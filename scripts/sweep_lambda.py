"""Sweep the penalty for both solvers, scoring each on its own objective.

The two solvers optimize different things, so tuning them against a single
criterion would hand the comparison to whichever one that criterion favours.
Isotropic ridge is selected on held-out R2; the attention-aligned solver is
selected on held-out metric-weighted R2. Each is then evaluated at its own best
setting.

The attention-aligned solver has two knobs, not one. ``lambda`` sets how much
regularization there is; ``alpha`` sets how much the attention metric is
allowed to redistribute it, through a penalty of ``lambda / L_j**alpha``. At
alpha=0 the solver is identically plain ridge, so the sweep contains its own
baseline and the comparison cannot be won by a tuning asymmetry. Sweeping alpha
matters because the two ends genuinely disagree: alpha=1 fits its own objective
better in sample while generalizing worse, since it regularizes least exactly
where the design Gram is worst conditioned.

This costs no GPU time beyond collecting the metrics: every fit and every score
comes from cached statistics, which is the point of accumulating sufficient
statistics in the first place.

Example:
    python scripts/sweep_lambda.py \\
        --artifacts artifacts/qwen3-0.6b__to__qwen3-1.7b/web --target Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kvxfer.data import build_calibration
from kvxfer.geometry import load_geometry
from kvxfer.metrics import key_metric, value_metric
from kvxfer.models import load_model, load_tokenizer
from kvxfer.queries import collect_query_moments
from kvxfer.solvers.ridge import held_out_r2, select_top_k, solve_ridge
from kvxfer.solvers.whitened import (
    clear_design_cache,
    held_out_metric_r2,
    solve_whitened,
)
from kvxfer.stats import GramStats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--lambdas", default="1e-5,1e-4,1e-3,1e-2,1e-1,1.0",
        help="penalties to try, relative to the design scale",
    )
    parser.add_argument(
        "--alphas",
        default="0,0.25,0.5,0.75,1.0",
        help="how far the metric may reshape the penalty; 0 is plain ridge",
    )
    parser.add_argument(
        "--layers", default="", help="comma-separated target layers; default is a spread"
    )
    parser.add_argument("--floor", type=float, default=1e-6)
    parser.add_argument("--out", default="results/sweeps")
    args = parser.parse_args()

    art = Path(args.artifacts)
    geom = load_geometry(args.target)
    lambdas = [float(v) for v in args.lambdas.split(",")]
    alphas = [float(v) for v in args.alphas.split(",")]

    stats = {
        f"{split}_{kind}": GramStats.load(art / f"{split}_{kind}.pt")
        for split in ("fit", "val")
        for kind in ("keys", "values")
    }
    has_head_moments = stats["val_keys"].yty_head is not None

    if args.layers:
        layers = [int(v) for v in args.layers.split(",")]
    else:
        layers = sorted({0, geom.n_layers // 4, geom.n_layers // 2, geom.n_layers - 1})

    print(f"target {args.target}, probing layers {layers}")
    if not has_head_moments:
        print(
            "note: these statistics predate per-head target moments, so the "
            "attention-weighted score is unavailable and only isotropic R2 is "
            "reported. Recalibrate to select the whitened solver properly."
        )

    model = load_model(args.target, dtype=torch.bfloat16)
    tokenizer = load_tokenizer(args.target)
    probe = build_calibration(tokenizer, seq_len=512, n_sequences=16)
    moments = collect_query_moments(model, geom, probe, n_sequences=16, token_stride=8)
    metrics = {
        "keys": key_metric(moments, geom.head_dim, geom.rope_theta).normalized(),
        "values": value_metric(model, geom).normalized(),
    }
    del model

    report: dict = {
        "target": args.target,
        "k": args.k,
        "layers": layers,
        "alphas": alphas,
        "sweeps": {},
    }

    for kind in ("keys", "values"):
        fit_stats, val_stats = stats[f"fit_{kind}"], stats[f"val_{kind}"]
        print(f"\n=== {kind} ===")
        header = (
            f"{'lambda':>8} {'alpha':>6} {'ridge iso R2':>14} {'whitened iso R2':>17}"
        )
        if has_head_moments:
            header += f" {'ridge metR2':>13} {'whitened metR2':>16}"
        print(header)

        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

        def score(fits: list) -> tuple[float, float]:
            """Held-out isotropic and metric R2, averaged over probe layers."""
            iso = mean([float(held_out_r2(val_stats, f).mean()) for f in fits])
            met = (
                mean([
                    held_out_metric_r2(
                        val_stats, f, metrics[kind], geom.n_kv_heads, geom.head_dim
                    )
                    for f in fits
                ])
                if has_head_moments
                else float("nan")
            )
            return iso, met

        rows = []
        for lam in lambdas:
            # The selection is made once per penalty and reused across alpha,
            # so that alpha is compared against a fixed set of predictors
            # rather than being credited with a different layer choice.
            selected = {
                layer: select_top_k(fit_stats, val_stats, layer, k=args.k, lam=lam)
                for layer in layers
            }
            iso_r, met_r = score(
                [solve_ridge(fit_stats, selected[l], l, lam=lam) for l in layers]
            )
            for alpha in alphas:
                iso_w, met_w = score([
                    solve_whitened(
                        fit_stats, selected[l], l, metrics[kind],
                        geom.n_kv_heads, geom.head_dim, lam=lam,
                        eigenvalue_floor=args.floor, alpha=alpha,
                    )
                    for l in layers
                ])
                row = {
                    "lambda": lam,
                    "alpha": alpha,
                    "ridge_isotropic_r2": iso_r,
                    "ridge_metric_r2": met_r,
                    "whitened_isotropic_r2": iso_w,
                    "whitened_metric_r2": met_w,
                }
                rows.append(row)
                line = (
                    f"{lam:>8.0e} {alpha:>6.2f} {iso_r:>14.4f} {iso_w:>17.4f}"
                )
                if has_head_moments:
                    line += f" {met_r:>13.4f} {met_w:>16.4f}"
                print(line, flush=True)
            clear_design_cache()

        best_ridge = max(rows, key=lambda r: r["ridge_isotropic_r2"])
        key = "whitened_metric_r2" if has_head_moments else "whitened_isotropic_r2"
        best_whitened = max(rows, key=lambda r: r[key])
        print(
            f"  best ridge lambda={best_ridge['lambda']:.0e} (isotropic R2)  |  "
            f"best whitened lambda={best_whitened['lambda']:.0e} "
            f"alpha={best_whitened['alpha']:.2f} ({key})"
        )
        report["sweeps"][kind] = {
            "rows": rows,
            "best_ridge_lambda": best_ridge["lambda"],
            "best_whitened_lambda": best_whitened["lambda"],
            "best_whitened_alpha": best_whitened["alpha"],
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{art.parent.name}__{art.name}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
