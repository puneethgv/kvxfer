"""Fit mappers from cached statistics and measure retention.

Reads the artifacts produced by ``scripts/calibrate.py`` and needs no further
calibration passes: layer selection, penalty choice, and both solvers all run
off the cached Gram. The only GPU work here is the evaluation itself.

Conditions compared:

* ``target``  -- the target model with its own prefill; the ceiling.
* ``floor``   -- a destroyed (zeroed) cache; anchors the normalized scale.
* ``ridge``   -- isotropic least squares, reproducing the reference method.
* ``whitened``-- the attention-aligned objective.

Example:
    python scripts/fit_and_eval.py \\
        --artifacts artifacts/qwen3-0.6b__to__qwen3-1.7b/web \\
        --tasks arc_easy,hellaswag --limit 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kvxfer.data import build_calibration
from kvxfer.eval.retention import evaluate_task
from kvxfer.eval.tasks import load_task
from kvxfer.geometry import load_geometry
from kvxfer.mappers import FittedMapper, ZeroMapper
from kvxfer.metrics import HeadMetrics, identity_metrics, key_metric, value_metric
from kvxfer.models import load_model, load_tokenizer
from kvxfer.queries import collect_query_moments
from kvxfer.solvers.ridge import (
    held_out_r2,
    select_source_layers,
    select_top_k,
    solve_ridge,
)
from kvxfer.solvers.whitened import clear_design_cache, solve_whitened
from kvxfer.stats import GramStats


def choose_layers(
    fit_stats: GramStats,
    val_stats: GramStats,
    n_target_layers: int,
    k: int,
    lam: float,
    n_candidates: int,
    label: str,
    strategy: str = "topk",
) -> dict[int, tuple[int, ...]]:
    """Select source layers per target layer, once, out of sample.

    The selection is deliberately shared by every solver variant. If each
    variant chose its own layers, a difference in retention would confound the
    objective with the selection, and the objective is the thing under test.
    Selection uses the isotropic fit so that the baseline is not disadvantaged
    by being scored against a choice made for someone else.
    """
    chosen: dict[int, tuple[int, ...]] = {}
    for layer in range(n_target_layers):
        if strategy == "topk":
            chosen[layer] = select_top_k(fit_stats, val_stats, layer, k=k, lam=lam)
        else:
            chosen[layer] = select_source_layers(
                fit_stats, val_stats, layer, k=k, lam=lam, n_candidates=n_candidates
            )
        print(f"  [{label}] L{layer:02d} <- {chosen[layer]}", flush=True)
    return chosen


def fit_all_layers(
    fit_stats: GramStats,
    val_stats: GramStats,
    target_geom,
    metrics: HeadMetrics | None,
    selection: dict[int, tuple[int, ...]],
    lam: float,
    label: str,
) -> tuple[dict, dict]:
    """Fit one map per target layer on a fixed layer selection.

    Returns:
        ``(maps, diagnostics)`` keyed by target layer.
    """
    maps: dict[int, object] = {}
    diagnostics: dict[int, dict] = {}

    for layer in range(target_geom.n_layers):
        selected = selection[layer]
        if metrics is None:
            fit = solve_ridge(fit_stats, selected, layer, lam=lam)
        else:
            fit = solve_whitened(
                fit_stats,
                selected,
                layer,
                metrics,
                target_geom.n_kv_heads,
                target_geom.head_dim,
                lam=lam,
            )
        maps[layer] = fit
        diagnostics[layer] = {
            "source_layers": list(selected),
            "in_sample_r2": round(fit.r2, 4),
            "held_out_r2": round(float(held_out_r2(val_stats, fit).mean()), 4),
        }
        print(
            f"  [{label}] L{layer:02d} in-sample R2={fit.r2:.3f}  "
            f"held-out R2={diagnostics[layer]['held_out_r2']:.3f}",
            flush=True,
        )

    return maps, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, help="directory from calibrate.py")
    parser.add_argument("--tasks", default="arc_easy,hellaswag")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--k", type=int, default=4, help="source layers per target layer")
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--n-candidates", type=int, default=6)
    parser.add_argument(
        "--selection",
        default="topk",
        choices=["topk", "greedy"],
        help="topk reproduces the reference rule (ranked out of sample); "
        "greedy does forward selection, which is slower and rarely differs",
    )
    parser.add_argument("--metric-offset", type=int, default=2048)
    parser.add_argument("--query-sequences", type=int, default=32)
    parser.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    art = Path(args.artifacts)
    manifest = json.loads((art / "manifest.json").read_text())
    source_id, target_id = manifest["source"], manifest["target"]
    print(f"pair: {source_id} -> {target_id}")

    source_geom = load_geometry(source_id)
    target_geom = load_geometry(target_id)

    stats = {
        f"{split}_{kind}": GramStats.load(art / f"{split}_{kind}.pt")
        for split in ("fit", "val")
        for kind in ("keys", "values")
    }
    print(
        f"statistics: {stats['fit_keys'].n_tokens:,} fit tokens, "
        f"{stats['val_keys'].n_tokens:,} val tokens, "
        f"{stats['fit_keys'].n_source_layers} candidate source layers"
    )

    dtype = getattr(torch, args.dtype)
    tokenizer = load_tokenizer(source_id)
    target_model = load_model(target_id, dtype=dtype)

    print("\ncollecting query second moments for the key metric")
    probe = build_calibration(tokenizer, seq_len=512, n_sequences=args.query_sequences)
    q_moments = collect_query_moments(
        target_model, target_geom, probe, n_sequences=args.query_sequences
    )
    key_m = key_metric(
        q_moments, target_geom.head_dim, target_geom.rope_theta,
        max_offset=args.metric_offset,
    ).normalized()
    value_m = value_metric(target_model, target_geom).normalized()
    identity_k = identity_metrics(target_geom, "keys")
    identity_v = identity_metrics(target_geom, "values")

    variants = {
        "ridge": (None, None),
        "whitened": (key_m, value_m),
    }

    mappers: dict[str, object] = {"floor": ZeroMapper(target_geom)}
    diagnostics: dict[str, dict] = {}

    # Selected once and shared, so the variants differ only in objective.
    print("\nselecting source layers (out of sample, shared by all variants)")
    selection = {
        "keys": choose_layers(
            stats["fit_keys"], stats["val_keys"], target_geom.n_layers,
            args.k, args.lam, args.n_candidates, "select/K", args.selection,
        ),
        "values": choose_layers(
            stats["fit_values"], stats["val_values"], target_geom.n_layers,
            args.k, args.lam, args.n_candidates, "select/V", args.selection,
        ),
    }

    for name, (km, vm) in variants.items():
        print(f"\nfitting {name}")
        started = time.time()
        key_maps, key_diag = fit_all_layers(
            stats["fit_keys"], stats["val_keys"], target_geom, km,
            selection["keys"], args.lam, f"{name}/K",
        )
        value_maps, value_diag = fit_all_layers(
            stats["fit_values"], stats["val_values"], target_geom, vm,
            selection["values"], args.lam, f"{name}/V",
        )
        clear_design_cache()
        mappers[name] = FittedMapper(key_maps, value_maps, target_geom, label=name)
        diagnostics[name] = {
            "keys": key_diag,
            "values": value_diag,
            "fit_seconds": round(time.time() - started, 1),
        }
        print(f"  fitted in {diagnostics[name]['fit_seconds']}s")

    source_model = load_model(source_id, dtype=dtype)

    results = {}
    for task_name in args.tasks.split(","):
        print(f"\nevaluating {task_name} ({args.limit} items)")
        examples = load_task(task_name, limit=args.limit)
        outcome = evaluate_task(
            task_name, examples, target_model, tokenizer,
            source_model=source_model, mappers=mappers, dtype=dtype,
        )
        results[task_name] = outcome

        print(f"\n  {task_name}:")
        for name, cond in outcome.conditions.items():
            line = f"    {name:10s} acc={cond.accuracy:.4f} +/- {cond.stderr():.4f}"
            if name not in ("target", "floor"):
                line += (
                    f"  retention={outcome.retention(name):.1%}"
                    f"  floor-normalized={outcome.floor_normalized_retention(name):.1%}"
                )
            print(line)

    out_dir = Path(args.out) / art.parent.name / art.name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer_selection": {
            kind: {str(l): list(v) for l, v in sel.items()}
            for kind, sel in selection.items()
        },
        "source": source_id,
        "target": target_id,
        "settings": vars(args),
        "n_fit_tokens": stats["fit_keys"].n_tokens,
        "n_val_tokens": stats["val_keys"].n_tokens,
        "diagnostics": diagnostics,
        "results": {
            task: {
                "conditions": {
                    name: {
                        "accuracy": cond.accuracy,
                        "accuracy_normalized": cond.accuracy_normalized,
                        "stderr": cond.stderr(),
                        "n_items": cond.n_items,
                    }
                    for name, cond in outcome.conditions.items()
                },
                "retention": {
                    name: outcome.retention(name)
                    for name in outcome.conditions
                    if name not in ("target",)
                },
                "floor_normalized_retention": {
                    name: outcome.floor_normalized_retention(name)
                    for name in outcome.conditions
                    if name not in ("target", "floor")
                },
            }
            for task, outcome in results.items()
        },
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
