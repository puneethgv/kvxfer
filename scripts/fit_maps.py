"""Fit mappers from cached statistics and write them to disk.

The first half of the experiment, run as its own job. Fitting holds gigabytes
of calibration statistics and evaluation holds two models; running both in one
process makes the peak their sum, which is what the out-of-memory killer reacts
to on a 16 GB machine.

Splitting them also makes re-evaluation free: changing a task or an item limit
no longer costs a refit, and the fit is the slow half.

Example:
    python scripts/fit_maps.py \\
        --artifacts artifacts/qwen3-0.6b__to__qwen3-1.7b/web \\
        --lambdas results/sweeps/qwen3-0.6b__to__qwen3-1.7b__web.json \\
        --out maps/qwen3-0.6b__to__qwen3-1.7b__web
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kvxfer.experiment import ExperimentConfig, fit_mappers, settings_from_sweep
from kvxfer.mapstore import save_maps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, help="directory from calibrate.py")
    parser.add_argument("--out", required=True, help="where to write the fitted maps")
    parser.add_argument("--k", type=int, default=4, help="source layers per target layer")
    parser.add_argument(
        "--lam",
        type=float,
        default=1e-3,
        help="penalty used for layer selection, and for any solver the "
        "--lambdas report does not cover",
    )
    parser.add_argument(
        "--lambdas",
        default="",
        help="a sweep_lambda.py report; gives each solver the penalty and "
        "metric exponent chosen on its own held-out objective",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="how far the attention metric reshapes the penalty, for any cache "
        "kind the --lambdas report does not cover; 0 is plain ridge",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="if set, also fit rank-constrained maps at this rank, isotropic "
        "and attention-aligned, against the same rank budget",
    )
    parser.add_argument("--n-candidates", type=int, default=6)
    parser.add_argument(
        "--selection", default="topk", choices=["topk", "greedy"],
        help="topk reproduces the reference rule, ranked out of sample",
    )
    parser.add_argument("--metric-offset", type=int, default=2048)
    parser.add_argument("--query-sequences", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    lambdas, alphas = settings_from_sweep(args.lambdas) if args.lambdas else ({}, {})
    if lambdas:
        print(f"settings from {args.lambdas}: lambdas={lambdas} alphas={alphas}")

    config = ExperimentConfig(
        k=args.k,
        lam=args.lam,
        lambdas=lambdas,
        alpha=args.alpha,
        alphas=alphas,
        rank=args.rank,
        n_candidates=args.n_candidates,
        selection=args.selection,
        metric_offset=args.metric_offset,
        query_sequences=args.query_sequences,
        dtype=args.dtype,
    )

    maps, metadata = fit_mappers(args.artifacts, config)
    metadata["artifacts"] = str(args.artifacts)
    out = save_maps(args.out, maps, metadata)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
