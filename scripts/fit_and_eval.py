"""Fit mappers from cached statistics and measure retention.

Reads the artifacts produced by ``scripts/calibrate.py`` and needs no further
calibration passes: layer selection, penalty choice, and both solvers all run
off the cached Gram. The only GPU work here is the evaluation itself.

The experiment itself lives in :mod:`kvxfer.experiment` so that the Modal
entrypoint runs the same code on pairs too large for a laptop.

Conditions compared:

* ``target``  -- the target model with its own prefill; the ceiling.
* ``floor``   -- a destroyed (zeroed) cache; anchors the normalized scale.
* ``source``  -- the source model standalone; the decision baseline.
* ``ridge``   -- isotropic least squares, reproducing the reference method.
* ``whitened``-- the attention-aligned objective.

Example:
    python scripts/fit_and_eval.py \\
        --artifacts artifacts/qwen3-0.6b__to__qwen3-1.7b/web \\
        --lambdas results/sweeps/qwen3-0.6b__to__qwen3-1.7b__web.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvxfer.experiment import (
    ExperimentConfig,
    evaluate_mappers,
    run_experiment,
    settings_from_sweep,
)
from kvxfer.geometry import load_geometry
from kvxfer.mapstore import load_maps


def _target_of(artifacts: Path) -> str:
    """Read the target model id from a calibration manifest."""
    return json.loads((artifacts / "manifest.json").read_text())["target"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, help="directory from calibrate.py")
    parser.add_argument(
        "--maps",
        default="",
        help="a directory from scripts/fit_maps.py. Given one, this evaluates "
        "those maps instead of refitting, so the statistics never have to be "
        "resident alongside the models",
    )
    parser.add_argument(
        "--tasks",
        default="arc_easy,arc_challenge",
        help="comma-separated multiple-choice tasks; pass an empty string to "
        "measure perplexity alone, which is the high-resolution metric and "
        "enough for sweeps where the tasks would only add noise and time",
    )
    parser.add_argument(
        "--ppl-documents",
        type=int,
        default=64,
        help="documents for prefix-conditioned perplexity, the primary metric",
    )
    parser.add_argument("--ppl-seq-len", type=int, default=512)
    parser.add_argument("--ppl-prefix", type=int, default=256)
    parser.add_argument("--ppl-batch-size", type=int, default=4)
    parser.add_argument(
        "--calibration-sequences",
        type=int,
        default=640,
        help="how many sequences calibration consumed; evaluation documents "
        "skip past these so the two corpora stay disjoint",
    )
    parser.add_argument("--limit", type=int, default=300)
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
        "metric exponent chosen on its own held-out objective rather than a "
        "shared guess",
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
        "and attention-aligned, so the metric's contribution to truncation is "
        "measured against the same rank budget",
    )
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
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="matches the dtype calibration harvested in and the one these "
        "models are served in; float32 doubles the resident weights",
    )
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    lambdas, alphas = settings_from_sweep(args.lambdas) if args.lambdas else ({}, {})
    if lambdas:
        print(f"settings from {args.lambdas}: lambdas={lambdas} alphas={alphas}")

    config = ExperimentConfig(
        tasks=tuple(t for t in args.tasks.split(",") if t.strip()),
        limit=args.limit,
        ppl_documents=args.ppl_documents,
        ppl_seq_len=args.ppl_seq_len,
        ppl_prefix=args.ppl_prefix,
        ppl_batch_size=args.ppl_batch_size,
        calibration_sequences=args.calibration_sequences,
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

    art = Path(args.artifacts)
    if args.maps:
        mappers, metadata = load_maps(args.maps, load_geometry(_target_of(art)))
        print(f"loaded {len(mappers)} fitted variants from {args.maps}")
        payload = evaluate_mappers(mappers, metadata, config)
    else:
        payload = run_experiment(art, config)

    out_dir = Path(args.out) / art.parent.name / art.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
