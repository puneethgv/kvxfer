"""Characterize how anisotropic the attention-induced metrics actually are.

This is the measurement that decides whether an attention-aligned objective can
help at all. If the metric were near-isotropic, the whitened solver would
reduce to plain ridge and there would be nothing to test. Reported per model so
the prediction can be made before any retention number is collected.

The summary statistics:

* **condition number** -- ratio of largest to smallest eigenvalue, per head.
* **effective rank** -- the perplexity of the normalized eigenvalue spectrum,
  i.e. how many directions genuinely carry weight. This is the more meaningful
  figure: a condition number can be inflated by a single near-null direction,
  whereas effective rank says how much of the space the model actually reads.

Example:
    python scripts/metric_diagnostics.py --model Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kvxfer.data import build_calibration
from kvxfer.geometry import load_geometry
from kvxfer.metrics import HeadMetrics, key_metric, value_metric
from kvxfer.models import load_model, load_tokenizer
from kvxfer.queries import collect_query_moments


def summarize(metrics: HeadMetrics) -> dict:
    """Per-head spectral summary of a metric."""
    evals = torch.linalg.eigvalsh(metrics.matrices).clamp_min(1e-12)
    condition = evals[..., -1] / evals[..., 0]
    share = evals / evals.sum(-1, keepdim=True)
    effective_rank = torch.exp(-(share * share.log()).sum(-1))

    return {
        "n_dims": int(evals.shape[-1]),
        "condition_median": float(condition.median()),
        "condition_p90": float(condition.flatten().quantile(0.9)),
        "effective_rank_median": float(effective_rank.median()),
        "effective_rank_min": float(effective_rank.min()),
        "top_eigenvalue_share_median": float((evals[..., -1] / evals.sum(-1)).median()),
        "per_layer_effective_rank": [
            round(float(v), 2) for v in effective_rank.mean(dim=1)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="the target model")
    parser.add_argument("--sequences", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-offset", type=int, default=2048)
    parser.add_argument("--out", default="results/metrics")
    args = parser.parse_args()

    geometry = load_geometry(args.model)
    print(geometry)

    model = load_model(args.model, dtype=torch.bfloat16)
    tokenizer = load_tokenizer(args.model)
    probe = build_calibration(
        tokenizer, seq_len=args.seq_len, n_sequences=args.sequences
    )

    moments = collect_query_moments(
        model, geometry, probe, n_sequences=args.sequences, token_stride=8
    )
    keys = key_metric(
        moments, geometry.head_dim, geometry.rope_theta, max_offset=args.max_offset
    ).normalized()
    values = value_metric(model, geometry).normalized()

    report = {
        "model": args.model,
        "n_sequences": args.sequences,
        "max_offset": args.max_offset,
        "keys": summarize(keys),
        "values": summarize(values),
    }

    for kind in ("keys", "values"):
        stats = report[kind]
        print(
            f"\n{kind.upper()}:"
            f"\n  condition number  median={stats['condition_median']:.1f}"
            f"  p90={stats['condition_p90']:.1f}"
            f"\n  effective rank    median={stats['effective_rank_median']:.1f}"
            f" of {stats['n_dims']}  min={stats['effective_rank_min']:.1f}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.model.split('/')[-1].lower()}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
