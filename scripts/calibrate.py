"""Run a calibration pass for one source -> target pair and cache the statistics.

This is the only step that needs both models resident. Everything downstream --
layer selection, penalty sweeps, the attention-aligned solver, ablations --
reads the cached statistics and never touches a GPU again.

Fit and validation statistics are accumulated over disjoint sequences. The
validation split is not optional: in-sample R² saturates at 1.0 in this regime
and cannot rank candidates at all.

Example:
    python scripts/calibrate.py --source Qwen/Qwen3-0.6B --target Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kvxfer.data import build_calibration
from kvxfer.geometry import load_geometry
from kvxfer.harvest import harvest
from kvxfer.models import load_model, load_tokenizer


def pair_slug(source: str, target: str) -> str:
    """Filesystem-safe identifier for a model pair."""
    clean = lambda m: m.split("/")[-1].lower()
    return f"{clean(source)}__to__{clean(target)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--fit-sequences", type=int, default=512)
    parser.add_argument("--val-sequences", type=int, default=128)
    parser.add_argument(
        "--token-stride",
        type=int,
        default=4,
        help="keep every nth token; adjacent tokens carry near-duplicate KV states",
    )
    parser.add_argument(
        "--layer-stride",
        type=int,
        default=1,
        help="candidate source layers to consider (memory grows quadratically)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--mixture",
        default="web=1.0",
        help="calibration domain mixture, e.g. 'web=0.5,code=0.3,math=0.2'",
    )
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    mixture = {
        part.split("=")[0]: float(part.split("=")[1])
        for part in args.mixture.split(",")
    }
    domain_tag = "-".join(sorted(mixture))

    source_geom = load_geometry(args.source)
    target_geom = load_geometry(args.target)
    print(source_geom)
    print(target_geom)

    layers = tuple(range(0, source_geom.n_layers, args.layer_stride))
    dim = len(layers) * source_geom.kv_dim
    projected = (dim * dim + dim * target_geom.n_layers * target_geom.kv_dim) * 4
    print(
        f"\ncandidate source layers: {len(layers)} -> design dim {dim:,}"
        f"\nprojected accumulator: {projected / 1024**3:.2f} GB"
    )

    tokenizer = load_tokenizer(args.source)
    total = args.fit_sequences + args.val_sequences
    print(f"\nbuilding calibration set: {total} x {args.seq_len} tokens, {mixture}")
    corpus = build_calibration(
        tokenizer, seq_len=args.seq_len, n_sequences=total, mixture=mixture
    )

    from kvxfer.data import CalibrationSet

    splits = {
        "fit": CalibrationSet(
            corpus.input_ids[: args.fit_sequences],
            corpus.domains[: args.fit_sequences],
        ),
        "val": CalibrationSet(
            corpus.input_ids[args.fit_sequences :],
            corpus.domains[args.fit_sequences :],
        ),
    }

    dtype = getattr(torch, args.dtype)
    print(f"\nloading models in {args.dtype}")
    source_model = load_model(args.source, dtype=dtype)
    target_model = load_model(args.target, dtype=dtype)

    out_dir = Path(args.out) / pair_slug(args.source, args.target) / domain_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": args.source,
        "target": args.target,
        "seq_len": args.seq_len,
        "token_stride": args.token_stride,
        "source_layers": list(layers),
        "mixture": mixture,
        "dtype": args.dtype,
        "splits": {},
    }

    started = time.time()
    for split, corpus_split in splits.items():
        for kind in ("keys", "values"):
            print(f"\n[{split}/{kind}] harvesting {len(corpus_split)} sequences")
            stats, report = harvest(
                source_model,
                target_model,
                source_geom,
                target_geom,
                corpus_split,
                kind=kind,
                source_layers=layers,
                token_stride=args.token_stride,
                batch_size=args.batch_size,
            )
            path = out_dir / f"{split}_{kind}.pt"
            stats.save(path)
            print(f"  {report}\n  saved {path}")
            manifest["splits"][f"{split}_{kind}"] = {
                "n_tokens": report.n_tokens,
                "seconds": round(report.seconds, 1),
                "path": str(path),
            }
            del stats

    manifest["total_seconds"] = round(time.time() - started, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone in {manifest['total_seconds']}s -> {out_dir}")


if __name__ == "__main__":
    main()
