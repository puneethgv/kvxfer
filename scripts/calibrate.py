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
from kvxfer.harvest import harvest, harvest_both
from kvxfer.models import load_model, load_tokenizer


def pair_slug(source: str, target: str) -> str:
    """Filesystem-safe identifier for a model pair."""
    clean = lambda m: m.split("/")[-1].lower()
    return f"{clean(source)}__to__{clean(target)}"



def _choose_passes(
    setting: str, accumulator_bytes: int, source_model, target_model
) -> bool:
    """Decide whether to sweep keys and values together.

    Sweeping together halves the model forward passes, which normally dominate
    the cost. But it holds two accumulators at once, and on a memory-constrained
    machine that is a false economy: exceeding physical memory costs far more
    than the forward passes it saves. Measured on a 16 GB laptop, the
    single-pass variant drove the system to 10.7 GB of swap and ran roughly
    twenty times slower per sequence than two separate passes.

    The estimate deliberately errs toward splitting. Model size is measured from
    the loaded parameters rather than inferred from the config, which previously
    understated it by about a gigabyte, and the headroom factor reflects that
    the first attempt swapped at a projected 84% of the reported budget --
    allocator fragmentation and the host's own demand are not visible here.

    Args:
        setting: "auto", "single", or "split".
        accumulator_bytes: footprint of one accumulator.
        source_model: the loaded source model.
        target_model: the loaded target model.

    Returns:
        True to sweep both kinds together.
    """
    if setting == "single":
        return True
    if setting == "split":
        return False

    def model_bytes(model) -> int:
        return sum(p.numel() * p.element_size() for p in model.parameters())

    weights = model_bytes(source_model) + model_bytes(target_model)

    try:
        budget = torch.mps.recommended_max_memory()
    except Exception:
        budget = 0
    if not budget:
        budget = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    if not budget:
        return False  # unknown budget: take the memory-safe option

    # Activations, the float32 content copies for both models, and allocator
    # slack. Set from what the failed run actually consumed, not from theory.
    overhead = 3 * 1024**3
    needed = 2 * accumulator_bytes + weights + overhead
    fits = needed < 0.85 * budget
    print(
        f"  memory check: single-pass needs ~{needed / 1024**3:.1f} GB "
        f"(weights {weights / 1024**3:.1f} + accumulators "
        f"{2 * accumulator_bytes / 1024**3:.1f} + overhead 3.0) against a "
        f"{budget / 1024**3:.1f} GB budget -> {'single' if fits else 'split'}"
    )
    return fits


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
    parser.add_argument(
        "--passes",
        default="auto",
        choices=["auto", "single", "split"],
        help="single sweeps keys and values together, halving forward passes but "
        "holding two accumulators; split sweeps them separately, halving peak "
        "memory. auto chooses by comparing the projected footprint against "
        "available memory -- on a machine that would swap, split is far faster "
        "despite doing twice the model work",
    )
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

    use_single = _choose_passes(args.passes, projected, source_model, target_model)
    manifest["passes"] = "single" if use_single else "split"
    print(f"harvest strategy: {manifest['passes']}")

    started = time.time()
    for split, corpus_split in splits.items():
        print(f"\n[{split}] harvesting {len(corpus_split)} sequences")
        if use_single:
            # Keys and values share the same forward passes, which dominate,
            # so one sweep does both.
            produced, report = harvest_both(
                source_model, target_model, source_geom, target_geom, corpus_split,
                source_layers=layers, token_stride=args.token_stride,
                batch_size=args.batch_size,
            )
            print(f"  {report}")
        else:
            # Two sweeps, one accumulator at a time. Twice the model work, half
            # the peak memory -- the right trade whenever holding both would
            # push the machine into swap.
            produced = {}
            for kind in ("keys", "values"):
                print(f"  [{kind}]")
                stats, report = harvest(
                    source_model, target_model, source_geom, target_geom, corpus_split,
                    kind=kind, source_layers=layers, token_stride=args.token_stride,
                    batch_size=args.batch_size,
                )
                print(f"  {report}")
                produced[kind] = stats

        for kind, stats in produced.items():
            path = out_dir / f"{split}_{kind}.pt"
            stats.save(path)
            print(f"  saved {path}")
            manifest["splits"][f"{split}_{kind}"] = {
                "n_tokens": report.n_tokens,
                "seconds": round(report.seconds, 1),
                "path": str(path),
            }
        del produced

    manifest["total_seconds"] = round(time.time() - started, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone in {manifest['total_seconds']}s -> {out_dir}")


if __name__ == "__main__":
    main()
