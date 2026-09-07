"""Train the residual correction on top of a fitted closed-form map.

The closed-form solvers here all reduce to plain ridge once their settings are
chosen honestly, because a closed form can only express the attention metric as
a penalty. This trains the part that cannot be solved for, against the
objective that actually matters: that attention produce the same output.

Example:
    python scripts/train_residual.py \\
        --artifacts artifacts/qwen3-0.6b__to__qwen3-1.7b/web \\
        --maps maps/qwen3-0.6b__to__qwen3-1.7b__web \\
        --out maps/qwen3-0.6b__to__qwen3-1.7b__web__residual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kvxfer.data import build_calibration
from kvxfer.geometry import load_geometry
from kvxfer.mapstore import load_maps
from kvxfer.models import load_model, load_tokenizer
from kvxfer.solvers.neural import ResidualConfig, ResidualMapper, train_residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, help="directory from calibrate.py")
    parser.add_argument("--maps", required=True, help="directory from fit_maps.py")
    parser.add_argument("--out", required=True, help="where to write the trained residual")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--inner-steps", type=int, default=4)
    parser.add_argument("--batch-sequences", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-sequences", type=int, default=256)
    parser.add_argument(
        "--calibration-sequences",
        type=int,
        default=640,
        help="sequences the calibration consumed; training documents skip past "
        "them so the residual is not fitted on the maps' own data",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    manifest = json.loads((Path(args.artifacts) / "manifest.json").read_text())
    source_id, target_id = manifest["source"], manifest["target"]
    target_geom = load_geometry(target_id)
    dtype = getattr(torch, args.dtype)

    mappers, metadata = load_maps(args.maps, target_geom)
    base = mappers["ridge"]
    design_dim = base.key_maps[0].dense().shape[0]
    print(f"pair: {source_id} -> {target_id}, design dim {design_dim}")

    mapper = ResidualMapper(
        base.key_maps, base.value_maps, target_geom, design_dim, hidden=args.hidden
    )
    print(f"residual: {mapper.n_trained_parameters() / 1e6:.1f}M trained parameters")

    tokenizer = load_tokenizer(source_id)
    corpus = build_calibration(
        tokenizer,
        seq_len=args.seq_len,
        n_sequences=args.train_sequences,
        skip=args.calibration_sequences,
    )
    source_model = load_model(source_id, dtype=dtype)
    target_model = load_model(target_id, dtype=dtype)

    report = train_residual(
        mapper,
        source_model,
        target_model,
        corpus,
        ResidualConfig(
            hidden=args.hidden,
            learning_rate=args.learning_rate,
            steps=args.steps,
            inner_steps=args.inner_steps,
            batch_sequences=args.batch_sequences,
            seq_len=args.seq_len,
        ),
    )
    print(f"\n{report}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "key_residual": mapper.key_residual.state_dict(),
            "value_residual": mapper.value_residual.state_dict(),
            "hidden": args.hidden,
            "design_dim": design_dim,
        },
        out / "residual.pt",
    )
    (out / "training.json").write_text(
        json.dumps(
            {
                "source": source_id,
                "target": target_id,
                "base_maps": str(args.maps),
                "steps": report.steps,
                "trained_parameters": report.trained_parameters,
                "initial_loss": report.initial_loss,
                "final_loss": report.final_loss,
                "history": report.history,
                "settings": vars(args),
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
