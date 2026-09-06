"""Modal entrypoints for the pairs that do not fit on a laptop.

Cost discipline is the design constraint here, not throughput. On a $30 budget
the dominant risk is not slow GPUs, it is paying repeatedly for work that
should happen once:

* Model weights live on a Volume with ``HF_HOME`` pointed at it, so a 16 GB
  download happens once ever rather than once per run. This is the single
  largest avoidable cost.
* Calibration and solving are separate functions. The solve is CPU linear
  algebra and must never hold a GPU; it can equally run on a laptop against the
  downloaded statistics.
* Statistics are written to the Volume, so every later ablation -- layer
  selection, penalty sweeps, both solvers -- is free.

Usage:
    modal run modal_app.py --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B
    modal run modal_app.py::evaluate --artifacts qwen3-1.7b__to__qwen3-4b/web
    modal volume get kvxfer-artifacts /artifacts/... ./artifacts/
"""

from __future__ import annotations

import modal

CACHE_DIR = "/cache"
ARTIFACT_DIR = "/artifacts"

weights_volume = modal.Volume.from_name("kvxfer-weights", create_if_missing=True)
artifact_volume = modal.Volume.from_name("kvxfer-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.51",
        "datasets>=2.20",
        "numpy>=1.26",
        "hf_transfer>=0.1",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("kvxfer")
)

app = modal.App("kvxfer", image=image)


@app.function(
    volumes={CACHE_DIR: weights_volume},
    timeout=60 * 60,
    # CPU-only: downloading weights does not need an accelerator, and doing it
    # on one is how a small budget disappears.
)
def fetch_weights(model_ids: list[str]) -> dict[str, str]:
    """Pre-download weights onto the Volume, off the GPU clock."""
    from huggingface_hub import snapshot_download

    out = {}
    for model_id in model_ids:
        path = snapshot_download(
            model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
        )
        out[model_id] = path
        print(f"cached {model_id}")
    weights_volume.commit()
    return out


@app.function(
    gpu="L4",
    volumes={CACHE_DIR: weights_volume, ARTIFACT_DIR: artifact_volume},
    timeout=4 * 60 * 60,
)
def calibrate(
    source: str,
    target: str,
    seq_len: int = 1024,
    fit_sequences: int = 512,
    val_sequences: int = 128,
    token_stride: int = 4,
    layer_stride: int = 1,
    batch_size: int = 4,
    mixture: str = "web=1.0",
    passes: str = "auto",
) -> dict:
    """Run a calibration pass and write statistics to the artifact Volume.

    The GPU here is doing prefill, which is compute-light relative to its
    memory footprint, so an L4 is usually the right price point. Step up only
    when the target model does not fit.

    ``passes`` is decided by the same measured-memory rule the laptop uses. It
    matters more here, not less: a 1.7B-to-4B pair over all 28 source layers
    needs about 7.5 GB per accumulator, and holding two alongside 11 GB of
    weights does not fit an L4 at all.
    """
    import json
    import time
    from pathlib import Path

    import torch

    from kvxfer.data import CalibrationSet, build_calibration
    from kvxfer.geometry import load_geometry
    from kvxfer.harvest import harvest, harvest_both
    from kvxfer.models import load_model, load_tokenizer
    from kvxfer.planning import accumulator_bytes, choose_harvest_strategy

    weights = {
        part.split("=")[0]: float(part.split("=")[1]) for part in mixture.split(",")
    }
    source_geom, target_geom = load_geometry(source), load_geometry(target)
    print(source_geom, "\n", target_geom)

    layers = tuple(range(0, source_geom.n_layers, layer_stride))
    projected = accumulator_bytes(source_geom, target_geom, layers)
    print(
        f"candidate source layers: {len(layers)} -> design dim "
        f"{len(layers) * source_geom.kv_dim:,}\n"
        f"projected accumulator: {projected / 1024**3:.2f} GB"
    )

    tokenizer = load_tokenizer(source)
    corpus = build_calibration(
        tokenizer,
        seq_len=seq_len,
        n_sequences=fit_sequences + val_sequences,
        mixture=weights,
    )

    source_model = load_model(source, dtype=torch.bfloat16)
    target_model = load_model(target, dtype=torch.bfloat16)

    slug = f"{source.split('/')[-1].lower()}__to__{target.split('/')[-1].lower()}"
    out_dir = Path(ARTIFACT_DIR) / slug / "-".join(sorted(weights))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": source,
        "target": target,
        "seq_len": seq_len,
        "token_stride": token_stride,
        "source_layers": list(layers),
        "mixture": weights,
        "dtype": "bfloat16",
        "splits": {},
    }

    splits = {
        "fit": CalibrationSet(corpus.input_ids[:fit_sequences], corpus.domains[:fit_sequences]),
        "val": CalibrationSet(corpus.input_ids[fit_sequences:], corpus.domains[fit_sequences:]),
    }

    use_single = choose_harvest_strategy(
        passes, projected, source_model, target_model
    )
    manifest["passes"] = "single" if use_single else "split"
    print(f"harvest strategy: {manifest['passes']}")

    started = time.time()
    for split, subset in splits.items():
        print(f"\n[{split}] {len(subset)} sequences")
        if use_single:
            produced, report = harvest_both(
                source_model, target_model, source_geom, target_geom, subset,
                source_layers=layers, token_stride=token_stride,
                batch_size=batch_size,
            )
            print(f"  {report}")
        else:
            produced = {}
            for kind in ("keys", "values"):
                print(f"  [{kind}]")
                stats, report = harvest(
                    source_model, target_model, source_geom, target_geom, subset,
                    kind=kind, source_layers=layers, token_stride=token_stride,
                    batch_size=batch_size,
                )
                print(f"  {report}")
                produced[kind] = stats

        for kind, stats in produced.items():
            stats.save(out_dir / f"{split}_{kind}.pt")
            manifest["splits"][f"{split}_{kind}"] = {
                "n_tokens": report.n_tokens,
                "seconds": round(report.seconds, 1),
            }
        del produced

    manifest["total_seconds"] = round(time.time() - started, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    artifact_volume.commit()
    print(f"\nwrote {out_dir}")
    return manifest


@app.function(
    gpu="L4",
    volumes={CACHE_DIR: weights_volume, ARTIFACT_DIR: artifact_volume},
    timeout=4 * 60 * 60,
)
def evaluate(
    artifacts: str,
    tasks: str = "arc_easy,arc_challenge",
    limit: int = 300,
    ppl_documents: int = 64,
    k: int = 4,
    lam: float = 1e-3,
    alpha: float = 1.0,
    dtype: str = "float32",
) -> dict:
    """Fit both solvers from cached statistics and evaluate them.

    Runs :func:`kvxfer.experiment.run_experiment`, the same function the laptop
    runs, so a pair measured here is comparable with a pair measured there.
    Fitting is CPU linear algebra and could run anywhere; it stays in the same
    function only because the evaluation immediately needs both models resident
    and a second job would re-download nothing but still re-load them.

    Args:
        artifacts: path under the artifact Volume, e.g. ``pair-slug/web``.
        tasks: comma-separated multiple-choice tasks.
        limit: items per task.
        ppl_documents: documents for the prefix-conditioned perplexity metric.
        k: source layers per target layer.
        lam: penalty, relative to the design scale.
        alpha: how far the attention metric reshapes that penalty.
        dtype: evaluation precision.

    Returns:
        The results payload, also written beside the statistics.
    """
    import json
    from pathlib import Path

    from kvxfer.experiment import ExperimentConfig, run_experiment

    art = Path(ARTIFACT_DIR) / artifacts
    config = ExperimentConfig(
        tasks=tuple(tasks.split(",")),
        limit=limit,
        ppl_documents=ppl_documents,
        k=k,
        lam=lam,
        alpha=alpha,
        dtype=dtype,
    )
    payload = run_experiment(art, config)
    (art / "results.json").write_text(json.dumps(payload, indent=2))
    artifact_volume.commit()
    print(f"\nwrote {art / 'results.json'}")
    return payload


@app.local_entrypoint()
def main(source: str = "Qwen/Qwen3-1.7B", target: str = "Qwen/Qwen3-4B") -> None:
    """Fetch weights once, calibrate the pair, then evaluate it."""
    fetch_weights.remote([source, target])
    manifest = calibrate.remote(source=source, target=target)
    print(f"calibration finished in {manifest['total_seconds']}s")

    slug = f"{source.split('/')[-1].lower()}__to__{target.split('/')[-1].lower()}"
    domain = "-".join(sorted(manifest["mixture"]))
    payload = evaluate.remote(artifacts=f"{slug}/{domain}")
    for name, result in payload["perplexity"].items():
        print(f"  {name:10s} ppl={result['perplexity']:.4f}")
