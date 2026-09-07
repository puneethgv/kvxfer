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

Always launch paid work with ``--detach``. Without it the app is ephemeral and
tied to the local client, so anything that kills the client -- including a
local out-of-memory kill that has nothing to do with the remote job -- stops a
GPU run that was minutes from finishing. That happened here, and cost a second
evaluation pass.

Usage:
    modal run --detach modal_app.py --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B
    modal run --detach modal_app.py::evaluate --artifacts qwen3-1.7b__to__qwen3-4b/web
    modal volume get kvxfer-artifacts qwen3-1.7b__to__qwen3-4b/web/results.json .
"""

from __future__ import annotations

import modal

CACHE_DIR = "/cache"
ARTIFACT_DIR = "/artifacts"

weights_volume = modal.Volume.from_name("kvxfer-weights", create_if_missing=True)
artifact_volume = modal.Volume.from_name("kvxfer-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Pinned, not floored. The cache path builds a DynamicCache through the
    # transformers 5 constructor (``ddp_cache_data``) and reads layers as
    # ``cache.layers[i].keys``; a range of ">=4.51" resolves happily to a 4.x
    # release where neither exists, and the failure would land partway into a
    # paid GPU run. These are the versions the local results were produced on,
    # so remote numbers stay comparable with them.
    .pip_install(
        "torch==2.14.0",
        "transformers==5.16.1",
        "datasets==5.0.1",
        "numpy>=1.26",
        "hf_transfer>=0.1",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("kvxfer")
)

# vLLM brings its own torch and pins hard, so it gets its own image rather
# than fighting the pinned one above. It is a measurement baseline here, not a
# dependency of the method.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "hf_transfer>=0.1")
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

app = modal.App("kvxfer", image=image)


@app.function(
    gpu="L4",
    image=vllm_image,
    volumes={CACHE_DIR: weights_volume},
    timeout=90 * 60,
)
def vllm_prefill(
    model: str = "Qwen/Qwen3-4B",
    lengths: str = "512,1024,2048,4096,8192",
    repeats: int = 5,
) -> dict:
    """Time an optimized engine's prefill, as the baseline to quote against.

    The speedup this project can claim is decided by how fast the thing being
    replaced is. Timing against a slow prefill would inflate the result exactly
    the way retention-against-target inflates the quality result, so the
    baseline has to be the best prefill available rather than the most
    convenient one.

    Measured as time-to-first-token with a single sequence and one output
    token, which is prefill plus one decode step. The decode step is a constant
    of a few milliseconds and is reported so it can be subtracted.

    Args:
        model: model id, matching the transfer target being compared against.
        lengths: comma-separated prompt lengths in tokens.
        repeats: timed runs per length; the median is reported.

    Returns:
        Median milliseconds per length.
    """
    import time

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        max_model_len=16384,
        disable_log_stats=True,
    )
    tokenizer = llm.get_tokenizer()
    one_token = SamplingParams(max_tokens=1, temperature=0.0)

    out = {"model": model, "engine": "vllm", "points": []}
    for n_tokens in (int(v) for v in lengths.split(",")):
        # A real token sequence, not a repeated id: prefix caching would
        # otherwise short-circuit the very work being measured.
        prompt_ids = list(range(1000, 1000 + n_tokens))
        prompt = tokenizer.decode(prompt_ids)

        llm.generate([prompt], one_token)  # warmup / compile
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            llm.generate([prompt], one_token)
            samples.append((time.perf_counter() - start) * 1000.0)
        samples.sort()
        median = samples[len(samples) // 2]
        out["points"].append({"n_tokens": n_tokens, "prefill_ms": median})
        print(f"  {n_tokens:>6} tok | vllm prefill+1 {median:8.1f} ms", flush=True)

    return out


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
    from kvxfer.planning import (
        accumulator_bytes,
        choose_harvest_strategy,
        report_headroom,
        working_set_bytes,
    )

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

    working = working_set_bytes(source_geom, target_geom, batch_size, seq_len)
    use_single = choose_harvest_strategy(
        passes, projected, source_model, target_model, working_set=working
    )
    manifest["passes"] = "single" if use_single else "split"
    print(f"harvest strategy: {manifest['passes']}")
    report_headroom(projected, working, split=not use_single)

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
    dtype: str = "bfloat16",
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
        dtype: evaluation precision. bfloat16 by default, matching both the
            harvest and the local runs. float32 is not a safe default here: a
            4B target costs 16 GB of weights rather than 8, and with the source
            model alongside it does not fit a 22 GiB L4 at all.

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


@app.function(
    gpu="L4",
    volumes={CACHE_DIR: weights_volume, ARTIFACT_DIR: artifact_volume},
    timeout=2 * 60 * 60,
)
def latency(
    artifacts: str,
    lengths: str = "512,1024,2048,4096,8192",
    repeats: int = 5,
    k: int = 4,
    lam: float = 1e-3,
    dtype: str = "bfloat16",
) -> dict:
    """Time map-and-inject against letting the target model prefill.

    Needs a GPU and needs it to be the *same* GPU for every condition, which is
    why this is one function rather than a comparison assembled from separate
    runs. A speedup measured across two machines measures the machines.

    Args:
        artifacts: path under the artifact Volume, e.g. ``pair-slug/web``.
        lengths: comma-separated context lengths in tokens.
        repeats: timed runs per measurement; the median is reported.
        k: source layers per target layer.
        lam: penalty, relative to the design scale.
        dtype: cache and compute dtype.

    Returns:
        A serialisable record of every timing.
    """
    import json
    from pathlib import Path

    import torch

    from kvxfer.experiment import ExperimentConfig, fit_mappers
    from kvxfer.eval.latency import benchmark_lengths
    from kvxfer.geometry import load_geometry
    from kvxfer.mappers import FittedMapper
    from kvxfer.models import load_model
    from kvxfer.planning import release_memory

    art = Path(ARTIFACT_DIR) / artifacts
    config = ExperimentConfig(k=k, lam=lam, dtype=dtype)
    maps, metadata = fit_mappers(art, config)
    release_memory()

    target_geom = load_geometry(metadata["target"])
    mapper = FittedMapper(
        maps["ridge"]["keys"], maps["ridge"]["values"], target_geom, label="ridge"
    )
    del maps
    release_memory()

    torch_dtype = getattr(torch, dtype)
    target_model = load_model(metadata["target"], dtype=torch_dtype)
    source_model = load_model(metadata["source"], dtype=torch_dtype)

    print(f"\nlatency: {metadata['source']} -> {metadata['target']}")
    points = benchmark_lengths(
        source_model,
        target_model,
        mapper,
        lengths=tuple(int(v) for v in lengths.split(",")),
        repeats=repeats,
        dtype=torch_dtype,
    )

    payload = {
        "source": metadata["source"],
        "target": metadata["target"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dtype": dtype,
        "k": k,
        "points": [
            {
                "n_tokens": p.n_tokens,
                "target_prefill_ms": p.target_prefill_ms,
                "source_prefill_ms": p.source_prefill_ms,
                "map_ms": p.map_ms,
                "inject_ms": p.inject_ms,
                "warm_speedup": p.warm_speedup,
                "cold_speedup": p.cold_speedup,
            }
            for p in points
        ],
    }
    (art / "latency.json").write_text(json.dumps(payload, indent=2))
    artifact_volume.commit()
    print(f"\nwrote {art / 'latency.json'}")
    return payload


@app.local_entrypoint()
def main(
    source: str = "Qwen/Qwen3-1.7B",
    target: str = "Qwen/Qwen3-4B",
    layer_stride: int = 2,
    passes: str = "split",
    batch_size: int = 2,
    tasks: str = "arc_easy,arc_challenge",
    limit: int = 300,
    lam: float = 1e-3,
) -> None:
    """Fetch weights once, calibrate the pair, then evaluate it.

    ``layer_stride`` defaults to 2 rather than 1 deliberately. At stride 1 a
    1.7B-to-4B design is 28,672 wide, which is 7.5 GB per accumulator; even
    sweeping the cache kinds separately that peaks near 22 GB against an L4's
    24 GB. Stride 2 halves the pool to 14 candidate layers, peaks around
    17 GB, and matches the candidate pool the local 0.6B-to-1.7B run used, so
    top-k selection stays comparable across pairs.

    ``batch_size`` defaults to 2 for the same reason. Every harvested batch is
    converted to float32 content for keys and values across every layer of
    both models, which at batch 4 is about 3.2 GB for this pair against the
    10.5 GB left free once the weights are resident.
    """
    fetch_weights.remote([source, target])
    manifest = calibrate.remote(
        source=source,
        target=target,
        layer_stride=layer_stride,
        passes=passes,
        batch_size=batch_size,
    )
    print(f"calibration finished in {manifest['total_seconds']}s")

    slug = f"{source.split('/')[-1].lower()}__to__{target.split('/')[-1].lower()}"
    domain = "-".join(sorted(manifest["mixture"]))
    payload = evaluate.remote(
        artifacts=f"{slug}/{domain}", tasks=tasks, limit=limit, lam=lam
    )

    print("\nprefix-conditioned perplexity:")
    for name, result in payload["perplexity"].items():
        print(f"  {name:16s} ppl={result['perplexity']:.4f}")
    for task, block in payload["results"].items():
        print(f"\n{task}:")
        for name, cond in block["conditions"].items():
            retention = block["retention"].get(name)
            suffix = "" if retention is None else f"  retention={retention:.1%}"
            print(f"  {name:16s} acc={cond['accuracy']:.4f}{suffix}")
