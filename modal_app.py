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
    # transformers is pinned below 5 because vLLM 0.11 still calls
    # tokenizer.all_special_tokens_extended, which v5 removed -- the run dies
    # at tokenizer load, before any timing. This differs from the transformers
    # version the method itself runs on, which is fine: what is being timed
    # here is vLLM's own prefill kernels, not anything transformers does.
    .pip_install("vllm==0.11.0", "transformers<5", "hf_transfer>=0.1")
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

    A sanity floor worth keeping in mind when reading the output: a 4B model
    over 8192 tokens is 65.9 TFLOP, which an L4 cannot do in under about
    544 ms. Anything faster is not measuring prefill.

    Args:
        model: model id, matching the transfer target being compared against.
        lengths: comma-separated prompt lengths in tokens.
        repeats: timed runs per length; the median is reported.

    Returns:
        Median milliseconds per length.
    """
    import time

    from vllm import LLM, SamplingParams

    # Prefix caching off. With it on, the warmup populates the cache and every
    # timed run of the same prompt is served from it: the first version of this
    # reported 74.5 ms to prefill 8192 tokens, which is below the 544 ms floor
    # the card's peak throughput allows, so it was timing cache hits.
    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        max_model_len=16384,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    one_token = SamplingParams(max_tokens=1, temperature=0.0)

    out = {"model": model, "engine": "vllm", "points": []}
    for n_tokens in (int(v) for v in lengths.split(",")):
        # Token ids are passed directly, so the prompt is exactly n_tokens long
        # rather than however many tokens a decoded string re-encodes to. Each
        # timed run also uses a different prompt, belt and braces against any
        # caching that survives the flag above.
        def prompt_at(offset: int) -> dict:
            return {"prompt_token_ids": list(range(offset, offset + n_tokens))}

        llm.generate([prompt_at(1000)], one_token)  # warmup / compile
        samples = []
        for run in range(repeats):
            start = time.perf_counter()
            llm.generate([prompt_at(2000 + run * n_tokens)], one_token)
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
    maps: str = "",
    residual: str = "",
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
        maps: subdirectory of stored maps to evaluate instead of refitting.
        residual: subdirectory of a trained residual to add as a condition. It
            is scored alongside its own frozen base, so the comparison isolates
            what training added rather than comparing across configurations. bfloat16 by default, matching both the
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
        tasks=tuple(t for t in tasks.split(",") if t.strip()),
        limit=limit,
        ppl_documents=ppl_documents,
        k=k,
        lam=lam,
        alpha=alpha,
        dtype=dtype,
        variants=("ridge",) if residual else (),
    )

    if maps or residual:
        from kvxfer.experiment import evaluate_mappers
        from kvxfer.geometry import load_geometry
        from kvxfer.mapstore import load_maps
        from kvxfer.solvers.neural import load_residual

        target_geom = load_geometry(
            json.loads((art / "manifest.json").read_text())["target"]
        )
        mappers, metadata = load_maps(art / (maps or "maps"), target_geom)
        if residual:
            base = mappers["ridge"]
            # Trained residual and its own frozen base, side by side: the
            # difference between them is what training bought and nothing else.
            mappers["residual"] = load_residual(
                art / residual, base.key_maps, base.value_maps, target_geom
            )
        payload = evaluate_mappers(mappers, metadata, config)
    else:
        payload = run_experiment(art, config)
    (art / "results.json").write_text(json.dumps(payload, indent=2))
    artifact_volume.commit()
    print(f"\nwrote {art / 'results.json'}")
    return payload


@app.function(
    cpu=8.0,
    memory=32768,
    volumes={CACHE_DIR: weights_volume, ARTIFACT_DIR: artifact_volume},
    timeout=4 * 60 * 60,
)
def fit(
    artifacts: str,
    variants: str = "ridge",
    k: int = 4,
    lam: float = 1e-3,
    rank: int = 0,
    name: str = "maps",
) -> dict:
    """Fit maps once and store them beside the statistics.

    Fitting is closed-form linear algebra over cached moments. It was being
    redone inside every latency and evaluation run, which meant paying a GPU
    to recompute maps that never change -- three latency reruns cost about
    thirty minutes of L4 time producing byte-identical output.

    This runs on CPU because a ridge-only fit needs no model: the attention
    metrics are the only thing that loads one, and only the aligned solver
    reads them. Ask for "ridge,whitened" and it will still work, but it will
    want a GPU to be quick about the metrics.

    Args:
        artifacts: path under the artifact Volume, e.g. ``pair-slug/web``.
        variants: comma-separated variant names to fit.
        k: source layers per target layer.
        lam: penalty, relative to the design scale.
        rank: optional rank budget for the constrained variants.
        name: subdirectory to write the maps into.

    Returns:
        Metadata describing what was fitted and where it landed.
    """
    from pathlib import Path

    from kvxfer.experiment import ExperimentConfig, fit_mappers
    from kvxfer.mapstore import save_maps

    art = Path(ARTIFACT_DIR) / artifacts
    config = ExperimentConfig(
        k=k,
        lam=lam,
        rank=rank,
        dtype="float32",  # CPU: bfloat16 matmuls there are slow and emulated
        variants=tuple(v for v in variants.split(",") if v.strip()),
    )
    maps, metadata = fit_mappers(art, config)
    out = save_maps(art / name, maps, metadata)
    artifact_volume.commit()
    print(f"\nwrote {out}")
    return {"maps": str(out), "variants": metadata["variants"] if "variants" in metadata else list(maps)}


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
    maps: str = "maps",
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
        maps: subdirectory of saved maps to reuse. Fitted on demand only when
            it does not exist, so repeat measurements do not re-pay for maps
            that never change.

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
    # Reuse stored maps when they exist. Only the isotropic map is timed, so
    # only it is fitted when they do not.
    from kvxfer.mapstore import load_maps, save_maps

    saved = art / maps
    config = ExperimentConfig(k=k, lam=lam, dtype=dtype, variants=("ridge",))
    if (saved / "maps.json").exists():
        print(f"loading maps from {saved}")
        target_geom = load_geometry(
            json.loads((art / "manifest.json").read_text())["target"]
        )
        mappers, metadata = load_maps(saved, target_geom)
        mapper = mappers["ridge"]
        del mappers
    else:
        print(f"no maps at {saved}; fitting them once and storing them there")
        fitted, metadata = fit_mappers(art, config)
        save_maps(saved, fitted, metadata)
        artifact_volume.commit()
        target_geom = load_geometry(metadata["target"])
        mapper = FittedMapper(
            fitted["ridge"]["keys"], fitted["ridge"]["values"], target_geom,
            label="ridge",
        )
        del fitted
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
        "lengths_requested": [int(v) for v in lengths.split(",")],
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


@app.function(
    gpu="L4",
    volumes={CACHE_DIR: weights_volume, ARTIFACT_DIR: artifact_volume},
    timeout=6 * 60 * 60,
)
def train(
    artifacts: str,
    maps: str = "maps",
    out: str = "residual",
    hidden: int = 256,
    steps: int = 2000,
    inner_steps: int = 4,
    batch_sequences: int = 2,
    seq_len: int = 512,
    learning_rate: float = 1e-3,
    train_sequences: int = 256,
    dtype: str = "bfloat16",
) -> dict:
    """Train the residual correction against the attention-output objective.

    An L4 rather than something larger: measured at 1.4 s per step, of which
    1.36 s is the two model forwards that generate the caches. The training
    itself is 5% of the cost, so paying for a faster card buys almost nothing.

    Args:
        artifacts: path under the artifact Volume, e.g. ``pair-slug/web``.
        maps: subdirectory holding the frozen base maps.
        out: subdirectory to write the trained residual into.
        hidden: residual bottleneck width.
        steps: outer steps, each generating a fresh batch of caches.
        inner_steps: gradient steps per generated batch.
        batch_sequences: sequences per batch.
        seq_len: tokens per sequence.
        learning_rate: peak learning rate for the one-cycle schedule.
        train_sequences: size of the training corpus, disjoint from calibration.
        dtype: model dtype.

    Returns:
        The training report.
    """
    import json
    from pathlib import Path

    import torch

    from kvxfer.data import build_calibration
    from kvxfer.geometry import load_geometry
    from kvxfer.mapstore import load_maps
    from kvxfer.models import load_model, load_tokenizer
    from kvxfer.solvers.neural import ResidualConfig, ResidualMapper, train_residual

    art = Path(ARTIFACT_DIR) / artifacts
    manifest = json.loads((art / "manifest.json").read_text())
    source_id, target_id = manifest["source"], manifest["target"]
    target_geom = load_geometry(target_id)
    torch_dtype = getattr(torch, dtype)

    mappers, _ = load_maps(art / maps, target_geom)
    base = mappers["ridge"]
    design_dim = base.key_maps[0].dense().shape[0]
    mapper = ResidualMapper(
        base.key_maps, base.value_maps, target_geom, design_dim, hidden=hidden
    )
    print(
        f"pair: {source_id} -> {target_id}\n"
        f"residual: {mapper.n_trained_parameters() / 1e6:.1f}M trained parameters "
        f"at hidden width {hidden}"
    )

    tokenizer = load_tokenizer(source_id)
    corpus = build_calibration(
        tokenizer, seq_len=seq_len, n_sequences=train_sequences, skip=640
    )
    source_model = load_model(source_id, dtype=torch_dtype)
    target_model = load_model(target_id, dtype=torch_dtype)

    report = train_residual(
        mapper, source_model, target_model, corpus,
        ResidualConfig(
            hidden=hidden, learning_rate=learning_rate, steps=steps,
            inner_steps=inner_steps, batch_sequences=batch_sequences,
            seq_len=seq_len,
        ),
    )
    print(f"\n{report}")

    out_dir = art / out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "key_residual": mapper.key_residual.state_dict(),
            "value_residual": mapper.value_residual.state_dict(),
            "hidden": hidden,
            "design_dim": design_dim,
        },
        out_dir / "residual.pt",
    )
    payload = {
        "source": source_id,
        "target": target_id,
        "steps": report.steps,
        "trained_parameters": report.trained_parameters,
        "initial_loss": report.initial_loss,
        "final_loss": report.final_loss,
        "history": report.history,
    }
    (out_dir / "training.json").write_text(json.dumps(payload, indent=2))
    artifact_volume.commit()
    print(f"wrote {out_dir}")
    return payload


@app.local_entrypoint()
def big(
    source: str = "mistralai/Mistral-7B-v0.3",
    target: str = "mistralai/Ministral-8B-Instruct-2410",
    layer_stride: int = 2,
    batch_size: int = 2,
    gpu: str = "A100-40GB",
    steps: int = 500,
) -> None:
    """Run a pair whose weights do not fit a 22 GiB card.

    Mistral-7B and Ministral-8B are 30.5 GB of bfloat16 between them, so the
    L4 the other entrypoints use is not an option. The functions are the same;
    only the accelerator is swapped.

    Only ridge and the trained residual are fitted. The attention-aligned
    solver is skipped deliberately: on a family without per-head query
    normalization its metric is severely ill-conditioned, and at alpha=1 it
    drove held-out R2 to -10.1 on Qwen2.5 and perplexity to six figures. It is
    a known failure, not something to pay to rediscover.

    ``steps`` defaults well below the Qwen3 run's 2000 because that run's loss
    was flat from roughly step 25; the extra steps bought nothing and here they
    would cost several dollars an hour more. Pass 0 to skip training entirely
    and measure the closed form on its own.
    """
    fetch_weights.remote([source, target])
    manifest = calibrate.with_options(gpu=gpu).remote(
        source=source, target=target, layer_stride=layer_stride,
        passes="split", batch_size=batch_size,
    )
    print(f"calibration finished in {manifest['total_seconds']}s")

    slug = f"{source.split('/')[-1].lower()}__to__{target.split('/')[-1].lower()}"
    artifacts = f"{slug}/" + "-".join(sorted(manifest["mixture"]))

    fit.remote(artifacts=artifacts, variants="ridge")

    # steps=0 evaluates the closed form alone. Worth doing first on an
    # unfamiliar pair: training is only interesting where ridge fails, and
    # finding that out costs a fraction of finding it out afterwards.
    residual_dir = ""
    if steps:
        train.with_options(gpu=gpu).remote(artifacts=artifacts, steps=steps)
        residual_dir = "residual"

    payload = evaluate.with_options(gpu=gpu).remote(
        artifacts=artifacts, maps="maps", residual=residual_dir,
    )
    for name, result in payload["perplexity"].items():
        print(f"  {name:10s} ppl={result['perplexity']:.4f}")
    for task, block in payload["results"].items():
        print(f"\n{task}:")
        for name, cond in block["conditions"].items():
            ret = block["retention"].get(name)
            print(f"  {name:10s} acc={cond['accuracy']:.4f}"
                  + (f"  retention={ret:.1%}" if ret is not None else ""))


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
