"""Run one transfer experiment end to end from cached statistics.

The experiment lives here rather than inside a script so that the laptop CLI
and the Modal entrypoint execute the same code. A pair that only fits on a
rented GPU should not be measured by a second, subtly different path, or the
cross-pair comparison in the writeup means nothing.

Nothing here harvests: every fit reads the sufficient statistics written by
``scripts/calibrate.py``. The only GPU work is the evaluation itself.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from kvxfer.data import build_calibration
from kvxfer.eval.ppl import evaluate_perplexity, paired_nll
from kvxfer.eval.retention import evaluate_task
from kvxfer.eval.tasks import load_task
from kvxfer.geometry import KVGeometry, load_geometry
from kvxfer.mappers import FittedMapper, ZeroMapper
from kvxfer.metrics import HeadMetrics, key_metric, value_metric
from kvxfer.models import load_model, load_tokenizer
from kvxfer.queries import collect_query_moments
from kvxfer.solvers.ridge import (
    held_out_r2,
    select_source_layers,
    select_top_k,
    solve_ridge,
)
from kvxfer.solvers.lowrank import solve_low_rank, stored_parameters
from kvxfer.solvers.whitened import clear_design_cache, solve_whitened
from kvxfer.stats import GramStats

# The two full-rank conditions, always run: the reference method and the
# attention-aligned objective. Rank-constrained variants are added on request.
VARIANTS = ("ridge", "whitened")
KINDS = ("keys", "values")


def variant_plan(config: "ExperimentConfig") -> dict[str, tuple[str, int]]:
    """Which maps to fit: name -> (metric to use, rank budget).

    The rank-constrained pair is deliberately isotropic-versus-aligned at the
    *same* rank. That isolates what the metric contributes from what the
    truncation contributes, which a comparison against the full-rank map would
    confound.
    """
    plan = {"ridge": ("ridge", 0), "whitened": ("whitened", 0)}
    if config.rank:
        plan[f"rank{config.rank}_iso"] = ("ridge", config.rank)
        plan[f"rank{config.rank}_aligned"] = ("whitened", config.rank)
    return plan


@dataclass
class ExperimentConfig:
    """Everything that defines an experiment except where it runs.

    ``lam`` is the fallback penalty and the one used for layer selection, which
    is deliberately shared across variants. ``lambdas`` overrides it per
    (variant, kind), which is how the penalty sweep's answer gets used: each
    solver is entitled to its own best penalty because each was tuned on its
    own objective, and forcing them to share one would hand the comparison to
    whichever solver that single value happened to suit.
    """

    tasks: tuple[str, ...] = ("arc_easy", "arc_challenge")
    limit: int = 300
    ppl_documents: int = 64
    ppl_seq_len: int = 512
    ppl_prefix: int = 256
    ppl_batch_size: int = 4
    calibration_sequences: int = 640
    k: int = 4
    lam: float = 1e-3
    lambdas: dict[str, dict[str, float]] = field(default_factory=dict)
    alpha: float = 1.0
    alphas: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    n_candidates: int = 6
    selection: str = "topk"
    metric_offset: int = 2048
    query_sequences: int = 32
    dtype: str = "float32"

    def penalty(self, variant: str, kind: str) -> float:
        """Penalty for one solver on one cache kind, falling back to ``lam``."""
        return float(self.lambdas.get(variant, {}).get(kind, self.lam))

    def metric_exponent(self, kind: str) -> float:
        """How far the metric reshapes the penalty for one cache kind."""
        return float(self.alphas.get(kind, self.alpha))


def settings_from_sweep(
    path: str | Path,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Read the selected penalties and metric exponents from a sweep report.

    Args:
        path: JSON written by ``scripts/sweep_lambda.py``.

    Returns:
        ``({variant: {kind: lambda}}, {kind: alpha})``. Reports written before
        alpha was swept carry no exponent, and those cache kinds are left out
        so the config's default applies.
    """
    report = json.loads(Path(path).read_text())
    lambdas: dict[str, dict[str, float]] = {name: {} for name in VARIANTS}
    alphas: dict[str, float] = {}
    for kind, sweep in report["sweeps"].items():
        lambdas["ridge"][kind] = float(sweep["best_ridge_lambda"])
        lambdas["whitened"][kind] = float(sweep["best_whitened_lambda"])
        if "best_whitened_alpha" in sweep:
            alphas[kind] = float(sweep["best_whitened_alpha"])
    return lambdas, alphas


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
    target_geom: KVGeometry,
    metrics: HeadMetrics | None,
    selection: dict[int, tuple[int, ...]],
    lam: float,
    label: str,
    alpha: float = 1.0,
    rank: int = 0,
) -> tuple[dict, dict]:
    """Fit one map per target layer on a fixed layer selection.

    Args:
        metrics: the attention-induced metric, or ``None`` for isotropic ridge.
        alpha: how far that metric reshapes the penalty; ignored when
            ``metrics`` is ``None``.
        rank: if positive, constrain each map to this rank, with ``metrics``
            deciding which directions survive.

    Returns:
        ``(maps, diagnostics)`` keyed by target layer.
    """
    maps: dict[int, object] = {}
    diagnostics: dict[int, dict] = {}

    for layer in range(target_geom.n_layers):
        selected = selection[layer]
        if rank:
            fit = solve_low_rank(
                fit_stats,
                selected,
                layer,
                rank=rank,
                metrics=metrics,
                n_kv_heads=target_geom.n_kv_heads,
                head_dim=target_geom.head_dim,
                lam=lam,
            )
        elif metrics is None:
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
                alpha=alpha,
            )
        maps[layer] = fit
        diagnostics[layer] = {
            "source_layers": list(selected),
            "in_sample_r2": round(fit.r2, 4),
            "held_out_r2": round(float(held_out_r2(val_stats, fit).mean()), 4),
            "stored_parameters": stored_parameters(fit),
        }
        print(
            f"  [{label}] L{layer:02d} in-sample R2={fit.r2:.3f}  "
            f"held-out R2={diagnostics[layer]['held_out_r2']:.3f}",
            flush=True,
        )

    return maps, diagnostics


def _comparisons(names) -> list[tuple[str, str]]:
    """Which conditions to test against each other, as (baseline, contender).

    Every comparison is isotropic-versus-aligned at otherwise identical
    settings, because that difference is the thing under test. Comparing across
    ranks, or against the floor, would measure something nobody disputes.
    """
    names = set(names)
    pairs = [("ridge", "whitened")]
    pairs += [
        (iso, iso.replace("_iso", "_aligned"))
        for iso in sorted(names)
        if iso.endswith("_iso") and iso.replace("_iso", "_aligned") in names
    ]
    return [pair for pair in pairs if set(pair) <= names]


def _paired_nll_payload(ppl: dict) -> dict:
    """Summarise every paired perplexity comparison that can be made."""
    payload = {}
    for baseline, contender in _comparisons(ppl):
        paired = paired_nll(ppl[baseline], ppl[contender])
        payload[f"{baseline}_vs_{contender}"] = {
            "mean_difference": paired.mean_difference,
            "stderr": paired.stderr,
            "t_statistic": paired.t_statistic,
            "n_documents": paired.n_documents,
            "n_better": paired.n_better,
        }
    return payload


def _task_payload(outcome) -> dict:
    """Serialise one task's outcome, keeping per-item results for re-analysis."""
    paired = {}
    for baseline, contender in _comparisons(outcome.conditions):
        test = outcome.paired_test(baseline, contender)
        paired[f"{baseline}_vs_{contender}"] = {
            "difference": test.difference,
            "p_value": test.p_value,
            f"{baseline}_only": test.a_only,
            f"{contender}_only": test.b_only,
        }
    return {
        "conditions": {
            name: {
                "accuracy": cond.accuracy,
                "accuracy_normalized": cond.accuracy_normalized,
                "stderr": cond.stderr(),
                "n_items": cond.n_items,
                "outcomes": [int(v) for v in cond.outcomes],
            }
            for name, cond in outcome.conditions.items()
        },
        "retention": {
            name: outcome.retention(name)
            for name in outcome.conditions
            if name != "target"
        },
        "floor_normalized_retention": {
            name: outcome.floor_normalized_retention(name)
            for name in outcome.conditions
            if name not in ("target", "floor")
        },
        "paired": paired,
    }


def fit_mappers(
    artifacts: str | Path, config: ExperimentConfig
) -> tuple[dict[str, dict[str, dict[int, object]]], dict]:
    """Fit every variant's maps from cached statistics.

    Holds no models beyond the one pass needed to build the attention metrics,
    and no statistics beyond the cache kind being fitted. Kept separate from
    evaluation because the two have disjoint working sets: overlapping them
    made the peak their sum, which a 16 GB machine does not survive.

    Args:
        artifacts: directory written by ``scripts/calibrate.py``.
        config: the experiment definition.

    Returns:
        ``(maps, metadata)`` where maps is ``{variant: {kind: {layer: map}}}``.
    """
    art = Path(artifacts)
    manifest = json.loads((art / "manifest.json").read_text())
    source_id, target_id = manifest["source"], manifest["target"]
    print(f"pair: {source_id} -> {target_id}")

    target_geom = load_geometry(target_id)
    dtype = getattr(torch, config.dtype)
    tokenizer = load_tokenizer(source_id)
    target_model = load_model(target_id, dtype=dtype)

    print("\ncollecting query second moments for the key metric")
    probe = build_calibration(tokenizer, seq_len=512, n_sequences=config.query_sequences)
    q_moments = collect_query_moments(
        target_model, target_geom, probe, n_sequences=config.query_sequences
    )
    metrics = {
        "ridge": {"keys": None, "values": None},
        "whitened": {
            "keys": key_metric(
                q_moments,
                target_geom.head_dim,
                target_geom.rope_theta,
                max_offset=config.metric_offset,
            ).normalized(),
            "values": value_metric(target_model, target_geom).normalized(),
        },
    }

    # The target model is not needed again here, and the fitting that follows
    # wants the memory for statistics.
    del q_moments, probe, target_model, tokenizer
    gc.collect()

    plan = variant_plan(config)
    maps: dict[str, dict[str, dict[int, object]]] = {name: {} for name in plan}
    selection: dict[str, dict[int, tuple[int, ...]]] = {}
    diagnostics: dict[str, dict] = {name: {} for name in plan}
    elapsed = dict.fromkeys(plan, 0.0)
    tokens = {}

    # One cache kind at a time, released as soon as its maps exist. Nothing
    # after this reads the statistics: held-out scoring happens here, while
    # they are still open.
    for kind in KINDS:
        fit_stats = GramStats.load(art / f"fit_{kind}.pt")
        val_stats = GramStats.load(art / f"val_{kind}.pt")
        tokens[kind] = (fit_stats.n_tokens, val_stats.n_tokens)
        print(
            f"\n[{kind}] {fit_stats.n_tokens:,} fit tokens, "
            f"{val_stats.n_tokens:,} held out, "
            f"{fit_stats.n_source_layers} candidate source layers"
        )

        # Selected once per kind and shared, so variants differ only in
        # objective and never in which predictors they were handed.
        selection[kind] = choose_layers(
            fit_stats,
            val_stats,
            target_geom.n_layers,
            config.k,
            config.lam,
            config.n_candidates,
            f"select/{kind[0].upper()}",
            config.selection,
        )
        gc.collect()

        for name, (metric_name, rank) in plan.items():
            penalty = config.penalty(metric_name, kind)
            exponent = config.metric_exponent(kind)
            described = f"lambda={penalty:.0e}"
            if metric_name == "whitened" and not rank:
                described += f", alpha={exponent:.2f}"
            if rank:
                described += f", rank={rank}"
            print(f"\nfitting {name} on {kind} ({described})")

            started = time.time()
            layer_maps, layer_diag = fit_all_layers(
                fit_stats,
                val_stats,
                target_geom,
                metrics[metric_name][kind],
                selection[kind],
                penalty,
                f"{name}/{kind[0].upper()}",
                alpha=exponent,
                rank=rank,
            )
            elapsed[name] += time.time() - started
            maps[name][kind] = layer_maps
            diagnostics[name][kind] = layer_diag
            clear_design_cache()
            gc.collect()

        del fit_stats, val_stats
        gc.collect()

    for name, (metric_name, rank) in plan.items():
        stored = sum(
            layer["stored_parameters"]
            for kind in KINDS
            for layer in diagnostics[name][kind].values()
        )
        diagnostics[name]["lambda"] = {
            kind: config.penalty(metric_name, kind) for kind in KINDS
        }
        diagnostics[name]["alpha"] = (
            {kind: config.metric_exponent(kind) for kind in KINDS}
            if metric_name == "whitened" and not rank
            else None
        )
        diagnostics[name]["rank"] = rank or None
        diagnostics[name]["stored_parameters"] = stored
        diagnostics[name]["fit_seconds"] = round(elapsed[name], 1)
        print(
            f"  {name}: fitted in {elapsed[name]:.1f}s, "
            f"{stored / 1e6:.1f}M stored parameters"
        )

    metadata = {
        "source": source_id,
        "target": target_id,
        "settings": asdict(config),
        "n_fit_tokens": tokens["keys"][0],
        "n_val_tokens": tokens["keys"][1],
        "layer_selection": {
            kind: {str(layer): list(v) for layer, v in sel.items()}
            for kind, sel in selection.items()
        },
        "diagnostics": diagnostics,
    }
    return maps, metadata


def evaluate_mappers(
    mappers: dict[str, object],
    metadata: dict,
    config: ExperimentConfig,
) -> dict:
    """Measure what a set of fitted mappers costs downstream.

    Conditions compared: the target model's own prefill (the ceiling), a zeroed
    cache (the floor), the source model standalone, and each fitted mapper.

    Args:
        mappers: fitted mappers by name; the floor is added here.
        metadata: what :func:`fit_mappers` returned alongside the maps.
        config: the experiment definition.

    Returns:
        The results payload, ready to serialise.
    """
    source_id, target_id = metadata["source"], metadata["target"]
    target_geom = load_geometry(target_id)
    dtype = getattr(torch, config.dtype)

    tokenizer = load_tokenizer(source_id)
    target_model = load_model(target_id, dtype=dtype)
    source_model = load_model(source_id, dtype=dtype)

    conditions: dict[str, object] = {"floor": ZeroMapper(target_geom)}
    conditions.update(mappers)

    # Perplexity first: it yields one measurement per token rather than one per
    # item, which is the difference between resolving a mapper's effect and not.
    scored = config.ppl_seq_len - config.ppl_prefix
    print(
        f"\nprefix-conditioned perplexity: {config.ppl_documents} documents, "
        f"{config.ppl_prefix} prefix + {scored} scored tokens"
    )
    eval_docs = build_calibration(
        tokenizer,
        seq_len=config.ppl_seq_len,
        n_sequences=config.ppl_documents,
        skip=config.calibration_sequences,
    )
    ppl = evaluate_perplexity(
        eval_docs.input_ids,
        config.ppl_prefix,
        target_model=target_model,
        source_model=source_model,
        mappers=conditions,
        dtype=dtype,
        batch_size=config.ppl_batch_size,
    )
    print()
    for name, res in ppl.items():
        print(
            f"    {name:16s} ppl={res.perplexity:8.4f}  "
            f"nll={res.mean_nll:.5f} +/- {res.stderr():.5f}"
        )
    for baseline, contender in _comparisons(ppl):
        print(
            f"    paired {baseline} vs {contender}: "
            f"{paired_nll(ppl[baseline], ppl[contender])}"
        )

    results = {}
    for task_name in config.tasks:
        print(f"\nevaluating {task_name} ({config.limit} items)")
        examples = load_task(task_name, limit=config.limit)
        outcome = evaluate_task(
            task_name,
            examples,
            target_model,
            tokenizer,
            source_model=source_model,
            mappers=conditions,
            dtype=dtype,
        )
        results[task_name] = outcome

        print(f"\n  {task_name}:")
        for name, cond in outcome.conditions.items():
            line = f"    {name:16s} acc={cond.accuracy:.4f} +/- {cond.stderr():.4f}"
            if name not in ("target", "floor", "source"):
                line += (
                    f"  retention={outcome.retention(name):.1%}"
                    f"  floor-normalized={outcome.floor_normalized_retention(name):.1%}"
                )
            print(line)

        # The comparison that matters is paired: conditions are scored on
        # identical items, so per-condition standard errors overstate the
        # uncertainty of the difference between them.
        for baseline, contender in _comparisons(outcome.conditions):
            print(
                f"    paired {baseline} vs {contender}: "
                f"{outcome.paired_test(baseline, contender)}"
            )

    payload = dict(metadata)
    payload["perplexity"] = {
        name: {
            "perplexity": res.perplexity,
            "mean_nll": res.mean_nll,
            "stderr": res.stderr(),
            "n_tokens": res.n_tokens,
            "document_nll": res.document_nll,
        }
        for name, res in ppl.items()
    }
    payload["perplexity_paired"] = _paired_nll_payload(ppl)
    payload["results"] = {
        task: _task_payload(outcome) for task, outcome in results.items()
    }
    return payload


def run_experiment(artifacts: str | Path, config: ExperimentConfig) -> dict:
    """Fit and evaluate in one process.

    Convenient where memory is not the binding constraint. On a machine where
    it is, run ``scripts/fit_maps.py`` and ``scripts/fit_and_eval.py --maps``
    instead, so the two working sets never coexist.
    """
    maps, metadata = fit_mappers(artifacts, config)
    target_geom = load_geometry(metadata["target"])
    mappers = {
        name: FittedMapper(by_kind["keys"], by_kind["values"], target_geom, label=name)
        for name, by_kind in maps.items()
    }
    del maps
    gc.collect()
    return evaluate_mappers(mappers, metadata, config)
