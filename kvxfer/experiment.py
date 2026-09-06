"""Run one transfer experiment end to end from cached statistics.

The experiment lives here rather than inside a script so that the laptop CLI
and the Modal entrypoint execute the same code. A pair that only fits on a
rented GPU should not be measured by a second, subtly different path, or the
cross-pair comparison in the writeup means nothing.

Nothing here harvests: every fit reads the sufficient statistics written by
``scripts/calibrate.py``. The only GPU work is the evaluation itself.
"""

from __future__ import annotations

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
from kvxfer.solvers.whitened import clear_design_cache, solve_whitened
from kvxfer.stats import GramStats

VARIANTS = ("ridge", "whitened")


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
) -> tuple[dict, dict]:
    """Fit one map per target layer on a fixed layer selection.

    Args:
        metrics: the attention-induced metric, or ``None`` for isotropic ridge.
        alpha: how far that metric reshapes the penalty; ignored when
            ``metrics`` is ``None``.

    Returns:
        ``(maps, diagnostics)`` keyed by target layer.
    """
    maps: dict[int, object] = {}
    diagnostics: dict[int, dict] = {}

    for layer in range(target_geom.n_layers):
        selected = selection[layer]
        if metrics is None:
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
        }
        print(
            f"  [{label}] L{layer:02d} in-sample R2={fit.r2:.3f}  "
            f"held-out R2={diagnostics[layer]['held_out_r2']:.3f}",
            flush=True,
        )

    return maps, diagnostics


def _paired_nll_payload(ppl: dict) -> dict | None:
    """Summarise the paired perplexity comparison, if both solvers ran."""
    if not set(VARIANTS) <= set(ppl):
        return None
    paired = paired_nll(ppl["ridge"], ppl["whitened"])
    return {
        "mean_difference": paired.mean_difference,
        "stderr": paired.stderr,
        "t_statistic": paired.t_statistic,
        "n_documents": paired.n_documents,
        "n_better": paired.n_better,
    }


def _task_payload(outcome) -> dict:
    """Serialise one task's outcome, keeping per-item results for re-analysis."""
    paired = (
        outcome.paired_test("ridge", "whitened")
        if set(VARIANTS) <= set(outcome.conditions)
        else None
    )
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
        "paired_ridge_vs_whitened": (
            {
                "difference": paired.difference,
                "p_value": paired.p_value,
                "ridge_only": paired.a_only,
                "whitened_only": paired.b_only,
            }
            if paired is not None
            else None
        ),
    }


def run_experiment(artifacts: str | Path, config: ExperimentConfig) -> dict:
    """Fit both solvers from cached statistics and measure what they cost.

    Conditions compared: the target model's own prefill (the ceiling), a zeroed
    cache (the floor), the source model standalone, and each solver's mapped
    cache.

    Args:
        artifacts: directory written by ``scripts/calibrate.py``.
        config: the experiment definition.

    Returns:
        The results payload, ready to serialise.
    """
    art = Path(artifacts)
    manifest = json.loads((art / "manifest.json").read_text())
    source_id, target_id = manifest["source"], manifest["target"]
    print(f"pair: {source_id} -> {target_id}")

    target_geom = load_geometry(target_id)

    stats = {
        f"{split}_{kind}": GramStats.load(art / f"{split}_{kind}.pt")
        for split in ("fit", "val")
        for kind in ("keys", "values")
    }
    print(
        f"statistics: {stats['fit_keys'].n_tokens:,} fit tokens, "
        f"{stats['val_keys'].n_tokens:,} val tokens, "
        f"{stats['fit_keys'].n_source_layers} candidate source layers"
    )

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

    mappers: dict[str, object] = {"floor": ZeroMapper(target_geom)}
    diagnostics: dict[str, dict] = {}

    # Selected once and shared, so the variants differ only in objective.
    print("\nselecting source layers (out of sample, shared by all variants)")
    selection = {
        kind: choose_layers(
            stats[f"fit_{kind}"],
            stats[f"val_{kind}"],
            target_geom.n_layers,
            config.k,
            config.lam,
            config.n_candidates,
            f"select/{kind[0].upper()}",
            config.selection,
        )
        for kind in ("keys", "values")
    }

    for name in VARIANTS:
        penalties = {kind: config.penalty(name, kind) for kind in ("keys", "values")}
        exponents = {kind: config.metric_exponent(kind) for kind in ("keys", "values")}
        described = ", ".join(
            f"{kind} lambda={penalties[kind]:.0e}"
            + (f" alpha={exponents[kind]:.2f}" if name == "whitened" else "")
            for kind in ("keys", "values")
        )
        print(f"\nfitting {name} ({described})")
        started = time.time()
        fitted = {}
        for kind in ("keys", "values"):
            fitted[kind] = fit_all_layers(
                stats[f"fit_{kind}"],
                stats[f"val_{kind}"],
                target_geom,
                metrics[name][kind],
                selection[kind],
                penalties[kind],
                f"{name}/{kind[0].upper()}",
                alpha=exponents[kind],
            )
        clear_design_cache()
        mappers[name] = FittedMapper(
            fitted["keys"][0], fitted["values"][0], target_geom, label=name
        )
        diagnostics[name] = {
            "lambda": penalties,
            "alpha": exponents if name == "whitened" else None,
            "keys": fitted["keys"][1],
            "values": fitted["values"][1],
            "fit_seconds": round(time.time() - started, 1),
        }
        print(f"  fitted in {diagnostics[name]['fit_seconds']}s")

    source_model = load_model(source_id, dtype=dtype)

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
        mappers=mappers,
        dtype=dtype,
        batch_size=config.ppl_batch_size,
    )
    print()
    for name, res in ppl.items():
        print(
            f"    {name:10s} ppl={res.perplexity:8.4f}  "
            f"nll={res.mean_nll:.5f} +/- {res.stderr():.5f}"
        )
    if set(VARIANTS) <= set(ppl):
        print(f"    paired: {paired_nll(ppl['ridge'], ppl['whitened'])}")

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
            mappers=mappers,
            dtype=dtype,
        )
        results[task_name] = outcome

        print(f"\n  {task_name}:")
        for name, cond in outcome.conditions.items():
            line = f"    {name:10s} acc={cond.accuracy:.4f} +/- {cond.stderr():.4f}"
            if name not in ("target", "floor", "source"):
                line += (
                    f"  retention={outcome.retention(name):.1%}"
                    f"  floor-normalized={outcome.floor_normalized_retention(name):.1%}"
                )
            print(line)

        # The comparison that matters is paired: conditions are scored on
        # identical items, so per-condition standard errors overstate the
        # uncertainty of the difference between them.
        if set(VARIANTS) <= set(outcome.conditions):
            print(f"    paired: {outcome.paired_test('ridge', 'whitened')}")

    return {
        "source": source_id,
        "target": target_id,
        "settings": asdict(config),
        "n_fit_tokens": stats["fit_keys"].n_tokens,
        "n_val_tokens": stats["val_keys"].n_tokens,
        "perplexity": {
            name: {
                "perplexity": res.perplexity,
                "mean_nll": res.mean_nll,
                "stderr": res.stderr(),
                "n_tokens": res.n_tokens,
                "document_nll": res.document_nll,
            }
            for name, res in ppl.items()
        },
        "perplexity_paired_ridge_vs_whitened": _paired_nll_payload(ppl),
        "layer_selection": {
            kind: {str(layer): list(v) for layer, v in sel.items()}
            for kind, sel in selection.items()
        },
        "diagnostics": diagnostics,
        "results": {task: _task_payload(outcome) for task, outcome in results.items()},
    }
