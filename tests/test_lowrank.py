"""Rank-constrained mapping, and whether the metric earns its place there.

The attention metric failed to help as a penalty reweighting. The claim this
module tests is narrower and structural: under a *hard* rank budget, something
must be discarded, and a metric is exactly the object that says what. If
metric-weighted truncation does not beat isotropic truncation at matched rank
on metric-weighted error, the idea has no support and should be dropped.
"""

from __future__ import annotations

import pytest
import torch

from kvxfer.geometry import KVGeometry
from kvxfer.metrics import HeadMetrics, identity_metrics
from kvxfer.solvers.lowrank import solve_low_rank, stored_parameters
from kvxfer.solvers.ridge import solve_ridge
from kvxfer.stats import GramAccumulator

N_KV_HEADS = 2
HEAD_DIM = 8
KV_DIM = N_KV_HEADS * HEAD_DIM
N_SRC_LAYERS = 3
N_TGT_LAYERS = 2
LAM = 1e-2


def _geom(n_layers: int) -> KVGeometry:
    return KVGeometry(
        model_id="synthetic",
        n_layers=n_layers,
        n_q_heads=N_KV_HEADS * 2,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=64,
        rope_theta=1e6,
    )


def _stats(data_seed: int = 0, n_tokens: int = 8000, noise: float = 0.4):
    """Draw one split from a *fixed* ground truth.

    The ground truth is seeded separately from the data. Reseeding both would
    make the validation split a different regression problem, and every
    generalization claim measured against it would be meaningless.
    """
    dim_in = N_SRC_LAYERS * KV_DIM
    truth = torch.Generator().manual_seed(1234)
    w_true = torch.randn(dim_in, KV_DIM, generator=truth, dtype=torch.float64)
    w_true /= dim_in**0.5

    draw = torch.Generator().manual_seed(data_seed)
    acc = GramAccumulator(_geom(N_SRC_LAYERS), _geom(N_TGT_LAYERS), kind="keys")
    for _ in range(4):
        rows = n_tokens // 4
        x = torch.randn(rows, dim_in, generator=draw, dtype=torch.float64)
        y0 = x @ w_true + noise * torch.randn(
            rows, KV_DIM, generator=draw, dtype=torch.float64
        )
        other = torch.randn(rows, KV_DIM, generator=draw, dtype=torch.float64)
        acc.update(x, torch.stack([y0, other], dim=1))
    return acc.finalize()


def _anisotropic(seed: int = 5) -> HeadMetrics:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(
        N_TGT_LAYERS, N_KV_HEADS, HEAD_DIM, HEAD_DIM, generator=g, dtype=torch.float64
    )
    return HeadMetrics(a @ a.transpose(-1, -2) / HEAD_DIM, "keys").normalized()


def _metric_error(stats, fit, metrics, target_layer: int = 0) -> float:
    """Metric-weighted residual on ``stats``, computed from moments."""
    xtx, xty, x_sum = stats.select(tuple(fit.source_layers))
    cpu64 = dict(device="cpu", dtype=torch.float64)
    xtx, x_sum = xtx.to(**cpu64), x_sum.to(**cpu64)
    xty = xty[:, target_layer].to(**cpu64)
    yty = stats.yty_head[target_layer].to(**cpu64)
    y_sum = stats.y_sum[target_layer].to(**cpu64)
    n = float(stats.n_tokens)

    w = fit.weight.to(**cpu64)
    b = fit.bias.to(**cpu64)

    # E[(Xw + b - y)(Xw + b - y)'] assembled from moments, then read under M.
    second = (
        w.T @ xtx @ w
        + torch.outer(b, x_sum @ w)
        + torch.outer(x_sum @ w, b)
        + n * torch.outer(b, b)
        - w.T @ xty
        - xty.T @ w
        - torch.outer(b, y_sum)
        - torch.outer(y_sum, b)
    )
    blocks = second.reshape(N_KV_HEADS, HEAD_DIM, N_KV_HEADS, HEAD_DIM)
    total = 0.0
    for head in range(N_KV_HEADS):
        block = blocks[head, :, head, :] + yty[head]
        total += float((block * metrics.matrices[target_layer, head].to(**cpu64)).sum())
    return total / n


def test_full_rank_reproduces_ridge_for_any_metric():
    """At full rank the constraint is vacuous, so the metric must not matter."""
    stats = _stats()
    layers = tuple(range(N_SRC_LAYERS))

    plain = solve_ridge(stats, layers, 0, lam=LAM)
    full = solve_low_rank(
        stats, layers, 0, rank=KV_DIM, metrics=_anisotropic(),
        n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, lam=LAM,
    )

    assert torch.allclose(plain.weight, full.weight, atol=1e-6), (
        f"max difference {(plain.weight - full.weight).abs().max():.2e}"
    )
    assert torch.allclose(plain.bias, full.bias, atol=1e-6)


def test_factors_reconstruct_the_weight_and_shrink_storage():
    """The saving must be in the stored object, not only in the matrix rank."""
    stats = _stats()
    layers = tuple(range(N_SRC_LAYERS))
    rank = 4

    fit = solve_low_rank(
        stats, layers, 0, rank=rank, metrics=_anisotropic(),
        n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, lam=LAM,
    )
    left, right = fit.factors

    assert fit.rank == rank
    assert torch.allclose(left @ right, fit.weight, atol=1e-5)
    # Relative tolerance: the weight is stored in float32, so the singular
    # values past the true rank sit at that precision rather than at zero.
    assert torch.linalg.matrix_rank(fit.weight.double(), rtol=1e-5) <= rank
    assert stored_parameters(fit) < fit.weight.numel()


def test_identity_metric_matches_ordinary_reduced_rank_regression():
    """Passing M = I must equal passing no metric at all."""
    stats = _stats()
    layers = tuple(range(N_SRC_LAYERS))
    shared = dict(rank=5, lam=LAM)

    isotropic = solve_low_rank(stats, layers, 0, metrics=None, **shared)
    explicit = solve_low_rank(
        stats, layers, 0, metrics=identity_metrics(_geom(N_TGT_LAYERS), "keys"),
        n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, **shared,
    )

    assert torch.allclose(isotropic.weight, explicit.weight, atol=1e-6)


def test_metric_truncation_beats_isotropic_truncation_at_matched_rank():
    """The claim: under a rank budget, the metric picks better directions.

    Scored on held-out moments, so this is generalization and not a restatement
    of the objective. Ranks near full are excluded: there the constraint barely
    binds and the two solutions coincide by construction.
    """
    fit_stats = _stats(data_seed=0)
    val_stats = _stats(data_seed=99, n_tokens=4000)
    layers = tuple(range(N_SRC_LAYERS))
    metrics = _anisotropic()

    for rank in (2, 4, 6):
        aligned = solve_low_rank(
            fit_stats, layers, 0, rank=rank, metrics=metrics,
            n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, lam=LAM,
        )
        isotropic = solve_low_rank(fit_stats, layers, 0, rank=rank, lam=LAM)

        aligned_error = _metric_error(val_stats, aligned, metrics)
        isotropic_error = _metric_error(val_stats, isotropic, metrics)
        assert aligned_error < isotropic_error, (
            f"rank {rank}: metric-aligned {aligned_error:.6f} "
            f"did not beat isotropic {isotropic_error:.6f}"
        )


def test_rank_must_be_positive():
    stats = _stats()
    with pytest.raises(ValueError, match="rank must be positive"):
        solve_low_rank(stats, tuple(range(N_SRC_LAYERS)), 0, rank=0)
