"""Rank-constrained KV mapping, with the attention metric choosing the rank.

A full map is not free. For Qwen3-0.6B to 1.7B with four source layers it is
``4096 x 1024`` per target layer per cache kind: about 235M parameters across
the model, roughly 470 MB in half precision, against a 1.2 GB source model.
A method whose stated purpose is to avoid recomputing prefill should say what
it costs to carry, and the reference work does not.

Constraining the rank makes that cost a dial. It also puts the attention
metric where it has first-order leverage. As a penalty reweighting the metric
competes with the design's conditioning and loses (see
:mod:`kvxfer.solvers.whitened`): it decides how hard to fit directions that
were all going to be fitted anyway. Under a hard rank budget something must be
discarded, and then the only question is *which* directions -- which is exactly
what a metric answers.

The solve stays closed form. Writing ``W*`` for the ridge solution, the
rank-``r`` minimizer of ``||(XW - Y) M^{1/2}||_F`` is::

    W_r = W* M^{1/2} V_r V_r' M^{-1/2}

where ``V_r`` holds the top ``r`` eigenvectors of ``M^{1/2} W*' X'X W* M^{1/2}``
-- the fitted signal, measured in the metric. With ``M = I`` this is ordinary
reduced-rank regression, so the isotropic baseline is the same code with a
different metric, and the comparison is not confounded by implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from kvxfer.metrics import HeadMetrics
from kvxfer.solvers.ridge import (
    LinearMap,
    _centered_moments,
    resolve_lambda,
)
from kvxfer.solvers.whitened import _block_diagonal_basis


def _metric_roots(
    metrics: HeadMetrics,
    target_layer: int,
    n_kv_heads: int,
    head_dim: int,
    eigenvalue_floor: float,
) -> tuple[Tensor, Tensor]:
    """Return ``(M^{1/2}, M^{-1/2})`` for one target layer.

    The inverse root is why the floor is not optional here: directions
    attention cannot see have near-zero metric eigenvalues, and inverting them
    would amplify precisely the directions the metric declared irrelevant.
    """
    evals, basis = _block_diagonal_basis(metrics, target_layer, n_kv_heads, head_dim)

    per_head = evals.reshape(n_kv_heads, head_dim)
    floor = eigenvalue_floor * per_head.mean(dim=1, keepdim=True)
    evals = torch.maximum(per_head, floor).reshape(-1)

    root = evals.sqrt()
    half = (basis * root) @ basis.T
    inv_half = (basis / root) @ basis.T
    return half, inv_half


def solve_low_rank(
    stats,
    layers: tuple[int, ...],
    target_layer: int,
    rank: int,
    metrics: HeadMetrics | None = None,
    n_kv_heads: int = 0,
    head_dim: int = 0,
    lam: float = 1e-3,
    relative_lambda: bool = True,
    eigenvalue_floor: float = 1e-6,
) -> LinearMap:
    """Fit a rank-constrained map for one target layer.

    Args:
        stats: accumulated calibration statistics.
        layers: source layers forming the design.
        target_layer: the target layer to predict.
        rank: how many directions the map may span. At full rank this returns
            the ridge solution exactly, whatever the metric.
        metrics: the metric deciding which directions to keep. ``None`` gives
            ordinary reduced-rank regression, the isotropic control.
        n_kv_heads: target KV head count; required when ``metrics`` is given.
        head_dim: target head dimension; required when ``metrics`` is given.
        lam: ridge penalty applied before truncation.
        relative_lambda: see :func:`kvxfer.solvers.ridge.resolve_lambda`.
        eigenvalue_floor: clamp on metric eigenvalues, as a fraction of each
            head's mean.

    Returns:
        The fitted :class:`LinearMap`, carrying its ``rank`` and the factors
        that make the storage saving real rather than notional.

    Raises:
        ValueError: if ``rank`` is not positive, or the metric's shape does not
            match the declared head geometry.
    """
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")

    xtx_c, xty_c, x_mean, y_mean, tss = _centered_moments(stats, layers, target_layer)
    kv_dim = xty_c.shape[1]
    rank = min(rank, kv_dim)

    lam_abs = resolve_lambda(xtx_c, lam, relative_lambda)
    ridge = xtx_c + lam_abs * torch.eye(xtx_c.shape[0], dtype=xtx_c.dtype)
    full = torch.linalg.solve(ridge, xty_c)

    if metrics is None:
        half = inv_half = torch.eye(kv_dim, dtype=xtx_c.dtype)
    else:
        if n_kv_heads * head_dim != kv_dim:
            raise ValueError(
                f"target width {kv_dim} does not match {n_kv_heads}x{head_dim}"
            )
        half, inv_half = _metric_roots(
            metrics, target_layer, n_kv_heads, head_dim, eigenvalue_floor
        )

    # The fitted signal, seen through the metric. Its leading eigenvectors are
    # the directions worth spending rank on.
    whitened_fit = half @ (full.T @ xtx_c @ full) @ half
    whitened_fit = 0.5 * (whitened_fit + whitened_fit.T)
    _, vecs = torch.linalg.eigh(whitened_fit)
    kept = vecs[:, -rank:]

    # Factored, so the saving is in the stored object and not only in the
    # rank of a matrix that is still dense on disk.
    left = full @ half @ kept                  # (D, rank)
    right = kept.T @ inv_half                  # (rank, kv_dim)
    weight = left @ right
    bias = y_mean - x_mean @ weight

    rss = tss - 2.0 * (weight * xty_c).sum(0) + (weight * (xtx_c @ weight)).sum(0)
    r2 = (1.0 - rss / tss.clamp_min(1e-12)).clamp(min=-1.0, max=1.0)

    return LinearMap(
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
        source_layers=tuple(layers),
        target_layer=target_layer,
        r2=float(r2.mean()),
        rank=rank,
        factors=(left.to(torch.float32), right.to(torch.float32)),
    )


def stored_parameters(fit: LinearMap) -> int:
    """Parameters actually needed to carry a map, honouring its factorization."""
    if fit.factors is None:
        return fit.weight.numel() + fit.bias.numel()
    left, right = fit.factors
    return left.numel() + right.numel() + fit.bias.numel()
