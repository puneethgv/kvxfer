"""Attention-aligned closed-form KV mapping.

Plain ridge minimizes reconstruction error isotropically, weighting every
coordinate of a head equally. The reference work's own diagnostics show that
this is the wrong target: calibration R2 *anti*-correlates with downstream
retention, and what predicts retention is whether the error lands in
directions attention is sensitive to.

This solver minimizes error under that sensitivity metric instead::

    min_W  sum_i (x_i W + b - y_i) M (x_i W + b - y_i)'  +  lambda ||W||^2

with ``M`` from :mod:`kvxfer.metrics`. The key property is that this stays
closed-form. Diagonalizing ``M = U L U'`` and rotating the targets into that
basis decouples the problem into independent ridge regressions, one per output
coordinate, each with penalty ``lambda / L_j``::

    (X'X + (lambda / L_j) I) w_j = X' y_j

So the objective change amounts to a per-direction reweighting of the penalty:
directions the model's queries actually probe are fitted hard, directions
attention cannot see are shrunk away. A single eigendecomposition of the design
Gram serves every output coordinate, so the cost is essentially that of the
isotropic solve.

Setting ``M = I`` recovers plain ridge exactly, which is asserted in the tests.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor

from kvxfer.metrics import HeadMetrics
from kvxfer.solvers.ridge import LinearMap, _centered_moments, resolve_lambda
from kvxfer.stats import GramStats


class _EighCache:
    """Caches design-Gram eigendecompositions across target layers.

    Different target layers frequently select overlapping source layers, and
    the eigendecomposition is by far the dominant cost of the solve, so reusing
    it across layers that chose the same subset matters more than it looks.
    """

    def __init__(self, maxsize: int = 8) -> None:
        self._store: dict[tuple, tuple[Tensor, Tensor]] = {}
        self._order: list[tuple] = []
        self._maxsize = maxsize

    def get(self, key: tuple, matrix: Tensor) -> tuple[Tensor, Tensor]:
        if key in self._store:
            return self._store[key]
        evals, evecs = torch.linalg.eigh(matrix)
        evals = evals.clamp_min(0.0)
        self._store[key] = (evals, evecs)
        self._order.append(key)
        if len(self._order) > self._maxsize:
            del self._store[self._order.pop(0)]
        return evals, evecs

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()


_DESIGN_CACHE = _EighCache()


def _block_diagonal_basis(
    metrics: HeadMetrics, target_layer: int, n_kv_heads: int, head_dim: int
) -> tuple[Tensor, Tensor]:
    """Assemble the metric's eigenbasis for one target layer.

    The metric is per-head, so the full ``kv_dim x kv_dim`` basis is block
    diagonal. Returns ``(eigenvalues, basis)`` shaped ``(kv_dim,)`` and
    ``(kv_dim, kv_dim)``.
    """
    kv_dim = n_kv_heads * head_dim
    evals = torch.zeros(kv_dim, dtype=torch.float64)
    basis = torch.zeros(kv_dim, kv_dim, dtype=torch.float64)

    per_head = metrics.matrices[target_layer].to(torch.float64)
    for head in range(n_kv_heads):
        vals, vecs = torch.linalg.eigh(per_head[head])
        lo, hi = head * head_dim, (head + 1) * head_dim
        evals[lo:hi] = vals.clamp_min(0.0)
        basis[lo:hi, lo:hi] = vecs

    return evals, basis


def solve_whitened(
    stats: GramStats,
    layers: tuple[int, ...],
    target_layer: int,
    metrics: HeadMetrics,
    n_kv_heads: int,
    head_dim: int,
    lam: float = 1e-3,
    relative_lambda: bool = True,
    eigenvalue_floor: float = 1e-6,
) -> LinearMap:
    """Fit one target layer's map under an attention-induced metric.

    Args:
        stats: accumulated calibration statistics.
        layers: source layers to use as predictors.
        target_layer: the target layer to predict.
        metrics: per-head error metrics for this target model. Normalize them
            first so that ``lam`` means the same thing across layers.
        n_kv_heads: target KV head count.
        head_dim: target head dimension.
        lam: ridge penalty, relative to the design scale by default.
        relative_lambda: see :func:`kvxfer.solvers.ridge.resolve_lambda`.
        eigenvalue_floor: metric eigenvalues below this fraction of the head's
            mean are clamped. Directions attention genuinely cannot see carry
            near-zero weight, and without a floor their effective penalty
            diverges, which is numerically ugly for no benefit -- the fit in
            those directions is irrelevant either way.

    Returns:
        The fitted :class:`LinearMap`. Its ``r2`` field remains the *isotropic*
        calibration R2, so that it stays directly comparable with the baseline
        rather than being scored under a different objective.
    """
    xtx_c, xty_c, x_mean, y_mean, tss = _centered_moments(stats, layers, target_layer)

    kv_dim = n_kv_heads * head_dim
    if xty_c.shape[1] != kv_dim:
        raise ValueError(
            f"target width {xty_c.shape[1]} does not match {n_kv_heads}x{head_dim}"
        )

    evals, basis = _block_diagonal_basis(metrics, target_layer, n_kv_heads, head_dim)

    # Clamp per head, so one quiet head cannot drag the whole layer's floor.
    per_head = evals.reshape(n_kv_heads, head_dim)
    floor = eigenvalue_floor * per_head.mean(dim=1, keepdim=True)
    evals = torch.maximum(per_head, floor).reshape(-1)

    lam_abs = resolve_lambda(xtx_c, lam, relative_lambda)

    # One eigendecomposition of the design Gram serves every output coordinate.
    key = (id(stats), tuple(layers), stats.kind)
    design_evals, design_basis = _DESIGN_CACHE.get(key, xtx_c)

    # Rotate the cross-moments into the metric's eigenbasis.
    rotated_xty = xty_c @ basis                      # (D, kv_dim)
    projected = design_basis.T @ rotated_xty         # (D, kv_dim)

    # Per-coordinate ridge: penalty lambda / L_j, applied in the design's
    # eigenbasis where the solve is a division.
    penalties = lam_abs / evals                      # (kv_dim,)
    denom = design_evals.unsqueeze(1) + penalties.unsqueeze(0)   # (D, kv_dim)
    weight_rotated = design_basis @ (projected / denom)

    # Back out of the metric basis.
    weight = weight_rotated @ basis.T
    bias = y_mean - x_mean @ weight

    rss = tss - 2.0 * (weight * xty_c).sum(0) + (weight * (xtx_c @ weight)).sum(0)
    r2 = (1.0 - rss / tss.clamp_min(1e-12)).clamp(min=-1.0, max=1.0)

    return LinearMap(
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
        source_layers=tuple(layers),
        target_layer=target_layer,
        r2=float(r2.mean()),
    )


def clear_design_cache() -> None:
    """Drop cached eigendecompositions. Call between pairs to free memory."""
    _DESIGN_CACHE.clear()


def held_out_metric_r2(
    stats: GramStats,
    fit: LinearMap,
    metrics: HeadMetrics,
    n_kv_heads: int,
    head_dim: int,
) -> float:
    """Held-out fit quality measured under the attention metric.

    Plain R² asks how much of the target's variance a map reproduces. This asks
    how much of the variance *attention can see* it reproduces, which is the
    quantity the reference work's diagnostics implicate: they find calibration
    R² anti-correlates with downstream retention, while an attention-sensitive
    measure predicts it.

    Having both makes that testable directly. If retention tracks this figure
    while moving against plain R², the objective mismatch is demonstrated
    rather than argued.

    Computed from moments, like :func:`kvxfer.solvers.ridge.held_out_r2`, but
    it needs the full per-head target second moments rather than just their
    diagonal, since the metric is not diagonal.

    Args:
        stats: statistics from a split not used to fit ``fit``. Must carry
            ``yty_head``.
        fit: the map to score.
        metrics: the metric to score under.
        n_kv_heads: target KV head count.
        head_dim: target head dimension.

    Returns:
        Metric-weighted R², averaged over heads.

    Raises:
        ValueError: if ``stats`` predates per-head moment accumulation.
    """
    if stats.yty_head is None:
        raise ValueError(
            "these statistics carry no per-head target moments; recalibrate to "
            "score under a non-diagonal metric"
        )

    cpu64 = dict(device="cpu", dtype=torch.float64)
    xtx, xty, x_sum = stats.select(tuple(fit.source_layers))
    xtx = xtx.to(**cpu64)
    xty = xty[:, fit.target_layer].to(**cpu64)
    x_sum = x_sum.to(**cpu64)
    y_sum = stats.y_sum[fit.target_layer].to(**cpu64)
    yty = stats.yty_head[fit.target_layer].to(**cpu64)
    n = float(stats.n_tokens)

    w = fit.weight.to(**cpu64)
    b = fit.bias.to(**cpu64)

    # Second moment of the prediction, and its cross moment with the target.
    pred_sq = w.T @ xtx @ w
    cross_x = torch.outer(w.T @ x_sum, b)
    pred_sq = pred_sq + cross_x + cross_x.T + n * torch.outer(b, b)
    pred_target = w.T @ xty + torch.outer(b, y_sum)

    metric = metrics.matrices[fit.target_layer].to(**cpu64)
    mean = y_sum / n

    scores = []
    for head in range(n_kv_heads):
        lo, hi = head * head_dim, (head + 1) * head_dim
        block = yty[head]

        residual = (
            pred_sq[lo:hi, lo:hi]
            - pred_target[lo:hi, lo:hi]
            - pred_target[lo:hi, lo:hi].T
            + block
        )
        centered = block - n * torch.outer(mean[lo:hi], mean[lo:hi])

        rss = torch.einsum("ij,ji->", metric[head], residual)
        tss = torch.einsum("ij,ji->", metric[head], centered)
        scores.append(1.0 - rss / tss.clamp_min(1e-12))

    return float(torch.stack(scores).mean())
