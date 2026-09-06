"""Closed-form ridge mapping from source KV to target KV.

This is the baseline method: for each target layer, regress its KV states on
the concatenated KV states of a few source layers, with an isotropic L2
penalty. It reproduces the approach of arXiv:2608.03893 and is the reference
that the attention-aligned solver is measured against.

One structural note. The regression shares a single design matrix across all
output coordinates, and ridge with a shared design decouples across outputs.
So a "per-head" fit is not a separate loop over heads -- factorizing the design
Gram once and solving for all ``kv_dim`` output columns at once *is* the
per-head solution, and is far cheaper than fitting heads independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kvxfer.stats import GramStats


@dataclass
class LinearMap:
    """An affine map from concatenated source layers to one target layer.

    Attributes:
        weight: ``(k * kv_dim, target_kv_dim)``.
        bias: ``(target_kv_dim,)``.
        source_layers: which source layers the design concatenates, in order.
        target_layer: the target layer this map produces.
        r2: mean coefficient of determination on the calibration data. Useful
            for diagnostics, but note that it is a poor predictor of downstream
            retention -- that mismatch is the motivation for the attention
            aligned solver.
        rank: the rank the map was constrained to, or ``None`` if unconstrained.
        factors: ``(left, right)`` with ``weight = left @ right``, when the map
            was fitted under a rank constraint. Kept so that the storage saving
            is a property of the object rather than a claim about it.
    """

    weight: Tensor
    bias: Tensor
    source_layers: tuple[int, ...]
    target_layer: int
    r2: float
    rank: int | None = None
    factors: tuple[Tensor, Tensor] | None = None

    def apply(self, x: Tensor) -> Tensor:
        """Map design rows ``(n_tokens, k * kv_dim)`` to ``(n_tokens, kv_dim)``."""
        return x @ self.weight + self.bias


def _centered_moments(
    stats: GramStats, layers: tuple[int, ...], target_layer: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return mean-centered (xtx, xty, x_mean, y_mean, tss) in float64.

    Centering is done on the accumulated moments rather than the samples, which
    is the whole point of keeping sufficient statistics: an intercept costs a
    rank-one correction, not another pass over the corpus.
    """
    xtx, xty, x_sum = stats.select(layers)
    n = float(stats.n_tokens)

    # Solve on CPU in float64. The Gram is accumulated on an accelerator for
    # throughput, but the solve is a one-off whose conditioning matters far
    # more than its speed -- and MPS has no float64 at all.
    cpu64 = dict(device="cpu", dtype=torch.float64)
    xtx = xtx.to(**cpu64)
    xty = xty[:, target_layer].to(**cpu64)
    x_mean = (x_sum.to(**cpu64) / n).unsqueeze(1)
    y_mean = (stats.y_sum[target_layer].to(**cpu64) / n).unsqueeze(0)

    xtx_c = xtx - n * (x_mean @ x_mean.T)
    xty_c = xty - n * (x_mean @ y_mean)
    tss = stats.yty_diag[target_layer].to(**cpu64) - n * y_mean.squeeze(0) ** 2

    return xtx_c, xty_c, x_mean.squeeze(1), y_mean.squeeze(0), tss


def resolve_lambda(xtx_c: Tensor, lam: float, relative: bool = True) -> float:
    """Turn a penalty setting into an absolute ridge value.

    Args:
        xtx_c: the centered design Gram.
        lam: penalty. Interpreted relative to the mean diagonal of ``xtx_c``
            when ``relative`` is set, which keeps a single setting meaningful
            across models, layer counts and calibration sizes. An absolute
            value is only comparable within one exact configuration.
        relative: whether ``lam`` is a fraction of the mean eigenvalue scale.
    """
    if not relative:
        return lam
    scale = (torch.diagonal(xtx_c).mean()).item()
    return lam * max(scale, 1e-12)


def solve_ridge(
    stats: GramStats,
    layers: tuple[int, ...],
    target_layer: int,
    lam: float = 1e-3,
    relative_lambda: bool = True,
) -> LinearMap:
    """Fit one target layer's map in closed form.

    Args:
        stats: accumulated calibration statistics.
        layers: source layers to use as predictors.
        target_layer: the target layer to predict.
        lam: ridge penalty, relative to the design scale by default.
        relative_lambda: see :func:`resolve_lambda`.

    Returns:
        The fitted :class:`LinearMap`, with calibration R² attached.
    """
    xtx_c, xty_c, x_mean, y_mean, tss = _centered_moments(stats, layers, target_layer)

    lam_abs = resolve_lambda(xtx_c, lam, relative_lambda)
    dim = xtx_c.shape[0]
    regularized = xtx_c + lam_abs * torch.eye(dim, dtype=torch.float64)

    # Cholesky is valid here: the centered Gram is PSD and the ridge term makes
    # it strictly PD. Fall back to a general solve if conditioning defeats it.
    try:
        factor = torch.linalg.cholesky(regularized)
        weight = torch.cholesky_solve(xty_c, factor)
    except RuntimeError:
        weight = torch.linalg.solve(regularized, xty_c)

    bias = y_mean - x_mean @ weight

    # RSS at the ridge optimum, from moments alone.
    rss = tss - 2.0 * (weight * xty_c).sum(0) + (weight * (xtx_c @ weight)).sum(0)
    r2 = (1.0 - rss / tss.clamp_min(1e-12)).clamp(min=-1.0, max=1.0)

    return LinearMap(
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
        source_layers=tuple(layers),
        target_layer=target_layer,
        r2=float(r2.mean()),
    )


def rank_source_layers(
    stats: GramStats, target_layer: int, lam: float = 1e-3
) -> list[tuple[int, float]]:
    """Rank single source layers by how well each alone predicts a target layer.

    This is the selection signal for top-k: fit each candidate on its own and
    order by R². Cheap, because every fit is a submatrix of the cached Gram.

    Returns:
        ``(source_layer, r2)`` pairs, best first.
    """
    scored = [
        (layer, solve_ridge(stats, (layer,), target_layer, lam).r2)
        for layer in stats.source_layers
    ]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def held_out_r2(stats: GramStats, fit: LinearMap) -> Tensor:
    """Per-coordinate R² of a fitted map on *independent* statistics.

    Selecting source layers on in-sample R² is not merely optimistic, it is
    uninformative: with ``k * kv_dim`` free parameters against a comparable
    number of calibration tokens the design interpolates, and every candidate
    scores 1.0. Selection has to be made against held-out data.

    Crucially this needs no held-out *samples*, only held-out *moments*. The
    residual sum of squares expands entirely into second moments::

        RSS_j = Y'Y_j - 2 w_j'(X'Y)_j - 2 b_j (1'Y)_j
                + w_j'(X'X) w_j + 2 b_j (1'X) w_j + n b_j^2

    so a validation split costs one more accumulator pass and nothing else.

    Args:
        stats: statistics from a split *not* used to fit ``fit``.
        fit: the map to score.

    Returns:
        ``(target_kv_dim,)`` R² per output coordinate.
    """
    if fit.source_layers != tuple(fit.source_layers):
        raise ValueError("fit.source_layers must be an ordered tuple")

    xtx, xty, x_sum = stats.select(tuple(fit.source_layers))
    cpu64 = dict(device="cpu", dtype=torch.float64)
    xtx = xtx.to(**cpu64)
    xty = xty[:, fit.target_layer].to(**cpu64)
    x_sum = x_sum.to(**cpu64)
    y_sum = stats.y_sum[fit.target_layer].to(**cpu64)
    yty = stats.yty_diag[fit.target_layer].to(**cpu64)
    n = float(stats.n_tokens)

    w = fit.weight.to(**cpu64)
    b = fit.bias.to(**cpu64)

    rss = (
        yty
        - 2.0 * (w * xty).sum(0)
        - 2.0 * b * y_sum
        + (w * (xtx @ w)).sum(0)
        + 2.0 * b * (x_sum @ w)
        + n * b**2
    )
    tss = yty - n * (y_sum / n) ** 2
    return 1.0 - rss / tss.clamp_min(1e-12)


def select_source_layers(
    fit_stats: GramStats,
    val_stats: GramStats,
    target_layer: int,
    k: int,
    lam: float = 1e-3,
    n_candidates: int | None = None,
) -> tuple[int, ...]:
    """Choose k source layers for one target layer, scored out of sample.

    Greedy forward selection: repeatedly add whichever remaining candidate most
    improves held-out R². Greedy rather than exhaustive because the number of
    subsets is prohibitive, and forward rather than backward because the useful
    values of k are small.

    Args:
        fit_stats: statistics to fit candidate maps on.
        val_stats: independent statistics to score them on.
        target_layer: the target layer being predicted.
        k: how many source layers to select.
        lam: ridge penalty used during selection.
        n_candidates: restrict the search to the this many best single layers,
            which cuts selection cost with little effect on the result.

    Returns:
        The selected source layers, in the order they were chosen.
    """
    pool = list(fit_stats.source_layers)
    if n_candidates is not None and n_candidates < len(pool):
        singles = [
            (l, held_out_r2(val_stats, solve_ridge(fit_stats, (l,), target_layer, lam)).mean().item())
            for l in pool
        ]
        singles.sort(key=lambda pair: pair[1], reverse=True)
        pool = [l for l, _ in singles[:n_candidates]]

    chosen: list[int] = []
    for _ in range(min(k, len(pool))):
        scored = []
        for layer in pool:
            if layer in chosen:
                continue
            trial = tuple(sorted(chosen + [layer]))
            fit = solve_ridge(fit_stats, trial, target_layer, lam)
            scored.append((held_out_r2(val_stats, fit).mean().item(), layer))
        if not scored:
            break
        best_score, best_layer = max(scored)
        chosen.append(best_layer)

    return tuple(sorted(chosen))


def select_top_k(
    fit_stats: GramStats,
    val_stats: GramStats,
    target_layer: int,
    k: int,
    lam: float = 1e-3,
) -> tuple[int, ...]:
    """Pick the k best single source layers, ranked out of sample.

    This is the reference work's selection rule -- rank each candidate layer by
    how well it alone predicts the target, then concatenate the top k -- with
    the one change that ranking happens on held-out rather than in-sample
    statistics. It is much cheaper than forward selection (linear rather than
    quadratic in k) and, because layers within a family are highly correlated
    with their neighbours, usually lands on the same set.

    Args:
        fit_stats: statistics to fit candidates on.
        val_stats: independent statistics to rank them on.
        target_layer: the target layer being predicted.
        k: how many source layers to keep.
        lam: ridge penalty used while ranking.

    Returns:
        The selected source layers, in ascending order.
    """
    scored = [
        (
            float(held_out_r2(val_stats, solve_ridge(fit_stats, (layer,), target_layer, lam)).mean()),
            layer,
        )
        for layer in fit_stats.source_layers
    ]
    scored.sort(reverse=True)
    return tuple(sorted(layer for _, layer in scored[:k]))
