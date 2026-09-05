"""Attention-induced metrics on KV error.

A mapper fitted by plain least squares treats every coordinate of a head as
equally important. The model does not. This module derives the metric under
which KV error actually costs the model something, which the attention-aligned
solver then minimizes against.

Keys
----
A content-space key error ``eps`` at position n reaches the model only through
attention logits against queries at positions m::

    logit = q_c' R_{n-m} k_c        so       logit error = (R_{n-m}' q_c)' eps

The cost of ``eps`` is therefore ``eps' M_K eps`` with::

    M_K = E_delta[ R_-delta C R_-delta' ],      C = E[q_c q_c']

RoPE rotates each coordinate pair ``(i, i + head_dim/2)`` at its own rate, and
the conjugation acts independently within pairs, so ``M_K`` is block diagonal
with 2x2 blocks. The two regimes matter and this module handles both rather
than assuming either:

* High-frequency bands rotate through many turns across the offsets present in
  real contexts, so the average washes out to an isotropic block -- the metric
  becomes a per-band energy weight.
* Low-frequency bands barely rotate at all over a few thousand tokens, so
  ``R_delta`` is nearly the identity and the block keeps the full anisotropic
  structure of ``C``.

Averaging over the actual offset distribution interpolates between them without
having to choose.

Values
------
A value error reaches the residual stream only through the output projection,
scaled by the attention mass the token receives. With no positional
dependence the metric is just ``W_O' W_O`` for the query heads sharing that KV
head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kvxfer.geometry import KVGeometry


@dataclass
class HeadMetrics:
    """Per-(layer, kv head) metrics on content-space KV error.

    Attributes:
        matrices: ``(n_layers, n_kv_heads, head_dim, head_dim)``, symmetric PSD.
        kind: ``"keys"`` or ``"values"``.
    """

    matrices: Tensor
    kind: str

    def normalized(self) -> "HeadMetrics":
        """Rescale each head's metric to unit mean eigenvalue.

        The solver's penalty is expressed relative to the metric, so putting
        every head on a common scale keeps one lambda meaningful across layers
        and heads instead of silently varying with activation magnitude.
        """
        trace = self.matrices.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True)
        scale = trace.unsqueeze(-1).clamp_min(1e-12)
        return HeadMetrics(self.matrices / scale, self.kind)

    def eigendecompose(self) -> tuple[Tensor, Tensor]:
        """Return ``(eigenvalues, eigenvectors)`` per head.

        The solver works in this basis, where the metric is diagonal and every
        output coordinate becomes an independent ridge with its own penalty.
        """
        evals, evecs = torch.linalg.eigh(self.matrices.to(torch.float64))
        return evals.clamp_min(0.0), evecs


def rope_frequencies(head_dim: int, rope_theta: float) -> Tensor:
    """Angular rate of each rotary coordinate pair."""
    return 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    )


def offset_distribution(max_offset: int, n_samples: int = 512) -> Tensor:
    """Relative query-key offsets to average the key metric over.

    Under causal attention a key at position n is read by queries at every
    later position, so offsets run from 1 to the remaining context. Sampling
    them log-uniformly rather than uniformly reflects that attention mass
    concentrates at short range while still covering long-range reads, and it
    resolves the low-frequency bands -- where the metric is most anisotropic --
    far better than linear spacing at the same sample count.
    """
    if max_offset < 1:
        raise ValueError(f"max_offset must be >= 1, got {max_offset}")
    return torch.logspace(0, torch.log10(torch.tensor(float(max_offset))), n_samples, dtype=torch.float64)


def key_metric(
    query_second_moment: Tensor,
    head_dim: int,
    rope_theta: float,
    max_offset: int = 4096,
    n_offsets: int = 512,
) -> HeadMetrics:
    """Build the key-error metric from content-space query second moments.

    Args:
        query_second_moment: ``(n_layers, n_kv_heads, head_dim, head_dim)``,
            ``E[q_c q_c']`` for the query heads sharing each KV head.
        head_dim: head dimension.
        rope_theta: the model's rotary base.
        max_offset: largest relative offset to average over. Should reflect the
            context lengths the mapper will serve.
        n_offsets: how many offsets to average.

    Returns:
        Block-diagonal metrics, one per (layer, kv head).
    """
    device = query_second_moment.device
    c = query_second_moment.to(torch.float64)
    n_layers, n_kv, dim, _ = c.shape
    if dim != head_dim:
        raise ValueError(f"expected head_dim {head_dim}, got {dim}")

    half = head_dim // 2
    freqs = rope_frequencies(head_dim, rope_theta).to(device)
    offsets = offset_distribution(max_offset, n_offsets).to(device)

    # angles[b, s] = rotation of band b at offset s
    angles = offsets[None, :] * freqs[:, None]
    cos, sin = angles.cos(), angles.sin()

    metric = torch.zeros_like(c)
    idx_lo = torch.arange(half, device=device)
    idx_hi = idx_lo + half

    # Each band is the 2x2 sub-block on coordinates (i, i + half).
    a = c[..., idx_lo, idx_lo]
    d = c[..., idx_hi, idx_hi]
    b = c[..., idx_lo, idx_hi]

    # Average R_theta [[a, b], [b, d]] R_theta' over the offset distribution.
    cc = (cos**2).mean(-1)
    ss = (sin**2).mean(-1)
    cs = (cos * sin).mean(-1)

    top = a * cc + d * ss - 2.0 * b * cs
    bot = a * ss + d * cc + 2.0 * b * cs
    off = (a - d) * cs + b * (cc - ss)

    metric[..., idx_lo, idx_lo] = top
    metric[..., idx_hi, idx_hi] = bot
    metric[..., idx_lo, idx_hi] = off
    metric[..., idx_hi, idx_lo] = off

    return HeadMetrics(metric, kind="keys")


def value_metric(model: torch.nn.Module, geometry: KVGeometry) -> HeadMetrics:
    """Build the value-error metric from the output projections.

    A value error on KV head h is read by every query head in its group and
    reaches the residual stream through that group's slice of ``W_O``, so the
    metric is the summed Gram of those slices.
    """
    group = geometry.n_q_heads // geometry.n_kv_heads
    head_dim = geometry.head_dim

    layers = model.model.layers
    metrics = torch.zeros(
        geometry.n_layers, geometry.n_kv_heads, head_dim, head_dim, dtype=torch.float64
    )

    for layer_idx, layer in enumerate(layers):
        # .cpu() first: MPS has no float64, and this must be exact.
        w_o = layer.self_attn.o_proj.weight.detach().cpu().to(torch.float64)
        # o_proj maps (n_q_heads * head_dim) -> hidden, so columns are per head.
        for kv_head in range(geometry.n_kv_heads):
            acc = torch.zeros(head_dim, head_dim, dtype=torch.float64)
            for g in range(group):
                q_head = kv_head * group + g
                block = w_o[:, q_head * head_dim : (q_head + 1) * head_dim]
                acc += block.T @ block
            metrics[layer_idx, kv_head] = acc

    return HeadMetrics(metrics, kind="values")


def identity_metrics(geometry: KVGeometry, kind: str) -> HeadMetrics:
    """Isotropic metrics, reproducing plain least squares.

    The baseline the attention-aligned solver is compared against.
    """
    eye = torch.eye(geometry.head_dim, dtype=torch.float64)
    matrices = eye.expand(
        geometry.n_layers, geometry.n_kv_heads, geometry.head_dim, geometry.head_dim
    ).clone()
    return HeadMetrics(matrices, kind=kind)
