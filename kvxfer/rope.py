"""Rotary position embedding: stripping it off keys, and putting it back.

The KV cache stores keys *after* RoPE has been applied, so a cached key
entangles content with absolute position. A mapper fitted on post-RoPE keys
would have to learn the rotation as well as the cross-model map, and would not
generalize to positions it never saw during calibration.

So we map in *content space*: strip the rotation from the source keys, fit and
apply the cross-model map there, then re-apply the target model's rotation at
the positions the keys will actually occupy. Values carry no rotation and pass
through untouched.

The rotation is orthogonal, so its inverse is a rotation by the negated angle,
which in the half-split parameterization is just a sign flip on the sine term.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the halves of the last dimension: (x1, x2) -> (-x2, x1)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate content vectors into position-dependent space.

    Args:
        x: ``(batch, n_heads, seq, head_dim)`` content vectors.
        cos: ``(batch, seq, head_dim)`` cosine table.
        sin: ``(batch, seq, head_dim)`` sine table.

    Returns:
        Tensor of the same shape as ``x``, rotated by each position's angle.
    """
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return x * cos + rotate_half(x) * sin


def unapply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Invert :func:`apply_rope`, recovering position-free content vectors.

    The rotation matrix is orthogonal, so the inverse is a rotation by the
    negated angle -- identical to :func:`apply_rope` with ``sin`` negated.
    """
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return x * cos - rotate_half(x) * sin


def rope_tables(
    model: torch.nn.Module,
    position_ids: Tensor,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Get ``(cos, sin)`` from a model's own rotary embedding module.

    Deriving the tables from the model rather than recomputing them keeps us
    exactly consistent with whatever rope_theta, scaling, or partial-rotary
    configuration the model actually uses.

    Args:
        model: a causal LM whose backbone exposes ``rotary_emb``.
        position_ids: ``(batch, seq)`` absolute positions.
        dtype: dtype to compute the tables in. Keep this float32 even for a
            bf16 model -- a bf16 round trip through strip/re-apply loses about
            three decimal digits, which shows up directly in mapper error.

    Returns:
        ``(cos, sin)``, each ``(batch, seq, head_dim)``.
    """
    backbone = getattr(model, "model", model)
    rotary = getattr(backbone, "rotary_emb", None)
    if rotary is None:
        raise AttributeError(
            f"{type(model).__name__} exposes no rotary_emb; cannot derive RoPE tables"
        )

    device = next(model.parameters()).device
    position_ids = position_ids.to(device)
    # rotary_emb only reads dtype/device off its first argument.
    probe = torch.zeros(1, dtype=dtype, device=device)
    cos, sin = rotary(probe, position_ids)
    return cos.to(dtype), sin.to(dtype)


def strip_keys(keys: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Convert cached (post-RoPE) keys to content space, in float32."""
    return unapply_rope(keys.to(torch.float32), cos, sin)


def restore_keys(content: Tensor, cos: Tensor, sin: Tensor, dtype: torch.dtype) -> Tensor:
    """Convert content-space keys back to cache (post-RoPE) form."""
    return apply_rope(content.to(torch.float32), cos, sin).to(dtype)
