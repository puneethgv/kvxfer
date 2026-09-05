"""Correctness gate 2: RoPE strip/restore must be an exact inverse pair,
and must agree with the reference implementation in transformers."""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3 import modeling_qwen3 as ref

from kvxfer.geometry import load_geometry
from kvxfer.rope import apply_rope, rotate_half, unapply_rope

HEAD_DIM = 128
N_KV_HEADS = 8


def _tables(
    seq: int,
    head_dim: int = HEAD_DIM,
    theta: float = 1e6,
    offset: int = 0,
    dtype: torch.dtype = torch.float32,
):
    """Build (cos, sin) tables the same way a Qwen3 rotary embedding would."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    pos = torch.arange(offset, offset + seq, dtype=dtype)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None], emb.sin()[None]


def test_rotate_half_is_a_quarter_turn():
    """Applying rotate_half four times returns the original vector."""
    x = torch.randn(2, 4, 6, HEAD_DIM)
    assert torch.allclose(rotate_half(rotate_half(rotate_half(rotate_half(x)))), x)
    # Two applications negate.
    assert torch.allclose(rotate_half(rotate_half(x)), -x)


@pytest.mark.parametrize("offset", [0, 1, 512, 8192, 32768])
def test_strip_then_restore_recovers_keys(offset):
    """The gate: unapply(apply(x)) == x at every position offset we will use.

    Position offsets far beyond the calibration window are included because
    content-space mapping is only sound if the round trip holds there too.
    """
    seq = 16
    cos, sin = _tables(seq, offset=offset)
    keys = torch.randn(1, N_KV_HEADS, seq, HEAD_DIM)

    recovered = unapply_rope(apply_rope(keys, cos, sin), cos, sin)

    assert torch.allclose(recovered, keys, atol=1e-5, rtol=1e-4), (
        f"round trip drifted at offset={offset}: "
        f"max abs err {(recovered - keys).abs().max().item():.3e}"
    )


def test_rope_preserves_norm():
    """RoPE is orthogonal, so per-head vector norms must be unchanged."""
    seq = 32
    cos, sin = _tables(seq)
    keys = torch.randn(1, N_KV_HEADS, seq, HEAD_DIM)

    rotated = apply_rope(keys, cos, sin)

    assert torch.allclose(keys.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_apply_rope_matches_transformers_reference():
    """Our apply_rope must be bit-comparable to the shipped Qwen3 implementation.

    If transformers changes its rotary convention, this fails loudly rather
    than silently producing a mapper fitted in the wrong basis.
    """
    seq = 24
    cos, sin = _tables(seq)
    q = torch.randn(1, 16, seq, HEAD_DIM)
    k = torch.randn(1, N_KV_HEADS, seq, HEAD_DIM)

    ref_q, ref_k = ref.apply_rotary_pos_emb(q, k, cos, sin)

    assert torch.allclose(apply_rope(q, cos, sin), ref_q, atol=1e-6)
    assert torch.allclose(apply_rope(k, cos, sin), ref_k, atol=1e-6)


def test_relative_position_structure():
    """Content-space mapping relies on q.k depending only on relative position.

    A query rotated at position m against a key rotated at position n must give
    the same logit as any other pair with the same offset n - m. This is the
    property that lets a mapper fitted at one context length work at another.

    Checked in float64 to test the mathematics rather than the float32 tables;
    the precision cost of float32 is pinned separately below.
    """
    torch.manual_seed(0)
    q_c = torch.randn(1, 1, 1, HEAD_DIM, dtype=torch.float64)
    k_c = torch.randn(1, 1, 1, HEAD_DIM, dtype=torch.float64)

    def logit(m: int, n: int) -> float:
        cos_m, sin_m = _tables(1, offset=m, dtype=torch.float64)
        cos_n, sin_n = _tables(1, offset=n, dtype=torch.float64)
        rq = apply_rope(q_c, cos_m, sin_m)
        rk = apply_rope(k_c, cos_n, sin_n)
        return (rq * rk).sum().item()

    base = logit(0, 8)
    for start in (1, 1000, 20_000, 100_000):
        assert logit(start, start + 8) == pytest.approx(base, abs=1e-9), (
            f"relative-position structure broke at absolute position {start}"
        )


def test_float32_rope_tables_drift_at_long_context():
    """Pin the precision cost of float32 RoPE tables at large absolute positions.

    With theta=1e6 the highest-frequency band reaches ~N radians at position N,
    where float32 resolution is ~1e-3 rad. The strip/restore round trip is
    unaffected (the same table is used both ways, so the error cancels), but
    relative-position geometry degrades. This test documents the size of that
    effect so a regression in table handling is distinguishable from it.
    """
    torch.manual_seed(0)
    q_c = torch.randn(1, 1, 1, HEAD_DIM)
    k_c = torch.randn(1, 1, 1, HEAD_DIM)

    def logit(m: int, n: int) -> float:
        cos_m, sin_m = _tables(1, offset=m)
        cos_n, sin_n = _tables(1, offset=n)
        return (apply_rope(q_c, cos_m, sin_m) * apply_rope(k_c, cos_n, sin_n)).sum().item()

    base = logit(0, 8)
    drift_1k = abs(logit(1_000, 1_008) - base)
    drift_100k = abs(logit(100_000, 100_008) - base)

    assert drift_1k < 1e-3, f"unexpected drift at 1k: {drift_1k:.2e}"
    assert drift_100k < 5e-2, f"unexpected drift at 100k: {drift_100k:.2e}"
    assert drift_100k > drift_1k, "drift should grow with absolute position"


def test_qwen3_family_rope_theta_is_uniform():
    """All Qwen3 dense sizes must share rope_theta for content-space transfer.

    Marked slow only in the sense that it hits the HF config endpoint; it
    downloads no weights.
    """
    pytest.importorskip("huggingface_hub")
    geoms = {}
    for size in ["0.6B", "1.7B", "4B", "8B"]:
        g = load_geometry(f"Qwen/Qwen3-{size}")
        geoms[size] = (g.rope_theta, g.head_dim, g.n_kv_heads)
    assert len(set(geoms.values())) == 1, f"Qwen3 geometry is not uniform: {geoms}"
    assert geoms["0.6B"] == (1e6, 128, 8)
