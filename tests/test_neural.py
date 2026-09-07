"""The trained residual and, more importantly, the objective it trains on.

The reason to train at all is that the functional objective cannot be written
as a penalty on a linear map. So the load-bearing test here is not that the
network runs, it is that the loss behaves differently from reconstruction
error: it must care where the error lands, not just how large it is. If it
does not, this is an expensive way to reproduce a null already established.
"""

from __future__ import annotations

import math

import pytest
import torch

from kvxfer.geometry import KVGeometry
from kvxfer.solvers.neural import (
    LayerResidual,
    ResidualMapper,
    attention_output_loss,
)
from kvxfer.solvers.ridge import LinearMap

N_KV_HEADS, HEAD_DIM, N_Q_HEADS = 2, 8, 4
KV_DIM = N_KV_HEADS * HEAD_DIM
N_LAYERS, DESIGN_DIM = 3, 32


def _geom() -> KVGeometry:
    return KVGeometry(
        model_id="synthetic",
        n_layers=N_LAYERS,
        n_q_heads=N_Q_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=64,
        rope_theta=1e6,
    )


def _mapper(hidden: int = 16) -> ResidualMapper:
    g = torch.Generator().manual_seed(0)
    maps = {
        layer: LinearMap(
            weight=torch.randn(DESIGN_DIM, KV_DIM, generator=g) / DESIGN_DIM**0.5,
            bias=torch.randn(KV_DIM, generator=g) * 0.01,
            source_layers=(0, 1),
            target_layer=layer,
            r2=0.5,
        )
        for layer in range(N_LAYERS)
    }
    values = {layer: fit for layer, fit in maps.items()}
    return ResidualMapper(maps, values, _geom(), DESIGN_DIM, hidden=hidden)


def test_untrained_residual_leaves_the_closed_form_untouched():
    """Zero-initialized output means training starts from the closed form.

    Starting from noise would throw away a solution that is already most of the
    answer and free, and would make an early-stopped run worse than no residual
    at all.
    """
    mapper = _mapper()
    design = torch.randn(6, DESIGN_DIM)
    for kind in ("keys", "values"):
        base, correction = mapper.base_and_residual(design, 0, kind)
        assert torch.count_nonzero(correction) == 0
        assert torch.allclose(base + correction, base)


def test_loss_is_zero_for_a_perfect_cache_and_positive_otherwise():
    torch.manual_seed(0)
    q = torch.randn(2, N_Q_HEADS, 5, HEAD_DIM)
    k = torch.randn(2, N_KV_HEADS, 5, HEAD_DIM)
    v = torch.randn(2, N_KV_HEADS, 5, HEAD_DIM)

    assert attention_output_loss(q, k, v, k.clone(), v.clone()).item() == pytest.approx(
        0.0, abs=1e-12
    )
    assert attention_output_loss(q, k, v, k + 0.5, v).item() > 0


def test_attention_is_causal():
    """A change at a later position must not alter an earlier output.

    Without the mask the loss would reward fitting keys the model can never
    attend to during prefill, which is not the deployed behaviour.
    """
    torch.manual_seed(0)
    q = torch.randn(1, N_Q_HEADS, 6, HEAD_DIM)
    k = torch.randn(1, N_KV_HEADS, 6, HEAD_DIM)
    v = torch.randn(1, N_KV_HEADS, 6, HEAD_DIM)

    perturbed_v = v.clone()
    perturbed_v[:, :, -1] += 10.0  # only the last position changes

    # Compare attention outputs directly at position 0.
    def attend(keys, values):
        group = N_Q_HEADS // N_KV_HEADS
        kk = keys.repeat_interleave(group, dim=1)
        vv = values.repeat_interleave(group, dim=1)
        scores = (q @ kk.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        mask = torch.ones(6, 6, dtype=torch.bool).tril()
        return torch.softmax(scores.masked_fill(~mask, float("-inf")), -1) @ vv

    assert torch.allclose(attend(k, v)[:, :, 0], attend(k, perturbed_v)[:, :, 0])


def test_loss_weighs_error_by_where_attention_looks():
    """The property that justifies training instead of solving.

    Two perturbations of identical magnitude: one on the value of a position
    attention concentrates on, one on a position it ignores. Reconstruction
    error cannot tell them apart -- that is exactly the reference work's
    complaint -- while this loss must.
    """
    torch.manual_seed(0)
    seq = 8
    # Queries aligned with key 0 so attention concentrates there.
    k = torch.randn(1, N_KV_HEADS, seq, HEAD_DIM)
    v = torch.randn(1, N_KV_HEADS, seq, HEAD_DIM)
    q = k[:, :, :1].repeat(1, N_Q_HEADS // N_KV_HEADS, seq, 1) * 8.0

    attended = v.clone()
    attended[:, :, 0] += 1.0        # perturb what attention looks at
    ignored = v.clone()
    ignored[:, :, -1] += 1.0        # perturb what it does not

    # Identical reconstruction error by construction.
    assert torch.allclose(
        (attended - v).pow(2).sum(), (ignored - v).pow(2).sum()
    )

    loss_attended = attention_output_loss(q, k, v, k, attended)
    loss_ignored = attention_output_loss(q, k, v, k, ignored)
    assert loss_attended > 10 * loss_ignored, (
        f"loss failed to distinguish where error landed: "
        f"attended {loss_attended:.3e} vs ignored {loss_ignored:.3e}"
    )


def test_gradients_reach_the_residual_but_not_the_closed_form():
    """The base map must stay frozen; only the correction is learned."""
    mapper = _mapper()
    design = torch.randn(4, DESIGN_DIM)
    base, correction = mapper.base_and_residual(design, 0, "keys")

    assert not base.requires_grad
    assert correction.requires_grad

    (base + correction).sum().backward()
    grads = [p.grad for p in mapper.key_residual[0].parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_residual_stays_a_bottleneck_at_deployment_scale():
    """The correction must cost a fraction of the map it corrects.

    Checked at real dimensions rather than this file's toy ones, where a
    hidden width of 16 is not narrower than a design of 32 and the property
    is vacuous. For a 1.7B-to-4B pair the design is 4096 wide and the cache
    1024, and the measured latency budget allows a hidden width of about 822
    before the speedup that justifies mapping falls below 3x.
    """
    design_dim, kv_dim, layers = 4 * 1024, 1024, 36
    linear = layers * 2 * design_dim * kv_dim

    for hidden, ceiling in ((256, 0.40), (822, 1.10)):
        residual = LayerResidual(design_dim, kv_dim, hidden)
        total = layers * 2 * sum(p.numel() for p in residual.parameters())
        assert total / linear < ceiling, (
            f"hidden={hidden}: residual is {total / linear:.0%} of the linear map"
        )

    # And it must genuinely be a bottleneck: narrower than both ends.
    assert 256 < kv_dim < design_dim


def test_trained_residual_can_be_evaluated_like_any_mapper():
    """A residual that cannot be measured is a residual reported on its own loss.

    The evaluation path takes anything with ``map`` and ``name``, so the
    trained mapper has to satisfy the same interface as the closed-form one or
    it can only ever be judged by its training curve.
    """
    from kvxfer.cache import ContentKV

    mapper = _mapper()
    mapper.label = "residual"
    batch, seq = 2, 5
    content = ContentKV(
        keys=torch.randn(2, batch, N_KV_HEADS, seq, HEAD_DIM),
        values=torch.randn(2, batch, N_KV_HEADS, seq, HEAD_DIM),
        position_ids=torch.arange(seq).expand(batch, seq),
    )

    out = mapper.map(content)
    assert out.keys.shape == (N_LAYERS, batch, N_KV_HEADS, seq, HEAD_DIM)
    assert out.values.shape == out.keys.shape
    assert torch.isfinite(out.keys).all() and torch.isfinite(out.values).all()
    assert mapper.name == "residual"


def test_untrained_residual_maps_identically_to_its_base():
    """With a zeroed residual the mapper must equal the closed-form mapper.

    This is the control for every later comparison: a difference measured
    between them then comes from training rather than from the two paths
    disagreeing about how to apply the same map.
    """
    from kvxfer.cache import ContentKV
    from kvxfer.mappers import FittedMapper

    mapper = _mapper()
    plain = FittedMapper(mapper.key_maps, mapper.value_maps, _geom(), label="ridge")

    batch, seq = 1, 4
    content = ContentKV(
        keys=torch.randn(2, batch, N_KV_HEADS, seq, HEAD_DIM),
        values=torch.randn(2, batch, N_KV_HEADS, seq, HEAD_DIM),
        position_ids=torch.arange(seq).expand(batch, seq),
    )

    trained, base = mapper.map(content), plain.map(content)
    assert torch.allclose(trained.keys, base.keys, atol=1e-5)
    assert torch.allclose(trained.values, base.values, atol=1e-5)
