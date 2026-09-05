"""FittedMapper must gather, apply, and re-layout without scrambling axes.

The mapper moves between a layer-stacked cache view and a token-major design
matrix and back. Those permutes are exactly the kind of code that produces
plausible-looking garbage rather than an error, so it is checked against a case
with a known answer: maps that are literally the identity must leave a cache
bit-for-bit unchanged.
"""

from __future__ import annotations

import pytest
import torch

from kvxfer.cache import ContentKV
from kvxfer.geometry import KVGeometry
from kvxfer.mappers import FittedMapper, IdentityMapper
from kvxfer.solvers.ridge import LinearMap

N_LAYERS = 4
N_KV_HEADS = 3
HEAD_DIM = 8
KV_DIM = N_KV_HEADS * HEAD_DIM


def _geom() -> KVGeometry:
    return KVGeometry(
        model_id="synthetic",
        n_layers=N_LAYERS,
        n_q_heads=N_KV_HEADS * 2,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=32,
        rope_theta=1e6,
    )


def _content(batch: int = 2, seq: int = 5) -> ContentKV:
    torch.manual_seed(0)
    shape = (N_LAYERS, batch, N_KV_HEADS, seq, HEAD_DIM)
    return ContentKV(
        keys=torch.randn(shape),
        values=torch.randn(shape),
        position_ids=torch.arange(seq).unsqueeze(0).expand(batch, -1),
    )


def _identity_maps() -> dict[int, LinearMap]:
    """One map per target layer, copying the same-numbered source layer."""
    return {
        layer: LinearMap(
            weight=torch.eye(KV_DIM),
            bias=torch.zeros(KV_DIM),
            source_layers=(layer,),
            target_layer=layer,
            r2=1.0,
        )
        for layer in range(N_LAYERS)
    }


@pytest.mark.parametrize("batch,seq", [(1, 5), (2, 5), (3, 17)])
def test_identity_maps_reproduce_the_source_cache(batch, seq):
    """Identity weights must be a genuine no-op at any batch and length."""
    source = _content(batch, seq)
    mapper = FittedMapper(_identity_maps(), _identity_maps(), _geom())

    mapped = mapper.map(source)

    assert mapped.keys.shape == source.keys.shape
    assert torch.allclose(mapped.keys, source.keys, atol=1e-5), (
        f"max key error {(mapped.keys - source.keys).abs().max():.3e}"
    )
    assert torch.allclose(mapped.values, source.values, atol=1e-5)


def test_matches_identity_mapper():
    """Composed with the reference IdentityMapper, results must agree."""
    source = _content()
    fitted = FittedMapper(_identity_maps(), _identity_maps(), _geom()).map(source)
    reference = IdentityMapper().map(source)
    assert torch.allclose(fitted.keys, reference.keys, atol=1e-5)


def test_permutation_map_moves_the_right_layer():
    """A map reading a different source layer must actually read that layer.

    Guards against the gather silently ignoring source_layers.
    """
    source = _content()
    shifted = {
        layer: LinearMap(
            weight=torch.eye(KV_DIM),
            bias=torch.zeros(KV_DIM),
            source_layers=((layer + 1) % N_LAYERS,),
            target_layer=layer,
            r2=1.0,
        )
        for layer in range(N_LAYERS)
    }

    mapped = FittedMapper(shifted, shifted, _geom()).map(source)

    for layer in range(N_LAYERS):
        expected = source.keys[(layer + 1) % N_LAYERS]
        assert torch.allclose(mapped.keys[layer], expected, atol=1e-5), (
            f"target layer {layer} did not read source layer {(layer + 1) % N_LAYERS}"
        )


def test_multi_layer_design_concatenates_in_order():
    """With k source layers, the design must concatenate them in the given order.

    Built so that the correct answer is a specific layer: the map averages two
    source layers, and reading them in the wrong order or from the wrong
    offsets would give a different result.
    """
    source = _content()
    picked = (0, 2)
    weight = torch.zeros(len(picked) * KV_DIM, KV_DIM)
    weight[:KV_DIM] = 0.25 * torch.eye(KV_DIM)
    weight[KV_DIM:] = 0.75 * torch.eye(KV_DIM)

    maps = {
        layer: LinearMap(
            weight=weight, bias=torch.zeros(KV_DIM),
            source_layers=picked, target_layer=layer, r2=1.0,
        )
        for layer in range(N_LAYERS)
    }

    mapped = FittedMapper(maps, maps, _geom()).map(source)
    expected = 0.25 * source.keys[0] + 0.75 * source.keys[2]

    assert torch.allclose(mapped.keys[0], expected, atol=1e-5)


def test_missing_target_layer_is_rejected():
    """A mapper that cannot fill every target layer must fail at construction."""
    incomplete = _identity_maps()
    del incomplete[2]
    with pytest.raises(ValueError, match="no key map for target layers"):
        FittedMapper(incomplete, _identity_maps(), _geom())
