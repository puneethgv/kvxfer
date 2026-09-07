"""Mapper interface and the trivial mappers used to validate the harness.

A mapper takes a source model's content-space KV and produces content-space KV
in the target model's layout. Everything downstream -- injection, evaluation,
retention scoring -- is identical regardless of which mapper produced the KV,
which is what makes the correctness gates meaningful: an identity or oracle
mapper exercises the whole pipeline except the regression itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dataclasses import replace

import torch
from torch import Tensor

from kvxfer.cache import ContentKV
from kvxfer.geometry import KVGeometry
from kvxfer.solvers.ridge import LinearMap


class Mapper(ABC):
    """Maps source-model content KV onto target-model content KV."""

    @abstractmethod
    def map(self, source: ContentKV) -> ContentKV:
        """Produce target-layout content KV from source content KV."""

    @property
    def name(self) -> str:
        return type(self).__name__


class IdentityMapper(Mapper):
    """Passes KV through unchanged.

    Only meaningful when source and target are the same model. Used by the
    identity-injection gate: if evaluating through this mapper does not exactly
    reproduce the model's standalone scores, the bug is in the harness -- cache
    construction, position handling, or scoring -- and no transfer result can
    be trusted until it is fixed.
    """

    def map(self, source: ContentKV) -> ContentKV:
        return source


class OracleMapper(Mapper):
    """Returns pre-computed ground-truth target KV, ignoring its input.

    This is the ceiling: it isolates harness error from mapper error. Retention
    measured through an oracle mapper must be 100%. Anything less means the
    evaluation path is lossy and every mapper number is depressed by that same
    amount.
    """

    def __init__(self, target_kv: ContentKV) -> None:
        self._target = target_kv

    def map(self, source: ContentKV) -> ContentKV:
        if source.seq_len != self._target.seq_len:
            raise ValueError(
                f"oracle holds {self._target.seq_len} tokens but was given "
                f"{source.seq_len}; the oracle must be built for this exact input"
            )
        return self._target


class ZeroMapper(Mapper):
    """Emits an all-zero cache of the target's shape.

    A floor, not a method. Retention through this mapper shows what the target
    model scores when its context is destroyed but the prompt length is
    preserved, which is the right baseline for judging whether a weak mapper is
    transferring anything at all.
    """

    def __init__(self, target_geometry: KVGeometry) -> None:
        self._geom = target_geometry

    def map(self, source: ContentKV) -> ContentKV:
        _, batch, _, seq, _ = source.keys.shape
        shape = (self._geom.n_layers, batch, self._geom.n_kv_heads, seq, self._geom.head_dim)
        zeros = torch.zeros(shape, dtype=torch.float32, device=source.keys.device)
        return ContentKV(
            keys=zeros, values=zeros.clone(), position_ids=source.position_ids
        )


def design_rows(tensor: Tensor, layers: tuple[int, ...]) -> Tensor:
    """Gather layers into ``(n_tokens, len(layers) * kv_dim)`` design rows.

    Shared with the trained residual rather than duplicated. The feature order
    here has to match the order the maps were fitted against, and a divergence
    would not raise -- it would apply the map to correctly shaped, wrongly
    ordered inputs and quietly produce a worse mapper.
    """
    picked = tensor[list(layers)]
    n_layers, batch, n_kv, seq, head_dim = picked.shape
    return picked.permute(1, 3, 0, 2, 4).reshape(
        batch * seq, n_layers * n_kv * head_dim
    )


def _apply_on(fit: LinearMap, design: Tensor) -> Tensor:
    """Apply a fitted map to a design block, on the design's device and dtype.

    Routes through the factors when the map has them, so a rank-constrained
    map never reconstitutes the dense matrix it exists to avoid.

    The ``.to`` calls are no-ops once :meth:`FittedMapper.to` has placed the
    maps on the compute device. Left on the host they are not: the maps are
    302M parameters for a 1.7B-to-4B pair, so moving them per layer per call
    pushes about 1.2 GB across PCIe every time a cache is mapped, which
    measured as 195 ms to map a 512-token cache that the target model prefills
    in 99 ms.
    """
    parts = fit.factors if fit.factors is not None else (fit.weight,)
    if parts[0].device != design.device:
        parts = tuple(t.to(design.device) for t in parts)

    # The design follows the map's dtype rather than the reverse. Casting the
    # map up to float32 would keep the multiply off the accelerator's tensor
    # cores, and the cache it feeds is bfloat16 anyway, so the precision was
    # never going to survive.
    x = design.to(parts[0].dtype)
    bias = fit.bias.to(x.device, x.dtype)
    if len(parts) == 2:
        return (x @ parts[0]) @ parts[1] + bias
    return x @ parts[0] + bias


class FittedMapper(Mapper):
    """Applies fitted per-target-layer linear maps to a source cache.

    Each target layer draws on its own selected source layers, so mapping is a
    gather followed by one matrix multiply per layer. Keys and values are
    mapped independently, by maps fitted on their own statistics.
    """

    def __init__(
        self,
        key_maps: dict[int, LinearMap],
        value_maps: dict[int, LinearMap],
        target_geometry: KVGeometry,
        label: str = "fitted",
    ) -> None:
        """
        Args:
            key_maps: target layer index -> fitted key map.
            value_maps: target layer index -> fitted value map.
            target_geometry: geometry of the model the cache is destined for.
            label: name used in results tables.
        """
        missing = set(range(target_geometry.n_layers)) - set(key_maps)
        if missing:
            raise ValueError(f"no key map for target layers {sorted(missing)}")
        missing = set(range(target_geometry.n_layers)) - set(value_maps)
        if missing:
            raise ValueError(f"no value map for target layers {sorted(missing)}")

        self.key_maps = key_maps
        self.value_maps = value_maps
        self.geometry = target_geometry
        self.label = label

    @property
    def name(self) -> str:
        return self.label

    _design = staticmethod(design_rows)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "FittedMapper":
        """Place every stored map on ``device``, once, in place.

        Mapping is a per-layer matrix multiply, so leaving the maps on the host
        makes every call pay a host-to-device copy of the whole map. Doing it
        once here is the difference between the map costing more than the
        prefill it replaces and costing a fraction of it.

        ``dtype`` should be the compute dtype the models run in. The maps are
        fitted and stored in float32 for conditioning, but applying them in
        float32 gives up the accelerator's tensor cores for no accuracy that
        survives a bfloat16 cache.

        Returns:
            ``self``, so this can be chained onto construction.
        """
        for maps in (self.key_maps, self.value_maps):
            for layer, fit in maps.items():
                moved = {"bias": fit.bias.to(device=device, dtype=dtype)}
                if fit.weight is not None:
                    moved["weight"] = fit.weight.to(device=device, dtype=dtype)
                if fit.factors is not None:
                    moved["factors"] = tuple(
                        f.to(device=device, dtype=dtype) for f in fit.factors
                    )
                maps[layer] = replace(fit, **moved)
        return self

    def _map_one(
        self, source: Tensor, maps: dict[int, LinearMap], batch: int, seq: int
    ) -> Tensor:
        out = torch.empty(
            self.geometry.n_layers,
            batch,
            self.geometry.n_kv_heads,
            seq,
            self.geometry.head_dim,
            dtype=torch.float32,
            device=source.device,
        )
        for layer in range(self.geometry.n_layers):
            fit = maps[layer]
            design = self._design(source, fit.source_layers)
            predicted = _apply_on(fit, design).to(out.dtype)
            out[layer] = predicted.reshape(
                batch, seq, self.geometry.n_kv_heads, self.geometry.head_dim
            ).permute(0, 2, 1, 3)
        return out

    def map(self, source: ContentKV) -> ContentKV:
        _, batch, _, seq, _ = source.keys.shape
        return ContentKV(
            keys=self._map_one(source.keys, self.key_maps, batch, seq),
            values=self._map_one(source.values, self.value_maps, batch, seq),
            position_ids=source.position_ids,
        )
