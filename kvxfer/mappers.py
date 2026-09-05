"""Mapper interface and the trivial mappers used to validate the harness.

A mapper takes a source model's content-space KV and produces content-space KV
in the target model's layout. Everything downstream -- injection, evaluation,
retention scoring -- is identical regardless of which mapper produced the KV,
which is what makes the correctness gates meaningful: an identity or oracle
mapper exercises the whole pipeline except the regression itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from kvxfer.cache import ContentKV
from kvxfer.geometry import KVGeometry


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
