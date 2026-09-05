"""Moving between a live KV cache and the content-space tensors mappers work on.

A ``DynamicCache`` holds keys already rotated by RoPE and values untouched.
Mappers operate on *content space*: keys with the rotation stripped. These
helpers are the only place that conversion happens, so the rest of the codebase
never has to reason about which space a tensor is in.

Layout convention throughout: content tensors are stacked over layers as
``(n_layers, batch, n_kv_heads, seq, head_dim)``, in float32.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import DynamicCache

from kvxfer.rope import rope_tables, strip_keys, restore_keys


@dataclass
class ContentKV:
    """A model's KV cache with RoPE stripped from the keys.

    Attributes:
        keys: ``(n_layers, batch, n_kv_heads, seq, head_dim)`` content-space keys.
        values: ``(n_layers, batch, n_kv_heads, seq, head_dim)`` values, as cached.
        position_ids: ``(batch, seq)`` absolute positions the keys came from.
    """

    keys: Tensor
    values: Tensor
    position_ids: Tensor

    @property
    def n_layers(self) -> int:
        return self.keys.shape[0]

    @property
    def seq_len(self) -> int:
        return self.keys.shape[3]

    def token_matrix(self, which: str = "keys") -> Tensor:
        """Flatten to ``(batch * seq, n_layers * n_kv_heads * head_dim)``.

        This is the design matrix layout the regression solvers consume: one
        row per token, one column per (layer, head, dim) coordinate.
        """
        t = self.keys if which == "keys" else self.values
        n_layers, batch, n_kv, seq, head_dim = t.shape
        # -> (batch, seq, n_layers, n_kv, head_dim) -> (batch*seq, -1)
        return t.permute(1, 3, 0, 2, 4).reshape(batch * seq, n_layers * n_kv * head_dim)

    def to(self, device: torch.device | str) -> "ContentKV":
        return ContentKV(
            self.keys.to(device), self.values.to(device), self.position_ids.to(device)
        )


@torch.no_grad()
def prefill(model: torch.nn.Module, input_ids: Tensor) -> DynamicCache:
    """Run a forward pass and return the resulting KV cache."""
    device = next(model.parameters()).device
    out = model(input_ids=input_ids.to(device), use_cache=True)
    return out.past_key_values


def stack_cache(cache: DynamicCache) -> tuple[Tensor, Tensor]:
    """Stack a cache's per-layer tensors into ``(n_layers, batch, n_kv, seq, d)``."""
    keys = torch.stack([layer.keys for layer in cache.layers], dim=0)
    values = torch.stack([layer.values for layer in cache.layers], dim=0)
    return keys, values


def cache_to_content(
    cache: DynamicCache, model: torch.nn.Module, position_ids: Tensor | None = None
) -> ContentKV:
    """Strip RoPE from a cache's keys, yielding mapper-ready content tensors."""
    keys, values = stack_cache(cache)
    _, batch, _, seq, _ = keys.shape

    if position_ids is None:
        position_ids = torch.arange(seq, device=keys.device).unsqueeze(0).expand(batch, -1)

    cos, sin = rope_tables(model, position_ids)
    content = strip_keys(keys, cos.unsqueeze(0), sin.unsqueeze(0))
    return ContentKV(
        keys=content, values=values.to(torch.float32), position_ids=position_ids
    )


def content_to_cache(
    content: ContentKV,
    model: torch.nn.Module,
    dtype: torch.dtype,
    position_ids: Tensor | None = None,
) -> DynamicCache:
    """Re-apply the target model's RoPE and build a cache ready for decoding.

    Args:
        content: content-space keys and values to install.
        model: the model that will consume the cache; supplies the RoPE tables.
        dtype: cache dtype, normally the model's own compute dtype.
        position_ids: positions to rotate to. Defaults to the positions the
            content was captured at, which is what a straight prefill reuse
            wants; pass explicitly to relocate a cache in the sequence.
    """
    if position_ids is None:
        position_ids = content.position_ids

    cos, sin = rope_tables(model, position_ids)
    keys = restore_keys(content.keys, cos.unsqueeze(0), sin.unsqueeze(0), dtype)
    values = content.values.to(dtype)

    device = next(model.parameters()).device
    return DynamicCache(
        ddp_cache_data=[
            (keys[i].to(device), values[i].to(device)) for i in range(keys.shape[0])
        ]
    )
