"""Collecting content-space query second moments from the target model.

The key metric needs ``E[q_c q_c']`` -- the second moment of queries *before*
RoPE is applied, grouped by the KV head each query head reads from. Qwen-style
attention computes exactly that quantity as ``q_norm(q_proj(x))``, so a forward
hook on ``q_norm`` captures it without recomputing or reimplementing anything.

This is cheap. A covariance over 128 dimensions converges in far fewer tokens
than a regression over thousands, so a few dozen sequences suffice -- the metric
is not where calibration budget needs to go.
"""

from __future__ import annotations

import torch
from torch import Tensor

from kvxfer.data import CalibrationSet
from kvxfer.geometry import KVGeometry


class _QueryMomentHook:
    """Accumulates ``sum q q'`` per (layer, kv head) from q_norm outputs."""

    def __init__(self, geometry: KVGeometry, device: torch.device) -> None:
        self.geometry = geometry
        self.group = geometry.n_q_heads // geometry.n_kv_heads
        self.moments = torch.zeros(
            geometry.n_layers,
            geometry.n_kv_heads,
            geometry.head_dim,
            geometry.head_dim,
            dtype=torch.float32,
            device=device,
        )
        self.n_tokens = 0
        self._stride = 1

    def set_stride(self, stride: int) -> None:
        self._stride = max(1, stride)

    def make(self, layer_idx: int):
        def hook(_module, _inputs, output: Tensor) -> None:
            # (batch, seq, n_q_heads, head_dim), pre-RoPE and post q_norm.
            q = output.detach()
            if q.dim() != 4:
                raise RuntimeError(
                    f"expected (batch, seq, heads, dim) from q_norm, got {tuple(q.shape)}"
                )
            q = q[:, :: self._stride].to(torch.float32)
            batch, seq, n_heads, head_dim = q.shape

            # Query heads are grouped onto KV heads in contiguous blocks.
            grouped = q.reshape(batch, seq, self.geometry.n_kv_heads, self.group, head_dim)
            flat = grouped.permute(2, 0, 1, 3, 4).reshape(
                self.geometry.n_kv_heads, batch * seq * self.group, head_dim
            )
            self.moments[layer_idx] += torch.bmm(flat.transpose(1, 2), flat)
            if layer_idx == 0:
                self.n_tokens += batch * seq * self.group

        return hook


@torch.no_grad()
def collect_query_moments(
    model: torch.nn.Module,
    geometry: KVGeometry,
    calibration: CalibrationSet,
    n_sequences: int = 32,
    token_stride: int = 4,
    batch_size: int = 1,
) -> Tensor:
    """Estimate ``E[q_c q_c']`` per (layer, kv head) for the target model.

    Args:
        model: the *target* model -- its queries are what the mapped keys will
            be read by.
        geometry: that model's geometry.
        calibration: sequences to measure over.
        n_sequences: how many to use. A few dozen is plenty for a 128x128
            covariance.
        token_stride: keep every nth token position.
        batch_size: sequences per forward pass.

    Returns:
        ``(n_layers, n_kv_heads, head_dim, head_dim)`` second moments.
    """
    device = next(model.parameters()).device
    collector = _QueryMomentHook(geometry, device)
    collector.set_stride(token_stride)

    handles = [
        layer.self_attn.q_norm.register_forward_hook(collector.make(idx))
        for idx, layer in enumerate(model.model.layers)
    ]
    try:
        subset = CalibrationSet(
            calibration.input_ids[:n_sequences], calibration.domains[:n_sequences]
        )
        for batch in subset.batches(batch_size):
            model(input_ids=batch.to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if collector.n_tokens == 0:
        raise RuntimeError("no queries captured; is q_norm present on this model?")
    return (collector.moments / collector.n_tokens).cpu()
