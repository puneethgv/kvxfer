"""Running the calibration pass that produces sufficient statistics.

Source and target are prefilled on the same token sequences, converted to
content space, and folded into a :class:`GramStats`. Nothing is written to disk
except the finished statistics -- activations for a calibration set of this size
would run to tens of gigabytes, and the Gram is all the solver ever needs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from kvxfer.cache import cache_to_content, prefill
from kvxfer.data import CalibrationSet
from kvxfer.geometry import KVGeometry, check_pair
from kvxfer.stats import GramAccumulator, GramStats


@dataclass
class HarvestReport:
    """What a calibration pass cost and covered."""

    n_tokens: int
    n_sequences: int
    seconds: float
    accumulator_bytes: int

    def __str__(self) -> str:
        gb = self.accumulator_bytes / 1024**3
        rate = self.n_tokens / max(self.seconds, 1e-9)
        return (
            f"{self.n_tokens:,} tokens from {self.n_sequences} sequences in "
            f"{self.seconds:.1f}s ({rate:,.0f} tok/s), accumulator {gb:.2f} GB"
        )


def _design_rows(content, layers: tuple[int, ...], which: str) -> Tensor:
    """Flatten selected layers of a ContentKV into ``(n_tokens, k * kv_dim)``."""
    tensor = content.keys if which == "keys" else content.values
    picked = tensor[list(layers)]
    n_layers, batch, n_kv, seq, head_dim = picked.shape
    return picked.permute(1, 3, 0, 2, 4).reshape(batch * seq, n_layers * n_kv * head_dim)


def _target_rows(content, which: str) -> Tensor:
    """Reshape a ContentKV into ``(n_tokens, n_layers, kv_dim)``."""
    tensor = content.keys if which == "keys" else content.values
    n_layers, batch, n_kv, seq, head_dim = tensor.shape
    return tensor.permute(1, 3, 0, 2, 4).reshape(batch * seq, n_layers, n_kv * head_dim)


@torch.no_grad()
def harvest(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_geom: KVGeometry,
    target_geom: KVGeometry,
    calibration: CalibrationSet,
    kind: str = "keys",
    source_layers: tuple[int, ...] | None = None,
    token_stride: int = 2,
    batch_size: int = 1,
    accumulate_on: str | torch.device | None = None,
    progress: bool = True,
) -> tuple[GramStats, HarvestReport]:
    """Accumulate calibration statistics for one KV kind.

    Args:
        source_model: model whose cache will be mapped from.
        target_model: model whose cache will be mapped to.
        source_geom: source geometry.
        target_geom: target geometry.
        calibration: tokenized sequences to calibrate on.
        kind: ``"keys"`` or ``"values"``.
        source_layers: candidate source layers. Memory grows with the square of
            ``len(source_layers) * kv_dim``, so this is the dial to turn when
            the accumulator does not fit.
        token_stride: keep every nth token position. Adjacent tokens carry
            highly correlated KV states, so subsampling costs little
            information and buys proportional time and conditioning headroom.
        batch_size: sequences per forward pass.
        accumulate_on: device for the Gram update. Defaults to the models'
            device; the update is a large matmul and is far faster on an
            accelerator than on CPU.
        progress: print per-batch progress.

    Returns:
        The finished statistics and a report on the pass.
    """
    check_pair(source_geom, target_geom)
    layers = source_layers or tuple(range(source_geom.n_layers))
    device = accumulate_on or next(target_model.parameters()).device

    accumulator = GramAccumulator(
        source_geom, target_geom, kind=kind, source_layers=layers, device=device
    )
    if progress:
        print(f"  accumulator: {accumulator.nbytes / 1024**3:.2f} GB on {device}", flush=True)

    started = time.time()
    n_sequences = 0
    for batch in calibration.batches(batch_size):
        source_content = cache_to_content(prefill(source_model, batch), source_model)
        target_content = cache_to_content(prefill(target_model, batch), target_model)

        x = _design_rows(source_content, layers, kind)
        y = _target_rows(target_content, kind)

        if token_stride > 1:
            x = x[::token_stride]
            y = y[::token_stride]

        accumulator.update(x, y)
        n_sequences += batch.shape[0]

        del source_content, target_content, x, y
        if progress and n_sequences % (batch_size * 16) == 0:
            print(
                f"  {n_sequences}/{len(calibration)} sequences, "
                f"{accumulator.n_tokens:,} tokens",
                flush=True,
            )

    report = HarvestReport(
        n_tokens=accumulator.n_tokens,
        n_sequences=n_sequences,
        seconds=time.time() - started,
        accumulator_bytes=accumulator.nbytes,
    )
    return accumulator.finalize(), report


@torch.no_grad()
def harvest_both(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_geom: KVGeometry,
    target_geom: KVGeometry,
    calibration: CalibrationSet,
    source_layers: tuple[int, ...] | None = None,
    token_stride: int = 2,
    batch_size: int = 1,
    accumulate_on: str | torch.device | None = None,
    progress: bool = True,
) -> tuple[dict[str, GramStats], HarvestReport]:
    """Accumulate key and value statistics in a single pass.

    Keys and values need separate statistics but identical forward passes, so
    running them as two passes doubles the dominant cost for nothing. This
    halves calibration time at the price of holding two accumulators at once --
    the right trade whenever they both fit, and the difference between one and
    two GPU-hours on rented hardware.

    Args:
        See :func:`harvest`. ``source_layers`` remains the memory dial, and now
        governs two accumulators rather than one.

    Returns:
        ``({"keys": ..., "values": ...}, report)``.
    """
    check_pair(source_geom, target_geom)
    layers = source_layers or tuple(range(source_geom.n_layers))
    device = accumulate_on or next(target_model.parameters()).device

    accumulators = {
        kind: GramAccumulator(
            source_geom, target_geom, kind=kind, source_layers=layers, device=device
        )
        for kind in ("keys", "values")
    }
    if progress:
        total = sum(a.nbytes for a in accumulators.values())
        print(f"  accumulators: {total / 1024**3:.2f} GB on {device}", flush=True)

    started = time.time()
    n_sequences = 0
    for batch in calibration.batches(batch_size):
        source_content = cache_to_content(prefill(source_model, batch), source_model)
        target_content = cache_to_content(prefill(target_model, batch), target_model)

        for kind, accumulator in accumulators.items():
            x = _design_rows(source_content, layers, kind)
            y = _target_rows(target_content, kind)
            if token_stride > 1:
                x = x[::token_stride]
                y = y[::token_stride]
            accumulator.update(x, y)
            del x, y

        n_sequences += batch.shape[0]
        del source_content, target_content

        if progress and n_sequences % (batch_size * 16) == 0:
            print(
                f"  {n_sequences}/{len(calibration)} sequences, "
                f"{accumulators['keys'].n_tokens:,} tokens",
                flush=True,
            )

    report = HarvestReport(
        n_tokens=accumulators["keys"].n_tokens,
        n_sequences=n_sequences,
        seconds=time.time() - started,
        accumulator_bytes=sum(a.nbytes for a in accumulators.values()),
    )
    return {k: a.finalize() for k, a in accumulators.items()}, report
