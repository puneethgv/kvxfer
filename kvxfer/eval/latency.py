"""Wall-clock cost of transferring a cache versus recomputing it.

This is the measurement the method's premise rests on. Cross-model KV transfer
does not have to improve quality -- it has to be *cheaper* than letting the
target model prefill the prompt itself. On the pairs measured here it does not
improve downstream accuracy over simply running the source model, so latency is
not a secondary number, it is the entire remaining case.

Two comparisons are reported, because they answer different questions:

* **warm** -- the escalation setting the reference work assumes. The source
  model already ran (that is *why* you are escalating), so its prefill is sunk
  cost and only mapping and injection count against re-prefilling the target.
* **cold** -- nobody has run anything yet. The source prefill is then part of
  the price, and the comparison is much less favourable. Reporting only the
  warm number would overstate the method exactly the way retention-against-
  target does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from kvxfer.cache import CacheTemplate, cache_to_content, prefill
from kvxfer.mappers import Mapper


@dataclass
class LatencyPoint:
    """Timings at one context length, in milliseconds."""

    n_tokens: int
    target_prefill_ms: float
    source_prefill_ms: float
    map_ms: float
    inject_ms: float

    @property
    def warm_ms(self) -> float:
        """Cost when the source cache already exists."""
        return self.map_ms + self.inject_ms

    @property
    def cold_ms(self) -> float:
        """Cost when the source must be prefilled too."""
        return self.source_prefill_ms + self.warm_ms

    @property
    def warm_speedup(self) -> float:
        return self.target_prefill_ms / max(self.warm_ms, 1e-9)

    @property
    def cold_speedup(self) -> float:
        return self.target_prefill_ms / max(self.cold_ms, 1e-9)

    def __str__(self) -> str:
        return (
            f"{self.n_tokens:>6} tok | target prefill {self.target_prefill_ms:8.1f} ms | "
            f"map {self.map_ms:7.1f} + inject {self.inject_ms:6.1f} ms | "
            f"warm {self.warm_speedup:5.2f}x | cold {self.cold_speedup:5.2f}x"
        )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time(fn, device: torch.device, repeats: int, warmup: int = 1) -> tuple[float, object]:
    """Median wall-clock milliseconds over ``repeats`` runs, plus the last result.

    The median rather than the mean: a single scheduling hiccup on a shared GPU
    should not decide a speedup number.
    """
    out = None
    for _ in range(warmup):
        out = fn()
    _sync(device)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2], out


@torch.no_grad()
def benchmark_lengths(
    source_model,
    target_model,
    mapper: Mapper,
    lengths: tuple[int, ...] = (512, 1024, 2048, 4096, 8192),
    repeats: int = 5,
    dtype: torch.dtype = torch.bfloat16,
    vocab_size: int | None = None,
    seed: int = 0,
) -> list[LatencyPoint]:
    """Time re-prefill against map-and-inject across context lengths.

    Each phase releases its tensors before the next begins. That is not tidiness:
    holding the target cache, the source cache, both float32 content copies, the
    template and the built cache at once comes to 18.9 GB at 8192 tokens for a
    1.7B-to-4B pair, which does not fit a 22 GiB card alongside 10.7 GB of
    weights. Staged, the same measurement peaks around 15 GB.

    Lengths are attempted in order and a length that runs out of memory ends the
    sweep rather than the run, so the shorter points already measured survive.

    Args:
        source_model: the model whose cache is being mapped from.
        target_model: the model whose cache is being produced.
        mapper: the fitted map under test.
        lengths: context lengths in tokens, ascending.
        repeats: timed runs per measurement; the median is reported.
        dtype: cache dtype, matching how the models are served.
        vocab_size: sampling range for synthetic tokens; taken from the target
            config when omitted.
        seed: token sampling seed.

    Returns:
        One :class:`LatencyPoint` per length that completed.
    """
    device = next(target_model.parameters()).device
    vocab = vocab_size or int(target_model.config.vocab_size)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Place the maps on the accelerator once. A deployed mapper would load them
    # there and keep them there; timing a per-call host-to-device copy would
    # measure a loading strategy nobody would ship.
    if hasattr(mapper, "to"):
        mapper.to(device, dtype)

    def release() -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    points: list[LatencyPoint] = []
    for n_tokens in sorted(lengths):
        ids = torch.randint(0, vocab, (1, n_tokens), generator=generator)
        try:
            target_ms, out = _time(lambda: prefill(target_model, ids), device, repeats)
            del out
            release()

            source_ms, source_cache = _time(
                lambda: prefill(source_model, ids), device, repeats
            )
            content = cache_to_content(source_cache, source_model)
            del source_cache
            release()

            # Content conversion is part of mapping: the map lives in RoPE-free
            # space, so stripping and restoring are not overhead to omit.
            map_ms, mapped = _time(lambda: mapper.map(content), device, repeats)
            del content
            release()

            inject_ms, built = _time(
                lambda: CacheTemplate.from_content(
                    mapped, target_model, dtype=dtype
                ).build(),
                device,
                repeats,
            )
            del mapped, built
            release()
        except torch.OutOfMemoryError:
            print(
                f"  {n_tokens:>6} tok | out of memory; stopping the sweep and "
                f"keeping the {len(points)} shorter point(s)",
                flush=True,
            )
            release()
            break

        point = LatencyPoint(
            n_tokens=n_tokens,
            target_prefill_ms=target_ms,
            source_prefill_ms=source_ms,
            map_ms=map_ms,
            inject_ms=inject_ms,
        )
        points.append(point)
        print(f"  {point}", flush=True)

    return points
