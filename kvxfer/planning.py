"""Deciding how a harvest should be run on the machine actually running it.

Harvesting keys and values in one sweep halves the model forward passes, which
normally dominate the cost. But it holds two Gram accumulators at once, and on
a memory-constrained machine that is a false economy: exceeding physical memory
costs far more than the forward passes it saves. Measured on a 16 GB laptop,
the combined sweep drove the system to 10.7 GB of swap and ran roughly twenty
times slower per sequence than two separate sweeps.

The decision therefore has to be made from measured memory rather than from
token counts, and it has to be made the same way locally and on a rented GPU --
a run that silently thrashes is indistinguishable from a run that is merely
slow until the bill arrives.
"""

from __future__ import annotations

import torch

from kvxfer.geometry import KVGeometry

# Allocator slack and transient activations that are not worth modelling
# individually. Small now that the working set below is computed rather than
# guessed, and that prefill no longer materializes logits.
_SLACK_BYTES = 1536 * 1024**2

# The first attempt at this heuristic swapped at a projected 84% of the
# reported budget: allocator fragmentation and the host's own demand are not
# visible from inside the process, so leave room for them.
_HEADROOM = 0.85


def accumulator_bytes(
    source_geom: KVGeometry, target_geom: KVGeometry, source_layers: tuple[int, ...]
) -> int:
    """Footprint of one float32 Gram accumulator for a given layer pool.

    Args:
        source_geom: geometry of the model being mapped from.
        target_geom: geometry of the model being mapped to.
        source_layers: candidate source layers, all of which enter the design.

    Returns:
        Size in bytes, dominated by ``X'X`` and ``X'Y``.
    """
    dim = len(source_layers) * source_geom.kv_dim
    cross = dim * target_geom.n_layers * target_geom.kv_dim
    per_head = target_geom.n_layers * target_geom.n_kv_heads * target_geom.head_dim**2
    return (dim * dim + cross + per_head) * 4


def model_bytes(model) -> int:
    """Parameter footprint of a loaded model.

    Measured rather than inferred from the config, which understated a 0.6B and
    1.7B pair by about a gigabyte -- enough to flip the decision below.
    """
    return sum(p.numel() * p.element_size() for p in model.parameters())


def working_set_bytes(
    source_geom: KVGeometry,
    target_geom: KVGeometry,
    batch_size: int,
    seq_len: int,
) -> int:
    """Bytes held per harvested batch, outside the accumulators.

    Each model's cache is materialized in its compute dtype and then converted
    to float32 content for both keys and values across every layer. That
    conversion, not the forward pass, is the large transient: a 4B target at
    batch 4 and 1024 tokens is about 1.2 GB of float32 content on its own.

    Args:
        source_geom: geometry of the model being mapped from.
        target_geom: geometry of the model being mapped to.
        batch_size: sequences per forward pass.
        seq_len: tokens per sequence.

    Returns:
        Estimated bytes, counting float32 content plus the bfloat16 caches it
        was converted from, for keys and values across both models.
    """
    per_model = 0
    for geom in (source_geom, target_geom):
        elements = geom.n_layers * batch_size * seq_len * geom.kv_dim
        per_model += elements * 2 * (4 + 2)  # keys and values, float32 + bf16
    return per_model


def device_budget() -> int:
    """Memory actually available to this process, or 0 if unknown.

    Reports *free* memory rather than the card's nominal capacity. The two
    differ by more than rounding: an L4 advertised as 24 GB reports 23.66 GB
    total and only 22.03 GiB usable, and by the time this is consulted the
    model weights are already resident, so free memory is the quantity that
    answers the question being asked.
    """
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    try:
        return int(torch.mps.recommended_max_memory())
    except Exception:
        return 0


def choose_harvest_strategy(
    setting: str,
    accumulator: int,
    source_model,
    target_model,
    working_set: int = 0,
    verbose: bool = True,
) -> bool:
    """Decide whether to sweep keys and values together.

    Consulted after both models are resident, so their weights are already
    subtracted from the free memory this compares against; only what the
    harvest is about to allocate has to be predicted.

    Args:
        setting: ``"auto"``, ``"single"`` (together) or ``"split"`` (separately).
        accumulator: bytes for one accumulator, from :func:`accumulator_bytes`.
        source_model: the loaded source model, for reporting only.
        target_model: the loaded target model, for reporting only.
        working_set: per-batch bytes from :func:`working_set_bytes`.
        verbose: print the arithmetic behind an ``auto`` decision.

    Returns:
        True to sweep both kinds together. When the budget cannot be read,
        returns False: the memory-safe option, since the penalty for splitting
        unnecessarily is a factor of two and the penalty for not splitting when
        it was needed is a factor of twenty locally, or a failed paid run
        remotely.
    """
    if setting == "single":
        return True
    if setting == "split":
        return False

    free = device_budget()
    if not free:
        if verbose:
            print("  memory check: no device budget reported -> split")
        return False

    weights = model_bytes(source_model) + model_bytes(target_model)
    needed = 2 * accumulator + working_set + _SLACK_BYTES
    fits = needed < _HEADROOM * free
    if verbose:
        print(
            f"  memory check: {weights / 1024**3:.1f} GB of weights resident, "
            f"{free / 1024**3:.1f} GB free. Single-pass would add "
            f"~{needed / 1024**3:.1f} GB (accumulators "
            f"{2 * accumulator / 1024**3:.1f} + working set "
            f"{working_set / 1024**3:.1f} + slack "
            f"{_SLACK_BYTES / 1024**3:.1f}) -> {'single' if fits else 'split'}"
        )
    return fits


def report_headroom(
    accumulator: int, working_set: int, split: bool, verbose: bool = True
) -> bool:
    """Check that the chosen strategy actually fits, and say so.

    Called for an explicit ``--passes`` choice as well as an automatic one, so
    that forcing ``split`` on a machine where even one accumulator does not fit
    fails loudly at the start rather than partway through a paid run.

    Returns:
        True if the projection fits within the headroom factor.
    """
    free = device_budget()
    if not free:
        return True
    needed = (1 if split else 2) * accumulator + working_set + _SLACK_BYTES
    fits = needed < _HEADROOM * free
    if verbose:
        verdict = "fits" if fits else "DOES NOT FIT -- expect an allocation failure"
        print(
            f"  projected peak: {needed / 1024**3:.1f} GB against "
            f"{free / 1024**3:.1f} GB free ({_HEADROOM:.0%} headroom) -> {verdict}"
        )
    return fits
