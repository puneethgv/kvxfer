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

# Activations, the float32 content copies for both models, and allocator slack.
# Set from what a failed run actually consumed, not from theory.
_OVERHEAD_BYTES = 3 * 1024**3

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


def device_budget() -> int:
    """Usable device memory in bytes, or 0 if it cannot be determined."""
    if torch.cuda.is_available():
        return int(torch.cuda.get_device_properties(0).total_memory)
    try:
        return int(torch.mps.recommended_max_memory())
    except Exception:
        return 0


def choose_harvest_strategy(
    setting: str, accumulator: int, source_model, target_model, verbose: bool = True
) -> bool:
    """Decide whether to sweep keys and values together.

    Args:
        setting: ``"auto"``, ``"single"`` (together) or ``"split"`` (separately).
        accumulator: bytes for one accumulator, from :func:`accumulator_bytes`.
        source_model: the loaded source model.
        target_model: the loaded target model.
        verbose: print the arithmetic behind an ``auto`` decision.

    Returns:
        True to sweep both kinds together. When the budget cannot be read,
        returns False: the memory-safe option, since the penalty for splitting
        unnecessarily is a factor of two and the penalty for not splitting when
        it was needed is a factor of twenty.
    """
    if setting == "single":
        return True
    if setting == "split":
        return False

    weights = model_bytes(source_model) + model_bytes(target_model)
    budget = device_budget()
    if not budget:
        if verbose:
            print("  memory check: no device budget reported -> split")
        return False

    needed = 2 * accumulator + weights + _OVERHEAD_BYTES
    fits = needed < _HEADROOM * budget
    if verbose:
        print(
            f"  memory check: single-pass needs ~{needed / 1024**3:.1f} GB "
            f"(weights {weights / 1024**3:.1f} + accumulators "
            f"{2 * accumulator / 1024**3:.1f} + overhead "
            f"{_OVERHEAD_BYTES / 1024**3:.1f}) against a "
            f"{budget / 1024**3:.1f} GB budget -> {'single' if fits else 'split'}"
        )
    return fits
