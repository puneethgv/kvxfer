"""Persisting fitted maps so that fitting and evaluation are separate jobs.

Fitting holds gigabytes of calibration statistics; evaluation holds two models.
Neither needs what the other holds, and doing both in one process makes the
peak their sum. On a 16 GB machine that sum is what the out-of-memory killer
reacts to -- twice, in this project's case, once losing a completed fit that
then had to be redone.

Separating them also makes re-evaluation free. Changing a task, an item limit
or a batch size no longer costs a refit, which matters because the fit is the
slow half and the part least likely to be what changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from kvxfer.geometry import KVGeometry
from kvxfer.mappers import FittedMapper
from kvxfer.solvers.ridge import LinearMap

KINDS = ("keys", "values")


def _map_to_state(fit: LinearMap) -> dict:
    return {
        "weight": fit.weight,
        "bias": fit.bias,
        "source_layers": list(fit.source_layers),
        "target_layer": fit.target_layer,
        "r2": fit.r2,
        "rank": fit.rank,
        "factors": list(fit.factors) if fit.factors is not None else None,
    }


def _state_to_map(state: dict) -> LinearMap:
    factors = state.get("factors")
    return LinearMap(
        weight=state["weight"],
        bias=state["bias"],
        source_layers=tuple(state["source_layers"]),
        target_layer=state["target_layer"],
        r2=state["r2"],
        rank=state.get("rank"),
        factors=tuple(factors) if factors is not None else None,
    )


def save_maps(
    out_dir: str | Path,
    maps: dict[str, dict[str, dict[int, LinearMap]]],
    metadata: dict,
) -> Path:
    """Write every variant's fitted maps and the metadata describing them.

    Args:
        out_dir: directory to write into; created if absent.
        maps: ``{variant: {kind: {target_layer: LinearMap}}}``.
        metadata: diagnostics, layer selection, token counts and settings.

    Returns:
        The directory written to.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for variant, by_kind in maps.items():
        for kind, layers in by_kind.items():
            torch.save(
                {layer: _map_to_state(fit) for layer, fit in layers.items()},
                out / f"{variant}__{kind}.pt",
            )

    payload = dict(metadata)
    payload["variants"] = sorted(maps)
    (out / "maps.json").write_text(json.dumps(payload, indent=2))
    return out


def load_maps(
    maps_dir: str | Path, target_geom: KVGeometry
) -> tuple[dict[str, FittedMapper], dict]:
    """Load previously fitted maps as ready-to-use mappers.

    Args:
        maps_dir: a directory written by :func:`save_maps`.
        target_geom: geometry of the target model.

    Returns:
        ``(mappers, metadata)``.

    Raises:
        FileNotFoundError: if the directory holds no map manifest.
    """
    path = Path(maps_dir)
    manifest = path / "maps.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no maps.json in {path}")

    metadata = json.loads(manifest.read_text())
    mappers = {}
    for variant in metadata["variants"]:
        loaded = {}
        for kind in KINDS:
            state = torch.load(path / f"{variant}__{kind}.pt", weights_only=False)
            loaded[kind] = {int(k): _state_to_map(v) for k, v in state.items()}
        mappers[variant] = FittedMapper(
            loaded["keys"], loaded["values"], target_geom, label=variant
        )
    return mappers, metadata
