"""Streaming sufficient statistics for closed-form KV mappers.

Ridge regression needs only ``X'X`` and ``X'Y``, never the samples themselves.
So calibration is a single streaming pass that never stores an activation: we
accumulate the Gram over *all* source layers at once, and every later choice of
source-layer subset is a submatrix of it.

That is what makes the study affordable. Selecting k, sweeping lambda, and
re-running layer-selection ablations all become linear algebra on a cached
matrix rather than fresh GPU passes over the calibration corpus.

Keys and values are accumulated in separate passes. They need separate
statistics (target keys are predicted from source keys, values from values) and
splitting the passes halves peak memory, which matters more than the extra
forward pass -- prefilling the calibration set is a couple of GPU-minutes even
for a 4B model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from kvxfer.geometry import KVGeometry


@dataclass
class GramStats:
    """Accumulated second moments linking source and target KV states.

    All matrices are float32. Accumulation happens per chunk in float32 too:
    a chunk's Gram has magnitude on the order of the chunk size, so summing a
    few dozen of them stays far inside float32's precision, unlike accumulating
    sample by sample.

    Attributes:
        xtx: ``(D, D)`` source Gram, where ``D = n_source_layers * kv_dim``.
        xty: ``(D, n_target_layers, kv_dim)`` source-target cross moments.
        x_sum: ``(D,)`` source totals, for the intercept.
        y_sum: ``(n_target_layers, kv_dim)`` target totals, for the intercept.
        yty_diag: ``(n_target_layers, kv_dim)`` target second moments, the
            denominator of per-coordinate R².
        yty_head: ``(n_target_layers, n_kv_heads, head_dim, head_dim)`` full
            per-head target second moments. Needed to score a fit under a
            non-diagonal metric, which the diagonal alone cannot support. Only
            the per-head blocks are kept, not the whole ``kv_dim`` square,
            because the metrics are block diagonal by head -- that is 15 MB
            rather than 117 MB for a 28-layer target. Optional so that
            statistics written before this existed still load.
        n_tokens: number of tokens accumulated.
        kind: ``"keys"`` or ``"values"``.
        source_layers: which source layers ``D`` indexes, in order.
    """

    xtx: Tensor
    xty: Tensor
    x_sum: Tensor
    y_sum: Tensor
    yty_diag: Tensor
    n_tokens: int
    kind: str
    source_layers: tuple[int, ...]
    kv_dim: int
    yty_head: Tensor | None = None

    @property
    def n_source_layers(self) -> int:
        return len(self.source_layers)

    @property
    def n_target_layers(self) -> int:
        return self.xty.shape[1]

    def layer_slice(self, layer: int) -> slice:
        """Column range of one source layer inside the concatenated design."""
        idx = self.source_layers.index(layer)
        return slice(idx * self.kv_dim, (idx + 1) * self.kv_dim)

    def select(self, layers: tuple[int, ...]) -> tuple[Tensor, Tensor, Tensor]:
        """Extract the design statistics for a subset of source layers.

        This is the payoff of accumulating the full Gram: choosing a different
        set of source layers costs an index operation, not a calibration run.

        Returns:
            ``(xtx, xty, x_sum)`` restricted to ``layers``, with ``xtx`` of
            shape ``(k*kv_dim, k*kv_dim)``.
        """
        idx = torch.cat(
            [
                torch.arange(
                    self.source_layers.index(l) * self.kv_dim,
                    (self.source_layers.index(l) + 1) * self.kv_dim,
                )
                for l in layers
            ]
        )
        return self.xtx[idx][:, idx], self.xty[idx], self.x_sum[idx]

    def save(self, path: str | Path) -> None:
        """Persist to disk. These are the expensive artifacts; everything else
        downstream is cheap to recompute from them."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "xtx": self.xtx.cpu(),
                "xty": self.xty.cpu(),
                "x_sum": self.x_sum.cpu(),
                "y_sum": self.y_sum.cpu(),
                "yty_diag": self.yty_diag.cpu(),
                "yty_head": None if self.yty_head is None else self.yty_head.cpu(),
                "n_tokens": self.n_tokens,
                "kind": self.kind,
                "source_layers": self.source_layers,
                "kv_dim": self.kv_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GramStats":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**blob)


class GramAccumulator:
    """Builds :class:`GramStats` incrementally over calibration batches."""

    def __init__(
        self,
        source: KVGeometry,
        target: KVGeometry,
        kind: str,
        source_layers: tuple[int, ...] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Args:
            source: source model geometry.
            target: target model geometry.
            kind: ``"keys"`` or ``"values"``.
            source_layers: candidate source layers. Defaults to all of them.
                Restricting this is the memory dial: the Gram is quadratic in
                ``len(source_layers) * kv_dim``, so halving the candidates
                quarters the accumulator.
            device: where to accumulate. CPU keeps accelerator memory free for
                the models, which is usually the binding constraint.
        """
        if kind not in ("keys", "values"):
            raise ValueError(f"kind must be 'keys' or 'values', got {kind!r}")

        self.source = source
        self.target = target
        self.kind = kind
        self.source_layers = source_layers or tuple(range(source.n_layers))
        self.kv_dim = source.kv_dim
        self.device = torch.device(device)

        dim = len(self.source_layers) * self.kv_dim
        self.xtx = torch.zeros(dim, dim, dtype=torch.float32, device=self.device)
        self.xty = torch.zeros(
            dim, target.n_layers, target.kv_dim, dtype=torch.float32, device=self.device
        )
        self.x_sum = torch.zeros(dim, dtype=torch.float32, device=self.device)
        self.y_sum = torch.zeros(
            target.n_layers, target.kv_dim, dtype=torch.float32, device=self.device
        )
        self.yty_diag = torch.zeros(
            target.n_layers, target.kv_dim, dtype=torch.float32, device=self.device
        )
        self.yty_head = torch.zeros(
            target.n_layers,
            target.n_kv_heads,
            target.head_dim,
            target.head_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.n_tokens = 0
        self._n_kv_heads = target.n_kv_heads
        self._head_dim = target.head_dim

    @property
    def nbytes(self) -> int:
        """Accumulator footprint, for sizing a run before starting it."""
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.xtx,
                self.xty,
                self.x_sum,
                self.y_sum,
                self.yty_diag,
                self.yty_head,
            )
        )

    @torch.no_grad()
    def update(self, source_tokens: Tensor, target_tokens: Tensor) -> None:
        """Accumulate one chunk.

        Args:
            source_tokens: ``(n_tokens, n_source_layers * kv_dim)`` design rows,
                restricted to the candidate source layers.
            target_tokens: ``(n_tokens, n_target_layers, kv_dim)`` targets.
        """
        x = source_tokens.to(self.device, torch.float32)
        y = target_tokens.to(self.device, torch.float32)

        if x.shape[1] != self.xtx.shape[0]:
            raise ValueError(
                f"design width {x.shape[1]} does not match accumulator "
                f"{self.xtx.shape[0]}; check the source-layer selection"
            )

        n, n_tgt, kv = y.shape
        self.xtx += x.T @ x
        # (D, n) @ (n, n_tgt*kv) -> (D, n_tgt, kv)
        self.xty += (x.T @ y.reshape(n, n_tgt * kv)).reshape(-1, n_tgt, kv)
        self.x_sum += x.sum(dim=0)
        self.y_sum += y.sum(dim=0)
        self.yty_diag += (y * y).sum(dim=0)

        # Per-head blocks, needed to score fits under a non-diagonal metric.
        heads = y.reshape(n, n_tgt, self._n_kv_heads, self._head_dim)
        self.yty_head += torch.einsum("nlhi,nlhj->lhij", heads, heads)

        self.n_tokens += n

    def finalize(self) -> GramStats:
        """Freeze into an immutable :class:`GramStats`."""
        if self.n_tokens == 0:
            raise RuntimeError("no tokens accumulated; nothing to finalize")
        return GramStats(
            xtx=self.xtx,
            xty=self.xty,
            x_sum=self.x_sum,
            y_sum=self.y_sum,
            yty_diag=self.yty_diag,
            n_tokens=self.n_tokens,
            kind=self.kind,
            source_layers=self.source_layers,
            kv_dim=self.kv_dim,
            yty_head=self.yty_head,
        )
