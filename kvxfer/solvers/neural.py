"""A trained residual on top of the closed-form map.

Every closed-form variant tried here failed to improve on plain ridge, and the
reason is structural rather than incidental. In a closed-form solve the
attention metric can only enter through the penalty, and sweeping its influence
out of sample drives that influence to zero: the tuned solver *is* ridge. The
objective everyone actually cares about -- that attention produces the same
output -- cannot be written as a penalty on a linear map.

Training removes that constraint. The loss here is functional, comparing what
attention computes rather than how close the cache entries are:

    || softmax(q K_hat' / sqrt(d)) V_hat  -  softmax(q K' / sqrt(d)) V ||

which is the quantity the reference work's diagnostics implicate and which no
solver in this repository has been able to optimize directly. Reconstruction
loss is deliberately *not* the objective: the central finding of this project
is that reconstruction quality does not predict downstream behaviour, so an
MLP trained on mean squared error would be expected to reproduce the same null
and would teach nothing.

The closed-form map is kept and frozen as the base. It is free, it is already
a strong predictor, and starting from it means the network only has to learn
what the linear map cannot represent::

    KV_target = ridge(X) + g(X)

Design constraint. Mapping is only worth doing because it is cheaper than
letting the target model prefill: measured at roughly 4x on an L4 for a
1.7B-to-4B pair. A residual wide enough to erase that margin would be
self-defeating, so ``g`` is a bottleneck rather than a wide MLP -- at hidden
width 256 it adds about a third of the linear map's parameters, and the
measured budget allows up to about 822 before the speedup falls below 3x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kvxfer.geometry import KVGeometry
from kvxfer.mappers import design_rows
from kvxfer.solvers.ridge import LinearMap


class LayerResidual(nn.Module):
    """The correction one target layer's map could not express.

    A bottleneck rather than a wide layer: the design is ``k * kv_dim`` wide
    (4096 for a four-layer selection on eight KV heads), so a full-width hidden
    layer would cost more than the linear map it is correcting and eat the
    latency advantage that justifies mapping at all.

    Initialized to output zeros, so an untrained residual leaves the closed-form
    solution exactly unchanged and training starts from a known-good baseline
    rather than from noise.
    """

    def __init__(self, design_dim: int, kv_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.down = nn.Linear(design_dim, hidden, bias=True)
        self.up = nn.Linear(hidden, kv_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, design: Tensor) -> Tensor:
        return self.up(torch.nn.functional.gelu(self.down(design)))


@dataclass
class ResidualConfig:
    """Training settings for the residual mapper."""

    hidden: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    steps: int = 2000
    inner_steps: int = 4
    batch_sequences: int = 2
    seq_len: int = 512
    grad_clip: float = 1.0
    kl_weight: float = 0.0
    warmup: int = 100


def attention_output_loss(
    queries: Tensor,
    keys_true: Tensor,
    values_true: Tensor,
    keys_pred: Tensor,
    values_pred: Tensor,
) -> Tensor:
    """Distance between what attention computes from true and predicted caches.

    This is the objective the closed-form solvers could only approximate through
    a penalty. Keys enter through a softmax over positions, so an error matters
    exactly insofar as it moves attention weight, and values enter weighted by
    the attention they receive -- neither of which is what a reconstruction loss
    measures.

    Args:
        queries: ``(batch, n_q_heads, seq, head_dim)``, RoPE already applied.
        keys_true: ``(batch, n_kv_heads, seq, head_dim)``, RoPE already applied.
        values_true: ``(batch, n_kv_heads, seq, head_dim)``.
        keys_pred: predicted keys, same shape as ``keys_true``.
        values_pred: predicted values, same shape as ``values_true``.

    Returns:
        Scalar mean squared difference of attention outputs.
    """
    batch, n_q_heads, seq, head_dim = queries.shape
    n_kv_heads = keys_true.shape[1]
    group = n_q_heads // n_kv_heads

    def attend(keys: Tensor, values: Tensor) -> Tensor:
        k = keys.repeat_interleave(group, dim=1)
        v = values.repeat_interleave(group, dim=1)
        scores = (queries @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        # Causal: position i may only attend to j <= i, as during prefill.
        mask = torch.ones(seq, seq, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v

    with torch.no_grad():
        reference = attend(keys_true, values_true)
    return torch.nn.functional.mse_loss(attend(keys_pred, values_pred), reference)


class ResidualMapper(nn.Module):
    """Frozen closed-form map plus a trained per-layer correction.

    The base map is not a starting point that gets overwritten; it stays fixed
    and the network learns only the difference. That keeps the good properties
    of the closed form -- it costs nothing to fit, it generalizes across
    positions, and it is already most of the answer -- while giving the
    correction a functional objective the closed form cannot take.
    """

    def __init__(
        self,
        key_maps: dict[int, LinearMap],
        value_maps: dict[int, LinearMap],
        geometry: KVGeometry,
        design_dim: int,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.key_maps = key_maps
        self.value_maps = value_maps
        kv_dim = geometry.n_kv_heads * geometry.head_dim
        self.key_residual = nn.ModuleList(
            LayerResidual(design_dim, kv_dim, hidden) for _ in range(geometry.n_layers)
        )
        self.value_residual = nn.ModuleList(
            LayerResidual(design_dim, kv_dim, hidden) for _ in range(geometry.n_layers)
        )

    def base_and_residual(
        self, design: Tensor, layer: int, kind: str
    ) -> tuple[Tensor, Tensor]:
        """Return ``(closed_form, correction)`` for one layer's design block."""
        maps = self.key_maps if kind == "keys" else self.value_maps
        residual = self.key_residual if kind == "keys" else self.value_residual
        fit = maps[layer]

        weight = fit.weight if fit.weight is not None else fit.factors[0] @ fit.factors[1]
        weight = weight.to(design.device, design.dtype)
        bias = fit.bias.to(design.device, design.dtype)
        with torch.no_grad():
            base = design @ weight + bias
        return base, residual[layer](design)

    def n_trained_parameters(self) -> int:
        """Parameters the training actually updates."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _QueryTap:
    """Captures each layer's queries during a forward pass.

    The attention loss needs the target model's actual queries, not their
    second moments, so this keeps the vectors rather than accumulating
    ``q q'`` the way :mod:`kvxfer.queries` does for the closed-form metric.
    """

    def __init__(self, n_layers: int) -> None:
        self.queries: list[Tensor | None] = [None] * n_layers

    def make(self, layer: int):
        def hook(_module, _inputs, output: Tensor) -> None:
            self.queries[layer] = output.detach()

        return hook


def _attach_query_taps(model, geometry: KVGeometry) -> tuple[_QueryTap, list]:
    """Hook every layer's query projection. Returns ``(tap, handles)``.

    Raises:
        AttributeError: if the model exposes neither ``q_norm`` nor ``q_proj``,
            in which case the queries cannot be captured without
            reimplementing the attention block.
    """
    tap = _QueryTap(geometry.n_layers)
    backbone = getattr(model, "model", model)
    handles = []
    for idx, layer in enumerate(backbone.layers):
        attn = layer.self_attn
        # Qwen3 normalizes queries per head before RoPE; Qwen2 does not, so
        # fall back to the projection itself.
        module = getattr(attn, "q_norm", None) or getattr(attn, "q_proj", None)
        if module is None:
            raise AttributeError(
                f"layer {idx} exposes neither q_norm nor q_proj; cannot capture queries"
            )
        handles.append(module.register_forward_hook(tap.make(idx)))
    return tap, handles


def _shape_queries(raw: Tensor, geometry: KVGeometry) -> Tensor:
    """Normalize a captured query tensor to ``(batch, n_q_heads, seq, head_dim)``."""
    if raw.dim() == 4:                      # (batch, seq, heads, dim), from q_norm
        return raw.permute(0, 2, 1, 3)
    batch, seq, _ = raw.shape               # (batch, seq, heads * dim), from q_proj
    return raw.view(batch, seq, geometry.n_q_heads, geometry.head_dim).permute(
        0, 2, 1, 3
    )


@dataclass
class TrainingReport:
    """What a training run did, for the record and for the writeup."""

    steps: int
    trained_parameters: int
    initial_loss: float
    final_loss: float
    history: list[float]

    def __str__(self) -> str:
        change = (
            (self.final_loss / self.initial_loss - 1.0) * 100.0
            if self.initial_loss
            else 0.0
        )
        return (
            f"{self.steps} steps, {self.trained_parameters / 1e6:.1f}M trained "
            f"parameters, attention loss {self.initial_loss:.5f} -> "
            f"{self.final_loss:.5f} ({change:+.1f}%)"
        )


def train_residual(
    mapper: ResidualMapper,
    source_model,
    target_model,
    corpus,
    config: ResidualConfig,
    seed: int = 0,
) -> TrainingReport:
    """Train the residual on attention-output error.

    Caches are produced on the fly rather than stored. Storing them would be
    the usual choice, but for this pair a single 512-token sequence is about
    130 MB of content across both models, so a training set large enough to
    matter does not fit anywhere convenient. Each batch is instead reused for
    ``inner_steps`` gradient steps, which amortizes the two model forwards --
    the dominant cost -- without needing the storage.

    Args:
        mapper: the residual mapper to train, base map already frozen.
        source_model: model the cache is mapped from.
        target_model: model the cache is mapped to.
        corpus: a :class:`kvxfer.data.CalibrationSet` of token sequences.
        config: training settings.
        seed: batch sampling seed.

    Returns:
        A :class:`TrainingReport`.
    """
    from kvxfer.cache import cache_to_content, prefill, stack_cache
    from kvxfer.rope import apply_rope, restore_keys, rope_tables

    device = next(target_model.parameters()).device
    geometry = mapper.geometry
    mapper.to(device).train()

    trainable = [p for p in mapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=max(config.steps, 1),
        pct_start=min(0.3, config.warmup / max(config.steps, 1)),
    )

    generator = torch.Generator().manual_seed(seed)
    history: list[float] = []

    for step in range(config.steps):
        pick = torch.randint(
            0, len(corpus.input_ids), (config.batch_sequences,), generator=generator
        )
        batch = corpus.input_ids[pick][:, : config.seq_len]

        with torch.no_grad():
            source_content = cache_to_content(prefill(source_model, batch), source_model)

            tap, handles = _attach_query_taps(target_model, geometry)
            try:
                target_cache = prefill(target_model, batch)
            finally:
                for handle in handles:
                    handle.remove()
            true_keys, true_values = stack_cache(target_cache)

            positions = source_content.position_ids
            cos, sin = rope_tables(target_model, positions)
            design_cache = source_content.keys, source_content.values
            del target_cache

        # Several gradient steps per pair of forwards. The forwards dominate,
        # and the targets are deterministic, so repeating them is free signal.
        for _ in range(config.inner_steps):
            total = torch.zeros((), device=device)
            for layer in range(geometry.n_layers):
                predicted = {}
                for kind, source_side in zip(("keys", "values"), design_cache):
                    maps = mapper.key_maps if kind == "keys" else mapper.value_maps
                    selected = maps[layer].source_layers
                    # Same helper the fitted mapper uses, so the frozen base
                    # map sees features in the order it was fitted against.
                    design = design_rows(source_side, selected).to(device)
                    b_, s_ = source_side.shape[1], source_side.shape[3]

                    base, correction = mapper.base_and_residual(design, layer, kind)
                    out = (base + correction).reshape(
                        b_, s_, geometry.n_kv_heads, geometry.head_dim
                    ).permute(0, 2, 1, 3)
                    predicted[kind] = out

                # Keys are predicted in content space; attention sees them
                # rotated, so the loss must too.
                keys_pred = apply_rope(predicted["keys"], cos, sin)
                queries = apply_rope(
                    _shape_queries(tap.queries[layer], geometry).float(), cos, sin
                )
                total = total + attention_output_loss(
                    queries,
                    true_keys[layer].float(),
                    true_values[layer].float(),
                    keys_pred,
                    predicted["values"],
                )

            loss = total / geometry.n_layers
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()
            history.append(float(loss.detach()))

        schedule.step()
        if step % 25 == 0:
            print(f"  step {step:>5} attention loss {history[-1]:.6f}", flush=True)

    mapper.eval()
    return TrainingReport(
        steps=config.steps,
        trained_parameters=mapper.n_trained_parameters(),
        initial_loss=history[0] if history else float("nan"),
        final_loss=history[-1] if history else float("nan"),
        history=history,
    )


def _residual_map(self, source: "ContentKV") -> "ContentKV":
    """Map a source cache through the frozen base plus the trained residual.

    Mirrors :meth:`kvxfer.mappers.FittedMapper.map` so a trained residual can
    be dropped into the same evaluation and latency paths as any other mapper.
    Without this the residual could be trained but not measured, which is how a
    model ends up reported on its training loss.
    """
    from kvxfer.cache import ContentKV

    _, batch, _, seq, _ = source.keys.shape
    mapped = {}
    for kind, side in (("keys", source.keys), ("values", source.values)):
        out = torch.empty(
            self.geometry.n_layers,
            batch,
            self.geometry.n_kv_heads,
            seq,
            self.geometry.head_dim,
            dtype=torch.float32,
            device=side.device,
        )
        for layer in range(self.geometry.n_layers):
            maps = self.key_maps if kind == "keys" else self.value_maps
            design = design_rows(side, maps[layer].source_layers)
            base, correction = self.base_and_residual(design, layer, kind)
            out[layer] = (
                (base + correction)
                .reshape(batch, seq, self.geometry.n_kv_heads, self.geometry.head_dim)
                .permute(0, 2, 1, 3)
                .to(torch.float32)
            )
        mapped[kind] = out

    return ContentKV(
        keys=mapped["keys"], values=mapped["values"], position_ids=source.position_ids
    )


def _residual_name(self) -> str:
    return getattr(self, "label", "residual")


ResidualMapper.map = _residual_map
ResidualMapper.name = property(_residual_name)


def load_residual(
    path,
    key_maps: dict[int, LinearMap],
    value_maps: dict[int, LinearMap],
    geometry: KVGeometry,
    label: str = "residual",
) -> ResidualMapper:
    """Rebuild a trained residual mapper from disk.

    Args:
        path: directory containing ``residual.pt``.
        key_maps: the frozen base key maps it was trained on top of.
        value_maps: the frozen base value maps.
        geometry: target model geometry.
        label: name used in results tables.

    Returns:
        The mapper, in eval mode.
    """
    from pathlib import Path

    state = torch.load(Path(path) / "residual.pt", weights_only=False)
    mapper = ResidualMapper(
        key_maps, value_maps, geometry, state["design_dim"], hidden=state["hidden"]
    )
    mapper.key_residual.load_state_dict(state["key_residual"])
    mapper.value_residual.load_state_dict(state["value_residual"])
    mapper.label = label
    mapper.eval()
    return mapper
