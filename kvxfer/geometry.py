"""KV-cache geometry introspection and source/target pair compatibility.

The transfer method requires *matched KV geometry*: the source and target models
must agree on ``n_kv_heads`` and ``head_dim``, so a source head maps onto the
corresponding target head without any reshaping. Layer counts may differ -- the
mapper selects source layers per target layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoConfig


@dataclass(frozen=True)
class KVGeometry:
    """Everything about a model that determines its KV cache layout."""

    model_id: str
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    rope_theta: float
    vocab_size: int = 0

    @property
    def kv_dim(self) -> int:
        """Width of one layer's K (or V) tensor for a single token."""
        return self.n_kv_heads * self.head_dim

    @property
    def content_dim(self) -> int:
        """Width of the full-model content vector used as regression input.

        This is the dimensionality of one token's KV state across *all* layers,
        which is the input space the full-Gram sufficient statistics live in.
        """
        return self.n_layers * self.kv_dim

    def bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Cache footprint of a single token across all layers, K and V."""
        return 2 * self.n_layers * self.kv_dim * dtype_bytes

    def __str__(self) -> str:
        return (
            f"{self.model_id}: {self.n_layers}L x {self.n_kv_heads}kv "
            f"x {self.head_dim}d (q_heads={self.n_q_heads}, hidden={self.hidden_size})"
        )


class IncompatiblePairError(ValueError):
    """Raised when two models cannot share a KV mapping."""


def load_geometry(model_id: str) -> KVGeometry:
    """Read KV geometry from a model's config without downloading weights."""
    cfg = AutoConfig.from_pretrained(model_id)

    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads

    n_kv_heads = getattr(cfg, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = cfg.num_attention_heads

    return KVGeometry(
        model_id=model_id,
        n_layers=cfg.num_hidden_layers,
        n_q_heads=cfg.num_attention_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=cfg.hidden_size,
        rope_theta=_rope_theta(cfg),
        vocab_size=int(getattr(cfg, "vocab_size", 0)),
    )


def _rope_theta(cfg) -> float:
    """Read rope_theta across transformers versions.

    transformers>=5 nests it under ``rope_parameters``; earlier versions expose
    it as a top-level attribute. There is deliberately no default: a silently
    wrong rope_theta would corrupt every content-space mapping without any
    visible error, so an unreadable config must raise.
    """
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])

    try:
        return float(cfg.rope_theta)
    except AttributeError:
        pass

    raise AttributeError(
        f"cannot determine rope_theta for {cfg.__class__.__name__}; "
        "checked cfg.rope_parameters['rope_theta'] and cfg.rope_theta"
    )


def check_pair(source: KVGeometry, target: KVGeometry) -> None:
    """Assert that a source -> target KV mapping is well posed.

    Matched KV geometry is necessary but not sufficient. The pair must also
    share a tokenizer, because one set of token ids is prefilled through both
    models and the cache positions have to correspond. Differing vocabulary
    sizes are the reliable signal that they do not: Mistral-7B-v0.3 tokenizes
    to 32,768 ids and Ministral-8B to 131,072, and feeding the first model's
    ids to the second produced chance-level accuracy on every condition that
    ran through the target -- while the source, scored with its own tokenizer,
    looked perfectly healthy at 0.82.

    Raises:
        IncompatiblePairError: if head counts, head dims, or vocabularies differ.
    """
    problems = []
    if source.vocab_size != target.vocab_size:
        problems.append(
            f"vocab_size differs: source={source.vocab_size} "
            f"target={target.vocab_size}, so the two do not share a tokenizer"
        )
    if source.n_kv_heads != target.n_kv_heads:
        problems.append(
            f"n_kv_heads differ: source={source.n_kv_heads} target={target.n_kv_heads}"
        )
    if source.head_dim != target.head_dim:
        problems.append(
            f"head_dim differs: source={source.head_dim} target={target.head_dim}"
        )
    if problems:
        raise IncompatiblePairError(
            f"{source.model_id} -> {target.model_id} is not a matched-KV pair: "
            + "; ".join(problems)
        )


def rope_compatible(source: KVGeometry, target: KVGeometry) -> bool:
    """Whether both models rotate content vectors at identical frequencies.

    When true, a content-space mapping is position-independent in the strong
    sense: the same relative-position rotation applies on both sides. When
    false, the mapper still works (RoPE is stripped and re-applied with each
    model's own frequencies) but relative-position geometry is not preserved,
    which is expected to cost accuracy at long context.
    """
    return source.rope_theta == target.rope_theta and source.head_dim == target.head_dim
