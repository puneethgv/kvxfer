"""Correctness gates 1, 3 and 4: the evaluation path must be lossless.

These load real weights (marked ``slow``). They deliberately test the harness
rather than any mapper: if injecting a model's own cache does not reproduce its
standalone behaviour, every transfer number downstream is wrong by an unknown
offset.
"""

from __future__ import annotations

import pytest
import torch

from kvxfer.cache import cache_to_content, content_to_cache, prefill, stack_cache
from kvxfer.eval.scoring import score_continuation
from kvxfer.geometry import load_geometry
from kvxfer.mappers import IdentityMapper, OracleMapper, ZeroMapper
from kvxfer.models import load_model, load_tokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

PROMPT = (
    "The Antikythera mechanism is an ancient Greek hand-powered device that has "
    "been identified as the world's oldest known analogue computer. It was used "
    "to predict astronomical positions and eclipses decades in advance, and to "
    "track the four-year cycle of athletic games. It was recovered in 1901 from "
    "a shipwreck off the coast of the Greek island of"
)
CONTINUATION = " Antikythera, and has been studied ever since by historians."

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    # float32 so that gate failures mean real bugs, not bf16 rounding.
    return load_model(MODEL_ID, dtype=torch.float32)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(MODEL_ID)


@pytest.fixture(scope="module")
def ids(tokenizer):
    context = tokenizer(PROMPT, return_tensors="pt").input_ids
    full = tokenizer(PROMPT + CONTINUATION, return_tensors="pt").input_ids
    return full, context.shape[1]


def test_cache_content_roundtrip_is_lossless(model, ids):
    """cache -> content space -> cache must return the original keys."""
    full_ids, n_context = ids
    cache = prefill(model, full_ids[:, :n_context])
    original_keys, original_values = stack_cache(cache)

    content = cache_to_content(cache, model)
    rebuilt = content_to_cache(content, model, dtype=torch.float32)
    rebuilt_keys, rebuilt_values = stack_cache(rebuilt)

    key_err = (rebuilt_keys - original_keys).abs().max().item()
    val_err = (rebuilt_values - original_values).abs().max().item()
    assert key_err < 1e-4, f"key round trip lost precision: max err {key_err:.3e}"
    assert val_err == 0.0, f"values must pass through untouched, got {val_err:.3e}"


def test_identity_injection_matches_standalone(model, ids):
    """GATE 1: scoring through an injected self-cache equals standalone scoring.

    This is the load-bearing test of the whole project.
    """
    full_ids, n_context = ids

    standalone = score_continuation(model, full_ids, n_context, cache=None)

    n_cached = n_context - 1
    cache = prefill(model, full_ids[:, :n_cached])
    content = IdentityMapper().map(cache_to_content(cache, model))
    injected_cache = content_to_cache(content, model, dtype=torch.float32)
    injected = score_continuation(
        model, full_ids, n_context, cache=injected_cache, n_cached=n_cached
    )

    assert injected.n_tokens == standalone.n_tokens
    delta = abs(injected.total_logprob - standalone.total_logprob)
    assert delta < 1e-2, (
        f"injected cache changed the score by {delta:.4f} nats over "
        f"{standalone.n_tokens} tokens (standalone={standalone.total_logprob:.4f}, "
        f"injected={injected.total_logprob:.4f})"
    )


def test_oracle_mapper_is_exact(model, ids):
    """GATE 3: an oracle mapper reproduces standalone scores exactly.

    Same guarantee as gate 1, but routed through the Mapper interface, so it
    also covers the plumbing that real mappers will use.
    """
    full_ids, n_context = ids
    n_cached = n_context - 1

    standalone = score_continuation(model, full_ids, n_context, cache=None)

    truth = cache_to_content(prefill(model, full_ids[:, :n_cached]), model)
    # Feed the oracle deliberately corrupted input; it must ignore it entirely.
    garbage = type(truth)(
        keys=torch.randn_like(truth.keys),
        values=torch.randn_like(truth.values),
        position_ids=truth.position_ids,
    )
    mapped = OracleMapper(truth).map(garbage)

    oracle = score_continuation(
        model,
        full_ids,
        n_context,
        cache=content_to_cache(mapped, model, dtype=torch.float32),
        n_cached=n_cached,
    )

    assert abs(oracle.total_logprob - standalone.total_logprob) < 1e-2


def test_zero_mapper_is_clearly_worse(model, ids):
    """The scoring metric must be sensitive enough to detect a destroyed cache.

    Establishes the floor that retention is measured against, and guards
    against a harness that would score well no matter what is injected.
    """
    full_ids, n_context = ids
    n_cached = n_context - 1
    geom = load_geometry(MODEL_ID)

    standalone = score_continuation(model, full_ids, n_context, cache=None)

    source = cache_to_content(prefill(model, full_ids[:, :n_cached]), model)
    destroyed = ZeroMapper(geom).map(source)
    zeroed = score_continuation(
        model,
        full_ids,
        n_context,
        cache=content_to_cache(destroyed, model, dtype=torch.float32),
        n_cached=n_cached,
    )

    assert zeroed.total_logprob < standalone.total_logprob - 1.0, (
        "zeroing the cache barely changed the score, so the harness is probably "
        f"not using it (standalone={standalone.total_logprob:.3f}, "
        f"zeroed={zeroed.total_logprob:.3f})"
    )


@pytest.mark.parametrize("offset", [0, 1, 137, 4096])
def test_content_mapping_generalizes_across_positions(model, ids, offset):
    """GATE 4: content-space KV can be re-rotated to positions never calibrated on.

    Strip RoPE at the original positions, re-apply at shifted positions, and
    confirm the result equals a genuine prefill at those shifted positions.
    This is the property that lets a mapper fitted on 1k contexts serve longer
    ones, and it is what content-space mapping exists to provide.
    """
    full_ids, n_context = ids
    context_ids = full_ids[:, : n_context - 1]
    seq = context_ids.shape[1]

    content = cache_to_content(prefill(model, context_ids), model)

    shifted_positions = (
        torch.arange(offset, offset + seq, device=content.keys.device).unsqueeze(0)
    )
    relocated = content_to_cache(
        content, model, dtype=torch.float32, position_ids=shifted_positions
    )
    relocated_keys, _ = stack_cache(relocated)

    reference = prefill_at_positions(model, context_ids, shifted_positions)
    reference_keys, _ = stack_cache(reference)

    err = (relocated_keys - reference_keys).abs().max().item()
    scale = reference_keys.abs().max().item()
    assert err / scale < 1e-3, (
        f"relocating a cache to offset {offset} diverged from a real prefill "
        f"there: max abs err {err:.3e} (scale {scale:.3e})"
    )


@torch.no_grad()
def prefill_at_positions(model, input_ids, position_ids):
    """Prefill with explicit absolute positions, for the relocation gate."""
    device = next(model.parameters()).device
    out = model(
        input_ids=input_ids.to(device),
        position_ids=position_ids.to(device),
        use_cache=True,
    )
    return out.past_key_values


@pytest.mark.parametrize("batch_size", [1, 3])
def test_layered_roundtrip_is_batch_correct(model, tokenizer, batch_size):
    """RoPE stripping must align batch and sequence axes at any batch size.

    Layer-stacked KV is (n_layers, batch, n_kv_heads, seq, head_dim). Using the
    4-D (batch, heads, seq, dim) broadcasting convention on it happens to work
    at batch size 1 and silently misaligns batch against heads above it, so
    this is checked at more than one batch size on purpose.
    """
    prompts = [
        "The capital of France is Paris, a city on the river",
        "In 1969 the Apollo 11 mission landed the first humans on the",
        "Photosynthesis converts light energy into chemical energy stored in",
    ][:batch_size]
    batch = tokenizer(prompts, return_tensors="pt", padding=True).input_ids

    cache = prefill(model, batch)
    original_keys, _ = stack_cache(cache)

    content = cache_to_content(cache, model)
    rebuilt, _ = stack_cache(content_to_cache(content, model, dtype=torch.float32))

    assert rebuilt.shape == original_keys.shape
    err = (rebuilt - original_keys).abs().max().item()
    assert err < 1e-4, f"batch={batch_size} round trip drifted by {err:.3e}"


def test_scoring_mutates_a_raw_cache(model, ids):
    """Document the hazard that CacheTemplate exists to avoid.

    Attention appends to whatever cache it is handed, even under
    use_cache=False, so a cache is longer after being scored against and reusing
    it returns a different -- wrong -- answer instead of failing loudly.
    """
    full_ids, n_context = ids
    n_cached = n_context - 1
    content = cache_to_content(prefill(model, full_ids[:, :n_cached]), model)
    cache = content_to_cache(content, model, dtype=torch.float32)

    assert cache.get_seq_length() == n_cached
    first = score_continuation(model, full_ids, n_context, cache=cache, n_cached=n_cached)
    assert cache.get_seq_length() > n_cached, "expected the cache to have grown"

    second = score_continuation(model, full_ids, n_context, cache=cache, n_cached=n_cached)
    assert abs(first.total_logprob - second.total_logprob) > 1e-3, (
        "reusing a raw cache should give a different (wrong) answer; if this "
        "now matches, transformers changed its behaviour and CacheTemplate's "
        "cloning may no longer be necessary"
    )


def test_cache_template_is_reusable(model, ids):
    """Repeated scoring through a template must be exactly repeatable.

    This is what makes the multiple-choice protocol sound: one prefill, one
    mapping, many continuations scored against them.
    """
    from kvxfer.cache import CacheTemplate

    full_ids, n_context = ids
    n_cached = n_context - 1
    content = cache_to_content(prefill(model, full_ids[:, :n_cached]), model)
    template = CacheTemplate.from_content(content, model, dtype=torch.float32)

    scores = [
        score_continuation(
            model, full_ids, n_context, cache=template.build(), n_cached=n_cached
        ).total_logprob
        for _ in range(3)
    ]

    assert max(scores) - min(scores) < 1e-6, f"template not reusable: {scores}"

    standalone = score_continuation(model, full_ids, n_context, cache=None)
    assert abs(scores[0] - standalone.total_logprob) < 1e-2


def test_injection_gate_passes_on_a_supported_model(model, tokenizer):
    """The per-pair gate must agree with the unit-test version of gate 1.

    The gate exists because this check was only ever run in tests, against one
    family. A Mistral pair was calibrated, trained and evaluated on rented
    hardware before anyone noticed its target model scored at chance through
    its own cache -- it uses sliding-window attention, which this cache path
    does not implement.
    """
    from kvxfer.eval.gates import check_injection

    delta = check_injection(model, tokenizer, dtype=torch.float32)
    assert delta < 1e-2, f"gate reported {delta:.4f} nats on a supported model"


def test_injection_gate_rejects_a_model_that_reads_caches_differently(model, tokenizer):
    """A model that ignores its injected cache must fail the gate loudly.

    Simulated by a wrapper that drops the cache, which is the observable
    behaviour of an unsupported attention layout: the scores simply stop
    matching, with nothing raising on its own.
    """
    from kvxfer.eval.gates import InjectionGateError, check_injection

    class IgnoresCache(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.config = inner.config

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

        def forward(self, *args, **kwargs):
            kwargs.pop("past_key_values", None)
            return self.inner(*args, **kwargs)

    with pytest.raises(InjectionGateError, match="differently through its own"):
        check_injection(IgnoresCache(model), tokenizer, dtype=torch.float32)
