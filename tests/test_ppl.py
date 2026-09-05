"""Prefix-conditioned perplexity: batching, protocol, and paired statistics.

Perplexity is meant to be the high-resolution comparison, so the machinery has
to be exactly right -- a subtle indexing error would produce plausible numbers
with no visible symptom.
"""

from __future__ import annotations

import math

import pytest
import torch

from kvxfer.eval.ppl import PerplexityResult, evaluate_perplexity, paired_nll


def test_paired_nll_removes_document_difficulty():
    """Pairing must cancel per-document difficulty, not average over it.

    Documents here differ enormously in absolute NLL while b is uniformly
    better by exactly 0.1 nats. Unpaired standard errors would be huge; the
    paired one must be zero.
    """
    hard_easy = [0.5, 3.0, 1.2, 8.0, 2.2]
    a = PerplexityResult("a", sum(hard_easy), 500, hard_easy)
    b = PerplexityResult("b", sum(v - 0.1 for v in hard_easy), 500,
                         [v - 0.1 for v in hard_easy])

    comparison = paired_nll(a, b)

    assert comparison.mean_difference == pytest.approx(-0.1)
    assert comparison.stderr == pytest.approx(0.0, abs=1e-12)
    assert comparison.n_better == 5


def test_paired_nll_reports_mixed_outcomes():
    """A genuinely noisy difference must show a small t statistic."""
    a = PerplexityResult("a", 0.0, 100, [1.0, 1.0, 1.0, 1.0])
    b = PerplexityResult("b", 0.0, 100, [0.9, 1.1, 0.9, 1.1])

    comparison = paired_nll(a, b)

    assert comparison.mean_difference == pytest.approx(0.0, abs=1e-12)
    assert comparison.n_better == 2
    assert abs(comparison.t_statistic) < 1e-6


def test_perplexity_is_exp_of_mean_nll():
    result = PerplexityResult("x", total_nll=200.0, n_tokens=100)
    assert result.mean_nll == pytest.approx(2.0)
    assert result.perplexity == pytest.approx(math.exp(2.0))


def test_mismatched_document_counts_are_rejected():
    a = PerplexityResult("a", 1.0, 10, [1.0, 2.0])
    b = PerplexityResult("b", 1.0, 10, [1.0])
    with pytest.raises(ValueError, match="document counts differ"):
        paired_nll(a, b)


@pytest.fixture(scope="module")
def setup():
    """A small real model and a batch of identical fixed-length documents."""
    from kvxfer.models import load_model, load_tokenizer

    model_id = "Qwen/Qwen3-0.6B"
    model = load_model(model_id, dtype=torch.float32)
    tokenizer = load_tokenizer(model_id)
    text = (
        "The history of computing is often told through hardware, but the "
        "decisive shifts were conceptual. Stored-program architecture meant "
        "instructions and data shared one memory, which made programs into "
        "objects that other programs could manipulate. Compilers followed, "
        "and with them the idea that a language could be designed for people "
        "rather than machines. Each step traded execution efficiency for "
        "human leverage, a bargain that has almost always paid off."
    )
    ids = tokenizer(text, return_tensors="pt").input_ids
    return model, ids[:, :64].repeat(4, 1)

@pytest.mark.slow
def test_batched_scoring_matches_unbatched(setup):
    """The batched path must agree with the single-sequence path exactly."""
    from kvxfer.eval.scoring import score_continuation, score_tokens_batch

    model, docs = setup
    n_prefix = 20

    batched = score_tokens_batch(model, docs, n_prefix)
    single = score_continuation(model, docs[:1], n_prefix)

    assert batched.shape == (4, docs.shape[1] - n_prefix)
    assert float(batched[0].sum()) == pytest.approx(single.total_logprob, abs=1e-3)
    # Identical rows must score identically.
    assert torch.allclose(batched[0], batched[3], atol=1e-4)

@pytest.mark.slow
def test_oracle_mapper_reproduces_target_perplexity(setup):
    """GATE: transferring the target's own cache must change nothing.

    The perplexity analogue of the identity-injection gate, and a much more
    sensitive one -- it compares every token rather than an argmax.
    """
    from kvxfer.mappers import IdentityMapper

    model, docs = setup
    n_prefix = 20

    results = evaluate_perplexity(
        docs,
        n_prefix,
        target_model=model,
        source_model=model,
        mappers={"identity": IdentityMapper()},
        dtype=torch.float32,
        batch_size=2,
        score_source_baseline=False,
        progress_every=0,
    )

    assert results["identity"].mean_nll == pytest.approx(
        results["target"].mean_nll, abs=1e-3
    )

@pytest.mark.slow
def test_zeroed_cache_is_measurably_worse(setup):
    """The floor must be clearly worse, or the metric is not using the cache."""
    from kvxfer.geometry import load_geometry
    from kvxfer.mappers import ZeroMapper

    model, docs = setup
    geom = load_geometry("Qwen/Qwen3-0.6B")

    results = evaluate_perplexity(
        docs,
        20,
        target_model=model,
        source_model=model,
        mappers={"floor": ZeroMapper(geom)},
        dtype=torch.float32,
        batch_size=2,
        score_source_baseline=False,
        progress_every=0,
    )

    assert results["floor"].mean_nll > results["target"].mean_nll + 0.1

@pytest.mark.slow
def test_per_document_values_are_recorded(setup):
    """Paired statistics depend on these being present and correctly sized."""
    model, docs = setup
    results = evaluate_perplexity(
        docs, 20, target_model=model, dtype=torch.float32,
        batch_size=3, progress_every=0,
    )
    assert len(results["target"].document_nll) == docs.shape[0]
