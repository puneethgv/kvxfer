"""Correctness gates that run against the models actually being measured.

The identity-injection gate lives in the test suite, where it runs against one
family. That is not where it is needed. A Mistral pair was calibrated,
evaluated and trained on an A100 before anyone noticed the target model was
scoring at chance through its own cache, because Ministral-8B uses interleaved
sliding-window attention that this cache path does not implement. The gate
would have caught it in seconds.

So the gate runs per pair, before anything expensive, against the specific
models in front of it.
"""

from __future__ import annotations

import torch

from kvxfer.cache import cache_to_content, content_to_cache, prefill
from kvxfer.eval.scoring import score_continuation
from kvxfer.mappers import IdentityMapper


class InjectionGateError(RuntimeError):
    """The model does not score the same through its own injected cache."""


def check_injection(
    model,
    tokenizer,
    text: str = (
        "The history of scientific instruments is in large part a history of "
        "measurement error, and of the slow work of telling one source of it "
        "from another. Early astronomers knew their observations disagreed."
    ),
    n_context: int | None = None,
    tolerance: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Verify that injecting a model's own cache reproduces standalone scoring.

    Every retention number depends on this holding. If a model reads an
    injected cache differently from one it built itself -- because it uses
    sliding-window attention, or a cache layout this code does not implement --
    then the ``target`` condition is wrong, and every retention ratio computed
    against it is meaningless rather than merely noisy.

    Args:
        model: the model to check, usually the transfer target.
        tokenizer: its tokenizer.
        text: probe text; anything long enough to split into context and
            continuation will do.
        n_context: context length in tokens; defaults to half the probe.
        tolerance: allowed difference in nats per scored token. Defaults to a
            value appropriate to ``dtype``, which matters more than it sounds:
            a cache round trip in bfloat16 costs about 0.006 nats per token on
            a model known to be supported, so a threshold calibrated in float32
            rejects everything. Doing exactly that killed a paid run on a pair
            that turned out to be fine.
        dtype: dtype for the injected cache.

    Returns:
        The measured absolute difference in nats per token.

    Raises:
        InjectionGateError: if the difference exceeds ``tolerance``.
    """
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    if ids.shape[1] < 8:
        raise ValueError("probe text is too short to split")
    n_context = n_context or ids.shape[1] // 2

    standalone = score_continuation(model, ids, n_context, cache=None)

    n_cached = n_context - 1
    cache = prefill(model, ids[:, :n_cached])
    content = IdentityMapper().map(cache_to_content(cache, model))
    injected = score_continuation(
        model,
        ids,
        n_context,
        cache=content_to_cache(content, model, dtype=dtype),
        n_cached=n_cached,
    )

    # Per token, so the threshold does not depend on probe length.
    delta = abs(injected.total_logprob - standalone.total_logprob)
    delta /= max(standalone.n_tokens, 1)
    if tolerance is None:
        # float32 round trips are exact to ~1e-6; bfloat16 measures 0.006 on a
        # supported model, and a genuine cache-layout mismatch is orders of
        # magnitude larger than either -- it drives accuracy to chance.
        tolerance = 1e-3 if dtype == torch.float32 else 5e-2
    if delta > tolerance:
        raise InjectionGateError(
            f"{getattr(model.config, 'name_or_path', type(model).__name__)} scores "
            f"differently through its own injected cache: {delta:.4f} nats per "
            f"token over {standalone.n_tokens} tokens (standalone "
            f"{standalone.total_logprob:.4f}, injected {injected.total_logprob:.4f}). "
            "Every retention figure measured against this model would be "
            "meaningless. Models using sliding-window attention are a known "
            "cause; this cache path implements full attention only."
        )
    return delta
