"""Scoring continuations against a KV cache, with or without cache transfer.

The protocol needs one non-obvious detail. A cache covering context tokens
``[0, n)`` lets the model predict token ``n``, but the logit that does so is
produced *by the prefill itself*. In a transfer setting that prefill was run by
the source model, so its logits are unusable -- we need the target model's own
logit at that position.

The fix is to hold back the final context token: the mapped cache covers
``[0, n-1)`` and the target model forwards ``[n-1, end)``. That single real
token is what the target uses to enter the mapped cache, and it yields target
logits at every position needed to score the continuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import DynamicCache


@dataclass
class ScoredContinuation:
    """Log-probability of a continuation given a context."""

    total_logprob: float
    n_tokens: int

    @property
    def mean_logprob(self) -> float:
        """Length-normalized score, for comparing choices of unequal length."""
        return self.total_logprob / max(self.n_tokens, 1)


@torch.no_grad()
def score_continuation(
    model: torch.nn.Module,
    full_ids: Tensor,
    n_context: int,
    cache: DynamicCache | None = None,
    n_cached: int | None = None,
) -> ScoredContinuation:
    """Sum log P(continuation | context) under ``model``.

    Args:
        model: the model producing the logits (the *target* model in transfer).
        full_ids: ``(1, seq)`` context tokens followed by continuation tokens.
        n_context: number of leading tokens that count as context; everything
            from this index onward is scored.
        cache: a prepopulated cache covering ``full_ids[:, :n_cached]``. When
            None, the model processes the whole sequence itself.
        n_cached: how many leading tokens ``cache`` covers. Defaults to
            ``n_context - 1``, the largest prefix that still leaves the target
            model producing the logit for the first scored token.

    Returns:
        The summed log-probability and the number of tokens scored.
    """
    if full_ids.dim() != 2 or full_ids.shape[0] != 1:
        raise ValueError(f"expected (1, seq) input_ids, got {tuple(full_ids.shape)}")
    total_len = full_ids.shape[1]
    if n_context >= total_len:
        raise ValueError(
            f"no continuation to score: n_context={n_context} but sequence is {total_len}"
        )

    device = next(model.parameters()).device
    full_ids = full_ids.to(device)

    if cache is None:
        n_cached = 0
    elif n_cached is None:
        n_cached = n_context - 1

    if n_cached > n_context - 1:
        raise ValueError(
            f"cache covers {n_cached} tokens but scoring starts at {n_context}; "
            f"the target model must forward the token at index {n_context - 1} "
            "itself to produce the logit for the first scored token"
        )

    forward_ids = full_ids[:, n_cached:]
    positions = torch.arange(n_cached, total_len, device=device).unsqueeze(0)

    out = model(
        input_ids=forward_ids,
        position_ids=positions,
        past_key_values=cache,
        use_cache=False,
    )
    logits = out.logits.float()

    # logits[:, j] predicts the token at absolute index n_cached + j + 1.
    # We want absolute indices [n_context, total_len), i.e. j in
    # [n_context - n_cached - 1, total_len - n_cached - 1).
    start = n_context - n_cached - 1
    pred = logits[:, start : total_len - n_cached - 1, :]
    targets = full_ids[:, n_context:total_len]

    logprobs = F.log_softmax(pred, dim=-1)
    picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return ScoredContinuation(
        total_logprob=picked.sum().item(), n_tokens=int(targets.numel())
    )
