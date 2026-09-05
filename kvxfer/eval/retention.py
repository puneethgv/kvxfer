"""Measuring how much accuracy survives a KV cache transfer.

Every condition is scored on the same items, so comparisons are paired: the
difference between two mappers is measured on identical questions rather than
on independent samples, which removes most of the noise that would otherwise
swamp a few-hundred-item benchmark.

Retention is reported two ways. The plain ratio to the target model's own
accuracy is what the reference work reports, and is what makes numbers
comparable to it. But it is misleading near chance -- a mapper that destroys
the cache entirely still "retains" 25% on a 4-way task. The floor-normalized
figure places a destroyed cache at 0 and the intact target at 100, which is the
scale that answers whether a mapper transferred anything at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from kvxfer.cache import CacheTemplate, cache_to_content, prefill
from kvxfer.eval.scoring import score_continuation
from kvxfer.eval.tasks import Example
from kvxfer.mappers import Mapper


@dataclass
class ConditionResult:
    """Accuracy of one condition on one task.

    Per-item outcomes are kept, not just totals. Conditions are scored on
    identical items, so the meaningful comparison between two mappers is
    paired, and a paired test needs to know *which* items each got right --
    information that aggregate counts throw away. With a few hundred items the
    difference matters: an aggregate standard error of a few points can hide an
    effect that is unambiguous once the shared items are accounted for.
    """

    name: str
    n_items: int
    n_correct: int
    n_correct_normalized: int
    outcomes: list[bool] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.n_correct / max(self.n_items, 1)

    @property
    def accuracy_normalized(self) -> float:
        """Accuracy using length-normalized choice scores."""
        return self.n_correct_normalized / max(self.n_items, 1)

    def stderr(self) -> float:
        """Binomial standard error, for judging whether gaps are real."""
        p = self.accuracy
        return math.sqrt(max(p * (1 - p), 0.0) / max(self.n_items, 1))


@dataclass
class PairedComparison:
    """McNemar comparison of two conditions scored on the same items."""

    a: str
    b: str
    a_only: int
    b_only: int
    difference: float
    p_value: float

    @property
    def n_discordant(self) -> int:
        return self.a_only + self.b_only

    def __str__(self) -> str:
        return (
            f"{self.b} - {self.a} = {self.difference:+.4f}  "
            f"({self.b_only} vs {self.a_only} of {self.n_discordant} disagreements, "
            f"p={self.p_value:.4f})"
        )


@dataclass
class RetentionResult:
    """All conditions for one task, plus the derived retention figures."""

    task: str
    conditions: dict[str, ConditionResult] = field(default_factory=dict)

    def paired_test(self, a: str, b: str) -> PairedComparison:
        """Exact McNemar test between two conditions on the same items.

        Only items the two conditions disagree on carry information; items both
        get right, or both get wrong, say nothing about which is better. Under
        the null those disagreements split evenly, so the exact two-sided
        binomial test on that split is the whole test -- and it is exact rather
        than asymptotic, which matters because the discordant count is often
        small even when the item count is not.
        """
        left = self.conditions[a].outcomes
        right = self.conditions[b].outcomes
        if not left or not right:
            raise ValueError(
                "per-item outcomes are unavailable; these results predate "
                "outcome recording and cannot support a paired test"
            )
        if len(left) != len(right):
            raise ValueError(f"condition lengths differ: {len(left)} vs {len(right)}")

        a_only = sum(1 for x, y in zip(left, right) if x and not y)
        b_only = sum(1 for x, y in zip(left, right) if y and not x)
        n = a_only + b_only

        if n == 0:
            p_value = 1.0
        else:
            # Two-sided exact binomial: P(|X - n/2| >= |observed - n/2|).
            extreme = max(a_only, b_only)
            tail = sum(math.comb(n, i) for i in range(extreme, n + 1)) / 2**n
            p_value = min(1.0, 2.0 * tail)

        return PairedComparison(
            a=a,
            b=b,
            a_only=a_only,
            b_only=b_only,
            difference=(b_only - a_only) / max(len(left), 1),
            p_value=p_value,
        )

    def retention(self, condition: str, reference: str = "target") -> float:
        """Transferred accuracy as a fraction of the target's own."""
        ref = self.conditions[reference].accuracy
        return self.conditions[condition].accuracy / ref if ref > 0 else float("nan")

    def floor_normalized_retention(
        self, condition: str, reference: str = "target", floor: str = "floor"
    ) -> float:
        """Retention rescaled so a destroyed cache is 0 and the target is 1."""
        lo = self.conditions[floor].accuracy
        hi = self.conditions[reference].accuracy
        if hi - lo <= 0:
            return float("nan")
        return (self.conditions[condition].accuracy - lo) / (hi - lo)


@torch.no_grad()
def _score_choices(
    target_model,
    tokenizer,
    example: Example,
    template: CacheTemplate | None,
    n_cached: int,
) -> tuple[int, int]:
    """Return the argmax choice under summed and length-normalized scores."""
    context_ids = tokenizer(example.context, return_tensors="pt").input_ids
    totals: list[float] = []
    means: list[float] = []

    for choice in example.choices:
        full = tokenizer(example.context + choice, return_tensors="pt").input_ids
        n_context = context_ids.shape[1]
        if full.shape[1] <= n_context:
            totals.append(-math.inf)
            means.append(-math.inf)
            continue

        scored = score_continuation(
            target_model,
            full,
            n_context,
            cache=template.build() if template is not None else None,
            n_cached=n_cached if template is not None else None,
        )
        totals.append(scored.total_logprob)
        means.append(scored.mean_logprob)

    return int(max(range(len(totals)), key=totals.__getitem__)), int(
        max(range(len(means)), key=means.__getitem__)
    )


@torch.no_grad()
def evaluate_task(
    task_name: str,
    examples: list[Example],
    target_model,
    tokenizer,
    source_model=None,
    mappers: dict[str, Mapper] | None = None,
    dtype: torch.dtype = torch.float32,
    progress_every: int = 50,
    score_source_baseline: bool = True,
) -> RetentionResult:
    """Score a task under the target model and each supplied mapper.

    Args:
        task_name: label for the result.
        examples: items to score. Every condition sees exactly these.
        target_model: the model whose accuracy is being preserved.
        tokenizer: shared by both models.
        source_model: model whose prefill is transferred. Required if
            ``mappers`` is non-empty.
        mappers: condition name -> mapper. Typically includes a ``floor``
            entry using a zero mapper, which anchors the normalized scale.
        dtype: cache dtype for injection.
        progress_every: print progress every n items; 0 disables.
        score_source_baseline: also score the source model standalone. This is
            the decision-relevant comparison -- transfer is only worth doing if
            it beats running the smaller model on its own.

    Returns:
        A :class:`RetentionResult` holding every condition's accuracy.
    """
    mappers = mappers or {}
    if mappers and source_model is None:
        raise ValueError("source_model is required when mappers are given")

    names = ["target", *mappers]
    if source_model is not None and score_source_baseline:
        names.append("source")
    counters = {name: [0, 0] for name in names}
    outcomes: dict[str, list[bool]] = {name: [] for name in names}

    for index, example in enumerate(examples):
        correct, correct_norm = _score_choices(
            target_model, tokenizer, example, template=None, n_cached=0
        )
        counters["target"][0] += int(correct == example.answer)
        counters["target"][1] += int(correct_norm == example.answer)
        outcomes["target"].append(correct == example.answer)

        if "source" in counters:
            # The practical decision baseline: if a transferred cache does not
            # beat simply running the small model, transfer buys nothing.
            hit, hit_norm = _score_choices(
                source_model, tokenizer, example, template=None, n_cached=0
            )
            counters["source"][0] += int(hit == example.answer)
            counters["source"][1] += int(hit_norm == example.answer)
            outcomes["source"].append(hit == example.answer)

        if mappers:
            context_ids = tokenizer(example.context, return_tensors="pt").input_ids
            n_cached = context_ids.shape[1] - 1
            if n_cached < 1:
                # Nothing to transfer; every mapper trivially matches the target.
                for name in mappers:
                    counters[name][0] += int(correct == example.answer)
                    counters[name][1] += int(correct_norm == example.answer)
                    outcomes[name].append(correct == example.answer)
                continue

            source_content = cache_to_content(
                prefill(source_model, context_ids[:, :n_cached]), source_model
            )
            for name, mapper in mappers.items():
                template = CacheTemplate.from_content(
                    mapper.map(source_content), target_model, dtype=dtype
                )
                hit, hit_norm = _score_choices(
                    target_model, tokenizer, example, template, n_cached
                )
                counters[name][0] += int(hit == example.answer)
                counters[name][1] += int(hit_norm == example.answer)
                outcomes[name].append(hit == example.answer)

        if progress_every and (index + 1) % progress_every == 0:
            print(f"    {index + 1}/{len(examples)} items", flush=True)

    return RetentionResult(
        task=task_name,
        conditions={
            name: ConditionResult(
                name, len(examples), hits, hits_norm, outcomes[name]
            )
            for name, (hits, hits_norm) in counters.items()
        },
    )
