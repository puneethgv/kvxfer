"""Prefix-conditioned perplexity: the high-resolution way to compare mappers.

Multiple-choice accuracy is easy to read but statistically blunt. A few hundred
items resolve differences of a couple of items, which is far coarser than the
effects a mapper produces. The same forward passes over the same text yield one
measurement per token instead, so a comparison that accuracy cannot resolve
becomes unambiguous.

The protocol mirrors the deployment case. A document is split at ``n_prefix``:
the source model prefills the prefix, the mapper converts that cache into the
target's format, and the target model is scored on the remainder. The target
model prefilling the same prefix itself is the ceiling; a zeroed cache is the
floor.

Statistics are computed per document, not per token. Tokens within a document
are strongly dependent, so treating them as independent samples would understate
the standard error by roughly the square root of the document length -- turning
noise into apparent significance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from kvxfer.cache import CacheTemplate, cache_to_content, prefill
from kvxfer.eval.scoring import score_tokens_batch
from kvxfer.mappers import Mapper


@dataclass
class PerplexityResult:
    """Perplexity of one condition, with per-document detail for paired tests."""

    name: str
    total_nll: float
    n_tokens: int
    document_nll: list[float] = field(default_factory=list)

    @property
    def mean_nll(self) -> float:
        """Mean negative log-likelihood per token, in nats."""
        return self.total_nll / max(self.n_tokens, 1)

    @property
    def perplexity(self) -> float:
        return math.exp(self.mean_nll)

    def stderr(self) -> float:
        """Standard error of mean NLL, computed across documents."""
        n = len(self.document_nll)
        if n < 2:
            return float("nan")
        mean = sum(self.document_nll) / n
        variance = sum((v - mean) ** 2 for v in self.document_nll) / (n - 1)
        return math.sqrt(variance / n)


@dataclass
class PairedNLL:
    """Paired comparison of two conditions over the same documents."""

    a: str
    b: str
    mean_difference: float
    stderr: float
    n_documents: int
    n_better: int

    @property
    def t_statistic(self) -> float:
        return self.mean_difference / self.stderr if self.stderr > 0 else float("nan")

    def __str__(self) -> str:
        direction = "better" if self.mean_difference < 0 else "worse"
        return (
            f"{self.b} vs {self.a}: dNLL={self.mean_difference:+.5f} "
            f"+/- {self.stderr:.5f} nats/token ({direction}), "
            f"t={self.t_statistic:+.2f}, "
            f"{self.n_better}/{self.n_documents} documents improved"
        )


def paired_nll(a: PerplexityResult, b: PerplexityResult) -> PairedNLL:
    """Compare two conditions document by document.

    Both conditions see identical documents, so the difference is taken within
    each document before averaging. That removes document difficulty -- by far
    the largest source of variance -- from the comparison.
    """
    if len(a.document_nll) != len(b.document_nll):
        raise ValueError(
            f"document counts differ: {len(a.document_nll)} vs {len(b.document_nll)}"
        )
    if not a.document_nll:
        raise ValueError("no per-document values recorded")

    deltas = [y - x for x, y in zip(a.document_nll, b.document_nll)]
    n = len(deltas)
    mean = sum(deltas) / n
    if n < 2:
        stderr = float("nan")
    else:
        variance = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        stderr = math.sqrt(variance / n)

    return PairedNLL(
        a=a.name,
        b=b.name,
        mean_difference=mean,
        stderr=stderr,
        n_documents=n,
        n_better=sum(1 for d in deltas if d < 0),
    )


@torch.no_grad()
def evaluate_perplexity(
    documents: Tensor,
    n_prefix: int,
    target_model: torch.nn.Module,
    source_model: torch.nn.Module | None = None,
    mappers: dict[str, Mapper] | None = None,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 4,
    score_source_baseline: bool = True,
    progress_every: int = 8,
) -> dict[str, PerplexityResult]:
    """Score every condition on the continuation of each document.

    Args:
        documents: ``(n_documents, seq_len)`` fixed-length token sequences.
            Uniform length is what allows batching, and these must be disjoint
            from the calibration corpus.
        n_prefix: split point. Tokens from here on are scored; the cache covers
            ``n_prefix - 1`` tokens so the target model produces the logit for
            the first scored token itself.
        target_model: the model whose quality is being preserved.
        source_model: the model whose prefill is transferred.
        mappers: condition name -> mapper.
        dtype: cache dtype.
        batch_size: documents per forward pass.
        score_source_baseline: also score the source model alone.
        progress_every: print progress every n batches; 0 disables.

    Returns:
        Condition name -> :class:`PerplexityResult`.
    """
    mappers = mappers or {}
    if mappers and source_model is None:
        raise ValueError("source_model is required when mappers are given")

    n_docs, seq_len = documents.shape
    if not 1 < n_prefix < seq_len:
        raise ValueError(f"n_prefix must lie in (1, {seq_len}), got {n_prefix}")

    names = ["target", *mappers]
    if source_model is not None and score_source_baseline:
        names.append("source")
    totals = {name: [0.0, 0] for name in names}
    per_document = {name: [] for name in names}

    n_cached = n_prefix - 1

    def record(name: str, logprobs: Tensor) -> None:
        nll = -logprobs
        totals[name][0] += float(nll.sum())
        totals[name][1] += int(nll.numel())
        per_document[name].extend(nll.mean(dim=1).tolist())

    for start in range(0, n_docs, batch_size):
        batch = documents[start : start + batch_size]

        record("target", score_tokens_batch(target_model, batch, n_prefix))
        if "source" in totals:
            record("source", score_tokens_batch(source_model, batch, n_prefix))

        if mappers:
            source_content = cache_to_content(
                prefill(source_model, batch[:, :n_cached]), source_model
            )
            for name, mapper in mappers.items():
                template = CacheTemplate.from_content(
                    mapper.map(source_content), target_model, dtype=dtype
                )
                record(
                    name,
                    score_tokens_batch(
                        target_model,
                        batch,
                        n_prefix,
                        cache=template.build(),
                        n_cached=n_cached,
                    ),
                )
            del source_content

        if progress_every and (start // batch_size + 1) % progress_every == 0:
            print(
                f"    {min(start + batch_size, n_docs)}/{n_docs} documents",
                flush=True,
            )

    return {
        name: PerplexityResult(
            name=name,
            total_nll=total,
            n_tokens=count,
            document_nll=per_document[name],
        )
        for name, (total, count) in totals.items()
    }
