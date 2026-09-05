"""Calibration and evaluation corpora.

The reference work calibrates on a single web corpus and reports a measurable
accuracy drop when the mapper meets code instead. Domain mixture is therefore a
first-class option here rather than an afterthought: ``build_calibration`` takes
a mixture spec so that calibrate-on-X / evaluate-on-Y is a configuration change,
not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

# Streaming avoids materializing corpora we only need a few hundred documents of.
DOMAINS: dict[str, dict] = {
    "web": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "text_key": "text",
    },
    "code": {
        "path": "bigcode/the-stack-smol",
        "name": "data/python",
        "split": "train",
        "text_key": "content",
    },
    "math": {
        "path": "open-r1/OpenR1-Math-220k",
        "name": "default",
        "split": "train",
        "text_key": "problem",
    },
}


@dataclass
class CalibrationSet:
    """Tokenized fixed-length sequences used to fit mappers.

    Attributes:
        input_ids: ``(n_sequences, seq_len)``.
        domains: the domain each sequence came from, parallel to ``input_ids``.
        seq_len: sequence length every row was truncated or packed to.
    """

    input_ids: Tensor
    domains: list[str] = field(default_factory=list)

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def batches(self, batch_size: int):
        """Yield ``(batch, seq_len)`` chunks."""
        for start in range(0, len(self), batch_size):
            yield self.input_ids[start : start + batch_size]


def build_calibration(
    tokenizer,
    seq_len: int = 1024,
    n_sequences: int = 256,
    mixture: dict[str, float] | None = None,
    seed: int = 0,
    skip: int = 0,
) -> CalibrationSet:
    """Tokenize a fixed-length calibration set from one or more domains.

    Args:
        tokenizer: tokenizer of the model family. Source and target must share
            one, so that token boundaries -- and therefore cache positions --
            line up exactly between the two models.
        seq_len: tokens per sequence.
        n_sequences: total sequences across all domains.
        mixture: domain name -> proportion. Defaults to web only, matching the
            reference setup; pass e.g. ``{"web": .5, "code": .3, "math": .2}``
            to test domain robustness.
        seed: shuffles the assembled set.
        skip: discard this many sequences per domain before collecting. Use it to
            carve evaluation documents that are disjoint from the calibration
            corpus -- scoring a mapper on the text it was fitted on would
            measure memorization rather than transfer.

    Returns:
        A :class:`CalibrationSet` of exactly ``n_sequences`` rows, each a full
        ``seq_len`` tokens (documents are packed and split, never padded, so no
        row contains padding that would pollute the Gram).
    """
    from datasets import load_dataset

    mixture = mixture or {"web": 1.0}
    total = sum(mixture.values())

    rows: list[Tensor] = []
    labels: list[str] = []

    for domain, weight in mixture.items():
        if domain not in DOMAINS:
            raise KeyError(f"unknown domain {domain!r}; known: {sorted(DOMAINS)}")
        spec = DOMAINS[domain]
        want = int(round(n_sequences * weight / total))
        if want == 0:
            continue

        stream = load_dataset(
            spec["path"], name=spec["name"], split=spec["split"], streaming=True
        )

        buffer: list[int] = []
        produced = 0
        skipped = 0
        for record in stream:
            text = record.get(spec["text_key"]) or ""
            if not text.strip():
                continue
            buffer.extend(tokenizer(text, add_special_tokens=False).input_ids)
            buffer.append(tokenizer.eos_token_id)

            while len(buffer) >= seq_len and produced < want:
                chunk = buffer[:seq_len]
                buffer = buffer[seq_len:]
                if skipped < skip:
                    skipped += 1
                    continue
                rows.append(torch.tensor(chunk, dtype=torch.long))
                labels.append(domain)
                produced += 1
            if produced >= want:
                break

        if produced < want:
            raise RuntimeError(
                f"domain {domain!r} yielded only {produced} of {want} sequences"
            )

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(rows), generator=generator)
    return CalibrationSet(
        input_ids=torch.stack([rows[i] for i in order.tolist()]),
        domains=[labels[i] for i in order.tolist()],
    )
