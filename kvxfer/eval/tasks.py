"""Multiple-choice benchmarks, shaped for prefix-conditioned cache transfer.

Each item is a shared context plus several candidate continuations. That shape
is what makes transfer measurable the way it would actually be used: the
context is prefilled once by the source model, mapped once, and every candidate
is then scored by the target model against that single mapped cache.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Example:
    """One multiple-choice item.

    Attributes:
        context: text prefilled by the source model and shared by all choices.
        choices: candidate continuations, scored by the target model.
        answer: index of the correct choice.
    """

    context: str
    choices: list[str]
    answer: int


def _arc(record: dict) -> Example | None:
    labels = record["choices"]["label"]
    texts = record["choices"]["text"]
    if record["answerKey"] not in labels:
        return None
    return Example(
        context=f"Question: {record['question']}\nAnswer:",
        choices=[f" {t}" for t in texts],
        answer=labels.index(record["answerKey"]),
    )


def _hellaswag(record: dict) -> Example | None:
    if not str(record.get("label", "")).isdigit():
        return None
    context = f"{record['activity_label']}: {record['ctx']}"
    return Example(
        context=context,
        choices=[f" {ending}" for ending in record["endings"]],
        answer=int(record["label"]),
    )


def _piqa(record: dict) -> Example | None:
    if record.get("label") not in (0, 1):
        return None
    return Example(
        context=f"Question: {record['goal']}\nAnswer:",
        choices=[f" {record['sol1']}", f" {record['sol2']}"],
        answer=int(record["label"]),
    )


TASKS: dict[str, dict] = {
    "arc_easy": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Easy",
        "split": "test",
        "adapter": _arc,
    },
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "test",
        "adapter": _arc,
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "name": None,
        "split": "validation",
        "adapter": _hellaswag,
    },
    "piqa": {
        "path": "ybisk/piqa",
        "name": None,
        "split": "validation",
        "adapter": _piqa,
    },
}


def load_task(name: str, limit: int = 500, seed: int = 0) -> list[Example]:
    """Load a fixed, shuffled subset of a benchmark.

    A fixed subset keeps every condition scored on identical items, so the
    comparison between mappers is paired and the sampling noise is shared
    rather than independent across conditions.

    Args:
        name: key into :data:`TASKS`.
        limit: how many items to keep.
        seed: shuffle seed; the same seed always yields the same subset.

    Returns:
        Parsed examples, at most ``limit`` of them.
    """
    from datasets import load_dataset

    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    spec = TASKS[name]

    dataset = load_dataset(
        spec["path"],
        name=spec["name"],
        split=spec["split"],
        trust_remote_code=False,
    )
    dataset = dataset.shuffle(seed=seed)

    adapter = spec["adapter"]
    examples: list[Example] = []
    for record in dataset:
        parsed = adapter(record)
        if parsed is not None and len(parsed.choices) >= 2:
            examples.append(parsed)
        if len(examples) >= limit:
            break
    return examples
