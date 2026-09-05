"""The paired test must reflect disagreements, not aggregate accuracy.

Two mappers scored on identical items can differ by a fraction of a point in
aggregate while disagreeing on very few items -- and whether that difference is
real depends entirely on how those disagreements split. These tests pin the
behaviour that separates the two readings.
"""

from __future__ import annotations

import pytest

from kvxfer.eval.retention import ConditionResult, RetentionResult


def _result(**conditions: list[bool]) -> RetentionResult:
    return RetentionResult(
        task="synthetic",
        conditions={
            name: ConditionResult(
                name=name,
                n_items=len(outcomes),
                n_correct=sum(outcomes),
                n_correct_normalized=sum(outcomes),
                outcomes=outcomes,
            )
            for name, outcomes in conditions.items()
        },
    )


def test_identical_conditions_are_indistinguishable():
    """No disagreements means no evidence, whatever the accuracy."""
    outcomes = [True, False, True, True, False] * 20
    comparison = _result(a=outcomes, b=list(outcomes)).paired_test("a", "b")

    assert comparison.n_discordant == 0
    assert comparison.difference == 0.0
    assert comparison.p_value == 1.0


def test_one_sided_disagreement_is_significant():
    """When every disagreement favours one side, the result is unambiguous.

    Twelve straight disagreements in one direction is p < 0.001 even though the
    aggregate accuracy gap is only 6 points on 200 items.
    """
    a = [True] * 100 + [False] * 100
    b = list(a)
    for i in range(100, 112):  # b rescues 12 items a got wrong
        b[i] = True

    comparison = _result(a=a, b=b).paired_test("a", "b")

    assert comparison.a_only == 0
    assert comparison.b_only == 12
    assert comparison.difference == pytest.approx(12 / 200)
    assert comparison.p_value < 0.001


def test_balanced_disagreement_is_not_significant():
    """Equal disagreements in both directions is exactly the null."""
    a = [True] * 100 + [False] * 100
    b = list(a)
    for i in range(0, 10):
        b[i] = False        # b loses 10
    for i in range(100, 110):
        b[i] = True         # b gains 10

    comparison = _result(a=a, b=b).paired_test("a", "b")

    assert comparison.a_only == 10
    assert comparison.b_only == 10
    assert comparison.difference == 0.0
    assert comparison.p_value == pytest.approx(1.0)


def test_small_aggregate_gap_can_still_be_significant():
    """The point of pairing: a tiny accuracy gap with lopsided disagreements.

    Here accuracy moves by 2pp -- far inside any per-condition standard error on
    250 items -- but every one of the disagreements favours b, so the paired
    test rejects while an unpaired comparison could not.
    """
    a = [True] * 180 + [False] * 70
    b = list(a)
    for i in range(180, 185):
        b[i] = True

    comparison = _result(a=a, b=b).paired_test("a", "b")
    unpaired_gap = (sum(b) - sum(a)) / 250

    assert unpaired_gap == pytest.approx(0.02)
    assert comparison.p_value < 0.07, "five one-sided disagreements should be suggestive"
    assert comparison.b_only == 5 and comparison.a_only == 0


def test_missing_outcomes_are_rejected():
    """Results without per-item records must refuse rather than mislead."""
    stale = RetentionResult(
        task="old",
        conditions={
            "a": ConditionResult("a", 100, 70, 70),
            "b": ConditionResult("b", 100, 72, 72),
        },
    )
    with pytest.raises(ValueError, match="per-item outcomes are unavailable"):
        stale.paired_test("a", "b")
