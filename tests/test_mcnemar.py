"""Spec for the exact McNemar test and the paired 2x2 table."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import binomtest

from agentcheck.stats.mcnemar import PairedCounts, exact_mcnemar_p, mcnemar, paired_counts

# --- p-values worked out by hand -----------------------------------------------------


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [
        (0, 0, 1.0),  # no discordant pairs: no evidence either way
        (3, 3, 1.0),
        (1, 0, 1.0),  # 2 * 1/2 = 1
        (5, 0, 2 / 32),  # 2 * (1/2)^5
        (0, 5, 2 / 32),
        (6, 0, 2 / 64),
        (10, 2, 2 * (1 + 12 + 66) / 4096),
    ],
)
def test_hand_checked(b: int, c: int, p: float) -> None:
    assert exact_mcnemar_p(b, c) == pytest.approx(p, rel=1e-12)


def test_five_discordant_can_never_reach_five_percent() -> None:
    """Smallest possible p with 5 discordant pairs is 0.0625: a 5% test needs at least 6."""
    assert min(exact_mcnemar_p(b, 5 - b) for b in range(6)) > 0.05
    assert exact_mcnemar_p(6, 0) < 0.05


# --- against SciPy -------------------------------------------------------------------

pairs = st.tuples(st.integers(0, 400), st.integers(0, 400))


@given(pairs)
def test_matches_scipy(bc: tuple[int, int]) -> None:
    b, c = bc
    ref = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    assert exact_mcnemar_p(b, c) == pytest.approx(ref, rel=1e-9, abs=1e-300)


# --- properties ----------------------------------------------------------------------


@given(pairs)
def test_symmetric_and_in_range(bc: tuple[int, int]) -> None:
    b, c = bc
    p = exact_mcnemar_p(b, c)
    assert 0.0 < p <= 1.0
    assert p == exact_mcnemar_p(c, b)


@given(st.integers(0, 200), st.integers(1, 50))
def test_more_imbalance_is_more_significant(b: int, extra: int) -> None:
    assert exact_mcnemar_p(b + extra, b) <= exact_mcnemar_p(b, b)


def test_huge_counts_are_finite() -> None:
    p = exact_mcnemar_p(6000, 5000)
    assert 0.0 < p < 1e-15


@pytest.mark.parametrize(("b", "c"), [(-1, 0), (0, -1)])
def test_rejects_negative_counts(b: int, c: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        exact_mcnemar_p(b, c)


# --- paired table --------------------------------------------------------------------


def test_paired_counts() -> None:
    base = {"a": True, "b": True, "c": False, "d": False, "e": True}
    cand = {"a": True, "b": False, "c": True, "d": False, "e": False}
    assert paired_counts(base, cand) == PairedCounts(
        both_pass=1, baseline_only=2, candidate_only=1, both_fail=1
    )


def test_paired_counts_ignores_dict_order() -> None:
    base = {"a": True, "b": False}
    cand = {"b": True, "a": True}
    assert paired_counts(base, cand) == PairedCounts(1, 0, 1, 0)


def test_mismatched_samples_are_refused() -> None:
    with pytest.raises(ValueError, match="different samples"):
        paired_counts({"a": True, "b": True}, {"a": True, "c": True})


def test_empty_is_refused() -> None:
    with pytest.raises(ValueError, match="no samples"):
        paired_counts({}, {})


def test_mcnemar_end_to_end() -> None:
    base = {f"s{i}": True for i in range(60)}
    cand = {f"s{i}": i >= 6 for i in range(60)}  # 6 regressions, 0 improvements
    result = mcnemar(base, cand)
    assert result.counts == PairedCounts(54, 6, 0, 0)
    assert result.p_value == pytest.approx(2 / 64)
    assert result.difference == pytest.approx(-0.1)


def test_aa_identical_runs_never_fire() -> None:
    run = {f"s{i}": i % 3 != 0 for i in range(60)}
    result = mcnemar(run, dict(run))
    assert result.p_value == 1.0
    assert result.difference == 0.0
