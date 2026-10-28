"""Spec for Benjamini-Hochberg."""

import math
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import false_discovery_control

from agentcheck.stats.multiple import bh_adjust, bh_reject

p_lists = st.lists(st.floats(0.0, 1.0), min_size=1, max_size=50)


def test_hand_checked() -> None:
    # m = 4, sorted: 0.01*4/1 = 0.04, 0.02*4/2 = 0.04, 0.03*4/3 = 0.04, 0.5*4/4 = 0.5
    assert bh_adjust([0.5, 0.01, 0.03, 0.02]) == pytest.approx([0.5, 0.04, 0.04, 0.04])


def test_step_up_rescues_a_larger_p() -> None:
    """0.04 alone fails 0.05 * 1/2, but the step-up lets the larger p carry it."""
    assert bh_reject([0.04, 0.05]) == [True, True]
    assert bh_reject([0.04, 0.06]) == [False, False]


def test_single_p_is_unchanged() -> None:
    assert bh_adjust([0.03]) == [0.03]


def test_ties_get_equal_adjusted_values() -> None:
    adjusted = bh_adjust([0.02, 0.02, 0.02])
    assert adjusted == pytest.approx([0.02, 0.02, 0.02])


@given(p_lists)
def test_matches_scipy(ps: list[float]) -> None:
    ref = false_discovery_control(ps, method="bh")
    assert bh_adjust(ps) == pytest.approx(list(ref), abs=1e-12)


@given(p_lists)
def test_properties(ps: list[float]) -> None:
    adjusted = bh_adjust(ps)
    assert all(p <= q <= 1.0 for p, q in zip(ps, adjusted, strict=True))
    pairs = sorted(zip(ps, adjusted, strict=True))
    assert all(a[1] <= b[1] for a, b in pairwise(pairs))


@given(p_lists, st.sampled_from([0.01, 0.05, 0.1]))
def test_rejects_at_least_as_many_as_bonferroni(ps: list[float], fdr: float) -> None:
    bonferroni = sum(p <= fdr / len(ps) for p in ps)
    assert sum(bh_reject(ps, fdr)) >= bonferroni


@pytest.mark.parametrize("ps", [[], [-0.1], [1.1], [math.nan]])
def test_bad_p_values_are_refused(ps: list[float]) -> None:
    with pytest.raises(ValueError, match="p-values"):
        bh_adjust(ps)


@pytest.mark.parametrize("fdr", [0.0, 1.0, 1.5])
def test_bad_fdr_is_refused(fdr: float) -> None:
    with pytest.raises(ValueError, match="fdr"):
        bh_reject([0.01], fdr)
