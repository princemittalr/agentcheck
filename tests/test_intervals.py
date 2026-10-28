"""Spec for wilson_interval. Do not weaken these tests to make them pass."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import binomtest

from agentcheck.stats.intervals import wilson_interval

# --- textbook values -----------------------------------------------------------------
# Newcombe (1998), "Two-sided confidence intervals for the single proportion",
# Statistics in Medicine 17:857-872, Table I (Wilson score method, 95%).


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (81, 263, 0.2553, 0.3662),
        (15, 148, 0.0624, 0.1605),
        (0, 20, 0.0000, 0.1611),
        (1, 29, 0.0061, 0.1718),
    ],
)
def test_newcombe_table(k: int, n: int, low: float, high: float) -> None:
    ci = wilson_interval(k, n)
    assert ci.low == pytest.approx(low, abs=5e-5)
    assert ci.high == pytest.approx(high, abs=5e-5)


# --- closed forms at the edges -------------------------------------------------------

Z95 = 1.959963984540054


@pytest.mark.parametrize("n", [1, 2, 5, 60, 10_000])
def test_zero_successes(n: int) -> None:
    ci = wilson_interval(0, n)
    assert ci.low == 0.0  # exactly, not -1e-17
    assert ci.high == pytest.approx(Z95**2 / (n + Z95**2), rel=1e-12)


@pytest.mark.parametrize("n", [1, 2, 5, 60, 10_000])
def test_all_successes(n: int) -> None:
    ci = wilson_interval(n, n)
    assert ci.high == 1.0  # exactly, not 1.0000000000000002
    assert ci.low == pytest.approx(n / (n + Z95**2), rel=1e-12)


def test_ceiling_case_is_not_degenerate() -> None:
    """58/60 (our qwen v3 run): a Wald interval would be far too narrow here; Wilson isn't."""
    ci = wilson_interval(58, 60)
    assert ci.low == pytest.approx(0.8864, abs=5e-5)
    assert ci.high == pytest.approx(0.9908, abs=5e-5)


# --- against SciPy on random inputs --------------------------------------------------

counts = st.integers(min_value=1, max_value=5000).flatmap(
    lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
)
levels = st.sampled_from([0.5, 0.8, 0.9, 0.95, 0.99, 0.999])


@given(counts, levels)
def test_matches_scipy(kn: tuple[int, int], confidence: float) -> None:
    k, n = kn
    ours = wilson_interval(k, n, confidence)
    ref = binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
    assert ours.low == pytest.approx(ref.low, abs=1e-9)
    assert ours.high == pytest.approx(ref.high, abs=1e-9)


# --- properties every correct interval must have -------------------------------------


@given(counts, levels)
def test_bounds_are_ordered_and_contain_estimate(kn: tuple[int, int], confidence: float) -> None:
    k, n = kn
    ci = wilson_interval(k, n, confidence)
    assert 0.0 <= ci.low <= k / n <= ci.high <= 1.0


@given(counts)
def test_symmetry_passes_vs_failures(kn: tuple[int, int]) -> None:
    """Swapping passes and failures mirrors the interval around 1/2."""
    k, n = kn
    a, b = wilson_interval(k, n), wilson_interval(n - k, n)
    assert a.low == pytest.approx(1 - b.high, abs=1e-12)
    assert a.high == pytest.approx(1 - b.low, abs=1e-12)


@given(counts)
def test_higher_confidence_is_wider(kn: tuple[int, int]) -> None:
    k, n = kn
    narrow, wide = wilson_interval(k, n, 0.90), wilson_interval(k, n, 0.99)
    assert wide.low <= narrow.low
    assert wide.high >= narrow.high


def test_more_data_is_narrower() -> None:
    small, large = wilson_interval(30, 60), wilson_interval(300, 600)
    assert (large.high - large.low) < (small.high - small.low)


def test_huge_n_is_finite_and_tight() -> None:
    ci = wilson_interval(500_000_000, 10**9)
    assert math.isfinite(ci.low)
    assert math.isfinite(ci.high)
    assert ci.high - ci.low < 1e-4


# --- input validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n", "confidence"),
    [
        (0, 0, 0.95),  # no trials: the proportion is undefined
        (-1, 10, 0.95),
        (11, 10, 0.95),
        (5, 10, 0.0),
        (5, 10, 1.0),
        (5, 10, 1.5),
    ],
)
def test_rejects_invalid_input(k: int, n: int, confidence: float) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - any clear ValueError message is fine
        wilson_interval(k, n, confidence)
