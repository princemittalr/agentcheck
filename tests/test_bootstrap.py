"""Spec for the paired cluster bootstrap."""

import random
import statistics

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import bootstrap as scipy_bootstrap

from agentcheck.stats.bootstrap import paired_bootstrap, per_sample_differences, quantile

Runs = dict[str, list[bool]]


def make_runs(n: int, epochs: int, p_base: float, p_cand: float, seed: int) -> tuple[Runs, Runs]:
    rng = random.Random(seed)
    base = {f"s{i:03}": [rng.random() < p_base for _ in range(epochs)] for i in range(n)}
    cand = {f"s{i:03}": [rng.random() < p_cand for _ in range(epochs)] for i in range(n)}
    return base, cand


# --- per-sample differences ----------------------------------------------------------


def test_per_sample_differences() -> None:
    base = {"a": [True, True, False, False], "b": [True]}
    cand = {"a": [True, True, True, False], "b": [False, False]}
    assert per_sample_differences(base, cand) == [0.25, -1.0]


def test_epoch_counts_may_differ_between_runs() -> None:
    assert per_sample_differences({"a": [True]}, {"a": [True, False]}) == [-0.5]


@pytest.mark.parametrize(
    ("base", "cand", "match"),
    [
        ({}, {}, "no samples"),
        ({"a": [True]}, {"b": [True]}, "different samples"),
        ({"a": []}, {"a": [True]}, "no epochs"),
    ],
)
def test_bad_runs_are_refused(base: Runs, cand: Runs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        per_sample_differences(base, cand)


# --- quantile ------------------------------------------------------------------------


@given(
    st.lists(st.floats(-1, 1), min_size=1, max_size=200),
    st.floats(0, 1),
)
def test_quantile_matches_numpy(values: list[float], q: float) -> None:
    xs = sorted(values)
    assert quantile(xs, q) == pytest.approx(float(np.quantile(xs, q)), abs=1e-12)


# --- the bootstrap -------------------------------------------------------------------


def test_deterministic_for_a_seed() -> None:
    base, cand = make_runs(60, 5, 0.8, 0.7, seed=1)
    assert paired_bootstrap(base, cand, seed=7) == paired_bootstrap(base, cand, seed=7)


def test_identical_runs_give_a_zero_width_interval_at_zero() -> None:
    base, _ = make_runs(60, 5, 0.8, 0.8, seed=2)
    result = paired_bootstrap(base, base)
    assert (result.difference, result.low, result.high) == (0.0, 0.0, 0.0)
    assert result.p_value == 1.0


def test_interval_contains_estimate_and_detects_a_big_drop() -> None:
    base, cand = make_runs(60, 5, 0.95, 0.55, seed=3)
    result = paired_bootstrap(base, cand)
    assert result.low <= result.difference <= result.high
    assert result.high < 0
    assert result.p_value < 0.01


def test_p_value_agrees_with_the_interval() -> None:
    """p < 0.05 exactly when the 95% interval excludes 0 (up to resampling granularity)."""
    for seed in range(10):
        base, cand = make_runs(40, 3, 0.8, 0.7, seed=100 + seed)
        r = paired_bootstrap(base, cand)
        if r.low > 0.005 or r.high < -0.005:
            assert r.p_value < 0.05
        if r.low < -0.005 and r.high > 0.005:
            assert r.p_value > 0.05


def test_agrees_with_scipy_percentile_bootstrap() -> None:
    base, cand = make_runs(60, 5, 0.8, 0.7, seed=4)
    diffs = per_sample_differences(base, cand)
    ours = paired_bootstrap(base, cand, n_resamples=20_000)
    ref = scipy_bootstrap(
        (np.array(diffs),), np.mean, n_resamples=20_000, method="percentile", rng=0
    ).confidence_interval
    assert ours.low == pytest.approx(ref.low, abs=0.01)
    assert ours.high == pytest.approx(ref.high, abs=0.01)


def test_width_matches_normal_theory() -> None:
    """For a mean of n iid values, the bootstrap SE is about sd * sqrt((n - 1) / n) / sqrt(n)."""
    base, cand = make_runs(200, 3, 0.7, 0.7, seed=5)
    diffs = per_sample_differences(base, cand)
    n = len(diffs)
    se = statistics.pstdev(diffs) / n**0.5
    result = paired_bootstrap(base, cand, n_resamples=20_000)
    assert (result.high - result.low) == pytest.approx(2 * 1.959964 * se, rel=0.05)


def test_copied_epochs_do_not_shrink_the_interval() -> None:
    """The clustering point: 5 identical epochs carry no more information than 1."""
    base1, cand1 = make_runs(60, 1, 0.8, 0.7, seed=6)
    base5 = {k: v * 5 for k, v in base1.items()}
    cand5 = {k: v * 5 for k, v in cand1.items()}
    assert paired_bootstrap(base1, cand1) == paired_bootstrap(base5, cand5)


def test_higher_confidence_is_wider() -> None:
    base, cand = make_runs(60, 5, 0.8, 0.7, seed=8)
    narrow = paired_bootstrap(base, cand, confidence=0.80)
    wide = paired_bootstrap(base, cand, confidence=0.99)
    assert wide.low <= narrow.low
    assert wide.high >= narrow.high


@pytest.mark.parametrize(
    ("n_resamples", "confidence"), [(0, 0.95), (100, 0.0), (100, 1.0), (100, 1.5)]
)
def test_bad_settings_are_refused(n_resamples: int, confidence: float) -> None:
    base, cand = make_runs(5, 1, 0.5, 0.5, seed=9)
    with pytest.raises(ValueError, match="must be"):
        paired_bootstrap(base, cand, n_resamples=n_resamples, confidence=confidence)
