"""Spec for the sign-flip permutation test."""

import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import permutation_test

from agentcheck.stats.mcnemar import exact_mcnemar_p
from agentcheck.stats.permutation import sign_flip_p

fractions = st.sampled_from([-1.0, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 1.0])


def test_hand_checked() -> None:
    # 3 non-zero, all positive: only ++ + and --- reach |sum| = 3 -> 2 / 8
    assert sign_flip_p([1.0, 1.0, 1.0, 0.0]) == pytest.approx(2 / 8)
    assert sign_flip_p([0.0, 0.0]) == 1.0
    assert sign_flip_p([0.5, -0.5]) == 1.0


@given(st.integers(0, 8), st.integers(0, 8))
def test_equals_exact_mcnemar_for_single_epoch(b: int, c: int) -> None:
    """With differences in {-1, 0, +1}, sign flipping is the exact McNemar test.

    (Up to 16 discordant pairs, where both are exact.)
    """
    diffs = [-1.0] * b + [1.0] * c + [0.0] * 5
    assert sign_flip_p(diffs) == pytest.approx(exact_mcnemar_p(b, c), rel=1e-12)


@settings(max_examples=40, deadline=None)
@given(st.lists(fractions, min_size=2, max_size=12))
def test_matches_scipy_exact(diffs: list[float]) -> None:
    if abs(sum(diffs)) < 1e-9:
        # Observed mean is exactly 0 in real arithmetic, so p = 1. SciPy can return less
        # here because it compares against a float residue such as 5e-17.
        assert sign_flip_p(diffs) == 1.0
        return
    x = np.array(diffs)
    ref = permutation_test(
        (x, np.zeros_like(x)),
        lambda a, b: np.mean(a - b),
        permutation_type="samples",
        alternative="two-sided",
        n_resamples=np.inf,
    ).pvalue
    assert sign_flip_p(diffs) == pytest.approx(ref, abs=1e-9)


def test_monte_carlo_is_close_to_exact() -> None:
    rng = random.Random(0)
    diffs = [rng.choice([-0.4, -0.2, 0.2, 0.4, 0.6]) for _ in range(16)]
    exact = sign_flip_p(diffs)
    assert sign_flip_p([*diffs, 0.0, 0.0]) == exact  # zeros never count: still exact
    # 32 non-zero (two copies) takes the Monte Carlo path; compare with SciPy's own MC
    doubled = [*diffs, *diffs]
    x = np.array(doubled)
    ref = permutation_test(
        (x, np.zeros_like(x)),
        lambda a, b: np.mean(a - b),
        permutation_type="samples",
        n_resamples=100_000,
        rng=0,
    ).pvalue
    assert sign_flip_p(doubled, n_resamples=100_000) == pytest.approx(ref, abs=0.005)


def test_monte_carlo_never_returns_zero() -> None:
    assert sign_flip_p([1.0] * 40, n_resamples=1000) == pytest.approx(1 / 1001)


def test_aa_false_alarms_at_most_alpha() -> None:
    """Exchangeable runs: the rejection rate must not exceed 5% (by construction)."""
    rng = random.Random(1)
    rejections = 0
    reps = 400
    for _ in range(reps):
        ps = [rng.betavariate(3, 1) for _ in range(60)]
        diffs = [
            sum(rng.random() < p for _ in range(5)) / 5
            - sum(rng.random() < p for _ in range(5)) / 5
            for p in ps
        ]
        rejections += sign_flip_p(diffs, n_resamples=1000, seed=rng.randrange(2**31)) < 0.05
    assert rejections / reps <= 0.05 + 2 * (0.05 * 0.95 / reps) ** 0.5


@pytest.mark.parametrize(("diffs", "n"), [([], 100), ([1.0], 0)])
def test_bad_input_is_refused(diffs: list[float], n: int) -> None:
    with pytest.raises(ValueError, match=r"no differences|n_resamples"):
        sign_flip_p(diffs, n_resamples=n)
