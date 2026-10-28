"""Spec for alpha spending and sequential decisions."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentcheck.calibrate import calibrate, synthetic_pool
from agentcheck.compare import Verdict, decide, sequential_decide
from agentcheck.stats.sequential import look_alphas, obf_spent


def test_known_values() -> None:
    # z_0.975 / sqrt(0.5) = 2.7718 -> 2 * (1 - Phi(2.7718)) = 0.005575
    assert obf_spent(0.5) == pytest.approx(0.005575, abs=2e-6)
    # z_0.975 / sqrt(0.2) = 4.3826 -> about 1.17e-5
    assert obf_spent(0.2) == pytest.approx(1.173e-5, rel=0.01)
    assert obf_spent(1.0) == 0.05


def test_one_look_is_the_plain_test() -> None:
    assert look_alphas(1) == [0.05]


@given(st.integers(1, 20), st.sampled_from([0.01, 0.05, 0.1]))
def test_spending_sums_to_alpha_and_grows(looks: int, alpha: float) -> None:
    alphas = look_alphas(looks, alpha)
    assert len(alphas) == looks
    assert sum(alphas) == pytest.approx(alpha, rel=1e-12)
    assert all(a >= 0 for a in alphas)  # very early looks can spend ~0 (underflow)
    spent = [obf_spent(k / looks, alpha) for k in range(1, looks + 1)]
    assert spent == sorted(spent)


def test_early_looks_are_strict() -> None:
    alphas = look_alphas(5)
    assert alphas[0] < 1e-4
    # spent(0.8) = 2(1 - Phi(1.96 / sqrt(0.8))) = 0.0284, so the last look gets 0.0216
    assert alphas[-1] == pytest.approx(0.05 - obf_spent(0.8), rel=1e-12)
    assert alphas[-1] > 0.02


@pytest.mark.parametrize(("t", "alpha"), [(0.0, 0.05), (1.5, 0.05), (0.5, 0.0), (0.5, 1.0)])
def test_bad_input_is_refused(t: float, alpha: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        obf_spent(t, alpha)


def test_zero_looks_is_refused() -> None:
    with pytest.raises(ValueError, match="looks"):
        look_alphas(0)


# --- sequential decisions -------------------------------------------------------------


def test_one_look_equals_the_fixed_design() -> None:
    base = {f"s{i}": [True] * 4 for i in range(40)}
    cand = {f"s{i}": [i >= 8] * 4 for i in range(40)}
    assert sequential_decide(base, cand, looks=1) == (decide(base, cand), 4)


def test_a_clear_regression_stops_early() -> None:
    base = {f"s{i}": [True] * 4 for i in range(60)}
    cand = {f"s{i}": [i >= 30] * 4 for i in range(60)}  # half the samples broken
    verdict, used = sequential_decide(base, cand, looks=4)
    assert verdict is Verdict.REGRESSION
    assert used == 1


def test_identical_runs_use_every_epoch() -> None:
    run = {f"s{i}": [i % 2 == 0, True] for i in range(30)}
    assert sequential_decide(run, run, looks=2) == (Verdict.INCONCLUSIVE, 2)


def test_uneven_batches_are_refused() -> None:
    run = {"a": [True, False, True]}
    with pytest.raises(ValueError, match="equal looks"):
        sequential_decide(run, run, looks=2)


def test_sequential_calibration_keeps_false_alarms_and_saves_epochs() -> None:
    pool = synthetic_pool(n_samples=120, mean=0.8, seed=11)
    c = calibrate(pool, run_epochs=4, looks=4, reps=300, n_resamples=500, seed=5)
    assert c.false_alarms.rate <= 0.05 + 2 * (0.05 * 0.95 / 300) ** 0.5
    assert c.mean_epochs_drop < 4  # stops early on real drops
    assert c.mean_epochs_aa > 3.8  # almost never stops under A/A


def test_looks_must_divide_epochs() -> None:
    with pytest.raises(ValueError, match="looks must divide"):
        calibrate(synthetic_pool(seed=1), run_epochs=5, looks=2, reps=5)
