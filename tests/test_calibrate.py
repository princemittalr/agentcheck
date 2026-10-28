"""Tests for the calibration harness."""

import random
from pathlib import Path
from statistics import fmean

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from agentcheck.calibrate import (
    break_samples,
    calibrate,
    check_pool,
    draw_run,
    pool_from_logs,
    pool_rate,
    synthetic_pool,
)
from agentcheck.cli import main
from agentcheck.compare import compare_records, decide
from agentcheck.summary import SampleRecord
from agentcheck.tasks.inventory import inventory_lookup


def test_one_epoch_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="need >= 2"):
        check_pool({"a": [True, False], "b": [True]})
    with pytest.raises(ValueError, match="empty"):
        check_pool({})


def test_draws_come_from_the_pool() -> None:
    pool = {"always": [True, True], "never": [False, False], "half": [True, False]}
    rng = random.Random(0)
    for _ in range(50):
        run = draw_run(pool, 3, rng)
        assert run["always"] == [True] * 3
        assert run["never"] == [False] * 3
        assert len(run["half"]) == 3


def test_break_samples_extremes() -> None:
    run = {"a": [True, False], "b": [True, True]}
    rng = random.Random(0)
    assert break_samples(run, 0.0, rng) == run
    assert break_samples(run, 1.0, rng) == {"a": [False, False], "b": [False, False]}


def test_injected_drop_has_the_right_size_on_average() -> None:
    pool = synthetic_pool(mean=0.8, seed=3)
    base_rate = pool_rate(pool)
    rng = random.Random(1)
    rates = []
    for _ in range(3000):
        run = break_samples(draw_run(pool, 1, rng), 0.10 / base_rate, rng)
        rates.append(fmean(fmean(v) for v in run.values()))
    assert fmean(rates) == pytest.approx(base_rate - 0.10, abs=0.005)


runs = st.integers(0, 2**32 - 1).flatmap(
    lambda seed: st.tuples(st.just(seed), st.sampled_from([1, 3]))
)


@settings(max_examples=30, deadline=None)
@given(runs)
def test_decide_agrees_with_compare_records(seed_epochs: tuple[int, int]) -> None:
    seed, epochs = seed_epochs
    rng = random.Random(seed)
    pool = synthetic_pool(n_samples=40, mean=0.7, seed=seed % 1000)
    base = draw_run(pool, epochs, rng)
    cand = break_samples(draw_run(pool, epochs, rng), rng.random() * 0.3, rng)
    full = compare_records(
        {k: SampleRecord(k, "c", v) for k, v in base.items()},
        {k: SampleRecord(k, "c", v) for k, v in cand.items()},
        n_resamples=500,
        seed=7,
    )
    assert decide(base, cand, n_resamples=500, seed=7) is full.verdict


def test_deterministic_for_a_seed() -> None:
    pool = synthetic_pool(seed=4)
    assert calibrate(pool, reps=50, seed=9) == calibrate(pool, reps=50, seed=9)


def test_mcnemar_aa_false_alarms_stay_under_alpha() -> None:
    """Exact McNemar is conservative: A/A false alarms at or below 5%."""
    c = calibrate(synthetic_pool(mean=0.7, spread=4.0, seed=5), reps=1000, seed=1)
    assert c.false_alarms.rate <= 0.05


def test_a_huge_drop_is_almost_always_detected() -> None:
    c = calibrate(synthetic_pool(mean=0.9, seed=6), drop=0.5, reps=200, seed=2)
    assert c.detections.rate >= 0.95


def test_more_epochs_detect_more() -> None:
    pool = synthetic_pool(mean=0.8, seed=7)
    one = calibrate(pool, run_epochs=1, reps=200, seed=3, n_resamples=500)
    five = calibrate(pool, run_epochs=5, reps=200, seed=3, n_resamples=500)
    assert five.detections.rate > one.detections.rate


@pytest.mark.parametrize("drop", [0.0, -0.1, 0.99])
def test_bad_drop_is_refused(drop: float) -> None:
    with pytest.raises(ValueError, match="drop"):
        calibrate(synthetic_pool(mean=0.5, seed=8), drop=drop, reps=10)


def test_cli_synthetic(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["calibrate", "--synthetic", "--reps", "50"]) == 0
    out = capsys.readouterr().out
    assert "A/A false alarms" in out
    assert "detect -0.10 drop" in out


def test_cli_needs_exactly_one_pool_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["calibrate"]) == 2
    assert "not both or neither" in capsys.readouterr().err


# --- pool from real Inspect logs -----------------------------------------------------

MOCK = "mockllm/model"


def alternating_log(log_dir: Path, epochs: int) -> EvalLog:
    calls = {"n": 0}

    def model(input: list[object], *args: object) -> ModelOutput:
        calls["n"] += 1
        right = "37" if "SKU-1042" in str(input) else "0"
        out = ModelOutput.from_content(MOCK, f"ANSWER: {right if calls['n'] % 2 else 'x'}")
        out.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return out

    [log] = inspect_eval(
        inventory_lookup(),
        model=get_model(MOCK, custom_outputs=model),
        sample_id=["single-SKU-1042", "single-SKU-2177"],
        epochs=epochs,
        max_samples=1,
        log_dir=str(log_dir),
        display="none",
    )
    return log


def test_pool_concatenates_logs_and_cli_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = alternating_log(tmp_path / "a", epochs=2)
    b = alternating_log(tmp_path / "b", epochs=1)
    pool = pool_from_logs([a, b])
    assert {len(v) for v in pool.values()} == {3}
    assert main(["calibrate", a.location, "--reps", "20", "--drop", "0.2"]) == 0
    assert "2 samples" in capsys.readouterr().out
