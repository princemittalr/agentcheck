"""Tests for power-by-simulation."""

import random

import pytest

from agentcheck.calibrate import estimate, synthetic_pool
from agentcheck.cli import main
from agentcheck.power import PowerCell, cheapest, expand_pool, power_grid, render


def test_expand_pool_draws_whole_samples() -> None:
    pool = {"a": [True, False], "b": [False, False]}
    big = expand_pool(pool, 50, random.Random(0))
    assert len(big) == 50
    assert all(v in ([True, False], [False, False]) for v in big.values())


def test_expand_pool_at_pool_size_keeps_every_sample_once() -> None:
    pool = {"a": [True, True], "b": [False, True]}
    assert sorted(v for v in expand_pool(pool, 2, random.Random(0)).values()) == sorted(
        pool.values()
    )


def test_expand_pool_refuses_zero() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        expand_pool({"a": [True, False]}, 0, random.Random(0))


def test_more_samples_and_epochs_detect_more() -> None:
    pool = synthetic_pool(mean=0.8, seed=1)
    cells = {
        (c.n_samples, c.epochs): c.detections.rate
        for c in power_grid(pool, [60, 240], [1, 5], reps=150, n_resamples=500)
    }
    assert cells[(240, 1)] > cells[(60, 1)]
    assert cells[(60, 5)] > cells[(60, 1)]
    assert cells[(240, 5)] >= 0.9


def test_cheapest_needs_the_whole_interval_above_target() -> None:
    cells = [
        PowerCell(60, 5, estimate(90, 100)),  # 300 sample-epochs, CI low ~0.83
        PowerCell(120, 1, estimate(82, 100)),  # 120 sample-epochs, CI low ~0.73: not enough
        PowerCell(240, 5, estimate(100, 100)),
    ]
    best = cheapest(cells)
    assert best is not None
    assert (best.n_samples, best.epochs) == (60, 5)
    assert cheapest(cells[1:2]) is None


def test_render_reports_cheapest_or_none() -> None:
    cells = [PowerCell(60, 1, estimate(10, 100)), PowerCell(60, 3, estimate(95, 100))]
    assert "60 samples x 3 epochs" in render(cells, 0.1, "x")
    assert "No design" in render(cells[:1], 0.1, "x")


def test_cli_power(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["power", "--synthetic", "--samples", "60", "--epochs", "1,3", "--reps", "20"])
    assert code == 0
    assert "Power to detect a -0.10 drop" in capsys.readouterr().out


def test_cli_power_needs_a_pool(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["power"]) == 2
    assert "not both or neither" in capsys.readouterr().err
