"""Spec for Cohen's kappa and the position-bias test."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from statsmodels.stats.inter_rater import cohens_kappa as sm_kappa

from agentcheck.stats.agreement import cohens_kappa, kappa_from_labels, position_bias


def test_hand_checked() -> None:
    # p_o = (20 + 15) / 50 = 0.7; p_e = 0.5 * 0.6 + 0.5 * 0.4 = 0.5; kappa = 0.2 / 0.5
    k = cohens_kappa([[20, 5], [10, 15]])
    assert k.observed == pytest.approx(0.7)
    assert k.expected == pytest.approx(0.5)
    assert k.kappa == pytest.approx(0.4)


def test_high_raw_agreement_can_mean_low_kappa() -> None:
    """The reason to use kappa: 90% raw agreement, but no better than chance."""
    k = cohens_kappa([[90, 5], [5, 0]])
    assert k.observed == pytest.approx(0.90)
    assert k.kappa < 0


def test_perfect_agreement() -> None:
    k = cohens_kappa([[30, 0], [0, 20]])
    assert k.kappa == pytest.approx(1.0)
    assert k.se == pytest.approx(0.0, abs=1e-12)


tables = st.integers(2, 4).flatmap(
    lambda k: st.lists(st.lists(st.integers(0, 40), min_size=k, max_size=k), min_size=k, max_size=k)
)


@settings(max_examples=200)
@given(tables)
def test_matches_statsmodels(table: list[list[int]]) -> None:
    for i in range(len(table)):
        table[i][i] += 1  # avoid all-zero tables
    rows = [sum(r) for r in table]
    cols = [sum(c) for c in zip(*table, strict=True)]
    if max(rows) == max(cols) == sum(rows) and rows.index(max(rows)) == cols.index(max(cols)):
        with pytest.raises(ValueError, match="undefined"):  # one shared label: no kappa
            cohens_kappa(table)
        return
    ours = cohens_kappa(table)
    with np.errstate(invalid="ignore"):  # statsmodels takes sqrt(-1e-17) at kappa = 1
        ref = sm_kappa(np.array(table))
    assert ours.kappa == pytest.approx(float(ref.kappa), abs=1e-9)
    ref_se = 0.0 if np.isnan(ref.std_kappa) else float(ref.std_kappa)
    # sqrt of a variance that rounds to ~1e-16 instead of 0 gives ~1e-8: allow for it
    assert ours.se == pytest.approx(ref_se, abs=1e-7)
    assert -1.0 <= ours.low <= ours.kappa <= ours.high <= 1.0


def test_from_labels() -> None:
    judge = ["C", "C", "I", "C", "I", "I"]
    truth = ["C", "I", "I", "C", "I", "C"]
    assert kappa_from_labels(judge, truth) == cohens_kappa([[2, 1], [1, 2]])


@pytest.mark.parametrize(
    ("table", "match"),
    [
        ([[5]], "square"),
        ([[1, 2], [3]], "square"),
        ([[0, 0], [0, 0]], "empty"),
        ([[-1, 0], [0, 1]], "non-negative"),
        ([[10, 0], [0, 0]], "undefined"),
    ],
)
def test_bad_tables_are_refused(table: list[list[int]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cohens_kappa(table)


def test_label_lists_must_match() -> None:
    with pytest.raises(ValueError, match="length"):
        kappa_from_labels(["C"], ["C", "I"])


# --- position bias -------------------------------------------------------------------


def test_unbiased_consistent_judge() -> None:
    pb = position_bias(["A", "B", "A", "B"], ["A", "B", "A", "B"])
    assert pb.consistency == 1.0
    assert (pb.first_both, pb.second_both) == (0, 0)
    assert pb.p_value == 1.0


def test_judge_that_always_picks_the_first_answer() -> None:
    n = 12
    pb = position_bias(["A"] * n, ["B"] * n)  # (A,B): A first wins; (B,A): B first wins
    assert pb.consistency == 0.0
    assert pb.first_both == n
    assert pb.p_value == pytest.approx(2 / 2**n)


def test_position_bias_input_checks() -> None:
    with pytest.raises(ValueError, match="equal length"):
        position_bias(["A"], [])
    with pytest.raises(ValueError, match='"A" or "B"'):
        position_bias(["A"], ["C"])
