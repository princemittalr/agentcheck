"""Judge calibration: Cohen's kappa against reference labels, and position bias.

Kappa corrects raw agreement for the agreement two raters would reach by chance given
their label frequencies: kappa = (p_o - p_e) / (1 - p_e). An LLM judge that says
"correct" 95% of the time on a suite where 95% of answers are correct agrees 90%+ by
chance alone, which raw accuracy hides and kappa does not.
"""

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

from agentcheck.stats.mcnemar import exact_mcnemar_p


@dataclass(frozen=True)
class Kappa:
    kappa: float
    se: float  # large-sample standard error, Fleiss, Cohen & Everitt (1969)
    low: float
    high: float
    n: int
    observed: float  # raw agreement p_o
    expected: float  # chance agreement p_e


def cohens_kappa(table: Sequence[Sequence[int]], confidence: float = 0.95) -> Kappa:
    """Kappa for a k x k contingency table (rows: rater A's label, columns: rater B's)."""
    k = len(table)
    if k < 2 or any(len(row) != k for row in table):
        raise ValueError("table must be square and at least 2 x 2")
    if any(c < 0 for row in table for c in row):
        raise ValueError("counts must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    n = sum(sum(row) for row in table)
    if n == 0:
        raise ValueError("empty table")
    p = [[c / n for c in row] for row in table]
    rows = [sum(r) for r in p]
    cols = [sum(p[i][j] for i in range(k)) for j in range(k)]
    p_o = sum(p[i][i] for i in range(k))
    p_e = sum(rows[i] * cols[i] for i in range(k))
    if math.isclose(p_e, 1.0):
        raise ValueError("kappa is undefined: both raters use a single, identical label")
    kappa = (p_o - p_e) / (1 - p_e)

    # Fleiss, Cohen & Everitt (1969), eq. 13: variance of kappa when kappa != 0.
    diag = sum(p[i][i] * ((1 - p_e) - (rows[i] + cols[i]) * (1 - p_o)) ** 2 for i in range(k))
    off = (1 - p_o) ** 2 * sum(
        p[i][j] * (cols[i] + rows[j]) ** 2 for i in range(k) for j in range(k) if i != j
    )
    var = (diag + off - (p_o * p_e - 2 * p_e + p_o) ** 2) / (n * (1 - p_e) ** 4)
    se = math.sqrt(max(var, 0.0))
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    return Kappa(
        kappa=kappa,
        se=se,
        low=max(-1.0, kappa - z * se),
        high=min(1.0, kappa + z * se),
        n=n,
        observed=p_o,
        expected=p_e,
    )


def kappa_from_labels(a: Sequence[Hashable], b: Sequence[Hashable]) -> Kappa:
    if len(a) != len(b):
        raise ValueError(f"label lists differ in length: {len(a)} vs {len(b)}")
    labels = sorted({*a, *b}, key=repr)
    if len(labels) == 1:
        labels.append(object())  # a 2 x 2 table with an empty second label
    index = {lab: i for i, lab in enumerate(labels)}
    table = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b, strict=True):
        table[index[x]][index[y]] += 1
    return cohens_kappa(table)


@dataclass(frozen=True)
class PositionBias:
    n: int
    consistent: int  # same winner in both orders
    first_both: int  # the answer shown first won in both orders
    second_both: int  # the answer shown second won in both orders
    p_value: float  # exact two-sided test of first_both vs second_both

    @property
    def consistency(self) -> float:
        return self.consistent / self.n


def position_bias(winners_ab: Sequence[str], winners_ba: Sequence[str]) -> PositionBias:
    """Pairwise judge shown (A, B) then (B, A). Each entry is "A" or "B", the item judged
    better. A judge without position bias flips its stated position when the order flips;
    among inconsistent items, "first shown wins both times" and "second shown wins both
    times" are then equally likely, which an exact McNemar/sign test checks.
    """
    if len(winners_ab) != len(winners_ba) or not winners_ab:
        raise ValueError("need two non-empty lists of equal length")
    if any(w not in ("A", "B") for w in (*winners_ab, *winners_ba)):
        raise ValueError('winners must be "A" or "B"')
    consistent = sum(x == y for x, y in zip(winners_ab, winners_ba, strict=True))
    # (A,B) order: A is first. (B,A) order: B is first.
    first_both = sum(x == "A" and y == "B" for x, y in zip(winners_ab, winners_ba, strict=True))
    second_both = sum(x == "B" and y == "A" for x, y in zip(winners_ab, winners_ba, strict=True))
    return PositionBias(
        n=len(winners_ab),
        consistent=consistent,
        first_both=first_both,
        second_both=second_both,
        p_value=exact_mcnemar_p(first_both, second_both),
    )
