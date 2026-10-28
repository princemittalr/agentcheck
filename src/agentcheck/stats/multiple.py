"""Multiple comparisons: Benjamini-Hochberg false discovery rate control.

Used when one comparison is split into several tests (e.g. one per task category).
"""

from collections.abc import Sequence


def _check(p_values: Sequence[float]) -> None:
    if not p_values:
        raise ValueError("no p-values")
    for p in p_values:
        if not 0.0 <= p <= 1.0:  # also rejects nan
            raise ValueError(f"p-values must be in [0, 1], got {p}")


def bh_adjust(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order.

    adjusted_(i) = min over j >= i of m * p_(j) / j, capped at 1, for sorted p_(1) <= ... <= p_(m).
    """
    _check(p_values)
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        # max(p, ...) guards rounding: m * p / m can come out a hair below p.
        running = min(running, max(p_values[i], m * p_values[i] / rank))
        adjusted[i] = running
    return adjusted


def bh_reject(p_values: Sequence[float], fdr: float = 0.05) -> list[bool]:
    """Which hypotheses to reject while keeping the false discovery rate at `fdr`.

    Assumes independent or positively dependent tests (Benjamini & Yekutieli 2001).
    """
    if not 0.0 < fdr < 1.0:
        raise ValueError(f"fdr must be in (0, 1), got {fdr}")
    return [q <= fdr for q in bh_adjust(p_values)]
