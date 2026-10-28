"""Confidence intervals for pass rates."""

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Two-sided Wilson score interval for a binomial proportion.

    Args:
        successes: number of passes, 0 <= successes <= n.
        n: number of trials, n >= 1.
        confidence: two-sided confidence level, strictly between 0 and 1.

    Returns:
        Interval with 0 <= low <= p_hat <= high <= 1, where p_hat = successes / n.

    Raises:
        ValueError: on n < 1, successes outside [0, n], or confidence outside (0, 1).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    z = NormalDist().inv_cdf((1 + confidence) / 2)
    z2 = z * z
    p = successes / n
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))

    # Exact edges: floating point can otherwise give -1e-17 or 1.0000000000000002,
    # and clamping to p_hat keeps the estimate inside its own interval.
    low = 0.0 if successes == 0 else max(0.0, min(p, centre - half))
    high = 1.0 if successes == n else min(1.0, max(p, centre + half))
    return Interval(low, high)
