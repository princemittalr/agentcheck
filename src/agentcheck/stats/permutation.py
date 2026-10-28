"""Paired sign-flip permutation test on per-sample differences.

Under the null that baseline and candidate are exchangeable (an A/A comparison), each
sample's difference d_i = candidate rate - baseline rate is as likely to be +d_i as
-d_i. Flipping signs at random gives the exact null distribution of the mean, so the
false-alarm rate is at most alpha by construction, whatever n is. That is the guarantee
the percentile bootstrap lacks at n = 60.
"""

import itertools
import random
from collections.abc import Sequence

EXACT_MAX_NONZERO = 16  # 2**16 = 65,536 sign patterns: exact enumeration stays fast
_TOL = 1e-12


def sign_flip_p(diffs: Sequence[float], n_resamples: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for mean(diffs) == 0.

    Exact when at most EXACT_MAX_NONZERO differences are non-zero (zeros never change
    the statistic). Otherwise Monte Carlo with the +1 correction (Phipson & Smyth 2010),
    which keeps the p-value valid: never smaller than the exact one in expectation.
    """
    if not diffs:
        raise ValueError("no differences")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    nonzero = [d for d in diffs if d != 0.0]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    if len(nonzero) <= EXACT_MAX_NONZERO:
        extreme = sum(
            abs(sum(s * d for s, d in zip(signs, nonzero, strict=True))) >= observed - _TOL
            for signs in itertools.product((1, -1), repeat=len(nonzero))
        )
        return extreme / (1 << len(nonzero))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        total = sum(d if rng.random() < 0.5 else -d for d in nonzero)
        extreme += abs(total) >= observed - _TOL
    return (extreme + 1) / (n_resamples + 1)
