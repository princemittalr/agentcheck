"""Paired sign-flip permutation test on per-sample differences.

Under the null that baseline and candidate are exchangeable (an A/A comparison), each
sample's difference d_i = candidate rate - baseline rate is as likely to be +d_i as
-d_i. Flipping signs at random gives the exact null distribution of the mean, so the
false-alarm rate is at most alpha by construction, whatever n is. That is the guarantee
the percentile bootstrap lacks at n = 60.
"""

import itertools
from collections.abc import Sequence
from functools import cache

import numpy as np
import numpy.typing as npt

EXACT_MAX_NONZERO = 16  # 2**16 = 65,536 sign patterns: exact enumeration stays fast
_TOL = 1e-12


@cache
def _all_signs(k: int) -> npt.NDArray[np.int8]:
    return np.array(list(itertools.product((1, -1), repeat=k)), dtype=np.int8)


def sign_flip_p(diffs: Sequence[float], n_resamples: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for mean(diffs) == 0.

    Exact when at most EXACT_MAX_NONZERO differences are non-zero (zeros never change
    the statistic). Otherwise Monte Carlo with the +1 correction (Phipson & Smyth 2010),
    which keeps the p-value valid. NumPy vectorises the sign matrix; it is a runtime
    dependency of Inspect already.
    """
    if not diffs:
        raise ValueError("no differences")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    nonzero = np.array([d for d in diffs if d != 0.0], dtype=np.float64)
    if nonzero.size == 0:
        return 1.0
    observed = abs(float(nonzero.sum()))
    if nonzero.size <= EXACT_MAX_NONZERO:
        totals = _all_signs(nonzero.size) @ nonzero
        return float(np.count_nonzero(np.abs(totals) >= observed - _TOL)) / totals.size
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_resamples, nonzero.size), dtype=np.int8) * 2 - 1
    extreme = int(np.count_nonzero(np.abs(signs @ nonzero) >= observed - _TOL))
    return (extreme + 1) / (n_resamples + 1)
