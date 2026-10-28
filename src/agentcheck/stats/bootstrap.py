"""Paired cluster bootstrap for the pass-rate difference when samples have several epochs.

Epochs of one sample are correlated, so the sample (not the epoch) is the unit that is
resampled. Each sample contributes one number: candidate rate minus baseline rate.
"""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class BootstrapResult:
    difference: float  # candidate pass rate minus baseline, averaged over samples
    low: float
    high: float
    n_samples: int
    n_resamples: int


def per_sample_differences(
    baseline: Mapping[str, Sequence[bool]], candidate: Mapping[str, Sequence[bool]]
) -> list[float]:
    """Candidate minus baseline pass rate for each sample, in sorted sample-id order."""
    if not baseline:
        raise ValueError("no samples to compare")
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() ^ candidate.keys())
        raise ValueError(f"runs cover different samples; not in both: {missing[:5]}")
    diffs = []
    for sample_id in sorted(baseline):
        base, cand = baseline[sample_id], candidate[sample_id]
        if not base or not cand:
            raise ValueError(f"sample {sample_id!r} has no epochs in one run")
        diffs.append(fmean(cand) - fmean(base))
    return diffs


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (NumPy's default method) of already sorted values."""
    if not sorted_values:
        raise ValueError("no values")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


def paired_bootstrap(
    baseline: Mapping[str, Sequence[bool]],
    candidate: Mapping[str, Sequence[bool]],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile interval for the mean per-sample difference, resampling samples."""
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    diffs = per_sample_differences(baseline, candidate)
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(fmean(rng.choices(diffs, k=n)) for _ in range(n_resamples))
    alpha = 1 - confidence
    return BootstrapResult(
        difference=fmean(diffs),
        low=quantile(means, alpha / 2),
        high=quantile(means, 1 - alpha / 2),
        n_samples=n,
        n_resamples=n_resamples,
    )
