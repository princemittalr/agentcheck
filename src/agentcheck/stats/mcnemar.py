"""Paired comparison of two agents on the same samples: exact McNemar test."""

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedCounts:
    """2x2 table of per-sample outcomes, baseline vs candidate."""

    both_pass: int
    baseline_only: int  # passed on baseline, failed on candidate: a regression
    candidate_only: int  # failed on baseline, passed on candidate: an improvement
    both_fail: int

    @property
    def n(self) -> int:
        return self.both_pass + self.baseline_only + self.candidate_only + self.both_fail

    @property
    def discordant(self) -> int:
        return self.baseline_only + self.candidate_only


@dataclass(frozen=True)
class McNemarResult:
    counts: PairedCounts
    p_value: float

    @property
    def difference(self) -> float:
        """Candidate pass rate minus baseline pass rate."""
        return (self.counts.candidate_only - self.counts.baseline_only) / self.counts.n


def paired_counts(baseline: Mapping[str, bool], candidate: Mapping[str, bool]) -> PairedCounts:
    """Build the 2x2 table. Both runs must cover exactly the same, non-empty set of samples."""
    if not baseline:
        raise ValueError("no samples to compare")
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() ^ candidate.keys())
        raise ValueError(f"runs cover different samples; not in both: {missing[:5]}")
    table = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for sample_id, passed in baseline.items():
        table[(passed, candidate[sample_id])] += 1
    return PairedCounts(
        both_pass=table[(True, True)],
        baseline_only=table[(True, False)],
        candidate_only=table[(False, True)],
        both_fail=table[(False, False)],
    )


def exact_mcnemar_p(baseline_only: int, candidate_only: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of b out of b + c at p = 1/2.

    With no discordant pairs there is no evidence either way, so p = 1.
    Integer arithmetic keeps it exact for any n (Python's int / int is correctly rounded).
    """
    if baseline_only < 0 or candidate_only < 0:
        raise ValueError("counts must be non-negative")
    m = baseline_only + candidate_only
    k = min(baseline_only, candidate_only)
    if 2 * k >= m:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (1 << m))


def mcnemar(baseline: Mapping[str, bool], candidate: Mapping[str, bool]) -> McNemarResult:
    counts = paired_counts(baseline, candidate)
    return McNemarResult(counts, exact_mcnemar_p(counts.baseline_only, counts.candidate_only))
