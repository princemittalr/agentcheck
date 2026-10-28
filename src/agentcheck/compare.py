"""Compare a baseline and a candidate eval log: regression, improvement or inconclusive.

Primary test (decides the verdict):
- one epoch per sample in both logs: exact McNemar on the paired pass/fail table;
- otherwise: paired sign-flip permutation test on per-sample differences (exact under
  exchangeability, so A/A false alarms stay <= alpha); the interval comes from the
  paired cluster bootstrap. Samples are the unit; epochs are never resampled alone.
Per-category tests use the same method and are adjusted with Benjamini-Hochberg; they
explain where a change happened but do not change the verdict.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from inspect_ai.log import EvalLog

from agentcheck.stats.bootstrap import paired_bootstrap, per_sample_differences
from agentcheck.stats.mcnemar import exact_mcnemar_p, paired_counts
from agentcheck.stats.multiple import bh_adjust
from agentcheck.stats.permutation import sign_flip_p
from agentcheck.summary import SampleRecord, per_sample


class Verdict(StrEnum):
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class TestResult:
    method: str  # "mcnemar" or "permutation"
    n_samples: int
    baseline_rate: float
    candidate_rate: float
    difference: float  # candidate minus baseline
    low: float
    high: float
    p_value: float


@dataclass(frozen=True)
class CategoryResult:
    category: str
    test: TestResult
    p_adjusted: float
    flagged: bool


@dataclass(frozen=True)
class Comparison:
    verdict: Verdict
    overall: TestResult
    categories: list[CategoryResult]
    alpha: float


def _rate(records: Mapping[str, SampleRecord]) -> float:
    """Mean over samples of each sample's pass rate (each sample weighs the same)."""
    return sum(sum(r.passes) / len(r.passes) for r in records.values()) / len(records)


def _test(
    base: Mapping[str, SampleRecord],
    cand: Mapping[str, SampleRecord],
    alpha: float,
    n_resamples: int,
    seed: int,
) -> TestResult:
    base_passes = {k: r.passes for k, r in base.items()}
    cand_passes = {k: r.passes for k, r in cand.items()}
    boot = paired_bootstrap(
        base_passes, cand_passes, n_resamples=n_resamples, confidence=1 - alpha, seed=seed
    )
    single_epoch = all(len(v) == 1 for v in (*base_passes.values(), *cand_passes.values()))
    if single_epoch:
        counts = paired_counts(
            {k: v[0] for k, v in base_passes.items()}, {k: v[0] for k, v in cand_passes.items()}
        )
        p = exact_mcnemar_p(counts.baseline_only, counts.candidate_only)
        method = "mcnemar"
    else:
        diffs = per_sample_differences(base_passes, cand_passes)
        p = sign_flip_p(diffs, n_resamples=n_resamples, seed=seed)
        method = "permutation"
    return TestResult(
        method=method,
        n_samples=boot.n_samples,
        baseline_rate=_rate(base),
        candidate_rate=_rate(cand),
        difference=boot.difference,
        low=boot.low,
        high=boot.high,
        p_value=p,
    )


def decide(
    base: Mapping[str, Sequence[bool]],
    cand: Mapping[str, Sequence[bool]],
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Verdict:
    """The primary test only, without the report: same verdict as compare_records, faster."""
    if all(len(v) == 1 for v in (*base.values(), *cand.values())):
        counts = paired_counts(
            {k: v[0] for k, v in base.items()}, {k: v[0] for k, v in cand.items()}
        )
        p = exact_mcnemar_p(counts.baseline_only, counts.candidate_only)
        diff = float(counts.candidate_only - counts.baseline_only)
    else:
        diffs = per_sample_differences(base, cand)
        p = sign_flip_p(diffs, n_resamples=n_resamples, seed=seed)
        diff = sum(diffs)
    if p >= alpha:
        return Verdict.INCONCLUSIVE
    return Verdict.REGRESSION if diff < 0 else Verdict.IMPROVEMENT


def compare_records(
    base: Mapping[str, SampleRecord],
    cand: Mapping[str, SampleRecord],
    *,
    alpha: float = 0.05,
    fdr: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Comparison:
    if base.keys() != cand.keys():
        missing = sorted(base.keys() ^ cand.keys())
        raise ValueError(f"logs cover different samples; not in both: {missing[:5]}")
    overall = _test(base, cand, alpha, n_resamples, seed)
    if overall.p_value < alpha:
        verdict = Verdict.REGRESSION if overall.difference < 0 else Verdict.IMPROVEMENT
    else:
        verdict = Verdict.INCONCLUSIVE

    by_cat: dict[str, list[str]] = defaultdict(list)
    for key, rec in base.items():
        by_cat[rec.category].append(key)
    cats = sorted(by_cat)
    tests = [
        _test(
            {k: base[k] for k in by_cat[c]},
            {k: cand[k] for k in by_cat[c]},
            alpha,
            n_resamples,
            seed,
        )
        for c in cats
    ]
    adjusted = bh_adjust([t.p_value for t in tests])
    categories = [
        CategoryResult(c, t, q, q <= fdr) for c, t, q in zip(cats, tests, adjusted, strict=True)
    ]
    return Comparison(verdict, overall, categories, alpha)


def compare_logs(
    baseline: EvalLog,
    candidate: EvalLog,
    *,
    alpha: float = 0.05,
    fdr: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Comparison:
    if baseline.eval.task != candidate.eval.task:
        raise ValueError(f"different tasks: {baseline.eval.task!r} vs {candidate.eval.task!r}")
    return compare_records(
        per_sample(baseline),
        per_sample(candidate),
        alpha=alpha,
        fdr=fdr,
        n_resamples=n_resamples,
        seed=seed,
    )


def render(c: Comparison, baseline: EvalLog, candidate: EvalLog) -> str:
    o = c.overall
    conf = round((1 - c.alpha) * 100)
    lines = [
        f"Baseline:  {baseline.eval.model}  ({baseline.location})",
        f"Candidate: {candidate.eval.model}  ({candidate.location})",
        "",
        f"VERDICT: {c.verdict.value.upper()}",
        f"  pass rate {o.baseline_rate:.3f} -> {o.candidate_rate:.3f}"
        f"  (difference {o.difference:+.3f}, {conf}% CI [{o.low:+.3f}, {o.high:+.3f}])",
        f"  {o.method}, {o.n_samples} samples, p = {o.p_value:.4g}, alpha = {c.alpha}",
        "",
        f"{'category':<10} {'base':>6} {'cand':>6} {'diff':>7} {'p':>8} {'p(BH)':>8}",
    ]
    for cr in c.categories:
        t = cr.test
        flag = "  <-" if cr.flagged else ""
        lines.append(
            f"{cr.category:<10} {t.baseline_rate:>6.3f} {t.candidate_rate:>6.3f} "
            f"{t.difference:>+7.3f} {t.p_value:>8.4f} {cr.p_adjusted:>8.4f}{flag}"
        )
    if c.verdict is Verdict.INCONCLUSIVE:
        lines += ["", "Inconclusive: no significant difference at this sample size."]
    return "\n".join(lines)
