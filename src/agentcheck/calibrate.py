"""Calibration: measure AgentCheck's false-alarm and detection rates by resampling.

Fresh model runs are too expensive on a free tier, so runs are simulated from a fixed
*pool* of logged epochs: for each sample, a simulated run draws its epochs (with
replacement) from that sample's pooled outcomes. This treats each sample's pooled pass
rate as its true rate, keeps samples as the independent unit, and keeps every sample's
own difficulty.

- A/A: baseline and candidate both drawn from the same pool. Any verdict other than
  INCONCLUSIVE is a false alarm. Target: at most 5% at alpha = 0.05.
- Injected drop: the candidate additionally has a random subset of samples broken
  (every epoch fails), sized so the expected pass rate falls by `drop`. Target: at least
  80% REGRESSION verdicts for a 10-point drop.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from inspect_ai.log import EvalLog

from agentcheck.compare import Verdict, decide
from agentcheck.stats.intervals import Interval, wilson_interval
from agentcheck.summary import per_sample

Pool = Mapping[str, Sequence[bool]]
Run = dict[str, list[bool]]


def check_pool(pool: Pool) -> None:
    if not pool:
        raise ValueError("empty pool")
    for sample_id, outcomes in pool.items():
        if len(outcomes) < 2:
            raise ValueError(
                f"sample {sample_id!r} has {len(outcomes)} pooled epoch(s); need >= 2, "
                "or every simulated run would be identical"
            )


def draw_run(pool: Pool, epochs: int, rng: random.Random) -> Run:
    return {k: rng.choices(list(v), k=epochs) for k, v in sorted(pool.items())}


def pool_rate(pool: Pool) -> float:
    return fmean(fmean(v) for v in pool.values())


def break_samples(run: Run, probability: float, rng: random.Random) -> Run:
    """Each sample independently fails in every epoch with `probability`."""
    return {k: [False] * len(v) if rng.random() < probability else list(v) for k, v in run.items()}


@dataclass(frozen=True)
class RateEstimate:
    hits: int
    reps: int
    interval: Interval  # 95% Wilson

    @property
    def rate(self) -> float:
        return self.hits / self.reps


@dataclass(frozen=True)
class Calibration:
    n_samples: int
    pool_epochs: int
    run_epochs: int
    pool_pass_rate: float
    reps: int
    alpha: float
    false_alarms: RateEstimate  # A/A runs with any non-inconclusive verdict
    drop: float
    detections: RateEstimate  # injected-drop runs with a REGRESSION verdict


def estimate(hits: int, reps: int) -> RateEstimate:
    return RateEstimate(hits, reps, wilson_interval(hits, reps))


def calibrate(
    pool: Pool,
    *,
    run_epochs: int = 1,
    drop: float = 0.10,
    reps: int = 1000,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Calibration:
    check_pool(pool)
    if run_epochs < 1 or reps < 1:
        raise ValueError("run_epochs and reps must be >= 1")
    base_rate = pool_rate(pool)
    if not 0.0 < drop <= base_rate:
        raise ValueError(f"drop must be in (0, pool pass rate {base_rate:.3f}], got {drop}")
    rng = random.Random(seed)
    false_alarms = detections = 0
    for rep in range(reps):
        base, same = draw_run(pool, run_epochs, rng), draw_run(pool, run_epochs, rng)
        v = decide(base, same, alpha=alpha, n_resamples=n_resamples, seed=rep)
        false_alarms += v is not Verdict.INCONCLUSIVE

        worse = break_samples(draw_run(pool, run_epochs, rng), drop / base_rate, rng)
        v = decide(base, worse, alpha=alpha, n_resamples=n_resamples, seed=rep)
        detections += v is Verdict.REGRESSION
    return Calibration(
        n_samples=len(pool),
        pool_epochs=min(len(v) for v in pool.values()),
        run_epochs=run_epochs,
        pool_pass_rate=base_rate,
        reps=reps,
        alpha=alpha,
        false_alarms=estimate(false_alarms, reps),
        drop=drop,
        detections=estimate(detections, reps),
    )


def pool_from_logs(logs: Sequence[EvalLog]) -> Run:
    """Concatenate every epoch of every log, per sample. All logs must cover the same samples."""
    if not logs:
        raise ValueError("no logs")
    pool: Run = {}
    for log in logs:
        records = per_sample(log)
        if pool and records.keys() != pool.keys():
            raise ValueError(f"{log.location} covers different samples from the first log")
        for k, rec in records.items():
            pool.setdefault(k, []).extend(rec.passes)
    return pool


def synthetic_pool(
    n_samples: int = 60, epochs: int = 5, mean: float = 0.8, spread: float = 4.0, seed: int = 0
) -> Run:
    """Per-sample pass probabilities from Beta(mean * spread, (1 - mean) * spread).

    A small `spread` gives mostly always-pass / always-fail samples; a large one makes
    every sample flaky around `mean`.
    """
    rng = random.Random(seed)
    a, b = mean * spread, (1 - mean) * spread
    ps = [rng.betavariate(a, b) for _ in range(n_samples)]
    return {f"s{i:03}": [rng.random() < p for _ in range(epochs)] for i, p in enumerate(ps)}


def render(c: Calibration, label: str) -> str:
    def line(name: str, r: RateEstimate, target: str) -> str:
        return (
            f"  {name:<22} {r.hits:>5}/{r.reps:<5} = {r.rate:.3f}  "
            f"95% CI [{r.interval.low:.3f}, {r.interval.high:.3f}]   target {target}"
        )

    return "\n".join(
        [
            f"Calibration: {label}",
            f"  pool: {c.n_samples} samples x {c.pool_epochs}+ epochs, pass rate "
            f"{c.pool_pass_rate:.3f}; simulated runs: {c.run_epochs} epoch(s); alpha {c.alpha}",
            line("A/A false alarms", c.false_alarms, "<= 0.05"),
            line(f"detect -{c.drop:.2f} drop", c.detections, ">= 0.80"),
        ]
    )
