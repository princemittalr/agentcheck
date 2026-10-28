"""Power by simulation: how many samples and epochs it takes to catch a given drop.

Uses the calibration machinery. A suite with more samples than the pool is simulated by
drawing samples with replacement from the pool, i.e. new questions as hard as the
existing ones on average. Each grid cell reports the detection rate of the injected
drop; the false-alarm rate needs no grid, because the tests are exact (<= alpha).
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from agentcheck.calibrate import (
    Pool,
    RateEstimate,
    Run,
    break_samples,
    check_pool,
    draw_run,
    estimate,
    pool_rate,
)
from agentcheck.compare import Verdict, decide


@dataclass(frozen=True)
class PowerCell:
    n_samples: int
    epochs: int
    detections: RateEstimate

    @property
    def cost(self) -> int:
        """Sample-epochs per run: what a model provider bills, roughly."""
        return self.n_samples * self.epochs


def expand_pool(pool: Pool, n_samples: int, rng: random.Random) -> Run:
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    keys = sorted(pool)
    picked = keys if n_samples == len(keys) else rng.choices(keys, k=n_samples)
    return {f"{k}#{i}": list(pool[k]) for i, k in enumerate(picked)}


def power_grid(
    pool: Pool,
    sample_counts: Sequence[int],
    epoch_counts: Sequence[int],
    *,
    drop: float = 0.10,
    reps: int = 300,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> list[PowerCell]:
    check_pool(pool)
    base_rate = pool_rate(pool)
    if not 0.0 < drop <= base_rate:
        raise ValueError(f"drop must be in (0, pool pass rate {base_rate:.3f}], got {drop}")
    cells = []
    for n in sample_counts:
        for e in epoch_counts:
            rng = random.Random(f"{seed}-{n}-{e}")
            hits = 0
            for rep in range(reps):
                suite = expand_pool(pool, n, rng)
                rate = pool_rate(suite)
                base = draw_run(suite, e, rng)
                worse = break_samples(draw_run(suite, e, rng), min(1.0, drop / rate), rng)
                verdict = decide(base, worse, alpha=alpha, n_resamples=n_resamples, seed=rep)
                hits += verdict is Verdict.REGRESSION
            cells.append(PowerCell(n, e, estimate(hits, reps)))
    return cells


def cheapest(cells: Sequence[PowerCell], target: float = 0.80) -> PowerCell | None:
    """Fewest sample-epochs whose 95% interval lies entirely at or above `target`."""
    good = [c for c in cells if c.detections.interval.low >= target]
    return min(good, key=lambda c: (c.cost, c.epochs), default=None)


def render(cells: Sequence[PowerCell], drop: float, label: str, target: float = 0.80) -> str:
    epochs = sorted({c.epochs for c in cells})
    lines = [
        f"Power to detect a -{drop:.2f} drop: {label}",
        f"(rows: samples per run; columns: epochs; detection rate, {cells[0].detections.reps} "
        "reps each)",
        "",
        f"{'samples':>8} " + " ".join(f"{e:>7}ep" for e in epochs),
    ]
    by_n: dict[int, dict[int, PowerCell]] = {}
    for c in cells:
        by_n.setdefault(c.n_samples, {})[c.epochs] = c
    for n in sorted(by_n):
        row = by_n[n]
        lines.append(f"{n:>8} " + " ".join(f"{row[e].detections.rate:>9.3f}" for e in epochs))
    best = cheapest(cells, target)
    lines.append("")
    if best is None:
        lines.append(f"No design in this grid reaches {target:.0%} with confidence.")
    else:
        ci = best.detections.interval
        lines.append(
            f"Cheapest design reaching {target:.0%}: {best.n_samples} samples x {best.epochs} "
            f"epochs = {best.cost} sample-epochs per run (detection {best.detections.rate:.3f}, "
            f"95% CI [{ci.low:.3f}, {ci.high:.3f}])"
        )
    return "\n".join(lines)
