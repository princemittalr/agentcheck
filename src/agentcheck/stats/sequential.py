"""Group-sequential testing: look at the comparison after each batch of epochs, stop early.

Testing at the full alpha after every batch inflates false alarms (5 looks at 0.05 give
roughly 0.14). Instead the total alpha is *spent* across the planned looks with the
Lan-DeMets O'Brien-Fleming-type function

    spent(t) = 2 * (1 - Phi(z_{1 - alpha/2} / sqrt(t))),   t = fraction of epochs run,

which spends almost nothing early and nearly all of alpha at the last look. Look k tests
at the increment spent(t_k) - spent(t_{k-1}). The increments sum to alpha, so by the union
bound the chance of any false alarm is at most alpha whatever the correlation between
looks. That is conservative (exact boundaries would use the correlation), but needs no
normality assumption and works with the exact tests used here.
"""

import math
from itertools import pairwise
from statistics import NormalDist

_N = NormalDist()


def obf_spent(t: float, alpha: float = 0.05) -> float:
    """Alpha spent by information fraction t in (0, 1]."""
    if not 0.0 < t <= 1.0:
        raise ValueError(f"t must be in (0, 1], got {t}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if t == 1.0:
        return alpha  # exact at the end, not 2 * (1 - Phi(z)) with rounding
    return 2.0 * (1.0 - _N.cdf(_N.inv_cdf(1.0 - alpha / 2.0) / math.sqrt(t)))


def look_alphas(looks: int, alpha: float = 0.05) -> list[float]:
    """Nominal significance level for each of `looks` equally spaced looks."""
    if looks < 1:
        raise ValueError(f"looks must be >= 1, got {looks}")
    spent = [obf_spent(k / looks, alpha) for k in range(1, looks + 1)]
    return [spent[0]] + [b - a for a, b in pairwise(spent)]
