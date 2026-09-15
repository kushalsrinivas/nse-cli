"""L3 — the conditional distribution of the next move.

Shaped by what the evaluation actually found, not by what would look
impressive:

**Scale is conditional, location mostly is not.** Quantile regression on
the tradeable targets did not beat unconditional quantiles out of sample
(pinball delta +0.002 to +0.003, CI straddling zero at both decision
points). Volatility, by contrast, is genuinely forecastable — VIX does it.
So the distribution is built as `location + scale x shape`, where scale
comes from L2 and location is left at zero unless a model that actually
beat its baseline supplies one. At PREOPEN that model exists for the gap
(AUC 0.783, Brier skill +0.213); for the session it does not, and the
location stays at zero rather than inventing a drift.

**Shape is empirical.** Standardised historical moves, not a Gaussian.
Index returns have fat tails, and a normal assumption understates exactly
the scenario an option buyer is paying for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@dataclass(frozen=True)
class EmpiricalShape:
    """Standardised historical moves — the distribution's shape, unitless."""

    samples: tuple[float, ...]
    n: int
    kurtosis: float
    skew: float

    @classmethod
    def from_history(cls, moves_pct, sigma_pct) -> EmpiricalShape:
        m = np.asarray(moves_pct, dtype=float)
        s = np.asarray(sigma_pct, dtype=float)
        ok = np.isfinite(m) & np.isfinite(s) & (s > 1e-9)
        z = m[ok] / s[ok]
        if len(z) < 50:               # too little to shape; fall back to normal
            z = np.random.default_rng(0).normal(size=2000)
        z = z - np.median(z)          # centre without letting outliers drag it
        sd = float(np.std(z)) or 1.0
        z = z / sd                    # unit scale, so `scale` means what it says
        return cls(samples=tuple(z.tolist()), n=int(len(z)),
                   kurtosis=round(float(np.mean(z ** 4)), 3),
                   skew=round(float(np.mean(z ** 3)), 3))

    @classmethod
    def normal(cls, n: int = 4000) -> EmpiricalShape:
        z = np.random.default_rng(0).normal(size=n)
        return cls(tuple(z.tolist()), n, 3.0, 0.0)

    @property
    def fat_tailed(self) -> bool:
        return self.kurtosis > 3.5


@dataclass
class ConditionalDistribution:
    """A forecast distribution in percent, with its provenance attached."""

    location_pct: float
    scale_pct: float
    horizon: str                       # "gap" | "session" | "c2c"
    shape: EmpiricalShape
    location_source: str = "zero (no model beat its baseline)"
    _draws: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        z = np.asarray(self.shape.samples, dtype=float)
        self._draws = self.location_pct + self.scale_pct * z

    # -- summary -----------------------------------------------------------

    def quantiles(self, qs=QUANTILES) -> dict[str, float]:
        vals = np.percentile(self._draws, [q * 100 for q in qs])
        return {f"p{int(q * 100)}": round(float(v), 4)
                for q, v in zip(qs, vals, strict=False)}

    @property
    def mean_pct(self) -> float:
        return round(float(self._draws.mean()), 4)

    @property
    def median_pct(self) -> float:
        return round(float(np.median(self._draws)), 4)

    def p_up(self) -> float:
        return round(float((self._draws > 0).mean()), 4)

    def p_beyond(self, move_pct: float) -> float:
        """P(move further than `move_pct` in that direction), signed."""
        if move_pct >= 0:
            return round(float((self._draws > move_pct).mean()), 4)
        return round(float((self._draws < move_pct).mean()), 4)

    def p_abs_beyond(self, size_pct: float) -> float:
        """P(|move| > size) — the number that matters for a straddle."""
        return round(float((np.abs(self._draws) > abs(size_pct)).mean()), 4)

    def expected_shortfall(self, alpha: float = 0.05) -> float:
        """Mean of the worst `alpha` tail — the loss to size against."""
        cut = np.percentile(self._draws, alpha * 100)
        tail = self._draws[self._draws <= cut]
        return round(float(tail.mean()) if len(tail) else float(cut), 4)

    def draws(self) -> np.ndarray:
        return self._draws.copy()

    def describe(self) -> str:
        q = self.quantiles()
        return (f"{self.horizon}: loc {self.location_pct:+.3f}% "
                f"scale {self.scale_pct:.3f}% "
                f"[p10 {q['p10']:+.2f}, p50 {q['p50']:+.2f}, p90 {q['p90']:+.2f}] "
                f"{'fat-tailed' if self.shape.fat_tailed else 'near-normal'} "
                f"(kurt {self.shape.kurtosis})")


def build_distribution(scale_pct: float, horizon: str, shape: EmpiricalShape,
                       location_pct: float = 0.0,
                       location_source: str = "zero (no model beat its baseline)"
                       ) -> ConditionalDistribution:
    return ConditionalDistribution(
        location_pct=float(location_pct), scale_pct=float(max(scale_pct, 1e-6)),
        horizon=horizon, shape=shape, location_source=location_source)


def implied_distribution(implied_sigma_pct: float, horizon: str,
                         shape: EmpiricalShape | None = None
                         ) -> ConditionalDistribution:
    """What the option chain is pricing, as a distribution we can compare to.

    Risk-neutral, so the location is zero by construction: the market does
    not take a directional view, it prices one.
    """
    return build_distribution(
        implied_sigma_pct, horizon, shape or EmpiricalShape.normal(),
        location_pct=0.0, location_source="risk-neutral (implied)")
