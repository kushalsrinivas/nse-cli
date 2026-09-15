"""L2 — volatility, with the horizon stated every time.

Two hard-won rules are encoded here, both of them audit findings:

**Never mix horizons.** The old engine compared a calendar-scaled VIX move
against the standard deviation of overnight *gaps only*, and separately
annualised a gap sigma by sqrt(252) as if it covered a whole session. Gaps
are 43.6% of daily variance in this sample, so that understated vol by
about 1.5x and made every option read "expensive" as a constant. Each
quantity below carries its horizon in its name and `Horizon` converts
between them explicitly.

**Do not claim an edge you do not have.** Measured on 2021-2026, a ridge on
the full feature set forecasts the session range better than trailing
realised vol (MAE 0.337 vs 0.381, CI excludes 0) but *not* better than
India VIX alone (delta -0.004, CI [-0.022, +0.014]). VIX already contains
what our features know. So the production forecast is VIX-anchored, and the
model only supplies the shape adjustment VIX cannot: the split between the
overnight gap and the session that follows it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252.0
#: E|N(0,1)| — converts a single |move| observation into a sigma estimate.
E_ABS_N = math.sqrt(2.0 / math.pi)
#: Parkinson: sigma = range / (2*sqrt(ln 2)). Intraday diffusion ONLY —
#: it excludes the overnight gap, so it is never comparable to VIX directly.
PARKINSON = 1.0 / (2.0 * math.sqrt(math.log(2.0)))

#: Share of close-to-close variance that arrives in the overnight gap,
#: measured on this repo's 5y sample (gap var / c2c var = 0.436).
GAP_VARIANCE_SHARE = 0.436


@dataclass(frozen=True)
class VolForecast:
    """One night's volatility view. Every field names its own horizon."""

    sigma_c2c_pct: float          # close-to-close, the full 24h an option covers
    sigma_gap_pct: float          # previous close -> next open
    sigma_session_pct: float      # open -> close
    expected_range_pct: float     # expected high-low as % of open
    implied_sigma_c2c_pct: float  # VIX/sqrt(252) — same 24h, comparable
    vix_level: float
    source: str = "vix-anchored"

    @property
    def variance_ratio(self) -> float:
        """Implied / forecast. > 1 means the chain charges more than we expect."""
        return (self.implied_sigma_c2c_pct / self.sigma_c2c_pct
                if self.sigma_c2c_pct > 1e-9 else float("nan"))

    def scaled(self, horizon: str) -> float:
        return {"c2c": self.sigma_c2c_pct, "gap": self.sigma_gap_pct,
                "session": self.sigma_session_pct}[horizon]


def vix_to_daily_sigma(vix_level: float) -> float:
    """India VIX (annualised %, 365-day convention) -> 1-session sigma %.

    Trading days, not calendar days: the variance accrues when the market
    is open. Using sqrt(365) here is a common and expensive error.
    """
    return float(max(vix_level, 1.0) / math.sqrt(TRADING_DAYS))


def realized_sigma_from_abs_move(abs_move_pct: float) -> float:
    """Unbiased sigma from a single |return| observation."""
    return float(abs_move_pct / E_ABS_N)


def parkinson_sigma(range_pct: float) -> float:
    """Intraday sigma from the high-low range. Excludes the gap — do not
    compare this to VIX without adding the gap variance back."""
    return float(range_pct * PARKINSON)


def split_sigma(sigma_c2c_pct: float,
                gap_share: float = GAP_VARIANCE_SHARE) -> tuple[float, float]:
    """Split a 24h sigma into (gap, session) components by variance share."""
    share = min(max(gap_share, 0.01), 0.99)
    var = sigma_c2c_pct ** 2
    return (math.sqrt(var * share), math.sqrt(var * (1.0 - share)))


def blend_forecast(vix_level: float, realized_sigma_20d_pct: float | None = None,
                   *, vix_weight: float = 0.75,
                   gap_share: float = GAP_VARIANCE_SHARE) -> VolForecast:
    """The production volatility view.

    VIX carries most of the weight because it measurably wins: our feature
    set adds nothing to it out of sample. Trailing realised gets the
    remainder as an anchor against a stale or dislocated VIX print, not
    because it is expected to add information.
    """
    implied = vix_to_daily_sigma(vix_level)
    if realized_sigma_20d_pct and realized_sigma_20d_pct > 0:
        rv_daily = realized_sigma_20d_pct / math.sqrt(TRADING_DAYS)
        w = min(max(vix_weight, 0.0), 1.0)
        sigma = w * implied + (1 - w) * rv_daily
    else:
        sigma = implied
    gap_s, sess_s = split_sigma(sigma, gap_share)
    # E[range] for a Brownian session with sigma_session; the 1/PARKINSON
    # factor inverts the Parkinson relation.
    exp_range = sess_s / PARKINSON
    return VolForecast(
        sigma_c2c_pct=round(sigma, 4),
        sigma_gap_pct=round(gap_s, 4),
        sigma_session_pct=round(sess_s, 4),
        expected_range_pct=round(exp_range, 4),
        implied_sigma_c2c_pct=round(implied, 4),
        vix_level=round(float(vix_level), 2),
    )


# ---------------------------------------------------------------------------
# Volatility risk premium
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolPremium:
    """How rich options have been, measured — not assumed."""

    ratio: float                  # mean implied sigma / mean realised sigma
    mean_implied_pct: float
    mean_realized_pct: float
    ci_low: float                 # CI on the mean difference (implied - realised)
    ci_high: float
    p_realized_exceeds: float     # P(|move| > 1 implied sigma); 0.317 if fair
    n: int

    @property
    def options_are_rich(self) -> bool:
        return self.ci_low > 0

    @property
    def long_premium_hurdle_pct(self) -> float:
        """How much a long-premium trade must overcome before it is even.

        Buying options means paying this premium every night. It is the
        single most robust number in the dataset and the old engine, which
        only ever bought premium, was on the wrong side of it.
        """
        return round((self.ratio - 1.0) * 100, 1)


def measure_vol_premium(abs_c2c_pct, vix_levels, *, block: int = 10) -> VolPremium:
    """Realised vs implied over a history. Both on the same 24h horizon."""
    from model.forecast.evaluate import block_bootstrap_ci

    a = np.asarray(abs_c2c_pct, dtype=float)
    v = np.asarray(vix_levels, dtype=float)
    m = np.isfinite(a) & np.isfinite(v)
    a, v = a[m], v[m]
    if len(a) < 30:
        return VolPremium(float("nan"), float("nan"), float("nan"),
                          float("nan"), float("nan"), float("nan"), len(a))
    implied = v / math.sqrt(TRADING_DAYS)
    realized = a / E_ABS_N
    lo, hi = block_bootstrap_ci(implied - realized, block=block)
    return VolPremium(
        ratio=round(float(implied.mean() / realized.mean()), 4),
        mean_implied_pct=round(float(implied.mean()), 4),
        mean_realized_pct=round(float(realized.mean()), 4),
        ci_low=round(lo, 5), ci_high=round(hi, 5),
        p_realized_exceeds=round(float((a > implied).mean()), 4),
        n=int(len(a)))
