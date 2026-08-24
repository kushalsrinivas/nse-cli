"""Overnight option-buying strategy.

Your play: buy a CE/PE near today's close, expecting tomorrow's open to
gap/continue in your favor. This module answers the only question that
matters for it, empirically:

    "After days that looked like TODAY, what did NIFTY do by the next open?"

It replays history day by day with the same composite model used live,
records every qualifying signal's next-open / next-close outcome, and
buckets results by the conditions that could plausibly drive an
overnight edge:

  * close location inside the day's range (strong close = fuel for gaps)
  * alignment with the EMA-50 trend (with-trend vs counter-trend)
  * relative volume (participation behind the move)
  * regime (trending vs sideways vs high vol)
  * score band (65-74 vs 75+)

The overnight cost side is explicit: an option held overnight pays one
session of theta plus spread. The setup card converts the historical gap
distribution into an estimated premium return via delta/theta so "the
index gaps my way" is never confused with "my option makes money".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.signals import Direction
from config import SETTINGS
from model.backtest import _base_frame
from model.composite import MIN_TRADEABLE_CONFIDENCE, compute_composite
from model.indicators import assess_all
from model.regime import detect_regime
from model.weights import compute_effective_weights, load_learned_weights


@dataclass(frozen=True)
class OvernightSignal:
    timestamp: object
    direction: Direction
    score: float
    regime: str
    entry_close: float          # buy point (day T close)
    next_open: float            # exit candidate (day T+1 open)
    next_close: float           # alternative exit (day T+1 close)
    next_low: float
    next_high: float
    gap_pct: float              # (next_open - entry) / entry * 100, signed long-basis
    close_move_pct: float       # same through T+1 close
    close_pos: float            # where T closed within its own range (0-1)
    rel_volume: float
    with_trend: bool
    next_timestamp: object = None   # T+1 session date (for weekend/expiry filters)

    @property
    def calendar_gap_days(self) -> int:
        """Calendar days held; >1 means a weekend/holiday in between."""
        if self.next_timestamp is None:
            return 1
        return (self.next_timestamp - self.timestamp).days


@dataclass
class BucketStats:
    name: str
    n: int = 0
    wins_open: int = 0
    avg_gap_pct: float = 0.0
    median_gap_pct: float = 0.0
    avg_close_move_pct: float = 0.0
    p90_gap: float = 0.0
    p10_gap: float = 0.0
    worst_gap: float = 0.0

    @property
    def win_rate_open(self) -> float:
        return self.wins_open / self.n if self.n else 0.0

    def as_row(self) -> str:
        return (f"{self.name:<34} {self.n:>4}  "
                f"{self.win_rate_open * 100:5.1f}%  "
                f"{self.avg_gap_pct:+.3f}%  {self.median_gap_pct:+.3f}%  "
                f"[{self.p10_gap:+.2f}, {self.p90_gap:+.2f}]  "
                f"worst {self.worst_gap:+.2f}%")


def collect_overnight_signals(candles, settings=SETTINGS,
                              min_score: float = MIN_TRADEABLE_CONFIDENCE,
                              learned=None) -> list[OvernightSignal]:
    """Replay history; every bar scoring >= min_score becomes a signal."""
    frame = _base_frame(candles)
    n = len(frame)
    signals: list[OvernightSignal] = []

    for i in range(200, n - 1):
        window = frame.iloc[: i + 1]
        try:
            regime = detect_regime(window)
            assessments = assess_all(window)
        except Exception:
            continue
        if not assessments:
            continue
        weights = compute_effective_weights(regime.regime,
                                            {a.group for a in assessments},
                                            learned or load_learned_weights())
        comp = compute_composite(assessments, weights, regime)
        if comp.score < min_score or comp.direction is Direction.NEUTRAL:
            continue

        row, nxt = frame.iloc[i], frame.iloc[i + 1]
        entry, op = float(row["close"]), float(nxt["open"])
        rng = float(row["high"]) - float(row["low"])
        e9, e50 = float(row["ema9"]), float(row["ema50"])
        sign = 1 if comp.direction is Direction.BULLISH else -1
        signals.append(OvernightSignal(
            timestamp=frame.index[i],
            direction=comp.direction,
            score=comp.score,
            regime=regime.regime.value,
            entry_close=entry,
            next_open=op,
            next_close=float(nxt["close"]),
            next_low=float(nxt["low"]),
            next_high=float(nxt["high"]),
            gap_pct=sign * (op - entry) / entry * 100,
            close_move_pct=sign * (float(nxt["close"]) - entry) / entry * 100,
            close_pos=(float(row["close"]) - float(row["low"])) / rng if rng > 0 else 0.5,
            rel_volume=float(row.get("rel_volume", np.nan)),
            with_trend=(e9 > e50) == (sign > 0),
            next_timestamp=frame.index[i + 1],
        ))
    return signals


# ---------------------------------------------------------------------------
# Discipline filters
# ---------------------------------------------------------------------------

# NIFTY weekly expiry weekday by era: Thursday until NSE moved index
# derivatives to Tuesday effective Sep 2025. (weekday(): Mon=0)
_EXPIRY_ERAS = ((pd.Timestamp("2025-09-01"), 1), (pd.Timestamp("1970-01-01"), 3))


def _expiry_weekday(ts) -> int:
    for start, wd in _EXPIRY_ERAS:
        if ts >= start:
            return wd
    return 3


def _is_last_expiry_of_month(ts) -> bool:
    """Heuristic: an expiry weekday in the final 7 days of the month is the
    monthly contract; earlier ones are weeklies."""
    nxt = ts + pd.Timedelta(days=7)
    return nxt.month != ts.month


def classify_next_expiry(next_ts) -> str:
    """'monthly' | 'weekly' — for labelling the gamma risk of tonight's hold."""
    wd = _expiry_weekday(next_ts)
    # Walk forward to that weekday
    delta = (wd - next_ts.weekday()) % 5 or 5
    exp_ts = next_ts + pd.Timedelta(days=delta)
    return "monthly" if _is_last_expiry_of_month(exp_ts) else "weekly"


def apply_discipline(signals: list[OvernightSignal], *,
                     skip_weekends: bool = True,
                     skip_hold_into_expiry: bool = True,
                     skip_entry_on_expiry: bool = True) -> list[OvernightSignal]:
    """Trading-discipline filter.

    - skip_weekends: no entries whose exit lands after a weekend/holiday
      (calendar gap > 1 day) — extra decay days with no edge.
    - skip_hold_into_expiry: never buy at close when the NEXT session is a
      weekly expiry — that overnight is max gamma/theta.
    - skip_entry_on_expiry: don't initiate on expiry day itself.
    """
    out = []
    for s in signals:
        if skip_weekends and s.calendar_gap_days > 1:
            continue
        if skip_hold_into_expiry and s.next_timestamp is not None \
                and s.next_timestamp.weekday() == _expiry_weekday(s.next_timestamp):
            continue
        if skip_entry_on_expiry and s.timestamp.weekday() == _expiry_weekday(s.timestamp):
            continue
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Condition buckets
# ---------------------------------------------------------------------------

_FILTERS: tuple[tuple[str, object], ...] = (
    ("ALL SIGNALS", lambda s: True),
    ("score >= 75 (high conviction)", lambda s: s.score >= 75),
    ("strong close (pos > 0.7 / < 0.3)",
     lambda s: s.close_pos > 0.7 if s.direction is Direction.BULLISH else s.close_pos < 0.3),
    ("weak close (faded into bell)", lambda s: 0.35 < s.close_pos < 0.65),
    ("with EMA-50 trend", lambda s: s.with_trend),
    ("counter-trend", lambda s: not s.with_trend),
    ("rel volume >= 1.3x", lambda s: not np.isnan(s.rel_volume) and s.rel_volume >= 1.3),
    ("rel volume < 0.8x (thin day)", lambda s: np.isnan(s.rel_volume) or s.rel_volume < 0.8),
    ("trending regime", lambda s: "trending" in s.regime),
    ("sideways regime", lambda s: s.regime == "sideways"),
    ("high volatility regime", lambda s: s.regime == "high_volatility"),
)


def bucket_stats(signals: list[OvernightSignal]) -> list[BucketStats]:
    out = []
    for name, pred in _FILTERS:
        subset = [s for s in signals if pred(s)]
        out.append(_stats_for(name, subset))
    return out


def combo_stats(signals: list[OvernightSignal], min_n: int = 8) -> list[BucketStats]:
    """Intersections most likely to carry an edge: strong close + volume +
    trend alignment."""
    combos = {
        "strong close + with-trend": lambda s: (
            (s.close_pos > 0.7 if s.direction is Direction.BULLISH else s.close_pos < 0.3)
            and s.with_trend),
        "strong close + vol>=1.3 + with-trend": lambda s: (
            (s.close_pos > 0.7 if s.direction is Direction.BULLISH else s.close_pos < 0.3)
            and s.with_trend and not np.isnan(s.rel_volume) and s.rel_volume >= 1.3),
        "score>=75 + strong close": lambda s: (
            s.score >= 75
            and (s.close_pos > 0.7 if s.direction is Direction.BULLISH else s.close_pos < 0.3)),
    }
    out = []
    for name, pred in combos.items():
        st = _stats_for(name, [s for s in signals if pred(s)])
        if st.n >= min_n:
            out.append(st)
    return sorted(out, key=lambda b: -b.avg_gap_pct)


def _stats_for(name: str, subset: list[OvernightSignal]) -> BucketStats:
    if not subset:
        return BucketStats(name=name)
    gaps = np.array([s.gap_pct for s in subset])
    return BucketStats(
        name=name,
        n=len(subset),
        wins_open=int((gaps > 0).sum()),
        avg_gap_pct=float(gaps.mean()),
        median_gap_pct=float(np.median(gaps)),
        avg_close_move_pct=float(np.mean([s.close_move_pct for s in subset])),
        p90_gap=float(np.percentile(gaps, 90)),
        p10_gap=float(np.percentile(gaps, 10)),
        worst_gap=float(gaps.min()),
    )


# ---------------------------------------------------------------------------
# Premium economics: index gap → option P&L
# ---------------------------------------------------------------------------

def premium_outlook(gaps, spot: float, atm_iv_pct: float,
                    dte_days: int, settings=SETTINGS) -> dict:
    """Translate a historical gap distribution into expected premium moves.

    premium_pnl ≈ delta × spot_move − theta × 1 session (held one night).
    Uses an ATM-ish delta approximation; exact numbers come from the chosen
    contract at scan time.
    """
    from model.options_scan import bs_greeks
    if len(gaps) == 0:
        return {}
    iv = max(atm_iv_pct, 1.0) / 100.0
    greeks = bs_greeks(spot, round(spot), dte_days, iv, is_call=True)
    delta = abs(greeks["delta"])
    theta = abs(greeks["theta"])
    # Rough ATM premium estimate via 1σ to expiry.
    est_premium = spot * iv * math.sqrt(max(dte_days, 1) / 365.0) * 0.8
    if est_premium <= 0:
        return {}
    breakeven_gap = theta / (delta * spot) * 100

    def prem_return(move_pct: float) -> float:
        rupees = delta * spot * move_pct / 100 - theta
        return rupees / est_premium * 100

    gaps = np.asarray(list(gaps))
    return {
        "est_atm_premium": round(est_premium, 1),
        "delta_used": delta,
        "theta_overnight": round(theta, 2),
        "breakeven_gap_pct": round(breakeven_gap, 3),
        "avg_prem_return_pct": round(prem_return(float(gaps.mean())), 2),
        "median_prem_return_pct": round(prem_return(float(np.median(gaps))), 2),
        "p90_prem_return_pct": round(prem_return(float(np.percentile(gaps, 90))), 2),
        "worst_prem_return_pct": round(prem_return(float(gaps.min())), 2),
        "prem_win_prob": round(float((gaps > breakeven_gap).mean()), 3),
    }


def format_research(signals: list[OvernightSignal]) -> str:
    lines = [
        "Overnight research — what happened after qualifying closes",
        "(exit at next open; gap % signed toward the signal direction)",
        "",
        f"{'condition':<34} {'n':>4}  {'win%':>5}  {'avg':>6}  {'med':>6}  "
        f"{'[p10,p90]':>15}  {'worst':>7}",
        "-" * 92,
    ]
    for st in bucket_stats(signals):
        lines.append(st.as_row())
    combos = combo_stats(signals)
    if combos:
        lines.append("")
        lines.append("Best condition combinations:")
        for st in combos:
            lines.append(st.as_row())
    return "\n".join(lines)
