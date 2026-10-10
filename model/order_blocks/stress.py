"""Overnight options stress engine.

An overnight option position is exposed to four things at once between
15:20 and the next open: the gap, the IV change, a night (or weekend) of
decay, and an opening spread wider than the one it was entered at. A
single "predicted option price" hides all of that. This module reprices
the exact structure across a grid of next-open outcomes instead:

    underlying move   −3σ … +3σ of the VIX-implied gap sigma (signed in the
                      trade's favour), plus named scenarios
    IV change         −2 … +5 vol points (the measured weekday change is the
                      centre of the IV weighting; Friday holds lean up)
    time              DTE at the next session's 09:15 (weekends included)
    exit price        bid for long legs / ask for short legs, with the half
                      spread widened by `open_spread_mult` for the open

Outputs, all per unit and per lot after round-trip charges:
    expected_pnl      probability-weighted over the grid (normal gap, normal IV)
    expected_loss     E[min(P&L, 0)] — the average size of the bad outcomes
    stress_loss       worst of the named adverse scenarios — used for sizing
    worst             the single worst grid cell, with its move and IV shift
    breakeven_move    underlying move (points) where the trade is flat at the
                      measured IV change

Scenarios are illustrations of what the position does, not forecasts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from model.forecast.position import MEASURED_IV_CHANGE
from model.forecast.volatility import split_sigma, vix_to_daily_sigma
from model.options_ev import bs_price

MOVES = tuple(x / 2 for x in range(-6, 7))           # −3σ … +3σ in 0.5σ steps
IV_SHIFTS = (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0)          # vol points
IV_SHIFT_SD = 0.95                                    # sd of overnight VIX change (pts)


@dataclass(frozen=True)
class Scenario:
    name: str
    move_sigmas: float        # + = in the trade's favour
    iv_shift: float           # vol points
    adverse: bool


NAMED = (
    Scenario("favourable gap, IV falls", +1.0, -1.0, False),
    Scenario("flat open, IV falls", 0.0, -1.0, False),
    Scenario("flat open, IV unchanged", 0.0, 0.0, False),
    Scenario("adverse gap, IV rises", -1.5, +1.5, True),
    Scenario("volatility shock, against", -2.5, +5.0, True),
    Scenario("volatility shock, with", +2.5, +5.0, False),
    Scenario("tail gap against", -3.0, +3.0, True),
)


@dataclass
class ScenarioResult:
    name: str
    move_points: float
    iv_shift: float
    exit_value: float          # per unit, executable
    pnl_unit: float
    pnl_lot: float


@dataclass
class StressReport:
    entry_net: float
    lot: int
    sigma_gap_pct: float
    exit_dte: float
    scenarios: list[ScenarioResult] = field(default_factory=list)
    expected_pnl_lot: float = 0.0
    expected_loss_lot: float = 0.0
    stress_loss_unit: float = 0.0
    stress_loss_lot: float = 0.0
    stress_scenario: str = ""
    worst_lot: float = 0.0
    worst_move_points: float = 0.0
    worst_iv_shift: float = 0.0
    breakeven_move_points: float | None = None

    def to_dict(self) -> dict:
        return {
            "entry_net": self.entry_net, "lot": self.lot,
            "sigma_gap_pct": round(self.sigma_gap_pct, 4), "exit_dte": round(self.exit_dte, 3),
            "expected_pnl_lot": round(self.expected_pnl_lot, 1),
            "expected_loss_lot": round(self.expected_loss_lot, 1),
            "stress_loss_lot": round(self.stress_loss_lot, 1),
            "stress_scenario": self.stress_scenario,
            "worst_lot": round(self.worst_lot, 1),
            "worst_move_points": round(self.worst_move_points, 1),
            "worst_iv_shift": self.worst_iv_shift,
            "breakeven_move_points": (round(self.breakeven_move_points, 1)
                                      if self.breakeven_move_points is not None else None),
            "scenarios": [{"name": s.name, "move_points": round(s.move_points, 1),
                           "iv_shift": s.iv_shift, "pnl_lot": round(s.pnl_lot, 1)}
                          for s in self.scenarios],
        }


def next_open(now: datetime) -> datetime:
    """Next weekday 09:15 after `now` (exchange holidays are not modelled)."""
    d = now + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.replace(hour=9, minute=15, second=0, microsecond=0)


def _half_spread(q) -> float:
    if q.bid and q.ask and q.ask >= q.bid:
        return (q.ask - q.bid) / 2
    px = q.ltp or q.ask or q.bid or 0.0
    return max(px * 0.01, 0.05)


def _exit_prices(legs, sides, spot: float, dte: float, iv_shift: float,
                 open_spread_mult: float) -> list[float]:
    """Executable exit price per leg: longs sold at the bid, shorts bought
    back at the ask, with the opening half-spread widened."""
    out = []
    for q, s in zip(legs, sides, strict=True):
        iv = max((q.iv or 14.0) + iv_shift, 1.0) / 100.0
        mid = bs_price(spot, q.strike, max(dte, 0.01), iv, q.is_call)
        hs = _half_spread(q) * open_spread_mult
        out.append(max(mid - hs, 0.05) if s > 0 else mid + hs)
    return out


def _exit_value(legs, sides, spot: float, dte: float, iv_shift: float,
                open_spread_mult: float) -> float:
    return sum(s * px for s, px in zip(
        sides, _exit_prices(legs, sides, spot, dte, iv_shift, open_spread_mult), strict=True))


def stress_overnight(legs, sides, *, entry_net: float, spot: float, direction: str,
                     vix: float, now: datetime, expiry: str, lot: int, costs=None,
                     open_spread_mult: float = 2.0) -> StressReport:
    """Reprice the structure at the next open across the stress grid."""
    from execution.costs import DEFAULT_COSTS
    costs = costs or DEFAULT_COSTS
    sign = 1 if direction == "bullish" else -1
    gap_sigma, _ = split_sigma(vix_to_daily_sigma(vix))
    exit_at = next_open(now)
    exp_dt = datetime.fromisoformat(expiry).replace(hour=15, minute=30)
    dte = max((exp_dt - exit_at).total_seconds() / 86400.0, 0.01)
    rep = StressReport(entry_net, lot, gap_sigma, dte)

    def pnl(move_sig: float, iv_shift: float) -> tuple[float, float, float, float]:
        move_pts = sign * move_sig * gap_sigma / 100.0 * spot
        exits = _exit_prices(legs, sides, spot + move_pts, dte, iv_shift, open_spread_mult)
        val = sum(s * px for s, px in zip(sides, exits, strict=True))
        unit = val - entry_net
        charges = 0.0
        for q, s, px_out in zip(legs, sides, exits, strict=True):
            px_in = (q.ask if s > 0 else q.bid) or q.ltp or 0.05
            charges += costs.round_trip(px_in, px_out, lot, long=s > 0)
        return move_pts, val, unit, unit * lot - charges

    measured = MEASURED_IV_CHANGE.get(now.weekday(), -0.12)
    wsum = ev = el = 0.0
    worst = None
    for m in MOVES:
        wm = math.exp(-0.5 * m * m)
        for iv in IV_SHIFTS:
            wi = math.exp(-0.5 * ((iv - measured) / IV_SHIFT_SD) ** 2)
            move_pts, _val, _unit, lotpnl = pnl(m, iv)
            w = wm * wi
            wsum += w
            ev += w * lotpnl
            el += w * min(lotpnl, 0.0)
            if worst is None or lotpnl < worst[0]:
                worst = (lotpnl, move_pts, iv)
    rep.expected_pnl_lot = ev / wsum
    rep.expected_loss_lot = el / wsum
    rep.worst_lot, rep.worst_move_points, rep.worst_iv_shift = worst

    for sc in NAMED:
        move_pts, val, unit, lotpnl = pnl(sc.move_sigmas, sc.iv_shift)
        rep.scenarios.append(ScenarioResult(sc.name, move_pts, sc.iv_shift, val, unit, lotpnl))
        if sc.adverse and -lotpnl > rep.stress_loss_lot:
            rep.stress_loss_lot = -lotpnl
            rep.stress_loss_unit = max(-unit, 0.0)
            rep.stress_scenario = sc.name

    rep.breakeven_move_points = _breakeven(legs, sides, entry_net, spot, dte, measured,
                                           open_spread_mult, gap_sigma)
    return rep


def _breakeven(legs, sides, entry_net, spot, dte, iv_shift, mult, gap_sigma) -> float | None:
    """Signed underlying move (points) at which the exit value equals entry."""
    span = 5 * gap_sigma / 100.0 * spot

    def f(move: float) -> float:
        return _exit_value(legs, sides, spot + move, dte, iv_shift, mult) - entry_net

    roots = []
    grid = [(-span + 2 * span * i / 40) for i in range(41)]
    for a, b in zip(grid, grid[1:], strict=False):
        fa, fb = f(a), f(b)
        if fa == 0:
            roots.append(a)
        elif fa * fb < 0:
            lo, hi = a, b
            for _ in range(50):
                mid = (lo + hi) / 2
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
    if not roots:
        return None
    return min(roots, key=abs)
