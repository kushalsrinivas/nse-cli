"""Exit support for a position already held: hold, or fold at the open?

This closes the loop on the buy-tonight-sell-tomorrow workflow. You bought
last night; at 08:30 the gap model — the one component that measurably
beats its baseline (AUC 0.748) — says roughly where you will open. That is
directly actionable in a way the directional call never was, because it
informs an *exit* you are already committed to rather than an entry the
evidence does not support.

Two measured facts drive the recommendation:

**The session is directionless.** Session-direction models score AUC 0.539,
indistinguishable from a coin flip. Holding past the open therefore adds
variance with no expected return.

**Theta is certain.** It is the only term in the P&L that does not have a
distribution. For a long-premium position, variance without drift plus
guaranteed decay is a negative-expectation combination, and the engine will
usually say take the open — but it computes it rather than assuming it,
because a position far enough in the money inverts the argument.

The overnight IV change is *measured* from India VIX history, not assumed.
The engine this replaces hard-coded a Friday hold at -0.8 points; measured
over 1,229 sessions the Friday-to-Monday change is **+0.51** (95% CI
[+0.36, +0.65]) — the opposite sign, on the case where it matters most.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from model.options_ev import bs_price, estimate_fees_per_lot, estimate_spread_cost

#: Measured mean change in India VIX (points) over a hold entered on this
#: weekday, from 1,229 sessions of 2021-2026. Monday-to-Tuesday etc.; the
#: Friday entry carries the weekend. Used only when live history is not
#: supplied to `measure_overnight_iv_change`.
MEASURED_IV_CHANGE = {0: -0.169, 1: -0.105, 2: -0.153, 3: -0.079, 4: +0.507}

_LEG_RE = re.compile(r"^\s*(?P<sign>[+-]?)(?P<strike>\d+(?:\.\d+)?)\s*"
                     r"(?P<kind>CE|PE|C|P)\s*@\s*(?P<price>\d+(?:\.\d+)?)\s*$",
                     re.IGNORECASE)


class PositionSpecError(ValueError):
    """Raised for a leg spec the parser cannot read."""


@dataclass(frozen=True)
class PositionLeg:
    strike: float
    is_call: bool
    qty: int                 # +1 long, -1 short
    entry_price: float
    iv: float = 0.135

    @property
    def label(self) -> str:
        side = "" if self.qty > 0 else "-"
        return f"{side}{self.strike:g}{'CE' if self.is_call else 'PE'}@{self.entry_price:g}"


def parse_leg(spec: str) -> PositionLeg:
    """Parse `23100CE@223.55`, or `-23300CE@120.1` for a short leg."""
    m = _LEG_RE.match(spec)
    if not m:
        raise PositionSpecError(
            f"cannot read {spec!r} — expected STRIKE{{CE|PE}}@PRICE, "
            f"e.g. 23100CE@223.55 (prefix '-' for a short leg)")
    kind = m.group("kind").upper()
    return PositionLeg(
        strike=float(m.group("strike")),
        is_call=kind.startswith("C"),
        qty=-1 if m.group("sign") == "-" else 1,
        entry_price=float(m.group("price")))


def implied_vol(price: float, spot: float, strike: float, dte: float,
                is_call: bool, lo: float = 0.005, hi: float = 3.0) -> float | None:
    """Bisect for the vol that reprices `price` exactly. None if unattainable."""
    intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
    if price <= intrinsic or price <= 0:
        return None
    if bs_price(spot, strike, dte, hi, is_call) < price:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if bs_price(spot, strike, dte, mid, is_call) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def attach_ivs(legs, chain, spot: float, dte: int) -> list[PositionLeg]:
    """Calibrate each leg's IV to the price actually paid.

    Marking the entry at the chain's quoted IV instead of the one implied
    by your fill injects a phantom P&L at time zero: on a real 23100CE
    filled at 223.55 the chain IV of 14.93% reprices it at 215.12, an
    instant -Rs632/lot the position never lost. Solving for the entry vol
    makes the mark consistent, so the only P&L drivers left are the move,
    the decay and the IV change -- which is what we are trying to measure.

    Falls back to the chain quote when the price cannot be inverted (below
    intrinsic, or a stale print).
    """
    out = []
    for leg in legs:
        iv = leg.iv
        chain_iv = None
        if chain is not None and getattr(chain, "rows", None):
            row = min(chain.rows, key=lambda r: abs(r.strike - leg.strike))
            if abs(row.strike - leg.strike) < 1e-6:
                quote = row.call if leg.is_call else row.put
                if quote.iv and quote.iv > 0:
                    chain_iv = quote.iv / 100.0
        solved = implied_vol(leg.entry_price, spot, leg.strike, dte, leg.is_call)
        iv = solved if solved is not None else (chain_iv or iv)
        out.append(PositionLeg(leg.strike, leg.is_call, leg.qty,
                               leg.entry_price, iv))
    return out


def measure_overnight_iv_change(vix_series=None, entry_weekday: int = 0
                                ) -> tuple[float, float, str]:
    """(mean change, sd, provenance) in VIX points for this hold.

    Measured from history when a series is supplied. The change is
    close-to-close, which spans the whole next session rather than only the
    move to the open, so treat it as an upper bound on the overnight-only
    component — an honest proxy, and the only one this repo has data for.
    """
    if vix_series is not None and len(vix_series) > 200:
        import pandas as pd
        s = pd.Series(vix_series).dropna()
        d = s.diff().dropna()
        # The change realised on the EXIT day belongs to the entry the day
        # before, so a Friday entry maps to the Monday change.
        exit_dow = (entry_weekday + 3) % 7 if entry_weekday == 4 else entry_weekday + 1
        sub = d[d.index.dayofweek == exit_dow] if hasattr(d.index, "dayofweek") else d
        if len(sub) >= 30:
            return (float(sub.mean()), float(sub.std()),
                    f"measured from {len(sub)} matching sessions")
        return float(d.mean()), float(d.std()), f"measured from {len(d)} sessions"
    return (MEASURED_IV_CHANGE.get(entry_weekday, -0.12), 0.95,
            "table measured over 1,229 sessions (2021-2026)")


@dataclass
class ExitOutlook:
    """What the position is likely worth at the open, and whether to hold."""

    legs: tuple[PositionLeg, ...]
    lots: int
    lot_size: int
    entry_cost: float                  # rupees, total, + = paid
    expected_open: float
    spot: float

    ev_at_open: float
    p_profit_at_open: float
    quantiles_at_open: dict[str, float]

    ev_at_close: float
    p_profit_at_close: float
    quantiles_at_close: dict[str, float]

    breakevens: tuple[float, ...]
    iv_change_pts: float
    iv_change_note: str
    exit_friction: float
    recommendation: str
    rationale: list[str] = field(default_factory=list)

    @property
    def hold_gains(self) -> float:
        return self.ev_at_close - self.ev_at_open


def _mark(legs, spot_exit: float, dte: float, iv_shift_pts: float) -> float:
    total = 0.0
    for leg in legs:
        iv = max(leg.iv + iv_shift_pts / 100.0, 0.005)
        total += leg.qty * bs_price(spot_exit, leg.strike, dte, iv, leg.is_call)
    return total


def evaluate_exit(legs, spot: float, gap_dist, session_dist, dte: int, *,
                  lots: int = 1, lot_size: int = 75,
                  iv_change_pts: float = 0.0,
                  iv_change_note: str = "") -> ExitOutlook:
    """Mark the position across the gap distribution, then across the day.

    `gap_dist` prices the exit at tomorrow's open. Holding through the
    session adds the session distribution on top — gap and session correlate
    -0.057 (95% CI [-0.112, -0.001]) on this sample, close enough to
    independent to convolve — plus another third of a day of decay.
    """
    legs = tuple(legs)
    units = lots * lot_size
    entry_per_unit = sum(leg.qty * leg.entry_price for leg in legs)
    entry_cost = entry_per_unit * units

    exit_friction = units * sum(
        estimate_spread_cost(None, None, abs(leg.entry_price))
        + estimate_fees_per_lot(abs(leg.entry_price), lot_size) / lot_size
        for leg in legs)

    gap = gap_dist.draws()
    dte_open = max(dte - 1.0, 0.01)
    open_vals = np.array([_mark(legs, spot * (1 + g / 100.0), dte_open,
                                iv_change_pts) for g in gap])
    pnl_open = (open_vals - entry_per_unit) * units - exit_friction

    # Holding to the close: same calendar day, so roughly another third of a
    # day of time value goes, and the session variance stacks on the gap.
    rng = np.random.default_rng(11)
    sess = rng.choice(session_dist.draws(), size=len(gap), replace=True)
    dte_close = max(dte - 1.3, 0.01)
    close_vals = np.array([_mark(legs, spot * (1 + (g + s) / 100.0), dte_close,
                                 iv_change_pts)
                           for g, s in zip(gap, sess, strict=True)])
    pnl_close = (close_vals - entry_per_unit) * units - exit_friction

    def qs(arr):
        return {f"p{p}": round(float(np.percentile(arr, p)), 0)
                for p in (10, 25, 50, 75, 90)}

    lo, hi = sorted((legs[0].strike * 0.9, legs[-1].strike * 1.1))
    grid = np.linspace(lo, hi, 500)
    curve = np.array([(_mark(legs, s, dte_open, iv_change_pts) - entry_per_unit)
                      * units - exit_friction for s in grid])
    crosses = tuple(float(round(grid[i], 0)) for i in range(1, len(grid))
                    if curve[i - 1] * curve[i] < 0)

    ev_open, ev_close = float(pnl_open.mean()), float(pnl_close.mean())
    p_open = float((pnl_open > 0).mean())
    p_close = float((pnl_close > 0).mean())
    delta = ev_close - ev_open

    # A verdict is only worth stating when the difference is bigger than the
    # cost of acting on it. Declaring "EXIT" off a Rs64 edge on a Rs16,766
    # position is the fake precision this whole rebuild exists to remove.
    material = max(exit_friction, 0.02 * abs(entry_cost))
    rationale = []
    if abs(delta) < material:
        rec = "TOO CLOSE TO CALL — take the open"
        rationale.append(
            f"holding changes expected value by only ₹{delta:+,.0f}, inside the "
            f"₹{material:,.0f} that friction and forecast error already cover — "
            f"not a real difference. Prefer the open: it is the certain exit")
    elif delta > 0:
        rec = "HOLD past the open"
        rationale.append(
            f"holding adds ₹{delta:+,.0f} of expected value — the position's "
            f"convexity outweighs a third of a day of decay")
    else:
        rec = "EXIT at the open"
        rationale.append(
            f"holding costs ₹{-delta:,.0f} of expected value: session direction "
            f"is unforecastable (AUC 0.539), so the day adds variance with no "
            f"drift while theta is certain")

    moved = "rises" if p_close > p_open else "falls" if p_close < p_open else "holds"
    rationale.append(
        f"P(profit) {moved} from {p_open:.0%} at the open to {p_close:.0%} by "
        f"the close, but the spread of outcomes widens "
        f"(P10 ₹{np.percentile(pnl_open, 10):+,.0f} → "
        f"₹{np.percentile(pnl_close, 10):+,.0f})")
    if ev_open < 0:
        rationale.append(
            "expected value at the open is already negative — this is a "
            "damage-control decision, not an optimisation")

    return ExitOutlook(
        legs=legs, lots=lots, lot_size=lot_size,
        entry_cost=round(entry_cost, 1),
        expected_open=round(spot * (1 + gap_dist.location_pct / 100.0), 2),
        spot=round(spot, 2),
        ev_at_open=round(ev_open, 1),
        p_profit_at_open=round(p_open, 4),
        quantiles_at_open=qs(pnl_open),
        ev_at_close=round(ev_close, 1),
        p_profit_at_close=round(p_close, 4),
        quantiles_at_close=qs(pnl_close),
        breakevens=crosses,
        iv_change_pts=round(iv_change_pts, 3),
        iv_change_note=iv_change_note,
        exit_friction=round(exit_friction, 1),
        recommendation=rec, rationale=rationale)


def render_exit(outlook: ExitOutlook, console=None) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    c = console or Console()
    legs = " · ".join(leg.label for leg in outlook.legs)
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("Position", f"[bold]{legs}[/]  × {outlook.lots} lot"
                          f"{'s' if outlook.lots != 1 else ''}")
    t.add_row("Paid", f"₹{outlook.entry_cost:,.0f}")
    t.add_row("Last close", f"{outlook.spot:,.2f}")
    t.add_row("Expected open", f"[bold]{outlook.expected_open:,.0f}[/]")
    t.add_row("IV assumption", f"{outlook.iv_change_pts:+.2f} pts "
                               f"[dim]({outlook.iv_change_note})[/]")
    if outlook.breakevens:
        t.add_row("Breakeven at open",
                  " / ".join(f"{b:,.0f}" for b in outlook.breakevens))
    c.print(Panel(t, title="[bold]POSITION EXIT — 08:30 IST[/]", expand=False))

    o = Table(title="P&L on the position (₹, net of exit friction)", expand=False)
    o.add_column("Exit")
    for col in ("EV", "P(profit)", "P10", "P25", "Median", "P75", "P90"):
        o.add_column(col, justify="right")
    for label, ev, pp, q in (
            ("at tomorrow's open", outlook.ev_at_open,
             outlook.p_profit_at_open, outlook.quantiles_at_open),
            ("hold to the close", outlook.ev_at_close,
             outlook.p_profit_at_close, outlook.quantiles_at_close)):
        o.add_row(label, f"₹{ev:+,.0f}", f"{pp:.0%}",
                  *[f"₹{q[k]:+,.0f}" for k in ("p10", "p25", "p50", "p75", "p90")])
    c.print(o)

    style = ("green" if outlook.recommendation.startswith("HOLD")
             else "cyan" if outlook.recommendation.startswith("TOO CLOSE")
             else "yellow")
    body = [f"[bold]{outlook.recommendation}[/]", ""]
    body += [f"· {r}" for r in outlook.rationale]
    c.print(Panel("\n".join(body), title=f"[bold {style}]VERDICT[/]", expand=False))
