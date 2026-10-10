"""Underlying setup → NIFTY option structure (docs §5).

Three candidate structures per direction (bullish shown; bearish mirrors):

    long_call         +1 CE, |delta| in [delta_lo, delta_hi]
    bull_call_spread  +1 CE (as above), -1 CE at/just past u_target
    bull_put_spread   -1 PE below u_stop, +1 PE `credit_wing_steps` strikes lower

Each is priced with `options_edge.evaluate_structure` (price paid and
friction included) over a horizon-matched distribution. The gate metric is
`ev_model`: rupees/lot net of spread and fees under OUR distribution, whose
scale is the implied sigma divided by the measured implied/realised ratio
(1.19, docs/AUDIT.md). That prices the volatility premium in exactly once,
so the separate long-premium hurdle is not subtracted again. Horizons:

- intraday: VIX session sigma scaled to the minutes left until 15:15
- overnight: VIX gap sigma + session sigma to the 10:30 time exit

Location of that distribution (the only directional input):
- `p_win` given (calibrated, §4.4) → location = p*(target-entry) +
  (1-p)*(stop-entry), as % of spot
- otherwise → ZERO. With no calibrated evidence the setup is not allowed
  to claim a drift, so the EV gate measures only what the structure costs
  versus what it pays. This is deliberate and matches docs/AUDIT.md: until
  the harness shows an edge, long premium will usually fail this gate.

Contract filters (§5.2): DTE, delta band, spread, top-of-book depth, OI and
quote age. A candidate failing any filter is dropped with a reason.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime

from model.forecast.distribution import (
    EmpiricalShape,
    build_distribution,
    implied_distribution,
)
from model.forecast.options_edge import Leg, Structure, StructureEV, evaluate_structure
from model.forecast.volatility import (
    GAP_VARIANCE_SHARE,
    split_sigma,
    vix_to_daily_sigma,
)
from model.options_ev import bs_greeks, bs_price
from model.order_blocks.types import BULLISH, Setup

SESSION_MINUTES = 375.0
_DEFAULT_SHAPE = EmpiricalShape.normal(1500)


@dataclass(frozen=True)
class ContractRules:
    delta_lo: float = 0.45
    delta_hi: float = 0.60
    min_dte_intraday: float = 0.0      # DTE 0 allowed before expiry_cutoff_hour
    expiry_cutoff_hour: int = 13
    min_dte_overnight: int = 2
    max_spread_ticks: float = 2.0
    max_spread_frac: float = 0.015
    tick: float = 0.05
    min_oi: int = 50_000
    max_quote_age_sec: float = 10.0
    credit_wing_steps: int = 2
    overnight_exit_minutes: float = 75.0   # 09:15 -> 10:30 managed window
    o_stop_frac: float = 0.40              # option stop on debit structures
    min_ev_rupees: float = 0.0
    vrp_ratio: float = 1.19                # measured implied / realised


@dataclass
class LegQuote:
    """One chain leg as the selector sees it."""
    tradingsymbol: str
    strike: float
    is_call: bool
    expiry: str
    ltp: float | None
    bid: float | None
    ask: float | None
    bid_qty: int | None = None
    ask_qty: int | None = None
    oi: int | None = None
    iv: float | None = None            # percent
    quote_age_sec: float | None = None
    lot_size: int | None = None

    @property
    def mid(self) -> float | None:
        if self.bid and self.ask and self.ask >= self.bid > 0:
            return (self.bid + self.ask) / 2
        return self.ltp


@dataclass
class ContractChoice:
    structure: Structure
    ev: StructureEV
    legs: list[LegQuote]
    sides: list[int]                   # +1 buy / -1 sell, aligned with legs
    lot_size: int
    o_entry: float                     # net premium per unit (+ debit / - credit)
    o_stop: float | None
    o_target: float | None
    delta: float
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.structure.name

    @property
    def max_loss_per_unit(self) -> float:
        """Defined-risk bound per unit (debit paid, or width - credit)."""
        if self.o_entry >= 0:
            return self.o_entry
        strikes = sorted(leg.strike for leg in self.legs)
        return (strikes[-1] - strikes[0]) + self.o_entry

    def legs_json(self) -> str:
        return json.dumps([{
            "tradingsymbol": q.tradingsymbol, "side": "BUY" if s > 0 else "SELL",
            "qty": s, "strike": q.strike, "type": "CE" if q.is_call else "PE",
            "expiry": q.expiry, "bid": q.bid, "ask": q.ask, "iv": q.iv,
        } for q, s in zip(self.legs, self.sides, strict=True)])


@dataclass
class Selection:
    choice: ContractChoice | None
    candidates: list[ContractChoice] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.choice is not None


# ---------------------------------------------------------------------------
# Horizon distribution
# ---------------------------------------------------------------------------

def horizon_sigma_pct(horizon: str, vix: float, now: datetime,
                      rules: ContractRules) -> float:
    daily = vix_to_daily_sigma(vix)
    gap, session = split_sigma(daily)
    if horizon == "intraday":
        cutoff = now.replace(hour=15, minute=15, second=0, microsecond=0)
        left = max((cutoff - now).total_seconds() / 60.0, 5.0)
        return session * math.sqrt(min(left / SESSION_MINUTES, 1.0))
    frac = min(rules.overnight_exit_minutes / SESSION_MINUTES, 1.0)
    return math.sqrt(gap ** 2 + (session ** 2) * frac)


def holding_days(horizon: str, now: datetime, rules: ContractRules | None = None) -> float:
    """Time decay charged to the hold, in calendar days, variance-consistent.

    BS decays in calendar time, but the move distribution is built from
    trading-session variance. Charging an intraday hold its calendar hours
    (0.2 day) while giving it ~40% of a session's variance hands every long
    option free gamma. So the hold is converted from the variance it is
    exposed to: trading-day fraction x 365/252, and never less than the
    calendar time actually held.
    """
    rules = rules or ContractRules()
    cal_per_trading = 365.0 / 252.0
    if horizon == "intraday":
        cutoff = now.replace(hour=15, minute=15, second=0, microsecond=0)
        left = max((cutoff - now).total_seconds() / 60.0, 5.0)
        frac = (1 - GAP_VARIANCE_SHARE) * min(left / SESSION_MINUTES, 1.0)
        return max(frac * cal_per_trading, left / 1440.0)
    frac = GAP_VARIANCE_SHARE + (1 - GAP_VARIANCE_SHARE) * min(
        rules.overnight_exit_minutes / SESSION_MINUTES, 1.0)
    calendar = 3.0 if now.weekday() == 4 else 1.0      # Friday holds over the weekend
    return max(frac * cal_per_trading, calendar)


def location_pct(setup: Setup, spot: float, p_win: float | None) -> tuple[float, str]:
    if p_win is None:
        return 0.0, "zero (no calibrated edge)"
    plan = setup.plan
    move = p_win * (plan.u_target - plan.u_entry) + (1 - p_win) * (plan.u_stop - plan.u_entry)
    return move / spot * 100.0, f"calibrated p_win={p_win:.2f}"


# ---------------------------------------------------------------------------
# Filters and candidates
# ---------------------------------------------------------------------------

def _dte(expiry: str, now: datetime) -> float:
    exp = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    return max((exp - now).total_seconds() / 86400.0, 0.0)


def expiry_allowed(expiry: str, horizon: str, now: datetime,
                   rules: ContractRules) -> tuple[bool, str]:
    dte = _dte(expiry, now)
    if horizon == "overnight":
        if dte < rules.min_dte_overnight:
            return False, f"{expiry}: DTE {dte:.1f} < {rules.min_dte_overnight} for overnight"
        return True, ""
    if dte < 1.0 and now.hour >= rules.expiry_cutoff_hour:
        return False, f"{expiry}: expiry day after {rules.expiry_cutoff_hour}:00"
    return True, ""


def leg_filter(q: LegQuote, qty_units: int, rules: ContractRules, *,
               buying: bool) -> str:
    """'' if tradeable, else the reason."""
    if q.bid is None or q.ask is None or q.bid <= 0 or q.ask <= 0:
        return f"{q.tradingsymbol}: one-sided book"
    spread = q.ask - q.bid
    mid = (q.ask + q.bid) / 2
    if spread > max(rules.max_spread_ticks * rules.tick, rules.max_spread_frac * mid):
        return f"{q.tradingsymbol}: spread {spread:.2f} on mid {mid:.2f}"
    if q.oi is not None and q.oi < rules.min_oi:
        return f"{q.tradingsymbol}: OI {q.oi} < {rules.min_oi}"
    if q.quote_age_sec is not None and q.quote_age_sec > rules.max_quote_age_sec:
        return f"{q.tradingsymbol}: quote {q.quote_age_sec:.0f}s old"
    depth = q.ask_qty if buying else q.bid_qty
    if depth is not None and qty_units and depth < qty_units:
        return ""          # thin top of book: allowed, fill walks depth (priced in)
    return ""


def _iv(q: LegQuote, spot: float, dte: float) -> float:
    if q.iv:
        return q.iv / 100.0
    from data.kite.chain import implied_vol
    px = q.mid
    iv = implied_vol(spot, q.strike, dte, px, q.is_call) if px else None
    return (iv or 14.0) / 100.0


def _delta(q: LegQuote, spot: float, dte: float) -> float:
    return bs_greeks(spot, q.strike, max(dte, 0.01), _iv(q, spot, dte), q.is_call)["delta"]


def _leg(q: LegQuote, side: int, spot: float, dte: float) -> Leg:
    price = q.ask if side > 0 else q.bid
    return Leg(q.strike, q.is_call, side, float(price), _iv(q, spot, dte), q.bid, q.ask)


def _pick_long(legs: list[LegQuote], spot: float, dte: float,
               rules: ContractRules) -> LegQuote | None:
    best, best_gap = None, None
    for q in legs:
        d = abs(_delta(q, spot, dte))
        if rules.delta_lo <= d <= rules.delta_hi:
            gap = abs(d - (rules.delta_lo + rules.delta_hi) / 2)
            if best_gap is None or gap < best_gap:
                best, best_gap = q, gap
    return best


def build_candidates(setup: Setup, legs: list[LegQuote], spot: float,
                     now: datetime, rules: ContractRules) -> tuple[list, list[str]]:
    """(list of (name, kind, [(LegQuote, side)]), rejections) for one expiry."""
    bull = setup.plan.direction == BULLISH
    rejected: list[str] = []
    if not legs:
        return [], ["no legs"]
    expiry = legs[0].expiry
    dte = _dte(expiry, now)
    want_call = bull
    same = sorted([q for q in legs if q.is_call == want_call], key=lambda q: q.strike)
    other = sorted([q for q in legs if q.is_call != want_call], key=lambda q: q.strike)
    out = []

    long_leg = _pick_long(same, spot, dte, rules)
    if long_leg is None:
        rejected.append(f"{expiry}: no strike with |δ| in [{rules.delta_lo}, {rules.delta_hi}]")
    else:
        name = "long_call" if bull else "long_put"
        out.append((name, "long_premium", [(long_leg, +1)]))
        target = setup.plan.u_target
        if bull:
            shorts = [q for q in same if q.strike >= target and q.strike > long_leg.strike]
            short = shorts[0] if shorts else None
        else:
            shorts = [q for q in reversed(same) if q.strike <= target and q.strike < long_leg.strike]
            short = shorts[0] if shorts else None
        if short is not None:
            out.append(("bull_call_spread" if bull else "bear_put_spread", "spread",
                        [(long_leg, +1), (short, -1)]))
        else:
            rejected.append(f"{expiry}: no short strike at/through target {target}")

    stop = setup.plan.u_stop
    if bull:
        sells = [q for q in reversed(other) if q.strike <= stop]
    else:
        sells = [q for q in other if q.strike >= stop]
    if sells:
        sell = sells[0]
        idx = other.index(sell)
        j = idx - rules.credit_wing_steps if bull else idx + rules.credit_wing_steps
        if 0 <= j < len(other):
            out.append(("bull_put_spread" if bull else "bear_call_spread", "spread",
                        [(sell, -1), (other[j], +1)]))
        else:
            rejected.append(f"{expiry}: no protective wing for credit spread")
    else:
        rejected.append(f"{expiry}: no strike beyond stop {stop} for credit spread")
    return out, rejected


def _mark(legs_sides, spot: float, dte: float) -> float:
    return sum(s * bs_price(spot, q.strike, max(dte, 0.01), _iv(q, spot, dte), q.is_call)
               for q, s in legs_sides)


def select_contract(setup: Setup, chains: dict[str, list[LegQuote]], spot: float,
                    now: datetime, *, vix: float, lot_size_for,
                    rules: ContractRules | None = None, p_win: float | None = None,
                    shape: EmpiricalShape | None = None, lots_hint: int = 1,
                    expected_lot: int | None = None) -> Selection:
    """Pick the max-EV structure that passes every filter.

    `chains` maps expiry → legs. `lot_size_for(tradingsymbol)` returns the
    contract's lot size as of the trade date (master history), never config.
    """
    rules = rules or ContractRules()
    # 1,500 draws: EV and its CI are stable to a few rupees per lot, and a
    # live decision must land inside the 15:15-15:25 window (4,000 took ~5 s).
    shape = shape or _DEFAULT_SHAPE
    sel = Selection(None)
    sigma_implied = horizon_sigma_pct(setup.horizon, vix, now, rules)
    loc, loc_src = location_pct(setup, spot, p_win)
    dist = build_distribution(sigma_implied / rules.vrp_ratio, setup.horizon,
                              shape, loc, loc_src)
    implied = implied_distribution(sigma_implied, setup.horizon, shape)
    hold = holding_days(setup.horizon, now, rules)

    for expiry in sorted(chains):
        ok, why = expiry_allowed(expiry, setup.horizon, now, rules)
        if not ok:
            sel.rejected.append(why)
            continue
        cands, rej = build_candidates(setup, chains[expiry], spot, now, rules)
        sel.rejected.extend(rej)
        dte = _dte(expiry, now)
        for name, kind, legs_sides in cands:
            # Lot size comes from the contract master as of the trade date
            # (data/lots.py). Unknown, inconsistent across legs, or different
            # from config.lot_size → the structure is rejected, never sized.
            lots = set()
            for q, _s in legs_sides:
                master = lot_size_for(q.tradingsymbol)
                if not master:
                    lots.add(None)
                    break
                if q.lot_size and q.lot_size != master:
                    lots.add(-1)
                lots.add(master)
            if None in lots:
                sel.rejected.append(f"{name} {expiry}: lot size unknown for trade date")
                continue
            if -1 in lots or len(lots) != 1:
                sel.rejected.append(f"{name} {expiry}: legs disagree on lot size {sorted(lots)}")
                continue
            lot = lots.pop()
            if expected_lot is not None and lot != expected_lot:
                sel.rejected.append(f"{name} {expiry}: contract lot {lot} ≠ config.lot_size "
                                    f"{expected_lot} — fix config before sizing")
                continue
            bad = [leg_filter(q, lot * lots_hint, rules, buying=s > 0) for q, s in legs_sides]
            bad = [b for b in bad if b]
            if bad:
                sel.rejected.append(f"{name} {expiry}: " + "; ".join(bad))
                continue
            # Fractional DTE on purpose: evaluate_structure reprices at
            # dte - hold, and a rounded-up DTE gives the exit more time
            # value than the entry had — free EV on every long option.
            structure = Structure(name, kind, tuple(_leg(q, s, spot, dte) for q, s in legs_sides),
                                  round(dte, 4), expiry)
            ev = evaluate_structure(structure, spot, dist, lot_size=lot,
                                    holding_days=hold, implied_dist=implied)
            entry = structure.net_debit
            dte_exit = max(dte - hold, 0.01)
            o_stop_model = _mark(legs_sides, setup.plan.u_stop, dte_exit)
            o_target = _mark(legs_sides, setup.plan.u_target, dte_exit)
            if entry > 0:
                o_stop = max(o_stop_model, entry * (1 - rules.o_stop_frac))
            else:
                o_stop = o_stop_model
            delta = sum(s * _delta(q, spot, dte) for q, s in legs_sides)
            sel.candidates.append(ContractChoice(
                structure, ev, [q for q, _ in legs_sides], [s for _, s in legs_sides],
                lot, round(entry, 2), round(o_stop, 2), round(o_target, 2),
                round(delta, 3), notes=list(ev.notes)))

    if not sel.candidates:
        return sel
    ranked = sorted(sel.candidates, key=lambda c: c.ev.ev_model, reverse=True)
    best = ranked[0]
    if best.ev.ev_model > rules.min_ev_rupees:
        sel.choice = best
    else:
        sel.rejected.append(
            f"best structure {best.name} EV ₹{best.ev.ev_model:,.0f}/lot ≤ "
            f"{rules.min_ev_rupees:,.0f} after friction ({loc_src})")
    return sel


def legs_from_chain(chain, expiry: str, *, symbol_for, quote_age_sec=None,
                    depth_for=None) -> list[LegQuote]:
    """OptionChain (data/options.py) → LegQuote list for one expiry.

    `symbol_for(expiry, strike, 'CE'|'PE')` resolves tradingsymbols from the
    master; `depth_for(tradingsymbol)` optionally returns (bid_qty, ask_qty).
    """
    out = []
    for row in chain.for_expiry(expiry):
        for leg, is_call in ((row.call, True), (row.put, False)):
            if leg is None or leg.ltp is None:
                continue
            sym = symbol_for(expiry, row.strike, "CE" if is_call else "PE")
            if not sym:
                continue
            bq, aq = depth_for(sym) if depth_for else (None, None)
            out.append(LegQuote(sym, row.strike, is_call, expiry, leg.ltp, leg.bid,
                                leg.ask, bq, aq, leg.open_interest, leg.iv,
                                quote_age_sec))
    return out
