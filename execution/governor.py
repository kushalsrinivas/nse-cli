"""RiskGovernor: the engine proposes, the governor decides (docs §7).

Sizing (§7.2), per structure, not per single option:

    R      = equity x tier   (score 75-84 → risk_normal, 85+ → risk_high;
                              risk_exceptional is never used here)
    L_unit = per-unit loss under STRESS, not at the planned stop:
             intraday  — structure marked at u_stop, plus one tick per leg
                         and the exit half-spread
             overnight — structure marked at spot x (1 -/+ stress_sigmas x
                         sigma_gap), with IV moved against the position by the
                         measured weekday change + 1 sd, DTE reduced by the hold
             capped at the structure's defined max loss per unit
    lots   = floor(R / (L_unit x lot_size)), then the premium deploy ceiling
             (max_premium_deploy_pct, same backstop as model/risk.py), and the
             near-expiry throttle.

Hard limits (§7.3) are checked before sizing and copied verbatim into the
signal when they block. Nothing here can increase an open position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from config import SETTINGS
from model.forecast.position import MEASURED_IV_CHANGE
from model.forecast.volatility import split_sigma, vix_to_daily_sigma
from model.options_ev import bs_price


@dataclass(frozen=True)
class GovernorLimits:
    daily_loss_pct: float = SETTINGS.max_daily_loss_pct          # 2%
    max_consecutive_losses: int = 3
    max_intraday_positions: int = 2
    max_overnight_positions: int = 1
    max_total_positions: int = SETTINGS.max_open_setups           # 3
    max_direction_risk_pct: float = SETTINGS.max_direction_exposure_risk  # 1.5%
    max_expiry_stress_pct: float = 0.01
    max_entries_per_session: int = 3
    stress_sigmas: float = 1.5
    iv_shift_sd: float = 0.95                 # sd of overnight VIX change (pts)
    near_expiry_days: int = SETTINGS.near_expiry_days
    near_expiry_scale: float = SETTINGS.near_expiry_risk_scale
    max_premium_deploy_pct: float = SETTINGS.max_premium_deploy_pct
    risk_normal: float = SETTINGS.risk_normal
    risk_high: float = SETTINGS.risk_high


@dataclass
class BookState:
    """What the governor needs to know about the account right now."""
    equity: float
    realized_today: float = 0.0
    unrealized: float = 0.0                       # MTM of open positions (negative = loss)
    entries_today: int = 0
    consecutive_losses: int = 0
    open_by_horizon: dict[str, int] = field(default_factory=lambda: {"intraday": 0, "overnight": 0})
    open_risk_by_direction: dict[str, float] = field(default_factory=lambda: {"bullish": 0.0, "bearish": 0.0})
    open_stress_by_expiry: dict[str, float] = field(default_factory=dict)
    open_signal_zone_ids: set[str] = field(default_factory=set)

    @property
    def open_total(self) -> int:
        return sum(self.open_by_horizon.values())


@dataclass
class GovernorDecision:
    allowed: bool
    reason: str = ""
    tier: str = "none"
    risk_budget: float = 0.0
    stress_loss_per_unit: float = 0.0
    stress_loss_per_lot: float = 0.0
    lots: int = 0
    lot_size: int = 0
    risk_rupees: float = 0.0
    premium_deployed: float = 0.0
    bound_by: str = ""


def _leg_value(legs, sides, spot: float, dte: float, iv_shift_pts: float) -> float:
    total = 0.0
    for q, s in zip(legs, sides, strict=True):
        iv = max((q.iv or 14.0) + iv_shift_pts, 1.0) / 100.0
        total += s * bs_price(spot, q.strike, max(dte, 0.01), iv, q.is_call)
    return total


def stress_loss_per_unit(choice, setup, spot: float, now: datetime, *, vix: float,
                         limits: GovernorLimits, hold_days: float) -> float:
    """Per-unit loss under the §7.2 stress scenario (>= 0)."""
    legs, sides = choice.legs, choice.sides
    entry = choice.o_entry
    dte = choice.structure.dte
    bull = setup.plan.direction == "bullish"
    if setup.horizon == "intraday":
        val = _leg_value(legs, sides, setup.plan.u_stop, dte - hold_days, 0.0)
        half_spreads = sum(((q.ask or 0) - (q.bid or 0)) / 2 for q in legs)
        loss = entry - val + 0.05 * len(legs) + half_spreads
    else:
        gap_sigma, _ = split_sigma(vix_to_daily_sigma(vix))
        move = limits.stress_sigmas * gap_sigma / 100.0
        s_stress = spot * (1 - move) if bull else spot * (1 + move)
        measured = MEASURED_IV_CHANGE.get(now.weekday(), -0.12)
        long_vega = entry > 0
        iv_shift = (measured - limits.iv_shift_sd) if long_vega else (measured + limits.iv_shift_sd)
        val = _leg_value(legs, sides, s_stress, dte - hold_days, iv_shift)
        half_spreads = sum(((q.ask or 0) - (q.bid or 0)) / 2 for q in legs)
        loss = entry - val + half_spreads
    return round(min(max(loss, 0.0), choice.max_loss_per_unit), 2)


class RiskGovernor:
    def __init__(self, limits: GovernorLimits | None = None) -> None:
        self.l = limits or GovernorLimits()

    def hard_limits(self, setup, book: BookState) -> str:
        """'' when no limit blocks, else the first blocking reason."""
        lim = self.l
        loss = -(book.realized_today + min(book.unrealized, 0.0))
        if loss >= lim.daily_loss_pct * book.equity:
            return f"daily loss ₹{loss:,.0f} ≥ {lim.daily_loss_pct:.1%} of equity"
        if book.consecutive_losses >= lim.max_consecutive_losses:
            return f"{book.consecutive_losses} consecutive losses — halted for the day"
        if book.entries_today >= lim.max_entries_per_session:
            return f"{book.entries_today} entries today (max {lim.max_entries_per_session})"
        if book.open_total >= lim.max_total_positions:
            return f"{book.open_total} open positions (max {lim.max_total_positions})"
        cap = (lim.max_intraday_positions if setup.horizon == "intraday"
               else lim.max_overnight_positions)
        if book.open_by_horizon.get(setup.horizon, 0) >= cap:
            return f"max {setup.horizon} positions ({cap}) open"
        if setup.zone.zone_id in book.open_signal_zone_ids:
            return "zone already traded (one attempt per zone)"
        dir_used = book.open_risk_by_direction.get(setup.plan.direction, 0.0)
        if dir_used >= lim.max_direction_risk_pct * book.equity:
            return f"{setup.plan.direction} exposure ₹{dir_used:,.0f} at limit"
        return ""

    def size(self, setup, choice, book: BookState, spot: float, now: datetime, *,
             vix: float, hold_days: float, dte_days: float) -> GovernorDecision:
        lim = self.l
        block = self.hard_limits(setup, book)
        if block:
            return GovernorDecision(False, block)
        score = setup.score.total
        if score >= 85:
            tier, risk_pct = "high", lim.risk_high
        elif score >= 75:
            tier, risk_pct = "normal", lim.risk_normal
        else:
            return GovernorDecision(False, f"score {score:.0f} below sizing tiers", "none")
        if dte_days <= lim.near_expiry_days:
            risk_pct *= lim.near_expiry_scale
        budget = book.equity * risk_pct
        dir_room = lim.max_direction_risk_pct * book.equity - book.open_risk_by_direction.get(
            setup.plan.direction, 0.0)
        budget = min(budget, max(dir_room, 0.0))

        lot = choice.lot_size
        l_unit = stress_loss_per_unit(choice, setup, spot, now, vix=vix, limits=lim,
                                      hold_days=hold_days)
        if l_unit <= 0:
            l_unit = max(abs(choice.o_entry) * 0.05, 0.05)   # never size on a zero loss
        per_lot = l_unit * lot
        by_risk = int(budget // per_lot) if per_lot > 0 else 0
        margin_unit = abs(choice.o_entry) if choice.o_entry > 0 else choice.max_loss_per_unit
        premium_lot = margin_unit * lot
        deploy_cap = book.equity * lim.max_premium_deploy_pct
        by_deploy = int(deploy_cap // premium_lot) if premium_lot > 0 else by_risk
        expiry = choice.structure.expiry
        exp_room = lim.max_expiry_stress_pct * book.equity - book.open_stress_by_expiry.get(expiry, 0.0)
        by_expiry = int(max(exp_room, 0.0) // per_lot) if per_lot > 0 else 0
        lots = min(by_risk, by_deploy, by_expiry)
        # Ties report the most fundamental limit first: risk budget, then the
        # deploy ceiling, then per-expiry stress.
        bound = min(((by_risk, 0, "risk_budget"), (by_deploy, 1, "deploy_ceiling"),
                     (by_expiry, 2, "expiry_stress")))[2]
        dec = GovernorDecision(
            lots >= 1, "", tier, round(budget, 2), l_unit, round(per_lot, 2),
            max(lots, 0), lot, round(max(lots, 0) * per_lot, 2),
            round(max(lots, 0) * premium_lot, 2), bound)
        if lots < 1:
            if bound == "deploy_ceiling":
                dec.reason = (f"one lot needs ₹{premium_lot:,.0f}, above the ₹{deploy_cap:,.0f} "
                              f"deploy ceiling")
            elif bound == "expiry_stress":
                dec.reason = f"expiry {expiry} stress budget exhausted"
            else:
                dec.reason = (f"risk budget ₹{budget:,.0f} too small for ₹{per_lot:,.0f}/lot "
                              f"stress loss")
        return dec


def ceil_lots(units: int, lot: int) -> int:
    return int(math.ceil(units / lot)) if lot else 0
