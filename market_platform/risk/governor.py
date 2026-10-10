"""Central Risk Governor (M8) — the only approver. Fails closed.

Checks run in the order of plan §7.2. Every failing check is recorded; the
first is the primary reason. Nothing is approved without a lot size from
the contract master and a positive size after rounding *down*.

1. Platform    KILL_SWITCH, NO_SESSION (calendar), FEED_DOWN, DATA_QUALITY
2. Metadata    NOT_IN_UNIVERSE, ROUTE_* (cannot resolve instrument/lot/expiry),
               EXPIRY_TODAY (overnight on an expiring future)
3. Executable  NOT_EXECUTABLE (no allowed route), NO_OVERNIGHT_CASH_SHORT,
               OPTIONS_UNEVALUABLE / OPTIONS_NO_EDGE for option routes
4. Geometry    STOP_SIDE, RR_BELOW_MIN
5. Liquidity   LIQUIDITY_ADV, SPREAD_WIDE
6. Portfolio   DUPLICATE (same underlying+direction+zone), MAX_POSITIONS,
               STRATEGY_CAP, CLUSTER_POSITIONS, DAILY_LOSS (realised + MTM),
               CONSECUTIVE_LOSSES, ENTRIES_PER_SESSION
7. Sizing      budget = equity × tier% (× near-expiry scale for options),
               capped by the head-room left under the aggregate, sector,
               underlying, cluster and overnight caps (BOUND_BY records which);
               per-unit loss on STRESS (intraday: stop + slippage band;
               overnight cash/futures: stop × (1 + gap mult) + band; options:
               premium at the option stop intraday, the full debit overnight);
               lots rounded down; then the order-value, premium-deploy and
               ADV-participation ceilings; zero lots → SIZE_ZERO.
8. Costs       COSTS_EXCEED when round-trip costs > max_cost_frac × reward.

Decisions are written once per signal (`risk_decisions.signal_id` UNIQUE),
with the exposure seen before the decision and the limits applied.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime

from market_platform.execution.costs import SegmentCosts
from market_platform.options.routes import Route
from model.order_blocks.types import short_hash


@dataclass
class RiskDecision:
    decision_id: str
    signal_id: str
    approved: bool
    reasons: list[str]
    route: Route | None = None
    lots: int = 0
    lot_size: int = 0
    quantity: int = 0
    entry_price: float = 0.0
    per_unit_loss: float = 0.0
    risk_rupees: float = 0.0
    budget: float = 0.0
    bound_by: str = ""
    tier: str = "normal"
    est_costs: float | None = None
    pricing: dict | None = None
    exposure_before: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)

    @property
    def primary(self) -> str:
        return self.reasons[0] if self.reasons else "APPROVED"


class CentralGovernor:
    def __init__(self, cfg, *, kill_switch=None, calendar=None) -> None:
        self.cfg = cfg
        self.r = cfg.risk
        self.liq = cfg.liquidity
        self.costs = SegmentCosts(cfg.costs)
        self.kill_switch = kill_switch or (lambda: False)
        self.calendar = calendar
        self.counters: dict[str, int] = {"approved": 0, "rejected": 0}

    def limits(self) -> dict:
        return {k: v for k, v in asdict(self.r).items()} | {
            "max_spread_bps_equity": self.liq.max_spread_bps_equity,
            "min_adv_value_cr": self.liq.min_adv_value_cr,
            "max_slippage_bps": self.liq.max_slippage_bps,
            "max_signals_per_cluster": self.cfg.scoring.max_signals_per_cluster}

    # ------------------------------------------------------------------------------------

    def decide(self, cand, *, instrument: dict | None, portfolio, now: datetime,
               route: Route | None, route_reason: str = "", pricing=None,
               feed_ok: bool = True, quarantined: set[str] = frozenset()) -> RiskDecision:
        r, eq = self.r, portfolio.equity
        reasons: list[str] = []
        exp = portfolio.exposure()
        dec = RiskDecision("D" + short_hash(cand.signal_id, "decision")[:15], cand.signal_id, False,
                           reasons, route=route, exposure_before=exp.to_dict(), limits=self.limits())
        if pricing is not None:
            dec.pricing = {k: getattr(pricing, k) for k in ("status", "structure", "o_entry", "o_stop",
                                                            "o_target", "lot_size", "ev_per_lot",
                                                            "max_loss_per_unit", "reasons")}
        # 1 platform
        if self.kill_switch():
            reasons.append("KILL_SWITCH")
        if self.calendar is not None and not self.calendar.is_trading_day(now.date()):
            reasons.append("NO_SESSION")
        if not feed_ok:
            reasons.append("FEED_DOWN")
        if cand.instrument_key in quarantined or cand.data_quality != "OK":
            reasons.append("DATA_QUALITY")
        # 2 metadata
        if instrument is None:
            reasons.append("NOT_IN_UNIVERSE")
        if route is None:
            reasons.append(f"ROUTE_{route_reason or 'NONE'}")
        elif route.segment == "futures" and cand.horizon == "overnight" and route.expiry \
                and route.expiry <= now.date().isoformat():
            reasons.append("EXPIRY_TODAY")
        # 3 executability
        if cand.executable is False:
            reasons.append("NOT_EXECUTABLE")
        if route is not None and route.is_option:
            if pricing is None or pricing.status == "UNEVALUABLE":
                why = "; ".join(pricing.reasons[:2]) if pricing else "not priced"
                reasons.append(f"OPTIONS_UNEVALUABLE:{why}")
            elif pricing.status != "OK":
                reasons.append("OPTIONS_NO_EDGE")
        # 4 geometry
        bull = cand.direction == "bullish"
        u_risk = (cand.entry - cand.stop) if bull else (cand.stop - cand.entry)
        if u_risk <= 0:
            reasons.append("STOP_SIDE")
        min_rr = (self.cfg.bullish if bull else self.cfg.bearish).min_rr
        if cand.rr < min_rr:
            reasons.append("RR_BELOW_MIN")
        # 5 liquidity
        if instrument and instrument.get("kind") == "equity":
            adv = instrument.get("adv_value_cr")
            if adv is not None and adv < self.liq.min_adv_value_cr:
                reasons.append("LIQUIDITY_ADV")
            sp = (cand.liquidity or {}).get("live_spread_bps") or instrument.get("median_spread_bps")
            if sp is not None and sp > self.liq.max_spread_bps_equity:
                reasons.append("SPREAD_WIDE")
        # 6 portfolio
        zone_key = f"{cand.underlying}|{cand.direction}|{cand.zone_id}"
        if zone_key in exp.zones or cand.signal_id in portfolio.positions:
            reasons.append("DUPLICATE")
        if exp.open_positions >= r.max_open_positions:
            reasons.append("MAX_POSITIONS")
        if exp.by_strategy.get(cand.strategy, 0) >= r.max_positions_per_strategy:
            reasons.append("STRATEGY_CAP")
        if cand.cluster_id and exp.cluster_count.get(cand.cluster_id, 0) >= r.max_positions_per_cluster:
            reasons.append("CLUSTER_POSITIONS")
        pnl_today = portfolio.realised_today(now.date()) + portfolio.unrealised()
        if pnl_today <= -eq * r.max_daily_loss_pct / 100:
            reasons.append("DAILY_LOSS")
        if portfolio.consecutive_losses >= r.max_consecutive_losses:
            reasons.append("CONSECUTIVE_LOSSES")
        if portfolio.entries_today(now.date()) >= r.max_entries_per_session:
            reasons.append("ENTRIES_PER_SESSION")
        if reasons:
            return self._done(dec)

        # 7 sizing
        pct = r.risk_high_tier_pct if (cand.score or 0) >= r.high_tier_score else r.risk_per_trade_pct
        dec.tier = "high" if pct == r.risk_high_tier_pct and pct != r.risk_per_trade_pct else "normal"
        budget = eq * pct / 100
        sector = (instrument or {}).get("sector") or "Unclassified"
        heads = {
            "aggregate": eq * r.max_aggregate_open_risk_pct / 100 - exp.total_risk,
            "sector": eq * r.max_risk_per_sector_pct / 100 - exp.by_sector.get(sector, 0),
            "underlying": eq * r.max_risk_per_underlying_pct / 100
            - exp.by_underlying.get(cand.underlying, 0),
        }
        if cand.cluster_id:
            heads["cluster"] = eq * r.max_correlated_cluster_pct / 100 - exp.by_cluster.get(
                cand.cluster_id, 0)
        if cand.horizon == "overnight":
            heads["overnight"] = eq * r.max_overnight_risk_pct / 100 - exp.overnight_risk
        tight = min(heads, key=heads.get)
        if heads[tight] < budget:
            budget, dec.bound_by = heads[tight], tight
        if budget <= 0:
            reasons.append(f"CAP_{tight.upper()}")
            return self._done(dec)

        slip = self.liq.max_slippage_bps / 1e4
        if route.is_option:
            lot = pricing.lot_size
            entry = pricing.o_entry
            if cand.horizon == "intraday" and pricing.o_stop is not None and entry > 0:
                per_unit = min(max(entry - pricing.o_stop, 0) + 2 * self.cfg.execution.tick * len(
                    pricing.legs), pricing.max_loss_per_unit)
            else:
                per_unit = pricing.max_loss_per_unit
            dte_scale = 1.0
            exp_str = pricing.legs[0]["expiry"] if pricing.legs else ""
            if exp_str:
                dte = (datetime.fromisoformat(exp_str).date() - now.date()).days
                if dte <= r.near_expiry_days:
                    dte_scale = r.near_expiry_scale
            budget *= dte_scale
            dec.route = Route(route.name, "options", route.product, "BUY", cand.instrument_key, lot,
                           exp_str)
        else:
            lot = route.lot_size
            entry = cand.entry
            band = 2 * slip * entry
            per_unit = u_risk * (1 + r.overnight_gap_stress_mult) if cand.horizon == "overnight" \
                else u_risk
            per_unit += band
        dec.per_unit_loss = round(per_unit, 4)
        dec.entry_price = entry
        dec.lot_size = lot
        if per_unit <= 0 or not lot:
            reasons.append("SIZE_ZERO")
            return self._done(dec)
        lots = math.floor(budget / (per_unit * lot))
        value_per_lot = abs(entry) * lot
        if value_per_lot > 0:
            lots = min(lots, math.floor(r.max_order_value_rupees / value_per_lot))
            if route.is_option:
                lots = min(lots, math.floor(eq * r.max_premium_deploy_pct / 100 / value_per_lot))
            adv = (instrument or {}).get("adv_value_cr")
            if route.segment == "equity" and adv:
                lots = min(lots, math.floor(adv * 1e7 * r.max_adv_participation_pct / 100
                                            / value_per_lot))
        if lots <= 0:
            reasons.append("SIZE_ZERO")
            return self._done(dec)
        dec.lots, dec.quantity = lots, lots * lot
        dec.risk_rupees = round(dec.quantity * per_unit, 2)
        dec.budget = round(budget, 2)

        # 8 costs vs reward
        if route.is_option:
            tgt = pricing.o_target if pricing.o_target is not None else entry
            reward = abs(tgt - entry) * dec.quantity
            n_legs = max(1, len(pricing.legs))
            costs = n_legs * self.costs.round_trip("options", route.product, abs(entry) / n_legs,
                                                   abs(tgt) / n_legs, dec.quantity)
        else:
            reward = abs(cand.targets[0] - cand.entry) * dec.quantity
            costs = self.costs.round_trip(route.segment, route.product, cand.entry, cand.targets[0],
                                          dec.quantity, long=bull)
        dec.est_costs = round(costs, 2)
        if reward <= 0 or costs > r.max_cost_frac_of_reward * reward:
            reasons.append("COSTS_EXCEED")
            return self._done(dec)
        dec.approved = True
        return self._done(dec)

    def _done(self, dec: RiskDecision) -> RiskDecision:
        self.counters["approved" if dec.approved else "rejected"] += 1
        if not dec.approved:
            p = dec.reasons[0].split(":")[0] if dec.reasons else "?"
            self.counters[p] = self.counters.get(p, 0) + 1
        return dec


def persist(app_conn, dec: RiskDecision, *, run_id: str, now: datetime) -> bool:
    route = dec.route
    cur = app_conn.execute(
        "INSERT OR IGNORE INTO risk_decisions (decision_id, signal_id, run_id, ts, approved, "
        "reason_codes, instrument_key, product, route, lots, lot_size, quantity, risk_rupees, "
        "stress_json, exposure_before_json, limits_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (dec.decision_id, dec.signal_id, run_id, now.isoformat(timespec="seconds"),
         int(dec.approved), json.dumps(dec.reasons), route.instrument_key if route else None,
         route.product if route else None, route.name if route else None, dec.lots, dec.lot_size,
         dec.quantity, dec.risk_rupees,
         json.dumps({"per_unit_loss": dec.per_unit_loss, "budget": dec.budget,
                     "bound_by": dec.bound_by, "tier": dec.tier, "est_costs": dec.est_costs,
                     "pricing": dec.pricing}, default=str),
         json.dumps(dec.exposure_before, default=str), json.dumps(dec.limits, default=str)))
    status = "APPROVED" if dec.approved else "REJECTED"
    app_conn.execute("UPDATE signals SET status=?, reject_reasons=CASE WHEN ?=0 THEN ? ELSE "
                     "reject_reasons END WHERE signal_id=? AND status IN ('QUALIFIED','NOT_EXECUTABLE')",
                     (status, int(dec.approved),
                      json.dumps([f"RISK:{x}" for x in dec.reasons]), dec.signal_id))
    app_conn.execute("INSERT INTO signal_status_history (signal_id, ts, status, reason) "
                     "VALUES (?,?,?,?)", (dec.signal_id, now.isoformat(timespec="seconds"), status,
                                          dec.primary))
    app_conn.commit()
    return bool(cur.rowcount)


def existing(app_conn, signal_id: str) -> dict | None:
    row = app_conn.execute("SELECT * FROM risk_decisions WHERE signal_id=?", (signal_id,)).fetchone()
    return dict(row) if row else None
