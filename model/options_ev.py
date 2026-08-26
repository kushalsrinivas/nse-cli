"""Options Expected Value (EV) Decision Engine.

Evaluates candidate option structures (ITM single-leg, ATM single-leg control,
and Debit Spreads) against the full conditional distribution of overnight moves.

Computes 2nd-order Greek decomposition (Delta, Gamma, Vega, Theta) alongside
exact Black-Scholes revaluation, explicitly factoring in bid-ask spread,
slippage, and execution fees (brokerage + STT + taxes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from analysis.signals import Direction
from config import SETTINGS
from data.options import OptionChain, OptionLeg
from model.magnitude import DistributionalMove


# ---------------------------------------------------------------------------
# Black-Scholes Pricing & Greeks
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def bs_price(spot: float, strike: float, dte_days: float, iv: float,
             is_call: bool, r: float = 0.065) -> float:
    """Exact Black-Scholes European option pricing."""
    if spot <= 0 or strike <= 0:
        return 0.0
    t = max(dte_days, 0.001) / 365.0
    v = max(iv, 0.001)
    d1 = (math.log(spot / strike) + (r + v * v / 2) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    disc = math.exp(-r * t)
    if is_call:
        price = spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    else:
        price = strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    return max(0.0, price)


def bs_greeks(spot: float, strike: float, dte_days: float, iv: float,
              is_call: bool, r: float = 0.065) -> dict[str, float]:
    """Analytical Black-Scholes greeks."""
    if spot <= 0 or strike <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    t = max(dte_days, 0.001) / 365.0
    v = max(iv, 0.001)
    d1 = (math.log(spot / strike) + (r + v * v / 2) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    sqrt_t = math.sqrt(t)
    disc = math.exp(-r * t)

    if is_call:
        delta = _norm_cdf(d1)
        theta = (-spot * _norm_pdf(d1) * v / (2 * sqrt_t)
                 - r * strike * disc * _norm_cdf(d2)) / 365.0
    else:
        delta = -_norm_cdf(-d1)
        theta = (-spot * _norm_pdf(d1) * v / (2 * sqrt_t)
                 + r * strike * disc * _norm_cdf(-d2)) / 365.0

    gamma = _norm_pdf(d1) / (spot * v * sqrt_t) if (spot * v * sqrt_t) > 0 else 0.0
    vega = spot * _norm_pdf(d1) * sqrt_t / 100.0   # per 1% IV move

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 3),   # ₹ decay per calendar day
        "vega": round(vega, 3),     # ₹ change per 1% IV change
    }


def days_to_expiry(expiry_iso: str, now: datetime | None = None) -> int:
    now = now or datetime.now()
    try:
        exp = datetime.strptime(expiry_iso, "%Y-%m-%d")
        return max(0, (exp.date() - now.date()).days)
    except Exception:
        return 7


# ---------------------------------------------------------------------------
# Strategy Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyCandidate:
    """A tradeable option candidate structure (Single-Leg or Spread)."""

    strategy_type: str                  # "ITM", "ATM", "DEBIT_SPREAD"
    name: str                           # e.g. "NIFTY 24300 PE (ITM)", "PE Spread 24400/24200"
    symbol: str
    is_call: bool
    dte: int
    expiry: str
    net_premium: float                  # Per unit entry premium (₹)
    max_risk_per_unit: float            # Max capital at risk per unit (₹)
    delta: float
    gamma: float
    theta: float
    vega: float
    spread_cost_per_unit: float         # Estimated round-trip spread & slippage (₹)
    fees_per_unit: float                # Brokerage, STT, exchange fees (₹)
    long_leg: OptionLeg | None = None
    short_leg: OptionLeg | None = None
    long_strike: float = 0.0
    short_strike: float | None = None
    liquidity_score: float = 50.0       # 0-100
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StrategyEV:
    """Expected Value & Risk decomposition across the empirical distribution."""

    candidate: StrategyCandidate
    net_ev_per_lot: float               # ₹ net EV per lot (75 units)
    net_ev_pct: float                   # EV / Max Risk (%)
    expected_delta_pnl_lot: float       # ₹ Delta component
    expected_gamma_pnl_lot: float       # ₹ Gamma component
    expected_vega_pnl_lot: float        # ₹ Vega component (IV change)
    expected_theta_cost_lot: float      # ₹ Theta decay cost
    spread_slippage_lot: float          # ₹ Total friction
    fees_lot: float                     # ₹ Total brokerage/taxes
    win_probability: float              # P(Net PnL > 0)
    p10_pnl_lot: float                  # ₹ 10th percentile outcome (downside)
    p90_pnl_lot: float                  # ₹ 90th percentile outcome (upside)
    median_pnl_lot: float               # ₹ Median outcome
    breakeven_gap_pct: float            # Underling gap % required to break even
    risk_adjusted_ev: float             # Score used for ranking
    is_tradeable: bool                  # Pass/fail hard sanity checks
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> str:
        return "GO" if (self.is_tradeable and self.net_ev_per_lot > 0) else "NO-GO"


# ---------------------------------------------------------------------------
# Friction & Fee Estimators (NSE India standard rates)
# ---------------------------------------------------------------------------

def estimate_fees_per_lot(premium: float, lot_size: int = 75) -> float:
    """Estimate round-trip fees for 1 lot (Brokerage ₹40 + STT + Exchange + GST + Stamp)."""
    turnover = premium * lot_size
    brokerage = 40.0                      # Standard discount broker ₹20 buy + ₹20 sell
    stt = turnover * 0.00125              # 0.125% on sell side (options)
    exchange_charges = turnover * 0.0005  # NSE turnover charge approx 0.05%
    gst = (brokerage + exchange_charges) * 0.18
    stamp_duty = turnover * 0.00003       # 0.003% on buy side
    sebi_charges = turnover * 0.000001
    total_fee = brokerage + stt + exchange_charges + gst + stamp_duty + sebi_charges
    return max(45.0, total_fee)


def estimate_spread_cost(bid: float | None, ask: float | None,
                         ltp: float, min_tick: float = 0.05) -> float:
    """Estimate roundtrip slippage & half-spread cost per unit."""
    if bid is not None and ask is not None and ask > bid:
        spread = ask - bid
        return max(spread * 0.8, min_tick * 2)
    # Default assumption: 0.8% of premium or 1 tick
    return max(ltp * 0.008, min_tick * 2)


# ---------------------------------------------------------------------------
# Candidate Generation
# ---------------------------------------------------------------------------

def _effective_premium(leg: OptionLeg, spot: float, strike: float, dte: int, is_call: bool) -> float:
    theo = bs_price(spot, strike, dte, (leg.iv or 14.0) / 100.0, is_call)
    intrinsic = max(0.0, spot - strike if is_call else strike - spot)
    if leg.ask and leg.ask >= intrinsic:
        return leg.ask
    if leg.ltp and leg.ltp >= intrinsic and abs(leg.ltp - theo) / max(theo, 1.0) < 0.35:
        return leg.ltp
    return max(theo, intrinsic)


def generate_strategy_candidates(
    chain: OptionChain,
    spot: float,
    direction: Direction,
    target_expiry: str | None = None,
    lot_size: int = 75,
) -> list[StrategyCandidate]:
    """Generate and price ITM, ATM, and Debit Spread candidates from option chain."""
    if direction not in (Direction.BULLISH, Direction.BEARISH) or not chain.rows:
        return []

    expiry = target_expiry or chain.expiries[0]
    rows = chain.for_expiry(expiry)
    if not rows:
        return []

    is_call = direction is Direction.BULLISH
    dte = days_to_expiry(expiry)
    candidates: list[StrategyCandidate] = []

    # Filter liquid strikes with valid LTP
    legs_data: list[tuple[float, OptionLeg, dict[str, float]]] = []
    for r in rows:
        leg = r.call if is_call else r.put
        if not leg.ltp or leg.ltp < 0.5:
            continue
        iv = (leg.iv or 14.0) / 100.0
        greeks = bs_greeks(spot, r.strike, dte, iv, is_call)
        legs_data.append((r.strike, leg, greeks))

    if not legs_data:
        return []

    # Sort by strike
    legs_data.sort(key=lambda item: item[0])

    # 1. Find ATM candidate (delta closest to 0.50)
    atm_item = min(legs_data, key=lambda item: abs(abs(item[2]["delta"]) - 0.50))
    atm_strike, atm_leg, atm_greeks = atm_item
    atm_prem = _effective_premium(atm_leg, spot, atm_strike, dte, is_call)
    atm_spread = estimate_spread_cost(atm_leg.bid, atm_leg.ask, atm_prem)
    atm_fee = estimate_fees_per_lot(atm_prem, lot_size) / lot_size
    candidates.append(StrategyCandidate(
        strategy_type="ATM",
        name=f"ATM {'CE' if is_call else 'PE'} {atm_strike:g}",
        symbol=f"NIFTY {atm_strike:g} {'CE' if is_call else 'PE'}",
        is_call=is_call,
        dte=dte,
        expiry=expiry,
        net_premium=round(atm_prem, 2),
        max_risk_per_unit=round(atm_prem, 2),
        delta=atm_greeks["delta"],
        gamma=atm_greeks["gamma"],
        theta=atm_greeks["theta"],
        vega=atm_greeks["vega"],
        spread_cost_per_unit=atm_spread,
        fees_per_unit=atm_fee,
        long_leg=atm_leg,
        long_strike=atm_strike,
        liquidity_score=_calc_liquidity_score(atm_leg),
        notes=("Control arm: ATM naked single-leg",),
    ))

    # 2. Find ITM candidate (delta closest to 0.75, range 0.65 - 0.85)
    itm_candidates = [
        item for item in legs_data
        if (0.65 <= abs(item[2]["delta"]) <= 0.85)
    ]
    if itm_candidates:
        itm_item = min(itm_candidates, key=lambda item: abs(abs(item[2]["delta"]) - 0.75))
        itm_strike, itm_leg, itm_greeks = itm_item
        itm_prem = _effective_premium(itm_leg, spot, itm_strike, dte, is_call)
        itm_spread = estimate_spread_cost(itm_leg.bid, itm_leg.ask, itm_prem)
        itm_fee = estimate_fees_per_lot(itm_prem, lot_size) / lot_size
        candidates.append(StrategyCandidate(
            strategy_type="ITM",
            name=f"ITM {'CE' if is_call else 'PE'} {itm_strike:g}",
            symbol=f"NIFTY {itm_strike:g} {'CE' if is_call else 'PE'}",
            is_call=is_call,
            dte=dte,
            expiry=expiry,
            net_premium=round(itm_prem, 2),
            max_risk_per_unit=round(itm_prem, 2),
            delta=itm_greeks["delta"],
            gamma=itm_greeks["gamma"],
            theta=itm_greeks["theta"],
            vega=itm_greeks["vega"],
            spread_cost_per_unit=itm_spread,
            fees_per_unit=itm_fee,
            long_leg=itm_leg,
            long_strike=itm_strike,
            liquidity_score=_calc_liquidity_score(itm_leg),
            notes=("Higher delta participation (0.70-0.80), lower relative % theta bleed",),
        ))

    # 3. Find Debit Spread candidate (Buy ATM/near-ITM + Sell OTM 150-300 pts away)
    long_leg_candidates = [
        item for item in legs_data
        if (0.45 <= abs(item[2]["delta"]) <= 0.65)
    ]
    short_leg_candidates = [
        item for item in legs_data
        if (0.20 <= abs(item[2]["delta"]) <= 0.38)
    ]

    if long_leg_candidates and short_leg_candidates:
        l_item = long_leg_candidates[0]
        s_item = None
        for s in short_leg_candidates:
            if is_call and s[0] > l_item[0] and (100 <= s[0] - l_item[0] <= 350):
                s_item = s
                break
            elif not is_call and s[0] < l_item[0] and (100 <= l_item[0] - s[0] <= 350):
                s_item = s
                break

        if s_item is not None:
            l_strike, l_leg, l_greeks = l_item
            s_strike, s_leg, s_greeks = s_item
            l_prem = _effective_premium(l_leg, spot, l_strike, dte, is_call)
            s_prem = _effective_premium(s_leg, spot, s_strike, dte, is_call)
            net_debit = max(1.0, l_prem - s_prem)
            spread_w = abs(s_strike - l_strike)
            spr_cost = (estimate_spread_cost(l_leg.bid, l_leg.ask, l_prem)
                        + estimate_spread_cost(s_leg.bid, s_leg.ask, s_prem))
            spr_fees = 2 * estimate_fees_per_lot(net_debit, lot_size) / lot_size
            candidates.append(StrategyCandidate(
                strategy_type="DEBIT_SPREAD",
                name=f"{'Bull' if is_call else 'Bear'} Spread {l_strike:g}/{s_strike:g}",
                symbol=f"NIFTY {l_strike:g}/{s_strike:g} {'CDS' if is_call else 'PDS'}",
                is_call=is_call,
                dte=dte,
                expiry=expiry,
                net_premium=round(net_debit, 2),
                max_risk_per_unit=round(net_debit, 2),
                delta=l_greeks["delta"] - s_greeks["delta"],
                gamma=l_greeks["gamma"] - s_greeks["gamma"],
                theta=l_greeks["theta"] - s_greeks["theta"],
                vega=l_greeks["vega"] - s_greeks["vega"],
                spread_cost_per_unit=spr_cost,
                fees_per_unit=spr_fees,
                long_leg=l_leg,
                short_leg=s_leg,
                long_strike=l_strike,
                short_strike=s_strike,
                liquidity_score=min(_calc_liquidity_score(l_leg), _calc_liquidity_score(s_leg)),
                notes=(f"Hedging short leg at {s_strike:g} reduces theta and IV exposure",),
            ))

    return candidates


def _calc_liquidity_score(leg: OptionLeg) -> float:
    score = 50.0
    oi = leg.open_interest or 0
    vol = leg.volume or 0
    if oi >= 1_000_000:
        score += 20
    elif oi >= 200_000:
        score += 10
    elif oi < 50_000:
        score -= 20

    if vol >= 100_000:
        score += 15
    elif vol == 0:
        score -= 25

    if leg.bid and leg.ask and leg.ask > 0:
        sp_pct = (leg.ask - leg.bid) / leg.ask * 100
        if sp_pct <= 1.0:
            score += 15
        elif sp_pct > 3.0:
            score -= 20
    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# EV Engine (Distributional Evaluation)
# ---------------------------------------------------------------------------

def evaluate_strategy(
    candidate: StrategyCandidate,
    spot: float,
    dist: DistributionalMove,
    expected_delta_iv: float = 0.0,
    holding_days: float = 1.0,
    lot_size: int = 75,
) -> StrategyEV:
    """Evaluate candidate payoff across the empirical distributional moves."""
    if dist.n_samples == 0 or not dist.samples:
        return StrategyEV(
            candidate=candidate,
            net_ev_per_lot=0.0,
            net_ev_pct=0.0,
            expected_delta_pnl_lot=0.0,
            expected_gamma_pnl_lot=0.0,
            expected_vega_pnl_lot=0.0,
            expected_theta_cost_lot=0.0,
            spread_slippage_lot=0.0,
            fees_lot=0.0,
            win_probability=0.0,
            p10_pnl_lot=0.0,
            p90_pnl_lot=0.0,
            median_pnl_lot=0.0,
            breakeven_gap_pct=0.0,
            risk_adjusted_ev=0.0,
            is_tradeable=False,
            rejection_reasons=("No historical distribution samples",),
        )

    gaps = np.array(dist.samples)
    sign = 1 if candidate.is_call else -1
    dte_entry = max(candidate.dte, 0.25)
    dte_exit = max(candidate.dte - holding_days, 0.01)

    # IV baseline
    long_iv = (candidate.long_leg.iv if candidate.long_leg and candidate.long_leg.iv else 14.0) / 100.0
    short_iv = (candidate.short_leg.iv if candidate.short_leg and candidate.short_leg.iv else 14.0) / 100.0
    exit_long_iv = max(0.01, long_iv + expected_delta_iv / 100.0)
    exit_short_iv = max(0.01, short_iv + expected_delta_iv / 100.0)

    # Calculate exact payoffs for each scenario in the distribution
    pnl_per_lot = []
    delta_pnls = []
    gamma_pnls = []
    vega_pnls = []
    theta_costs = []

    spread_and_fee_unit = candidate.spread_cost_per_unit + candidate.fees_per_unit
    spread_and_fee_lot = spread_and_fee_unit * lot_size

    for gap_pct in gaps:
        # Gap signed in trade direction: positive gap_pct means favorable move
        pct_move = gap_pct * sign
        s_exit = spot * (1.0 + pct_move / 100.0)
        ds = s_exit - spot

        # Consistent relative Black-Scholes revaluation (prevents phantom alpha from stale LTP)
        v_long_exit = bs_price(s_exit, candidate.long_strike, dte_exit, exit_long_iv, candidate.is_call)
        v_long_entry_theo = bs_price(spot, candidate.long_strike, dte_entry, long_iv, candidate.is_call)
        delta_v_long = v_long_exit - v_long_entry_theo

        if candidate.strategy_type == "DEBIT_SPREAD" and candidate.short_strike and candidate.short_leg:
            v_short_exit = bs_price(s_exit, candidate.short_strike, dte_exit, exit_short_iv, candidate.is_call)
            v_short_entry_theo = bs_price(spot, candidate.short_strike, dte_entry, short_iv, candidate.is_call)
            delta_v_short = v_short_exit - v_short_entry_theo
            delta_spread = delta_v_long - delta_v_short
            
            # Spread value bounded between 0 and strike width
            width = abs(candidate.short_strike - candidate.long_strike)
            exit_spread_val = max(0.0, min(width, candidate.net_premium + delta_spread))
            gross_pnl_unit = exit_spread_val - candidate.net_premium
        else:
            # Single-leg payoff bounded by -premium
            exit_unit_val = max(0.0, candidate.net_premium + delta_v_long)
            gross_pnl_unit = exit_unit_val - candidate.net_premium

        net_pnl_lot = (gross_pnl_unit * lot_size) - spread_and_fee_lot
        pnl_per_lot.append(net_pnl_lot)

        # 2nd Order Greek attribution (per lot)
        d_pnl = candidate.delta * ds * lot_size
        g_pnl = 0.5 * candidate.gamma * (ds ** 2) * lot_size
        v_pnl = candidate.vega * expected_delta_iv * lot_size
        th_cost = abs(candidate.theta) * holding_days * lot_size

        delta_pnls.append(d_pnl)
        gamma_pnls.append(g_pnl)
        vega_pnls.append(v_pnl)
        theta_costs.append(th_cost)

    pnl_arr = np.array(pnl_per_lot)
    net_ev = float(np.mean(pnl_arr))
    max_risk_lot = candidate.max_risk_per_unit * lot_size
    net_ev_pct = (net_ev / max_risk_lot * 100) if max_risk_lot > 0 else 0.0

    p10 = float(np.percentile(pnl_arr, 10))
    p90 = float(np.percentile(pnl_arr, 90))
    med = float(np.median(pnl_arr))
    win_p = float((pnl_arr > 0).mean())

    # Breakeven gap % calculation
    # Delta * S * gap/100 = Theta * days + Vega * dIV + Spread + Fees
    net_fixed_cost_unit = (abs(candidate.theta) * holding_days
                           + spread_and_fee_unit
                           - candidate.vega * expected_delta_iv)
    eff_delta = max(abs(candidate.delta), 0.10)
    be_gap_pct = (net_fixed_cost_unit / (eff_delta * spot)) * 100.0

    # Risk-adjusted EV metric
    # Reward/Downside: net_ev / (|p10| + 1e-4) or Sharpe-like ratio
    downside_risk = max(abs(p10), 100.0)
    risk_adj = (net_ev / downside_risk) * (win_p / 0.50)

    # Sanity checks
    rejections = []
    if candidate.liquidity_score < 30:
        rejections.append(f"Low contract liquidity (score {candidate.liquidity_score:.0f}/100)")
    if candidate.spread_cost_per_unit > candidate.net_premium * 0.15:
        rejections.append("Wide bid-ask spread (>15% of premium)")
    if candidate.dte == 0:
        rejections.append("0 DTE expiry evening")

    is_tradeable = (len(rejections) == 0)

    return StrategyEV(
        candidate=candidate,
        net_ev_per_lot=round(net_ev, 1),
        net_ev_pct=round(net_ev_pct, 2),
        expected_delta_pnl_lot=round(float(np.mean(delta_pnls)), 1),
        expected_gamma_pnl_lot=round(float(np.mean(gamma_pnls)), 1),
        expected_vega_pnl_lot=round(float(np.mean(vega_pnls)), 1),
        expected_theta_cost_lot=round(float(np.mean(theta_costs)), 1),
        spread_slippage_lot=round(candidate.spread_cost_per_unit * lot_size, 1),
        fees_lot=round(candidate.fees_per_unit * lot_size, 1),
        win_probability=round(win_p, 3),
        p10_pnl_lot=round(p10, 1),
        p90_pnl_lot=round(p90, 1),
        median_pnl_lot=round(med, 1),
        breakeven_gap_pct=round(be_gap_pct, 3),
        risk_adjusted_ev=round(risk_adj, 3),
        is_tradeable=is_tradeable,
        rejection_reasons=tuple(rejections),
    )


def rank_and_select_best_strategy(
    candidates: list[StrategyCandidate],
    spot: float,
    dist: DistributionalMove,
    expected_delta_iv: float = 0.0,
    holding_days: float = 1.0,
    lot_size: int = 75,
) -> tuple[StrategyEV | None, list[StrategyEV]]:
    """Evaluate all candidate structures and return the highest risk-adjusted EV."""
    if not candidates:
        return None, []

    evaluated: list[StrategyEV] = []
    for cand in candidates:
        ev = evaluate_strategy(cand, spot, dist, expected_delta_iv, holding_days, lot_size)
        evaluated.append(ev)

    # Filter tradeable with positive EV first
    tradeable_pos = [ev for ev in evaluated if ev.is_tradeable and ev.net_ev_per_lot > 0]

    if tradeable_pos:
        # Rank primarily by risk-adjusted EV
        tradeable_pos.sort(key=lambda ev: ev.risk_adjusted_ev, reverse=True)
        best = tradeable_pos[0]
    else:
        # If no positive EV, pick the highest EV / least negative
        evaluated.sort(key=lambda ev: ev.net_ev_per_lot, reverse=True)
        best = evaluated[0] if evaluated else None

    return best, evaluated
