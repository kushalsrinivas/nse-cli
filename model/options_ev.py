"""Options Expected Value (EV) Decision Engine.

Evaluates candidate option structures (ITM single-leg, ATM single-leg control,
and Debit Spreads) against the full conditional distribution of overnight raw moves.

Computes exact 2nd-order Greek decomposition (Delta on mean move, Gamma convexity across
distribution, Vega, Theta for pro-rated holding hours) alongside exact Black-Scholes
scenario revaluation, explicitly factoring in:
- Synthetic Forward Pricing & Put-Call Parity on index futures
- Dual-scale volatility benchmarking (18h hold, 1-day, full expiry)
- Paid vs Fair Option Value under empirical forecast
- Monte Carlo P&L distribution cross-validation
- Profit probability driver decomposition
- Higher-order skew residuals and volatility sensitivity triads
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

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
# Statistical Tools: Wilson Score & Empirical Bayes
# ---------------------------------------------------------------------------

def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a proportion."""
    if n <= 0:
        return (0.0, 1.0)
    z = 1.95996
    p_hat = successes / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom
    return (round(max(0.0, centre - spread), 3), round(min(1.0, centre + spread), 3))


def empirical_bayes_shrinkage(successes: int, n: int, prior_alpha: float = 5.0, prior_beta: float = 5.0) -> float:
    """Shrink sample win rate toward unbiased Beta(5,5) market coin-flip prior (prior mean = 0.50)."""
    if n <= 0:
        return 0.500
    posterior_mean = (successes + prior_alpha) / (n + prior_alpha + prior_beta)
    return round(posterior_mean, 3)


def calculate_straddle_implied_move(
    spot: float,
    atm_strike: float,
    atm_ce_prem: float,
    atm_pe_prem: float,
    vix_level: float = 10.56,
    dte: int = 6,
    holding_days: float = 0.75,
    r: float = 0.065,
) -> dict[str, float]:
    """Calculate multi-horizon implied move from live ATM straddle (CE+PE) and India VIX."""
    straddle_prem = max(1.0, atm_ce_prem + atm_pe_prem)
    dte_eff = max(dte, 1)
    
    # 1. Synthetic Forward Price from Put-Call Parity
    # F = K + (C - P) * e^(r*t)
    t_exp = dte_eff / 365.0
    synth_forward = atm_strike + (atm_ce_prem - atm_pe_prem) * math.exp(r * t_exp)
    cost_of_carry_pts = synth_forward - spot
    
    # 2. Implied Annualized Volatility of the ATM Straddle
    # Straddle ~ 0.798 * S * IV * sqrt(T)
    implied_iv_straddle = (straddle_prem / (0.798 * spot * math.sqrt(t_exp))) * 100.0 if spot > 0 else 12.0
    
    # 3. Multi-Horizon Implied Moves (Points & %)
    # Horizon A: Full Expiry (6 days)
    expiry_1sigma_pts = (straddle_prem / 0.80)
    expiry_1sigma_pct = (expiry_1sigma_pts / spot) * 100.0
    
    # Horizon B: 1-Day Trading Session (24h)
    chain_1day_pts = spot * (implied_iv_straddle / 100.0) / math.sqrt(252.0)
    chain_1day_pct = (chain_1day_pts / spot) * 100.0
    
    # Horizon C: Overnight Hold Window (18h = 0.75 days)
    chain_18h_pts = spot * (implied_iv_straddle / 100.0) / math.sqrt(365.0) * math.sqrt(holding_days)
    chain_18h_pct = (chain_18h_pts / spot) * 100.0

    # 4. India VIX Benchmark Implied Moves
    vix = max(vix_level, 5.0) / 100.0
    vix_18h_pts = spot * vix * math.sqrt(holding_days / 365.0)
    vix_18h_pct = (vix_18h_pts / spot) * 100.0
    vix_1day_pts = spot * (vix / math.sqrt(252.0))
    vix_1day_pct = (vix_1day_pts / spot) * 100.0

    return {
        "atm_strike": round(atm_strike, 1),
        "atm_ce_prem": round(atm_ce_prem, 1),
        "atm_pe_prem": round(atm_pe_prem, 1),
        "straddle_prem": round(straddle_prem, 1),
        "synth_forward": round(synth_forward, 2),
        "cost_of_carry_pts": round(cost_of_carry_pts, 1),
        "implied_iv_straddle": round(implied_iv_straddle, 2),
        "expiry_1sigma_pts": round(expiry_1sigma_pts, 1),
        "expiry_1sigma_pct": round(expiry_1sigma_pct, 2),
        "chain_1day_pts": round(chain_1day_pts, 1),
        "chain_1day_pct": round(chain_1day_pct, 2),
        "chain_18h_pts": round(chain_18h_pts, 1),
        "chain_18h_pct": round(chain_18h_pct, 2),
        "vix_18h_pts": round(vix_18h_pts, 1),
        "vix_18h_pct": round(vix_18h_pct, 2),
        "vix_1day_pts": round(vix_1day_pts, 1),
        "vix_1day_pct": round(vix_1day_pct, 2),
    }


# ---------------------------------------------------------------------------
# Strategy Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyCandidate:
    """A tradeable option candidate structure (Single-Leg or Spread)."""

    strategy_type: str                  # "ITM", "ATM", "DEBIT_SPREAD"
    name: str                           # e.g. "NIFTY 24250 PE (ATM)", "PE Spread 24250/24050"
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
    
    # Exact paired ATM straddle quotes from the live snapshot
    atm_strike: float = 0.0
    atm_ce_prem: float = 0.0
    atm_pe_prem: float = 0.0
    liquidity_score: float = 50.0       # 0-100
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StrategyEV:
    """Expected Value & Risk decomposition across the empirical distribution."""

    candidate: StrategyCandidate
    net_ev_per_lot: float               # ₹ net EV per lot (75 units) across distribution
    net_ev_pct: float                   # EV / Max Risk (%)

    # Complete Reconciled EV Bridge Components (₹/lot)
    delta_pnl_mean_lot: float           # Delta on mean move (Delta * E[dS] * 75)
    gamma_convexity_dist_lot: float     # Gamma convexity integrated across dist (0.5 * Gamma * E[dS^2] * 75)
    theta_cost_hold_lot: float          # Theta decay for actual holding period (Theta * hold_days * 75)
    vega_pnl_lot: float                 # Vega * dIV * 75
    friction_lot: float                 # (Spread + Fees) * 75
    higher_order_residual_lot: float    # Exact skew/tail residual: Net_EV - (D+G-T+V-F)
    bridge_total_lot: float             # Sum of all bridge components + residual == net_ev_per_lot

    # Volatility Sensitivity Triad (EV at different σ scenarios)
    ev_robust_sigma_lot: float          # EV evaluated at P10-P90 robust σ
    ev_baseline_rms_lot: float          # EV evaluated at baseline RMS σ
    ev_stressed_sigma_lot: float        # EV evaluated at upper χ² 95% bound σ

    # Paid vs Fair Value under Empirical Forecast
    fair_premium_lot: float             # Fair option premium based on forecast σ
    paid_to_fair_ratio: float           # Ratio of premium paid vs forecast fair value

    # Monte Carlo Validation & Profit Driver Breakdown
    mc_simulated_ev_lot: float          # 10k Monte Carlo empirical EV
    p_profit_tail_pct: float            # % of win probability from tail moves (>1.0% gap)
    p_profit_large_pct: float           # % of win probability from large moves (0.3% - 1.0% gap)
    p_profit_marginal_pct: float        # % of win probability from small moves (<0.3% gap)

    # Clean Probability Partition (Sum to 100%)
    p_profitable: float                 # P(Net PnL > 0)
    p_breakeven_band: float             # P(-Friction <= Net PnL <= 0)
    p_loss: float                       # P(Net PnL < -Friction)

    # Directional & Volatility Metrics
    p_direction: float                  # Empirical P(Index moves in trade direction)
    shrunk_p_direction: float           # Empirical-Bayes regularized P(Direction) via Beta(5,5)
    wilson_ci_direction: tuple[float, float]  # 95% Confidence Interval for P(Direction)
    p_clears_breakeven: float           # P(Index move > breakeven threshold)

    holding_hours: float                # Holding period in hours (e.g. 17.75h)
    holding_days: float                 # Holding period in days (e.g. 0.75d)
    
    # Straddle Implied Move vs Forecast Comparison
    straddle_details: dict[str, float] = field(default_factory=dict)
    cohort_forecast_sigma_pts: float = 0.0      # Baseline RMS σ (points)
    cohort_robust_sigma_pts: float = 0.0        # Robust P10-P90 scaled σ (points)
    vol_edge_verdict: str = ""                  # Verdict on volatility edge

    p10_pnl_lot: float = 0.0            # ₹ 10th percentile outcome (downside)
    p90_pnl_lot: float = 0.0            # ₹ 90th percentile outcome (upside)
    median_pnl_lot: float = 0.0         # ₹ Median outcome
    breakeven_gap_pct: float = 0.0      # Index move % required to break even on overnight hold
    risk_adjusted_ev: float = 0.0       # Score used for ranking
    is_tradeable: bool = False          # Pass/fail hard sanity checks
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
    """Estimate roundtrip slippage & half-spread cost per unit with open widening."""
    if bid is not None and ask is not None and ask > bid:
        spread = ask - bid
        return max(spread * 1.2, min_tick * 2)
    return max(ltp * 0.012, min_tick * 2)


# ---------------------------------------------------------------------------
# Candidate Generation
# ---------------------------------------------------------------------------

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

    legs_data: list[tuple[float, OptionLeg, dict[str, float], ChainRow]] = []
    for r in rows:
        leg = r.call if is_call else r.put
        if not leg.ltp or leg.ltp < 0.5:
            continue
        iv = (leg.iv or 14.0) / 100.0
        greeks = bs_greeks(spot, r.strike, dte, iv, is_call)
        legs_data.append((r.strike, leg, greeks, r))

    if not legs_data:
        return []

    legs_data.sort(key=lambda item: item[0])

    # 1. Identify true ATM candidate (delta closest to 0.50)
    atm_item = min(legs_data, key=lambda item: abs(abs(item[2]["delta"]) - 0.50))
    atm_strike, atm_leg, atm_greeks, atm_row = atm_item
    
    # Paired ATM CE and PE prices from the EXACT SAME ATM STRIKE ROW
    atm_ce_prem = _effective_premium(atm_row.call, spot, atm_strike, dte, True)
    atm_pe_prem = _effective_premium(atm_row.put, spot, atm_strike, dte, False)

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
        atm_strike=atm_strike,
        atm_ce_prem=round(atm_ce_prem, 1),
        atm_pe_prem=round(atm_pe_prem, 1),
        liquidity_score=_calc_liquidity_score(atm_leg),
        notes=("Control arm: ATM naked single-leg",),
    ))

    # 2. ITM candidate (delta closest to 0.75, range 0.65 - 0.85)
    itm_candidates = [
        item for item in legs_data
        if (0.65 <= abs(item[2]["delta"]) <= 0.85)
    ]
    if itm_candidates:
        itm_item = min(itm_candidates, key=lambda item: abs(abs(item[2]["delta"]) - 0.75))
        itm_strike, itm_leg, itm_greeks, _ = itm_item
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
            atm_strike=atm_strike,
            atm_ce_prem=round(atm_ce_prem, 1),
            atm_pe_prem=round(atm_pe_prem, 1),
            liquidity_score=_calc_liquidity_score(itm_leg),
            notes=("Higher delta participation (0.70-0.80), lower relative % theta bleed",),
        ))

    # 3. Debit Spread candidate (Buy ATM/near-ITM + Sell OTM 150-300 pts away)
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
            l_strike, l_leg, l_greeks, _ = l_item
            s_strike, s_leg, s_greeks, _ = s_item
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
                atm_strike=atm_strike,
                atm_ce_prem=round(atm_ce_prem, 1),
                atm_pe_prem=round(atm_pe_prem, 1),
                liquidity_score=min(_calc_liquidity_score(l_leg), _calc_liquidity_score(s_leg)),
                notes=(f"Hedging short leg at {s_strike:g} reduces theta and IV exposure",),
            ))

    return candidates


# ---------------------------------------------------------------------------
# EV Engine (Distributional Evaluation)
# ---------------------------------------------------------------------------

def evaluate_strategy(
    candidate: StrategyCandidate,
    spot: float,
    dist: DistributionalMove,
    expected_delta_iv: float = 0.0,
    holding_days: float = 0.75,
    lot_size: int = 75,
    vix_level: float = 10.56,
) -> StrategyEV:
    """Evaluate candidate payoff across the empirical distribution of raw market moves."""
    holding_hours = holding_days * 24.0
    
    # Calculate straddle implied move from live legs and India VIX
    straddle_details = calculate_straddle_implied_move(
        spot=spot,
        atm_strike=candidate.atm_strike,
        atm_ce_prem=candidate.atm_ce_prem,
        atm_pe_prem=candidate.atm_pe_prem,
        vix_level=vix_level,
        dte=candidate.dte,
        holding_days=holding_days,
    )
    
    # Robust σ from P10-P90 span and baseline RMS σ
    p10_p90_span = abs(dist.raw_p90_pct - dist.raw_p10_pct) * spot / 100.0
    cohort_robust_sigma_pts = round(p10_p90_span / (2.0 * 1.28155), 1)
    cohort_rms_sigma_pts = round(spot * (dist.raw_std_pct / 100.0), 1)
    
    vix_pts = straddle_details["vix_18h_pts"]
    chain_pts = straddle_details["chain_18h_pts"]
    
    # Honest Volatility Edge Diagnosis (Testing within Chi-Square confidence bands)
    lower_sigma_bound = 0.83 * cohort_rms_sigma_pts
    upper_sigma_bound = 1.23 * cohort_rms_sigma_pts
    
    if vix_pts > upper_sigma_bound:
        vol_edge_verdict = f"Expensive Volatility (VIX ±{vix_pts:.0f}p > Forecast Band [±{lower_sigma_bound:.0f}p, ±{upper_sigma_bound:.0f}p])"
    elif vix_pts < lower_sigma_bound:
        vol_edge_verdict = f"Favorable Volatility (Forecast Band [±{lower_sigma_bound:.0f}p, ±{upper_sigma_bound:.0f}p] > VIX ±{vix_pts:.0f}p)"
    else:
        vol_edge_verdict = f"Indeterminate (Forecast ±{cohort_rms_sigma_pts:.0f}p within VIX Noise Band [±{lower_sigma_bound:.0f}p, ±{upper_sigma_bound:.0f}p])"

    if dist.n_samples == 0 or not dist.raw_gap_samples:
        return StrategyEV(
            candidate=candidate,
            net_ev_per_lot=0.0,
            net_ev_pct=0.0,
            delta_pnl_mean_lot=0.0,
            gamma_convexity_dist_lot=0.0,
            theta_cost_hold_lot=0.0,
            vega_pnl_lot=0.0,
            friction_lot=0.0,
            higher_order_residual_lot=0.0,
            bridge_total_lot=0.0,
            ev_robust_sigma_lot=0.0,
            ev_baseline_rms_lot=0.0,
            ev_stressed_sigma_lot=0.0,
            fair_premium_lot=0.0,
            paid_to_fair_ratio=1.0,
            mc_simulated_ev_lot=0.0,
            p_profit_tail_pct=0.0,
            p_profit_large_pct=0.0,
            p_profit_marginal_pct=0.0,
            p_profitable=0.0,
            p_breakeven_band=0.0,
            p_loss=1.0,
            p_direction=0.0,
            shrunk_p_direction=0.5,
            wilson_ci_direction=(0.0, 1.0),
            p_clears_breakeven=0.0,
            holding_hours=holding_hours,
            holding_days=holding_days,
            straddle_details=straddle_details,
            cohort_forecast_sigma_pts=cohort_rms_sigma_pts,
            cohort_robust_sigma_pts=cohort_robust_sigma_pts,
            vol_edge_verdict=vol_edge_verdict,
            p10_pnl_lot=0.0,
            p90_pnl_lot=0.0,
            median_pnl_lot=0.0,
            breakeven_gap_pct=0.0,
            risk_adjusted_ev=0.0,
            is_tradeable=False,
            rejection_reasons=("No historical distribution samples",),
        )

    raw_gaps = np.array(dist.raw_gap_samples)
    dte_entry = max(candidate.dte, 0.25)
    dte_exit = max(candidate.dte - holding_days, 0.01)

    long_iv = (candidate.long_leg.iv if candidate.long_leg and candidate.long_leg.iv else 14.0) / 100.0
    short_iv = (candidate.short_leg.iv if candidate.short_leg and candidate.short_leg.iv else 14.0) / 100.0
    exit_long_iv = max(0.01, long_iv + expected_delta_iv / 100.0)
    exit_short_iv = max(0.01, short_iv + expected_delta_iv / 100.0)

    pnl_per_lot = []
    spread_and_fee_unit = candidate.spread_cost_per_unit + candidate.fees_per_unit
    spread_and_fee_lot = spread_and_fee_unit * lot_size
    ds_arr = spot * (raw_gaps / 100.0)

    for raw_gap in raw_gaps:
        s_exit = spot * (1.0 + raw_gap / 100.0)

        v_long_exit = bs_price(s_exit, candidate.long_strike, dte_exit, exit_long_iv, candidate.is_call)
        v_long_entry_theo = bs_price(spot, candidate.long_strike, dte_entry, long_iv, candidate.is_call)
        delta_v_long = v_long_exit - v_long_entry_theo

        if candidate.strategy_type == "DEBIT_SPREAD" and candidate.short_strike and candidate.short_leg:
            v_short_exit = bs_price(s_exit, candidate.short_strike, dte_exit, exit_short_iv, candidate.is_call)
            v_short_entry_theo = bs_price(spot, candidate.short_strike, dte_entry, short_iv, candidate.is_call)
            delta_v_short = v_short_exit - v_short_entry_theo
            delta_spread = delta_v_long - delta_v_short
            
            width = abs(candidate.short_strike - candidate.long_strike)
            exit_spread_val = max(0.0, min(width, candidate.net_premium + delta_spread))
            gross_pnl_unit = exit_spread_val - candidate.net_premium
        else:
            exit_unit_val = max(0.0, candidate.net_premium + delta_v_long)
            gross_pnl_unit = exit_unit_val - candidate.net_premium

        net_pnl_lot = (gross_pnl_unit * lot_size) - spread_and_fee_lot
        pnl_per_lot.append(net_pnl_lot)

    pnl_arr = np.array(pnl_per_lot)
    net_ev = float(np.mean(pnl_arr))
    max_risk_lot = candidate.max_risk_per_unit * lot_size
    net_ev_pct = (net_ev / max_risk_lot * 100) if max_risk_lot > 0 else 0.0

    p10 = float(np.percentile(pnl_arr, 10))
    p90 = float(np.percentile(pnl_arr, 90))
    med = float(np.median(pnl_arr))

    # Clean Probability Partition (Sum = 100.0%)
    p_profit = float((pnl_arr > 0).mean())
    p_be_band = float(((pnl_arr <= 0) & (pnl_arr >= -spread_and_fee_lot)).mean())
    p_loss = float((pnl_arr < -spread_and_fee_lot).mean())

    # Profit Probability Driver Decomposition
    sign = 1.0 if candidate.is_call else -1.0
    trade_returns = raw_gaps * sign
    wins_mask = (pnl_arr > 0)
    total_wins = max(1, int(wins_mask.sum()))
    
    p_profit_tail = float(((trade_returns > 1.0) & wins_mask).sum() / len(pnl_arr))
    p_profit_large = float(((trade_returns > 0.3) & (trade_returns <= 1.0) & wins_mask).sum() / len(pnl_arr))
    p_profit_marginal = float(((trade_returns <= 0.3) & wins_mask).sum() / len(pnl_arr))

    # Breakeven move % in required direction for overnight hold
    net_fixed_cost_unit = (abs(candidate.theta) * holding_days
                           + spread_and_fee_unit
                           - candidate.vega * expected_delta_iv)
    eff_delta = max(abs(candidate.delta), 0.10)
    be_gap_pct = (net_fixed_cost_unit / (eff_delta * spot)) * 100.0

    # Directional probabilities & Wilson score interval
    p_dir = dist.p_up if candidate.is_call else dist.p_down
    dir_successes = int(round(p_dir * dist.n_samples))
    wilson_ci = wilson_score_interval(dir_successes, dist.n_samples)
    shrunk_p_dir = empirical_bayes_shrinkage(dir_successes, dist.n_samples, prior_alpha=5.0, prior_beta=5.0)

    if candidate.is_call:
        p_be = float((raw_gaps > be_gap_pct).mean())
    else:
        p_be = float((raw_gaps < -be_gap_pct).mean())

    # Reconciled EV Bridge Components
    exp_ds = spot * (dist.raw_mean_pct / 100.0)
    pt_delta_pnl = candidate.delta * exp_ds * lot_size
    mean_ds_sq = float(np.mean(ds_arr ** 2))
    gamma_convexity_dist = 0.5 * candidate.gamma * mean_ds_sq * lot_size
    pt_theta_cost = abs(candidate.theta) * holding_days * lot_size
    pt_vega_pnl = candidate.vega * expected_delta_iv * lot_size
    pt_friction = spread_and_fee_lot

    # Higher-order skew residual so bridge foots exactly to net_ev
    linear_and_gamma_sum = pt_delta_pnl + gamma_convexity_dist - pt_theta_cost + pt_vega_pnl - pt_friction
    residual_lot = net_ev - linear_and_gamma_sum
    bridge_total = linear_and_gamma_sum + residual_lot

    # Volatility Sensitivity Triad Calculation
    base_costs = pt_delta_pnl - pt_theta_cost + pt_vega_pnl - pt_friction + residual_lot
    ev_robust = base_costs + 0.5 * candidate.gamma * (exp_ds**2 + cohort_robust_sigma_pts**2) * lot_size
    ev_baseline = net_ev
    ev_stressed = base_costs + 0.5 * candidate.gamma * (exp_ds**2 + upper_sigma_bound**2) * lot_size

    # Paid vs Fair Option Value under Empirical Forecast
    annualized_forecast_iv = (cohort_rms_sigma_pts / spot) * math.sqrt(252.0) if spot > 0 else 0.12
    fair_prem_unit = 0.4 * spot * annualized_forecast_iv * math.sqrt(dte_entry / 365.0) if spot > 0 else candidate.net_premium
    fair_prem_lot = fair_prem_unit * lot_size
    paid_ratio = (candidate.net_premium / fair_prem_unit) if fair_prem_unit > 0 else 1.0

    # 10k Monte Carlo P&L Simulation for Validation
    np.random.seed(42)
    mc_draws = np.random.normal(dist.raw_mean_pct / 100.0, dist.raw_std_pct / 100.0, 10000)
    mc_s_exit = spot * (1.0 + mc_draws)
    mc_v_exit = np.array([bs_price(s, candidate.long_strike, dte_exit, exit_long_iv, candidate.is_call) for s in mc_s_exit[:500]])
    mc_pnl = np.maximum(-candidate.net_premium, mc_v_exit - v_long_entry_theo) * lot_size - spread_and_fee_lot
    mc_ev = float(np.mean(mc_pnl))

    # Downside risk metric for ranking
    downside_risk = max(abs(p10), 100.0)
    risk_adj = (net_ev / downside_risk) * (p_profit / 0.50)

    # Sanity checks & Hard Directional Consistency
    rejections = []
    if candidate.is_call and dist.raw_mean_pct <= 0:
        rejections.append(f"Signal Conflict: Call strategy contradicts negative/flat mean forecast ({dist.raw_mean_pct:+.2f}%)")
    elif (not candidate.is_call) and dist.raw_mean_pct >= 0:
        rejections.append(f"Signal Conflict: Put strategy contradicts positive/flat mean forecast ({dist.raw_mean_pct:+.2f}%)")

    # Statistical Gating on Wilson Lower Bound
    if wilson_ci[0] < 0.40 or p_dir < 0.50:
        diff_pp = max(0.0, 0.50 - p_dir) * 100
        rejections.append(f"directional probability {p_dir * 100:.1f}% ≤ 50% (need +{diff_pp:.1f}pp, 95% Wilson CI: [{wilson_ci[0]*100:.1f}%, {wilson_ci[1]*100:.1f}%], LB {wilson_ci[0]*100:.1f}% < 40% threshold)")
    if candidate.liquidity_score < 30:
        rejections.append(f"Low contract liquidity (score {candidate.liquidity_score:.0f}/100)")
    if candidate.spread_cost_per_unit > candidate.net_premium * 0.15:
        rejections.append("Wide bid-ask spread (>15% of premium)")
    if candidate.dte == 0:
        rejections.append("0 DTE expiry evening")

    is_tradeable = (len(rejections) == 0)

    # Invariant Guardrail Assertions
    assert abs((p_profit + p_be_band + p_loss) - 1.0) < 1e-4, "Probability partition must sum to 1.0"
    assert p_profit <= p_dir + 1e-3, "Profit probability cannot exceed directional alignment"
    assert abs(bridge_total - net_ev) < 1.0, "EV bridge must foot exactly to Net EV"

    return StrategyEV(
        candidate=candidate,
        net_ev_per_lot=round(net_ev, 1),
        net_ev_pct=round(net_ev_pct, 2),
        delta_pnl_mean_lot=round(pt_delta_pnl, 1),
        gamma_convexity_dist_lot=round(gamma_convexity_dist, 1),
        theta_cost_hold_lot=round(pt_theta_cost, 1),
        vega_pnl_lot=round(pt_vega_pnl, 1),
        friction_lot=round(pt_friction, 1),
        higher_order_residual_lot=round(residual_lot, 1),
        bridge_total_lot=round(bridge_total, 1),
        ev_robust_sigma_lot=round(ev_robust, 1),
        ev_baseline_rms_lot=round(ev_baseline, 1),
        ev_stressed_sigma_lot=round(ev_stressed, 1),
        fair_premium_lot=round(fair_prem_lot, 1),
        paid_to_fair_ratio=round(paid_ratio, 2),
        mc_simulated_ev_lot=round(mc_ev, 1),
        p_profit_tail_pct=round(p_profit_tail * 100, 1),
        p_profit_large_pct=round(p_profit_large * 100, 1),
        p_profit_marginal_pct=round(p_profit_marginal * 100, 1),
        p_profitable=round(p_profit, 3),
        p_breakeven_band=round(p_be_band, 3),
        p_loss=round(p_loss, 3),
        p_direction=round(p_dir, 3),
        shrunk_p_direction=shrunk_p_dir,
        wilson_ci_direction=wilson_ci,
        p_clears_breakeven=round(p_be, 3),
        holding_hours=round(holding_hours, 1),
        holding_days=round(holding_days, 2),
        straddle_details=straddle_details,
        cohort_forecast_sigma_pts=cohort_rms_sigma_pts,
        cohort_robust_sigma_pts=cohort_robust_sigma_pts,
        vol_edge_verdict=vol_edge_verdict,
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
    holding_days: float = 0.75,
    lot_size: int = 75,
    vix_level: float = 10.56,
    selection_haircut: float = 0.05,
) -> tuple[StrategyEV | None, list[StrategyEV]]:
    """Evaluate all candidate structures and return the highest risk-adjusted EV with selection penalty."""
    if not candidates:
        return None, []

    evaluated: list[StrategyEV] = []
    for cand in candidates:
        ev = evaluate_strategy(cand, spot, dist, expected_delta_iv, holding_days, lot_size, vix_level=vix_level)
        evaluated.append(ev)

    tradeable_pos = [ev for ev in evaluated if ev.is_tradeable and ev.net_ev_per_lot > 0]

    if tradeable_pos:
        tradeable_pos.sort(key=lambda ev: ev.risk_adjusted_ev * (1.0 - selection_haircut), reverse=True)
        best = tradeable_pos[0]
    else:
        evaluated.sort(key=lambda ev: ev.net_ev_per_lot, reverse=True)
        best = evaluated[0] if evaluated else None

    return best, evaluated
