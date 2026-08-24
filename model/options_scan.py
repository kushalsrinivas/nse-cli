"""Options decision layer: score candidate contracts instead of blindly
buying the nearest strike.

Given the underlying direction from the technical model, evaluate every
liquid CE/PE candidate on: moneyness, liquidity (OI / volume / spread),
IV sanity, theta cost vs expected move, greeks profile and time to expiry.
Each candidate gets a 0-100 score; the top contract becomes the suggested
trade. Greeks are computed with Black-Scholes (European approximation —
adequate for NIFTY index options).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from analysis.signals import Direction
from config import SETTINGS
from data.options import OptionChain, OptionLeg


# ---------------------------------------------------------------------------
# Black-Scholes pricing + greeks (r, q kept simple for NIFTY index)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def bs_greeks(spot: float, strike: float, dte_days: float, iv: float,
              is_call: bool, r: float = 0.065) -> dict:
    t = max(dte_days, 0.25) / 365.0
    v = max(iv, 0.01)
    if spot <= 0 or strike <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
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
    gamma = _norm_pdf(d1) / (spot * v * sqrt_t)
    vega = spot * _norm_pdf(d1) * sqrt_t / 100.0
    return {
        "delta": round(delta, 3),
        "gamma": round(gamma, 6),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
    }


def days_to_expiry(expiry_iso: str, now: datetime | None = None) -> int:
    now = now or datetime.now()
    try:
        exp = datetime.strptime(expiry_iso, "%Y-%m-%d")
    except ValueError:
        return 999
    return max(0, (exp - now).days)


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionCandidate:
    symbol: str                       # e.g. "NIFTY 25400 CE"
    leg: OptionLeg
    is_call: bool
    dte: int
    premium: float
    delta: float
    theta: float
    vega: float
    gamma: float
    spread_pct: float | None
    distance_from_spot_pct: float
    score: float                      # 0-100
    notes: list[str]

    @property
    def stop_price(self) -> float:
        return round(self.premium * (1 - SETTINGS.default_stop_pct / 100), 2)

    @property
    def target_price(self) -> float:
        return round(self.premium
                     + (self.premium * SETTINGS.default_stop_pct / 100)
                     * SETTINGS.target_multiplier, 2)


def _spread_pct(leg: OptionLeg) -> float | None:
    if leg.bid and leg.ask and leg.ask > 0:
        return (leg.ask - leg.bid) / leg.ask * 100
    return None


def _score_candidate(leg: OptionLeg, is_call: bool, spot: float,
                     iv_median: float, expected_move_pct: float) -> tuple[float, list[str]]:
    score = 50.0
    notes: list[str] = []
    dte = days_to_expiry(leg.expiry)
    greeks = bs_greeks(spot, leg.strike, dte, (leg.iv or 15.0) / 100.0, is_call)
    delta = abs(greeks["delta"])
    dist = abs(leg.strike - spot) / spot * 100

    # --- Moneyness sweet spot: delta 0.35-0.60 ---
    if 0.40 <= delta <= 0.58:
        score += 18
        notes.append("ATM-ish delta")
    elif 0.30 <= delta < 0.40 or 0.58 < delta <= 0.68:
        score += 10
    elif delta > 0.80:
        score -= 12
        notes.append("deep ITM: expensive, low convexity")
    elif delta < 0.20:
        score -= 14
        notes.append("far OTM: low probability")

    # --- Liquidity ---
    oi = leg.open_interest or 0
    vol = leg.volume or 0
    if oi >= 1_000_000:
        score += 10
    elif oi >= 200_000:
        score += 6
    elif oi >= 50_000:
        score += 2
    else:
        score -= 10
        notes.append("thin OI")
    if vol >= 100_000:
        score += 5
    elif vol == 0:
        score -= 8
        notes.append("no traded volume today")

    sp = _spread_pct(leg)
    if sp is not None:
        if sp <= 1.0:
            score += 8
        elif sp <= 3.0:
            score += 3
        else:
            score -= 9
            notes.append(f"wide spread {sp:.1f}%")
    else:
        score -= 4
        notes.append("no quote")

    # --- IV sanity ---
    if leg.iv:
        ratio = leg.iv / iv_median if iv_median else 1.0
        if ratio > 1.35:
            score -= 10
            notes.append(f"IV rich ({leg.iv:.1f}% vs {iv_median:.1f}%)")
        elif ratio < 0.75:
            score += 6
            notes.append("IV cheap")

    # --- Theta cost vs expected move ---
    premium = leg.ltp or 0.0
    if premium > 0 and dte > 0:
        daily_theta_pct = abs(greeks["theta"]) / premium * 100
        move_capture = expected_move_pct * spot / premium * abs(delta)   # % of prem per 1σ day
        if move_capture > daily_theta_pct * 1.2:
            score += 8
        elif daily_theta_pct > move_capture * 1.5 and dte <= 3:
            score -= 12
            notes.append("theta bleed dominates")

    # --- Expiry proximity risk ---
    if dte == 0:
        score -= 15
        notes.append("expiry-day gamma/theta")
    elif dte <= 2:
        score -= 7
        notes.append("near expiry")

    return max(0.0, min(100.0, score)), notes


def scan_candidates(chain: OptionChain, direction: Direction,
                    expiry: str | None = None,
                    max_candidates: int = 5) -> list[OptionCandidate]:
    """Rank call candidates (bullish) or put candidates (bearish).

    Neutral direction yields no candidates — the caller should not trade.
    """
    if direction not in (Direction.BULLISH, Direction.BEARISH):
        return []
    spot = chain.underlying_value
    if not spot:
        return []

    if not chain.rows:
        return []
    nearest = sorted({r.call.expiry for r in chain.rows})[0]
    target_expiry = expiry or nearest
    rows = [r for r in chain.rows if r.call.expiry == target_expiry]
    if not rows:
        return []

    is_call = direction is Direction.BULLISH
    legs = [(r.strike, r.call if is_call else r.put) for r in rows]
    legs = [(s, l) for s, l in legs if l.ltp and l.ltp > 0.05]

    ivs = [l.iv for _, l in legs if l.iv and l.iv > 0]
    iv_median = sorted(ivs)[len(ivs) // 2] if ivs else 15.0

    # Expected daily move ~ 1σ from ATM IV.
    atm_leg = min(legs, key=lambda sl: abs(sl[0] - spot))[1]
    atm_iv = (atm_leg.iv or iv_median) / 100.0
    dte_ref = max(days_to_expiry(atm_leg.expiry), 1)
    expected_move_pct = spot * atm_iv / math.sqrt(252) / spot * 100 * math.sqrt(252 / 252)

    scored: list[OptionCandidate] = []
    for strike, leg in legs:
        raw_score, notes = _score_candidate(leg, is_call, spot, iv_median, expected_move_pct)
        sp = _spread_pct(leg)
        greeks = bs_greeks(spot, leg.strike, days_to_expiry(leg.expiry),
                           (leg.iv or 15.0) / 100.0, is_call)
        option_type = "CE" if is_call else "PE"
        scored.append(OptionCandidate(
            symbol=f"NIFTY {strike:g} {option_type}",
            leg=leg, is_call=is_call, dte=days_to_expiry(leg.expiry),
            premium=leg.ltp or 0.0,
            delta=greeks["delta"], theta=greeks["theta"],
            vega=greeks["vega"], gamma=greeks["gamma"],
            spread_pct=round(sp, 2) if sp is not None else None,
            distance_from_spot_pct=round((leg.strike - spot) / spot * 100, 2),
            score=round(raw_score, 1),
            notes=notes,
        ))
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:max_candidates]
