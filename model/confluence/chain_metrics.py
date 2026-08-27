"""Option-chain metrics for confluence setups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from data.options import ChainRow, OptionChain
from model.confluence.types import SuggestedContract
from model.options_scan import bs_greeks, days_to_expiry


@dataclass(frozen=True)
class ChainSnapshot:
    pcr: float
    max_pain: float
    spot: float
    expiry: str
    long_buildup_calls: bool
    long_buildup_puts: bool
    call_oi_wall: float | None
    put_oi_wall: float | None
    spot_vs_max_pain_pct: float


def _rows_for_expiry(chain: OptionChain, expiry: str | None = None) -> list[ChainRow]:
    exp = expiry or (chain.expiries[0] if chain.expiries else "")
    return chain.for_expiry(exp) if exp else list(chain.rows)


def compute_pcr(rows: list[ChainRow]) -> float:
    call_oi = sum(r.call.open_interest or 0 for r in rows)
    put_oi = sum(r.put.open_interest or 0 for r in rows)
    if call_oi <= 0:
        return 0.0
    return put_oi / call_oi


def compute_max_pain(rows: list[ChainRow], spot: float) -> float:
    """Strike minimizing total option writer payout at expiry."""
    if not rows:
        return spot
    strikes = [r.strike for r in rows]
    oi_by_strike = {r.strike: (r.call.open_interest or 0, r.put.open_interest or 0) for r in rows}
    best_strike, best_pain = spot, float("inf")
    for settle in strikes:
        pain = 0.0
        for strike, (coi, poi) in oi_by_strike.items():
            if settle > strike:
                pain += coi * (settle - strike)
            if settle < strike:
                pain += poi * (strike - settle)
        if pain < best_pain:
            best_pain, best_strike = pain, settle
    return best_strike


def _nearest_strikes(rows: list[ChainRow], spot: float, n: int = 3) -> list[ChainRow]:
    return sorted(rows, key=lambda r: abs(r.strike - spot))[:n]


def detect_long_buildup(rows: list[ChainRow], spot: float, is_call: bool) -> bool:
    """Price↑ + OI↑ at nearest strikes (proxy: positive change_in_oi)."""
    near = _nearest_strikes(rows, spot, 3)
    for r in near:
        leg = r.call if is_call else r.put
        oi_chg = leg.change_in_oi or 0
        if oi_chg > 0:
            return True
    return False


def heaviest_oi_wall(rows: list[ChainRow], spot: float, is_call: bool) -> float | None:
    """Strike with max OI above spot (calls) or below spot (puts)."""
    candidates: list[tuple[int, float]] = []
    for r in rows:
        leg = r.call if is_call else r.put
        oi = leg.open_interest or 0
        if oi <= 0:
            continue
        if is_call and r.strike >= spot:
            candidates.append((oi, r.strike))
        if not is_call and r.strike <= spot:
            candidates.append((oi, r.strike))
    if not candidates:
        return None
    return max(candidates)[1]


def build_chain_snapshot(chain: OptionChain, spot: float | None = None) -> ChainSnapshot | None:
    if not chain.expiries:
        return None
    spot = spot or chain.underlying_value or 0.0
    if spot <= 0:
        return None
    expiry = chain.expiries[0]
    rows = _rows_for_expiry(chain, expiry)
    if not rows:
        return None
    mp = compute_max_pain(rows, spot)
    return ChainSnapshot(
        pcr=round(compute_pcr(rows), 3),
        max_pain=mp,
        spot=spot,
        expiry=expiry,
        long_buildup_calls=detect_long_buildup(rows, spot, True),
        long_buildup_puts=detect_long_buildup(rows, spot, False),
        call_oi_wall=heaviest_oi_wall(rows, spot, True),
        put_oi_wall=heaviest_oi_wall(rows, spot, False),
        spot_vs_max_pain_pct=abs(spot - mp) / spot * 100.0 if spot else 0.0,
    )


def _weekly_expiry(chain: OptionChain, min_dte: int = 2) -> str | None:
    now = datetime.now()
    for exp in chain.expiries:
        dte = days_to_expiry(exp, now)
        if dte >= min_dte:
            return exp
    return chain.expiries[0] if chain.expiries else None


def pick_strike(
    chain: OptionChain,
    spot: float,
    is_call: bool,
    delta_min: float,
    delta_max: float,
    min_dte: int = 2,
    prefer_itm: bool = False,
) -> SuggestedContract | None:
    expiry = _weekly_expiry(chain, min_dte)
    if not expiry:
        return None
    rows = _rows_for_expiry(chain, expiry)
    dte = days_to_expiry(expiry)
    best: SuggestedContract | None = None
    best_dist = float("inf")

    for r in rows:
        leg = r.call if is_call else r.put
        iv = (leg.iv or 13.0) / 100.0
        g = bs_greeks(spot, r.strike, dte, iv, is_call)
        delta = abs(g["delta"])
        if delta < 0.25 or delta < delta_min or delta > delta_max:
            continue
        if prefer_itm:
            if is_call and r.strike > spot:
                continue
            if not is_call and r.strike < spot:
                continue
        dist = abs(delta - (delta_min + delta_max) / 2)
        if dist < best_dist:
            best_dist = dist
            opt = "CE" if is_call else "PE"
            best = SuggestedContract(
                symbol=f"NIFTY {int(r.strike)} {opt}",
                strike=r.strike,
                is_call=is_call,
                expiry=expiry,
                entry_price=leg.ltp,
                delta=round(delta, 3),
                dte=dte,
            )
    return best
