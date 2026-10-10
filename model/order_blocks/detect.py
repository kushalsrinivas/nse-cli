"""Displacement leg → source candle → order-block zone (docs §3.4).

Bullish shown; bearish mirrors every comparison.

1. Leg origin o: lowest low in (pivot of the broken swing, break bar b].
   Leg longer than `max_leg_bars` → reject (a grind, not displacement).
2. Displacement: max body over o..b >= disp_body_atr x ATR_b, OR
   leg range (max H over o..b - L_o) >= disp_range_atr x ATR_b.
3. Source candle s: the last bearish (or doji) candle in [o - source_lookback, o],
   i.e. the last opposing candle at or before the start of the move. If none,
   s = o. (The spec's first draft searched back from b-1, which picks
   mid-leg pullback candles; this is the standard definition.)
4. Zone = [L_s, H_s]; if wider than max_zone_atr x ATR, drop the upper wick
   ([L_s, max(O_s, C_s)]); if still wider, reject.
5. Volume gate: max rvol over o..b >= rvol_min when volume exists, else N/A.
6. FVG is attached one bar later (needs bar b+1): see `attach_fvg`.
7. Liquidity sweep: in [o - sweep_lookback, o] a bar took out a swing low
   that was confirmed before that bar opened, and closed back above it.
"""

from __future__ import annotations

from model.order_blocks.params import ObParams
from model.order_blocks.types import BULLISH, Bar, Break, Zone, short_hash


def _leg_origin(bars: list[Bar], start: int, end: int, bullish: bool) -> int:
    idx = range(start, end + 1)
    if bullish:
        return min(idx, key=lambda i: (bars[i].low, -i))
    return max(idx, key=lambda i: (bars[i].high, i))


def build_zone(brk: Break, bars: list[Bar], atr: list[float | None],
               rvol: list[float | None], swings, params: ObParams, *,
               series: str = "NIFTY_SPOT", trend_before: str = "none"
               ) -> tuple[Zone | None, str]:
    """Return (zone, '') or (None, rejection reason)."""
    b = brk.bar_index
    atr_b = atr[b]
    if atr_b is None or atr_b <= 0:
        return None, "ATR not yet defined"
    bullish = brk.direction == BULLISH
    piv = brk.swing.pivot_index
    if piv + 1 > b:
        return None, "no bars between swing and break"
    floor = getattr(bars, "first_index", 0)      # bounded ring: oldest bar held
    o = _leg_origin(bars, max(piv + 1, floor), b, bullish)
    leg_len = b - o + 1
    if leg_len > params.max_leg_bars:
        return None, f"leg {leg_len} bars > {params.max_leg_bars}"

    leg = bars[o:b + 1]
    body_atr = max(x.body for x in leg) / atr_b
    if bullish:
        range_atr = (max(x.high for x in leg) - bars[o].low) / atr_b
    else:
        range_atr = (bars[o].high - min(x.low for x in leg)) / atr_b
    if body_atr < params.disp_body_atr and range_atr < params.disp_range_atr:
        return None, f"displacement body {body_atr:.2f} / range {range_atr:.2f} ATR"

    s = o
    for j in range(o, max(o - params.source_lookback, floor) - 1, -1):
        x = bars[j]
        doji = x.body < params.doji_atr * atr_b
        if doji or (x.bearish if bullish else x.bullish):
            s = j
            break
    src = bars[s]
    lo, hi = src.low, src.high
    if hi - lo > params.max_zone_atr * atr_b:
        if bullish:
            hi = max(src.open, src.close)
        else:
            lo = min(src.open, src.close)
        if hi - lo > params.max_zone_atr * atr_b:
            return None, f"zone {(hi - lo) / atr_b:.2f} ATR wide"

    leg_rvol = [r for r in rvol[o:b + 1] if r is not None]
    max_rvol = max(leg_rvol) if leg_rvol else None
    if max_rvol is not None and max_rvol < params.rvol_min:
        return None, f"rvol {max_rvol:.2f} < {params.rvol_min}"

    swept = None
    kind = "low" if bullish else "high"
    for j in range(max(o - params.sweep_lookback, floor), o + 1):
        x = bars[j]
        for sw in swings.confirmed(x.ts, kind):
            if bullish and x.low < sw.price < x.close:
                swept = sw.price
            elif not bullish and x.high > sw.price > x.close:
                swept = sw.price

    tf = brk.bar.tf
    zone = Zone(
        zone_id=short_hash(series, tf, src.ts.isoformat(), brk.direction,
                           params.fingerprint()),
        series=series, timeframe=tf, direction=brk.direction,
        source_bar_ts=src.ts, bos_bar_ts=brk.bar.ts,
        first_eligible_ts=brk.bar.end,
        zone_low=round(lo, 2), zone_high=round(hi, 2),
        broken_swing=brk.swing.price, broken_swing_ts=brk.swing.pivot_ts,
        leg_origin=bars[o].low if bullish else bars[o].high,
        atr_at_bos=round(atr_b, 4), disp_body_atr=round(body_atr, 3),
        disp_range_atr=round(range_atr, 3),
        rvol=round(max_rvol, 3) if max_rvol is not None else None,
        kind=brk.kind, trend_before=trend_before, swept_level=swept,
        params_hash=params.fingerprint(), bos_index=b)
    zone._leg = (o, b)  # type: ignore[attr-defined]  # for attach_fvg
    return zone, ""


def attach_fvg(zone: Zone, bars: list[Bar]) -> bool:
    """At bar b+1: record the largest FVG in the leg. Returns True if found."""
    o, b = getattr(zone, "_leg", (None, None))
    zone.pending_fvg = False
    if o is None or b + 1 >= len(bars):
        return False
    best = None
    for j in range(max(o + 1, 1), b + 1):
        prev, nxt = bars[j - 1], bars[j + 1]
        if zone.direction == BULLISH and nxt.low > prev.high:
            gap = (prev.high, nxt.low)
        elif zone.direction != BULLISH and nxt.high < prev.low:
            gap = (nxt.high, prev.low)
        else:
            continue
        if best is None or gap[1] - gap[0] > best[1] - best[0]:
            best = gap
    if best is None:
        return False
    zone.fvg_low, zone.fvg_high = round(best[0], 2), round(best[1], 2)
    zone.fvg_in_leg = True
    return True
