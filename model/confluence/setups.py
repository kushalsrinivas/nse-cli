"""Setup A/B/C evaluators for live intraday confluence."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from data.options import OptionChain
from model.confluence.chain_metrics import build_chain_snapshot, pick_strike
from model.confluence.indicators import (
    ORBRange,
    compute_cpr,
    compute_orb,
    detect_rsi_divergence,
    ema21_sloping_down,
    ema21_sloping_up,
    enrich_frame,
    latest_swing_pivot,
    macd_histogram_expanding_positive,
    macd_histogram_shrinking_against_trend,
    orb_breakout_direction,
    avg_first_30min_volume,
    resample_bars,
    reversal_past_pivot,
)
from model.confluence.types import (
    ConditionCheck,
    ConditionStatus,
    ConfluenceSetupResult,
)


def _score(conditions: list[ConditionCheck]) -> float:
    applicable = [c for c in conditions if c.status is not ConditionStatus.NA]
    if not applicable:
        return 0.0
    passed = sum(1 for c in applicable if c.status is ConditionStatus.PASS)
    return round(passed / len(applicable) * 100, 1)


def _finalize(
    setup_id: str,
    title: str,
    conditions: list[ConditionCheck],
    direction: str = "neutral",
    suggested=None,
    notes: str = "",
    watch: bool = False,
) -> ConfluenceSetupResult:
    applicable = [c for c in conditions if c.status is not ConditionStatus.NA]
    failed = [c for c in applicable if c.status is ConditionStatus.FAIL]
    blocked = [c.detail or c.name for c in failed if c.detail or c.name]

    if watch:
        decision = "WATCH"
        rationale = blocked[0] if blocked else "Awaiting session window"
    elif not applicable or failed:
        decision = "NO-GO"
        rationale = f"{len(applicable) - len(failed)}/{len(applicable)} conditions met"
    else:
        decision = "GO"
        rationale = f"All {len(applicable)} conditions met — {direction} confluence"

    return ConfluenceSetupResult(
        setup_id=setup_id,
        title=title,
        decision=decision,
        direction=direction,
        confidence_score=_score(conditions),
        conditions=conditions,
        blocked_reasons=blocked,
        decision_rationale=rationale,
        suggested=suggested,
        notes=notes,
    )


def evaluate_setup_a(
    day_5m: pd.DataFrame,
    prev_day: pd.DataFrame,
    chain: OptionChain | None,
    spot: float,
    vix_level: float | None,
    vix_change: float | None,
) -> ConfluenceSetupResult:
    """Trend-Day Momentum Entry."""
    title = "Setup A — Trend-Day Momentum Entry"
    conditions: list[ConditionCheck] = []

    frame_5m = enrich_frame(day_5m)
    frame_15m = enrich_frame(resample_bars(day_5m, "15min"))
    cpr = compute_cpr(prev_day) if not prev_day.empty else None

    bullish = True
    if cpr and frame_15m is not None and not frame_15m.empty:
        close_15 = float(frame_15m["close"].iloc[-1])
        vwap_15 = frame_15m["vwap"].iloc[-1]
        above_cpr = close_15 > cpr.top
        above_vwap = not pd.isna(vwap_15) and close_15 > float(vwap_15)
        if not (above_cpr and above_vwap):
            below_cpr = close_15 < cpr.bottom
            below_vwap = not pd.isna(vwap_15) and close_15 < float(vwap_15)
            if below_cpr and below_vwap:
                bullish = False
            conditions.append(ConditionCheck(
                "15m close vs CPR top + VWAP",
                ConditionStatus.PASS if (above_cpr and above_vwap) or (below_cpr and below_vwap) else ConditionStatus.FAIL,
                f"close={close_15:.1f} CPR top={cpr.top:.1f} bottom={cpr.bottom:.1f}",
            ))
        else:
            conditions.append(ConditionCheck(
                "15m close vs CPR top + VWAP",
                ConditionStatus.PASS,
                f"close={close_15:.1f} > CPR top={cpr.top:.1f} & VWAP",
            ))
    else:
        conditions.append(ConditionCheck("15m close vs CPR top + VWAP", ConditionStatus.FAIL, "insufficient data"))

    ema_ok = (
        ema21_sloping_up(frame_5m) and ema21_sloping_up(frame_15m)
        if bullish
        else ema21_sloping_down(frame_5m) and ema21_sloping_down(frame_15m)
    )
    price_above_ema = False
    if not frame_5m.empty and "ema21" in frame_5m.columns:
        last = frame_5m.iloc[-1]
        ema = float(last["ema21"]) if not pd.isna(last["ema21"]) else 0
        price_above_ema = (float(last["close"]) > ema) if bullish else (float(last["close"]) < ema)
    conditions.append(ConditionCheck(
        "Price vs EMA-21 sloping (5m + 15m)",
        ConditionStatus.PASS if ema_ok and price_above_ema else ConditionStatus.FAIL,
        "bullish alignment" if bullish else "bearish alignment",
    ))

    macd_ok = macd_histogram_expanding_positive(frame_5m) if bullish else (
        not macd_histogram_expanding_positive(frame_5m)
        and float(frame_5m["macd_histogram"].iloc[-1]) < 0 if len(frame_5m) >= 2 else False
    )
    conditions.append(ConditionCheck(
        "MACD histogram expanding positive" if bullish else "MACD momentum bearish",
        ConditionStatus.PASS if macd_ok else ConditionStatus.FAIL,
    ))

    snap = build_chain_snapshot(chain, spot) if chain else None
    oi_ok = False
    if snap:
        oi_ok = snap.long_buildup_calls if bullish else snap.long_buildup_puts
    conditions.append(ConditionCheck(
        "OI long buildup at nearest strikes",
        ConditionStatus.PASS if oi_ok else ConditionStatus.FAIL if snap else ConditionStatus.FAIL,
        "calls" if bullish else "puts",
    ))

    vix_ok = vix_level is not None and vix_level < 22 and (vix_change is None or vix_change <= 1.5)
    conditions.append(ConditionCheck(
        "India VIX < 22, not spiking",
        ConditionStatus.PASS if vix_ok else ConditionStatus.FAIL,
        f"VIX={vix_level:.2f} Δ={vix_change:+.2f}" if vix_level and vix_change is not None else (
            f"VIX={vix_level:.2f}" if vix_level else "VIX unavailable"
        ),
    ))

    pcr_ok = snap is not None and 0.8 <= snap.pcr <= 1.2
    conditions.append(ConditionCheck(
        "PCR 0.8–1.2",
        ConditionStatus.PASS if pcr_ok else ConditionStatus.FAIL,
        f"PCR={snap.pcr:.2f}" if snap else "chain unavailable",
    ))

    suggested = None
    strike_ok = False
    if chain:
        suggested = pick_strike(
            chain, spot, is_call=bullish,
            delta_min=0.45, delta_max=0.55, min_dte=2, prefer_itm=True,
        )
        strike_ok = suggested is not None
    conditions.append(ConditionCheck(
        "Strike ATM/1-ITM, Δ 0.45–0.55, ≥2 DTE",
        ConditionStatus.PASS if strike_ok else ConditionStatus.FAIL,
        suggested.symbol if suggested else "no qualifying strike",
    ))

    direction = "bullish" if bullish else "bearish"
    return _finalize("A", title, conditions, direction=direction, suggested=suggested)


def evaluate_setup_b(
    day_5m: pd.DataFrame,
    prev_day: pd.DataFrame,
    chain: OptionChain | None,
    spot: float,
    vix_level: float | None,
    events: list[str] | None,
    now: datetime | None = None,
) -> ConfluenceSetupResult:
    """Opening Range Breakout with Confirmation."""
    title = "Setup B — ORB with Confirmation"
    now = now or datetime.now()
    orb_time = now.replace(hour=9, minute=30, second=0, microsecond=0)

    if now < orb_time:
        conditions = [ConditionCheck("ORB window 9:15–9:30", ConditionStatus.FAIL, "ORB window not complete")]
        return _finalize("B", title, conditions, watch=True, notes="stop=opposite ORB; book 50% at 1:1")

    conditions: list[ConditionCheck] = []
    orb: ORBRange = compute_orb(day_5m)
    conditions.append(ConditionCheck(
        "ORB marked 9:15–9:30",
        ConditionStatus.PASS if orb.complete else ConditionStatus.FAIL,
        f"H={orb.high:.1f} L={orb.low:.1f}",
    ))

    frame_5m = enrich_frame(day_5m)
    break_dir = orb_breakout_direction(day_5m, orb)
    conditions.append(ConditionCheck(
        "Full 5m close beyond ORB",
        ConditionStatus.PASS if break_dir else ConditionStatus.FAIL,
        break_dir or "inside range",
    ))

    vwap_aligned = False
    if break_dir and not frame_5m.empty:
        last = frame_5m.iloc[-1]
        vw = last.get("vwap")
        if not pd.isna(vw):
            close = float(last["close"])
            vwap_aligned = (break_dir == "bullish" and close > float(vw)) or (
                break_dir == "bearish" and close < float(vw)
            )
    conditions.append(ConditionCheck(
        "Break aligned with VWAP",
        ConditionStatus.PASS if vwap_aligned else ConditionStatus.FAIL if break_dir else ConditionStatus.FAIL,
    ))

    ema_confirms = False
    if break_dir == "bullish":
        ema_confirms = ema21_sloping_up(frame_5m) and float(frame_5m["close"].iloc[-1]) > float(frame_5m["ema21"].iloc[-1])
    elif break_dir == "bearish":
        ema_confirms = ema21_sloping_down(frame_5m) and float(frame_5m["close"].iloc[-1]) < float(frame_5m["ema21"].iloc[-1])
    conditions.append(ConditionCheck(
        "EMA-21 confirms on 5m",
        ConditionStatus.PASS if ema_confirms else ConditionStatus.FAIL,
    ))

    vol_ok = False
    post = day_5m.between_time("09:30", "15:30")
    if len(post) >= 1 and break_dir:
        avg_vol = avg_first_30min_volume(day_5m)
        vol_ok = float(post.iloc[-1]["volume"]) > avg_vol if avg_vol > 0 else False
    conditions.append(ConditionCheck(
        "Breakout volume > avg first 30 min",
        ConditionStatus.PASS if vol_ok else ConditionStatus.FAIL,
    ))

    cpr = compute_cpr(prev_day) if not prev_day.empty else None
    cpr_narrow = cpr is not None and cpr.width_pct < 1.2
    conditions.append(ConditionCheck(
        "CPR not wide (< 1.2% of spot)",
        ConditionStatus.PASS if cpr_narrow else ConditionStatus.FAIL,
        f"width={cpr.width_pct:.2f}%" if cpr else "n/a",
    ))

    event_block = bool(events)
    vix_block = vix_level is not None and vix_level > 25
    stand_aside = event_block or vix_block
    conditions.append(ConditionCheck(
        "VIX ≤ 25, no major event",
        ConditionStatus.PASS if not stand_aside else ConditionStatus.FAIL,
        "event scheduled" if event_block else f"VIX={vix_level:.1f}" if vix_block else "clear",
    ))

    direction = break_dir or "neutral"
    is_call = break_dir == "bullish"
    suggested = pick_strike(chain, spot, is_call, 0.45, 0.55, 2) if chain and break_dir else None
    return _finalize(
        "B", title, conditions,
        direction=direction,
        suggested=suggested,
        notes="stop=opposite ORB side; book 50% at 1:1, trail remainder",
    )


def evaluate_setup_c(
    day_5m: pd.DataFrame,
    chain: OptionChain | None,
    spot: float,
) -> ConfluenceSetupResult:
    """Reversal via Divergence + Max-Pain Magnet."""
    title = "Setup C — Reversal via Divergence + Max Pain"
    conditions: list[ConditionCheck] = []

    frame_15m = enrich_frame(resample_bars(day_5m, "15min"))
    div = detect_rsi_divergence(frame_15m)
    conditions.append(ConditionCheck(
        "RSI divergence at extreme",
        ConditionStatus.PASS if div else ConditionStatus.FAIL,
        div or "none detected",
    ))

    macd_ok = False
    if div == "bearish":
        macd_ok = macd_histogram_shrinking_against_trend(frame_15m, bullish=True)
    elif div == "bullish":
        macd_ok = macd_histogram_shrinking_against_trend(frame_15m, bullish=False)
    conditions.append(ConditionCheck(
        "MACD histogram shrinking vs trend",
        ConditionStatus.PASS if macd_ok else ConditionStatus.FAIL if div else ConditionStatus.FAIL,
    ))

    snap = build_chain_snapshot(chain, spot) if chain else None
    mp_ok = snap is not None and snap.spot_vs_max_pain_pct > 0.5
    conditions.append(ConditionCheck(
        "Spot far from max pain (> 0.5%)",
        ConditionStatus.PASS if mp_ok else ConditionStatus.FAIL,
        f"|spot-MP|={snap.spot_vs_max_pain_pct:.2f}%" if snap else "chain unavailable",
    ))

    oi_wall_ok = False
    if snap and div == "bearish":
        oi_wall_ok = snap.call_oi_wall is not None and snap.call_oi_wall >= spot * 0.998
    elif snap and div == "bullish":
        oi_wall_ok = snap.put_oi_wall is not None and snap.put_oi_wall <= spot * 1.002
    conditions.append(ConditionCheck(
        "Heavy OI wall at resistance/support",
        ConditionStatus.PASS if oi_wall_ok else ConditionStatus.FAIL if div else ConditionStatus.FAIL,
    ))

    pivot = latest_swing_pivot(frame_15m) if div else None
    rev_ok = pivot is not None and div is not None and reversal_past_pivot(frame_15m, pivot, div)
    conditions.append(ConditionCheck(
        "Reversal close past swing pivot",
        ConditionStatus.PASS if rev_ok else ConditionStatus.FAIL if div else ConditionStatus.FAIL,
        f"pivot={pivot.level:.1f} ({pivot.kind})" if pivot else "n/a",
    ))

    is_call = div == "bullish"
    suggested = pick_strike(chain, spot, is_call, 0.35, 0.40, 2) if chain and div else None
    strike_ok = suggested is not None and (suggested.delta or 0) >= 0.25
    conditions.append(ConditionCheck(
        "Strike slightly OTM, Δ 0.35–0.40",
        ConditionStatus.PASS if strike_ok else ConditionStatus.FAIL,
        suggested.symbol if suggested else "no qualifying strike",
    ))

    direction = "bullish" if div == "bullish" else "bearish" if div == "bearish" else "neutral"
    return _finalize("C", title, conditions, direction=direction, suggested=suggested)
