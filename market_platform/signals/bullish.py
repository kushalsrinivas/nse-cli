"""Bullish pipeline rules (plan §4.3, left column).

Same methodology as bearish (signals/shared.py); these are the
direction-specific parts. Thresholds come from `[bullish]` in the config.

| Element               | Bullish rule                                                        |
|-----------------------|---------------------------------------------------------------------|
| Zone                  | last red/doji candle at the leg low before a BOS/CHoCH up (core)    |
| Trigger               | 5m CHoCH up after the touch, or 15:15 close in/above the zone (core) |
| Volume confirmation   | up-volume share of the impulse leg ≥ min_directional_volume_share   |
| Relative strength     | RS vs benchmark over 5 and 20 sessions both > 0; vs sector reported  |
| Structure             | HH + HL on the detection timeframe                                  |
| Context               | alignment −1…+1, scored, never auto-rejected                        |
| Gap                   | open above zone_high + 1 ATR → GAP_AWAY (the move already happened); |
|                       | a gap *into* the zone waits for the first 5m close (core trigger)   |
| Executability         | equity: cash (MIS intraday / CNC overnight), long future, CE if F&O; |
|                       | index: long future / CE only when a derivative underlying exists    |
"""

from __future__ import annotations

from market_platform.signals.shared import (
    Confirmation,
    DirectionRules,
    SignalCandidate,
    swing_structure,
    volume_share,
)


class BullishRules(DirectionRules):
    direction = "bullish"

    def confirmations(self, cand: SignalCandidate, *, zone, leg_bars, swings,
                      ctx_view: dict, alignment: float) -> list[Confirmation]:
        out = []
        share = volume_share(leg_bars, up=True)
        thr = self.cfg.min_directional_volume_share
        out.append(Confirmation("VOLUME_DEMAND", share, thr,
                                None if share is None else share >= thr,
                                "up-volume share of the impulse leg"))
        rs5, rs20 = ctx_view.get("rs5_bench"), ctx_view.get("rs20_bench")
        ok = None if rs5 is None or rs20 is None else (rs5 > 0 and rs20 > 0)
        out.append(Confirmation("RS_BENCH", rs20, 0.0, ok, f"5d {rs5} / 20d {rs20} pts vs benchmark"))
        rs_sec = ctx_view.get("rs20_sector")
        out.append(Confirmation("RS_SECTOR", rs_sec, 0.0, None if rs_sec is None else rs_sec > 0,
                                "20d vs sector index"))
        st = swing_structure(swings, cand.detected_at, up=True) if swings is not None else None
        out.append(Confirmation("STRUCTURE_HH_HL", None if st is None else int(st), 1, st,
                                "last two swing highs and lows rising"))
        out.append(Confirmation("CTX_ALIGNMENT", alignment, 0.0, None, "market/sector/RS agreement"))
        return out

    def gap_reject(self, cand: SignalCandidate, ctx_view: dict) -> str:
        gap_open = ctx_view.get("day_open")
        if gap_open is not None and cand.atr and gap_open > cand.zone_high + cand.atr:
            return f"open {gap_open:.2f} > zone_high {cand.zone_high} + 1 ATR ({cand.atr:.2f})"
        return ""

    def executability(self, cand: SignalCandidate, instrument: dict | None) -> tuple[bool, list[str], str]:
        inst = instrument or {}
        fno = bool(inst.get("fno_eligible"))
        if inst.get("kind") == "index":
            if not inst.get("deriv_underlying"):
                return False, [], "INDEX_NO_DERIVATIVES"
            return True, ["fut_long", "ce"], ""
        routes = ["cash_mis" if cand.horizon == "intraday" else "cash_cnc"]
        if fno:
            routes += ["fut_long", "ce"]
        return True, routes, ""
