"""Bearish pipeline rules (plan §4.3, right column).

Bearish is first-class: its own thresholds (`[bearish]` in the config), its
own confirmations, and its own executability. A bearish *view* is always
recorded; whether it is an executable *trade* is a separate decision.

| Element               | Bearish rule                                                         |
|-----------------------|----------------------------------------------------------------------|
| Zone                  | last green/doji candle at the leg high before a BOS/CHoCH down (core)|
| Trigger               | 5m CHoCH down after the touch, or 15:15 close in/below the zone (core)|
| Volume confirmation   | selling pressure: down-volume share ≥ min_directional_volume_share   |
|                       | AND ≥ 50% of leg bars close in the lower third of their range        |
| Relative weakness     | RS vs benchmark over 5 and 20 sessions both < 0; vs sector reported  |
| Structure             | LH + LL on the detection timeframe                                   |
| Context               | alignment −1…+1, scored, never auto-rejected                         |
| Gap                   | open below zone_low − 1 ATR → GAP_AWAY (breakdown already happened)  |
| Executability         | intraday equity: MIS cash short, short future / PE if F&O;           |
|                       | overnight: F&O names only (short future / PE) — no overnight cash    |
|                       | shorts in India; index: short future / PE when derivatives exist     |
"""

from __future__ import annotations

from market_platform.signals.shared import (
    Confirmation,
    DirectionRules,
    SignalCandidate,
    close_location_share,
    swing_structure,
    volume_share,
)

LOWER_THIRD_MIN = 0.5


class BearishRules(DirectionRules):
    direction = "bearish"

    def confirmations(self, cand: SignalCandidate, *, zone, leg_bars, swings,
                      ctx_view: dict, alignment: float) -> list[Confirmation]:
        out = []
        share = volume_share(leg_bars, up=False)
        low3 = close_location_share(leg_bars, lower=True)
        thr = self.cfg.min_directional_volume_share
        ok = None if share is None else (share >= thr and (low3 or 0) >= LOWER_THIRD_MIN)
        out.append(Confirmation("VOLUME_SELLING", share, thr, ok,
                                f"down-volume share; lower-third closes {low3}"))
        rs5, rs20 = ctx_view.get("rs5_bench"), ctx_view.get("rs20_bench")
        weak = None if rs5 is None or rs20 is None else (rs5 < 0 and rs20 < 0)
        out.append(Confirmation("RS_WEAK_BENCH", rs20, 0.0, weak,
                                f"5d {rs5} / 20d {rs20} pts vs benchmark"))
        rs_sec = ctx_view.get("rs20_sector")
        out.append(Confirmation("RS_WEAK_SECTOR", rs_sec, 0.0, None if rs_sec is None else rs_sec < 0,
                                "20d vs sector index"))
        st = swing_structure(swings, cand.detected_at, up=False) if swings is not None else None
        out.append(Confirmation("STRUCTURE_LH_LL", None if st is None else int(st), 1, st,
                                "last two swing highs and lows falling"))
        out.append(Confirmation("CTX_ALIGNMENT", alignment, 0.0, None, "market/sector/RS agreement"))
        return out

    def gap_reject(self, cand: SignalCandidate, ctx_view: dict) -> str:
        gap_open = ctx_view.get("day_open")
        if gap_open is not None and cand.atr and gap_open < cand.zone_low - cand.atr:
            return f"open {gap_open:.2f} < zone_low {cand.zone_low} − 1 ATR ({cand.atr:.2f})"
        return ""

    def executability(self, cand: SignalCandidate, instrument: dict | None) -> tuple[bool, list[str], str]:
        inst = instrument or {}
        fno = bool(inst.get("fno_eligible"))
        if inst.get("kind") == "index":
            if not inst.get("deriv_underlying"):
                return False, [], "INDEX_NO_DERIVATIVES"
            return True, ["fut_short", "pe"], ""
        if cand.horizon == "overnight":
            if not fno:
                return False, [], "NO_FNO_OVERNIGHT_SHORT"
            return True, ["fut_short", "pe"], ""
        routes = ["cash_mis_short"]
        if fno:
            routes += ["fut_short", "pe"]
        return True, routes, ""
