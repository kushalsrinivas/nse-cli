"""NIFTY-vs-constituents divergence detection.

Divergence is where bottom-up data earns its keep: a NIFTY move the
constituents refuse to confirm (or a selloff they refuse to join) says
something about continuation vs reversal that the index series alone
cannot. Each flag carries a severity (0-3) and a plain-language rationale
so it can be audited, not just scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model.breadth.aggregate import BreadthSnapshot


@dataclass
class DivergenceSignal:
    flag: str                       # machine-readable name
    severity: int                   # 0 none, 1 watch, 2 caution, 3 strong
    direction: str                  # "bullish" | "bearish" | "neutral"
    rationale: str = ""
    detail: dict = field(default_factory=dict)


def detect_divergence(
    snap: BreadthSnapshot,
    nifty_close_pos_60: float | None = None,
    nifty_new_high_20: bool = False,
    nifty_new_low_20: bool = False,
) -> list[DivergenceSignal]:
    """Inspect one EOD snapshot for index-vs-breadth disagreement."""
    out: list[DivergenceSignal] = []
    if not snap.sufficient or snap.nifty_ret_1d is None:
        return out
    r = snap.nifty_ret_1d
    adv = snap.adv_pct or 0
    conf = snap.confirming_pct or 0
    accel = snap.breadth_accel

    if r > 0.3 and adv < 45 and conf < 50:
        out.append(DivergenceSignal(
            "bull_trap_risk", 3 if adv < 38 else 2, "bearish",
            f"NIFTY {r:+.2f}% but only {adv:.0f}% of constituents advanced "
            f"({conf:.0f}% confirming) — narrow leadership, fade risk.",
            {"adv_pct": adv, "confirming_pct": conf}))
    elif r > 0.3 and snap.heavy_drag is False and (snap.top5_contrib_share or 0) >= 65:
        out.append(DivergenceSignal(
            "concentrated_rally", 2, "neutral",
            f"NIFTY {r:+.2f}% with top-5 names producing "
            f"{snap.top5_contrib_share:.0f}% of the proxy move — "
            "continuation needs broader confirmation.",
            {"top5_share": snap.top5_contrib_share}))

    if r < -0.3 and adv > 55:
        out.append(DivergenceSignal(
            "bear_trap_relief", 2, "bullish",
            f"NIFTY {r:+.2f}% but {adv:.0f}% of constituents held up — "
            "selling looks index-led, bounce possible.",
            {"adv_pct": adv}))

    if nifty_new_high_20:
        hi = snap.new_high_pct or 0
        if hi >= 20:
            out.append(DivergenceSignal(
                "breakout_confirm", 1, "bullish",
                f"NIFTY 20d breakout confirmed by {hi:.0f}% of constituents "
                "at 20d highs — broad participation.",
                {"new_high_pct": hi}))
        elif hi < 10:
            out.append(DivergenceSignal(
                "breakout_divergence", 3, "bearish",
                f"NIFTY 20d breakout with only {hi:.0f}% of constituents at "
                "highs — classic thrust without breadth, reversal risk.",
                {"new_high_pct": hi}))
    if nifty_new_low_20:
        lo = snap.new_low_pct or 0
        if lo >= 20:
            out.append(DivergenceSignal(
                "breakdown_confirm", 1, "bearish",
                f"NIFTY 20d breakdown confirmed by {lo:.0f}% at 20d lows.",
                {"new_low_pct": lo}))
        elif lo < 10:
            out.append(DivergenceSignal(
                "breakdown_divergence", 2, "bullish",
                f"NIFTY 20d breakdown with only {lo:.0f}% of constituents at "
                "lows — washout candidate.",
                {"new_low_pct": lo}))

    if (nifty_close_pos_60 is not None and nifty_close_pos_60 >= 0.9
            and accel is not None and accel < -5):
        out.append(DivergenceSignal(
            "resistance_weakness", 2, "bearish",
            f"NIFTY pinned near 60d highs (pos {nifty_close_pos_60:.2f}) while "
            f"breadth fades ({accel:+.0f}pp acceleration) — late-stage tape.",
            {"close_pos_60": nifty_close_pos_60, "accel": accel}))

    if snap.heavy_drag and abs(r) > 0.3:
        out.append(DivergenceSignal(
            "heavyweight_divergence", 2,
            "bearish" if r > 0 else "bullish",
            f"Heavies average {snap.heavy_avg_ret:+.2f}% against a "
            f"{r:+.2f}% NIFTY move — index math, not market direction.",
            {"heavy_avg_ret": snap.heavy_avg_ret}))
    return out


def worst_severity(flags: list[DivergenceSignal]) -> int:
    return max((f.severity for f in flags), default=0)
