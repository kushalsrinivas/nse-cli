"""Turn a setup result into the text state Laya reads.

Laya is a text encoder: it reads words far better than it reads raw
numbers. The setup engines already explain themselves in words (each
condition carries a name, pass/fail and a detail string), so the state is
that audit trail plus a line of market context — not a feature vector.

Duck-typed over both engines' result types:
- model.confluence.types.ConfluenceSetupResult   (intraday A/B/C)
- model.overnight_setups.types.OvernightSetupResult (ON-A..ON-D)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SetupState:
    source: str                      # confluence | overnight-setup | overnight | premarket
    setup_id: str
    title: str
    direction: str                   # bullish | bearish | neutral
    decision: str                    # GO | NO-GO | WATCH
    confidence: float                # 0-100, engine's own pass share
    conditions: tuple[tuple[str, str, str], ...] = ()   # (name, status, detail)
    rationale: str = ""
    context: dict = field(default_factory=dict)

    def as_text(self) -> str:
        lines = [
            f"{self.source} setup {self.setup_id}: {self.title}.",
            f"Proposed direction: {self.direction}. Engine decision: "
            f"{self.decision} ({self.confidence:.0f}% of conditions met).",
        ]
        ctx = _context_text(self.context)
        if ctx:
            lines.append(ctx)
        for name, status, detail in self.conditions:
            if status == "info":             # evidence line, not a pass/fail check
                lines.append(f"- {name}" + (f": {detail}" if detail else ""))
                continue
            verdict = {"pass": "met", "fail": "NOT met"}.get(status, "unknown")
            lines.append(f"- {name}: {verdict}" + (f" ({detail})" if detail else ""))
        if self.rationale:
            lines.append(f"Rationale: {self.rationale}")
        return "\n".join(lines)


def setup_state(result, *, source: str, context: dict | None = None) -> SetupState:
    """Build a SetupState from either engine's result object."""
    decision = getattr(result, "decision", "")
    decision = getattr(decision, "value", decision)           # enum or str
    confidence = getattr(result, "confidence_score",
                         getattr(result, "confidence", 0.0)) or 0.0
    title = getattr(result, "title", None) or getattr(result, "name", "")
    rationale = (getattr(result, "decision_rationale", None)
                 or getattr(result, "rationale", "") or "")
    conds = tuple(
        (c.name, getattr(c.status, "value", str(c.status)), c.detail or "")
        for c in getattr(result, "conditions", []) or []
    )
    return SetupState(
        source=source,
        setup_id=str(result.setup_id),
        title=str(title),
        direction=str(getattr(result, "direction", "neutral") or "neutral"),
        decision=str(decision),
        confidence=float(confidence),
        conditions=conds,
        rationale=str(rationale),
        context=dict(context or {}),
    )


def _context_text(ctx: dict) -> str:
    parts = []
    spot = ctx.get("spot")
    if spot:
        parts.append(f"NIFTY at {spot:,.1f}")
    vix, vchg = ctx.get("vix"), ctx.get("vix_change")
    if vix is not None:
        v = f"India VIX {vix:.1f}"
        if vchg is not None:
            v += f" ({vchg:+.1f}% today, {_vix_word(vchg)})"
        parts.append(v)
    events = ctx.get("events") or []
    if events:
        parts.append("scheduled risk: " + "; ".join(events))
    return ("Market: " + ", ".join(parts) + ".") if parts else ""


def _vix_word(chg: float) -> str:
    if chg >= 8:
        return "volatility spiking"
    if chg >= 3:
        return "volatility rising"
    if chg <= -3:
        return "volatility easing"
    return "volatility steady"
