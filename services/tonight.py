"""The EOD 'tonight' workflow as pure data: fetch once, screen, decide.

No console output — notices carry displayable messages. Renderers live in
model_cli (Rich) and, later, the TUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: NIFTY-only screen below this is not worth a 50-ticker breadth fetch.
SCREEN_SCORE = 50.0


@dataclass
class TonightResult:
    setup: object = None
    screen: object = None
    snap: object = None
    kite_meta: dict | None = None
    notices: list[tuple[str, str]] = field(default_factory=list)


def run_tonight(*, period: str = "2y", source: str = "auto",
                cperiod: str = "6mo", no_breadth: bool = False,
                events: list[str] | None = None,
                journal: bool = False) -> TonightResult:
    """Full overnight card. Dry run unless `journal=True` (records)."""
    from model.overnight import collect_overnight_signals
    from model.overnight_card import build_overnight_setup
    from model.pipeline import evaluate
    from services.bundles import breadth_snapshot, tonight_bundle

    events = list(events or [])
    result = TonightResult()
    bundle = tonight_bundle(period=period, source=source)
    result.notices.extend(bundle.notices)
    result.kite_meta = bundle.kite_meta

    screen = evaluate(candles=bundle.candles, chain=bundle.chain,
                      use_breadth=False, persist=False)
    result.screen = screen

    breadth = breadth_snapshot(
        enabled=not no_breadth, source=source, cperiod=cperiod,
        nifty_candles=bundle.candles, screen_score=screen.composite.score,
        screen_threshold=SCREEN_SCORE)
    if no_breadth:
        result.notices.append(("info", "breadth disabled (--no-breadth)"))
    result.notices.extend(breadth.notices)
    result.snap = breadth.snap
    if events:
        result.notices.append(
            ("warn", "EVENT NIGHT: " + "; ".join(events) + " — gap "
                     "distribution is un-modelable; standing rule is NO-GO."))

    signals = collect_overnight_signals(bundle.candles)
    meta = bundle.kite_meta or {}
    result.setup = build_overnight_setup(
        bundle.candles, bundle.chain, signals=signals,
        events=events or None, breadth=breadth.snap, record=journal,
        fut_basis_bps=meta.get("basis_bps"),
        fut_oi_chg_pct=meta.get("fut_oi_chg_pct"))
    return result
