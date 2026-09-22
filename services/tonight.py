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
    laya: object = None                   # services.laya.LayaRun when --laya
    notices: list[tuple[str, str]] = field(default_factory=list)


def run_tonight(*, period: str = "2y", source: str = "auto",
                cperiod: str = "6mo", no_breadth: bool = False,
                events: list[str] | None = None,
                journal: bool = False, laya: bool = False,
                laya_enforce: bool = False,
                laya_filter=None) -> TonightResult:
    """Full overnight card. Dry run unless `journal=True` (records).

    `laya=True` asks Laya to judge the finished card (shadow unless
    `laya_enforce`). It runs before journaling, so an enforced veto is what
    gets recorded.
    """
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

    post_decision = None
    if laya:
        from datetime import date

        from model.laya_filter.cards import overnight_card_state, veto_overnight
        from services.laya import judge_card

        def post_decision(setup):
            vix = None
            try:
                from data import source as datasrc
                vix = datasrc.get_india_vix()[0] or None
            except Exception:
                pass
            result.laya = judge_card(
                overnight_card_state(setup, events=events, vix=vix),
                enforce=laya_enforce, journal=journal,
                run_id=f"ON-{date.today().isoformat()}",
                apply_veto=lambda v: veto_overnight(setup, v),
                laya_filter=laya_filter)
            result.notices.extend(result.laya.notices)
    result.setup = build_overnight_setup(
        bundle.candles, bundle.chain, signals=signals,
        events=events or None, breadth=breadth.snap, record=journal,
        fut_basis_bps=meta.get("basis_bps"),
        fut_oi_chg_pct=meta.get("fut_oi_chg_pct"),
        post_decision=post_decision)
    return result
