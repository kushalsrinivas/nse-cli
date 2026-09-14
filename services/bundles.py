"""Shared market-data bundles: one place builds what commands need.

Collapses the duplicated `_fetch_*` helpers that accumulated in model_cli:
tonight bundles (history + chain, either source) and breadth snapshots
(lazy-gated on the technical screen). No console output — see notices.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TonightBundle:
    candles: list = field(default_factory=list)
    chain: object | None = None
    kite_meta: dict | None = None   # None on the yahoo path
    notices: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class BreadthBundle:
    snap: object | None = None
    flags: list = field(default_factory=list)
    ctx: dict = field(default_factory=dict)
    notices: list[tuple[str, str]] = field(default_factory=list)


def tonight_bundle(*, period: str = "2y", source: str = "auto") -> TonightBundle:
    """NIFTY history + option chain, once. Kite first, legacy fallback."""
    from data import source as datasrc

    bundle = TonightBundle()
    result = datasrc.get_nifty_history(period=period, source=source,
                                       use_fut_volume=True)
    bundle.candles = result.candles
    try:
        bundle.chain = datasrc.get_nifty_chain(source=source)
    except Exception as exc:
        bundle.notices.append(("warn", f"option chain unavailable: {exc}"))
    used_kite = datasrc.session_available() if source == "auto" else source == "kite"
    if used_kite:
        basis, oi_chg = datasrc.get_futures_snapshot("NIFTY")
        bundle.kite_meta = {"source": "kite", "basis_bps": basis,
                            "fut_oi_chg_pct": oi_chg,
                            "volume_note": "volume proxied from front future"}
        bundle.notices.append(("info", f"kite: {len(bundle.candles)} bars · "
                                       f"basis {basis}bp · fut ΔOI {oi_chg}%"))
    return bundle


def breadth_snapshot(*, enabled: bool, source: str = "auto",
                     cperiod: str = "6mo", nifty_candles=None,
                     screen_score: float | None = None,
                     screen_threshold: float = 50.0) -> BreadthBundle:
    """Constituent snapshot, lazily gated on the technical screen.

    Returns an empty bundle (with a notice) when disabled, below the
    screen threshold, or on insufficient coverage — never raises for
    data problems. Kite first, Yahoo batch fallback.
    """
    from data import source as datasrc
    from model.breadth.live import build_live_snapshot

    out = BreadthBundle()
    if not enabled:
        return out
    if screen_score is not None and screen_score < screen_threshold:
        out.notices.append(
            ("info", f"breadth skipped: base score {screen_score:.0f} "
                     f"< screen {screen_threshold:.0f}"))
        return out
    try:
        bundle = datasrc.get_constituent_bundle(period=cperiod, source=source)
        if not bundle.sufficient:
            out.notices.append(
                ("warn", f"thin constituent coverage ({len(bundle.frames)} names, "
                         f"{bundle.weight_coverage * 100:.0f}% weight) — "
                         f"NIFTY-only baseline"))
            return out
        if datasrc.session_available() and source != "yahoo":
            out.notices.append(
                ("info", f"kite breadth: {len(bundle.frames)} names, "
                         f"{bundle.weight_coverage * 100:.0f}% weight"))
        snap, flags, ctx = build_live_snapshot(nifty_candles, bundle)
    except Exception as exc:
        out.notices.append(
            ("warn", f"constituent data unavailable: {exc} — NIFTY-only baseline"))
        return out
    out.snap, out.flags, out.ctx = snap, flags, ctx
    return out
