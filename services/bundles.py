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


def tonight_bundle(*, period: str = "2y", source: str = "yahoo") -> TonightBundle:
    """NIFTY history + option chain, once, either source."""
    if (source or "yahoo") == "kite":
        return _tonight_bundle_kite()
    from data import nifty, options as opts
    bundle = TonightBundle()
    bundle.candles = nifty.fetch_history(period=period).candles
    try:
        bundle.chain = opts.fetch_chain()
    except Exception as exc:
        bundle.notices.append(("warn", f"option chain unavailable: {exc}"))
    return bundle


def _tonight_bundle_kite() -> TonightBundle:
    from data.kite import instruments as ki
    from data.kite.auth import KiteAuthError
    from data.kite.chain import KiteChainProvider
    from data.kite.eod import eod_context
    from data.kite.rest import KiteRest
    from data.kite.store import InstrumentStore

    bundle = TonightBundle()
    rest = KiteRest()  # raises KiteAuthError w/o session; caller maps it
    store = InstrumentStore()
    ki.refresh_master(rest, store)
    ctx = eod_context("NIFTY", use_fut_volume=True, rest=rest, store=store)
    bundle.candles = ctx.candles
    bundle.notices.append(
        ("info", f"kite: {ctx.bars} bars · {ctx.volume_note} · "
                 f"basis {ctx.basis_bps}bp · fut ΔOI {ctx.fut_oi_chg_pct}%"))
    try:
        chain, _quotes = KiteChainProvider(rest, store).chain_for("NIFTY", ctx.spot)
        bundle.chain = chain
        bundle.notices.append(
            ("info", f"kite chain: {chain.expiries[0]} · {len(chain.rows)} strikes"))
    except Exception as exc:
        bundle.notices.append(("warn", f"kite chain failed ({exc}); NSE fallback"))
        try:
            from data import options as opts
            bundle.chain = opts.fetch_chain()
        except Exception as exc2:
            bundle.notices.append(("warn", f"option chain unavailable: {exc2}"))
    bundle.kite_meta = {"source": "kite", "basis_bps": ctx.basis_bps,
                        "fut_oi_chg_pct": ctx.fut_oi_chg_pct,
                        "volume_note": ctx.volume_note}
    return bundle


def breadth_snapshot(*, enabled: bool, source: str = "yahoo",
                     cperiod: str = "6mo", nifty_candles=None,
                     screen_score: float | None = None,
                     screen_threshold: float = 50.0) -> BreadthBundle:
    """Constituent snapshot, lazily gated on the technical screen.

    Returns an empty bundle (with a notice) when disabled, below the
    screen threshold, or on insufficient coverage — never raises for
    data problems.
    """
    out = BreadthBundle()
    if not enabled:
        return out
    if screen_score is not None and screen_score < screen_threshold:
        out.notices.append(
            ("info", f"breadth skipped: base score {screen_score:.0f} "
                     f"< screen {screen_threshold:.0f}"))
        return out
    try:
        if (source or "yahoo") == "kite":
            return _breadth_snapshot_kite(nifty_candles)
        from data.constituents import fetch_constituent_history
        from model.breadth.live import build_live_snapshot
        bundle = fetch_constituent_history(period=cperiod)
        if not bundle.sufficient:
            out.notices.append(
                ("warn", f"thin constituent coverage ({len(bundle.frames)} names, "
                         f"{bundle.weight_coverage * 100:.0f}% weight) — "
                         f"NIFTY-only baseline"))
            return out
        snap, flags, ctx = build_live_snapshot(nifty_candles, bundle)
    except Exception as exc:
        out.notices.append(
            ("warn", f"constituent data unavailable: {exc} — NIFTY-only baseline"))
        return out
    out.snap, out.flags, out.ctx = snap, flags, ctx
    return out


def _breadth_snapshot_kite(nifty_candles) -> BreadthBundle:
    from data.constituents import ConstituentBundle
    from data.kite.eod import constituent_frames
    from data.kite.rest import KiteRest
    from data.kite.store import InstrumentStore
    from model.breadth.live import build_live_snapshot
    from model.breadth.universe import get_universe

    out = BreadthBundle()
    rest, store = KiteRest(), InstrumentStore()
    shorts = [c.short for c in get_universe()]
    frames = constituent_frames(shorts, rest=rest, store=store)
    missing = [c.symbol for c in get_universe() if c.symbol not in frames]
    bundle = ConstituentBundle(frames=frames, missing=missing)
    if not bundle.sufficient:
        out.notices.append(
            ("warn", f"thin kite coverage ({len(frames)} names) — "
                     f"NIFTY-only baseline"))
        return out
    out.notices.append(
        ("info", f"kite breadth: {len(frames)} names, "
                 f"{bundle.weight_coverage * 100:.0f}% weight"))
    snap, flags, ctx = build_live_snapshot(nifty_candles, bundle)
    out.snap, out.flags, out.ctx = snap, flags, ctx
    return out
