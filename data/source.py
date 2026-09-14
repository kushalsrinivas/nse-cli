"""Market-data source router: Kite first, Yahoo/NSE fallback.

Policy (explicit user call): every NSE-sourced feed — NIFTY history,
constituent history, NSE + stock option chains/expiries, intraday bars,
India VIX — resolves to Kite when a session is available and falls back
to the legacy Yahoo/NSE-scrape path otherwise. Macro/global feeds
(`model/macro.py`: US indices, crude, FX, Asia) are deliberately untouched.

`source` tri-state, accepted by every getter:
- "auto"   (default): Kite when a session is valid, else legacy silently.
- "kite":   Kite or raise (fail loud; used by --source kite commands).
- "yahoo":  legacy path only (parity baselines, offline work).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

from config import SETTINGS

log = logging.getLogger(__name__)

SOURCE_AUTO = "auto"
SOURCE_KITE = "kite"
SOURCE_YAHOO = "yahoo"

_PERIOD_DAYS = {
    "5d": 5, "60d": 60, "1mo": 30, "3mo": 90, "6mo": 180,
    "1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "730d": 730, "max": 3650,
}

# App interval -> Kite interval (+ optional resample rule). Anything else
# falls back to Yahoo.
_KITE_INTERVALS = {
    "1d": ("day", None),
    "5m": ("minute", "5min"),
    "15m": ("minute", "15min"),
    "30m": ("minute", "30min"),
    "60m": ("minute", "60min"),
    "1h": ("minute", "60min"),
}

_MASTER_MAX_AGE_HOURS = 20

_vix_cache: dict = {"at": 0.0, "value": (None, None)}
VIX_CACHE_TTL = 120


def session_available() -> bool:
    """True when Kite is actually usable: creds present + session valid."""
    try:
        from data.kite.auth import credentials, load_session, session_valid
        credentials()
        return bool(session_valid(load_session()))
    except Exception:
        return False


def want_kite(source: str = SOURCE_AUTO) -> bool:
    """Resolve the tri-state. Explicit "kite" without a session raises."""
    source = (source or SOURCE_AUTO).lower()
    if source == SOURCE_YAHOO:
        return False
    if source == SOURCE_KITE:
        if not session_available():
            from data.kite.auth import KiteAuthError
            raise KiteAuthError(
                "source=kite requested but no valid session — run "
                "`model_cli.py kite-login` (sessions expire 6 AM IST daily)")
        return True
    return session_available()


def ensure_master(rest=None, store=None, max_age_hours: int = _MASTER_MAX_AGE_HOURS):
    """Instrument master, refreshing from Kite only when stale."""
    from data.kite import instruments as ki
    from data.kite.rest import KiteRest
    from data.kite.store import InstrumentStore

    rest = rest or KiteRest()
    store = store or InstrumentStore()
    fresh = False
    try:
        dates = store.as_of_dates()
        if dates:
            latest = datetime.strptime(dates[0], "%Y-%m-%d")
            fresh = (datetime.now() - latest) < timedelta(hours=max_age_hours)
    except Exception:
        fresh = False
    if not fresh:
        ki.refresh_master(rest, store)
    return store


def _kite_history_candles(token: int, period: str, interval: str,
                          rest=None):
    """Kite day/minute history as repo Candles (+ resampled frame)."""
    from data.kite.rest import KiteRest, history_to_candles, history_to_frame

    if interval not in _KITE_INTERVALS:
        raise ValueError(f"interval {interval!r} not supported via Kite")
    kite_iv, resample_rule = _KITE_INTERVALS[interval]
    days = _PERIOD_DAYS.get(period or SETTINGS.period, 365)
    to = datetime.now()
    frm = to - timedelta(days=days)
    rest = rest or KiteRest()
    records = rest.historical(token, kite_iv, frm, to)
    if resample_rule is None:
        return history_to_candles(records), None
    frame = history_to_frame(records)
    if frame.empty:
        return [], frame
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    resampled = frame.resample(resample_rule).agg(agg).dropna(subset=["close"])
    resampled.index = resampled.index.tz_localize(None)
    candles = history_to_candles(
        [{"date": ts.to_pydatetime(), **row.dropna().to_dict()}
         for ts, row in resampled.iterrows()])
    return candles, resampled


def get_nifty_history(period=None, interval=None, symbol: str = SETTINGS.symbol,
                      use_cache: bool = True, source: str = SOURCE_AUTO,
                      use_fut_volume: bool = False):
    """NIFTY history: Kite day/minute candles first, Yahoo fallback.

    Index packets carry no volume, so Kite index bars arrive with volume 0
    (same as Yahoo ^NSEI today). Pass use_fut_volume=True (tonight EOD) to
    proxy volume from the front future instead — the volume gate then sees
    real participation.
    """
    from data import nifty
    from data.nifty import HistoryResult, Quote

    period = period or SETTINGS.period
    interval = interval or SETTINGS.interval
    if symbol != SETTINGS.symbol or not want_kite(source):
        return nifty.fetch_history(period=period, interval=interval,
                                   symbol=symbol, use_cache=use_cache)
    try:
        from data.kite import instruments as ki
        from data.kite.rest import KiteRest
        from data.kite.store import InstrumentStore

        rest, store = KiteRest(), ensure_master()
        token = ki.nifty_spot_token(store)
        if token is None:
            raise ValueError("NIFTY spot token missing from master")
        candles, _ = _kite_history_candles(token, period, interval, rest)
        if len(candles) < 2:
            raise ValueError("empty Kite history for NIFTY")
        if use_fut_volume:
            candles = _with_fut_volume(candles, rest, store)
        last, prev = candles[-1], candles[-2]
        now = datetime.now()
        quote = Quote(
            price=last.close, previous_close=prev.close,
            change=round(last.close - prev.close, 2),
            change_pct=round((last.close - prev.close) / prev.close * 100, 2)
            if prev.close else None,
            day_open=last.open, day_high=last.high, day_low=last.low,
            volume=None, fetched_at=now)
        log.info("nifty history via kite (%d bars)", len(candles))
        return HistoryResult(candles=candles, quote=quote, period=period,
                             interval=interval, fetched_at=now, from_cache=False)
    except Exception as exc:
        log.warning("kite nifty history failed (%s); yahoo fallback", exc)
        return nifty.fetch_history(period=period, interval=interval,
                                   symbol=symbol, use_cache=use_cache)


def _with_fut_volume(candles, rest, store):
    """Rebuild candles with front-future volumes aligned by date."""
    import pandas as pd

    from data.kite import instruments as ki
    from data.nifty import Candle

    futs = ki.futures_chain(store, "NIFTY")
    if not futs:
        return candles
    to = datetime.now()
    recs = rest.historical(futs[0].instrument_token, "day",
                           to - timedelta(days=_PERIOD_DAYS.get("2y", 730)), to)
    vols = {}
    for r in recs:
        try:
            ts = pd.Timestamp(r["date"])
            ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
            vols[ts.normalize()] = int(r.get("volume") or 0)
        except (KeyError, TypeError, ValueError):
            continue
    if not vols:
        return candles
    out = []
    for c in candles:
        day = pd.Timestamp(c.timestamp).normalize()
        out.append(Candle(timestamp=c.timestamp, open=c.open, high=c.high,
                          low=c.low, close=c.close,
                          volume=vols.get(day, c.volume)))
    log.info("nifty volume proxied from front future")
    return out


def get_futures_snapshot(underlying: str):
    """(basis_bps, oi_day_chg_pct) or (None, None). Never raises."""
    try:
        from data.kite import instruments as ki
        from data.kite.rest import KiteRest
        from data.kite.store import InstrumentStore

        rest, store = KiteRest(), ensure_master()
        futs = ki.futures_chain(store, underlying)
        if not futs:
            return None, None
        to = datetime.now()
        recs = rest.historical(futs[0].instrument_token, "day",
                               to - timedelta(days=10), to, oi=True)
        closes = [(r["close"], r.get("oi")) for r in recs if r.get("close")]
        if not closes:
            return None, None
        basis = None
        if underlying == "NIFTY":
            spot = rest.ltp(["NSE:NIFTY 50"]).get("NSE:NIFTY 50", {}).get("last_price")
            if spot:
                basis = round((closes[-1][0] - spot) / spot * 10000, 1)
        oi_chg = None
        if len(closes) >= 2 and closes[-2][1]:
            oi_chg = round((closes[-1][1] - closes[-2][1]) / closes[-2][1] * 100, 1)
        return basis, oi_chg
    except Exception as exc:
        log.warning("futures snapshot failed (%s)", exc)
        return None, None


def get_constituent_bundle(period=None, use_cache: bool = True,
                           source: str = SOURCE_AUTO):
    """50-stock history bundle: Kite day candles first, Yahoo batch fallback."""
    from data.constituents import ConstituentBundle, fetch_constituent_history
    from data.kite.eod import constituent_frames
    from model.breadth.universe import get_universe

    period = period or "6mo"
    if not want_kite(source):
        return fetch_constituent_history(period=period, use_cache=use_cache)
    try:
        shorts = [c.short for c in get_universe()]
        frames = constituent_frames(shorts, days=_PERIOD_DAYS.get(period, 180))
        missing = [c.symbol for c in get_universe() if c.symbol not in frames]
        bundle = ConstituentBundle(frames=frames, missing=missing,
                                   period=period, interval="1d")
        if not bundle.sufficient:
            raise ValueError(
                f"thin kite coverage ({len(frames)} names)")
        log.info("constituent bundle via kite (%d names)", len(frames))
        return bundle
    except Exception as exc:
        log.warning("kite constituent bundle failed (%s); yahoo fallback", exc)
        return fetch_constituent_history(period=period, use_cache=use_cache)


def get_nifty_chain(expiry=None, use_cache: bool = True,
                    source: str = SOURCE_AUTO):
    """NIFTY chain: Kite-assembled primary, NSE scrape fallback."""
    from data import options as opts

    if not want_kite(source):
        return opts.fetch_chain(expiry=expiry, use_cache=use_cache)
    try:
        from data.kite.chain import KiteChainProvider
        from data.kite.rest import KiteRest
        from data.kite.store import InstrumentStore

        rest, store = KiteRest(), ensure_master()
        ltp = rest.ltp(["NSE:NIFTY 50"])
        spot = (ltp.get("NSE:NIFTY 50") or {}).get("last_price")
        if not spot:
            raise ValueError("no NIFTY spot LTP")
        chain, _ = KiteChainProvider(rest, store).chain_for("NIFTY", spot)
        if expiry and chain.expiries and chain.expiries[0] != expiry:
            raise ValueError(f"kite nearest {chain.expiries[0]} != {expiry}")
        log.info("nifty chain via kite (%s)", chain.expiries[0] if chain.expiries else "?")
        return chain
    except Exception as exc:
        log.warning("kite nifty chain failed (%s); NSE fallback", exc)
        return opts.fetch_chain(expiry=expiry, use_cache=use_cache)


def get_stock_chain(short: str, expiry=None, use_cache: bool = True,
                    source: str = SOURCE_AUTO):
    """Single-stock chain: Kite-assembled primary, NSE scrape fallback."""
    from data import options as opts

    if not want_kite(source):
        return opts.fetch_chain(symbol=short, expiry=expiry, use_cache=use_cache)
    try:
        from data.kite.chain import KiteChainProvider
        from data.kite.rest import KiteRest
        from data.kite.store import InstrumentStore

        rest, store = KiteRest(), ensure_master()
        ltp = rest.ltp([f"NSE:{short}"])
        spot = (ltp.get(f"NSE:{short}") or {}).get("last_price")
        if not spot:
            raise ValueError(f"no LTP for {short}")
        chain, _ = KiteChainProvider(rest, store).chain_for(short, spot)
        if expiry and chain.expiries and chain.expiries[0] != expiry:
            raise ValueError(f"kite nearest {chain.expiries[0]} != {expiry}")
        return chain
    except Exception as exc:
        log.warning("kite %s chain failed (%s); NSE fallback", short, exc)
        return opts.fetch_chain(symbol=short, expiry=expiry, use_cache=use_cache)


def get_stock_expiries(short: str, chain=None, source: str = SOURCE_AUTO):
    """Expiry list: chain > master > NSE contract-info."""
    if chain is not None:
        return list(chain.expiries)
    if want_kite(source):
        try:
            from data.kite import instruments as ki
            store = ensure_master()
            exp = ki.option_expiries(store, short)
            if exp:
                return exp
        except Exception as exc:
            log.warning("kite %s expiries failed (%s); NSE fallback", short, exc)
    from data import options as opts
    try:
        return list(opts.fetch_expiries(symbol=short))
    except Exception:
        return None


def get_india_vix(source: str = SOURCE_AUTO):
    """(level, day-change-%) or (None, None). Never raises."""
    global _vix_cache
    try:
        if time.time() - _vix_cache["at"] < VIX_CACHE_TTL:
            return _vix_cache["value"]
    except Exception:
        pass
    value = (None, None)
    try:
        use_kite = want_kite(source)
    except Exception:
        use_kite = False
    if use_kite:
        try:
            from data.kite.rest import KiteRest
            from data.kite.store import InstrumentStore

            rest, store = KiteRest(), ensure_master()
            row = store.find("NSE", "INDIA VIX")
            if row is None:
                raise ValueError("INDIA VIX missing from master")
            quote = rest.ltp(["NSE:INDIA VIX"]).get("NSE:INDIA VIX", {})
            level = quote.get("last_price")
            change = None
            if level:
                hist = rest.historical(row.instrument_token, "day",
                                       datetime.now() - timedelta(days=7),
                                       datetime.now())
                closes = [r["close"] for r in hist if r.get("close")]
                if len(closes) >= 2 and closes[-2]:
                    change = round((closes[-1] - closes[-2]) / closes[-2] * 100, 2)
                    level = round(float(closes[-1]), 2)
            value = (level, change)
        except Exception as exc:
            log.warning("kite india vix failed (%s)", exc)
            value = (None, None)
    _vix_cache = {"at": time.time(), "value": value}
    return value
