"""Kite REST access: rate-limited client + normalized loaders.

Buckets (per official limits): quote 1/s, historical 3/s, everything else
10/s. Loaders return the repo's own dataclasses (`Candle`, frames) so
strategy code never sees Kite payload shapes.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import pandas as pd

from data import nifty as nifty_data
from data.kite.auth import KiteAuthError, credentials, load_session, session_valid

log = logging.getLogger(__name__)

# Minimum seconds between calls, per endpoint bucket.
LIMITS = {"quote": 1.0, "historical": 1.0 / 3.0, "other": 0.1}


class RateLimiter:
    """Thread-safe minimum-interval gate with sleep (not drop)."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_ok = 0.0

    def acquire(self) -> float:
        """Block until a call is allowed. Returns seconds waited."""
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_ok - now)
            if wait:
                time.sleep(wait)
                now = time.monotonic()
            self._next_ok = now + self.min_interval
            return round(wait, 3)


_limiters = {name: RateLimiter(iv) for name, iv in LIMITS.items()}


def kite_client():
    """Authenticated KiteConnect from the stored session.

    Raises KiteAuthError pointing at `kite-login` when missing/expired.
    """
    creds = credentials()
    session = load_session()
    if not session_valid(session):
        raise KiteAuthError(
            "no valid kite session — run `model_cli.py kite-login` "
            "(sessions expire 6 AM IST daily)")
    try:
        from kiteconnect import KiteConnect
    except ImportError as exc:
        raise KiteAuthError(
            "kiteconnect is not installed (pip install kiteconnect)") from exc
    client = KiteConnect(api_key=creds.api_key)
    client.set_access_token(session["access_token"])
    return client


class KiteRest:
    """Thin rate-limited wrapper; payloads normalized by loaders below."""

    def __init__(self, client=None) -> None:
        self.client = client or kite_client()

    def instruments(self, exchange: str | None = None):
        _limiters["other"].acquire()
        return self.client.instruments(exchange)

    def quote(self, keys: list[str]) -> dict:
        """Full quotes for up to 500 `EXCHANGE:SYMBOL` keys."""
        if len(keys) > 500:
            raise ValueError(f"quote supports ≤500 keys, got {len(keys)}")
        _limiters["quote"].acquire()
        return self.client.quote(keys) or {}

    def ohlc(self, keys: list[str]) -> dict:
        if len(keys) > 1000:
            raise ValueError(f"ohlc supports ≤1000 keys, got {len(keys)}")
        _limiters["quote"].acquire()
        return self.client.ohlc(keys) or {}

    def ltp(self, keys: list[str]) -> dict:
        if len(keys) > 1000:
            raise ValueError(f"ltp supports ≤1000 keys, got {len(keys)}")
        _limiters["quote"].acquire()
        return self.client.ltp(keys) or {}

    def historical(self, token: int, interval: str, frm: datetime,
                   to: datetime, oi: bool = False,
                   continuous: bool = False) -> list[dict]:
        _limiters["historical"].acquire()
        return self.client.historical_data(
            token, frm, to, interval, continuous=continuous, oi=oi) or []


def history_to_candles(records: list[dict]) -> list[nifty_data.Candle]:
    """Kite historical records -> repo Candle list (tz-naive, oldest first)."""
    out: list[nifty_data.Candle] = []
    for r in records:
        try:
            ts = r["date"]
            ts = ts.tz_localize(None) if isinstance(ts, pd.Timestamp) \
                else pd.Timestamp(ts).tz_localize(None)
            out.append(nifty_data.Candle(
                timestamp=ts.to_pydatetime(),
                open=float(r["open"]), high=float(r["high"]),
                low=float(r["low"]), close=float(r["close"]),
                volume=int(r.get("volume") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda c: c.timestamp)


def history_to_frame(records: list[dict]) -> pd.DataFrame:
    """Kite historical records -> tidy OHLCV frame (constituent-compatible)."""
    candles = history_to_candles(records)
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        [{"open": c.open, "high": c.high, "low": c.low, "close": c.close,
          "volume": c.volume} for c in candles],
        index=pd.DatetimeIndex([c.timestamp for c in candles]))
    frame["volume"] = frame["volume"].astype("int64")
    return frame[["open", "high", "low", "close", "volume"]]


def fetch_history_token(token: int, interval: str, frm: datetime,
                        to: datetime, rest: KiteRest | None = None,
                        oi: bool = False) -> list[nifty_data.Candle]:
    """One token's history as repo Candles (single rate-limited call)."""
    rest = rest or KiteRest()
    return history_to_candles(rest.historical(token, interval, frm, to, oi=oi))
