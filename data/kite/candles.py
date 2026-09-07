"""Kite candle persistence (SQLite, settled candles only).

Only finished minute candles are stored — the in-flight minute lives in
the aggregator. 5m/15m are derived by resampling 1m on read (no duplicate
storage). Daily candles are stored explicitly (EOD flush) so history and
live share one read path. Provisional data never touches these tables.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kite_candles_1m (
    token INTEGER NOT NULL,
    ts TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL DEFAULT 0,
    oi INTEGER,
    n_ticks INTEGER NOT NULL DEFAULT 0,
    UNIQUE (token, ts)
);
CREATE INDEX IF NOT EXISTS idx_kc1m_token_ts ON kite_candles_1m(token, ts);
CREATE TABLE IF NOT EXISTS kite_candles_1d (
    token INTEGER NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL DEFAULT 0,
    oi INTEGER,
    UNIQUE (token, date)
);
CREATE INDEX IF NOT EXISTS idx_kc1d_token_date ON kite_candles_1d(token, date);
"""

RETENTION_1M_DAYS = 90


@dataclass
class MinuteCandle:
    token: int
    ts: str                  # "%Y-%m-%d %H:%M", IST wall, minute floor
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    oi: int | None = None
    n_ticks: int = 0


@dataclass
class DayCandle:
    token: int
    date: str                # "%Y-%m-%d"
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    oi: int | None = None


class CandleStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def upsert_1m(self, candles: list[MinuteCandle]) -> int:
        n = 0
        for c in candles:
            self.conn.execute(
                """INSERT INTO kite_candles_1m
                   (token, ts, open, high, low, close, volume, oi, n_ticks)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (token, ts) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume,
                     oi=excluded.oi, n_ticks=excluded.n_ticks""",
                (c.token, c.ts, c.open, c.high, c.low, c.close,
                 c.volume, c.oi, c.n_ticks))
            n += 1
        self.conn.commit()
        return n

    def upsert_1d(self, candles: list[DayCandle]) -> int:
        n = 0
        for c in candles:
            self.conn.execute(
                """INSERT INTO kite_candles_1d
                   (token, date, open, high, low, close, volume, oi)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT (token, date) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume, oi=excluded.oi""",
                (c.token, c.date, c.open, c.high, c.low, c.close,
                 c.volume, c.oi))
            n += 1
        self.conn.commit()
        return n

    def read_1m(self, token: int, frm: str, to: str) -> pd.DataFrame:
        """1m frame indexed by IST-naive DatetimeIndex (OHLCV + oi)."""
        rows = self.conn.execute(
            "SELECT ts, open, high, low, close, volume, oi FROM kite_candles_1m "
            "WHERE token=? AND ts>=? AND ts<=? ORDER BY ts",
            (token, frm, to)).fetchall()
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oi"])
        frame = pd.DataFrame([dict(r) for r in rows])
        frame.index = pd.to_datetime(frame.pop("ts"))
        return frame[["open", "high", "low", "close", "volume", "oi"]]

    def read_5m(self, token: int, frm: str, to: str) -> pd.DataFrame:
        """5m resampled from stored 1m (same OHLCV semantics)."""
        m1 = self.read_1m(token, frm, to)
        if m1.empty:
            return m1.drop(columns=["oi"], errors="ignore")
        agg: dict = {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}
        out = m1.drop(columns=["oi"]).resample("5min", origin="start_day").agg(agg)
        return out.dropna(subset=["close"])

    def read_1d(self, token: int, frm: str, to: str) -> pd.DataFrame:
        rows = self.conn.execute(
            "SELECT date, open, high, low, close, volume, oi FROM kite_candles_1d "
            "WHERE token=? AND date>=? AND date<=? ORDER BY date",
            (token, frm, to)).fetchall()
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oi"])
        frame = pd.DataFrame([dict(r) for r in rows])
        frame.index = pd.to_datetime(frame.pop("date"))
        return frame[["open", "high", "low", "close", "volume", "oi"]]

    def last_1m_ts(self, token: int) -> str | None:
        row = self.conn.execute(
            "SELECT MAX(ts) FROM kite_candles_1m WHERE token=?", (token,)).fetchone()
        return row[0] if row and row[0] else None

    def prune_1m(self, older_than_days: int = RETENTION_1M_DAYS,
                 now: datetime | None = None) -> int:
        cutoff = (now or datetime.now()).date() - timedelta(days=older_than_days)
        cur = self.conn.execute(
            "DELETE FROM kite_candles_1m WHERE date(ts) < ?", (cutoff.isoformat(),))
        self.conn.commit()
        return cur.rowcount
