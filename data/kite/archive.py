"""Long-lived market archive for the order-block system (SQLite, journal.db).

Three tables `kite_candles_1m` cannot serve:

- `ob_series_1m` — NIFTY spot and front-future minute bars kept for years
  under stable series names ('NIFTY_SPOT', 'NIFTY_FUT1'). `kite_candles_1m`
  prunes at 90 days and is keyed by token, which the exchange reuses.
- `option_candles_1m` — option minute bars keyed by (exchange,
  tradingsymbol). Kite's historical API does not serve expired option
  contracts, so whatever is not written here before expiry is gone.
- `option_quotes` — top of book. Candles say what traded; quotes say what
  was executable, which is what a paper fill must be priced from.

Timestamps follow the repo: bars are IST wall-clock minute floors
("%Y-%m-%d %H:%M"); quote capture times are ISO seconds, IST naive.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import SETTINGS

SERIES_SPOT = "NIFTY_SPOT"
SERIES_FUT1 = "NIFTY_FUT1"
SERIES = (SERIES_SPOT, SERIES_FUT1)
SOURCES = ("kite_hist", "kite_ws")
QUOTE_REASONS = ("periodic", "signal", "fill", "exit", "open_snapshot")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ob_series_1m (
    series TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume INTEGER,
    oi INTEGER,
    contract TEXT DEFAULT '',
    source TEXT NOT NULL CHECK(source IN ('kite_hist', 'kite_ws')),
    UNIQUE (series, ts)
);
CREATE TABLE IF NOT EXISTS option_candles_1m (
    exchange TEXT NOT NULL,
    tradingsymbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume INTEGER NOT NULL DEFAULT 0,
    oi INTEGER,
    source TEXT NOT NULL CHECK(source IN ('kite_hist', 'kite_ws')),
    UNIQUE (exchange, tradingsymbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_oc1m_sym_ts ON option_candles_1m(tradingsymbol, ts);
CREATE TABLE IF NOT EXISTS option_quotes (
    exchange TEXT NOT NULL,
    tradingsymbol TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    exchange_ts TEXT,
    spot REAL,
    ltp REAL,
    bid REAL, bid_qty INTEGER,
    ask REAL, ask_qty INTEGER,
    depth_json TEXT DEFAULT '',
    volume INTEGER, oi INTEGER,
    iv REAL,
    reason TEXT NOT NULL DEFAULT 'periodic'
        CHECK(reason IN ('periodic', 'signal', 'fill', 'exit', 'open_snapshot')),
    UNIQUE (exchange, tradingsymbol, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_oq_sym_ts ON option_quotes(tradingsymbol, captured_at);
"""


@dataclass
class SeriesBar:
    series: str
    ts: str                      # "%Y-%m-%d %H:%M", IST, minute floor
    open: float
    high: float
    low: float
    close: float
    volume: int | None = None    # None for spot: the index has no volume
    oi: int | None = None
    contract: str = ""           # tradingsymbol behind FUT1 on this bar
    source: str = "kite_hist"


@dataclass
class OptionBar:
    exchange: str
    tradingsymbol: str
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    oi: int | None = None
    source: str = "kite_hist"


@dataclass
class OptionQuote:
    exchange: str
    tradingsymbol: str
    captured_at: str             # ISO seconds, IST naive
    exchange_ts: str | None = None
    spot: float | None = None
    ltp: float | None = None
    bid: float | None = None
    bid_qty: int | None = None
    ask: float | None = None
    ask_qty: int | None = None
    depth_json: str = ""
    volume: int | None = None
    oi: int | None = None
    iv: float | None = None      # percent, like chain legs
    reason: str = "periodic"

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None or self.bid <= 0 or self.ask <= 0:
            return None
        return round(self.ask - self.bid, 2)


class MarketArchive:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    # -- underlying series --------------------------------------------------

    def upsert_series(self, bars: list[SeriesBar]) -> int:
        n = 0
        for b in bars:
            if b.series not in SERIES:
                raise ValueError(f"unknown series {b.series!r}")
            if b.source not in SOURCES:
                raise ValueError(f"unknown source {b.source!r}")
            self.conn.execute(
                """INSERT INTO ob_series_1m
                   (series, ts, open, high, low, close, volume, oi, contract, source)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (series, ts) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume,
                     oi=excluded.oi, contract=excluded.contract,
                     source=excluded.source""",
                (b.series, b.ts, b.open, b.high, b.low, b.close, b.volume,
                 b.oi, b.contract, b.source))
            n += 1
        self.conn.commit()
        return n

    def read_series(self, series: str, frm: str, to: str) -> pd.DataFrame:
        """1m frame indexed by IST-naive DatetimeIndex (OHLCV + oi + contract)."""
        rows = self.conn.execute(
            "SELECT ts, open, high, low, close, volume, oi, contract "
            "FROM ob_series_1m WHERE series=? AND ts>=? AND ts<=? ORDER BY ts",
            (series, frm, to)).fetchall()
        cols = ["open", "high", "low", "close", "volume", "oi", "contract"]
        if not rows:
            return pd.DataFrame(columns=cols)
        frame = pd.DataFrame([dict(r) for r in rows])
        frame.index = pd.to_datetime(frame.pop("ts"))
        return frame[cols]

    def series_bounds(self, series: str) -> tuple[str | None, str | None]:
        row = self.conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM ob_series_1m WHERE series=?",
            (series,)).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def series_sessions(self, series: str) -> dict[str, int]:
        """Bars per session date — the coverage check (375 = full session)."""
        rows = self.conn.execute(
            "SELECT substr(ts, 1, 10) AS d, COUNT(*) FROM ob_series_1m "
            "WHERE series=? GROUP BY d ORDER BY d", (series,)).fetchall()
        return {r[0]: r[1] for r in rows}

    # -- options --------------------------------------------------------------

    def upsert_option_bars(self, bars: list[OptionBar]) -> int:
        n = 0
        for b in bars:
            if b.source not in SOURCES:
                raise ValueError(f"unknown source {b.source!r}")
            self.conn.execute(
                """INSERT INTO option_candles_1m
                   (exchange, tradingsymbol, ts, open, high, low, close,
                    volume, oi, source)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (exchange, tradingsymbol, ts) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume,
                     oi=excluded.oi, source=excluded.source""",
                (b.exchange, b.tradingsymbol, b.ts, b.open, b.high, b.low,
                 b.close, b.volume, b.oi, b.source))
            n += 1
        self.conn.commit()
        return n

    def read_option_bars(self, tradingsymbol: str, frm: str, to: str,
                         exchange: str = "NFO") -> pd.DataFrame:
        rows = self.conn.execute(
            "SELECT ts, open, high, low, close, volume, oi FROM option_candles_1m "
            "WHERE exchange=? AND tradingsymbol=? AND ts>=? AND ts<=? ORDER BY ts",
            (exchange, tradingsymbol, frm, to)).fetchall()
        cols = ["open", "high", "low", "close", "volume", "oi"]
        if not rows:
            return pd.DataFrame(columns=cols)
        frame = pd.DataFrame([dict(r) for r in rows])
        frame.index = pd.to_datetime(frame.pop("ts"))
        return frame[cols]

    def last_option_ts(self, tradingsymbol: str,
                       exchange: str = "NFO") -> str | None:
        row = self.conn.execute(
            "SELECT MAX(ts) FROM option_candles_1m WHERE exchange=? "
            "AND tradingsymbol=?", (exchange, tradingsymbol)).fetchone()
        return row[0] if row and row[0] else None

    def add_quotes(self, quotes: list[OptionQuote]) -> int:
        """Insert quote snapshots; a repeat (symbol, captured_at) is ignored."""
        n = 0
        for q in quotes:
            if q.reason not in QUOTE_REASONS:
                raise ValueError(f"unknown quote reason {q.reason!r}")
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO option_quotes
                   (exchange, tradingsymbol, captured_at, exchange_ts, spot,
                    ltp, bid, bid_qty, ask, ask_qty, depth_json, volume, oi,
                    iv, reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (q.exchange, q.tradingsymbol, q.captured_at, q.exchange_ts,
                 q.spot, q.ltp, q.bid, q.bid_qty, q.ask, q.ask_qty,
                 q.depth_json, q.volume, q.oi, q.iv, q.reason))
            n += cur.rowcount
        self.conn.commit()
        return n

    def quotes(self, tradingsymbol: str | None = None, frm: str | None = None,
               to: str | None = None, limit: int = 100_000) -> list[OptionQuote]:
        sql = "SELECT * FROM option_quotes WHERE 1=1"
        params: list = []
        if tradingsymbol:
            sql += " AND tradingsymbol=?"
            params.append(tradingsymbol)
        if frm:
            sql += " AND captured_at>=?"
            params.append(frm)
        if to:
            sql += " AND captured_at<=?"
            params.append(to)
        sql += " ORDER BY captured_at LIMIT ?"
        params.append(limit)
        return [OptionQuote(**{k: r[k] for k in r.keys()})
                for r in self.conn.execute(sql, params)]

    def coverage(self) -> dict:
        """Row counts and date span per table, for the audit report."""
        out = {}
        for series in SERIES:
            lo, hi = self.series_bounds(series)
            n = self.conn.execute(
                "SELECT COUNT(*) FROM ob_series_1m WHERE series=?",
                (series,)).fetchone()[0]
            out[series] = {"bars": n, "from": lo, "to": hi}
        row = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT tradingsymbol), MIN(ts), MAX(ts) "
            "FROM option_candles_1m").fetchone()
        out["option_candles_1m"] = {"bars": row[0], "contracts": row[1],
                                    "from": row[2], "to": row[3]}
        row = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT tradingsymbol), MIN(captured_at), "
            "MAX(captured_at) FROM option_quotes").fetchone()
        out["option_quotes"] = {"rows": row[0], "contracts": row[1],
                                "from": row[2], "to": row[3]}
        return out


_shared: MarketArchive | None = None


def shared_market_archive() -> MarketArchive:
    global _shared
    if _shared is None:
        _shared = MarketArchive()
    return _shared
