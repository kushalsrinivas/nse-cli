"""Kite persistence layer (SQLite, shared journal.db).

Phase 1: instrument master only. Candle + chain-snapshot tables arrive
with the streaming phases. Expired contracts are KEPT (exchange reuses
tokens after expiry — the docs warn against token-only keys).

`kite_instruments` holds one row per (exchange, tradingsymbol) and is
overwritten on every refresh, so it only knows a contract's attributes as
of the latest master. `kite_instrument_history` is the versioned view: a
row is appended only when a tracked contract first appears or one of its
attributes (token, lot, tick, expiry, strike) changes. A backtest on date
D resolves lot size through `as_of_on(..., D)`, never through today's row.
History is limited to `HISTORY_UNDERLYINGS` — the full NFO master is
~90k rows and nothing downstream needs stock-option versions.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kite_instruments (
    instrument_token INTEGER NOT NULL,
    exchange TEXT NOT NULL,
    tradingsymbol TEXT NOT NULL,
    name TEXT DEFAULT '',
    expiry TEXT DEFAULT '',
    strike REAL,
    tick_size REAL,
    lot_size INTEGER,
    instrument_type TEXT DEFAULT '',
    segment TEXT DEFAULT '',
    as_of TEXT NOT NULL,
    UNIQUE (exchange, tradingsymbol)
);
CREATE INDEX IF NOT EXISTS idx_kite_inst_token ON kite_instruments(instrument_token);
CREATE INDEX IF NOT EXISTS idx_kite_inst_lookup
    ON kite_instruments(exchange, instrument_type, expiry);
CREATE TABLE IF NOT EXISTS kite_instrument_history (
    exchange TEXT NOT NULL,
    tradingsymbol TEXT NOT NULL,
    as_of TEXT NOT NULL,
    instrument_token INTEGER NOT NULL,
    name TEXT DEFAULT '',
    instrument_type TEXT NOT NULL,
    expiry TEXT DEFAULT '',
    strike REAL,
    lot_size INTEGER,
    tick_size REAL,
    UNIQUE (exchange, tradingsymbol, as_of)
);
CREATE INDEX IF NOT EXISTS idx_kih_lookup
    ON kite_instrument_history(exchange, instrument_type, expiry, as_of);
CREATE INDEX IF NOT EXISTS idx_kih_symbol
    ON kite_instrument_history(exchange, tradingsymbol, as_of);
"""

#: Underlyings whose master rows are versioned in kite_instrument_history.
HISTORY_UNDERLYINGS = ("NIFTY",)
#: NSE index rows versioned alongside (the spot token can change too).
HISTORY_INDEX_SYMBOLS = ("NIFTY 50", "INDIA VIX")

_VERSIONED = ("instrument_token", "instrument_type", "expiry", "strike",
              "lot_size", "tick_size")


def tracks_history(r: InstrumentRow) -> bool:
    """True for rows kite_instrument_history should version."""
    if r.exchange == "NSE":
        return r.tradingsymbol in HISTORY_INDEX_SYMBOLS
    if r.exchange != "NFO":
        return False
    for u in HISTORY_UNDERLYINGS:
        rest = r.tradingsymbol[len(u):len(u) + 1]
        if r.tradingsymbol.startswith(u) and rest.isdigit():
            return True
    return False


@dataclass
class InstrumentRow:
    instrument_token: int
    exchange: str
    tradingsymbol: str
    name: str = ""
    expiry: str = ""
    strike: float | None = None
    tick_size: float | None = None
    lot_size: int | None = None
    instrument_type: str = ""
    segment: str = ""
    as_of: str = ""


def _norm_expiry(value) -> str:
    if value in (None, "", "null"):
        return ""
    try:
        return value.strftime("%Y-%m-%d")
    except AttributeError:
        return str(value)[:10]


def normalize_dump_row(raw: dict, as_of: str) -> InstrumentRow:
    """kiteconnect instruments() dict -> InstrumentRow (never raises)."""
    def num(key, cast):
        try:
            v = raw.get(key)
            return cast(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    return InstrumentRow(
        instrument_token=int(raw.get("instrument_token") or 0),
        exchange=str(raw.get("exchange") or ""),
        tradingsymbol=str(raw.get("tradingsymbol") or ""),
        name=str(raw.get("name") or ""),
        expiry=_norm_expiry(raw.get("expiry")),
        strike=num("strike", float),
        tick_size=num("tick_size", float),
        lot_size=num("lot_size", int),
        instrument_type=str(raw.get("instrument_type") or ""),
        segment=str(raw.get("segment") or ""),
        as_of=as_of,
    )


class InstrumentStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def upsert(self, rows: list[InstrumentRow]) -> dict[str, int]:
        """Insert or replace by (exchange, tradingsymbol). Returns counts."""
        seen = 0
        for r in rows:
            if not r.tradingsymbol or not r.exchange or not r.instrument_token:
                continue
            seen += 1
            self.conn.execute(
                """INSERT INTO kite_instruments
                   (instrument_token, exchange, tradingsymbol, name, expiry,
                    strike, tick_size, lot_size, instrument_type, segment, as_of)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (exchange, tradingsymbol) DO UPDATE SET
                     instrument_token=excluded.instrument_token,
                     name=excluded.name, expiry=excluded.expiry,
                     strike=excluded.strike, tick_size=excluded.tick_size,
                     lot_size=excluded.lot_size,
                     instrument_type=excluded.instrument_type,
                     segment=excluded.segment, as_of=excluded.as_of""",
                (r.instrument_token, r.exchange, r.tradingsymbol, r.name,
                 r.expiry, r.strike, r.tick_size, r.lot_size,
                 r.instrument_type, r.segment, r.as_of))
        versioned = self._record_history(rows)
        self.conn.commit()
        total = self.conn.execute(
            "SELECT COUNT(*) FROM kite_instruments WHERE as_of=?",
            (rows[0].as_of if rows else "",)).fetchone()[0]
        return {"seen": seen, "as_of_total": total, "versioned": versioned}

    def _record_history(self, rows: list[InstrumentRow]) -> int:
        """Append a history version for each tracked row that changed."""
        tracked = [r for r in rows if r.tradingsymbol and r.instrument_token
                   and tracks_history(r)]
        if not tracked:
            return 0
        latest: dict[tuple[str, str], tuple] = {}
        for h in self.conn.execute(
                "SELECT h.* FROM kite_instrument_history h JOIN ("
                " SELECT exchange, tradingsymbol, MAX(as_of) AS m"
                " FROM kite_instrument_history GROUP BY exchange, tradingsymbol"
                ") last ON h.exchange=last.exchange"
                " AND h.tradingsymbol=last.tradingsymbol AND h.as_of=last.m"):
            latest[(h["exchange"], h["tradingsymbol"])] = (
                h["as_of"], tuple(h[c] for c in _VERSIONED))
        n = 0
        for r in tracked:
            attrs = tuple(getattr(r, c) for c in _VERSIONED)
            prev = latest.get((r.exchange, r.tradingsymbol))
            if prev is not None and (prev[1] == attrs or prev[0] > r.as_of):
                continue
            self.conn.execute(
                """INSERT INTO kite_instrument_history
                   (exchange, tradingsymbol, as_of, instrument_token, name,
                    instrument_type, expiry, strike, lot_size, tick_size)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (exchange, tradingsymbol, as_of) DO UPDATE SET
                     instrument_token=excluded.instrument_token,
                     name=excluded.name,
                     instrument_type=excluded.instrument_type,
                     expiry=excluded.expiry, strike=excluded.strike,
                     lot_size=excluded.lot_size, tick_size=excluded.tick_size""",
                (r.exchange, r.tradingsymbol, r.as_of, r.instrument_token,
                 r.name, r.instrument_type, r.expiry, r.strike, r.lot_size,
                 r.tick_size))
            n += 1
        return n

    def as_of_on(self, exchange: str, tradingsymbol: str,
                 date: str) -> InstrumentRow | None:
        """The contract's attributes as known on `date` (YYYY-MM-DD).

        Latest history version with as_of <= date. None when the contract
        was not yet tracked by then — callers must treat that as missing
        data, not fall back to today's master.
        """
        row = self.conn.execute(
            "SELECT * FROM kite_instrument_history WHERE exchange=? "
            "AND tradingsymbol=? AND as_of<=? ORDER BY as_of DESC LIMIT 1",
            (exchange, tradingsymbol, date)).fetchone()
        if row is None:
            return None
        return InstrumentRow(
            instrument_token=row["instrument_token"], exchange=row["exchange"],
            tradingsymbol=row["tradingsymbol"], name=row["name"],
            expiry=row["expiry"], strike=row["strike"],
            tick_size=row["tick_size"], lot_size=row["lot_size"],
            instrument_type=row["instrument_type"], segment="",
            as_of=row["as_of"])

    def lot_size_on(self, exchange: str, tradingsymbol: str,
                    date: str) -> int | None:
        row = self.as_of_on(exchange, tradingsymbol, date)
        return row.lot_size if row else None

    def history_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM kite_instrument_history").fetchone()[0]

    def as_of_dates(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT as_of FROM kite_instruments ORDER BY as_of DESC").fetchall()
        return [r[0] for r in rows if r[0]]

    def count_by_segment(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT segment, COUNT(*) FROM kite_instruments GROUP BY segment").fetchall()
        return {r[0] or "?": r[1] for r in rows}

    def by_token(self, token: int) -> InstrumentRow | None:
        row = self.conn.execute(
            "SELECT * FROM kite_instruments WHERE instrument_token=? "
            "ORDER BY as_of DESC LIMIT 1", (token,)).fetchone()
        return self._row(row) if row else None

    def find(self, exchange: str, tradingsymbol: str) -> InstrumentRow | None:
        # Newest snapshot wins: the same tradingsymbol appears once per
        # refresh, and tokens are reused after expiry.
        row = self.conn.execute(
            "SELECT * FROM kite_instruments WHERE exchange=? AND tradingsymbol=? "
            "ORDER BY as_of DESC LIMIT 1",
            (exchange, tradingsymbol)).fetchone()
        return self._row(row) if row else None

    def scan(self, exchange: str | None = None,
             instrument_type: str | tuple[str, ...] | None = None,
             prefix: str | None = None,
             latest_only: bool = True) -> list[InstrumentRow]:
        """Filtered scan for resolvers (NFO universes are small).

        Every refresh appends a new `as_of` snapshot without deleting the
        previous ones, so an unscoped scan returned contracts from every
        master ever fetched — long-settled expiries resurfaced as if they
        were live. `latest_only` pins the scan to the newest snapshot for
        the exchange being scanned (NSE and NFO refresh independently, so
        the bound must be per-exchange, not global).
        """
        sql = "SELECT * FROM kite_instruments WHERE 1=1"
        params: list = []
        if exchange:
            sql += " AND exchange=?"
            params.append(exchange)
        if instrument_type:
            if isinstance(instrument_type, str):
                instrument_type = (instrument_type,)
            sql += f" AND instrument_type IN ({','.join('?' * len(instrument_type))})"
            params.extend(instrument_type)
        if latest_only:
            if exchange:
                sql += (" AND as_of = (SELECT MAX(as_of) FROM kite_instruments "
                        "WHERE exchange=?)")
                params.append(exchange)
            else:
                sql += " AND as_of = (SELECT MAX(as_of) FROM kite_instruments)"
        rows = [self._row(r) for r in self.conn.execute(sql, params)]
        if prefix:
            rows = [r for r in rows if r.tradingsymbol.startswith(prefix)]
        return rows

    @staticmethod
    def _row(row: sqlite3.Row) -> InstrumentRow:
        return InstrumentRow(**{k: row[k] for k in (
            "instrument_token", "exchange", "tradingsymbol", "name", "expiry",
            "strike", "tick_size", "lot_size", "instrument_type", "segment",
            "as_of")})
