"""Kite persistence layer (SQLite, shared journal.db).

Phase 1: instrument master only. Candle + chain-snapshot tables arrive
with the streaming phases. Expired contracts are KEPT (exchange reuses
tokens after expiry — the docs warn against token-only keys).
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
"""


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
        self.conn.commit()
        total = self.conn.execute(
            "SELECT COUNT(*) FROM kite_instruments WHERE as_of=?",
            (rows[0].as_of if rows else "",)).fetchone()[0]
        return {"seen": seen, "as_of_total": total}

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
        row = self.conn.execute(
            "SELECT * FROM kite_instruments WHERE exchange=? AND tradingsymbol=?",
            (exchange, tradingsymbol)).fetchone()
        return self._row(row) if row else None

    def scan(self, exchange: str | None = None,
             instrument_type: str | tuple[str, ...] | None = None,
             prefix: str | None = None) -> list[InstrumentRow]:
        """Filtered scan for resolvers (NFO universes are small)."""
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
