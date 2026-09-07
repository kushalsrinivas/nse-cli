"""SQLite journal for per-stock overnight runs (separate table).

Same contract as the NIFTY overnight journal, plus symbol / lot_size /
lots. Fixed-lots mode: GO rows carry the requested lots, NO-GO rows carry
0 (hypothetical P&L still settles per-lot for measurement).
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stock_overnight_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    stock_close REAL,
    market_regime TEXT DEFAULT '',
    direction TEXT NOT NULL,
    decision TEXT NOT NULL,
    confidence_score REAL,
    option_type TEXT DEFAULT '',
    option_strike REAL,
    contract_name TEXT DEFAULT '',
    expiry TEXT DEFAULT '',
    entry_price REAL,
    expected_exit REAL,
    actual_exit_price REAL,
    actual_pnl REAL,
    actual_pnl_pct REAL,
    hypothetical_exit_price REAL,
    hypothetical_pnl REAL,
    hypothetical_pnl_pct REAL,
    outcome TEXT DEFAULT 'PENDING',
    is_actual_trade INTEGER DEFAULT 0,
    lot_size INTEGER DEFAULT 0,
    lots INTEGER DEFAULT 0,
    matched_bucket TEXT DEFAULT '',
    cohort_n INTEGER,
    expected_value_lot REAL,
    expected_value_pct REAL,
    p_direction REAL,
    p_profitable REAL,
    p10_loss_lot REAL,
    signal_scores TEXT DEFAULT '{}',
    decision_rationale TEXT DEFAULT '',
    blocked_reasons TEXT DEFAULT '',
    engine_version TEXT DEFAULT 'v1.0-stock',
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_soj_symbol ON stock_overnight_journal(symbol);
CREATE INDEX IF NOT EXISTS idx_soj_decision ON stock_overnight_journal(decision);
CREATE INDEX IF NOT EXISTS idx_soj_date ON stock_overnight_journal(trade_date);
"""


@dataclass
class StockOvernightRunRecord:
    id: int | None
    run_id: str
    timestamp: str
    trade_date: str
    symbol: str
    stock_close: float | None = None
    nifty_close: float | None = None
    market_regime: str = ""
    direction: str = "neutral"
    decision: str = "NO-GO"
    confidence_score: float | None = None
    option_type: str = ""
    option_strike: float | None = None
    contract_name: str = ""
    expiry: str = ""
    entry_price: float | None = None
    expected_exit: float | None = None
    actual_exit_price: float | None = None
    actual_pnl: float | None = None
    actual_pnl_pct: float | None = None
    hypothetical_exit_price: float | None = None
    hypothetical_pnl: float | None = None
    hypothetical_pnl_pct: float | None = None
    outcome: str = "PENDING"
    is_actual_trade: int = 0
    lot_size: int = 0
    lots: int = 0
    matched_bucket: str = ""
    cohort_n: int | None = None
    expected_value_lot: float | None = None
    expected_value_pct: float | None = None
    p_direction: float | None = None
    p_profitable: float | None = None
    p10_loss_lot: float | None = None
    signal_scores: str = "{}"
    decision_rationale: str = ""
    blocked_reasons: str = ""
    engine_version: str = "v1.0-stock"
    notes: str = ""
    created_at: str = ""

    @property
    def units(self) -> int:
        return max(int(self.lots or 0), 0) * max(int(self.lot_size or 0), 0)

    @property
    def pnl_display(self) -> str:
        pnl = self.actual_pnl if self.is_actual_trade else self.hypothetical_pnl
        if pnl is None:
            return "—"
        return f"₹{pnl:+,.0f}{'' if self.is_actual_trade else '*'}"


def _new_run_id() -> str:
    now = datetime.now()
    return f"ST-{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"


class StockOvernightJournal:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def _to_record(self, row: sqlite3.Row) -> StockOvernightRunRecord:
        data = {f.name: row[f.name] for f in fields(StockOvernightRunRecord)
                if f.name in row.keys() and f.name != "nifty_close"}
        return StockOvernightRunRecord(**data)

    def add(self, rec: StockOvernightRunRecord) -> StockOvernightRunRecord:
        import dataclasses
        if not rec.run_id:
            rec = dataclasses.replace(rec, run_id=_new_run_id())
        data = asdict(rec)
        data.pop("id")
        data.pop("nifty_close", None)
        cols = ", ".join(data)
        ph = ", ".join("?" * len(data))
        cur = self.conn.execute(
            f"INSERT INTO stock_overnight_journal ({cols}) VALUES ({ph})",
            list(data.values()))
        self.conn.commit()
        return dataclasses.replace(rec, id=cur.lastrowid)

    def get(self, rec_id: int) -> StockOvernightRunRecord | None:
        row = self.conn.execute(
            "SELECT * FROM stock_overnight_journal WHERE id=?", (rec_id,)).fetchone()
        return self._to_record(row) if row else None

    def list(self, symbol: str = "all", decision: str = "all",
             trade_type: str = "all", limit: int = 200) -> list[StockOvernightRunRecord]:
        sql = "SELECT * FROM stock_overnight_journal WHERE 1=1"
        params: list[Any] = []
        if symbol != "all":
            sql += " AND symbol=?"
            params.append(symbol.upper())
        if decision in ("GO", "NO-GO"):
            sql += " AND decision=?"
            params.append(decision)
        if trade_type == "actual":
            sql += " AND is_actual_trade=1"
        elif trade_type == "hypothetical":
            sql += " AND is_actual_trade=0"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [self._to_record(r) for r in self.conn.execute(sql, params)]

    def symbols_for_date(self, trade_date: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM stock_overnight_journal WHERE trade_date=?",
            (trade_date,)).fetchall()
        return {r[0] for r in rows}

    def settle(self, rec_id: int, exit_price: float,
               notes: str | None = None) -> StockOvernightRunRecord | None:
        rec = self.get(rec_id)
        if rec is None or rec.entry_price is None:
            return None
        # GO rows settle at their fixed lots; NO-GO rows (lots=0) settle
        # per-lot so avoided/missed moves stay measurable.
        units = rec.units or rec.lot_size or 0
        if units <= 0:
            return None
        pnl = round((exit_price - rec.entry_price) * units, 2)
        pct = round((exit_price - rec.entry_price) / rec.entry_price * 100, 2) \
            if rec.entry_price else 0.0
        if rec.is_actual_trade:
            self.conn.execute(
                "UPDATE stock_overnight_journal SET actual_exit_price=?, actual_pnl=?, "
                "actual_pnl_pct=?, outcome=?, notes=COALESCE(?, notes) WHERE id=?",
                (exit_price, pnl, pct, _outcome(pnl), notes, rec_id))
        else:
            self.conn.execute(
                "UPDATE stock_overnight_journal SET hypothetical_exit_price=?, hypothetical_pnl=?, "
                "hypothetical_pnl_pct=?, outcome=?, notes=COALESCE(?, notes) WHERE id=?",
                (exit_price, pnl, pct, _outcome(pnl), notes, rec_id))
        self.conn.commit()
        return self.get(rec_id)


def _outcome(pnl: float) -> str:
    if pnl > 100:
        return "WIN"
    if pnl < -100:
        return "LOSS"
    return "BREAKEVEN"


_shared: StockOvernightJournal | None = None


def shared_stock_overnight_journal() -> StockOvernightJournal:
    global _shared
    if _shared is None:
        _shared = StockOvernightJournal()
    return _shared
