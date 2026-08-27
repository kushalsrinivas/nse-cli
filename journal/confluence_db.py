"""SQLite journal for intraday confluence setup runs (A/B/C)."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS confluence_trade_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    setup_id TEXT NOT NULL CHECK(setup_id IN ('A', 'B', 'C')),
    timestamp TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    nifty_spot REAL NOT NULL,
    direction TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('GO', 'NO-GO', 'WATCH', 'ERROR')),
    confidence_score REAL NOT NULL,
    option_type TEXT DEFAULT '',
    option_strike REAL,
    contract_name TEXT DEFAULT '',
    expiry TEXT DEFAULT '',
    entry_price REAL,
    delta REAL,
    actual_exit_price REAL,
    actual_pnl REAL,
    actual_pnl_pct REAL,
    hypothetical_exit_price REAL,
    hypothetical_pnl REAL,
    hypothetical_pnl_pct REAL,
    outcome TEXT NOT NULL DEFAULT 'PENDING' CHECK(outcome IN ('WIN', 'LOSS', 'BREAKEVEN', 'NOT_TRADED', 'PENDING')),
    is_actual_trade INTEGER NOT NULL DEFAULT 0,
    conditions_json TEXT DEFAULT '[]',
    blocked_reasons TEXT DEFAULT '',
    decision_rationale TEXT DEFAULT '',
    vix_val REAL,
    engine_version TEXT NOT NULL DEFAULT 'v1.0-confluence',
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cj_setup ON confluence_trade_journal(setup_id);
CREATE INDEX IF NOT EXISTS idx_cj_decision ON confluence_trade_journal(decision);
CREATE INDEX IF NOT EXISTS idx_cj_trade_date ON confluence_trade_journal(trade_date);
CREATE INDEX IF NOT EXISTS idx_cj_run_id ON confluence_trade_journal(run_id);
"""


@dataclass
class ConfluenceRunRecord:
    id: int | None
    run_id: str
    setup_id: str
    timestamp: str
    trade_date: str
    nifty_spot: float
    direction: str
    decision: str
    confidence_score: float
    option_type: str = ""
    option_strike: float | None = None
    contract_name: str = ""
    expiry: str = ""
    entry_price: float | None = None
    delta: float | None = None
    actual_exit_price: float | None = None
    actual_pnl: float | None = None
    actual_pnl_pct: float | None = None
    hypothetical_exit_price: float | None = None
    hypothetical_pnl: float | None = None
    hypothetical_pnl_pct: float | None = None
    outcome: str = "PENDING"
    is_actual_trade: int = 0
    conditions_json: str = "[]"
    blocked_reasons: str = ""
    decision_rationale: str = ""
    vix_val: float | None = None
    engine_version: str = "v1.0-confluence"
    notes: str = ""
    created_at: str = ""

    @property
    def effective_pnl(self) -> float | None:
        if self.is_actual_trade and self.actual_pnl is not None:
            return self.actual_pnl
        return self.hypothetical_pnl

    @property
    def effective_exit(self) -> float | None:
        if self.is_actual_trade and self.actual_exit_price is not None:
            return self.actual_exit_price
        return self.hypothetical_exit_price

    @property
    def pnl_display(self) -> str:
        pnl = self.effective_pnl
        if pnl is None:
            return "—"
        star = "" if self.is_actual_trade else "*"
        return f"₹{pnl:+,.0f}{star}"


class ConfluenceJournal:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def _to_record(self, row: sqlite3.Row) -> ConfluenceRunRecord:
        data = {f.name: row[f.name] for f in fields(ConfluenceRunRecord) if f.name in row.keys()}
        return ConfluenceRunRecord(**data)

    @staticmethod
    def _cols() -> list[str]:
        return [f.name for f in fields(ConfluenceRunRecord) if f.name != "id"]

    def add(self, rec: ConfluenceRunRecord) -> ConfluenceRunRecord:
        if not rec.run_id:
            now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            rec.run_id = f"CF-{now_str}-{uuid.uuid4().hex[:4].upper()}"
        if not rec.created_at:
            rec.created_at = datetime.now().isoformat(timespec="seconds")
        if not rec.trade_date:
            rec.trade_date = rec.timestamp[:10] if rec.timestamp else datetime.now().strftime("%Y-%m-%d")

        cols = self._cols()
        values = [getattr(rec, c) for c in cols]
        placeholders = ", ".join("?" * len(cols))
        cur = self.conn.execute(
            f"INSERT INTO confluence_trade_journal ({', '.join(cols)}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        return replace(rec, id=cur.lastrowid)

    def get(self, record_id: int) -> ConfluenceRunRecord | None:
        row = self.conn.execute(
            "SELECT * FROM confluence_trade_journal WHERE id=?", (record_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def update(self, rec: ConfluenceRunRecord) -> None:
        cols = self._cols()
        assignments = ", ".join(f"{c}=?" for c in cols)
        values = [getattr(rec, c) for c in cols] + [rec.id]
        self.conn.execute(
            f"UPDATE confluence_trade_journal SET {assignments} WHERE id=?", values
        )
        self.conn.commit()

    def list(
        self,
        decision: str = "all",
        direction: str = "all",
        setup_id: str = "all",
        outcome: str = "all",
        trade_type: str = "all",
        search: str | None = None,
        limit: int = 200,
    ) -> list[ConfluenceRunRecord]:
        sql = "SELECT * FROM confluence_trade_journal WHERE 1=1"
        params: list[Any] = []

        if decision in ("GO", "NO-GO", "WATCH", "ERROR"):
            sql += " AND decision=?"
            params.append(decision)
        if direction in ("bullish", "bearish", "neutral"):
            sql += " AND direction=?"
            params.append(direction)
        if setup_id in ("A", "B", "C"):
            sql += " AND setup_id=?"
            params.append(setup_id)
        if outcome in ("WIN", "LOSS", "BREAKEVEN", "NOT_TRADED", "PENDING"):
            sql += " AND outcome=?"
            params.append(outcome)
        if trade_type == "actual":
            sql += " AND is_actual_trade=1"
        elif trade_type == "hypothetical":
            sql += " AND is_actual_trade=0"

        if search:
            like = f"%{search}%"
            sql += (" AND (contract_name LIKE ? OR blocked_reasons LIKE ? "
                    "OR decision_rationale LIKE ? OR notes LIKE ?)")
            params += [like] * 4

        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [self._to_record(r) for r in self.conn.execute(sql, params)]

    def settle(
        self,
        record_id: int,
        exit_price: float,
        is_actual: bool | None = None,
        notes: str | None = None,
    ) -> ConfluenceRunRecord | None:
        rec = self.get(record_id)
        if not rec or rec.entry_price is None or rec.entry_price <= 0:
            return None

        actual_flag = rec.is_actual_trade if is_actual is None else (1 if is_actual else 0)
        entry = rec.entry_price
        lot_qty = 75
        price_diff = exit_price - entry
        pnl = price_diff * lot_qty
        pnl_pct = (price_diff / entry) * 100.0 if entry > 0 else 0.0
        outcome = "WIN" if pnl > 100 else "LOSS" if pnl < -100 else "BREAKEVEN"

        if actual_flag:
            rec.actual_exit_price = round(exit_price, 2)
            rec.actual_pnl = round(pnl, 2)
            rec.actual_pnl_pct = round(pnl_pct, 2)
            rec.is_actual_trade = 1
        else:
            rec.hypothetical_exit_price = round(exit_price, 2)
            rec.hypothetical_pnl = round(pnl, 2)
            rec.hypothetical_pnl_pct = round(pnl_pct, 2)
            rec.is_actual_trade = 0

        rec.outcome = outcome
        if notes:
            rec.notes = f"{rec.notes}; {notes}".strip("; ")

        self.update(rec)
        return rec


_shared_confluence_journal: ConfluenceJournal | None = None


def shared_confluence_journal() -> ConfluenceJournal:
    global _shared_confluence_journal
    if _shared_confluence_journal is None:
        _shared_confluence_journal = ConfluenceJournal()
    return _shared_confluence_journal
