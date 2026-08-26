"""Dedicated SQLite-backed journal for Overnight Trade Decisions.

Records EVERY execution of the overnight decision engine (GO, NO-GO, ERROR),
capturing full technical indicators, Greek EV bridges, counterfactual
hypothetical outcomes, and settlement tracking.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS overnight_trade_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    nifty_close REAL NOT NULL,
    market_regime TEXT NOT NULL,
    direction TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('GO', 'NO-GO', 'ERROR', 'WATCH')),
    confidence_score REAL NOT NULL,
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
    outcome TEXT NOT NULL DEFAULT 'PENDING' CHECK(outcome IN ('WIN', 'LOSS', 'BREAKEVEN', 'NOT_TRADED', 'PENDING')),
    is_actual_trade INTEGER NOT NULL DEFAULT 0,
    rel_volume REAL,
    close_pos REAL,
    close_location TEXT DEFAULT '',
    micro_trend TEXT DEFAULT '',
    rsi_val REAL,
    macd_val REAL,
    adx_val REAL,
    atr_val REAL,
    vix_val REAL,
    matched_bucket TEXT DEFAULT '',
    cohort_n INTEGER DEFAULT 0,
    expected_value_lot REAL,
    expected_value_pct REAL,
    p_direction REAL,
    p_profitable REAL,
    p10_loss_lot REAL,
    contracts INTEGER,
    max_risk REAL,
    signal_scores TEXT DEFAULT '{}',
    decision_rationale TEXT DEFAULT '',
    blocked_reasons TEXT DEFAULT '',
    engine_version TEXT NOT NULL DEFAULT 'v2.2-ev-dist',
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_oj_decision ON overnight_trade_journal(decision);
CREATE INDEX IF NOT EXISTS idx_oj_trade_date ON overnight_trade_journal(trade_date);
CREATE INDEX IF NOT EXISTS idx_oj_outcome ON overnight_trade_journal(outcome);
CREATE INDEX IF NOT EXISTS idx_oj_direction ON overnight_trade_journal(direction);
"""


@dataclass
class OvernightRunRecord:
    id: int | None
    run_id: str
    timestamp: str                      # ISO execution timestamp
    trade_date: str                     # YYYY-MM-DD
    nifty_close: float
    market_regime: str
    direction: str                      # bullish, bearish, neutral
    decision: str                       # GO, NO-GO, ERROR
    confidence_score: float
    option_type: str = ""               # CE, PE, SPREAD, NONE
    option_strike: float | None = None
    contract_name: str = ""             # e.g. "NIFTY 24250 PE"
    expiry: str = ""                    # YYYY-MM-DD
    entry_price: float | None = None
    expected_exit: float | None = None
    actual_exit_price: float | None = None
    actual_pnl: float | None = None
    actual_pnl_pct: float | None = None
    hypothetical_exit_price: float | None = None
    hypothetical_pnl: float | None = None
    hypothetical_pnl_pct: float | None = None
    outcome: str = "PENDING"            # WIN, LOSS, BREAKEVEN, NOT_TRADED, PENDING
    is_actual_trade: int = 0            # 1 for GO, 0 for NO-GO / hypothetical
    rel_volume: float | None = None
    close_pos: float | None = None
    close_location: str = ""
    micro_trend: str = ""
    rsi_val: float | None = None
    macd_val: float | None = None
    adx_val: float | None = None
    atr_val: float | None = None
    vix_val: float | None = None
    matched_bucket: str = ""
    cohort_n: int = 0
    expected_value_lot: float | None = None
    expected_value_pct: float | None = None
    p_direction: float | None = None
    p_profitable: float | None = None
    p10_loss_lot: float | None = None
    contracts: int | None = None
    max_risk: float | None = None
    signal_scores: str = "{}"           # JSON string
    decision_rationale: str = ""
    blocked_reasons: str = ""
    engine_version: str = "v2.2-ev-dist"
    notes: str = ""
    created_at: str = ""

    @property
    def effective_pnl(self) -> float | None:
        """Actual P&L for GO trades, or hypothetical P&L for NO-GO trades."""
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


class OvernightJournal:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    # -- helpers -------------------------------------------------------------

    def _to_record(self, row: sqlite3.Row) -> OvernightRunRecord:
        data = {f.name: row[f.name] for f in fields(OvernightRunRecord) if f.name in row.keys()}
        return OvernightRunRecord(**data)

    @staticmethod
    def _cols() -> list[str]:
        return [f.name for f in fields(OvernightRunRecord) if f.name != "id"]

    # -- CRUD ------------------------------------------------------------------

    def add(self, rec: OvernightRunRecord) -> OvernightRunRecord:
        if not rec.run_id:
            now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            rec.run_id = f"ON-{now_str}-{uuid.uuid4().hex[:4].upper()}"
        if not rec.created_at:
            rec.created_at = datetime.now().isoformat(timespec="seconds")
        if not rec.trade_date:
            rec.trade_date = rec.timestamp[:10] if rec.timestamp else datetime.now().strftime("%Y-%m-%d")

        cols = self._cols()
        values = [getattr(rec, c) for c in cols]
        placeholders = ", ".join("?" * len(cols))
        cur = self.conn.execute(
            f"INSERT INTO overnight_trade_journal ({', '.join(cols)}) VALUES ({placeholders})", values
        )
        self.conn.commit()
        return replace(rec, id=cur.lastrowid)

    def get(self, record_id: int) -> OvernightRunRecord | None:
        row = self.conn.execute(
            "SELECT * FROM overnight_trade_journal WHERE id=?", (record_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def get_by_run_id(self, run_id: str) -> OvernightRunRecord | None:
        row = self.conn.execute(
            "SELECT * FROM overnight_trade_journal WHERE run_id=?", (run_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def update(self, rec: OvernightRunRecord) -> None:
        cols = self._cols()
        assignments = ", ".join(f"{c}=?" for c in cols)
        values = [getattr(rec, c) for c in cols] + [rec.id]
        self.conn.execute(f"UPDATE overnight_trade_journal SET {assignments} WHERE id=?", values)
        self.conn.commit()

    def delete(self, record_id: int) -> bool:
        cur = self.conn.execute(
            "DELETE FROM overnight_trade_journal WHERE id=?", (record_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # -- Queries & Filters -----------------------------------------------------

    def list(
        self,
        decision: str = "all",
        direction: str = "all",
        regime: str = "all",
        outcome: str = "all",
        trade_type: str = "all",        # 'all', 'actual', 'hypothetical'
        search: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 200,
    ) -> list[OvernightRunRecord]:
        sql = "SELECT * FROM overnight_trade_journal WHERE 1=1"
        params: list[Any] = []

        if decision in ("GO", "NO-GO", "ERROR", "WATCH"):
            sql += " AND decision=?"
            params.append(decision)
        if direction in ("bullish", "bearish", "neutral"):
            sql += " AND direction=?"
            params.append(direction)
        if regime and regime != "all":
            sql += " AND market_regime=?"
            params.append(regime)
        if outcome in ("WIN", "LOSS", "BREAKEVEN", "NOT_TRADED", "PENDING"):
            sql += " AND outcome=?"
            params.append(outcome)
        if trade_type == "actual":
            sql += " AND is_actual_trade=1"
        elif trade_type == "hypothetical":
            sql += " AND is_actual_trade=0"

        if date_from:
            sql += " AND trade_date >= ?"
            params.append(date_from)
        if date_to:
            sql += " AND trade_date <= ?"
            params.append(date_to)

        if search:
            like = f"%{search}%"
            sql += (" AND (contract_name LIKE ? OR matched_bucket LIKE ? "
                    "OR decision_rationale LIKE ? OR blocked_reasons LIKE ? OR notes LIKE ?)")
            params += [like] * 5

        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [self._to_record(r) for r in self.conn.execute(sql, params)]

    # -- Settlement ------------------------------------------------------------

    def settle(
        self,
        record_id: int,
        exit_price: float,
        is_actual: bool | None = None,
        notes: str | None = None,
    ) -> OvernightRunRecord | None:
        """Settle a trade or hypothetical run with next-open exit price."""
        rec = self.get(record_id)
        if not rec or rec.entry_price is None or rec.entry_price <= 0:
            return None

        actual_flag = rec.is_actual_trade if is_actual is None else (1 if is_actual else 0)
        entry = rec.entry_price
        lot_qty = (rec.contracts or 1) * 75
        price_diff = exit_price - entry
        
        # Long Option PnL
        pnl = price_diff * lot_qty
        pnl_pct = (price_diff / entry) * 100.0 if entry > 0 else 0.0
        
        # Outcome classification
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


_shared_overnight_journal: OvernightJournal | None = None


def shared_overnight_journal() -> OvernightJournal:
    global _shared_overnight_journal
    if _shared_overnight_journal is None:
        _shared_overnight_journal = OvernightJournal()
    return _shared_overnight_journal
