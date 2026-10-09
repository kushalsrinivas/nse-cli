"""Append-only journal for premarket paper candidates and blocked runs."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS premarket_paper_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    data_date TEXT DEFAULT '',
    history_source TEXT DEFAULT '',
    history_fetched_at TEXT DEFAULT '',
    history_cached INTEGER NOT NULL DEFAULT 0,
    chain_source TEXT DEFAULT '',
    chain_fetched_at TEXT DEFAULT '',
    model_action TEXT NOT NULL,
    paper_status TEXT NOT NULL,
    forced INTEGER NOT NULL DEFAULT 0,
    candidate_name TEXT DEFAULT '',
    candidate_kind TEXT DEFAULT '',
    expiry TEXT DEFAULT '',
    lots INTEGER NOT NULL DEFAULT 0,
    lot_size INTEGER NOT NULL DEFAULT 0,
    risk_budget_rupees REAL NOT NULL DEFAULT 0,
    net_debit REAL,
    max_loss_per_lot REAL,
    max_loss_total REAL,
    edge_after_hurdle REAL,
    edge_ci_low REAL,
    edge_ci_high REAL,
    p_profit REAL,
    notes TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_premarket_paper_date
    ON premarket_paper_journal(trade_date);
"""


@dataclass
class PremarketPaperRecord:
    id: int | None
    run_id: str
    created_at: str
    trade_date: str
    data_date: str = ""
    history_source: str = ""
    history_fetched_at: str = ""
    history_cached: int = 0
    chain_source: str = ""
    chain_fetched_at: str = ""
    model_action: str = "NOT EVALUATED"
    paper_status: str = "NO CANDIDATE"
    forced: int = 0
    candidate_name: str = ""
    candidate_kind: str = ""
    expiry: str = ""
    lots: int = 0
    lot_size: int = 0
    risk_budget_rupees: float = 0.0
    net_debit: float | None = None
    max_loss_per_lot: float | None = None
    max_loss_total: float | None = None
    edge_after_hurdle: float | None = None
    edge_ci_low: float | None = None
    edge_ci_high: float | None = None
    p_profit: float | None = None
    notes: str = ""


class PremarketJournal:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def add(self, record: PremarketPaperRecord) -> PremarketPaperRecord:
        if not record.run_id:
            record.run_id = f"PM-{uuid.uuid4().hex[:12].upper()}"
        if not record.created_at:
            now = datetime.now()
            record.created_at = now.isoformat(timespec="seconds")
            if not record.trade_date:
                record.trade_date = now.strftime("%Y-%m-%d")
        cols = [f.name for f in fields(PremarketPaperRecord) if f.name != "id"]
        values = [getattr(record, col) for col in cols]
        placeholders = ", ".join("?" * len(cols))
        cur = self.conn.execute(
            f"INSERT INTO premarket_paper_journal ({', '.join(cols)}) "
            f"VALUES ({placeholders})", values)
        self.conn.commit()
        return replace(record, id=cur.lastrowid)

    def record_run(self, *, history=None, chain=None, result=None,
                   requested_source: str = "auto", status: str | None = None,
                   note: str = "", at: datetime | None = None,
                   paper_entry: bool = True) -> PremarketPaperRecord:
        at = at or datetime.now()
        candles = getattr(history, "candles", []) or []
        candidate = (getattr(result, "paper_candidate", None)
                     if paper_entry else None)
        sizing = (getattr(result, "paper_risk", None)
                  if paper_entry else None)
        if status is None:
            if candidate is None:
                status = "NO_CANDIDATE"
            elif sizing is None or not sizing.allowed:
                status = "RISK_BLOCKED"
            elif getattr(result, "paper_forced", False):
                status = "FORCED_PAPER"
            else:
                status = "MODEL_GO_PAPER"

        strategy = candidate.structure if candidate is not None else None
        loss = abs(candidate.max_loss) if candidate is not None else None
        notes = [note] if note else []
        if sizing is not None and not sizing.allowed and sizing.reason:
            notes.append(sizing.reason)
        decision = getattr(result, "decision", None)
        edge_ci = getattr(candidate, "edge_ci", (None, None))
        record = PremarketPaperRecord(
            id=None,
            run_id=f"PM-{at.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}",
            created_at=at.isoformat(timespec="seconds"),
            trade_date=at.strftime("%Y-%m-%d"),
            data_date=(candles[-1].timestamp.strftime("%Y-%m-%d")
                       if candles else ""),
            history_source=getattr(history, "source", "") or "",
            history_fetched_at=(getattr(history, "fetched_at", at).isoformat()
                                if history is not None else ""),
            history_cached=int(bool(getattr(history, "from_cache", False))),
            chain_source=(getattr(chain, "source", "") if chain is not None
                          else ""),
            chain_fetched_at=(getattr(chain, "fetched_at", at).isoformat()
                              if chain is not None else ""),
            model_action=(getattr(decision, "action", "NOT EVALUATED")
                          if decision is not None else "NOT EVALUATED"),
            paper_status=status,
            forced=int(bool(getattr(result, "paper_forced", False)
                            and paper_entry)),
            candidate_name=(strategy.name if strategy is not None else ""),
            candidate_kind=(strategy.kind if strategy is not None else ""),
            expiry=(strategy.expiry if strategy is not None else ""),
            lots=(sizing.contracts if sizing is not None and sizing.allowed else 0),
            lot_size=(getattr(result, "paper_lot_size", 0) if result is not None else 0),
            risk_budget_rupees=(getattr(
                result, "paper_budget_rupees", SETTINGS.premarket_risk_budget_rupees)
                if result is not None else SETTINGS.premarket_risk_budget_rupees),
            net_debit=(strategy.net_debit if strategy is not None else None),
            max_loss_per_lot=loss,
            max_loss_total=(sizing.max_risk_rupees
                            if sizing is not None and sizing.allowed else None),
            edge_after_hurdle=(candidate.edge_after_hurdle
                               if candidate is not None else None),
            edge_ci_low=edge_ci[0], edge_ci_high=edge_ci[1],
            p_profit=(candidate.p_profit if candidate is not None else None),
            notes="; ".join(notes),
        )
        return self.add(record)

    def list(self, limit: int = 50) -> list[PremarketPaperRecord]:
        rows = self.conn.execute(
            "SELECT * FROM premarket_paper_journal ORDER BY id DESC LIMIT ?",
            (limit,))
        return [PremarketPaperRecord(**dict(row)) for row in rows]


_shared: PremarketJournal | None = None


def shared_premarket_journal() -> PremarketJournal:
    global _shared
    if _shared is None:
        _shared = PremarketJournal()
    return _shared
