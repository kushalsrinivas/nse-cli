"""SQLite-backed trade journal.

One table, one dataclass, plain sqlite3 — no ORM. All timestamps stored as
ISO strings (UTC-naive local market convention is fine for a journal).
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('long', 'short')),
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL DEFAULT 1,
    stop_loss REAL,
    target REAL,
    strategy TEXT NOT NULL DEFAULT 'manual',
    entry_reason TEXT DEFAULT '',
    exit_reason TEXT DEFAULT '',
    macd_state TEXT DEFAULT '',
    ema_state TEXT DEFAULT '',
    sma_state TEXT DEFAULT '',
    volume_state TEXT DEFAULT '',
    pnl REAL,
    pnl_pct REAL,
    duration_minutes REAL,
    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'closed')),
    notes TEXT DEFAULT '',
    option_type TEXT DEFAULT '',
    strike REAL,
    expiry TEXT DEFAULT '',
    lots INTEGER,
    lot_size INTEGER,
    delta_entry REAL
);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
"""

# Columns added after v1 — applied to existing databases via ALTER TABLE.
_MIGRATIONS = (
    ("option_type", "TEXT DEFAULT ''"),
    ("strike", "REAL"),
    ("expiry", "TEXT DEFAULT ''"),
    ("lots", "INTEGER"),
    ("lot_size", "INTEGER"),
    ("delta_entry", "REAL"),
)


@dataclass
class Trade:
    id: int | None
    timestamp: str                  # entry time ISO
    instrument: str
    direction: str                  # 'long' | 'short'
    entry_price: float
    exit_price: float | None = None
    quantity: float = 1.0           # total units (options: lots × lot_size)
    stop_loss: float | None = None
    target: float | None = None
    strategy: str = "manual"
    entry_reason: str = ""
    exit_reason: str = ""
    macd_state: str = ""
    ema_state: str = ""
    sma_state: str = ""
    volume_state: str = ""
    pnl: float | None = None
    pnl_pct: float | None = None
    duration_minutes: float | None = None
    status: str = "open"
    notes: str = ""
    # Options-specific (empty/None for plain index trades)
    option_type: str = ""           # 'CE' | 'PE'
    strike: float | None = None
    expiry: str = ""                # ISO date
    lots: int | None = None
    lot_size: int | None = None
    delta_entry: float | None = None

    @property
    def contract_name(self) -> str:
        if not self.option_type:
            return self.instrument
        return f"{self.instrument} {self.strike:g} {self.option_type} {self.expiry}"

    @staticmethod
    def compute_pnl(trade: "Trade") -> tuple[float, float] | None:
        """(pnl absolute, pnl %) for a closed trade; direction-aware."""
        if trade.exit_price is None:
            return None
        sign = 1 if trade.direction == "long" else -1
        diff = (trade.exit_price - trade.entry_price) * sign
        pnl = diff * trade.quantity
        pct = diff / trade.entry_price * 100 if trade.entry_price else 0.0
        return round(pnl, 2), round(pct, 4)


class Journal:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add post-v1 columns to databases created before options support."""
        existing = {r["name"] for r in self.conn.execute("PRAGMA table_info(trades)")}
        for col, decl in _MIGRATIONS:
            if col not in existing:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {decl}")
        self.conn.commit()

    # -- helpers -------------------------------------------------------------

    def _to_trade(self, row: sqlite3.Row) -> Trade:
        return Trade(**{f.name: row[f.name] for f in fields(Trade)})

    @staticmethod
    def _cols() -> list[str]:
        return [f.name for f in fields(Trade) if f.name != "id"]

    # -- CRUD ------------------------------------------------------------------

    def add(self, trade: Trade) -> Trade:
        cols = self._cols()
        values = [getattr(trade, c) for c in cols]
        placeholders = ", ".join("?" * len(cols))
        cur = self.conn.execute(
            f"INSERT INTO trades ({', '.join(cols)}) VALUES ({placeholders})", values
        )
        self.conn.commit()
        return replace(trade, id=cur.lastrowid)

    def get(self, trade_id: int) -> Trade | None:
        row = self.conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        return self._to_trade(row) if row else None

    def update(self, trade: Trade) -> None:
        cols = self._cols()
        assignments = ", ".join(f"{c}=?" for c in cols)
        values = [getattr(trade, c) for c in cols] + [trade.id]
        self.conn.execute(f"UPDATE trades SET {assignments} WHERE id=?", values)
        self.conn.commit()

    def delete(self, trade_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM trades WHERE id=?", (trade_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # -- queries ---------------------------------------------------------------

    def list(self, status: str = "all", search: str | None = None,
             strategy: str | None = None, limit: int = 200) -> list[Trade]:
        sql = "SELECT * FROM trades WHERE 1=1"
        params: list[Any] = []
        if status in ("open", "closed"):
            sql += " AND status=?"
            params.append(status)
        if strategy:
            sql += " AND strategy=?"
            params.append(strategy)
        if search:
            like = f"%{search}%"
            sql += (" AND (instrument LIKE ? OR strategy LIKE ? OR notes LIKE ? "
                    "OR entry_reason LIKE ? OR exit_reason LIKE ?)")
            params += [like] * 5
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return [self._to_trade(r) for r in self.conn.execute(sql, params)]

    def strategies(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT strategy FROM trades ORDER BY strategy"
        ).fetchall()
        return [r["strategy"] for r in rows]

    def all_closed(self) -> list[Trade]:
        """Closed trades oldest-first (for analytics)."""
        return [
            self._to_trade(r)
            for r in self.conn.execute(
                "SELECT * FROM trades WHERE status='closed' ORDER BY timestamp ASC"
            )
        ]

    # -- domain actions ----------------------------------------------------------

    def open_trade(self, *, instrument: str, direction: str, entry_price: float,
                   quantity: float = 1.0, stop_loss: float | None = None,
                   target: float | None = None, strategy: str = "manual",
                   entry_reason: str = "", states: dict[str, str] | None = None,
                   timestamp: datetime | None = None,
                   option_type: str = "", strike: float | None = None,
                   expiry: str = "", lots: int | None = None,
                   lot_size: int | None = None,
                   delta_entry: float | None = None) -> Trade:
        states = states or {}
        return self.add(Trade(
            id=None,
            timestamp=(timestamp or datetime.now()).isoformat(timespec="seconds"),
            instrument=instrument,
            direction="long" if direction.lower().startswith("l") else "short",
            entry_price=float(entry_price),
            quantity=float(quantity),
            stop_loss=stop_loss,
            target=target,
            strategy=strategy,
            entry_reason=entry_reason,
            macd_state=states.get("macd", ""),
            ema_state=states.get("ema", ""),
            sma_state=states.get("sma", ""),
            volume_state=states.get("volume", ""),
            option_type=option_type.upper(),
            strike=strike,
            expiry=expiry,
            lots=lots,
            lot_size=lot_size,
            delta_entry=delta_entry,
        ))

    def close_trade(self, trade_id: int, exit_price: float,
                    exit_reason: str = "", when: datetime | None = None) -> Trade | None:
        trade = self.get(trade_id)
        if not trade or trade.status == "closed":
            return None
        result = dict(zip(self._cols(), ([getattr(trade, c) for c in self._cols()])))
        closed = replace(
            trade,
            exit_price=float(exit_price),
            exit_reason=exit_reason,
            status="closed",
        )
        pnl = Trade.compute_pnl(closed)
        if pnl:
            closed.pnl, closed.pnl_pct = pnl
        entry_dt = datetime.fromisoformat(trade.timestamp)
        closed.duration_minutes = round(
            ((when or datetime.now()) - entry_dt).total_seconds() / 60, 1
        )
        self.update(closed)
        return closed

    def set_notes(self, trade_id: int, notes: str) -> bool:
        trade = self.get(trade_id)
        if not trade:
            return False
        self.update(replace(trade, notes=notes))
        return True


_shared_journal: Journal | None = None


def shared_journal() -> Journal:
    global _shared_journal
    if _shared_journal is None:
        _shared_journal = Journal()
    return _shared_journal
