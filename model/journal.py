"""Setup journal: every scored setup becomes a candidate trade record.

Separate `setups` table in the same SQLite DB as the manual trade journal.
Stores the full decision context — indicator scores, weights, regime, sizing
— so backtests and weight optimization can later measure whether the model's
edge is real. Outcome fields stay NULL until reviewed/closed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS setups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    nifty_price REAL,
    contract TEXT DEFAULT '',
    expiry TEXT DEFAULT '',
    direction TEXT DEFAULT '',
    composite_score REAL,
    classification TEXT DEFAULT '',
    win_probability REAL,
    rr_ratio REAL,
    grade TEXT DEFAULT '',
    regime TEXT DEFAULT '',
    entry REAL,
    stop REAL,
    target REAL,
    contracts INTEGER,
    max_risk REAL,
    indicator_scores TEXT DEFAULT '{}',
    group_weights TEXT DEFAULT '{}',
    conflict_penalty REAL DEFAULT 0,
    confirmation_bonus REAL DEFAULT 0,
    regime_penalty REAL DEFAULT 0,
    blocked_reason TEXT DEFAULT '',
    outcome TEXT DEFAULT '' CHECK(outcome IN ('', 'win', 'loss', 'scratch', 'expired')),
    pnl REAL,
    notes TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_setups_created ON setups(created_at);
"""


@dataclass
class SetupRecord:
    created_at: str
    nifty_price: float | None = None
    contract: str = ""
    expiry: str = ""
    direction: str = ""
    composite_score: float | None = None
    classification: str = ""
    win_probability: float | None = None
    rr_ratio: float | None = None
    grade: str = ""
    regime: str = ""
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    contracts: int | None = None
    max_risk: float | None = None
    indicator_scores: dict = field(default_factory=dict)
    group_weights: dict = field(default_factory=dict)
    conflict_penalty: float = 0.0
    confirmation_bonus: float = 0.0
    regime_penalty: float = 0.0
    blocked_reason: str = ""
    outcome: str = ""
    pnl: float | None = None
    notes: str = ""
    id: int | None = None


class SetupJournal:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def record(self, s: SetupRecord) -> SetupRecord:
        data = asdict(s)
        data.pop("id")
        data["indicator_scores"] = json.dumps(data["indicator_scores"])
        data["group_weights"] = json.dumps(data["group_weights"])
        cols = ", ".join(data)
        ph = ", ".join("?" * len(data))
        cur = self.conn.execute(
            f"INSERT INTO setups ({cols}) VALUES ({ph})", list(data.values())
        )
        self.conn.commit()
        return replace_id(s, cur.lastrowid)

    def set_outcome(self, setup_id: int, outcome: str, pnl: float | None = None,
                    notes: str | None = None) -> bool:
        cur = self.conn.execute(
            "UPDATE setups SET outcome=?, pnl=?, notes=COALESCE(?, notes) WHERE id=?",
            (outcome, pnl, notes, setup_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def closed_setups(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM setups WHERE outcome IN ('win','loss','scratch') ORDER BY created_at"
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for col in ("indicator_scores", "group_weights"):
                try:
                    d[col] = json.loads(d[col])
                except (json.JSONDecodeError, TypeError):
                    d[col] = {}
            out.append(d)
        return out

    def summary(self) -> dict | None:
        """Performance review of settled setups — the go/no-go evidence."""
        rows = self.closed_setups()
        if len(rows) < 2:
            return None
        pnls = [r["pnl"] or 0.0 for r in rows]
        wins = [r for r in rows if r["outcome"] == "win"]
        losses = [r for r in rows if r["outcome"] == "loss"]
        gross_win = sum(p for r, p in zip(rows, pnls) if p > 0)
        gross_loss = abs(sum(p for r, p in zip(rows, pnls) if p < 0))
        return {
            "settled": len(rows),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / max(len(wins) + len(losses), 1), 3),
            "net_pnl": round(sum(pnls), 2),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
            "avg_pnl": round(sum(pnls) / len(pnls), 2),
            "by_direction": _group_count(rows, "direction"),
            "by_regime": _group_pnl(rows, "regime"),
        }


def _group_count(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[r.get(key) or "?"] = out.get(r.get(key) or "?", 0) + 1
    return out


def _group_pnl(rows: list[dict], key: str) -> dict:
    out: dict[str, float] = {}
    for r in rows:
        k = r.get(key) or "?"
        out[k] = round(out.get(k, 0.0) + (r["pnl"] or 0.0), 2)
    return out


def replace_id(s: SetupRecord, new_id: int | None) -> SetupRecord:
    import dataclasses
    return dataclasses.replace(s, id=new_id)


def shared_setup_journal() -> SetupJournal:
    global _shared
    try:
        return _shared
    except NameError:
        _shared = SetupJournal()
        return _shared


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
