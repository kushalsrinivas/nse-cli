"""SQLite journal of Laya verdicts, keyed to the setup runs they judged.

Lives in the shared journal.db next to confluence_trade_journal so the
evaluation can join a verdict to its settled outcome on (run_id, setup_id).
Every verdict is recorded — kept and vetoed, GO and not — because the only
question that matters is whether vetoed setups actually do worse.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS laya_verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    run_id TEXT NOT NULL,
    setup_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    direction TEXT NOT NULL,
    would_veto INTEGER NOT NULL,
    enforced INTEGER NOT NULL DEFAULT 0,
    reasons TEXT DEFAULT '',
    p_fail REAL,
    p_execute REAL,
    quality REAL,
    laya_direction TEXT,
    regime TEXT,
    answers_json TEXT DEFAULT '{}',
    model TEXT DEFAULT '',
    latency_ms REAL,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lv_run ON laya_verdicts(run_id, setup_id);
"""

#: Below this many settled rows per arm, the evaluation reports but does
#: not conclude.
MIN_EVAL_ROWS = 30


class LayaJournal:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, run_id: str, verdict, *, enforced: bool = False) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO laya_verdicts (source, run_id, setup_id, decision,"
                " direction, would_veto, enforced, reasons, p_fail, p_execute,"
                " quality, laya_direction, regime, answers_json, model,"
                " latency_ms, error, created_at) VALUES"
                " (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (verdict.source, run_id, verdict.setup_id, verdict.decision,
                 verdict.direction, int(verdict.would_veto),
                 int(enforced and verdict.veto), "; ".join(verdict.reasons),
                 verdict.p_fail, verdict.p_execute, verdict.quality,
                 verdict.laya_direction, verdict.regime,
                 json.dumps(verdict.answers), verdict.model,
                 verdict.latency_ms, verdict.error,
                 datetime.now().isoformat(timespec="seconds")))
            return int(cur.lastrowid)

    def count(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM laya_verdicts").fetchone()[0])

    def settled_confluence(self) -> list[sqlite3.Row]:
        """Verdicts joined to settled confluence rows (WIN/LOSS/BREAKEVEN)."""
        with self._conn() as c:
            has_cj = c.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='confluence_trade_journal'").fetchone()
            if not has_cj:
                return []
            return c.execute(
                "SELECT v.*, j.outcome, j.decision AS engine_decision,"
                " COALESCE(CASE WHEN j.is_actual_trade=1 THEN j.actual_pnl END,"
                "          j.hypothetical_pnl) AS pnl"
                " FROM laya_verdicts v JOIN confluence_trade_journal j"
                "   ON j.run_id = v.run_id AND j.setup_id = v.setup_id"
                " WHERE v.source='confluence' AND v.error IS NULL"
                "   AND j.outcome IN ('WIN','LOSS','BREAKEVEN')").fetchall()


@dataclass
class ArmStats:
    n: int = 0
    wins: int = 0
    pnl_sum: float = 0.0

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.n if self.n else None

    @property
    def mean_pnl(self) -> float | None:
        return self.pnl_sum / self.n if self.n else None


def evaluate_rows(rows) -> dict:
    """Kept vs would-veto arms, plus rank AUC of P(fail) against losses.

    AUC > 0.5 means higher P(fail) goes with losing setups — the one
    property a filter needs. None when either class is empty.
    """
    kept, vetoed = ArmStats(), ArmStats()
    scored: list[tuple[float, int]] = []
    for r in rows:
        arm = vetoed if r["would_veto"] else kept
        arm.n += 1
        arm.wins += r["outcome"] == "WIN"
        arm.pnl_sum += float(r["pnl"] or 0.0)
        if r["p_fail"] is not None:
            scored.append((float(r["p_fail"]), int(r["outcome"] == "LOSS")))
    return {"kept": kept, "vetoed": vetoed, "auc_p_fail": _auc(scored),
            "n": kept.n + vetoed.n,
            "conclusive": min(kept.n, vetoed.n) >= MIN_EVAL_ROWS}


def _auc(pairs: list[tuple[float, int]]) -> float | None:
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
