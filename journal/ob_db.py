"""SQLite journal for the order-block system (zones, signals, paper trading).

Tables per docs/ORDER_BLOCKS.md §2.2. Rules:

- Ids are deterministic hashes (zone_id, signal_id, order tag), so a replay
  after a reconnect or restart hits UNIQUE and is a no-op, never a duplicate.
- Signals are immutable: a revised plan is a new row with `supersedes`.
- `ob_events` is append-only; recovery replays it.
- Live and backtest rows share tables, separated by `mode` + `run_id`, so the
  same performance code reads both.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ob_zones (
    zone_id TEXT PRIMARY KEY,
    series TEXT NOT NULL,
    timeframe TEXT NOT NULL CHECK(timeframe IN ('5m','15m','60m','1d')),
    direction TEXT NOT NULL CHECK(direction IN ('bullish','bearish')),
    source_bar_ts TEXT NOT NULL,
    bos_bar_ts TEXT NOT NULL,
    first_eligible_ts TEXT NOT NULL,
    zone_low REAL NOT NULL,
    zone_high REAL NOT NULL,
    zone_mid REAL NOT NULL,
    broken_swing REAL NOT NULL,
    broken_swing_ts TEXT NOT NULL,
    leg_origin REAL NOT NULL,
    atr_at_bos REAL NOT NULL,
    disp_body_atr REAL NOT NULL,
    disp_range_atr REAL NOT NULL,
    rvol REAL,
    fvg_low REAL, fvg_high REAL,
    swept_level REAL,
    kind TEXT NOT NULL DEFAULT 'BOS' CHECK(kind IN ('BOS','CHOCH')),
    status TEXT NOT NULL CHECK(status IN
        ('ACTIVE','TOUCHED','TRIGGERED','INVALIDATED','EXPIRED','CONSUMED')),
    touched_ts TEXT, closed_ts TEXT,
    close_reason TEXT DEFAULT '',
    bars_alive INTEGER NOT NULL DEFAULT 0,
    params_hash TEXT NOT NULL,
    engine_version TEXT NOT NULL DEFAULT 'ob-v1',
    mode TEXT NOT NULL CHECK(mode IN ('live','backtest')),
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obz_status ON ob_zones(status, timeframe);
CREATE INDEX IF NOT EXISTS idx_obz_run ON ob_zones(run_id);

CREATE TABLE IF NOT EXISTS ob_signals (
    signal_id TEXT PRIMARY KEY,
    zone_id TEXT NOT NULL,
    supersedes TEXT,
    horizon TEXT NOT NULL CHECK(horizon IN ('intraday','overnight')),
    trigger_ts TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('bullish','bearish')),
    u_entry REAL NOT NULL, u_stop REAL NOT NULL, u_target REAL NOT NULL,
    u_rr REAL NOT NULL,
    score REAL NOT NULL,
    score_json TEXT NOT NULL,
    p_win_calibrated REAL,
    structure TEXT DEFAULT '',
    legs_json TEXT DEFAULT '[]',
    o_entry REAL, o_stop REAL, o_target REAL,
    ev_rupees REAL,
    stress_loss_per_lot REAL,
    lots INTEGER NOT NULL DEFAULT 0,
    lot_size INTEGER,
    risk_rupees REAL,
    decision TEXT NOT NULL CHECK(decision IN ('GO','WATCH','NO-GO','SHADOW')),
    gates_json TEXT NOT NULL,
    blocked_reasons TEXT DEFAULT '',
    laya_json TEXT DEFAULT '',
    data_age_sec REAL,
    vix REAL,
    available_at TEXT,
    stress_json TEXT DEFAULT '',
    engine_version TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('live','backtest')),
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_decision ON ob_signals(decision, horizon);
CREATE INDEX IF NOT EXISTS idx_obs_run ON ob_signals(run_id);

CREATE TABLE IF NOT EXISTS ob_paper_orders (
    order_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    leg_index INTEGER NOT NULL,
    exchange TEXT NOT NULL DEFAULT 'NFO',
    tradingsymbol TEXT NOT NULL,
    transaction_type TEXT NOT NULL CHECK(transaction_type IN ('BUY','SELL')),
    product TEXT NOT NULL CHECK(product IN ('MIS','NRML')),
    order_type TEXT NOT NULL CHECK(order_type IN ('MARKET','LIMIT','SL','SL-M')),
    quantity INTEGER NOT NULL,
    price REAL, trigger_price REAL,
    purpose TEXT NOT NULL CHECK(purpose IN
        ('entry','stop','target','time_exit','gap_exit','manual','kill')),
    status TEXT NOT NULL CHECK(status IN ('OPEN','COMPLETE','CANCELLED','REJECTED')),
    filled_qty INTEGER NOT NULL DEFAULT 0,
    avg_price REAL,
    status_message TEXT DEFAULT '',
    placed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    UNIQUE (tag, purpose, leg_index, attempt)
);

CREATE TABLE IF NOT EXISTS ob_paper_fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    filled_at TEXT NOT NULL,
    qty INTEGER NOT NULL,
    price REAL NOT NULL,
    book_bid REAL, book_ask REAL,
    book_age_sec REAL,
    fill_model TEXT NOT NULL,
    charges REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ob_paper_positions (
    position_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL UNIQUE,
    horizon TEXT NOT NULL,
    structure TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT '',
    legs_json TEXT NOT NULL DEFAULT '[]',
    lots INTEGER NOT NULL, lot_size INTEGER NOT NULL,
    entry_net REAL NOT NULL,
    u_entry REAL, u_stop REAL NOT NULL, u_target REAL NOT NULL,
    o_stop REAL, o_target REAL,
    risk_rupees REAL,
    opened_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('OPEN','CLOSED')),
    closed_at TEXT, exit_net REAL,
    exit_reason TEXT CHECK(exit_reason IN
        ('u_stop','o_stop','target','time','gap','eod','expiry_guard','kill','manual','stale')),
    trigger_side TEXT,
    gross_pnl REAL, charges REAL, net_pnl REAL,
    r_multiple REAL,
    mae_rupees REAL, mfe_rupees REAL,
    gap_pnl REAL,
    mode TEXT NOT NULL DEFAULT 'live' CHECK(mode IN ('live','backtest')),
    run_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_obp_status ON ob_paper_positions(status, mode);

CREATE TABLE IF NOT EXISTS ob_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    ref_id TEXT,
    payload_json TEXT NOT NULL,
    run_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obe_run ON ob_events(run_id, seq);

CREATE TABLE IF NOT EXISTS ob_holdout_views (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    holdout_from TEXT NOT NULL,
    holdout_to TEXT NOT NULL,
    viewed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ob_backtest_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    git_sha TEXT NOT NULL,
    params_json TEXT NOT NULL,
    data_from TEXT NOT NULL, data_to TEXT NOT NULL,
    fold_spec_json TEXT NOT NULL,
    option_layer TEXT NOT NULL CHECK(option_layer IN ('none','archived','synthetic')),
    summary_json TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _iso(v) -> str | None:
    if v is None:
        return None
    return v.isoformat(timespec="seconds") if isinstance(v, datetime) else str(v)


@dataclass
class SignalRecord:
    signal_id: str
    zone_id: str
    horizon: str
    trigger_ts: str
    decided_at: str
    direction: str
    u_entry: float
    u_stop: float
    u_target: float
    u_rr: float
    score: float
    score_json: str
    decision: str
    gates_json: str
    engine_version: str
    params_hash: str
    mode: str
    run_id: str
    supersedes: str | None = None
    p_win_calibrated: float | None = None
    structure: str = ""
    legs_json: str = "[]"
    o_entry: float | None = None
    o_stop: float | None = None
    o_target: float | None = None
    ev_rupees: float | None = None
    stress_loss_per_lot: float | None = None
    lots: int = 0
    lot_size: int | None = None
    risk_rupees: float | None = None
    blocked_reasons: str = ""
    laya_json: str = ""
    data_age_sec: float | None = None
    vix: float | None = None
    available_at: str | None = None
    stress_json: str = ""
    created_at: str = ""

    @property
    def legs(self) -> list[dict]:
        try:
            return json.loads(self.legs_json or "[]")
        except ValueError:
            return []


@dataclass
class OrderRecord:
    order_id: str
    signal_id: str
    tag: str
    leg_index: int
    tradingsymbol: str
    transaction_type: str
    product: str
    order_type: str
    quantity: int
    purpose: str
    status: str
    placed_at: str
    updated_at: str
    exchange: str = "NFO"
    price: float | None = None
    trigger_price: float | None = None
    filled_qty: int = 0
    avg_price: float | None = None
    status_message: str = ""
    attempt: int = 0


@dataclass
class FillRecord:
    fill_id: str
    order_id: str
    filled_at: str
    qty: int
    price: float
    fill_model: str
    charges: float
    book_bid: float | None = None
    book_ask: float | None = None
    book_age_sec: float | None = None


@dataclass
class PositionRecord:
    position_id: str
    signal_id: str
    horizon: str
    structure: str
    lots: int
    lot_size: int
    entry_net: float
    u_stop: float
    u_target: float
    opened_at: str
    status: str = "OPEN"
    direction: str = ""
    legs_json: str = "[]"
    u_entry: float | None = None
    o_stop: float | None = None
    o_target: float | None = None
    risk_rupees: float | None = None
    closed_at: str | None = None
    exit_net: float | None = None
    exit_reason: str | None = None
    trigger_side: str | None = None
    gross_pnl: float | None = None
    charges: float | None = None
    net_pnl: float | None = None
    r_multiple: float | None = None
    mae_rupees: float | None = None
    mfe_rupees: float | None = None
    gap_pnl: float | None = None
    mode: str = "live"
    run_id: str = ""

    @property
    def legs(self) -> list[dict]:
        try:
            return json.loads(self.legs_json or "[]")
        except ValueError:
            return []

    @property
    def units(self) -> int:
        return self.lots * self.lot_size


def _cols(cls) -> list[str]:
    return [f.name for f in fields(cls)]


class ObJournal:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a table was first created."""
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(ob_signals)")}
        if "available_at" not in have:
            self.conn.execute("ALTER TABLE ob_signals ADD COLUMN available_at TEXT")
            self.conn.commit()
        if "stress_json" not in have:
            self.conn.execute("ALTER TABLE ob_signals ADD COLUMN stress_json TEXT DEFAULT ''")
            self.conn.commit()
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(ob_paper_orders)")]
        if "attempt" not in cols:
            # The UNIQUE key changed, which SQLite cannot alter: rebuild.
            self.conn.execute("ALTER TABLE ob_paper_orders RENAME TO ob_paper_orders_old")
            self.conn.executescript(_SCHEMA)
            names = ", ".join(cols)
            self.conn.execute(f"INSERT INTO ob_paper_orders ({names}) "
                              f"SELECT {names} FROM ob_paper_orders_old")
            self.conn.execute("DROP TABLE ob_paper_orders_old")
            self.conn.commit()

    # -- generic -------------------------------------------------------------

    def _insert(self, table: str, obj, ignore: bool = True) -> bool:
        cols = _cols(type(obj))
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        cur = self.conn.execute(
            f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [getattr(obj, c) for c in cols])
        self.conn.commit()
        return cur.rowcount > 0

    def _update(self, table: str, key: str, obj) -> None:
        cols = [c for c in _cols(type(obj)) if c != key]
        self.conn.execute(
            f"UPDATE {table} SET {', '.join(f'{c}=?' for c in cols)} WHERE {key}=?",
            [getattr(obj, c) for c in cols] + [getattr(obj, key)])
        self.conn.commit()

    @staticmethod
    def _row(cls, row: sqlite3.Row):
        return cls(**{c: row[c] for c in _cols(cls) if c in row.keys()})

    # -- zones ----------------------------------------------------------------

    def record_zone(self, zone, *, mode: str, run_id: str,
                    engine_version: str = "ob-v1") -> None:
        """Upsert a zone's current state (lifecycle changes overwrite)."""
        vals = {
            "zone_id": zone.zone_id, "series": zone.series,
            "timeframe": zone.timeframe, "direction": zone.direction,
            "source_bar_ts": _iso(zone.source_bar_ts), "bos_bar_ts": _iso(zone.bos_bar_ts),
            "first_eligible_ts": _iso(zone.first_eligible_ts),
            "zone_low": zone.zone_low, "zone_high": zone.zone_high,
            "zone_mid": zone.zone_mid, "broken_swing": zone.broken_swing,
            "broken_swing_ts": _iso(zone.broken_swing_ts),
            "leg_origin": zone.leg_origin, "atr_at_bos": zone.atr_at_bos,
            "disp_body_atr": zone.disp_body_atr, "disp_range_atr": zone.disp_range_atr,
            "rvol": zone.rvol, "fvg_low": zone.fvg_low, "fvg_high": zone.fvg_high,
            "swept_level": zone.swept_level, "kind": zone.kind, "status": zone.status,
            "touched_ts": _iso(zone.touched_ts), "closed_ts": _iso(zone.closed_ts),
            "close_reason": zone.close_reason, "bars_alive": zone.bars_alive,
            "params_hash": zone.params_hash, "engine_version": engine_version,
            "mode": mode, "run_id": run_id, "created_at": _now(),
        }
        cols = list(vals)
        update = [c for c in cols if c not in ("zone_id", "created_at")]
        self.conn.execute(
            f"INSERT INTO ob_zones ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
            f"ON CONFLICT (zone_id) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in update),
            [vals[c] for c in cols])
        self.conn.commit()

    def zones(self, status: str | None = None, run_id: str | None = None,
              limit: int = 200) -> list[dict]:
        sql, params = "SELECT * FROM ob_zones WHERE 1=1", []
        if status:
            sql += " AND status=?"
            params.append(status)
        if run_id:
            sql += " AND run_id=?"
            params.append(run_id)
        sql += " ORDER BY bos_bar_ts DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params)]

    # -- signals ----------------------------------------------------------------

    def add_signal(self, rec: SignalRecord) -> tuple[SignalRecord, bool]:
        """Insert; a replayed signal_id returns the stored row and False."""
        rec.created_at = rec.created_at or _now()
        inserted = self._insert("ob_signals", rec)
        return (rec if inserted else self.signal(rec.signal_id)), inserted

    def signal(self, signal_id: str) -> SignalRecord | None:
        row = self.conn.execute("SELECT * FROM ob_signals WHERE signal_id=?",
                                (signal_id,)).fetchone()
        return self._row(SignalRecord, row) if row else None

    def signals(self, *, decision: str | None = None, horizon: str | None = None,
                mode: str | None = None, run_id: str | None = None,
                limit: int = 200) -> list[SignalRecord]:
        sql, params = "SELECT * FROM ob_signals WHERE 1=1", []
        for col, val in (("decision", decision), ("horizon", horizon),
                         ("mode", mode), ("run_id", run_id)):
            if val:
                sql += f" AND {col}=?"
                params.append(val)
        sql += " ORDER BY trigger_ts DESC LIMIT ?"
        params.append(limit)
        return [self._row(SignalRecord, r) for r in self.conn.execute(sql, params)]

    # -- orders / fills -----------------------------------------------------------

    def add_order(self, rec: OrderRecord) -> tuple[OrderRecord, bool]:
        """Idempotent on (tag, purpose, leg_index) — except after a rejection.

        A replay of an order that is OPEN or COMPLETE returns that order and
        never fills twice. An order that was REJECTED (no book in the first
        seconds after the open, a stale feed) must be retryable, or the
        position it was closing could never close: the retry is stored as
        the next `attempt` of the same key.
        """
        rows = [self._row(OrderRecord, r) for r in self.conn.execute(
            "SELECT * FROM ob_paper_orders WHERE tag=? AND purpose=? AND leg_index=? "
            "ORDER BY attempt", (rec.tag, rec.purpose, rec.leg_index))]
        live = [o for o in rows if o.status in ("OPEN", "COMPLETE")]
        if live:
            return live[-1], False
        rec.attempt = len(rows)
        if self._insert("ob_paper_orders", rec):
            return rec, True
        return rows[-1], False

    def update_order(self, rec: OrderRecord) -> None:
        rec.updated_at = _now()
        self._update("ob_paper_orders", "order_id", rec)

    def orders(self, signal_id: str | None = None, status: str | None = None) -> list[OrderRecord]:
        sql, params = "SELECT * FROM ob_paper_orders WHERE 1=1", []
        if signal_id:
            sql += " AND signal_id=?"
            params.append(signal_id)
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY placed_at"
        return [self._row(OrderRecord, r) for r in self.conn.execute(sql, params)]

    def add_fill(self, rec: FillRecord) -> None:
        self._insert("ob_paper_fills", rec, ignore=False)

    def fills(self, order_id: str) -> list[FillRecord]:
        return [self._row(FillRecord, r) for r in self.conn.execute(
            "SELECT * FROM ob_paper_fills WHERE order_id=? ORDER BY filled_at", (order_id,))]

    # -- positions --------------------------------------------------------------------

    def open_position(self, rec: PositionRecord) -> tuple[PositionRecord, bool]:
        """One position per signal, ever (UNIQUE signal_id)."""
        if self._insert("ob_paper_positions", rec):
            return rec, True
        return self.position_for_signal(rec.signal_id), False

    def update_position(self, rec: PositionRecord) -> None:
        self._update("ob_paper_positions", "position_id", rec)

    def position(self, position_id: str) -> PositionRecord | None:
        row = self.conn.execute("SELECT * FROM ob_paper_positions WHERE position_id=?",
                                (position_id,)).fetchone()
        return self._row(PositionRecord, row) if row else None

    def position_for_signal(self, signal_id: str) -> PositionRecord | None:
        row = self.conn.execute("SELECT * FROM ob_paper_positions WHERE signal_id=?",
                                (signal_id,)).fetchone()
        return self._row(PositionRecord, row) if row else None

    def positions(self, *, status: str | None = None, mode: str | None = None,
                  run_id: str | None = None, limit: int = 10_000) -> list[PositionRecord]:
        sql, params = "SELECT * FROM ob_paper_positions WHERE 1=1", []
        for col, val in (("status", status), ("mode", mode), ("run_id", run_id)):
            if val:
                sql += f" AND {col}=?"
                params.append(val)
        sql += " ORDER BY opened_at LIMIT ?"
        params.append(limit)
        return [self._row(PositionRecord, r) for r in self.conn.execute(sql, params)]

    # -- events -------------------------------------------------------------------------

    def log_event(self, kind: str, payload: dict[str, Any], *, run_id: str,
                  ref_id: str | None = None, ts: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO ob_events (ts, kind, ref_id, payload_json, run_id) VALUES (?,?,?,?,?)",
            (ts or _now(), kind, ref_id, json.dumps(payload, default=str), run_id))
        self.conn.commit()
        return cur.lastrowid

    def events(self, run_id: str | None = None, since_seq: int = 0,
               kind: str | None = None, limit: int = 10_000) -> list[dict]:
        sql, params = "SELECT * FROM ob_events WHERE seq>?", [since_seq]
        if run_id:
            sql += " AND run_id=?"
            params.append(run_id)
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        sql += " ORDER BY seq LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params)]

    # -- backtest runs ------------------------------------------------------------------

    def save_run(self, *, run_id: str, git_sha: str, params_json: str,
                 data_from: str, data_to: str, fold_spec: dict,
                 option_layer: str, summary: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO ob_backtest_runs VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, _now(), git_sha, params_json, data_from, data_to,
             json.dumps(fold_spec, default=str), option_layer,
             json.dumps(summary, default=str)))
        self.conn.commit()

    def log_holdout_view(self, run_id: str, params_hash: str, frm: str, to: str) -> None:
        self.conn.execute(
            "INSERT INTO ob_holdout_views (run_id, params_hash, holdout_from, holdout_to, "
            "viewed_at) VALUES (?,?,?,?,?)", (run_id, params_hash, frm, to, _now()))
        self.conn.commit()

    def holdout_views(self, params_hash: str, frm: str, to: str) -> list[dict]:
        """Earlier unsealings of a holdout that overlaps [frm, to] for these params."""
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM ob_holdout_views WHERE params_hash=? AND holdout_from<=? "
            "AND holdout_to>=? ORDER BY seq", (params_hash, to, frm))]

    def runs(self, limit: int = 20) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM ob_backtest_runs ORDER BY created_at DESC LIMIT ?", (limit,))]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def as_dict(rec) -> dict:
    return asdict(rec)


_shared: ObJournal | None = None


def shared_ob_journal() -> ObJournal:
    global _shared
    if _shared is None:
        _shared = ObJournal()
    return _shared


__all__ = ["ObJournal", "SignalRecord", "OrderRecord", "FillRecord",
           "PositionRecord", "shared_ob_journal", "new_id", "field"]
