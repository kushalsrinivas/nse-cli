"""Run registry: every paper session and backtest records exactly what it used.

A run row pins the validated config (by hash, with its full body stored in
`config_versions`), the strategy version (engine version + git sha), the
universe snapshot and a data version (a fingerprint of market.db's bar
watermarks). Re-running with the same four values must reproduce the same
signals; tests/test_platform_phase1.py checks the bookkeeping.
"""

from __future__ import annotations

import hashlib
import subprocess
import uuid
from datetime import datetime

from model.order_blocks.params import ENGINE_VERSION


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def strategy_version() -> str:
    return f"{ENGINE_VERSION}+platform@{git_sha()}"


def data_version(market_conn) -> str:
    """Fingerprint of what bar data exists: count + max ts per instrument."""
    rows = market_conn.execute(
        "SELECT instrument_key, COUNT(*), MAX(ts) FROM bars_1m GROUP BY instrument_key "
        "ORDER BY instrument_key").fetchall()
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r[0]}|{r[1]}|{r[2]};".encode())
    return f"bars:{len(rows)}:{h.hexdigest()[:12]}"


def record_config(app_conn, cfg, note: str = "") -> str:
    app_conn.execute(
        "INSERT OR IGNORE INTO config_versions (config_hash, body_json, created_at, note) "
        "VALUES (?,?,?,?)", (cfg.hash, cfg.canonical(), datetime.now().isoformat(timespec="seconds"), note))
    app_conn.commit()
    return cfg.hash


def start_run(app_conn, market_conn, cfg, *, kind: str, universe_snapshot: str = "none",
              notes: str = "") -> str:
    record_config(app_conn, cfg)
    run_id = f"{kind}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    app_conn.execute(
        "INSERT INTO runs (run_id, kind, config_hash, strategy_version, universe_snapshot, "
        "data_version, started_at, status, notes) VALUES (?,?,?,?,?,?,?,?,?)",
        (run_id, kind, cfg.hash, strategy_version(), universe_snapshot,
         data_version(market_conn), datetime.now().isoformat(timespec="seconds"), "running", notes))
    app_conn.commit()
    return run_id


def end_run(app_conn, run_id: str, status: str = "completed", notes: str = "") -> None:
    app_conn.execute("UPDATE runs SET ended_at=?, status=?, notes=COALESCE(NULLIF(?, ''), notes) "
                     "WHERE run_id=?",
                     (datetime.now().isoformat(timespec="seconds"), status, notes, run_id))
    app_conn.commit()


def get_run(app_conn, run_id: str) -> dict | None:
    row = app_conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(row) if row else None
