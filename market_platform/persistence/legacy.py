"""Import the NIFTY order-block system's data into the platform databases.

Copies (idempotently, INSERT OR IGNORE) from the legacy `journal.db`:

    ob_series_1m        → market.bars_1m   NIFTY_SPOT → 'NSE:NIFTY 50',
                                           NIFTY_FUT1 → 'NFO:<contract>'
    kite_candles_1m     → market.bars_1m   token → 'EXCHANGE:SYMBOL' via the master
    option_quotes       → market.option_quotes
    option_candles_1m   → market.option_candles_1m
    ob_zones            → app.zones        (run 'import-legacy')
    ob_signals          → app.signals      (strategy 'nifty-ob-v1')

The legacy tables are only read. Every existing command keeps using them,
so nothing that works today changes; the import can be re-run at any time.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

LEGACY_RUN = "import-legacy"

_DECISION_TO_STATUS = {"GO": "APPROVED", "WATCH": "WATCH", "NO-GO": "REJECTED",
                       "SHADOW": "SUPPRESSED"}


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _instrument_key_for_token(legacy, token: int) -> str | None:
    row = legacy.execute("SELECT exchange, tradingsymbol FROM kite_instruments "
                         "WHERE instrument_token=? ORDER BY as_of DESC LIMIT 1", (token,)).fetchone()
    return f"{row[0]}:{row[1]}" if row else None


def import_legacy(legacy_path: str | Path, app, market, cfg) -> dict:
    """Returns counts per table. `app`/`market` are writer connections."""
    from market_platform.persistence.runs import record_config, strategy_version
    p = Path(legacy_path)
    if not p.exists():
        return {"skipped": f"{p} not found"}
    legacy = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    legacy.row_factory = sqlite3.Row
    have = _tables(legacy)
    out: dict[str, int] = {}
    now = datetime.now().isoformat(timespec="seconds")

    if "ob_series_1m" in have:
        rows = []
        for r in legacy.execute("SELECT * FROM ob_series_1m"):
            key = "NSE:NIFTY 50" if r["series"] == "NIFTY_SPOT" else f"NFO:{r['contract'] or 'NIFTY_FUT1'}"
            rows.append((key, r["ts"], r["open"], r["high"], r["low"], r["close"], r["volume"],
                         r["oi"], None, "legacy"))
        market.executemany("INSERT OR IGNORE INTO bars_1m VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        out["bars_1m(ob_series)"] = len(rows)

    if "kite_candles_1m" in have and "kite_instruments" in have:
        keys: dict[int, str | None] = {}
        rows = []
        for r in legacy.execute("SELECT * FROM kite_candles_1m"):
            tok = r["token"]
            if tok not in keys:
                keys[tok] = _instrument_key_for_token(legacy, tok)
            if keys[tok] is None:
                continue
            rows.append((keys[tok], r["ts"], r["open"], r["high"], r["low"], r["close"],
                         r["volume"], r["oi"], r["n_ticks"], "legacy"))
        market.executemany("INSERT OR IGNORE INTO bars_1m VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        out["bars_1m(kite_candles)"] = len(rows)

    if "option_quotes" in have:
        cols = [c[1] for c in legacy.execute("PRAGMA table_info(option_quotes)")]
        sel = [c for c in ("exchange", "tradingsymbol", "captured_at", "exchange_ts", "spot", "ltp",
                           "bid", "bid_qty", "ask", "ask_qty", "depth_json", "volume", "oi", "iv",
                           "reason", "expiry", "strike", "option_type") if c in cols]
        rows = [tuple(r) for r in legacy.execute(f"SELECT {', '.join(sel)} FROM option_quotes")]
        market.executemany(f"INSERT OR IGNORE INTO option_quotes ({', '.join(sel)}) VALUES "
                           f"({', '.join('?' * len(sel))})", rows)
        out["option_quotes"] = len(rows)

    if "option_candles_1m" in have:
        cols = [c[1] for c in legacy.execute("PRAGMA table_info(option_candles_1m)")]
        sel = [c for c in ("exchange", "tradingsymbol", "ts", "open", "high", "low", "close",
                           "volume", "oi", "source", "expiry", "strike", "option_type") if c in cols]
        rows = [tuple(r) for r in legacy.execute(f"SELECT {', '.join(sel)} FROM option_candles_1m")]
        market.executemany(f"INSERT OR IGNORE INTO option_candles_1m ({', '.join(sel)}) VALUES "
                           f"({', '.join('?' * len(sel))})", rows)
        out["option_candles_1m"] = len(rows)
    market.commit()

    if "ob_zones" in have or "ob_signals" in have:
        record_config(app, cfg, note="legacy import")
        app.execute("INSERT OR IGNORE INTO runs (run_id, kind, config_hash, strategy_version, "
                     "universe_snapshot, data_version, started_at, ended_at, status, notes) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (LEGACY_RUN, "import", cfg.hash, strategy_version(), "legacy-nifty",
                      "legacy", now, now, "completed", f"imported from {p}"))

    if "ob_zones" in have:
        rows = []
        for r in legacy.execute("SELECT * FROM ob_zones"):
            feats = {k: r[k] for k in ("broken_swing", "leg_origin", "atr_at_bos", "disp_body_atr",
                                       "disp_range_atr", "rvol", "fvg_low", "fvg_high", "swept_level")}
            rows.append((r["zone_id"], "NSE:NIFTY 50", r["timeframe"], r["direction"], r["kind"],
                         r["source_bar_ts"], r["bos_bar_ts"], r["first_eligible_ts"], r["zone_low"],
                         r["zone_high"], json.dumps(feats), r["status"], r["closed_ts"],
                         r["close_reason"], r["params_hash"], LEGACY_RUN))
        app.executemany("INSERT OR IGNORE INTO zones VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        out["zones"] = len(rows)

    if "ob_signals" in have:
        rows = []
        zones = {r[0]: (r[1], r[2]) for r in app.execute("SELECT zone_id, zone_low, zone_high FROM zones")}
        for r in legacy.execute("SELECT * FROM ob_signals WHERE mode='live'"):
            zl, zh = zones.get(r["zone_id"], (None, None))
            rows.append((
                r["signal_id"], LEGACY_RUN, r["direction"], "NSE:NIFTY 50", "NIFTY", r["direction"],
                "nifty-ob-v1", "order_block", "15m", r["horizon"], r["zone_id"], zl, zh,
                r["trigger_ts"], r["trigger_ts"], r["trigger_ts"][:10], r["u_entry"], r["u_stop"],
                r["u_stop"], json.dumps([r["u_target"]]), r["u_rr"], r["score"], r["score_json"],
                r["p_win_calibrated"], r["gates_json"], "{}", "legacy", "{}", None,
                json.dumps({"structure": r["structure"], "legs": r["legs_json"]}), None,
                _DECISION_TO_STATUS.get(r["decision"], "REJECTED"), "[]",
                json.dumps([x for x in (r["blocked_reasons"] or "").split("; ") if x]),
                cfg.hash, "legacy-nifty", now))
        app.executemany(f"INSERT OR IGNORE INTO signals VALUES ({', '.join('?' * 37)})", rows)
        out["signals"] = len(rows)
    app.commit()
    legacy.close()
    return out
