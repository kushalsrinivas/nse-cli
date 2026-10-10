"""Underlying evaluation: signal quality, separate from how a trade was executed.

Every signal the scorer QUALIFIED (its first status, before the governor
touched it) is walked forward on the instrument's own 1m bars with the
validated NIFTY simulator (`model.order_blocks.backtest.simulate`):

    entry   open of the first 1m bar at/after the signal became available
            (overnight: the 15:20 bar, as in the NIFTY system)
    exits   stop / target / end of session (intraday) / next-session rules
            (overnight), a gap through the stop fills at the open
    costs   index: backtest.cost_points_index points; equity:
            backtest.cost_bps_equity of the entry price

The result is in R and needs only underlying candles, so it can be measured
over any history that has bars. It answers "do the order-block signals
predict the underlying?" (Gate A). It says nothing about option P&L; that is
the executed-options basis, which needs archived real quotes (report.py keeps
the two apart and labels each).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta

from model.order_blocks import backtest as bt
from model.order_blocks.types import Bar

TS = "%Y-%m-%d %H:%M"
BASIS = ("UNDERLYING — every pre-risk QUALIFIED signal walked on its own 1m bars with the NIFTY "
         "simulator; costs as points (index) / bps (equity); independent of route, sizing and "
         "the governor. Measures signal quality, not option profitability.")


def pre_risk_status(app_conn, run_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for sid, st in app_conn.execute(
            "SELECT signal_id, status FROM signal_status_history WHERE run_id=? ORDER BY ts, rowid",
            (run_id,)):
        out.setdefault(sid, st)
    return out


def _book(market_conn, key: str, frm: str, to: str) -> bt._Book:
    rows = market_conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars_1m WHERE instrument_key=? AND ts>=? "
        "AND ts<=? ORDER BY ts", (key, frm, to)).fetchall()
    return bt._Book([(Bar(datetime.strptime(r[0][:16], TS), "1m", r[1], r[2], r[3], r[4], r[5]), "")
                     for r in rows])


def underlying_trades(app_conn, market_conn, cfg, run_id: str, *,
                      statuses: tuple[str, ...] = ("QUALIFIED",),
                      instruments: dict[str, dict] | None = None) -> list[dict]:
    first = pre_risk_status(app_conn, run_id)
    rows = app_conn.execute(
        "SELECT signal_id, instrument_key, direction, horizon, detected_at, available_at, entry, stop, "
        "targets_json, score, zone_id, session FROM signals WHERE run_id=? ORDER BY detected_at",
        (run_id,)).fetchall()
    by_key: dict[str, list] = defaultdict(list)
    for r in rows:
        if first.get(r["signal_id"], "") in statuses:
            by_key[r["instrument_key"]].append(r)
    out = []
    for key, sigs in by_key.items():
        kind = ((instruments or {}).get(key) or {}).get("kind")
        if kind is None:
            kind = "index" if key in ("NSE:NIFTY 50", "BSE:SENSEX") or " " in key.split(":")[-1] \
                else "equity"
        frm = min(s["detected_at"] for s in sigs)[:10] + " 00:00"
        to_day = max(s["detected_at"] for s in sigs)[:10]
        book = _book(market_conn, key, frm, _plus_days(to_day, 6))   # room for overnight exits
        if not book.bars:
            continue
        for s in sigs:
            det = datetime.fromisoformat(s["detected_at"])
            entry_ts = det + (bt.OVERNIGHT_FILL_DELAY if s["horizon"] == "overnight" else
                              timedelta(0))
            t = bt.Trade("OB", s["horizon"], s["direction"], s["session"], det, entry_ts, s["entry"],
                         s["stop"], json.loads(s["targets_json"])[0], s["score"] or 0.0, s["zone_id"])
            t.available_at = datetime.fromisoformat(s["available_at"])
            cost = cfg.backtest.cost_points_index if kind == "index" else \
                s["entry"] * cfg.backtest.cost_bps_equity / 1e4
            done = bt.simulate(t, book, cost)
            if done is None:
                continue
            out.append({"signal_id": s["signal_id"], "instrument_key": key, "direction": t.direction,
                        "horizon": t.horizon, "session": s["session"], "status": "CLOSED",
                        "r_multiple": done.r_net, "r_gross": done.r_gross, "exit_reason": done.reason,
                        "gap_r": done.gap_r, "score": done.score, "entry_ts": done.entry_ts.isoformat(),
                        "exit_ts": done.exit_ts.isoformat() if done.exit_ts else None,
                        "entry": done.entry, "exit": done.exit, "basis": "underlying"})
    return out


def _plus_days(day: str, n: int) -> str:
    from datetime import date
    return (date.fromisoformat(day) + timedelta(days=n)).isoformat() + " 23:59"
