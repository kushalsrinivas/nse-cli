"""Daily paper-validation check (Phase 9).

    python platform_cli.py daily-check [--date D] [--run RUN_ID]

After each live paper session:

1. Data quality for the day across the universe (candles/quality.py) — the
   blocking rules of the quality gate apply.
2. Coverage: instruments with bars / planned, minutes per instrument.
3. Reconciliation: the day is replayed through the same modules (with the
   same structure warm-up) and its signals are compared with the live
   run's — ids and pre-risk statuses. Differences come from data that
   differed between the live feed and the stored bars (e.g. repaired
   minutes) and are listed, not hidden. The unexplained share must stay
   below `shadow.max_unexplained_signal_mismatch` (conf/promotion.toml).
4. Health: degraded/down health events of the day, writer errors.
5. Portfolio: trades, net P&L, the four counts.

Writes reports/daily/<date>.json; exit code 3 when quality blocks or the
reconciliation mismatch exceeds the limit.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

PRE_RISK = {"QUALIFIED", "APPROVED", "EXECUTED", "REJECTED", "WATCH", "SUPPRESSED", "NOT_EXECUTABLE"}


def _pre_risk_status(app, run_id: str, signal_id: str, status: str) -> str:
    """Status as the scorer left it (before the governor/executor changed it)."""
    row = app.execute("SELECT status FROM signal_status_history WHERE run_id=? AND signal_id=? "
                      "ORDER BY ts LIMIT 1", (run_id, signal_id)).fetchone()
    return row[0] if row else status


def signals_of(app, run_id: str, day: str) -> dict[str, str]:
    out = {}
    for sid, st in app.execute("SELECT signal_id, status FROM signals WHERE run_id=? AND session=?",
                               (run_id, day)):
        out[sid] = _pre_risk_status(app, run_id, sid, st)
    return out


def reconcile(app, live_run: str, replay_run: str, day: str) -> dict:
    a, b = signals_of(app, live_run, day), signals_of(app, replay_run, day)
    only_live = sorted(set(a) - set(b))
    only_replay = sorted(set(b) - set(a))
    status_diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    total = max(len(set(a) | set(b)), 1)
    unexplained = (len(only_live) + len(only_replay) + len(status_diff)) / total
    return {"live": len(a), "replay": len(b), "matched": len(set(a) & set(b)),
            "only_live": only_live[:50], "only_replay": only_replay[:50],
            "status_differs": status_diff[:50], "mismatch_share": round(unexplained, 4)}


def run_daily(cfg, dbs, *, day: date, live_run: str | None, instruments: list[dict], store=None,
              calendar=None, root: Path | None = None) -> tuple[dict, int]:
    from market_platform.candles import quality
    from market_platform.research.replay import Replay, ReplaySpec
    from market_platform.research.report import load_promotion
    app, market = dbs.app, dbs.market
    ds = day.isoformat()
    rep, results = quality.run(market, instruments, day, day, calendar=calendar, app_conn=app,
                               jump_pct=cfg.data.jump_alert_pct)
    cov = {k: r["worst_coverage"] for k, r in results.items()}
    out: dict = {"date": ds, "generated_at": datetime.now().isoformat(timespec="seconds"),
                 "quality": rep.to_dict(),
                 "coverage": {"instruments": len(cov),
                              "with_bars": sum(1 for r in results.values() if r["bars"]),
                              "below_98pct": sorted(k for k, v in cov.items() if v is not None
                                                    and v < 0.98)[:50]}}
    code = rep.exit_code
    if live_run:
        prev = [r[0] for r in market.execute(
            "SELECT DISTINCT substr(ts,1,10) d FROM bars_1m WHERE ts<? ORDER BY d DESC LIMIT ?",
            (f"{ds} 00:00", cfg.data.warmup_sessions))]
        frm = date.fromisoformat(prev[-1]) if prev else day
        # same strategy version as the live run, or the deterministic ids differ
        sv = app.execute("SELECT strategy_version FROM runs WHERE run_id=?", (live_run,)).fetchone()
        rp = Replay(cfg, dbs, instruments, store=store, calendar=calendar,
                    universe_snapshot=f"reconcile:{live_run}",
                    strategy_version=sv[0] if sv else None)
        res = rp.run(ReplaySpec(frm, day, label=f"reconcile {ds} vs {live_run}"))
        rec = reconcile(app, live_run, res.run_id, ds)
        rec["replay_run"] = res.run_id
        limit = load_promotion()["shadow"]["max_unexplained_signal_mismatch"]
        rec["limit"] = limit
        rec["ok"] = rec["mismatch_share"] <= limit
        out["reconcile"] = rec
        if not rec["ok"]:
            code = max(code, 3)
        pos = app.execute("SELECT COUNT(*), COALESCE(SUM(net_pnl),0), COALESCE(SUM(charges),0) FROM "
                          "positions WHERE run_id=? AND substr(closed_at,1,10)=?", (live_run, ds)).fetchone()
        out["portfolio"] = {"closed_today": pos[0], "net_pnl": round(pos[1], 2),
                            "charges": round(pos[2], 2)}
    health = {}
    for comp, state, n in app.execute(
            "SELECT component, state, COUNT(*) FROM health_events WHERE substr(ts,1,10)=? AND state IN "
            "('degraded','down','DEGRADED','DOWN') GROUP BY component, state", (ds,)):
        health[f"{comp}:{state}"] = n
    out["health_incidents"] = health
    out["exit_code"] = code
    if root is not None:
        d = root / cfg.paths.reports_dir / "daily"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{ds}.json").write_text(json.dumps(out, indent=2, default=str))
    return out, code
