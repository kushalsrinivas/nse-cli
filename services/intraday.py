"""Intraday confluence workflow shared by CLI entry points.

One place pulls the required 5m timeframe (Kite-first, Yahoo fallback),
evaluates Setups A/B/C and optionally journals the run — so `model_cli
confluence`, `main.py --confluence` and future callers can never disagree
on data. No console output; callers render.
"""

from __future__ import annotations

import time as _time
from datetime import datetime
from datetime import time as dtime
from zoneinfo import ZoneInfo

SUPPORTED_TIMEFRAMES = ("5m",)

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)
EXIT_AFTER_MINUTES = 5  # final bar + buffer, then stop


def run_confluence(*, source: str = "auto", timeframe: str = "5m",
                   journal=False, events: list[str] | None = None,
                   chain=None):
    """Evaluate A/B/C on the required timeframe; journal only if asked.

    `journal`: False (dry-run), True (shared journal), or a journal object.
    Raises ValueError for unsupported timeframes — the engine derives 15m
    internally from 5m bars, so 5m is currently the only valid input.
    """
    from data import source as datasrc
    from model.confluence.engine import build_confluence_report

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"timeframe must be one of {SUPPORTED_TIMEFRAMES}, got {timeframe!r} "
            f"(setups derive 15m internally from 5m bars)")
    if chain is None:
        try:
            chain = datasrc.get_nifty_chain(source=source)
        except Exception:
            chain = None  # setups degrade to no-contract evaluation
    return build_confluence_report(chain=chain, events=events or None,
                                   journal=journal, timeframe=timeframe)


def session_state(now: datetime | None = None) -> str:
    """Market session state in IST: 'pre' | 'open' | 'closed' | 'weekend'."""
    now = now or datetime.now(IST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    now_ist = now.astimezone(IST)
    if now_ist.weekday() >= 5:
        return "weekend"
    t = now_ist.time()
    if t < SESSION_OPEN:
        return "pre"
    if t <= SESSION_CLOSE:
        return "open"
    return "closed"


def run_confluence_live(*, source: str = "auto", timeframe: str = "5m",
                        journal=True, events: list[str] | None = None,
                        poll_secs: int = 60, until: str = "15:35",
                        fetch_frame=None, evaluate=None, clock=None,
                        sleep=None, on_report=None, on_error=None) -> dict:
    """Evaluate A/B/C all session on every new 5m bar; journal each run.

    fetch_frame/evaluate/clock/sleep are injectable for tests; defaults hit
    live data (fresh bars, journal shared). Returns a summary dict. Never
    raises on data errors (reported via on_error, counted); KeyboardInterrupt
    stops cleanly with a partial summary. Restarting mid-bar may re-journal
    that one bar (dedup is in-memory only).
    """
    from model.confluence.engine import build_confluence_report
    from model.intraday_sr import fetch_intraday

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"timeframe must be one of {SUPPORTED_TIMEFRAMES}, got {timeframe!r}")

    clock = clock or (lambda: datetime.now(IST))
    sleep = sleep or _time.sleep
    fetch_frame = fetch_frame or (lambda: fetch_intraday(
        period="5d", interval="5m", use_cache=False))
    if evaluate is None:
        def evaluate(df):
            from data import source as datasrc
            try:
                chain = datasrc.get_nifty_chain(source=source)
            except Exception:
                chain = None
            return build_confluence_report(
                chain=chain, events=events or None, journal=journal,
                timeframe=timeframe, df_5m=df, use_cache=False)

    summary = {"evals": 0, "journaled": 0, "errors": 0,
               "status": "started", "interrupted": False}
    state = session_state(clock())
    if state == "weekend":
        summary["status"] = "market closed (weekend)"
        return summary

    exit_h, exit_m = (int(x) for x in until.split(":"))
    last_bar = None
    try:
        while True:
            now = clock()
            if now.tzinfo is None:
                now = now.replace(tzinfo=IST)
            now_ist = now.astimezone(IST)
            if (now_ist.time() >= dtime(exit_h, exit_m)
                    or now_ist.weekday() >= 5):
                break
            if session_state(now_ist) == "pre":
                sleep(poll_secs)
                continue
            try:
                df = fetch_frame()
                bar_ts = df.index[-1] if len(df) else None
                if bar_ts is not None and bar_ts != last_bar:
                    last_bar = bar_ts
                    report = evaluate(df)
                    summary["evals"] += 1
                    if journal is not False and not getattr(report, "error", None):
                        summary["journaled"] += 1
                    if on_report is not None:
                        on_report(report, journal is not False)
            except Exception as exc:  # per-poll failure must not kill the day
                summary["errors"] += 1
                if on_error is not None:
                    on_error(exc)
            sleep(poll_secs)
    except KeyboardInterrupt:
        summary["interrupted"] = True
    summary["status"] = "done"
    return summary
