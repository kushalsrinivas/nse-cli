"""Per-instrument data-quality checks on market.db (extends services/data_quality).

For every instrument and every session in the window:

    coverage      missing minutes vs the 375-minute grid      <98% WARN, <90% CRITICAL
    ohlc          low ≤ open,close ≤ high and low > 0          any violation CRITICAL
    jump          |open − prev close| > data.jump_alert_pct     WARN, unless a corporate
                  (bar to bar, within and across sessions)     action is recorded that day
    holiday_bar   bars on a day the calendar marks closed      WARN
    out_of_session bars before 09:15 or after 15:30            WARN
    flat          ≥ 30 consecutive zero-volume equity minutes   INFO

Results go to `quality_events` and into a `QualityReport` (same JSON +
exit-code contract as the NIFTY gate). Per-instrument status is returned
so the signal engines can mark or skip an instrument (`data_quality` on
every signal): CRITICAL → the instrument is quarantined for the window.

The report blocks (exit code 3) when an index (context/benchmark) has a
critical issue, or more than `max_bad_frac` of equities do — one bad
stock must not stop the platform, but a broken feed must.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta

from market_platform.candles.backfill import SESSION_MINUTES
from services.data_quality import QualityReport

FLAT_RUN = 30


def _sessions(frm: date, to: date, calendar) -> list[date]:
    out, d = [], frm
    while d <= to:
        if calendar.is_trading_day(d) if calendar else d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def check_instrument(market_conn, key: str, frm: date, to: date, *, kind: str = "equity",
                     calendar=None, corporate_days: set[str] | None = None,
                     jump_pct: float = 8.0) -> dict:
    rows = market_conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars_1m WHERE instrument_key=? "
        "AND ts>=? AND ts<=? ORDER BY ts",
        (key, f"{frm.isoformat()} 00:00", f"{to.isoformat()} 23:59")).fetchall()
    sessions = _sessions(frm, to, calendar)
    session_set = {d.isoformat() for d in sessions}
    per_day: dict[str, int] = defaultdict(int)
    issues: list[tuple[str, str, str]] = []          # (kind, severity, detail)
    prev_close = None
    flat = 0
    for ts, o, h, lo, c, v in rows:
        day, hm = ts[:10], ts[11:16]
        if hm < "09:15" or hm > "15:30":
            issues.append(("out_of_session", "WARN", ts))
            continue
        if day not in session_set:
            issues.append(("holiday_bar", "WARN", ts))
        if "09:15" <= hm <= "15:29":
            per_day[day] += 1
        if not (lo > 0 and lo <= min(o, c) and h >= max(o, c)):
            issues.append(("ohlc", "CRITICAL", f"{ts} o={o} h={h} l={lo} c={c}"))
        if prev_close and abs(o - prev_close) / prev_close * 100 > jump_pct \
                and day not in (corporate_days or set()):
            issues.append(("jump", "WARN",
                           f"{ts} {prev_close}→{o} ({(o / prev_close - 1) * 100:+.1f}%)"))
        prev_close = c
        if kind == "equity":
            flat = flat + 1 if not v else 0
            if flat == FLAT_RUN:
                issues.append(("flat", "INFO", f"{FLAT_RUN} zero-volume minutes to {ts}"))
    worst = 1.0
    for d in sessions:
        cov = per_day.get(d.isoformat(), 0) / SESSION_MINUTES
        worst = min(worst, cov)
        if cov < 0.90:
            issues.append(("gap", "CRITICAL", f"{d} {cov:.1%} of minutes"))
        elif cov < 0.98:
            issues.append(("gap", "WARN", f"{d} {cov:.1%} of minutes"))
    status = "CRITICAL" if any(s == "CRITICAL" for _, s, _ in issues) else \
        "WARN" if any(s == "WARN" for _, s, _ in issues) else "OK"
    return {"instrument_key": key, "kind": kind, "bars": len(rows), "sessions": len(sessions),
            "worst_coverage": round(worst, 4) if sessions else None, "status": status,
            "issues": issues}


def run(market_conn, instruments: list[dict], frm: date, to: date, *, calendar=None,
        app_conn=None, jump_pct: float = 8.0, max_bad_frac: float = 0.05,
        record: bool = True, now: datetime | None = None) -> tuple[QualityReport, dict[str, dict]]:
    now = now or datetime.now()
    rep = QualityReport("platform", now.isoformat(timespec="seconds"),
                        [frm.isoformat(), to.isoformat()])
    corp: dict[str, set[str]] = defaultdict(set)
    if app_conn is not None:
        for _isin, sym, ex in app_conn.execute("SELECT isin, symbol, ex_date FROM corporate_actions"):
            corp[f"NSE:{sym}"].add(ex)
    results: dict[str, dict] = {}
    events = []
    for inst in instruments:
        key = inst["instrument_key"]
        r = check_instrument(market_conn, key, frm, to, kind=inst.get("kind", "equity"),
                             calendar=calendar, corporate_days=corp.get(key), jump_pct=jump_pct)
        results[key] = r
        for kind, sev, detail in r["issues"]:
            events.append((now.isoformat(timespec="seconds"), key, kind, sev.lower(), detail))
    if record and events:
        market_conn.executemany("INSERT INTO quality_events (ts, instrument_key, kind, severity, "
                                "detail) VALUES (?,?,?,?,?)", events)
        market_conn.commit()

    idx_bad = [k for k, r in results.items() if r["kind"] == "index" and r["status"] == "CRITICAL"]
    eq = [r for r in results.values() if r["kind"] == "equity"]
    eq_bad = [r["instrument_key"] for r in eq if r["status"] == "CRITICAL"]
    frac = len(eq_bad) / len(eq) if eq else 0.0
    rep.add("indices.critical", "CRITICAL", not idx_bad, len(idx_bad), ", ".join(idx_bad[:10]))
    rep.add("equities.critical_fraction", "CRITICAL", frac <= max_bad_frac, f"{frac:.1%}",
            f"{len(eq_bad)}/{len(eq)} quarantined: {', '.join(eq_bad[:15])}")
    no_bars = [r["instrument_key"] for r in results.values() if r["bars"] == 0]
    rep.add("instruments.without_bars", "WARN", not no_bars, len(no_bars), ", ".join(no_bars[:15]))
    warn = [r["instrument_key"] for r in results.values() if r["status"] == "WARN"]
    rep.add("instruments.warnings", "WARN", not warn, len(warn), ", ".join(warn[:15]))
    if calendar is not None:
        years = sorted({frm.year, to.year})
        unknown = [y for y in years if not calendar.known(y)]
        rep.add("calendar.imported", "WARN", not unknown, unknown or "ok",
                "holidays unverified; sessions assumed Mon–Fri" if unknown else "")
    return rep, results


def quarantined(results: dict[str, dict]) -> set[str]:
    return {k for k, r in results.items() if r["status"] == "CRITICAL"}
