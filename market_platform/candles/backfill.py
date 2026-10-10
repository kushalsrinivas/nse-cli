"""Universe backfill and gap repair from Kite historical data.

* `backfill(...)` walks every instrument from its watermark (or
  `days` back) to now in ≤60-day minute windows, then pulls day candles.
  Watermarks (`backfill_watermarks`) make it resumable: stop it at any
  point and the next run continues where it left off. `max_requests`
  bounds one run; the REST client enforces 3 req/s.
* `find_gaps(...)` lists missing minutes of a session against the
  expected 09:15–15:29 grid (the calendar decides which days are sessions).
* `repair(...)` re-fetches a window for instruments with gaps and upserts
  it as source 'repair' (REST is the exchange's official record and
  replaces WS bars for the same minute).

Kite does not serve expired derivative contracts: futures/options history
must be recorded while they are live (see data/kite/legs.py); this module
targets spots, equities and live contracts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from data.kite.backfill import minute_chunks, to_ist_minute
from market_platform.candles.service import UPSERT_HIST

log = logging.getLogger(__name__)

SESSION_MINUTES = 375                    # 09:15 … 15:29


@dataclass
class BackfillReport:
    instruments: int = 0
    requests: int = 0
    bars_1m: int = 0
    bars_1d: int = 0
    complete: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)       # budget ran out
    errors: list[str] = field(default_factory=list)
    stopped_on_budget: bool = False


def expected_minutes(day: date) -> list[str]:
    start = datetime(day.year, day.month, day.day, 9, 15)
    return [(start + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M")
            for i in range(SESSION_MINUTES)]


def _rows(key: str, records: list[dict], source: str) -> list[tuple]:
    out = []
    for r in records:
        try:
            out.append((key, to_ist_minute(r["date"]), float(r["open"]), float(r["high"]),
                        float(r["low"]), float(r["close"]), int(r.get("volume") or 0),
                        r.get("oi"), None, source))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _watermark(market_conn, key: str, tf: str = "1m") -> str | None:
    row = market_conn.execute("SELECT last_ts FROM backfill_watermarks WHERE instrument_key=? "
                              "AND timeframe=?", (key, tf)).fetchone()
    return row[0] if row else None


def _set_watermark(market_conn, key: str, ts: str, source: str, tf: str = "1m") -> None:
    market_conn.execute(
        "INSERT INTO backfill_watermarks (instrument_key, timeframe, last_ts, source, updated_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT (instrument_key, timeframe) DO UPDATE SET "
        "last_ts=MAX(last_ts, excluded.last_ts), source=excluded.source, "
        "updated_at=excluded.updated_at",
        (key, tf, ts, source, datetime.now().isoformat(timespec="seconds")))


def backfill(rest, market_conn, instruments: list[dict], *, days: int, now: datetime | None = None,
             max_requests: int | None = None, daily: bool = True, progress=None) -> BackfillReport:
    """instruments: dicts with instrument_key and token (universe rows)."""
    now = (now or datetime.now()).replace(second=0, microsecond=0)
    rep = BackfillReport()
    floor = now - timedelta(days=days)
    for n, inst in enumerate(i for i in instruments if i.get("token")):
        key, tok = inst["instrument_key"], int(inst["token"])
        rep.instruments += 1
        wm = _watermark(market_conn, key)
        start = max(floor, datetime.strptime(wm, "%Y-%m-%d %H:%M") + timedelta(minutes=1)) \
            if wm else floor
        done = True
        for a, b in minute_chunks(start, now):
            if max_requests is not None and rep.requests >= max_requests:
                done = False
                rep.stopped_on_budget = True
                break
            rep.requests += 1
            try:
                recs = rest.historical(tok, "minute", a, b, oi=False) or []
            except Exception as exc:
                rep.errors.append(f"{key} {a:%Y-%m-%d}→{b:%Y-%m-%d}: {exc}")
                done = False
                break
            rows = _rows(key, recs, "kite_hist")
            if rows:
                market_conn.executemany(UPSERT_HIST, rows)
                rep.bars_1m += len(rows)
            # the window is covered even when empty (holidays): advance the watermark
            _set_watermark(market_conn, key, b.strftime("%Y-%m-%d %H:%M"), "kite_hist")
            market_conn.commit()
        if daily and done and (max_requests is None or rep.requests < max_requests):
            rep.requests += 1
            try:
                recs = rest.historical(tok, "day", floor, now, oi=False) or []
                rows = [(key, to_ist_minute(r["date"])[:10], float(r["open"]), float(r["high"]),
                         float(r["low"]), float(r["close"]), int(r.get("volume") or 0),
                         r.get("oi"), 0, "kite_hist") for r in recs]
                market_conn.executemany(
                    "INSERT INTO bars_1d (instrument_key, date, open, high, low, close, volume, oi, "
                    "adjusted, source) VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT (instrument_key, date) "
                    "DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low, "
                    "close=excluded.close, volume=excluded.volume, oi=excluded.oi, "
                    "source=excluded.source", rows)
                if rows:
                    _set_watermark(market_conn, key, rows[-1][1], "kite_hist", tf="1d")
                market_conn.commit()
                rep.bars_1d += len(rows)
            except Exception as exc:
                rep.errors.append(f"{key} day: {exc}")
        (rep.complete if done else rep.partial).append(key)
        if progress:
            progress(n + 1, key, rep)
        if rep.stopped_on_budget:
            remaining = [i["instrument_key"] for i in instruments[n + 1:] if i.get("token")]
            rep.partial.extend(remaining)
            break
    return rep


def find_gaps(market_conn, key: str, day: date) -> list[str]:
    have = {r[0] for r in market_conn.execute(
        "SELECT ts FROM bars_1m WHERE instrument_key=? AND ts>=? AND ts<?",
        (key, f"{day.isoformat()} 09:15", f"{day.isoformat()} 15:30"))}
    return [m for m in expected_minutes(day) if m not in have]


def repair(rest, market_conn, instruments: list[dict], frm: datetime, to: datetime, *,
           min_missing: int = 1, calendar=None) -> dict:
    """Re-fetch [frm, to] for instruments with ≥ min_missing missing minutes
    in any session of the window. Returns {key: bars_written}."""
    out: dict[str, int] = {}
    days = []
    d = frm.date()
    while d <= to.date():
        if calendar.is_trading_day(d) if calendar else d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    lo, hi = frm.strftime("%Y-%m-%d %H:%M"), to.strftime("%Y-%m-%d %H:%M")
    for inst in instruments:
        if not inst.get("token"):
            continue
        key = inst["instrument_key"]
        missing = [m for day in days for m in find_gaps(market_conn, key, day) if lo <= m <= hi]
        if len(missing) < min_missing:
            continue
        n = 0
        for a, b in minute_chunks(frm, to) or [(frm, to)]:      # frm == to: one minute
            try:
                recs = rest.historical(int(inst["token"]), "minute", a, b, oi=False) or []
            except Exception as exc:
                log.warning("repair %s failed: %s", key, exc)
                continue
            rows = _rows(key, recs, "repair")
            market_conn.executemany(UPSERT_HIST, rows)
            n += len(rows)
        market_conn.commit()
        out[key] = n
    return out
