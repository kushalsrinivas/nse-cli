"""REST minute backfill into the market archive.

Kite serves minute history in requests of at most 60 days, at 3 requests/s
(enforced by `KiteRest`). Every loader here is incremental: it resumes
from the newest stored bar, so a daily cron run costs a request or two.

What can and cannot be recovered:

- NIFTY spot: years of minute history. Fully recoverable.
- NIFTY front future: only contracts still listed today. Kite does not
  serve expired contracts (minute `continuous` data does not exist), so
  FUT1 history reaches back roughly one contract cycle and no further.
  Days whose front contract has expired are counted, not silently skipped.
- Options: only unexpired contracts, from their listing date. Every series
  that expires before it is backfilled is lost for good.

Roll rule (`fut1_for_date`): a contract is FUT1 through the close of the
second weekday before its expiry; the next contract takes over on the
following weekday. Exchange holidays are not modelled — a holiday inside
the T-2 window shifts the roll by one session, which only matters for the
volume baseline right at the roll, and rvol is NULLed across rolls anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from data.kite.archive import (
    SERIES_FUT1,
    SERIES_SPOT,
    MarketArchive,
    OptionBar,
    SeriesBar,
)

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
MAX_MINUTE_DAYS = 60          # Kite's per-request cap for minute candles
ROLL_WEEKDAYS_BEFORE = 2


@dataclass
class BackfillResult:
    target: str
    requests: int = 0
    bars: int = 0
    frm: str | None = None
    to: str | None = None
    unavailable_days: int = 0     # front contract expired → Kite won't serve
    errors: list[str] = field(default_factory=list)


def to_ist_minute(value) -> str:
    """Kite record timestamp (aware IST, or naive) → 'YYYY-MM-DD HH:MM'."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(IST).tz_localize(None)
    return ts.strftime("%Y-%m-%d %H:%M")


def minute_chunks(frm: datetime, to: datetime,
                  max_days: int = MAX_MINUTE_DAYS) -> list[tuple[datetime, datetime]]:
    """Split [frm, to] into request windows no longer than `max_days`."""
    out = []
    cur = frm
    while cur < to:
        end = min(cur + timedelta(days=max_days) - timedelta(minutes=1), to)
        out.append((cur, end))
        cur = end + timedelta(minutes=1)
    return out


def roll_date(expiry: str) -> date:
    """Last session a contract is FUT1: T-2 weekdays before expiry."""
    d = datetime.strptime(expiry, "%Y-%m-%d").date()
    n = 0
    while n < ROLL_WEEKDAYS_BEFORE:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d


def fut1_periods(futs) -> list[tuple[object, date, date]]:
    """(contract, first_day, last_day) FUT1 windows, nearest expiry first.

    `futs` are InstrumentRow-like with `.expiry`. A contract's window starts
    the day after the previous contract's roll. For the earliest contract
    the previous one is unknown, so its window is taken as one calendar
    month back from its roll (NIFTY futures are monthly).
    """
    rows = sorted((f for f in futs if f.expiry), key=lambda f: f.expiry)
    out = []
    prev_roll: date | None = None
    for f in rows:
        last = roll_date(f.expiry)
        if prev_roll is None:
            first = (pd.Timestamp(last) - pd.DateOffset(months=1)).date() + timedelta(days=1)
        else:
            first = prev_roll + timedelta(days=1)
        out.append((f, first, last))
        prev_roll = last
    return out


def fut1_for_date(futs, d: date):
    """The FUT1 contract on session date `d`, or None if unknown."""
    for f, first, last in fut1_periods(futs):
        if first <= d <= last:
            return f
    return None


def _start_from(last_ts: str | None, frm: datetime) -> datetime:
    if not last_ts:
        return frm
    resume = datetime.strptime(last_ts, "%Y-%m-%d %H:%M") + timedelta(minutes=1)
    return max(frm, resume)


def _fetch(rest, token: int, a: datetime, b: datetime, oi: bool,
           res: BackfillResult) -> list[dict]:
    res.requests += 1
    try:
        return rest.historical(token, "minute", a, b, oi=oi) or []
    except Exception as exc:     # degrade-don't-crash, but say so
        res.errors.append(f"{a:%Y-%m-%d}→{b:%Y-%m-%d}: {exc}")
        log.warning("backfill token %s %s→%s failed: %s", token, a, b, exc)
        return []


def _note_span(res: BackfillResult, ts: str) -> None:
    res.frm = ts if res.frm is None or ts < res.frm else res.frm
    res.to = ts if res.to is None or ts > res.to else res.to


def backfill_spot(rest, archive: MarketArchive, spot_token: int, *,
                  days: int, now: datetime | None = None) -> BackfillResult:
    """NIFTY 50 spot minute bars (volume stays NULL: the index has none)."""
    now = now or datetime.now()
    res = BackfillResult(SERIES_SPOT)
    _lo, last = archive.series_bounds(SERIES_SPOT)
    start = _start_from(last, now - timedelta(days=days))
    for a, b in minute_chunks(start, now):
        bars = []
        for r in _fetch(rest, spot_token, a, b, False, res):
            try:
                ts = to_ist_minute(r["date"])
                bars.append(SeriesBar(SERIES_SPOT, ts, float(r["open"]),
                                      float(r["high"]), float(r["low"]),
                                      float(r["close"]), None, None, "",
                                      "kite_hist"))
                _note_span(res, ts)
            except (KeyError, TypeError, ValueError):
                continue
        res.bars += archive.upsert_series(bars)
    return res


def backfill_fut1(rest, archive: MarketArchive, futs, *, days: int,
                  now: datetime | None = None) -> BackfillResult:
    """Front-future minute bars (volume + OI), tagged with the contract.

    `futs` should include expired rows still in the master
    (`futures_chain(..., include_expired=True)`) so the roll windows are
    right; windows whose contract has expired are counted as unavailable.
    """
    now = now or datetime.now()
    res = BackfillResult(SERIES_FUT1)
    _lo, last = archive.series_bounds(SERIES_FUT1)
    start = _start_from(last, now - timedelta(days=days))
    today = now.date()
    for f, first, last_day in fut1_periods(futs):
        lo = max(start, datetime.combine(first, datetime.min.time()))
        hi = min(now, datetime.combine(last_day, datetime.max.time()).replace(microsecond=0))
        if lo > hi:
            continue
        if f.expiry < today.isoformat():
            res.unavailable_days += sum(
                1 for i in range((hi.date() - lo.date()).days + 1)
                if (lo.date() + timedelta(days=i)).weekday() < 5)
            continue
        for a, b in minute_chunks(lo, hi):
            bars = []
            for r in _fetch(rest, f.instrument_token, a, b, True, res):
                try:
                    ts = to_ist_minute(r["date"])
                    oi = r.get("oi")
                    bars.append(SeriesBar(
                        SERIES_FUT1, ts, float(r["open"]), float(r["high"]),
                        float(r["low"]), float(r["close"]),
                        int(r.get("volume") or 0),
                        int(oi) if oi not in (None, "") else None,
                        f.tradingsymbol, "kite_hist"))
                    _note_span(res, ts)
                except (KeyError, TypeError, ValueError):
                    continue
            res.bars += archive.upsert_series(bars)
    return res


def backfill_options(rest, archive: MarketArchive, legs, *, days: int,
                     now: datetime | None = None) -> BackfillResult:
    """Minute bars for live option contracts, keyed by tradingsymbol.

    Rescues what Kite still serves for unexpired series. Contracts that
    have already expired are skipped (Kite returns nothing for them).
    """
    now = now or datetime.now()
    res = BackfillResult("OPTIONS")
    today = now.date().isoformat()
    for leg in legs:
        if leg.expiry and leg.expiry < today:
            continue
        start = _start_from(archive.last_option_ts(leg.tradingsymbol, leg.exchange),
                            now - timedelta(days=days))
        for a, b in minute_chunks(start, now):
            bars = []
            for r in _fetch(rest, leg.instrument_token, a, b, True, res):
                try:
                    ts = to_ist_minute(r["date"])
                    oi = r.get("oi")
                    bars.append(OptionBar(
                        leg.exchange, leg.tradingsymbol, ts, float(r["open"]),
                        float(r["high"]), float(r["low"]), float(r["close"]),
                        int(r.get("volume") or 0),
                        int(oi) if oi not in (None, "") else None, "kite_hist"))
                    _note_span(res, ts)
                except (KeyError, TypeError, ValueError):
                    continue
            res.bars += archive.upsert_option_bars(bars)
    return res
