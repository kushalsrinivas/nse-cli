"""Exchange trading calendar (fixes audit A12).

Holidays and special sessions come from files you import (the exchange
publishes a holiday list each year); nothing is invented here. For a year
with no imported rows the calendar falls back to "weekdays trade" and says
so: `known(year)` is False and the data-quality gate reports it as a
warning, because session counts, roll dates, next-open and DTE are only
approximate on that basis.

CSV format (header required):
    date,exchange,is_trading,open_time,close_time,note
    2026-01-26,NSE,0,,,Republic Day
    2026-11-08,NSE,1,18:00,19:00,Muhurat trading
`exchange` may be omitted (then the `--exchange` argument applies).
"""

from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta
from pathlib import Path

DEFAULT_OPEN = time(9, 15)
DEFAULT_CLOSE = time(15, 30)


class TradingCalendar:
    def __init__(self, conn=None) -> None:
        self.conn = conn
        self._rows: dict[tuple[str, str], dict] = {}
        self._years: dict[str, set[int]] = {}
        if conn is not None:
            self.reload()

    # -- data ----------------------------------------------------------------

    def reload(self) -> None:
        self._rows.clear()
        self._years.clear()
        for r in self.conn.execute("SELECT * FROM trading_calendar"):
            d = dict(r)
            self._rows[(d["exchange"], d["date"])] = d
            self._years.setdefault(d["exchange"], set()).add(int(d["date"][:4]))

    def import_csv(self, path: str | Path, *, exchange: str | None = None,
                   source: str | None = None) -> int:
        n = 0
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                ex = (row.get("exchange") or exchange or "").strip().upper()
                if not ex:
                    raise ValueError(f"{path}: row {row} has no exchange")
                d = datetime.strptime(row["date"].strip(), "%Y-%m-%d").date().isoformat()
                trading = str(row.get("is_trading", "0")).strip() in ("1", "true", "True", "yes")
                self.conn.execute(
                    "INSERT INTO trading_calendar (exchange, date, is_trading, open_time, close_time, "
                    "note, source) VALUES (?,?,?,?,?,?,?) ON CONFLICT (exchange, date) DO UPDATE SET "
                    "is_trading=excluded.is_trading, open_time=excluded.open_time, "
                    "close_time=excluded.close_time, note=excluded.note, source=excluded.source",
                    (ex, d, int(trading), (row.get("open_time") or None),
                     (row.get("close_time") or None), row.get("note", ""), source or str(path)))
                n += 1
        self.conn.commit()
        self.reload()
        return n

    def known(self, year: int, exchange: str = "NSE") -> bool:
        return year in self._years.get(exchange, set())

    # -- queries ---------------------------------------------------------------

    def is_trading_day(self, d: date, exchange: str = "NSE") -> bool:
        row = self._rows.get((exchange, d.isoformat()))
        if row is not None:
            return bool(row["is_trading"])
        return d.weekday() < 5

    def session_bounds(self, d: date, exchange: str = "NSE") -> tuple[datetime, datetime] | None:
        if not self.is_trading_day(d, exchange):
            return None
        row = self._rows.get((exchange, d.isoformat())) or {}
        o = time.fromisoformat(row["open_time"]) if row.get("open_time") else DEFAULT_OPEN
        c = time.fromisoformat(row["close_time"]) if row.get("close_time") else DEFAULT_CLOSE
        return datetime.combine(d, o), datetime.combine(d, c)

    def next_session(self, d: date, exchange: str = "NSE") -> date:
        x = d + timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(x, exchange):
                return x
            x += timedelta(days=1)
        raise RuntimeError(f"no trading day within 30 days after {d}")

    def previous_session(self, d: date, exchange: str = "NSE") -> date:
        x = d - timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(x, exchange):
                return x
            x -= timedelta(days=1)
        raise RuntimeError(f"no trading day within 30 days before {d}")

    def next_open(self, now: datetime, exchange: str = "NSE") -> datetime:
        """The next session open strictly after `now`."""
        d = now.date()
        b = self.session_bounds(d, exchange)
        if b and now < b[0]:
            return b[0]
        return self.session_bounds(self.next_session(d, exchange), exchange)[0]

    def sessions_between(self, start: date, end: date, exchange: str = "NSE") -> list[date]:
        out, d = [], start
        while d <= end:
            if self.is_trading_day(d, exchange):
                out.append(d)
            d += timedelta(days=1)
        return out

    def trading_days_until(self, start: date, end: date, exchange: str = "NSE") -> int:
        """Sessions after `start` up to and including `end` (DTE in sessions)."""
        if end <= start:
            return 0
        return len(self.sessions_between(start + timedelta(days=1), end, exchange))

    def coverage(self, years: list[int], exchange: str = "NSE") -> dict[int, bool]:
        return {y: self.known(y, exchange) for y in years}


_shared: TradingCalendar | None = None


def weekday_calendar() -> TradingCalendar:
    """Calendar with no imported rows (weekday fallback) — for code paths that
    have no database yet. Never treat it as verified."""
    global _shared
    if _shared is None:
        _shared = TradingCalendar(None)
    return _shared
