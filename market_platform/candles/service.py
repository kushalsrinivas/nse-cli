"""Candle Service: ticks → settled 1m bars → derived 5m/15m/60m bars.

Completion contract (plan §6.3)
-------------------------------
* A 1m bar [t, t+1m) is emitted once the clock passes t+1m+grace (or a
  tick for a later minute arrives). Its `available_at` is the emission
  time — never earlier than t+1m. Nothing downstream ever sees the
  in-flight minute.
* Higher timeframes are derived from the settled 1m bars with
  `model.order_blocks.bars.BarBuilder`, the *same* function replay uses
  (`replay()` below), so live and research cannot diverge. An HTF bar is
  available when the 1m bar completing it is.
* 1m bars are written to market.db `bars_1m` (source 'kite_ws') through
  the single writer. A WS bar never overwrites a bar already there: REST
  history ('kite_hist'/'repair') is the exchange's official record and wins.
* A checkpoint (`checkpoints['candles']`) records the last settled minute;
  on restart the gap from it to now is repaired from REST.

Volume baseline: Kite sends *cumulative* day volume, so a bar's volume is
the difference from the previous tick. The first bar per instrument after
a (re)start has no baseline: its volume is stored as NULL (unknown, not
0) and the bar is queued in `repair_queue` for the REST repair, which
replaces it with the exchange's minute record.

Bid/ask from `full` packets are sampled per instrument for the median
spread used by eligibility and liquidity checks.
"""

from __future__ import annotations

import statistics
from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from data.kite.aggregator import TickAggregator
from market_platform.marketdata.events import CandleClosed, Tick
from market_platform.persistence.writer import Write
from model.order_blocks.bars import BarBuilder
from model.order_blocks.types import Bar

IST = ZoneInfo("Asia/Kolkata")
HTF = ("5m", "15m", "60m")
INSERT_WS = ("INSERT OR IGNORE INTO bars_1m (instrument_key, ts, open, high, low, close, volume, "
             "oi, n_ticks, source) VALUES (?,?,?,?,?,?,?,?,?,?)")
UPSERT_HIST = ("INSERT INTO bars_1m (instrument_key, ts, open, high, low, close, volume, oi, "
               "n_ticks, source) VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT (instrument_key, ts) "
               "DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low, "
               "close=excluded.close, volume=excluded.volume, oi=excluded.oi, "
               "source=excluded.source")
TS_FMT = "%Y-%m-%d %H:%M"


def bar_from_row(r) -> Bar:
    return Bar(datetime.strptime(r[0][:16], TS_FMT), "1m", r[1], r[2], r[3], r[4], r[5])


class CandleService:
    def __init__(self, *, writer=None, bus=None, timeframes=("1m", "5m", "15m", "60m"),
                 grace_sec: float = 5.0, spread_samples: int = 300) -> None:
        self.writer = writer               # AsyncWriter-like (submit / submit_nowait) or None
        self.bus = bus
        self.tfs = tuple(tf for tf in timeframes if tf in HTF)
        self.grace = timedelta(seconds=grace_sec)
        self.agg = TickAggregator(late_grace_sec=grace_sec)
        self.keys: dict[int, str] = {}
        self.builders: dict[str, dict[str, BarBuilder]] = {}
        self.spreads: dict[str, deque] = {}
        self._spread_n = spread_samples
        self.last_minute: dict[str, str] = {}
        self._based: set[int] = set()
        self._unbased: set[tuple[int, str]] = set()
        self.repair_queue: list[tuple[str, str]] = []      # (instrument_key, minute)
        self.counters = {"ticks": 0, "bars_1m": 0, "bars_htf": 0, "write_backpressure": 0}

    # -- input ---------------------------------------------------------------------

    async def on_ticks(self, ticks: list[Tick], raw=None) -> list[CandleClosed]:
        out: list[CandleClosed] = []
        for t in ticks:
            self.keys[t.token] = t.instrument_key
            if t.token not in self._based and t.volume is not None:
                self._based.add(t.token)
                ts = t.exch_ts or t.recv_ts
                self._unbased.add((t.token, ts.astimezone(IST).strftime(TS_FMT)))
            sp = t.spread_bps
            if sp is not None:
                self.spreads.setdefault(t.instrument_key, deque(maxlen=self._spread_n)).append(sp)
            done = self.agg.on_tick({"token": t.token, "ltp": t.ltp, "exchange_ts": t.exch_ts,
                                     "volume": t.volume, "oi": t.oi}, arrived_at=t.recv_ts)
            self.counters["ticks"] += 1
            if done:
                out.extend(await self._emit(done, t.recv_ts))
        return out

    async def on_clock(self, now: datetime | None = None) -> list[CandleClosed]:
        """Call about once a second: settles bins whose minute is over."""
        now = now or datetime.now(tz=IST)
        done = self.agg.settle_before(now - self.grace)
        return await self._emit(done, now) if done else []

    async def flush(self, now: datetime | None = None) -> list[CandleClosed]:
        now = now or datetime.now(tz=IST)
        return await self._emit(self.agg.flush(), now)

    # -- emission ------------------------------------------------------------------

    async def _emit(self, candles, now: datetime) -> list[CandleClosed]:
        avail = now.astimezone(IST).replace(tzinfo=None)
        out: list[CandleClosed] = []
        rows = []
        for c in sorted(candles, key=lambda c: c.ts):
            key = self.keys.get(c.token)
            if key is None:
                continue
            vol = c.volume
            if (c.token, c.ts) in self._unbased:
                self._unbased.discard((c.token, c.ts))
                vol = None
                self.repair_queue.append((key, c.ts))
            bar = Bar(datetime.strptime(c.ts, TS_FMT), "1m", c.open, c.high, c.low, c.close, vol)
            rows.append((key, c.ts, c.open, c.high, c.low, c.close, vol, c.oi, c.n_ticks,
                         "kite_ws"))
            avail_bar = max(avail, bar.end)
            out.append(CandleClosed(key, "1m", bar, avail_bar, "kite_ws"))
            self.counters["bars_1m"] += 1
            if c.ts > self.last_minute.get(key, ""):
                self.last_minute[key] = c.ts
            for htf in self._feed(key, bar):
                out.append(CandleClosed(key, htf.tf, htf, max(avail_bar, htf.end), "kite_ws"))
                self.counters["bars_htf"] += 1
        if rows and self.writer is not None:
            w = Write(INSERT_WS, rows, many=True)
            if hasattr(self.writer, "submit"):
                if self.writer.queue.full():
                    self.counters["write_backpressure"] += 1
                await self.writer.submit(w)
            else:
                self.writer.write(w)
        if self.bus is not None:
            for ev in out:
                await self.bus.publish(f"candles.{ev.tf}", ev)
        return out

    def _feed(self, key: str, bar: Bar) -> list[Bar]:
        b = self.builders.get(key)
        if b is None:
            b = self.builders[key] = {tf: BarBuilder(tf) for tf in self.tfs}
        done: list[Bar] = []
        for builder in b.values():
            done.extend(builder.add(bar))
        return done

    def warm(self, key: str, bars: list[Bar]) -> None:
        """Rebuild HTF builder state from today's stored 1m bars after a
        restart, discarding outputs (those HTF bars were emitted before)."""
        self.builders.pop(key, None)
        for bar in bars:
            self._feed(key, bar)
            self.last_minute[key] = max(self.last_minute.get(key, ""), bar.ts.strftime(TS_FMT))

    # -- checkpoints / stats ---------------------------------------------------------

    def checkpoint(self, app_conn) -> str | None:
        if not self.last_minute:
            return None
        pos = max(self.last_minute.values())
        app_conn.execute("INSERT INTO checkpoints (consumer, position, ts) VALUES ('candles',?,?) "
                         "ON CONFLICT (consumer) DO UPDATE SET position=excluded.position, "
                         "ts=excluded.ts", (pos, datetime.now().isoformat(timespec="seconds")))
        app_conn.commit()
        return pos

    def median_spread_bps(self, key: str) -> float | None:
        s = self.spreads.get(key)
        return round(statistics.median(s), 2) if s and len(s) >= 5 else None

    def stats(self) -> dict:
        return {**self.counters, **{f"agg_{k}": v for k, v in self.agg.counters.items()},
                "instruments": len(self.builders)}


# ---------------------------------------------------------------------------
# Replay: the research path, through the same BarBuilder.
# ---------------------------------------------------------------------------

def load_1m(market_conn, key: str, frm: str, to: str) -> list[Bar]:
    rows = market_conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars_1m WHERE instrument_key=? "
        "AND ts>=? AND ts<=? ORDER BY ts", (key, frm, to)).fetchall()
    return [bar_from_row(r) for r in rows]


def replay(bars_1m: list[Bar], timeframes=HTF) -> dict[str, list[Bar]]:
    """Derive HTF bars from 1m bars exactly as the live service does."""
    builders = {tf: BarBuilder(tf) for tf in timeframes if tf in HTF}
    out: dict[str, list[Bar]] = {"1m": list(bars_1m), **{tf: [] for tf in builders}}
    for bar in bars_1m:
        for tf, b in builders.items():
            out[tf].extend(b.add(bar))
    return out


def checkpoint_position(app_conn) -> str | None:
    row = app_conn.execute("SELECT position FROM checkpoints WHERE consumer='candles'").fetchone()
    return row[0] if row else None
