"""Load harness (plan §9.6): synthetic ticks through the real live path.

    python platform_cli.py load-test --tokens 1500 --minutes 10 --mult 1,2,4

The token count is given explicitly; the multiplier scales the TICK RATE,
so 1×/2×/4× load means 1, 2 and 4 ticks per token per second (Kite sends a
`full` packet roughly once per second per actively traded token).

Path exercised, in one asyncio loop exactly like the runner:
    raw tick dicts → MarketDataPool.on_raw (normalise) → CandleService
    (aggregate, settle, write market.db via AsyncWriter, HTF) → bus
    → SignalLayer task (structure engine for every instrument, bullish /
    bearish pipelines) → TradingDesk task (governor, paper fills).

Measured per simulated minute:
    ingest_ticks_per_sec   ticks / wall time spent in on_raw
    settle_ms              CandleService.on_clock (completion overhead; the
                           completion *delay* is grace + this)
    pipeline_ms            from the minute's candles to all bus queues drained
                           (structure + signals + risk for every instrument)
    writer                 rows committed, commit p99, queue high-water
    memory                 tracemalloc current / peak
Results are written to reports/load/ and quoted in docs/PLATFORM.md.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class _Socket:
    """In-memory socket: accepts subscriptions, never connects."""

    def __init__(self, *a, **k):
        self.state, self.counters, self._sub = "connected", {}, {}
        self._stop = asyncio.Event()

    async def subscribe(self, tokens, mode="quote"):
        for t in tokens:
            self._sub[t] = mode

    async def unsubscribe(self, tokens):
        for t in tokens:
            self._sub.pop(t, None)

    async def set_mode(self, tokens, mode):
        await self.subscribe(tokens, mode)

    def subscribed(self):
        return dict(self._sub)

    async def run(self):
        await self._stop.wait()

    async def close(self):
        self._stop.set()


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


async def _drain(bus, timeout: float = 30.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        await asyncio.sleep(0)
        if all(s["depth"] == 0 for s in bus.stats()):
            await asyncio.sleep(0)
            if all(s["depth"] == 0 for s in bus.stats()):
                return


async def run_load(cfg, dbs, *, tokens: int, minutes: int, ticks_per_sec: float,
                   start: datetime | None = None, seed: int = 1) -> dict:
    from market_platform.marketdata.stream import DataPlane
    from market_platform.portfolio.book import Portfolio
    from market_platform.risk.desk import TradingDesk
    from market_platform.signals.pipeline import SignalLayer, SignalStore

    rng = random.Random(seed)
    inst = [{"instrument_key": f"NSE:S{i:04d}", "token": 100_000 + i, "kind": "equity",
             "symbol": f"S{i:04d}", "exchange": "NSE", "fno_eligible": i % 3 == 0,
             "deriv_underlying": f"S{i:04d}" if i % 3 == 0 else None,
             "sector": f"Sector{i % 12}", "liquidity_tier": "high", "adv_value_cr": 500.0,
             "median_spread_bps": 3.0, "indices": ["NSE:NIFTY 500"]} for i in range(tokens)]
    dp = DataPlane(cfg, dbs, ws_class=_Socket, instruments=inst)
    layer = SignalLayer(cfg, instruments={i["instrument_key"]: i for i in inst},
                        store=SignalStore(dbs.app), run_id="load", strategy_version="load")
    desk = TradingDesk(cfg, dbs.app, Portfolio(cfg.risk.equity_rupees, run_id="load"),
                       instruments={i["instrument_key"]: i for i in inst}, run_id="load")
    await dp.start(recover=False)
    tasks = [asyncio.create_task(layer.run(dp.bus)), asyncio.create_task(desk.run(dp.bus))]
    for t in dp._tasks:                        # the harness drives the clock itself
        t.cancel()
    t0 = start or datetime(2026, 10, 6, 9, 15, tzinfo=IST)
    px = [1000.0 + rng.random() * 2000 for _ in range(tokens)]
    vol = [0] * tokens
    drift = [rng.choice((-1, 1)) * 0.0004 for _ in range(tokens)]
    tracemalloc.start()
    ingest_s, ticks = 0.0, 0
    settle_ms, pipe_ms = [], []
    per_sec = max(1, int(round(ticks_per_sec)))
    for m in range(minutes):
        if m % 20 == 0:
            drift = [rng.choice((-1, 1)) * 0.0004 for _ in range(tokens)]
        for s in range(60):
            ts = t0 + timedelta(minutes=m, seconds=s)
            for k in range(per_sec):
                batch = []
                for i in range(tokens):
                    if ticks_per_sec < 1 and rng.random() > ticks_per_sec:
                        continue
                    px[i] = round(px[i] * (1 + drift[i] + rng.gauss(0, 0.0007)), 2)
                    vol[i] += rng.randint(1, 400)
                    batch.append({"token": 100_000 + i, "ltp": px[i], "volume": vol[i],
                                  "exchange_ts": ts + timedelta(milliseconds=k * 1000 // per_sec),
                                  "depth": {"buy": [{"price": round(px[i] - 0.05, 2), "quantity": 100}],
                                            "sell": [{"price": round(px[i] + 0.05, 2), "quantity": 100}]}})
                a = time.perf_counter()
                await dp.pool.on_raw(batch, recv_ts=ts)
                ingest_s += time.perf_counter() - a
                ticks += len(batch)
                # a real socket awaits recv() between frames: yield like it does,
                # so the writer and the pipeline tasks interleave as in live
                await asyncio.sleep(0)
        end = t0 + timedelta(minutes=m + 1, seconds=cfg.data.candle_grace_sec)
        a = time.perf_counter()
        await dp.candles.on_clock(end)
        b = time.perf_counter()
        await _drain(dp.bus)
        c = time.perf_counter()
        settle_ms.append((b - a) * 1000)
        pipe_ms.append((c - b) * 1000)
    await dp.writer.flush()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await dp.stop()
    w = dp.writer.stats
    return {
        "tokens": tokens, "minutes": minutes, "ticks_per_token_per_sec": ticks_per_sec,
        "ticks": ticks, "ingest_ticks_per_sec": round(ticks / ingest_s) if ingest_s else None,
        "settle_ms": {"p50": _pct(settle_ms, 50), "p99": _pct(settle_ms, 99), "max": _pct(settle_ms, 100)},
        "pipeline_ms": {"p50": _pct(pipe_ms, 50), "p99": _pct(pipe_ms, 99), "max": _pct(pipe_ms, 100)},
        "bars_1m": dp.candles.counters["bars_1m"], "bars_htf": dp.candles.counters["bars_htf"],
        "writer": {"committed": w.committed, "batches": w.batches, "errors": w.errors,
                   "dropped": w.dropped, "commit_ms_p99": _pct(w.recent_commit_ms, 99),
                   "max_commit_ms": round(w.max_commit_ms, 1),
                   "backpressure_waits": w.backpressure_waits},
        "bus_high_water": {s["name"]: s["high_water"] for s in dp.bus.stats()},
        "signals": layer.stats()["structure"]["setups"],
        "memory_mb": {"current": round(cur / 1e6, 1), "peak": round(peak / 1e6, 1)},
        "slo": {
            # sustained capacity ÷ offered tick rate (must stay > 1 with margin)
            "ingest_headroom_x": round(ticks / ingest_s / (tokens * ticks_per_sec), 1)
            if ingest_s else None,
            "candle_completion_p99_s": round(cfg.data.candle_grace_sec + (_pct(settle_ms, 99) or 0) / 1000, 3),
            "structure_all_instruments_p99_s": round((_pct(pipe_ms, 99) or 0) / 1000, 3),
        },
    }


def write(results: list[dict], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"load-{datetime.now():%Y%m%d-%H%M%S}.json"
    p.write_text(json.dumps(results, indent=2))
    return p
