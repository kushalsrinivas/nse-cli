"""Signal layer: structure events → bullish / bearish pipelines → stored signals.

    layer = SignalLayer(cfg, instruments=..., context=ctx, run_id=..., store=SignalStore(app))
    cands = layer.on_bars({"NSE:RELIANCE": bar, ...})        # sync (replay, tests)
    await layer.run(bus)                                       # live (separate tasks)

* One `StructureEngine` (shared core) produces events once per instrument.
* `setup` events are routed by zone direction to `pipelines["bullish"]` or
  `pipelines["bearish"]`. In live mode each pipeline is its own asyncio task
  reading its own bounded queue; an exception in one pipeline is recorded as
  a `strategy_error` health event and never stops the other.
* Zones are persisted on every lifecycle event; signals are written once
  (INSERT OR IGNORE on the deterministic id) with a status-history row.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from market_platform.scoring.score import Clusterer, Scorer
from market_platform.signals.bearish import BearishRules
from market_platform.signals.bullish import BullishRules
from market_platform.signals.shared import SignalCandidate, from_setup, leg_bars
from market_platform.structure.engine import (
    ZONE_UPSERT,
    StructureEngine,
    StructureEvent,
    zone_row,
)

log = logging.getLogger(__name__)

SIGNAL_COLUMNS = (
    "signal_id", "run_id", "pipeline", "instrument_key", "underlying", "direction", "strategy",
    "setup_type", "timeframe", "horizon", "zone_id", "zone_low", "zone_high", "detected_at",
    "available_at", "session", "entry", "invalidation", "stop", "targets_json", "rr", "score",
    "score_json", "p_calibrated", "confirmations_json", "context_json", "data_quality",
    "liquidity_json", "est_costs", "proposal_json", "cluster_id", "status", "qualify_reasons",
    "reject_reasons", "config_hash", "universe_snapshot", "created_at")


class SignalPipeline:
    """One direction. Pure evaluation; persistence is the caller's."""

    def __init__(self, cfg, direction: str, *, strategy_version: str) -> None:
        self.direction = direction
        sc = cfg.bullish if direction == "bullish" else cfg.bearish
        self.rules = BullishRules(sc) if direction == "bullish" else BearishRules(sc)
        self.scorer = Scorer(cfg, sc)
        self.clusterer = Clusterer(cfg.scoring.cluster_window_min,
                                   cfg.scoring.max_signals_per_cluster)
        self.strategy_version = strategy_version
        self.counters = {"setups": 0, "signals": 0, "errors": 0}
        self.by_status: dict[str, int] = {}

    def evaluate(self, ev: StructureEvent, *, instrument: dict | None, structure: StructureEngine,
                 context=None, quarantined: set[str] = frozenset(),
                 spread_bps: float | None = None) -> SignalCandidate:
        self.counters["setups"] += 1
        cand = from_setup(ev, pipeline=self.direction, instrument=instrument,
                          strategy_version=self.strategy_version)
        eng = structure.engines.get(ev.instrument_key)
        st = eng.tf.get(ev.zone.timeframe) if eng else None
        leg = leg_bars(ev.zone, st.bars) if st else []
        swings = st.swings if st else None
        view = context.instrument_view(ev.instrument_key) if context else {}
        align = context.alignment(self.direction, ev.instrument_key) if context else 0.0
        if context is not None and context.snapshot is not None:
            cand.context = {"snapshot_id": context.snapshot.snapshot_id,
                            "regime": context.snapshot.regime,
                            "regime_conf": context.snapshot.regime_conf,
                            "vol_regime": context.snapshot.vol_regime,
                            "alignment": align, **view}
        else:
            cand.context = {"alignment": align, **view}
        cand.confirmations = self.rules.confirmations(cand, zone=ev.zone, leg_bars=leg,
                                                      swings=swings, ctx_view=view, alignment=align)
        cand.executable, cand.routes, why = self.rules.executability(cand, instrument)
        gap = self.rules.gap_reject(cand, view)
        cand.data_quality = "QUARANTINED" if ev.instrument_key in quarantined else "OK"
        if instrument:
            cand.liquidity = {k: instrument.get(k) for k in ("liquidity_tier", "adv_value_cr",
                                                              "median_spread_bps", "lot_size",
                                                              "fno_eligible", "weekly_options")}
            if spread_bps is not None:
                cand.liquidity["live_spread_bps"] = spread_bps
        self.scorer.evaluate(cand, instrument=instrument, quarantined=quarantined, gap_reason=gap,
                             alignment=align, spread_bps=spread_bps)
        if cand.status == "NOT_EXECUTABLE":
            cand.reject_reasons = [*cand.reject_reasons, f"NOT_EXECUTABLE:{why}"]
        elif cand.executable is False:
            cand.qualify_reasons = [*cand.qualify_reasons, f"view only — NOT_EXECUTABLE:{why}"]
        self.clusterer.assign(cand, instrument)
        self.counters["signals"] += 1
        self.by_status[cand.status] = self.by_status.get(cand.status, 0) + 1
        return cand


class SignalStore:
    def __init__(self, app_conn) -> None:
        self.app = app_conn

    def save_zone(self, ev: StructureEvent, run_id: str) -> None:
        self.app.execute(ZONE_UPSERT, zone_row(ev, run_id))

    def save(self, cand: SignalCandidate, *, run_id: str, config_hash: str,
             universe_snapshot: str) -> bool:
        row = {
            "signal_id": cand.signal_id, "run_id": run_id, "pipeline": cand.pipeline,
            "instrument_key": cand.instrument_key, "underlying": cand.underlying,
            "direction": cand.direction, "strategy": cand.strategy, "setup_type": cand.setup_type,
            "timeframe": cand.timeframe, "horizon": cand.horizon, "zone_id": cand.zone_id,
            "zone_low": cand.zone_low, "zone_high": cand.zone_high,
            "detected_at": cand.detected_at.isoformat(), "available_at": cand.available_at.isoformat(),
            "session": cand.session, "entry": cand.entry, "invalidation": cand.invalidation,
            "stop": cand.stop, "targets_json": json.dumps(cand.targets), "rr": cand.rr,
            "score": cand.score,
            "score_json": json.dumps({"parts": cand.score_parts, "core": cand.core_components,
                                      "htf_trend": cand.htf_trend, "atr": cand.atr,
                                      "routes": cand.routes, "executable": cand.executable}),
            "p_calibrated": None,
            "confirmations_json": json.dumps([c.__dict__ for c in cand.confirmations], default=str),
            "context_json": json.dumps(cand.context, default=str),
            "data_quality": cand.data_quality, "liquidity_json": json.dumps(cand.liquidity),
            "est_costs": None, "proposal_json": None, "cluster_id": cand.cluster_id,
            "status": cand.status, "qualify_reasons": json.dumps(cand.qualify_reasons),
            "reject_reasons": json.dumps(cand.reject_reasons), "config_hash": config_hash,
            "universe_snapshot": universe_snapshot,
            "created_at": datetime.now().isoformat(timespec="seconds")}
        cur = self.app.execute(
            f"INSERT OR IGNORE INTO signals ({', '.join(SIGNAL_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(SIGNAL_COLUMNS))})", [row[c] for c in SIGNAL_COLUMNS])
        if cur.rowcount:
            reason = (cand.reject_reasons or cand.qualify_reasons or [""])[0]
            self.app.execute("INSERT INTO signal_status_history (signal_id, ts, status, reason) "
                             "VALUES (?,?,?,?)", (cand.signal_id, cand.available_at.isoformat(),
                                                  cand.status, reason))
        return bool(cur.rowcount)

    def commit(self) -> None:
        self.app.commit()


class SignalLayer:
    def __init__(self, cfg, *, instruments: dict[str, dict], structure: StructureEngine | None = None,
                 context=None, store: SignalStore | None = None, run_id: str = "adhoc",
                 universe_snapshot: str = "none", strategy_version: str | None = None,
                 quarantined: set[str] | None = None, spreads=None) -> None:
        from market_platform.persistence.runs import strategy_version as sv
        self.cfg = cfg
        self.instruments = instruments
        self.structure = structure or StructureEngine.from_config(cfg)
        self.context = context
        self.store = store
        self.run_id = run_id
        self.universe_snapshot = universe_snapshot
        version = strategy_version or sv()
        self.pipelines = {d: SignalPipeline(cfg, d, strategy_version=version)
                          for d in ("bullish", "bearish")}
        self.quarantined = quarantined or set()
        self.spreads = spreads                 # callable key -> median spread bps, or None
        self.errors: list[dict] = []

    # -- sync path (replay, tests, and inside the live tasks) -------------------------------

    def on_bars(self, bars: dict) -> list[SignalCandidate]:
        out: list[SignalCandidate] = []
        for key in sorted(bars):
            out.extend(self.on_bar(key, bars[key]))
        return out

    def on_bar(self, key: str, bar) -> list[SignalCandidate]:
        if self.context is not None:
            self.context.on_bar(key, bar)
        out = []
        for ev in self.structure.on_bar(key, bar):
            cand = self.handle(ev)
            if cand is not None:
                out.append(cand)
        return out

    def handle(self, ev: StructureEvent) -> SignalCandidate | None:
        if self.store is not None:
            self.store.save_zone(ev, self.run_id)
        if ev.kind != "setup":
            return None
        return self.evaluate(ev)

    def evaluate(self, ev: StructureEvent) -> SignalCandidate | None:
        pipe = self.pipelines[ev.direction]
        try:
            sp = self.spreads(ev.instrument_key) if self.spreads else None
            cand = pipe.evaluate(ev, instrument=self.instruments.get(ev.instrument_key),
                                 structure=self.structure, context=self.context,
                                 quarantined=self.quarantined, spread_bps=sp)
        except Exception as exc:                        # isolate the direction + instrument
            pipe.counters["errors"] += 1
            self.errors.append({"ts": ev.at.isoformat(), "pipeline": ev.direction,
                                "instrument_key": ev.instrument_key, "error": repr(exc)})
            log.exception("%s pipeline failed on %s", ev.direction, ev.instrument_key)
            return None
        if self.store is not None:
            self.store.save(cand, run_id=self.run_id, config_hash=self.cfg.hash,
                            universe_snapshot=self.universe_snapshot)
            self.store.commit()
        return cand

    # -- live path: separate tasks and queues per direction ---------------------------------

    async def run(self, bus, *, health=None) -> None:
        """Consume `candles.1m`; route setups to per-direction queues; each
        pipeline task publishes `signals.<direction>`."""
        q = {d: bus.subscribe(f"structure.{d}", f"pipeline.{d}",
                              maxsize=self.cfg.workers.signal_queue, policy="block")
             for d in self.pipelines}
        candles = bus.subscribe("candles.1m", "structure", maxsize=self.cfg.workers.candle_queue,
                                policy="block")

        async def structure_task():
            while True:
                ev = await candles.get()
                if self.context is not None:
                    self.context.on_bar(ev.instrument_key, ev.bar)
                for sev in self.structure.on_bar(ev.instrument_key, ev.bar):
                    if self.store is not None:
                        self.store.save_zone(sev, self.run_id)
                    if sev.kind == "setup":
                        await bus.publish(f"structure.{sev.direction}", sev)
                    else:
                        await bus.publish("structure.zones", sev)
                if self.store is not None:
                    self.store.commit()

        async def pipeline_task(direction: str):
            sub = q[direction]
            while True:
                sev = await sub.get()
                cand = self.evaluate(sev)
                if cand is None:
                    if health is not None:
                        health("strategy_error", direction, sev.instrument_key)
                    continue
                await bus.publish(f"signals.{direction}", cand)

        tasks = [asyncio.create_task(structure_task(), name="structure"),
                 *(asyncio.create_task(pipeline_task(d), name=f"pipeline.{d}")
                   for d in self.pipelines)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()

    def stats(self) -> dict:
        return {"structure": self.structure.stats(),
                **{d: {**p.counters, "by_status": p.by_status} for d, p in self.pipelines.items()},
                "errors": self.errors[-20:]}
