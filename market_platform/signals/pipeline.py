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
from dataclasses import dataclass
from datetime import datetime

from market_platform.candles.volume import join_index_volume
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


@dataclass(frozen=True)
class MinuteSetups:
    """Structure → pipeline: one direction's setups for one settled minute."""
    ts: datetime
    direction: str
    events: list


@dataclass(frozen=True)
class MinuteSignals:
    """Pipeline → desk: one direction's evaluated candidates for one minute
    (possibly empty — it doubles as the barrier for that minute)."""
    ts: datetime
    direction: str
    candidates: list


def desk_order(c) -> tuple:
    """Order in which the governor sees candidates: by availability, then
    merit (score, R:R), then direction and instrument only to break exact ties."""
    return (c.available_at, -(c.score or 0.0), -(c.rr or 0.0), c.direction, c.instrument_key)


class SwingsAt:
    """Frozen copy of a swing list with the tracker's `confirmed()` reader."""
    __slots__ = ("swings",)

    def __init__(self, swings) -> None:
        self.swings = list(swings)

    def confirmed(self, as_of, kind=None):
        return [s for s in self.swings if s.confirmed_ts <= as_of and (kind is None or s.kind == kind)]


def capture_inputs(ev: StructureEvent, structure: StructureEngine, context=None, spreads=None) -> dict:
    """Everything a pipeline reads about the market, frozen at detection time."""
    eng = structure.engines.get(ev.instrument_key)
    st = eng.tf.get(ev.zone.timeframe) if eng else None
    snap = {}
    if context is not None and context.snapshot is not None:
        snap = {"snapshot_id": context.snapshot.snapshot_id, "regime": context.snapshot.regime,
                "regime_conf": context.snapshot.regime_conf,
                "vol_regime": context.snapshot.vol_regime}
    return {
        "leg": leg_bars(ev.zone, st.bars) if st else [],
        "swings": SwingsAt(st.swings.swings) if st else None,
        "view": context.instrument_view(ev.instrument_key) if context else {},
        "alignment": {d: (context.alignment(d, ev.instrument_key) if context else 0.0)
                      for d in ("bullish", "bearish")},
        "snapshot": snap,
        "spread_bps": spreads(ev.instrument_key) if spreads else None,
    }


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

    def evaluate(self, ev: StructureEvent, *, instrument: dict | None, inputs: dict,
                 quarantined: set[str] = frozenset(), allocate: bool = True) -> SignalCandidate:
        """`inputs` is captured when the setup is detected (`capture_inputs`), so
        evaluation never reads engine or context state from a later minute."""
        self.counters["setups"] += 1
        cand = from_setup(ev, pipeline=self.direction, instrument=instrument,
                          strategy_version=self.strategy_version)
        leg, swings = inputs["leg"], inputs["swings"]
        view = inputs["view"]
        align = inputs["alignment"][self.direction]
        spread_bps = inputs.get("spread_bps")
        cand.context = {**inputs["snapshot"], "alignment": align, **view}
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
        self.counters["signals"] += 1
        if allocate:
            self.clusterer.allocate([cand], {cand.instrument_key: instrument or {}})
            self.by_status[cand.status] = self.by_status.get(cand.status, 0) + 1
        return cand

    def allocate(self, cands: list[SignalCandidate], instruments: dict[str, dict]) -> list:
        ranked = self.clusterer.allocate(cands, instruments)
        for c in ranked:
            self.by_status[c.status] = self.by_status.get(c.status, 0) + 1
        return ranked


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
                                      "routes": cand.routes, "executable": cand.executable,
                                      "allocation": cand.allocation}),
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
            self.app.execute("INSERT INTO signal_status_history (run_id, signal_id, ts, status, "
                             "reason) VALUES (?,?,?,?,?)", (run_id, cand.signal_id,
                                                            cand.available_at.isoformat(),
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
        #: index key → proxy future key (volume source); set per session
        self.volume_proxy: dict[str, str] = {}
        #: keys fed only as volume sources (not analysed as instruments)
        self.proxy_only: set[str] = set()

    def warm(self, bars: dict) -> int:
        """Feed one minute of historical bars to the structure engine only:
        events are discarded, nothing is stored or traded. Used before a live
        session so the engine starts with the state a continuous run would have."""
        bars = join_index_volume(bars, self.volume_proxy)
        n = 0
        for key in sorted(bars):
            if key in self.proxy_only:
                continue
            self.structure.on_bar(key, bars[key], self.volume_proxy.get(key))
            n += 1
        return n

    def set_volume_proxy(self, proxy: dict[str, str]) -> None:
        self.volume_proxy = dict(proxy)
        self.proxy_only = {f for f in proxy.values() if f not in self.instruments}

    # -- sync path (replay, tests, and inside the live tasks) -------------------------------

    def on_bars(self, bars: dict) -> list[SignalCandidate]:
        """All instruments' bars of one minute → that minute's candidates.

        Index bars get their proxy future's volume first. Every instrument is
        run through the structure engine before any candidate is judged, so
        the cluster allocation sees the whole minute at once, and the result
        is returned in DESK_ORDER — none of it depends on processing order."""
        bars = join_index_volume(bars, self.volume_proxy)
        setups: dict[str, list[StructureEvent]] = {d: [] for d in self.pipelines}
        for key in sorted(bars):
            if key in self.proxy_only:
                continue
            if self.context is not None:
                self.context.on_bar(key, bars[key])
            for ev in self.structure.on_bar(key, bars[key], self.volume_proxy.get(key)):
                if self.store is not None:
                    self.store.save_zone(ev, self.run_id)
                if ev.kind == "setup":
                    setups[ev.direction].append((ev, self._capture(ev)))
        out: list[SignalCandidate] = []
        for d, evs in setups.items():
            out.extend(self.evaluate_minute(d, evs))
        if self.store is not None:
            self.store.commit()
        return sorted(out, key=desk_order)

    def on_bar(self, key: str, bar) -> list[SignalCandidate]:
        return self.on_bars({key: bar})

    def handle(self, ev: StructureEvent) -> SignalCandidate | None:
        if self.store is not None:
            self.store.save_zone(ev, self.run_id)
        if ev.kind != "setup":
            return None
        return self.evaluate(ev)

    def _capture(self, ev: StructureEvent) -> dict:
        return capture_inputs(ev, self.structure, self.context, self.spreads)

    def evaluate(self, ev: StructureEvent) -> SignalCandidate | None:
        out = self.evaluate_minute(ev.direction, [(ev, self._capture(ev))])
        return out[0] if out else None

    def evaluate_minute(self, direction: str, events: list[tuple]) -> list[SignalCandidate]:
        """Evaluate one direction's setups of one minute, allocate clusters in
        merit order, persist. An exception isolates that setup only."""
        pipe = self.pipelines[direction]
        cands = []
        for ev, inputs in events:
            try:
                cands.append(pipe.evaluate(ev, instrument=self.instruments.get(ev.instrument_key),
                                           inputs=inputs, quarantined=self.quarantined,
                                           allocate=False))
            except Exception as exc:                        # isolate the direction + instrument
                pipe.counters["errors"] += 1
                self.errors.append({"ts": ev.at.isoformat(), "pipeline": ev.direction,
                                    "instrument_key": ev.instrument_key, "error": repr(exc)})
                log.exception("%s pipeline failed on %s", ev.direction, ev.instrument_key)
        ranked = pipe.allocate(cands, self.instruments)
        if self.store is not None:
            for c in ranked:
                self.store.save(c, run_id=self.run_id, config_hash=self.cfg.hash,
                                universe_snapshot=self.universe_snapshot)
            self.store.commit()
        return ranked

    # -- live path: separate tasks and queues per direction ---------------------------------

    async def run(self, bus, *, health=None) -> None:
        """Consume `candles.1m`. For every settled minute the structure task
        publishes one `MinuteSetups` per direction (possibly empty) on
        `structure.<direction>`; each pipeline task answers with one
        `MinuteSignals` on `signals.<direction>`. The desk decides a minute only
        when both directions have answered (a barrier), so the order in which
        the tasks happen to run cannot change any decision."""
        q = {d: bus.subscribe(f"structure.{d}", f"pipeline.{d}",
                              maxsize=self.cfg.workers.signal_queue, policy="block")
             for d in self.pipelines}
        candles = bus.subscribe("candles.1m", "structure", maxsize=self.cfg.workers.candle_queue,
                                policy="block")

        async def structure_task():
            while True:
                # The candle service publishes a whole settled batch at once:
                # drain it and process minute by minute, instruments sorted, so
                # live and replay see identical input (and index bars can be
                # joined with their proxy future's volume).
                batch = [await candles.get(), *candles.drain()]
                by_ts: dict = {}
                for ev in batch:
                    by_ts.setdefault(ev.bar.ts, {})[ev.instrument_key] = ev.bar
                for ts in sorted(by_ts):
                    bars = join_index_volume(by_ts[ts], self.volume_proxy)
                    setups: dict[str, list] = {d: [] for d in self.pipelines}
                    for key in sorted(bars):
                        if key in self.proxy_only:
                            continue
                        if self.context is not None:
                            self.context.on_bar(key, bars[key])
                        for sev in self.structure.on_bar(key, bars[key], self.volume_proxy.get(key)):
                            if self.store is not None:
                                self.store.save_zone(sev, self.run_id)
                            if sev.kind == "setup":
                                setups[sev.direction].append((sev, self._capture(sev)))
                            else:
                                await bus.publish("structure.zones", sev)
                    if self.store is not None:
                        self.store.commit()
                    for d, evs in setups.items():
                        await bus.publish(f"structure.{d}", MinuteSetups(ts, d, evs))

        async def pipeline_task(direction: str):
            sub = q[direction]
            while True:
                msg = await sub.get()
                n_err = self.pipelines[direction].counters["errors"]
                cands = self.evaluate_minute(direction, msg.events)
                if health is not None and self.pipelines[direction].counters["errors"] > n_err:
                    health("strategy_error", direction,
                           ",".join(e.instrument_key for e, _i in msg.events))
                await bus.publish(f"signals.{direction}", MinuteSignals(msg.ts, direction, cands))

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
