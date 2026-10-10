"""Platform Phase 4: bounded structure engine, shared signal contract with
order-block origin enforcement, bullish/bearish pipelines, context,
scoring, clustering, determinism and the NIFTY baseline equivalence."""

import asyncio
import re
import sys
import tempfile
import tracemalloc
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ob_detect import random_walk_bars  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NIFTY = "NSE:NIFTY 50"


def cfg_(**over):
    from market_platform.config import from_dict
    return from_dict(over)


def params():
    from model.order_blocks.params import ObParams
    return replace(ObParams(), rvol_min=1.0)


def nifty_inst():
    return {"instrument_key": NIFTY, "kind": "index", "symbol": "NIFTY 50",
            "deriv_underlying": "NIFTY", "indices": [NIFTY], "fno_eligible": True}


def eq_inst(key, *, fno=True, sector="Financial Services", tier="high"):
    return {"instrument_key": key, "kind": "equity", "symbol": key.split(":")[1],
            "fno_eligible": fno, "deriv_underlying": key.split(":")[1] if fno else None,
            "sector": sector, "industry": sector, "liquidity_tier": tier, "adv_value_cr": 900,
            "median_spread_bps": 3.0, "lot_size": 500 if fno else None, "indices": [NIFTY]}


BARS = random_walk_bars(sessions=30, seed=11)


def layer(cfg=None, instruments=None, **kw):
    from market_platform.signals.pipeline import SignalLayer
    from market_platform.structure.engine import StructureEngine
    return SignalLayer(cfg or cfg_(), instruments=instruments or {NIFTY: nifty_inst()},
                       structure=StructureEngine(params()), strategy_version="test", **kw)


def run_layer(lay, key=NIFTY, bars=BARS):
    out = []
    for b, _c in bars:
        out.extend(lay.on_bar(key, b))
    return out


class TestBoundedCore(unittest.TestCase):
    def test_ring_semantics(self):
        from model.order_blocks.ring import RingList
        r = RingList(3)
        for x in "abcde":
            r.append(x)
        self.assertEqual(len(r), 5)
        self.assertEqual((r[4], r[2], r[-1]), ("e", "c", "e"))
        with self.assertRaises(IndexError):
            r[1]
        self.assertEqual(r[0:5], ["c", "d", "e"])
        self.assertEqual(r[3:], ["d", "e"])

    def test_bounded_equals_unbounded_and_memory_flat(self):
        from model.order_blocks.engine import ObEngine
        bars = random_walk_bars(sessions=45, seed=21)

        def run(mb):
            e = ObEngine(params(), max_bars=mb)
            out = []
            for b, c in bars:
                for ev in e.on_minute(b, c):
                    s = ev.setup
                    out.append((ev.kind, ev.ts, ev.zone.zone_id if ev.zone else None,
                                (s.plan.u_entry, s.plan.u_stop, s.plan.u_target, s.score.total)
                                if s else None))
            return out, e
        a, _ = run(None)
        ring = {"5m": 400, "15m": 200, "60m": 120}
        b, e = run(ring)
        self.assertEqual(a, b)
        for tf, n in ring.items():
            self.assertLessEqual(e.tf[tf].bars.held(), n)
            self.assertLessEqual(e.tf[tf].swings.bars.held(), n)
            self.assertLessEqual(len(e.tf[tf].swings.swings), 300)

    def test_memory_flat_once_rings_are_full(self):
        """A5: per-instrument state stops growing once the rings fill (≈40
        sessions for 15m swings). Measured steady footprint ≈ 0.45 MB/instrument."""
        import gc

        from market_platform.structure.engine import StructureEngine
        se = StructureEngine(params())
        bars = random_walk_bars(sessions=70, seed=3)
        tracemalloc.start()
        s0 = tracemalloc.take_snapshot()
        for b, _c in bars[:375 * 45]:
            se.on_bar("NSE:X", b)
        gc.collect()
        s1 = tracemalloc.take_snapshot()
        for b, _c in bars[375 * 45:]:
            se.on_bar("NSE:X", b)
        gc.collect()
        s2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        steady = sum(x.size_diff for x in s1.compare_to(s0, "filename"))
        growth = sum(x.size_diff for x in s2.compare_to(s1, "filename"))
        self.assertLess(growth, 60_000, f"grew {growth / 1024:.0f} KB over 25 more sessions after 45")
        self.assertLess(steady, 700_000, f"steady {steady / 1024:.0f} KB")

    def test_config_params_equal_validated_defaults(self):
        from market_platform.structure.engine import params_from_config
        from model.order_blocks.params import ObParams
        self.assertEqual(params_from_config(cfg_()).fingerprint(), ObParams().fingerprint())

    def test_bad_instrument_quarantined_others_continue(self):
        from market_platform.structure.engine import StructureEngine
        from model.order_blocks.types import Bar
        se = StructureEngine(params(), quarantine_after=2)
        bad = Bar(datetime(2026, 6, 1, 9, 15), "1m", 1, 2, 0.5, 1.5, None)
        eng = se.engine("NSE:BAD")
        eng.on_minute = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        se.on_bar("NSE:BAD", bad)
        se.on_bar("NSE:BAD", bad)
        self.assertIn("NSE:BAD", se.quarantined)
        for b, _c in BARS[:400]:
            se.on_bar("NSE:OK", b)
        self.assertEqual(se.counters["bars"], 400)


class TestOrderBlockOrigin(unittest.TestCase):
    def test_candidate_cannot_be_constructed_directly(self):
        from market_platform.signals.shared import OriginError, SignalCandidate
        with self.assertRaises(OriginError):
            SignalCandidate(signal_id="x", pipeline="bullish", instrument_key="NSE:A",
                            underlying="A", direction="bullish", strategy="s", setup_type="t",
                            timeframe="15m", horizon="intraday", zone_id="z", zone_low=1,
                            zone_high=2, detected_at=datetime.now(), available_at=datetime.now(),
                            session="d", entry=2, invalidation=1, stop=0.9, targets=[3], rr=2,
                            core_score=80, core_components={}, htf_trend="up", atr=1)

    def test_only_setup_events_of_matching_direction(self):
        from market_platform.signals.shared import OriginError, from_setup
        from market_platform.structure.engine import StructureEngine
        se = StructureEngine(params())
        zone_ev = setup_ev = None
        for b, _c in BARS:
            for ev in se.on_bar(NIFTY, b):
                if ev.kind == "zone_new" and zone_ev is None:
                    zone_ev = ev
                if ev.kind == "setup" and setup_ev is None:
                    setup_ev = ev
            if zone_ev and setup_ev:
                break
        with self.assertRaises(OriginError):
            from_setup(zone_ev, pipeline=zone_ev.direction, instrument=None, strategy_version="t")
        other = "bearish" if setup_ev.direction == "bullish" else "bullish"
        with self.assertRaises(OriginError):
            from_setup(setup_ev, pipeline=other, instrument=None, strategy_version="t")
        with self.assertRaises(OriginError):
            from_setup({"kind": "setup"}, pipeline="bullish", instrument=None, strategy_version="t")
        c = from_setup(setup_ev, pipeline=setup_ev.direction, instrument=None, strategy_version="t")
        self.assertEqual(c.zone_id, setup_ev.zone.zone_id)

    def test_no_other_construction_site_in_source(self):
        pat = re.compile(r"\bSignalCandidate\(")
        offenders = []
        for p in (REPO / "market_platform").rglob("*.py"):
            if p.name == "shared.py" and p.parent.name == "signals":
                continue
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if pat.search(line):
                    offenders.append(f"{p.relative_to(REPO)}:{i}")
        self.assertEqual(offenders, [])

    def test_every_stored_signal_has_a_stored_zone(self):
        from market_platform.persistence.db import open_db
        from market_platform.signals.pipeline import SignalStore
        app = open_db(Path(tempfile.mkdtemp()) / "app.db", "app")
        run_layer(layer(store=SignalStore(app)))
        n = app.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        orphans = app.execute("SELECT COUNT(*) FROM signals s LEFT JOIN zones z "
                              "ON z.zone_id=s.zone_id AND z.run_id=s.run_id WHERE z.zone_id IS NULL").fetchone()[0]
        self.assertGreater(n, 0)
        self.assertEqual(orphans, 0)


class TestNiftyBaselineEquivalence(unittest.TestCase):
    def test_platform_setups_equal_obengine_setups(self):
        """No silent strategy change: same bars → the platform's bullish and
        bearish pipelines see exactly the setups the NIFTY ObEngine produces."""
        from model.order_blocks.engine import ObEngine
        eng = ObEngine(params())
        ref = []
        for b, _c in BARS:
            for ev in eng.on_minute(b, NIFTY):
                if ev.kind == "setup":
                    s = ev.setup
                    ref.append((s.zone.zone_id, s.zone.direction, s.trigger_ts, s.horizon,
                                s.plan.u_entry, s.plan.u_stop, s.plan.u_target, s.score.total))
        got = [(c.zone_id, c.pipeline, c.detected_at, c.horizon, c.entry, c.stop, c.targets[0],
                c.core_score) for c in run_layer(layer())]
        # same setups; within one minute the platform orders by merit, not emission order
        self.assertEqual(sorted(got, key=str), sorted(ref, key=str))
        self.assertTrue({d for _, d, *_ in ref} == {"bullish", "bearish"})


class TestDeterminism(unittest.TestCase):
    def test_same_bars_same_ids_and_statuses(self):
        a = [(c.signal_id, c.status, c.score, c.cluster_id) for c in run_layer(layer())]
        b = [(c.signal_id, c.status, c.score, c.cluster_id) for c in run_layer(layer())]
        self.assertEqual(a, b)
        self.assertEqual(len({x[0] for x in a}), len(a))


class TestDirectionRules(unittest.TestCase):
    def _cands(self, inst, horizon=None):
        key = inst["instrument_key"]
        cs = run_layer(layer(instruments={key: inst}), key=key)
        return [c for c in cs if horizon is None or c.horizon == horizon]

    def test_bearish_overnight_needs_fno(self):
        cs = [c for c in self._cands(eq_inst("NSE:CASHONLY", fno=False))
              if c.pipeline == "bearish"]
        on = [c for c in cs if c.horizon == "overnight"]
        intra = [c for c in cs if c.horizon == "intraday"]
        self.assertTrue(on and intra)
        for c in on:
            self.assertFalse(c.executable)
            if c.status == "NOT_EXECUTABLE":
                self.assertIn("NOT_EXECUTABLE:NO_FNO_OVERNIGHT_SHORT", c.reject_reasons)
        for c in intra:
            self.assertEqual(c.routes, ["cash_mis_short"])

    def test_bullish_routes_and_index_without_derivatives(self):
        cs = self._cands(eq_inst("NSE:BANKCO", fno=True))
        bull = [c for c in cs if c.pipeline == "bullish"]
        self.assertTrue(bull)
        for c in bull:
            self.assertIn("fut_long", c.routes)
            self.assertIn("ce", c.routes)
        idx = {"instrument_key": "NSE:NIFTY IT", "kind": "index", "symbol": "NIFTY IT",
               "deriv_underlying": None, "indices": []}
        for c in self._cands(idx):
            self.assertFalse(c.executable)

    def test_confirmation_codes_are_direction_specific(self):
        cs = self._cands(eq_inst("NSE:BANKCO"))
        for c in cs:
            codes = {x.code for x in c.confirmations}
            if c.pipeline == "bullish":
                self.assertTrue({"VOLUME_DEMAND", "RS_BENCH", "STRUCTURE_HH_HL"} <= codes)
            else:
                self.assertTrue({"VOLUME_SELLING", "RS_WEAK_BENCH", "STRUCTURE_LH_LL"} <= codes)

    def test_pipelines_are_isolated(self):
        lay = layer()
        lay.pipelines["bearish"].rules.confirmations = lambda *a, **k: 1 / 0
        cs = run_layer(lay)
        self.assertTrue(all(c.pipeline == "bullish" for c in cs))
        self.assertGreater(len(cs), 0)
        self.assertGreater(lay.pipelines["bearish"].counters["errors"], 0)
        self.assertTrue(all(e["pipeline"] == "bearish" for e in lay.errors))


class TestScoring(unittest.TestCase):
    def _cand(self, **over):
        from market_platform.signals.shared import from_setup
        from market_platform.structure.engine import StructureEngine
        se = StructureEngine(params())
        for b, _c in BARS:
            for ev in se.on_bar(NIFTY, b):
                if ev.kind == "setup" and ev.direction == "bullish" \
                        and ev.setup.horizon == "intraday":
                    c = from_setup(ev, pipeline="bullish", instrument=None, strategy_version="t")
                    for k, v in over.items():
                        setattr(c, k, v)
                    return c
        self.fail("no bullish setup")

    def test_filters_reason_codes(self):
        from market_platform.scoring.score import Scorer
        cfg = cfg_()
        sc = Scorer(cfg, cfg.bullish)
        c = self._cand()
        c.rr = 1.0
        c.detected_at = c.detected_at.replace(hour=15, minute=0)
        c.available_at = c.detected_at + timedelta(minutes=10)
        reasons = sc.filters(c, instrument=None, quarantined={NIFTY}, gap_reason="x",
                             spread_bps=None)
        codes = [r.split(":")[0] for r in reasons]
        self.assertEqual(codes, ["RR_BELOW_MIN", "OUTSIDE_WINDOW", "GAP_AWAY", "DATA_QUARANTINED",
                                 "DATA_STALE", "NOT_IN_UNIVERSE"])
        thin = sc.filters(self._cand(), instrument=eq_inst("NSE:T", tier="thin"), quarantined=set(),
                          gap_reason="", spread_bps=40.0)
        self.assertEqual([r.split(":")[0] for r in thin], ["LIQUIDITY_THIN", "SPREAD_WIDE"])

    def test_score_parts_are_transparent(self):
        from market_platform.scoring.score import Scorer
        from market_platform.signals.shared import Confirmation
        cfg = cfg_()
        c = self._cand()
        c.confirmations = [Confirmation("VOLUME_DEMAND", 0.7, 0.55, True),
                           Confirmation("RS_BENCH", -1.0, 0, False),
                           Confirmation("STRUCTURE_HH_HL", None, 1, None)]
        total, parts = Scorer(cfg, cfg.bullish).score(c, alignment=0.5)
        self.assertEqual(parts, {"core": c.core_score, "volume": 5.0, "rs": -5.0, "structure": 0.0,
                                 "context": 2.5})
        self.assertEqual(total, round(min(100, c.core_score + 2.5), 1))

    def test_cluster_cap_suppresses_same_move(self):
        from market_platform.scoring.score import Clusterer
        cl = Clusterer(window_min=15, max_per_cluster=1)
        a, b, c = self._cand(status="QUALIFIED"), self._cand(status="QUALIFIED"), \
            self._cand(status="QUALIFIED")
        a.instrument_key, b.instrument_key, c.instrument_key = "NSE:HDFCBANK", "NSE:ICICIBANK", "NSE:TCS"
        cl.assign(a, eq_inst("NSE:HDFCBANK"))
        cl.assign(b, eq_inst("NSE:ICICIBANK"))
        cl.assign(c, eq_inst("NSE:TCS", sector="Information Technology"))
        self.assertEqual(a.cluster_id, b.cluster_id)
        self.assertEqual((a.status, b.status, c.status), ("QUALIFIED", "SUPPRESSED", "QUALIFIED"))
        self.assertTrue(b.reject_reasons[-1].startswith("CLUSTER_CAP"))
        cl2 = Clusterer(corr_cluster={"NSE:HDFCBANK": "g1", "NSE:TCS": "g1"})
        self.assertEqual(cl2.group("NSE:TCS", None), cl2.group("NSE:HDFCBANK", None))


class TestContext(unittest.TestCase):
    def _engine(self):
        from market_platform.context.engine import ContextEngine
        ins = [nifty_inst(),
               {"instrument_key": "NSE:NIFTY BANK", "kind": "index", "indices": ["NSE:NIFTY BANK"]},
               {**eq_inst("NSE:UP"), "indices": [NIFTY, "NSE:NIFTY BANK"]},
               {**eq_inst("NSE:DOWN"), "indices": [NIFTY, "NSE:NIFTY BANK"]}]
        meta = {"NSE:NIFTY BANK": {"category": "sector"}, NIFTY: {"category": "broad"}}
        ctx = ContextEngine(cfg_(), instruments=ins, index_meta=meta)
        flat = [100.0] * 210
        ctx.set_daily(NIFTY, [100 + i * 0.01 for i in range(210)])
        ctx.set_daily("NSE:NIFTY BANK", flat)
        ctx.set_daily("NSE:UP", [100 + i * 0.5 for i in range(210)])
        ctx.set_daily("NSE:DOWN", [300 - i * 0.5 for i in range(210)])
        return ctx

    def test_rs_breadth_and_alignment(self):
        from model.order_blocks.types import Bar
        ctx = self._engine()
        self.assertGreater(ctx.rs("NSE:UP", 20), 0)
        self.assertLess(ctx.rs("NSE:DOWN", 20), 0)
        self.assertEqual(ctx.sector_index["NSE:UP"], "NSE:NIFTY BANK")
        t = datetime(2026, 10, 6, 9, 15)
        ctx.on_bar("NSE:UP", Bar(t, "1m", 205, 206, 204, 206, 10))
        ctx.on_bar("NSE:DOWN", Bar(t, "1m", 195, 196, 190, 190, 10))
        for i in range(30):                                   # NIFTY 15m uptrend
            ctx.on_bar(NIFTY, Bar(t + timedelta(minutes=15 * i), "15m", 100 + i, 101 + i, 99 + i,
                                  100.5 + i, None))
        snap = ctx.compute(datetime(2026, 10, 6, 10, 0))
        self.assertEqual(snap.breadth["advancing"], 1)
        self.assertEqual(snap.breadth["declining"], 1)
        self.assertEqual(snap.breadth["above_200dma"], 0.5)
        self.assertEqual(snap.benchmark["trend"], "up")
        self.assertGreater(ctx.alignment("bullish", "NSE:UP"), 0)
        self.assertLess(ctx.alignment("bearish", "NSE:UP"), 0)
        self.assertEqual(snap.snapshot_id, ctx.compute(datetime(2026, 10, 6, 10, 0)).snapshot_id)

    def test_daily_load_has_no_look_ahead(self):
        from market_platform.context.engine import ContextEngine
        from market_platform.persistence.db import open_db
        m = open_db(Path(tempfile.mkdtemp()) / "m.db", "market")
        for i in range(10):
            d = (date(2026, 9, 21) + timedelta(days=i)).isoformat()
            m.execute("INSERT INTO bars_1d VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (NIFTY, d, 1, 1, 1, 100 + i, 1, None, 0, "t"))
        m.commit()
        ctx = ContextEngine(cfg_(), instruments=[nifty_inst()])
        ctx.load_daily(m, date(2026, 9, 25))
        self.assertEqual(ctx.daily[NIFTY].closes[-1], 103)          # 09-24, not 09-25


class TestCorrelation(unittest.TestCase):
    def test_clusters_and_beta(self):
        import random

        from market_platform.context.correlation import _returns, beta, clusters
        rng = random.Random(1)
        base = [100.0]
        for _ in range(70):
            base.append(base[-1] * (1 + rng.gauss(0, 0.01)))
        twin = [x * 1.5 + rng.gauss(0, 0.05) for x in base]
        noise = [100.0]
        for _ in range(70):
            noise.append(noise[-1] * (1 + rng.gauss(0, 0.01)))
        cl = clusters({"A": base, "B": twin, "C": noise})
        self.assertEqual(cl["A"], cl["B"])
        self.assertNotEqual(cl["A"], cl["C"])
        self.assertAlmostEqual(beta(_returns(twin), _returns(base)), 1.0, delta=0.1)


class TestDeterministicAllocation(unittest.TestCase):
    """Cluster suppression and desk order must not depend on processing order
    or task timing (scoring/score.py: time priority across minutes, merit
    priority within a minute)."""

    def _cands(self, n=6):
        import copy

        from market_platform.signals.shared import from_setup
        from market_platform.structure.engine import StructureEngine
        se = StructureEngine(params())
        for b, _c in BARS:
            for ev in se.on_bar(NIFTY, b):
                if ev.kind == "setup" and ev.direction == "bullish":
                    base = from_setup(ev, pipeline="bullish", instrument=None, strategy_version="t")
                    out = []
                    for i in range(n):
                        c = copy.deepcopy(base)
                        c.instrument_key, c.status = f"NSE:B{i}", "QUALIFIED"
                        c.score, c.rr = 80.0 + (i % 3), 2.0 + i / 10
                        out.append(c)
                    return out
        self.fail("no setup")

    def test_merit_not_order_decides_who_is_kept(self):
        import copy
        import random

        from market_platform.scoring.score import Clusterer
        inst = {f"NSE:B{i}": eq_inst(f"NSE:B{i}") for i in range(6)}     # one sector: one cluster
        results = set()
        for seed in range(8):
            cands = copy.deepcopy(self._cands())
            random.Random(seed).shuffle(cands)
            Clusterer(15, 1).allocate(cands, inst)
            kept = tuple(sorted(c.instrument_key for c in cands if c.status == "QUALIFIED"))
            results.add(kept)
            for c in cands:
                self.assertEqual(c.allocation["policy"], "time-then-merit")
                self.assertEqual(c.allocation["candidates_this_minute"], 6)
        self.assertEqual(len(results), 1)
        self.assertEqual(results.pop(), ("NSE:B5",))     # score 82, highest R:R among 82s

    def test_time_priority_across_minutes(self):
        import copy
        from datetime import timedelta

        from market_platform.scoring.score import Clusterer
        inst = {f"NSE:B{i}": eq_inst(f"NSE:B{i}") for i in range(6)}
        early, late = copy.deepcopy(self._cands(2))
        early.score, late.score = 76.0, 95.0
        late.detected_at = early.detected_at + timedelta(minutes=1)
        cl = Clusterer(15, 1)
        cl.allocate([early], inst)
        cl.allocate([late], inst)
        self.assertEqual((early.status, late.status), ("QUALIFIED", "SUPPRESSED"))
        self.assertEqual(late.allocation["taken_before"], [early.instrument_key])

    def test_live_tasks_decide_like_replay_under_random_scheduling(self):
        """Random delays in both pipelines and the desk → identical decisions."""
        import random

        from market_platform.marketdata.bus import Bus
        from market_platform.marketdata.events import CandleClosed
        from market_platform.persistence.db import open_db
        from market_platform.portfolio.book import Portfolio
        from market_platform.risk.desk import TradingDesk
        from market_platform.signals.pipeline import SignalStore
        cfg = cfg_()
        instruments = {"NSE:A": eq_inst("NSE:A"), "NSE:B": eq_inst("NSE:B"),
                       "NSE:C": eq_inst("NSE:C", sector="IT")}
        bars = random_walk_bars(sessions=12, seed=11)

        def decisions(app):
            return [tuple(r) for r in app.execute(
                "SELECT signal_id, approved, reason_codes FROM risk_decisions ORDER BY signal_id")]

        ref_app = open_db(Path(tempfile.mkdtemp()) / "a.db", "app")
        lay = layer(instruments=instruments, store=SignalStore(ref_app), run_id="r")
        desk = TradingDesk(cfg, ref_app, Portfolio(500_000, run_id="r"), instruments=instruments,
                           run_id="r")
        for b, _c in bars:
            minute = {k: b for k in instruments}
            for k in sorted(minute):
                desk.on_bar(k, minute[k])
            for c in lay.on_bars(minute):
                desk.process(c, c.available_at)
        ref = decisions(ref_app)
        self.assertTrue(ref)

        for seed in (1, 2):
            app = open_db(Path(tempfile.mkdtemp()) / "b.db", "app")
            lay2 = layer(instruments=instruments, store=SignalStore(app), run_id="r")
            desk2 = TradingDesk(cfg, app, Portfolio(500_000, run_id="r"), instruments=instruments,
                                run_id="r")
            rng = random.Random(seed)

            async def jitter_publish(bus, rng=rng):
                real = bus.publish

                async def pub(topic, ev):
                    if topic.startswith(("signals.", "structure.")):
                        for _ in range(rng.randint(0, 3)):
                            await asyncio.sleep(0)
                    await real(topic, ev)
                bus.publish = pub

            async def go(lay2=lay2, desk2=desk2, rng=rng):
                bus = Bus()
                await jitter_publish(bus, rng)
                t1 = asyncio.create_task(lay2.run(bus))
                t2 = asyncio.create_task(desk2.run(bus))
                await asyncio.sleep(0)
                for b, _c in bars:
                    for k in sorted(instruments):
                        await bus.publish("candles.1m", CandleClosed(k, "1m", b, b.end, "replay"))
                    for _ in range(rng.randint(5, 40)):
                        await asyncio.sleep(0)
                for _ in range(400):
                    await asyncio.sleep(0)
                t1.cancel()
                t2.cancel()
            asyncio.run(go())
            self.assertEqual(decisions(app), ref, f"seed {seed}")


class TestLiveTasks(unittest.TestCase):
    def test_bus_driven_pipelines(self):
        from market_platform.marketdata.bus import Bus
        from market_platform.marketdata.events import CandleClosed

        async def go():
            bus = Bus()
            lay = layer()
            bull = bus.subscribe("signals.bullish", "t1", maxsize=50000)
            bear = bus.subscribe("signals.bearish", "t2", maxsize=50000)
            task = asyncio.create_task(lay.run(bus))
            await asyncio.sleep(0)
            for b, _c in BARS:
                await bus.publish("candles.1m", CandleClosed(NIFTY, "1m", b, b.end, "replay"))
            # wait until every bar went through and every queue is empty
            for _ in range(200000):
                await asyncio.sleep(0)
                if lay.structure.counters["bars"] == len(BARS) and \
                        all(x["depth"] == 0 for x in bus.stats() if x["name"] not in ("t1", "t2")):
                    break
            for _ in range(50):
                await asyncio.sleep(0)
            task.cancel()
            flat = lambda sub: [c for m in sub.drain(10**6) for c in m.candidates]  # noqa: E731
            return flat(bull), flat(bear)
        bull, bear = asyncio.run(go())
        ref = run_layer(layer())
        self.assertEqual(sorted(c.signal_id for c in bull + bear), sorted(c.signal_id for c in ref))
        self.assertTrue(all(c.pipeline == "bullish" for c in bull))
        self.assertTrue(all(c.pipeline == "bearish" for c in bear))


if __name__ == "__main__":
    unittest.main()
