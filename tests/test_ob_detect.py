"""Order-block detector: bars, indicators, swings, structure, zones, lifecycle,
triggers, scoring, exits — and the no-look-ahead property of the engine."""

import random
import sys
import unittest
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.order_blocks.types import Bar  # noqa: E402

T0 = datetime(2026, 10, 6, 9, 15)


def b15(i, o, h, l, c, v=None, start=T0):  # noqa: E741
    return Bar(start + timedelta(minutes=15 * i), "15m", o, h, l, c, v)


def random_walk_bars(sessions=25, seed=5, start=datetime(2026, 6, 1)):
    rng = random.Random(seed)
    out, px, d = [], 24000.0, start
    while len({b.ts.date() for b, _ in out}) < sessions:
        if d.weekday() < 5:
            t = d.replace(hour=9, minute=15)
            px += rng.gauss(0, 30)
            for i in range(375):
                o = px
                px += rng.gauss(0, 6) + (3 if (i // 50) % 2 else -3)
                h = max(o, px) + abs(rng.gauss(0, 3))
                lo = min(o, px) - abs(rng.gauss(0, 3))
                out.append((Bar(t, "1m", o, h, lo, px, rng.randint(500, 3000)), "FUT"))
                t += timedelta(minutes=1)
        d += timedelta(days=1)
    return out


# Hand-built bullish order block (15m, k=3).
SEQ = [
    (100, 101, 99, 100), (100, 102, 99.5, 101), (101, 103, 100, 102),
    (102, 110, 101, 103),                       # 3: swing high 110
    (103, 105, 100, 101), (101, 103, 96, 97),   # 5: bearish
    (97, 99, 95, 96),                           # 6: bearish, lowest low → origin + source
    (96, 104, 95.5, 103.5),                     # 7: impulse
    (103.5, 113, 103, 112),                     # 8: closes above 110 → BOS
]


def _structure(params):
    from model.order_blocks.indicators import AtrTracker, RvolTracker
    from model.order_blocks.structure import StructureTracker
    from model.order_blocks.swings import SwingTracker
    bars = [b15(i, *x) for i, x in enumerate(SEQ)]
    sw, st = SwingTracker(params.pivot_k), StructureTracker()
    atr, rv = AtrTracker(params.atr_n), RvolTracker(20, 20, 10)
    brk = None
    for i, b in enumerate(bars):
        atr.add(b)
        rv.add(b)
        st.add_swings(sw.add(b))
        x = st.on_bar(b, i)
        brk = brk or x
    return bars, sw, st, atr, rv, brk


class TestBars(unittest.TestCase):
    def test_15m_bins_anchor_at_0915_and_emit_on_last_minute(self):
        from model.order_blocks.bars import BarBuilder
        bb = BarBuilder("15m")
        out = []
        for m in range(30):
            ts = T0 + timedelta(minutes=m)
            out += bb.add(Bar(ts, "1m", 100 + m, 101 + m, 99 + m, 100.5 + m, 10))
            if m == 13:
                self.assertEqual(out, [])            # incomplete bin never exposed
        self.assertEqual([b.ts for b in out], [T0, T0 + timedelta(minutes=15)])
        self.assertEqual(out[0].volume, 150)
        self.assertEqual(out[0].high, 101 + 14)

    def test_60m_partial_last_bin_dropped(self):
        from model.order_blocks.bars import BarBuilder
        bb = BarBuilder("60m")
        out = []
        for m in range(375):
            out += bb.add(Bar(T0 + timedelta(minutes=m), "1m", 1, 1, 1, 1, None))
        self.assertEqual(len(out), 6)                # 09:15 … 14:15; 15:15 never emitted
        self.assertEqual(out[-1].ts.time(), dtime(14, 15))
        nxt = bb.add(Bar(T0 + timedelta(days=1), "1m", 1, 1, 1, 1, None))
        self.assertEqual(nxt, [])                    # next session discards the partial bin
        self.assertEqual(bb.dropped_partial, 1)

    def test_gap_in_minutes_still_closes_bin(self):
        from model.order_blocks.bars import BarBuilder
        bb = BarBuilder("5m")
        bb.add(Bar(T0, "1m", 1, 2, 0, 1, None))
        out = bb.add(Bar(T0 + timedelta(minutes=7), "1m", 1, 2, 0, 1, None))
        self.assertEqual([b.ts for b in out], [T0])


class TestIndicators(unittest.TestCase):
    def test_atr_wilder(self):
        from model.order_blocks.indicators import AtrTracker
        a = AtrTracker(3)
        bars = [b15(i, 10, 12, 9, 11) for i in range(5)]
        vals = [a.add(b) for b in bars]
        self.assertIsNone(vals[1])
        self.assertAlmostEqual(vals[2], 3.0)
        self.assertAlmostEqual(vals[3], (3.0 * 2 + 3.0) / 3)

    def test_rvol_excludes_current_bar(self):
        from model.order_blocks.indicators import RvolTracker
        r = RvolTracker(n=3, tod_sessions=20, tod_min_sessions=99)
        vals = [r.add(b15(i, 1, 1, 1, 1, v), "C") for i, v in enumerate([100, 100, 100, 300])]
        self.assertIsNone(vals[2])                  # baseline not yet full
        self.assertAlmostEqual(vals[3], 3.0)        # 300 / mean(100,100,100)

    def test_rvol_null_across_contract_roll(self):
        from model.order_blocks.indicators import RvolTracker
        r = RvolTracker(n=2, tod_sessions=20, tod_min_sessions=99)
        r.add(b15(0, 1, 1, 1, 1, 100), "OCT")
        r.add(b15(1, 1, 1, 1, 1, 100), "OCT")
        self.assertIsNone(r.add(b15(2, 1, 1, 1, 1, 100), "NOV"))
        self.assertIsNone(r.add(b15(3, 1, 1, 1, 1, 100), "NOV"))   # window still mixed
        self.assertIsNotNone(r.add(b15(4, 1, 1, 1, 1, 100), "NOV"))


class TestSwingsAndStructure(unittest.TestCase):
    def test_swing_confirmed_only_after_k_bars(self):
        from model.order_blocks.swings import SwingTracker
        sw = SwingTracker(3)
        bars = [b15(i, *x) for i, x in enumerate(SEQ)]
        for b in bars[:6]:
            sw.add(b)
        self.assertEqual(sw.confirmed(bars[5].end, "high"), [])
        sw.add(bars[6])
        highs = sw.confirmed(bars[6].end, "high")
        self.assertEqual([(s.price, s.pivot_index) for s in highs], [(110, 3)])
        self.assertEqual(highs[0].confirmed_ts, bars[6].end)
        self.assertEqual(sw.confirmed(bars[6].end - timedelta(seconds=1), "high"), [])

    def test_equal_highs_resolve_to_earlier_bar(self):
        from model.order_blocks.swings import SwingTracker
        sw = SwingTracker(1)
        for i, h in enumerate([1, 5, 5, 1, 1]):
            sw.add(b15(i, 1, h, 0.5, 1))
        self.assertEqual([s.pivot_index for s in sw.swings if s.kind == "high"], [1])

    def test_bos_then_choch(self):
        from model.order_blocks.params import ObParams
        _bars, _sw, st, _a, _r, brk = _structure(ObParams(atr_n=3))
        self.assertEqual((brk.direction, brk.kind, brk.bar_index), ("bullish", "BOS", 8))
        self.assertEqual(st.trend, "up")
        # wick through without close does nothing
        from model.order_blocks.structure import StructureTracker
        from model.order_blocks.types import Swing
        s = StructureTracker()
        s.add_swings([Swing("low", 90.0, T0, 0, T0)])
        s.trend = "up"
        self.assertIsNone(s.on_bar(b15(1, 95, 96, 85, 91), 1))
        x = s.on_bar(b15(2, 91, 92, 88, 89), 2)
        self.assertEqual((x.direction, x.kind), ("bearish", "CHOCH"))


class TestZone(unittest.TestCase):
    def test_bullish_zone_from_last_bearish_candle_at_origin(self):
        from model.order_blocks.detect import attach_fvg, build_zone
        from model.order_blocks.params import ObParams
        p = ObParams(atr_n=3, max_zone_atr=5.0)
        bars, sw, _st, atr, rv, brk = _structure(p)
        z, why = build_zone(brk, bars, atr.values, rv.values, sw, p)
        self.assertEqual(why, "")
        self.assertEqual(z.source_bar_ts, bars[6].ts)
        self.assertEqual((z.zone_low, z.zone_high), (95, 99))
        self.assertEqual(z.first_eligible_ts, bars[8].end)
        self.assertEqual(z.leg_origin, 95)
        self.assertIsNone(z.rvol)                    # no volume → N/A, not a fail
        bars.append(b15(9, 112, 114, 111, 113))
        self.assertTrue(attach_fvg(z, bars))
        # gaps: bar6.high 99 → bar8.low 103 (4) and bar7.high 104 → bar9.low 111 (7);
        # the largest is kept
        self.assertEqual((z.fvg_low, z.fvg_high), (104, 111))

    def test_wide_zone_trimmed_then_rejected(self):
        from model.order_blocks.detect import build_zone
        from model.order_blocks.params import ObParams
        p = ObParams(atr_n=3)
        bars, sw, _st, atr, rv, brk = _structure(p)
        atr_b = atr.values[8]
        p_trim = ObParams(atr_n=3, max_zone_atr=2.5 / atr_b)   # 4 wide > 2.5, body top 97 → 2 wide
        z, _ = build_zone(brk, bars, atr.values, rv.values, sw, p_trim)
        self.assertEqual((z.zone_low, z.zone_high), (95, 97))
        p_rej = ObParams(atr_n=3, max_zone_atr=1.0 / atr_b)
        z, why = build_zone(brk, bars, atr.values, rv.values, sw, p_rej)
        self.assertIsNone(z)
        self.assertIn("wide", why)

    def test_rejects_weak_displacement_and_long_leg(self):
        from model.order_blocks.detect import build_zone
        from model.order_blocks.params import ObParams
        bars, sw, _st, atr, rv, brk = _structure(ObParams(atr_n=3))
        z, why = build_zone(brk, bars, atr.values, rv.values, sw,
                            ObParams(atr_n=3, disp_body_atr=9, disp_range_atr=9))
        self.assertIsNone(z)
        self.assertIn("displacement", why)
        z, why = build_zone(brk, bars, atr.values, rv.values, sw,
                            ObParams(atr_n=3, max_leg_bars=2))
        self.assertIn("leg", why)

    def test_volume_gate(self):
        from model.order_blocks.detect import build_zone
        from model.order_blocks.params import ObParams
        p = ObParams(atr_n=3, max_zone_atr=5.0)
        bars, sw, _st, atr, _rv, brk = _structure(p)
        low = [0.9] * len(bars)
        z, why = build_zone(brk, bars, atr.values, low, sw, p)
        self.assertIn("rvol", why)
        high = [1.5] * len(bars)
        z, _ = build_zone(brk, bars, atr.values, high, sw, p)
        self.assertEqual(z.rvol, 1.5)


class TestLifecycle(unittest.TestCase):
    def _zone(self, **kw):
        from model.order_blocks.types import Zone
        base = dict(zone_id="z", series="S", timeframe="15m", direction="bullish",
                    source_bar_ts=T0, bos_bar_ts=T0, first_eligible_ts=T0 + timedelta(minutes=15),
                    zone_low=95, zone_high=99, broken_swing=110, broken_swing_ts=T0,
                    leg_origin=95, atr_at_bos=5, disp_body_atr=1.5, disp_range_atr=2,
                    rvol=None)
        base.update(kw)
        return Zone(**base)

    def test_not_evaluated_on_break_bar(self):
        from model.order_blocks.lifecycle import step
        from model.order_blocks.params import ObParams
        z = self._zone()
        self.assertEqual(step(z, b15(0, 100, 101, 90, 100), ObParams()), [])
        self.assertEqual(z.status, "ACTIVE")

    def test_touch_then_invalidate_on_close_through(self):
        from model.order_blocks.lifecycle import step
        from model.order_blocks.params import ObParams
        z = self._zone()
        ev = step(z, b15(1, 102, 103, 98, 101), ObParams())
        self.assertEqual([e.kind for e in ev], ["zone_touch"])
        ev = step(z, b15(2, 100, 100, 93, 96), ObParams())     # wick through, close inside
        self.assertEqual(z.status, "TOUCHED")
        ev = step(z, b15(3, 96, 97, 92, 94), ObParams())
        self.assertEqual(z.status, "INVALIDATED")

    def test_expiry_by_age_and_session_cutoff(self):
        from model.order_blocks.lifecycle import step
        from model.order_blocks.params import ObParams
        z = self._zone()
        p = ObParams(zone_age_bars=2)
        step(z, b15(1, 105, 106, 104, 105), p)
        step(z, b15(2, 105, 106, 104, 105), p)
        self.assertEqual((z.status, z.close_reason), ("EXPIRED", "age 2 bars"))
        z2 = self._zone()
        step(z2, b15(23, 105, 106, 104, 105), ObParams())        # 15:00 bar, ends 15:15
        self.assertTrue(z2.live)
        step(z2, b15(24, 105, 106, 104, 105), ObParams())        # 15:15 bar, ends 15:30
        self.assertEqual(z2.close_reason, "session cutoff")

    def test_supersede_overlapping(self):
        from model.order_blocks.lifecycle import ZoneBook
        from model.order_blocks.params import ObParams
        book = ZoneBook(ObParams())
        old = self._zone(zone_id="a")
        book.add(old, T0)
        book.add(self._zone(zone_id="b", zone_low=96, zone_high=100), T0)
        self.assertEqual((old.status, old.close_reason), ("EXPIRED", "superseded"))


class TestScoreAndDecision(unittest.TestCase):
    def test_component_formulas(self):
        from model.order_blocks.params import ObParams
        from model.order_blocks.score import score_zone
        from model.order_blocks.types import Zone
        z = Zone("z", "S", "15m", "bullish", T0, T0, T0, 95, 97.5, 110, T0, 95, 5.0,
                 1.3, 2.0, 1.5, kind="BOS", fvg_low=96, fvg_high=98, swept_level=94.0)
        z.touch_bars_after_bos = 3
        s = score_zone(z, "up", ObParams())
        c = s.components
        self.assertEqual(c["structure"], 20.0)
        self.assertAlmostEqual(c["displacement"], 10.0)        # (1.3-0.8)/1.0 * 20
        self.assertAlmostEqual(c["volume"], 7.5)               # (1.5-1)/1 * 15
        self.assertEqual((c["sweep"], c["fvg"], c["htf"], c["freshness"]), (15, 10, 10, 5))
        self.assertAlmostEqual(c["tightness"], 5 * (1 - 0.5) / 0.7, places=2)
        self.assertAlmostEqual(s.total, sum(c.values()), places=1)

    def test_choch_and_missing_volume(self):
        from model.order_blocks.params import ObParams
        from model.order_blocks.score import score_zone
        from model.order_blocks.types import Zone
        z = Zone("z", "S", "15m", "bearish", T0, T0, T0, 95, 99, 90, T0, 99, 5.0,
                 0.5, 1.0, None, kind="CHOCH")
        s = score_zone(z, "up", ObParams())
        self.assertEqual(s.components["structure"], 12.0)
        self.assertEqual(s.components["volume"], 0.0)
        self.assertIn("volume", s.na)
        self.assertEqual(s.components["htf"], 0.0)

    def test_decision_bands_and_shadow(self):
        from model.order_blocks.params import ObParams
        from model.order_blocks.score import GateInputs, decide, gates
        p = ObParams()
        ok = gates(GateInputs(u_rr=2.0))
        self.assertEqual(decide(55, ok, "intraday", None, p)[0], "NO-GO")
        self.assertEqual(decide(70, ok, "intraday", None, p)[0], "WATCH")
        self.assertEqual(decide(80, ok, "intraday", None, p)[0], "GO")
        self.assertEqual(decide(80, ok, "overnight", False, p)[0], "SHADOW")
        self.assertEqual(decide(80, ok, "overnight", True, p)[0], "GO")
        bad = gates(GateInputs(u_rr=1.0, killed=True))
        d, reasons = decide(90, bad, "intraday", None, p)
        self.assertEqual(d, "NO-GO")
        self.assertTrue(any("Kill" in r for r in reasons))
        self.assertTrue(any("R:R" in r for r in reasons))

    def test_stale_data_gate(self):
        from model.order_blocks.score import GateInputs, gates
        chk = {c.name: c.status.value for c in gates(GateInputs(u_rr=2, spot_age_sec=90))}
        self.assertEqual(chk["Data fresh"], "fail")
        chk = {c.name: c.status.value for c in gates(GateInputs(u_rr=2))}
        self.assertEqual(chk["Data fresh"], "n/a")


class TestPlanAndExits(unittest.TestCase):
    def test_plan_targets_nearest_unbroken_swing(self):
        from model.order_blocks.params import ObParams
        from model.order_blocks.triggers import build_plan
        from model.order_blocks.types import Swing, Zone
        z = Zone("z", "S", "15m", "bullish", T0, T0, T0, 95, 99, 110, T0, 95, 5.0, 1.5, 2, None)
        swings = [Swing("high", 120, T0, 0, T0), Swing("high", 115, T0, 1, T0),
                  Swing("high", 99, T0, 2, T0)]
        plan = build_plan(z, "intraday", 100.0, 10.0, swings, (130, 90), ObParams())
        self.assertEqual((plan.u_stop, plan.u_target, plan.target_source), (94.0, 115, "swing"))
        plan = build_plan(z, "intraday", 100.0, 10.0, [], None, ObParams())
        self.assertEqual((plan.u_target, plan.target_source), (112.0, "default_r"))

    def test_exit_stop_before_target_in_same_bar(self):
        from model.order_blocks.exits import check_exit
        bar = Bar(datetime(2026, 10, 6, 11, 0), "1m", 100, 120, 80, 100)
        sig = check_exit(direction="bullish", horizon="intraday", u_stop=90, u_target=110,
                         bar=bar, opened_on=bar.ts.date())
        self.assertEqual((sig.reason, sig.u_price), ("u_stop", 90))

    def test_overnight_gap_fills_at_open(self):
        from model.order_blocks.exits import check_exit
        bar = Bar(datetime(2026, 10, 7, 9, 15), "1m", 85, 88, 84, 87)
        sig = check_exit(direction="bullish", horizon="overnight", u_stop=90, u_target=110,
                         bar=bar, opened_on=datetime(2026, 10, 6).date())
        self.assertEqual((sig.reason, sig.u_price), ("gap", 85))
        same_day = Bar(datetime(2026, 10, 6, 15, 25), "1m", 85, 88, 84, 87)
        self.assertIsNone(check_exit(direction="bullish", horizon="overnight", u_stop=90,
                                     u_target=110, bar=same_day,
                                     opened_on=datetime(2026, 10, 6).date()))

    def test_time_exits(self):
        from model.order_blocks.exits import check_exit
        intr = Bar(datetime(2026, 10, 6, 15, 14), "1m", 100, 101, 99, 100)
        self.assertEqual(check_exit(direction="bullish", horizon="intraday", u_stop=90,
                                    u_target=110, bar=intr, opened_on=intr.ts.date()).reason, "time")
        exp = Bar(datetime(2026, 10, 13, 14, 59), "1m", 100, 101, 99, 100)
        self.assertEqual(check_exit(direction="bullish", horizon="intraday", u_stop=90,
                                    u_target=110, bar=exp, opened_on=exp.ts.date(),
                                    expiry="2026-10-13").reason, "expiry_guard")
        on = Bar(datetime(2026, 10, 7, 10, 29), "1m", 100, 101, 99, 100)
        self.assertEqual(check_exit(direction="bullish", horizon="overnight", u_stop=90,
                                    u_target=110, bar=on,
                                    opened_on=datetime(2026, 10, 6).date()).reason, "time")


class TestEngineNoLookAhead(unittest.TestCase):
    """Appending future bars must never change anything already decided."""

    @classmethod
    def setUpClass(cls):
        from model.order_blocks.params import ObParams
        cls.params = ObParams(rvol_min=1.0)
        cls.bars = random_walk_bars(sessions=20)
        cls.full = cls._run(cls.bars)

    @classmethod
    def _run(cls, bars):
        from model.order_blocks.engine import ObEngine
        eng = ObEngine(cls.params)
        zones, setups = {}, []
        for bar, c in bars:
            for ev in eng.on_minute(bar, c):
                if ev.kind == "zone_new":
                    z = ev.zone
                    zones[z.zone_id] = (z.zone_low, z.zone_high, z.bos_bar_ts,
                                        z.first_eligible_ts, z.source_bar_ts, z.kind, ev.ts)
                if ev.kind == "setup":
                    s = ev.setup
                    setups.append((s.zone.zone_id, s.trigger_ts, s.horizon,
                                   s.plan.u_entry, s.plan.u_stop))
        return zones, setups, eng

    def test_engine_finds_zones(self):
        zones, setups, _ = self.full
        self.assertGreater(len(zones), 0)

    def test_prefix_runs_agree_with_full_run(self):
        full_zones, full_setups, _ = self.full
        for cut in (len(self.bars) // 3, len(self.bars) // 2, (len(self.bars) * 4) // 5):
            prefix = self.bars[:cut]
            end = prefix[-1][0].end
            zones, setups, _ = self._run(prefix)
            for zid, attrs in zones.items():
                self.assertEqual(full_zones.get(zid), attrs, f"zone {zid} changed after cut {cut}")
                self.assertLessEqual(attrs[3], end)          # eligible no later than known
            for zid, attrs in full_zones.items():
                if attrs[6] <= end:
                    self.assertIn(zid, zones, "zone known by cut missing from prefix run")
            self.assertEqual(setups, [s for s in full_setups if s[1] <= end])

    def test_zone_never_eligible_before_break_close(self):
        full_zones, _, _ = self.full
        for _low, _high, bos, eligible, src, _kind, created in full_zones.values():
            self.assertGreater(eligible, bos)
            self.assertEqual(created, eligible)               # created at the BOS bar's close
            self.assertLessEqual(src, bos)

    def test_replay_is_idempotent(self):
        from model.order_blocks.engine import ObEngine
        eng = ObEngine(self.params)
        for bar, c in self.bars[:500]:
            eng.on_minute(bar, c)
        before = dict(eng.counters)
        for bar, c in self.bars[:500]:
            self.assertEqual(eng.on_minute(bar, c), [])
        self.assertEqual(eng.counters["zones"], before["zones"])

    def test_params_fingerprint_stable_and_grid_size(self):
        from model.order_blocks.params import ObParams, grid
        self.assertEqual(ObParams().fingerprint(), ObParams().fingerprint())
        self.assertNotEqual(ObParams().fingerprint(), ObParams(pivot_k=2).fingerprint())
        self.assertEqual(len(grid()), 81)


if __name__ == "__main__":
    unittest.main()


class TestAvailableAt(unittest.TestCase):
    """Higher timeframes come only from completed 1m bars; nothing acts early."""

    @classmethod
    def setUpClass(cls):
        cls.bars = random_walk_bars(sessions=6, seed=31)

    def test_resampled_bars_match_independent_resample(self):
        import pandas as pd

        from model.order_blocks.engine import ObEngine
        eng = ObEngine()
        for bar, c in self.bars:
            eng.on_minute(bar, c)
        m1 = pd.DataFrame([{"ts": b.ts, "open": b.open, "high": b.high, "low": b.low,
                            "close": b.close, "volume": b.volume} for b, _ in self.bars]).set_index("ts")
        for tf, rule in (("5m", "5min"), ("15m", "15min"), ("60m", "60min")):
            ref = []
            for _day, g in m1.groupby(m1.index.date):
                r = g.resample(rule, origin=g.index[0].replace(hour=9, minute=15)).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last",
                     "volume": "sum"}).dropna(subset=["close"])
                close = g.index[0].replace(hour=15, minute=30)
                ref.append(r[r.index + pd.Timedelta(rule) <= close])     # no partial bins
            ref = pd.concat(ref)
            got = eng.tf[tf].bars
            self.assertEqual(len(got), len(ref), tf)
            for b, (ts, row) in zip(got, ref.iterrows(), strict=True):
                self.assertEqual(b.ts, ts.to_pydatetime())
                self.assertAlmostEqual(b.high, row["high"])
                self.assertAlmostEqual(b.low, row["low"])
                self.assertAlmostEqual(b.close, row["close"])
                self.assertEqual(b.volume, int(row["volume"]))

    def test_setups_never_precede_their_data(self):
        from model.order_blocks.engine import ObEngine
        from model.order_blocks.params import ObParams
        eng = ObEngine(ObParams(rvol_min=1.0))
        n = 0
        for bar, c in self.bars:
            for ev in eng.on_minute(bar, c):
                if ev.kind == "setup":
                    n += 1
                    s = ev.setup
                    self.assertEqual(s.available_at, bar.end)        # the settling 1m bar
                    self.assertGreaterEqual(s.available_at, s.trigger_ts)
                    self.assertLessEqual(s.zone.first_eligible_ts, s.trigger_ts)
        self.assertGreater(n, 0)

    def test_backtest_refuses_fill_before_available(self):
        from model.order_blocks.backtest import Trade, _Book, simulate
        from model.order_blocks.types import LookAheadError
        t0 = datetime(2026, 10, 6, 9, 15)
        bars = [(Bar(t0 + timedelta(minutes=i), "1m", 100, 101, 99, 100), "") for i in range(60)]
        t = Trade("OB", "intraday", "bullish", "2026-10-06", t0 + timedelta(minutes=10),
                  t0 + timedelta(minutes=10), 100, 95, 110)
        t.available_at = t0 + timedelta(minutes=20)
        with self.assertRaises(LookAheadError):
            simulate(t, _Book(bars), 0.0)

    def test_decision_gate_fails_on_early_clock(self):
        from execution.governor import BookState
        from model.order_blocks.params import ObParams
        from model.order_blocks.score import score_zone
        from model.order_blocks.types import Setup, TradePlan, Zone
        from services.order_blocks import Context, evaluate
        now = datetime(2026, 10, 6, 11, 0)
        z = Zone("z", "S", "15m", "bullish", now, now, now, 24950, 24990, 25100, now, 24940,
                 40, 1.2, 1.8, 1.4)
        s = Setup(z, "intraday", now, TradePlan("bullish", "intraday", 25000, 24940, 25150, "swing"),
                  score_zone(z, "up", ObParams()), "up", 40, available_at=now + timedelta(seconds=30))
        ev = evaluate(s, Context(now, 25000, 13.0), chains={}, lot_size_for=lambda x: 65,
                      book=BookState(equity=1e6), expected_lot=65)
        gate = next(c for c in ev.checks if c.name == "Decision after data")
        self.assertEqual(gate.status.value, "fail")
        self.assertEqual(ev.decision, "NO-GO")
        self.assertEqual(ev.signal.available_at, "2026-10-06T11:00:30")
