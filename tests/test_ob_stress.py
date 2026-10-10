"""Overnight options stress engine (model/order_blocks/stress.py)."""

import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _leg(strike, is_call, dte, spot=25000.0, iv=0.13, spread=1.0):
    from model.options_ev import bs_price
    from model.order_blocks.contract import LegQuote
    p = bs_price(spot, strike, dte, iv, is_call)
    return LegQuote(f"N{strike}{'CE' if is_call else 'PE'}", strike, is_call, "2026-10-13", p,
                    p - spread / 2, p + spread / 2, iv=iv * 100)


def _run(legs, sides, now, direction="bullish", vix=13.0):
    from model.order_blocks.stress import stress_overnight
    entry = sum(s * (q.ask if s > 0 else q.bid) for q, s in zip(legs, sides, strict=True))
    return stress_overnight(legs, sides, entry_net=entry, spot=25000.0, direction=direction,
                            vix=vix, now=now, expiry="2026-10-13", lot=65)


class TestStress(unittest.TestCase):
    MON = datetime(2026, 10, 5, 15, 20)
    FRI = datetime(2026, 10, 9, 15, 20)

    def test_named_scenarios_rank_sensibly_for_a_long_call(self):
        rep = _run([_leg(25000, True, 8.0)], [1], self.MON)
        by = {s.name: s.pnl_lot for s in rep.scenarios}
        self.assertGreater(by["favourable gap, IV falls"], by["flat open, IV falls"])
        self.assertGreater(by["flat open, IV falls"], by["adverse gap, IV rises"])
        self.assertGreater(by["adverse gap, IV rises"], by["tail gap against"])
        self.assertLess(by["flat open, IV unchanged"], 0)          # decay + opening spread
        self.assertEqual(rep.stress_scenario, "tail gap against")
        self.assertLessEqual(rep.worst_lot, -rep.stress_loss_lot + 1e-6)
        self.assertGreater(rep.breakeven_move_points, 0)
        self.assertLess(rep.expected_loss_lot, 0)

    def test_weekend_hold_costs_more(self):
        mon = _run([_leg(25000, True, 8.0)], [1], self.MON)
        fri = _run([_leg(25000, True, 4.0)], [1], self.FRI)
        flat = "flat open, IV unchanged"
        m = next(s.pnl_lot for s in mon.scenarios if s.name == flat)
        f = next(s.pnl_lot for s in fri.scenarios if s.name == flat)
        self.assertLess(f, m)
        self.assertGreater(fri.exit_dte, 0)

    def test_bearish_put_mirrors(self):
        rep = _run([_leg(25000, False, 8.0)], [1], self.MON, direction="bearish")
        by = {s.name: s for s in rep.scenarios}
        self.assertLess(by["favourable gap, IV falls"].move_points, 0)   # favourable = down
        self.assertGreater(by["favourable gap, IV falls"].pnl_lot, by["tail gap against"].pnl_lot)
        self.assertLess(rep.breakeven_move_points, 0)

    def test_spread_loss_bounded_by_width(self):
        legs = [_leg(25000, True, 8.0), _leg(25200, True, 8.0)]
        rep = _run(legs, [1, -1], self.MON)
        debit = legs[0].ask - legs[1].bid
        self.assertLessEqual(rep.stress_loss_unit, debit + 2 * 1.0 + 1e-6)  # + widened spreads

    def test_governor_sizes_overnight_on_stress_and_reports_it(self):
        from execution.governor import BookState, RiskGovernor
        from model.forecast.options_edge import Leg, Structure
        from model.order_blocks.contract import ContractChoice
        from model.order_blocks.types import ScoreCard, Setup, TradePlan, Zone
        q = _leg(25000, True, 8.0)
        q.lot_size = 65
        st = Structure("long_call", "long_premium", (Leg(25000, True, 1, q.ask, 0.13),), 8.0,
                       "2026-10-13")
        choice = ContractChoice(st, None, [q], [1], 65, round(q.ask, 2), None, None, 0.5)
        z = Zone("z", "S", "15m", "bullish", self.MON, self.MON, self.MON, 24950, 24990, 25100,
                 self.MON, 24940, 40, 1.2, 1.8, 1.4)
        setup = Setup(z, "overnight", self.MON,
                      TradePlan("bullish", "overnight", 25000, 24940, 25150, "swing"),
                      ScoreCard(80.0, {}), "up", 40)
        dec = RiskGovernor().size(setup, choice, BookState(equity=10_000_000), 25000.0, self.MON,
                                  vix=13.0, hold_days=1.0, dte_days=8.0)
        self.assertIsNotNone(dec.stress)
        self.assertGreaterEqual(dec.stress_loss_per_lot + 1e-6, dec.stress["stress_loss_lot"] - 200)
        self.assertEqual(len(dec.stress["scenarios"]), 7)


if __name__ == "__main__":
    unittest.main()
