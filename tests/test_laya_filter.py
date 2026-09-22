"""Laya filter: state text, veto policy, shadow/enforce, journal + eval.

Network- and torch-free: a fake agent stands in for the Laya checkpoint.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from journal.laya_db import LayaJournal, evaluate_rows
from model.confluence.types import (
    ConditionCheck,
    ConditionStatus,
    ConfluenceReport,
    ConfluenceSetupResult,
)
from model.laya_filter import (
    SETUP_QUESTIONS,
    LayaFilter,
    LayaUnavailable,
    VetoPolicy,
    apply_verdicts,
    setup_state,
)
from model.overnight_setups.types import (
    OvernightSetupResult,
    SetupDecision,
    make_condition,
)
from services.laya import judge_confluence


def _answers(p_fail=0.2, p_exec=0.7, p_conflict=0.2, direction="bullish",
             dir_conf=0.8, quality=2.0):
    return {
        "direction": {"type": "choice", "choice": direction, "confidence": dir_conf},
        "market_regime": {"type": "choice", "choice": "trending", "confidence": 0.6},
        "setup_quality": {"type": "score", "score": quality, "confidence": 0.5},
        "likely_to_fail": {"type": "noul", "noul": p_fail},
        "warrants_execution": {"type": "noul", "noul": p_exec},
        "conflicting_signal": {"type": "noul", "noul": p_conflict},
    }


class FakeAgent:
    def __init__(self, per_setup=None, default=None, raise_for=()):
        self.per_setup = per_setup or {}
        self.default = default or _answers()
        self.raise_for = raise_for
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        for sid, ans in self.per_setup.items():
            if f"setup {sid}:" in state:
                return {"answers": ans}
        for sid in self.raise_for:
            if f"setup {sid}:" in state:
                raise RuntimeError("boom")
        return {"answers": self.default}


def _cf(setup_id="A", decision="GO", direction="bullish"):
    return ConfluenceSetupResult(
        setup_id=setup_id, title="CPR + VWAP trend", decision=decision,
        direction=direction, confidence_score=100.0,
        conditions=[ConditionCheck("15m close vs CPR top + VWAP",
                                   ConditionStatus.PASS, "close 24,610 > 24,580"),
                    ConditionCheck("RSI band", ConditionStatus.FAIL, "RSI 78")],
        decision_rationale="All conditions met")


def _report(setups):
    return ConfluenceReport(run_id="CF-TEST-1", timestamp="2026-09-23T10:00:00",
                            trade_date="2026-09-23", spot=24600.0, setups=setups,
                            vix_level=13.2, vix_change=9.0)


class TestState(unittest.TestCase):
    def test_confluence_state_reads_as_words(self):
        st = setup_state(_cf(), source="confluence",
                         context={"spot": 24600.0, "vix": 13.2, "vix_change": 9.0,
                                  "events": ["RBI policy"]})
        text = st.as_text()
        self.assertIn("confluence setup A", text)
        self.assertIn("Proposed direction: bullish", text)
        self.assertIn("RSI band: NOT met (RSI 78)", text)
        self.assertIn("volatility spiking", text)
        self.assertIn("RBI policy", text)

    def test_overnight_result_enum_decision(self):
        r = OvernightSetupResult(
            setup_id="ON-A", name="trend hold", direction="bearish",
            decision=SetupDecision.GO, confidence=80.0,
            conditions=[make_condition("vol regime sane (VIX<22)", True, "VIX 14")])
        st = setup_state(r, source="overnight")
        self.assertEqual(st.decision, "GO")
        self.assertEqual(st.title, "trend hold")
        self.assertIn("vol regime sane (VIX<22): met (VIX 14)", st.as_text())


class TestPolicy(unittest.TestCase):
    def test_quiet_answers_do_not_veto(self):
        self.assertEqual(VetoPolicy().reasons(_answers(), "bullish"), [])

    def test_each_rule_trips(self):
        pol = VetoPolicy(min_quality=1.0)
        self.assertTrue(pol.reasons(_answers(p_fail=0.9), "bullish"))
        self.assertTrue(pol.reasons(_answers(p_exec=0.1), "bullish"))
        self.assertTrue(pol.reasons(_answers(p_conflict=0.9), "bullish"))
        self.assertTrue(pol.reasons(_answers(direction="bearish", dir_conf=0.9), "bullish"))
        self.assertTrue(pol.reasons(_answers(quality=0.5), "bullish"))

    def test_weak_opposite_read_does_not_veto(self):
        self.assertEqual(VetoPolicy().reasons(
            _answers(direction="bearish", dir_conf=0.5), "bullish"), [])

    def test_neutral_setup_has_no_direction_rule(self):
        self.assertEqual(VetoPolicy().reasons(
            _answers(direction="bearish", dir_conf=0.99), "neutral"), [])


class TestFilter(unittest.TestCase):
    def test_one_pass_all_questions(self):
        agent = FakeAgent()
        LayaFilter(agent).judge(setup_state(_cf(), source="confluence"))
        self.assertEqual(len(agent.calls), 1)
        self.assertIs(agent.calls[0][1], SETUP_QUESTIONS)

    def test_only_go_can_be_vetoed(self):
        agent = FakeAgent(default=_answers(p_fail=0.95))
        flt = LayaFilter(agent)
        go = flt.judge(setup_state(_cf(decision="GO"), source="confluence"))
        nogo = flt.judge(setup_state(_cf(decision="NO-GO"), source="confluence"))
        self.assertTrue(go.veto)
        self.assertTrue(nogo.would_veto)
        self.assertFalse(nogo.veto)

    def test_inference_error_never_vetoes(self):
        flt = LayaFilter(FakeAgent(raise_for=("A",)))
        v = flt.judge(setup_state(_cf(), source="confluence"))
        self.assertEqual(v.error, "boom")
        self.assertFalse(v.veto)

    def test_missing_laya_raises_unavailable(self):
        import builtins
        real = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "laya":
                raise ImportError("no laya")
            return real(name, *a, **kw)

        builtins.__import__ = fake_import
        try:
            with self.assertRaises(LayaUnavailable):
                _ = LayaFilter().agent
        finally:
            builtins.__import__ = real


class TestApply(unittest.TestCase):
    def setUp(self):
        self.flt = LayaFilter(FakeAgent(default=_answers(p_fail=0.95)))

    def test_shadow_changes_nothing(self):
        results = [_cf()]
        v = self.flt.judge_many([setup_state(results[0], source="confluence")])
        out = apply_verdicts(results, v, enforce=False)
        self.assertEqual(out[0].decision, "GO")

    def test_enforce_downgrades_copy_not_original(self):
        orig = _cf()
        v = self.flt.judge_many([setup_state(orig, source="confluence")])
        out = apply_verdicts([orig], v, enforce=True)
        self.assertEqual(out[0].decision, "WATCH")
        self.assertTrue(out[0].blocked_reasons[0].startswith("laya veto"))
        self.assertEqual(orig.decision, "GO")

    def test_enforce_keeps_enum_type(self):
        r = OvernightSetupResult(setup_id="ON-A", name="x", direction="bullish",
                                 decision=SetupDecision.GO, confidence=90.0)
        v = self.flt.judge_many([setup_state(r, source="overnight")])
        out = apply_verdicts([r], v, enforce=True)
        self.assertIs(out[0].decision, SetupDecision.WATCH)
        self.assertTrue(out[0].rationale.startswith("laya veto"))


class TestServiceAndJournal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "j.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_judge_confluence_journals_every_verdict(self):
        agent = FakeAgent(per_setup={"A": _answers(p_fail=0.95)})
        report = _report([_cf("A"), _cf("B", decision="NO-GO"), _cf("C")])
        lj = LayaJournal(self.db)
        run = judge_confluence(report, enforce=True, journal=lj,
                               laya_filter=LayaFilter(agent))
        self.assertEqual([s.decision for s in run.report.setups],
                         ["WATCH", "NO-GO", "GO"])
        self.assertEqual(report.setups[0].decision, "GO")   # original untouched
        self.assertEqual(lj.count(), 3)

    def test_unavailable_is_a_notice(self):
        class Broken(LayaFilter):
            @property
            def agent(self):
                raise LayaUnavailable("not installed")
        report = _report([_cf()])
        run = judge_confluence(report, laya_filter=Broken())
        self.assertIs(run.report, report)
        self.assertTrue(any("unavailable" in m for _, m in run.notices))

    def test_eval_joins_settled_outcomes(self):
        from journal.confluence_db import ConfluenceJournal, ConfluenceRunRecord
        cj = ConfluenceJournal(self.db)
        lj = LayaJournal(self.db)
        flt = LayaFilter(FakeAgent(per_setup={"A": _answers(p_fail=0.95),
                                              "B": _answers(p_fail=0.1)}))
        for sid, outcome, pnl in (("A", "LOSS", -500.0), ("B", "WIN", 800.0)):
            rec = cj.add(ConfluenceRunRecord(
                id=None, run_id="CF-TEST-1", setup_id=sid,
                timestamp="2026-09-23T10:00:00", trade_date="2026-09-23",
                nifty_spot=24600.0, direction="bullish", decision="GO",
                confidence_score=100.0))
            rec.outcome, rec.hypothetical_pnl = outcome, pnl
            cj.update(rec)
            v = flt.judge(setup_state(_cf(sid), source="confluence"))
            lj.add("CF-TEST-1", v)
        res = evaluate_rows(lj.settled_confluence())
        self.assertEqual(res["n"], 2)
        self.assertEqual(res["vetoed"].n, 1)
        self.assertEqual(res["vetoed"].mean_pnl, -500.0)
        self.assertEqual(res["auc_p_fail"], 1.0)
        self.assertFalse(res["conclusive"])


def _night(go=True):
    from types import SimpleNamespace as NS
    return NS(
        composite=NS(direction=NS(value="bearish"), score=82.0),
        conditions=NS(close_location=NS(value="strong breakdown"), with_trend=True),
        matched_bucket="high conviction", hist_n=37, hist_win_rate_open=0.54,
        hist_avg_gap_pct=0.09, hist_p10_gap=-0.73, hist_p90_gap=0.46,
        chosen_strategy=NS(candidate=NS(name="ATM PE 23350"),
                           net_ev_per_lot=964.0, p_profitable=0.38),
        breadth=None, divergence_flags=[], scenarios=None,
        overnight_setups=NS(results=[OvernightSetupResult(
            setup_id="ON-A", name="Broad Trend Hold", direction="bearish",
            decision=SetupDecision.GO, confidence=90.0, rationale="8/8 green")]),
        reasons=[] if go else ["expiry day -> no new entries"],
        assessments=[], regime=NS(label="TRENDING BEAR", adx=31.0),
        spot=23329.0, go=go, verdict="GO" if go else "NO-GO")


def _premarket(trade=True):
    from types import SimpleNamespace as NS

    from model.forecast.decision import RiskView
    dec = NS(
        market=NS(regime_label="trending down", headline="UP 0.52% expected",
                  evidence=["gap model: +0.52%"]),
        execution=NS(note="expected open 23,450", trigger="work it"),
        instrument=NS(best=NS(structure=NS(name="Iron butterfly"), edge=597.0),
                      ranked=[]),
        trade=NS(has_edge=trade, blocking=[]),
        risk=RiskView(allowed=trade), invalidation=["above 23,489"],
        why=["Volatility: FAIR"], confidence="moderate",
        action="TRADE — Iron butterfly" if trade else "NO TRADE")
    gap = NS(available=True, confidence="high", expected_pct=0.52)
    return NS(decision=dec, gap=gap, spot=23329.0)


class TestCards(unittest.TestCase):
    def test_overnight_state(self):
        from model.laya_filter.cards import overnight_card_state
        st = overnight_card_state(_night(), events=["FOMC"], vix=13.1)
        text = st.as_text()
        self.assertEqual((st.source, st.decision, st.direction),
                         ("overnight", "GO", "bearish"))
        self.assertIn("ATM PE 23350, net EV ₹+964/lot", text)
        self.assertIn("ON-A Broad Trend Hold: met (8/8 green)", text)
        self.assertIn("FOMC", text)

    def test_premarket_state(self):
        from model.laya_filter.cards import premarket_state
        st = premarket_state(_premarket())
        self.assertEqual((st.decision, st.direction), ("GO", "bullish"))
        self.assertIn("Iron butterfly, edge ₹+597", st.as_text())
        self.assertEqual(premarket_state(_premarket(trade=False)).decision, "NO-GO")

    def test_veto_overnight_turns_go_to_nogo(self):
        from model.laya_filter.cards import overnight_card_state, veto_overnight
        night = _night()
        v = LayaFilter(FakeAgent(default=_answers(p_fail=0.9, direction="bearish"))
                       ).judge(overnight_card_state(night))
        veto_overnight(night, v)
        self.assertFalse(night.go)
        self.assertTrue(night.reasons[-1].startswith("laya veto"))

    def test_veto_premarket_kills_trade(self):
        from model.laya_filter.cards import premarket_state, veto_premarket
        out = _premarket()
        v = LayaFilter(FakeAgent(default=_answers(p_exec=0.1))).judge(premarket_state(out))
        veto_premarket(out, v)
        self.assertFalse(out.decision.trade.has_edge)
        self.assertFalse(out.decision.risk.allowed)
        self.assertIsNone(out.decision.instrument.best)

    def test_judge_card_shadow_vs_enforce(self):
        from model.laya_filter.cards import overnight_card_state, veto_overnight
        from services.laya import judge_card
        flt = LayaFilter(FakeAgent(default=_answers(p_fail=0.9, direction="bearish")))
        with tempfile.TemporaryDirectory() as d:
            lj = LayaJournal(Path(d) / "j.db")
            shadow = _night()
            run = judge_card(overnight_card_state(shadow), journal=lj, run_id="ON-x",
                             apply_veto=lambda v: veto_overnight(shadow, v),
                             laya_filter=flt)
            self.assertTrue(shadow.go)
            self.assertTrue(run.verdicts[0].veto)
            hard = _night()
            judge_card(overnight_card_state(hard), enforce=True, journal=lj,
                       run_id="ON-x", apply_veto=lambda v: veto_overnight(hard, v),
                       laya_filter=flt)
            self.assertFalse(hard.go)
            self.assertEqual(lj.count(), 2)

    def test_judge_card_nogo_is_never_touched(self):
        from model.laya_filter.cards import overnight_card_state, veto_overnight
        from services.laya import judge_card
        night = _night(go=False)
        before = list(night.reasons)
        judge_card(overnight_card_state(night), enforce=True, run_id="ON-x",
                   apply_veto=lambda v: veto_overnight(night, v),
                   laya_filter=LayaFilter(FakeAgent(default=_answers(p_fail=0.99))))
        self.assertEqual(night.reasons, before)


if __name__ == "__main__":
    unittest.main()
