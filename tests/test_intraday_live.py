"""Unit tests for the all-day intraday runner. Fully offline: fake clock,
fake frames, stub evaluator, no sleeping."""
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from services.intraday import run_confluence_live, session_state

IST = ZoneInfo("Asia/Kolkata")


def dt(day="2026-09-09", hm="10:00"):
    return datetime.strptime(f"{day} {hm}", "%Y-%m-%d %H:%M").replace(tzinfo=IST)


def frame(last_bar):
    idx = pd.DatetimeIndex([last_bar - timedelta(minutes=5), last_bar])
    return pd.DataFrame({"close": [100.0, 101.0]}, index=idx)


def report(verdicts=("GO", "NO-GO", "WATCH")):
    return SimpleNamespace(
        timestamp="2026-09-09T10:00:00", spot=25000.0, error=None,
        setups=[SimpleNamespace(setup_id=s, decision=d)
                for s, d in zip("ABC", verdicts, strict=True)])


class FakeClock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now

    def sleep(self, secs):
        self.now += timedelta(seconds=secs)


class TestSessionState(unittest.TestCase):
    def test_states(self):
        self.assertEqual(session_state(dt(hm="08:00")), "pre")
        self.assertEqual(session_state(dt(hm="09:15")), "open")
        self.assertEqual(session_state(dt(hm="15:30")), "open")
        self.assertEqual(session_state(dt(hm="15:31")), "closed")
        self.assertEqual(session_state(dt(day="2026-09-12", hm="10:00")), "weekend")

    def test_naive_treated_as_ist(self):
        naive = datetime(2026, 9, 9, 10, 0)
        self.assertEqual(session_state(naive), "open")


class TestLiveLoop(unittest.TestCase):
    def _run(self, start="10:00", until="10:10", journal=True, **kw):
        clock = FakeClock(dt(hm=start))
        calls = {"fetch": 0, "evals": []}

        def fetch_frame():
            calls["fetch"] += 1
            floored = clock().replace(second=0, microsecond=0)
            floored -= timedelta(minutes=(floored.minute % 5 or 5))
            return frame(floored)

        def evaluate(df):
            calls["evals"].append(df.index[-1])
            return report()

        got = []
        summary = run_confluence_live(
            journal=journal, poll_secs=60, until=until,
            fetch_frame=fetch_frame, evaluate=evaluate,
            clock=clock, sleep=clock.sleep,
            on_report=lambda r, j: got.append((r, j)), **kw)
        return summary, calls, got

    def test_evaluates_each_new_bar_and_journals(self):
        summary, calls, got = self._run()
        self.assertEqual(summary["status"], "done")
        self.assertFalse(summary["interrupted"])
        # bars 09:55 (at 10:00), 10:00 (at 10:05), 10:05 (at 10:09)
        self.assertEqual(summary["evals"], 3)
        self.assertEqual(summary["journaled"], 3)
        self.assertEqual(summary["errors"], 0)
        self.assertTrue(all(j for _, j in got))
        self.assertGreater(calls["fetch"], summary["evals"])  # polls > evals

    def test_dry_run_counts_zero_journaled(self):
        summary, _, got = self._run(journal=False)
        self.assertEqual(summary["evals"], 3)
        self.assertEqual(summary["journaled"], 0)
        self.assertTrue(all(not j for _, j in got))

    def test_same_bar_not_reevaluated(self):
        clock = FakeClock(dt(hm="10:00"))
        evals = []
        df = frame(dt(hm="09:55"))
        summary = run_confluence_live(
            poll_secs=60, until="10:03",
            fetch_frame=lambda: df,
            evaluate=lambda d: (evals.append(d), report())[1],
            clock=clock, sleep=clock.sleep)
        self.assertEqual(summary["evals"], 1)
        self.assertEqual(len(evals), 1)

    def test_fetch_error_does_not_kill_day(self):
        clock = FakeClock(dt(hm="10:00"))
        state = {"n": 0}
        errors = []

        def fetch_frame():
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("kite hiccup")
            floored = clock().replace(second=0, microsecond=0)
            floored -= timedelta(minutes=(floored.minute % 5 or 5))
            return frame(floored)

        summary = run_confluence_live(
            poll_secs=60, until="10:10", fetch_frame=fetch_frame,
            evaluate=lambda df: report(), clock=clock, sleep=clock.sleep,
            on_error=errors.append)
        self.assertEqual(summary["errors"], 1)
        self.assertGreaterEqual(summary["evals"], 1)
        self.assertEqual(len(errors), 1)

    def test_weekend_exits_immediately(self):
        clock = FakeClock(dt(day="2026-09-12", hm="10:00"))
        summary = run_confluence_live(
            fetch_frame=lambda: (_ for _ in ()).throw(AssertionError("no fetch")),
            evaluate=lambda df: (_ for _ in ()).throw(AssertionError("no eval")),
            clock=clock, sleep=clock.sleep)
        self.assertEqual(summary["status"], "market closed (weekend)")
        self.assertEqual(summary["evals"], 0)

    def test_after_close_exits(self):
        clock = FakeClock(dt(hm="16:00"))
        summary = run_confluence_live(
            fetch_frame=lambda: (_ for _ in ()).throw(AssertionError("no fetch")),
            evaluate=lambda df: (_ for _ in ()).throw(AssertionError("no eval")),
            clock=clock, sleep=clock.sleep)
        self.assertEqual(summary["evals"], 0)
        self.assertEqual(summary["status"], "done")

    def test_bad_timeframe_rejected(self):
        with self.assertRaises(ValueError):
            run_confluence_live(timeframe="15m", fetch_frame=lambda: None,
                                evaluate=lambda df: None)

    def test_keyboard_interrupt_is_graceful(self):
        clock = FakeClock(dt(hm="10:00"))

        def sleep(secs):
            raise KeyboardInterrupt

        summary = run_confluence_live(
            poll_secs=60, until="10:10",
            fetch_frame=lambda: frame(dt(hm="09:55")),
            evaluate=lambda df: report(), clock=clock, sleep=sleep)
        self.assertTrue(summary["interrupted"])
        self.assertEqual(summary["status"], "done")


class TestUntilValidation(unittest.TestCase):
    def test_validate_until(self):
        from model_cli import _validate_until
        _validate_until("15:35")
        _validate_until("9:15")
        for bad in ("25:00", "1535", "ab:cd", "15:35:00", ""):
            with self.assertRaises(ValueError, msg=bad):
                _validate_until(bad)


if __name__ == "__main__":
    unittest.main()
