"""Lot size single source of truth (data/lots.py)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _store(rows):
    from data.kite.store import InstrumentStore, normalize_dump_row
    st = InstrumentStore(os.path.join(tempfile.mkdtemp(), "s.db"))
    for as_of, raw in rows:
        st.upsert([normalize_dump_row(raw, as_of)])
    return st


def _fut(sym, expiry, lot, token=1):
    return {"instrument_token": token, "exchange": "NFO", "tradingsymbol": sym, "name": "NIFTY",
            "expiry": expiry, "lot_size": lot, "instrument_type": "FUT", "segment": "NFO-FUT"}


class TestLots(unittest.TestCase):
    def test_nifty_lot_from_master_and_history(self):
        from data.lots import nifty_lot
        st = _store([("2025-11-01", _fut("NIFTY25DECFUT", "2025-12-30", 75)),
                     ("2026-01-02", _fut("NIFTY25DECFUT", "2025-12-30", 65)),
                     ("2026-10-01", _fut("NIFTY26OCTFUT", "2026-10-27", 65, 2))])
        self.assertEqual(nifty_lot("2025-12-01", store=st).lot, 75)
        self.assertEqual(nifty_lot("2026-10-05", store=st).lot, 65)

    def test_config_mismatch_message(self):
        from data.lots import check_config_lot
        st = _store([("2026-10-01", _fut("NIFTY26OCTFUT", "2026-10-27", 65))])
        msg = check_config_lot(store=st, config_lot=75)
        self.assertIn("lot_size: int = 65", msg)
        self.assertEqual(check_config_lot(store=st, config_lot=65), "")
        empty = _store([])
        self.assertEqual(check_config_lot(store=empty, config_lot=75), "")   # no master yet

    def test_require_lot_never_falls_back(self):
        from data.lots import LotSizeError, require_lot
        st = _store([])
        with self.assertRaises(LotSizeError):
            require_lot("NIFTY26OCT25000CE", store=st)

    def test_settlement_uses_history_for_trade_date(self):
        from data.lots import lot_for_settlement
        st = _store([("2025-11-01", _fut("NIFTY25DECFUT", "2025-12-30", 75))])
        self.assertEqual(lot_for_settlement("2025-12-01", store=st), 75)

    def test_broker_rejects_entry_without_lot(self):
        from datetime import datetime

        from execution.paper_broker import Book, PaperBroker
        from journal.ob_db import ObJournal
        j = ObJournal(os.path.join(tempfile.mkdtemp(), "j.db"))
        b = PaperBroker(j, lambda s: Book(s, 100, 101, 650, 650), clock=lambda: datetime(2026, 10, 6, 10))
        oid = b.place_order(tradingsymbol="X", transaction_type="BUY", quantity=65, product="MIS",
                            order_type="MARKET", signal_id="S", purpose="entry")
        o = j.orders()[0]
        self.assertEqual((o.order_id, o.status), (oid, "REJECTED"))
        self.assertIn("lot size", o.status_message)


class TestCliGuard(unittest.TestCase):
    def test_sizing_commands_are_guarded(self):
        import model_cli
        for cmd in ("tonight", "overnight", "premarket", "ob", "ob-paper", "ob-backtest"):
            self.assertIn(cmd, model_cli.LOT_SIZED_COMMANDS)
        for cmd in ("kite-login", "kite-master", "ob-audit", "ob-record", "ob-backfill"):
            self.assertNotIn(cmd, model_cli.LOT_SIZED_COMMANDS)


if __name__ == "__main__":
    unittest.main()
