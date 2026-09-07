"""Protocol tests: frames built byte-for-byte from the spec table."""
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.kite import protocol as P


def pack_full(token=408065, ltp=141295, ltq=5, avg=141247, vol=7360198,
              buy_q=0, sell_q=5191, op=139600, high=142175, low=139555,
              close=138965, ltt=1711234567, oi=0, oi_h=0, oi_l=0,
              ets=1711234570):
    head = struct.pack(">i i i i i i i i i i i", token, ltp, ltq, avg, vol,
                       buy_q, sell_q, op, high, low, close)
    tail = struct.pack(">i i i i i", ltt, oi, oi_h, oi_l, ets)
    depth = b""
    for i in range(10):
        depth += struct.pack(">i i h 2x", 100 + i, 141200 + i, 3)
    return head + tail + depth


def pack_index(token=256265, ltp=2380000, high=2385000, low=2370000,
               op=2375000, close=2378000, change=2000, ets=1711234570,
               full=True):
    body = struct.pack(">i i i i i i i", token, ltp, high, low, op, close, change)
    return body + (struct.pack(">i", ets) if full else b"")


def frame(*payloads):
    out = struct.pack(">h", len(payloads))
    for p in payloads:
        out += struct.pack(">h", len(p)) + p
    return out


class TestProtocol(unittest.TestCase):
    def test_heartbeat(self):
        ticks, stats = P.parse_message(b"\x00")
        self.assertEqual((ticks, stats["heartbeats"]), ([], 1))

    def test_ltp_packet(self):
        payload = struct.pack(">i i", 408065, 141295)
        self.assertEqual(len(payload), 8)
        t = P.parse_packet(payload)
        self.assertEqual((t["token"], t["mode"], t["ltp"]), (408065, "ltp", 1412.95))

    def test_full_packet(self):
        t = P.parse_packet(pack_full())
        self.assertEqual(t["mode"], "full")
        self.assertEqual(t["ltp"], 1412.95)
        self.assertEqual(t["avg"], 1412.47)
        self.assertEqual(t["volume"], 7360198)
        self.assertEqual((t["open"], t["high"], t["low"], t["close"]),
                         (1396.0, 1421.75, 1395.55, 1389.65))
        self.assertIsNone(t["oi"])  # equities report 0 -> absent
        self.assertEqual(len(t["depth"]["buy"]), 5)
        self.assertEqual(len(t["depth"]["sell"]), 5)
        self.assertEqual(t["depth"]["buy"][0]["quantity"], 100)
        self.assertEqual(t["depth"]["buy"][0]["price"], 1412.0)
        self.assertIsNotNone(t["exchange_ts"])
        self.assertIsNotNone(t["last_trade_ts"])

    def test_full_packet_oi(self):
        t = P.parse_packet(pack_full(oi=150000, oi_h=160000, oi_l=140000))
        self.assertEqual((t["oi"], t["oi_high"], t["oi_low"]),
                         (150000, 160000, 140000))

    def test_quote_packet_no_depth(self):
        t = P.parse_packet(pack_full()[:44])
        self.assertEqual(t["mode"], "quote")
        self.assertNotIn("depth", t)
        self.assertNotIn("oi", t)
        self.assertEqual(t["ltp"], 1412.95)

    def test_index_packets(self):
        q = P.parse_packet(pack_index(full=False))
        self.assertEqual(len(pack_index(full=False)), 28)
        self.assertEqual(q["mode"], "index_quote")
        # index order is H/L/O/C — must not be scrambled
        self.assertEqual((q["high"], q["low"], q["open"], q["close"]),
                         (23850.0, 23700.0, 23750.0, 23780.0))
        self.assertEqual(q["change"], 20.0)
        self.assertNotIn("exchange_ts", q)
        f = P.parse_packet(pack_index(full=True))
        self.assertEqual(f["mode"], "index_full")
        self.assertIsNotNone(f["exchange_ts"])

    def test_multi_packet_message(self):
        msg = frame(pack_full(), pack_index(), struct.pack(">i i", 1, 100))
        ticks, stats = P.parse_message(msg)
        self.assertEqual(stats["packets"], 3)
        self.assertEqual([t["mode"] for t in ticks], ["full", "index_full", "ltp"])

    def test_unknown_length_skipped(self):
        msg = frame(b"\x00" * 17, pack_full())
        ticks, stats = P.parse_message(msg)
        self.assertEqual((stats["skipped"], stats["packets"]), (1, 1))
        self.assertEqual(ticks[0]["mode"], "full")

    def test_truncation_raises(self):
        with self.assertRaises(P.ProtocolError):
            P.parse_message(b"")
        with self.assertRaises(P.ProtocolError):
            P.parse_message(frame(pack_full())[:-10])
        with self.assertRaises(P.ProtocolError):
            P.parse_message(struct.pack(">h", 9999))

    def test_bad_timestamp_safe(self):
        t = P.parse_packet(pack_full(ltt=0, ets=2**31 - 1))
        self.assertIsNone(t["last_trade_ts"])


if __name__ == "__main__":
    unittest.main()
