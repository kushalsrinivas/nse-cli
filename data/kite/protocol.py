"""Kite WebSocket binary protocol parser (own code, per official spec).

Message: u16 packet-count, then per packet (u16 length + payload).
Payload lengths select the layout: 8B ltp, 28B index-quote, 32B index-full,
44B quote, 184B full. A 1-byte message is a heartbeat. All integers are
big-endian; prices are paise (/100). Index packets order day stats as
HIGH, LOW, OPEN, CLOSE (unlike tradeable packets).

Unknown payload lengths are skipped (boundary known from the prefix) and
counted, so spec drift degrades to a counter, not a crash. Truncated
messages raise ProtocolError.
"""

from __future__ import annotations

import logging
import struct
from datetime import datetime, timezone

log = logging.getLogger(__name__)

HEARTBEAT = b"\x00"  # 1-byte keepalive (content varies; length is the signal)

LEN_LTP = 8
LEN_INDEX_QUOTE = 28
LEN_INDEX_FULL = 32
LEN_QUOTE = 44
LEN_FULL = 184

_DEPTH_ENTRY = struct.Struct(">i i h 2x")  # qty, price, orders, pad
_DEPTH_COUNT = 10


class ProtocolError(RuntimeError):
    """Structurally invalid frame (truncation, bad counts)."""


def _price(raw: int) -> float:
    return round(raw / 100.0, 2)


def _ts(raw: int):
    if not raw:
        return None
    try:
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def parse_depth(payload: bytes, offset: int) -> tuple[dict, int]:
    bids, offers = [], []
    pos = offset
    for i in range(_DEPTH_COUNT):
        if pos + _DEPTH_ENTRY.size > len(payload):
            raise ProtocolError(f"depth entry {i} truncated")
        qty, price, orders = _DEPTH_ENTRY.unpack_from(payload, pos)
        pos += _DEPTH_ENTRY.size
        entry = {"quantity": qty, "price": _price(price), "orders": orders}
        (bids if i < 5 else offers).append(entry)
    return {"buy": bids, "sell": offers}, pos


def parse_packet(payload: bytes) -> dict:
    """One quote packet -> normalized tick dict."""
    n = len(payload)
    if n == LEN_LTP:
        (token, ltp) = struct.unpack(">i i", payload)
        return {"token": token, "mode": "ltp", "ltp": _price(ltp)}
    if n in (LEN_INDEX_QUOTE, LEN_INDEX_FULL):
        vals = struct.unpack(">i i i i i i i", payload[:28])
        token, ltp, high, low, op, close, change = vals
        tick = {"token": token,
                "mode": "index_full" if n == LEN_INDEX_FULL else "index_quote",
                "ltp": _price(ltp), "high": _price(high), "low": _price(low),
                "open": _price(op), "close": _price(close),
                "change": _price(change)}
        if n == LEN_INDEX_FULL:
            (ets,) = struct.unpack(">i", payload[28:32])
            tick["exchange_ts"] = _ts(ets)
        return tick
    if n in (LEN_QUOTE, LEN_FULL):
        head = struct.unpack(">i i i i i i i i i i i", payload[:44])
        (token, ltp, ltq, avg, vol, buy_q, sell_q,
         op, high, low, close) = head
        tick = {"token": token,
                "mode": "full" if n == LEN_FULL else "quote",
                "ltp": _price(ltp), "last_qty": ltq, "avg": _price(avg),
                "volume": vol, "buy_qty": buy_q, "sell_qty": sell_q,
                "open": _price(op), "high": _price(high), "low": _price(low),
                "close": _price(close)}
        if n == LEN_FULL:
            ltt, oi, oi_h, oi_l, ets = struct.unpack(">i i i i i", payload[44:64])
            tick["last_trade_ts"] = _ts(ltt)
            tick["oi"] = oi or None
            tick["oi_high"] = oi_h or None
            tick["oi_low"] = oi_l or None
            tick["exchange_ts"] = _ts(ets)
            tick["depth"], _ = parse_depth(payload, 64)
        return tick
    raise ProtocolError(f"unknown packet length {n}")


def parse_message(data: bytes) -> tuple[list[dict], dict]:
    """Binary frame -> (ticks, stats). Stats counts heartbeats/skips."""
    stats = {"heartbeats": 0, "packets": 0, "skipped": 0}
    if len(data) == 1:
        stats["heartbeats"] = 1
        return [], stats
    if len(data) < 2:
        raise ProtocolError(f"frame too short: {len(data)} bytes")
    (count,) = struct.unpack(">h", data[:2])
    if count < 0 or count > 3000:
        raise ProtocolError(f"implausible packet count {count}")
    ticks: list[dict] = []
    pos = 2
    for _ in range(count):
        if pos + 2 > len(data):
            raise ProtocolError("packet length prefix truncated")
        (length,) = struct.unpack(">h", data[pos:pos + 2])
        pos += 2
        if length <= 0 or pos + length > len(data):
            raise ProtocolError(f"packet overruns frame (len {length})")
        payload = data[pos:pos + length]
        pos += length
        try:
            ticks.append(parse_packet(payload))
            stats["packets"] += 1
        except ProtocolError as exc:
            stats["skipped"] += 1
            log.warning("skipping WS packet: %s", exc)
    return ticks, stats
