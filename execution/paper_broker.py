"""PaperBroker: kiteconnect-shaped order API with simulated, book-based fills.

`place_order` takes KiteConnect.place_order's keyword arguments, plus three
bookkeeping keywords (`signal_id`, `purpose`, `leg_index`) the journal needs
for idempotency. A replay of the same (tag, purpose, leg_index) returns the
existing order id and never fills twice.

Fill model (docs §8.2), all against a `Book` supplied by `book_source`:

1. MARKET buy walks the ask levels (sell side of depth); sells walk bids.
   Quantity left after the visible levels fills at the worst level + 1 tick.
2. A book older than `max_book_age_sec` fills the whole order at top of
   book + 1 tick adverse (`stale_book`) — still a fill, never a free one.
3. No book at all → REJECTED. There is no last-traded-price fallback: a
   price nobody quoted is not a price anyone could have traded at.
4. LIMIT fills when the top of the opposite side is at or through the
   limit; otherwise it rests OPEN until `poll()` sees a marketable book.
5. SL / SL-M rest OPEN and trigger on `poll()` when LTP crosses the
   trigger, then fill as MARKET (SL respects its limit).
6. Orders above `freeze_qty` are sliced; each slice is a separately
   charged executed order, as an exchange iceberg would be.
7. When the kill switch is engaged only exits are accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from execution import kill_switch
from execution.costs import DEFAULT_COSTS, CostModel
from journal.ob_db import FillRecord, ObJournal, OrderRecord, new_id

log = logging.getLogger(__name__)

EXIT_PURPOSES = ("stop", "target", "time_exit", "gap_exit", "manual", "kill")
#: NIFTY options freeze quantity (units per order). Exchange-set and
#: revised with lot size; verify against the current NSE circular.
DEFAULT_FREEZE_QTY = 1800


@dataclass
class Book:
    tradingsymbol: str
    bid: float | None
    ask: float | None
    bid_qty: int | None = None
    ask_qty: int | None = None
    bids: list[tuple[float, int]] = field(default_factory=list)   # best first
    asks: list[tuple[float, int]] = field(default_factory=list)
    ltp: float | None = None
    age_sec: float | None = None
    source: str = "live"            # live | archived | candle

    def side(self, transaction_type: str) -> list[tuple[float, int]]:
        """Levels a BUY lifts (asks) or a SELL hits (bids)."""
        levels = self.asks if transaction_type == "BUY" else self.bids
        if levels:
            return [(p, q) for p, q in levels if p and p > 0 and q and q > 0]
        top = (self.ask, self.ask_qty) if transaction_type == "BUY" else (self.bid, self.bid_qty)
        if top[0] and top[0] > 0:
            return [(top[0], top[1] or 0)]
        return []

    @classmethod
    def from_depth(cls, tradingsymbol: str, depth: dict | None, ltp=None,
                   age_sec=None, source: str = "live") -> Book:
        def lv(side):
            return [(float(x.get("price") or 0), int(x.get("quantity") or 0))
                    for x in (depth or {}).get(side, [])
                    if (x.get("price") or 0) > 0 and (x.get("quantity") or 0) > 0]
        bids, asks = lv("buy"), lv("sell")
        return cls(tradingsymbol,
                   bids[0][0] if bids else None, asks[0][0] if asks else None,
                   bids[0][1] if bids else None, asks[0][1] if asks else None,
                   bids, asks, ltp, age_sec, source)


class OrderRejected(RuntimeError):
    pass


class PaperBroker:
    def __init__(self, journal: ObJournal, book_source, *,
                 costs: CostModel = DEFAULT_COSTS, clock=datetime.now,
                 tick: float = 0.05, max_book_age_sec: float = 2.0,
                 freeze_qty: int = DEFAULT_FREEZE_QTY, run_id: str = "live") -> None:
        self.journal = journal
        self.book_source = book_source
        self.costs = costs
        self.clock = clock
        self.tick = tick
        self.max_book_age_sec = max_book_age_sec
        self.freeze_qty = freeze_qty
        self.run_id = run_id

    # -- kiteconnect-shaped API ----------------------------------------------

    def place_order(self, *, variety: str = "regular", exchange: str = "NFO",
                    tradingsymbol: str, transaction_type: str, quantity: int,
                    product: str, order_type: str, price: float | None = None,
                    trigger_price: float | None = None, tag: str | None = None,
                    signal_id: str = "", purpose: str = "entry", leg_index: int = 0,
                    lot_size: int | None = None) -> str:
        now = self.clock().isoformat(timespec="seconds")
        tag = (tag or signal_id[:12] or new_id("T"))[:20]
        rec = OrderRecord(
            order_id=new_id("PO"), signal_id=signal_id, tag=tag, leg_index=leg_index,
            tradingsymbol=tradingsymbol, transaction_type=transaction_type.upper(),
            product=product, order_type=order_type, quantity=int(quantity),
            purpose=purpose, status="OPEN", placed_at=now, updated_at=now,
            exchange=exchange, price=price, trigger_price=trigger_price)
        stored, inserted = self.journal.add_order(rec)
        if not inserted:
            self.journal.log_event("dup", {"tag": tag, "purpose": purpose,
                                           "leg_index": leg_index}, run_id=self.run_id,
                                   ref_id=stored.order_id, ts=now)
            return stored.order_id

        reject = self._validate(rec, lot_size)
        if reject:
            return self._reject(rec, reject)
        if order_type in ("MARKET", "LIMIT"):
            self._try_fill(rec)
        self.journal.log_event("order", {"order_id": rec.order_id, "status": rec.status,
                                         "purpose": purpose, "symbol": tradingsymbol,
                                         "side": rec.transaction_type, "qty": quantity},
                               run_id=self.run_id, ref_id=signal_id, ts=now)
        return rec.order_id

    def cancel_order(self, variety: str, order_id: str) -> str:
        for o in self.journal.orders(status="OPEN"):
            if o.order_id == order_id:
                o.status = "CANCELLED"
                self.journal.update_order(o)
        return order_id

    def orders(self) -> list[dict]:
        return [o.__dict__ for o in self.journal.orders()]

    def positions(self) -> dict:
        """Net units and average price per symbol, from fills."""
        net: dict[str, dict] = {}
        for o in self.journal.orders(status="COMPLETE"):
            sign = 1 if o.transaction_type == "BUY" else -1
            d = net.setdefault(o.tradingsymbol, {"quantity": 0, "value": 0.0})
            d["quantity"] += sign * o.filled_qty
            d["value"] += sign * o.filled_qty * (o.avg_price or 0.0)
        return {"net": [{"tradingsymbol": s, **v} for s, v in net.items() if v["quantity"]]}

    # -- resting orders --------------------------------------------------------

    def poll(self) -> list[str]:
        """Re-evaluate OPEN orders against fresh books. Returns filled ids."""
        done = []
        for o in self.journal.orders(status="OPEN"):
            if o.order_type in ("SL", "SL-M"):
                book = self.book_source(o.tradingsymbol)
                ltp = book.ltp if book else None
                if ltp is None or o.trigger_price is None:
                    continue
                hit = ltp >= o.trigger_price if o.transaction_type == "BUY" else ltp <= o.trigger_price
                if not hit:
                    continue
                if o.order_type == "SL-M":
                    o.order_type = "MARKET"
                else:
                    o.order_type = "LIMIT"
            if self._try_fill(o):
                done.append(o.order_id)
        return done

    # -- internals ---------------------------------------------------------------

    def _validate(self, rec: OrderRecord, lot_size: int | None) -> str:
        if rec.quantity <= 0:
            return "quantity must be positive"
        if rec.purpose == "entry" and not lot_size:
            return "entry without a contract lot size — refusing to size (data/lots.py)"
        if lot_size and rec.quantity % lot_size:
            return f"quantity {rec.quantity} not a multiple of lot {lot_size}"
        if rec.transaction_type not in ("BUY", "SELL"):
            return f"bad transaction_type {rec.transaction_type}"
        if rec.order_type in ("LIMIT", "SL") and rec.price is None:
            return f"{rec.order_type} needs a price"
        if rec.order_type in ("SL", "SL-M") and rec.trigger_price is None:
            return f"{rec.order_type} needs a trigger_price"
        if kill_switch.is_engaged() and rec.purpose not in EXIT_PURPOSES:
            return f"kill switch engaged ({kill_switch.reason()})"
        return ""

    def _reject(self, rec: OrderRecord, why: str) -> str:
        rec.status = "REJECTED"
        rec.status_message = why
        self.journal.update_order(rec)
        self.journal.log_event("order_reject", {"order_id": rec.order_id, "why": why},
                               run_id=self.run_id, ref_id=rec.signal_id)
        return rec.order_id

    def _try_fill(self, rec: OrderRecord) -> bool:
        book = self.book_source(rec.tradingsymbol)
        if book is None or not book.side(rec.transaction_type):
            if rec.order_type == "MARKET":
                self._reject(rec, "no executable book")
            return False
        levels = book.side(rec.transaction_type)
        buy = rec.transaction_type == "BUY"
        if rec.order_type == "LIMIT":
            top = levels[0][0]
            marketable = top <= rec.price if buy else top >= rec.price
            if not marketable:
                return False
        fills = self._walk(rec.quantity, levels, buy, book)
        model = fills[0][2]
        if rec.purpose == "gap_exit":
            model = "gap_open"
        filled_at = self.clock().isoformat(timespec="seconds")
        total_qty, total_val = 0, 0.0
        for qty, px, _m in fills:
            for slice_qty in self._slices(qty):
                charges = self.costs.charges(rec.transaction_type, px, slice_qty)
                self.journal.add_fill(FillRecord(
                    new_id("F"), rec.order_id, filled_at, slice_qty, round(px, 2),
                    model, charges, book.bid, book.ask, book.age_sec))
                total_qty += slice_qty
                total_val += slice_qty * px
        rec.filled_qty = total_qty
        rec.avg_price = round(total_val / total_qty, 4) if total_qty else None
        rec.status = "COMPLETE"
        self.journal.update_order(rec)
        return True

    def _walk(self, qty: int, levels: list[tuple[float, int]], buy: bool,
              book: Book) -> list[tuple[int, float, str]]:
        adverse = self.tick if buy else -self.tick
        stale = book.age_sec is not None and book.age_sec > self.max_book_age_sec
        if stale:
            return [(qty, levels[0][0] + adverse, "stale_book")]
        out, left = [], qty
        for px, avail in levels:
            if left <= 0:
                break
            # Unknown depth (archived top-of-book without qty) fills at the
            # quoted level; known depth is consumed level by level.
            take = min(left, avail) if avail else left
            if take:
                out.append((take, px, "touch" if px == levels[0][0] else "walk_depth"))
                left -= take
        if left > 0:
            worst = (out[-1][1] if out else levels[0][0]) + adverse
            out.append((left, worst, "walk_depth"))
        if len(out) > 1:
            out = [(q, p, "walk_depth") for q, p, _ in out]
        return out

    def _slices(self, qty: int) -> list[int]:
        if not self.freeze_qty or qty <= self.freeze_qty:
            return [qty]
        out = [self.freeze_qty] * (qty // self.freeze_qty)
        if qty % self.freeze_qty:
            out.append(qty % self.freeze_qty)
        return out

    def order_charges(self, order_id: str) -> float:
        return round(sum(f.charges for f in self.journal.fills(order_id)), 2)
