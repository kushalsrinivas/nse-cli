"""Transaction costs per segment and product (Zerodha-style schedule).

Rates live in `[costs]` of the platform config with an `as_of` label;
verify them against the broker's calculator and bump `as_of` when they
change — every run records the config hash, so old results keep the rates
they were computed with.

    equity delivery (CNC)  brokerage 0;  STT 0.1% both sides;  stamp 0.015% buy
    equity intraday (MIS)  brokerage min(₹20, 0.03%);  STT 0.025% sell;  stamp 0.003% buy
    futures (NRML/MIS)     brokerage min(₹20, 0.03%);  STT 0.02% sell;   stamp 0.002% buy
    options                brokerage ₹20 flat;  STT 0.1% of premium sell;  stamp 0.003% buy
    all                    exchange txn (segment rate), SEBI ₹10/crore, GST 18% on
                           brokerage + txn + SEBI

DP charges on delivery sells (a fixed ₹/scrip) are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    txn: float
    sebi: float
    stamp: float
    gst: float

    @property
    def total(self) -> float:
        return round(self.brokerage + self.stt + self.txn + self.sebi + self.stamp + self.gst, 2)


class SegmentCosts:
    def __init__(self, costs_cfg) -> None:
        self.c = costs_cfg

    def breakdown(self, segment: str, product: str, side: str, price: float, qty: int) -> CostBreakdown:
        c = self.c
        turnover = abs(price) * abs(qty)
        buy, sell = side == "BUY", side == "SELL"
        if qty == 0:
            return CostBreakdown(0, 0, 0, 0, 0, 0)
        if segment == "equity":
            if product == "CNC":
                brokerage = 0.0
                stt = turnover * c.stt_equity_delivery
                stamp = turnover * c.stamp_equity_delivery if buy else 0.0
            else:
                brokerage = min(c.brokerage_flat, turnover * c.brokerage_pct_cap)
                stt = turnover * c.stt_equity_intraday_sell if sell else 0.0
                stamp = turnover * c.stamp_equity_intraday if buy else 0.0
            txn = turnover * c.txn_equity
        elif segment == "futures":
            brokerage = min(c.brokerage_flat, turnover * c.brokerage_pct_cap)
            stt = turnover * c.stt_futures_sell if sell else 0.0
            stamp = turnover * c.stamp_futures if buy else 0.0
            txn = turnover * c.txn_futures
        elif segment == "options":
            brokerage = c.brokerage_flat
            stt = turnover * c.stt_options_sell if sell else 0.0
            stamp = turnover * c.stamp_options if buy else 0.0
            txn = turnover * c.txn_options
        else:
            raise ValueError(f"unknown segment {segment!r}")
        sebi = turnover * c.sebi_per_crore / 1e7
        gst = (brokerage + txn + sebi) * c.gst
        return CostBreakdown(round(brokerage, 4), round(stt, 4), round(txn, 4), round(sebi, 4),
                             round(stamp, 4), round(gst, 4))

    def charges(self, segment: str, product: str, side: str, price: float, qty: int) -> float:
        return self.breakdown(segment, product, side, price, qty).total

    def round_trip(self, segment: str, product: str, entry: float, exit_: float, qty: int, *,
                   long: bool = True) -> float:
        a, b = ("BUY", "SELL") if long else ("SELL", "BUY")
        return round(self.charges(segment, product, a, entry, qty)
                     + self.charges(segment, product, b, exit_, qty), 2)
