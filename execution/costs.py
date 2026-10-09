"""Transaction-cost model for NIFTY options (per executed order).

Rates change. These are pinned with `costs_as_of`; verify them against
Zerodha's brokerage calculator before trusting any P&L, and bump the date.
`model/options_ev.estimate_fees_per_lot` is the older engine's estimate
(0.125% STT); the order-block system uses this model only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    costs_as_of: str = "2025-10 (verify)"
    brokerage_per_order: float = 20.0          # flat, per executed order
    stt_sell_pct: float = 0.001                # 0.1% of premium, sell side
    exchange_txn_pct: float = 0.0003503        # NSE options, % of premium
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.00003             # 0.003% of premium, buy side
    gst_pct: float = 0.18                      # on brokerage + txn + SEBI

    def charges(self, side: str, price: float, qty: int) -> float:
        """Charges for one executed order (one fill slice)."""
        turnover = abs(price) * qty
        brokerage = self.brokerage_per_order if qty > 0 else 0.0
        txn = turnover * self.exchange_txn_pct
        sebi = turnover * self.sebi_per_crore / 1e7
        stt = turnover * self.stt_sell_pct if side == "SELL" else 0.0
        stamp = turnover * self.stamp_buy_pct if side == "BUY" else 0.0
        gst = (brokerage + txn + sebi) * self.gst_pct
        return round(brokerage + txn + sebi + stt + stamp + gst, 2)

    def round_trip(self, entry: float, exit_: float, qty: int, *, long: bool = True) -> float:
        first, second = ("BUY", "SELL") if long else ("SELL", "BUY")
        return round(self.charges(first, entry, qty) + self.charges(second, exit_, qty), 2)


DEFAULT_COSTS = CostModel()
