"""Performance analytics for the per-stock overnight journal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from journal.stock_overnight_db import (
    StockOvernightJournal,
    shared_stock_overnight_journal,
)


@dataclass
class StockOvernightPerformance:
    total: int = 0
    go: int = 0
    nogo: int = 0
    settled: int = 0
    win_rate: float | None = None
    net_pnl: float = 0.0
    profit_factor: float | None = None
    avoided_losses: float = 0.0
    missed_winners: float = 0.0
    by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_direction: dict[str, dict[str, Any]] = field(default_factory=dict)


def compute_stock_overnight_performance(
        journal: StockOvernightJournal | None = None) -> StockOvernightPerformance:
    j = journal or shared_stock_overnight_journal()
    recs = j.list(limit=100000)
    p = StockOvernightPerformance(total=len(recs))
    wins: list[float] = []
    gross_win, gross_loss = 0.0, 0.0
    for r in recs:
        if r.decision == "GO":
            p.go += 1
        else:
            p.nogo += 1
        pnl = r.actual_pnl if r.is_actual_trade else r.hypothetical_pnl
        if pnl is None:
            continue
        p.settled += 1
        if r.is_actual_trade:
            p.net_pnl += pnl
            wins.append(1.0 if pnl > 0 else 0.0)
            if pnl > 0:
                gross_win += pnl
            else:
                gross_loss += abs(pnl)
        else:
            # NO-GO hypotheticals: what did standing aside avoid / miss?
            if pnl < 0:
                p.avoided_losses += abs(pnl)
            else:
                p.missed_winners += pnl
        for bucket, key in ((p.by_symbol, r.symbol),
                            (p.by_direction, r.direction)):
            st = bucket.setdefault(key, {"trades": 0, "settled": 0, "net": 0.0})
            st["trades"] += 1
            st["settled"] += 1
            st["net"] = round(st["net"] + pnl, 2)
    if wins:
        p.win_rate = round(sum(wins) / len(wins), 3)
    if gross_loss > 0:
        p.profit_factor = round(gross_win / gross_loss, 2)
    elif gross_win > 0:
        p.profit_factor = float("inf")
    p.net_pnl = round(p.net_pnl, 2)
    p.avoided_losses = round(p.avoided_losses, 2)
    p.missed_winners = round(p.missed_winners, 2)
    return p
