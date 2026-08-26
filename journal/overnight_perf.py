"""Analytics & Performance Engine for Overnight Trade Journal.

Computes live statistics from `overnight_trade_journal`, comparing:
1. Actual GO execution performance (Win Rate, Net P&L, Profit Factor, Best/Worst).
2. Counterfactual NO-GO Filter Efficiency (Avoided Losses vs. Missed Winners).
3. Dimensional Breakdowns (Direction, Regime, Confidence Score Bins).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from journal.overnight_db import OvernightJournal, OvernightRunRecord, shared_overnight_journal


@dataclass(frozen=True)
class OvernightPerformanceSummary:
    total_runs: int
    go_count: int
    nogo_count: int
    actual_trades: int
    settled_actual: int
    settled_hypothetical: int

    # Actual GO Performance
    go_wins: int
    go_losses: int
    go_scratches: int
    go_win_rate: float
    go_net_pnl: float
    go_avg_pnl: float
    go_profit_factor: float
    go_best_trade: float
    go_worst_trade: float

    # NO-GO Filter Counterfactual Analytics
    avoided_losses_count: int
    avoided_losses_rupees: float
    missed_winners_count: int
    missed_winners_rupees: float
    filter_efficiency_pct: float

    # Breakdowns
    by_direction: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_regime: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_score_tier: dict[str, dict[str, Any]] = field(default_factory=dict)


def compute_overnight_performance(
    records: list[OvernightRunRecord] | None = None,
    journal: OvernightJournal | None = None,
) -> OvernightPerformanceSummary:
    """Compute comprehensive performance summary from journal records."""
    if records is None:
        j = journal or shared_overnight_journal()
        records = j.list(limit=1000)

    total_runs = len(records)
    go_recs = [r for r in records if r.decision == "GO"]
    nogo_recs = [r for r in records if r.decision == "NO-GO"]
    actual_recs = [r for r in records if r.is_actual_trade == 1 and r.actual_pnl is not None]
    hypo_recs = [r for r in records if r.is_actual_trade == 0 and r.hypothetical_pnl is not None]

    # 1. Actual GO Trades
    go_pnls = [r.actual_pnl for r in actual_recs if r.actual_pnl is not None]
    go_wins = sum(1 for p in go_pnls if p > 100)
    go_losses = sum(1 for p in go_pnls if p < -100)
    go_scratches = sum(1 for p in go_pnls if -100 <= p <= 100)
    
    n_go_settled = len(go_pnls)
    go_wr = (go_wins / n_go_settled) if n_go_settled > 0 else 0.0
    go_net = sum(go_pnls) if go_pnls else 0.0
    go_avg = (go_net / n_go_settled) if n_go_settled > 0 else 0.0

    gross_win = sum(p for p in go_pnls if p > 0)
    gross_loss = abs(sum(p for p in go_pnls if p < 0))
    go_pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    go_best = max(go_pnls) if go_pnls else 0.0
    go_worst = min(go_pnls) if go_pnls else 0.0

    # 2. NO-GO Filter Counterfactuals
    hypo_pnls = [r.hypothetical_pnl for r in hypo_recs if r.hypothetical_pnl is not None]
    avoided_losses = [p for p in hypo_pnls if p < -100]
    missed_winners = [p for p in hypo_pnls if p > 100]

    avoided_cnt = len(avoided_losses)
    avoided_rup = abs(sum(avoided_losses))
    missed_cnt = len(missed_winners)
    missed_rup = sum(missed_winners)

    tot_filter_decisions = avoided_cnt + missed_cnt
    filter_eff = (avoided_cnt / tot_filter_decisions * 100.0) if tot_filter_decisions > 0 else 50.0

    # 3. Dimensional Breakdowns
    by_dir: dict[str, dict[str, Any]] = {}
    for d in ("bullish", "bearish", "neutral"):
        sub = [r for r in records if r.direction == d]
        p_act = [r.actual_pnl for r in sub if r.actual_pnl is not None]
        by_dir[d] = {
            "runs": len(sub),
            "go_count": sum(1 for r in sub if r.decision == "GO"),
            "actual_trades": len(p_act),
            "win_rate": (sum(1 for p in p_act if p > 100) / len(p_act)) if p_act else 0.0,
            "net_pnl": sum(p_act) if p_act else 0.0,
        }

    by_regime: dict[str, dict[str, Any]] = {}
    for reg in ("sideways", "trending_bull", "trending_bear", "high_volatility"):
        sub = [r for r in records if r.market_regime == reg]
        p_act = [r.actual_pnl for r in sub if r.actual_pnl is not None]
        by_regime[reg] = {
            "runs": len(sub),
            "go_count": sum(1 for r in sub if r.decision == "GO"),
            "actual_trades": len(p_act),
            "win_rate": (sum(1 for p in p_act if p > 100) / len(p_act)) if p_act else 0.0,
            "net_pnl": sum(p_act) if p_act else 0.0,
        }

    by_score: dict[str, dict[str, Any]] = {
        "<50 (Weak)": {"min": 0, "max": 50},
        "50-64 (Watch)": {"min": 50, "max": 65},
        "65-74 (Valid)": {"min": 65, "max": 75},
        "75+ (High)": {"min": 75, "max": 101},
    }
    score_res: dict[str, dict[str, Any]] = {}
    for label, b in by_score.items():
        sub = [r for r in records if b["min"] <= r.confidence_score < b["max"]]
        p_act = [r.actual_pnl for r in sub if r.actual_pnl is not None]
        p_hyp = [r.hypothetical_pnl for r in sub if r.hypothetical_pnl is not None]
        score_res[label] = {
            "runs": len(sub),
            "go_count": sum(1 for r in sub if r.decision == "GO"),
            "actual_pnl": sum(p_act) if p_act else 0.0,
            "hypo_pnl": sum(p_hyp) if p_hyp else 0.0,
        }

    return OvernightPerformanceSummary(
        total_runs=total_runs,
        go_count=len(go_recs),
        nogo_count=len(nogo_recs),
        actual_trades=len(actual_recs),
        settled_actual=n_go_settled,
        settled_hypothetical=len(hypo_pnls),
        go_wins=go_wins,
        go_losses=go_losses,
        go_scratches=go_scratches,
        go_win_rate=round(go_wr, 3),
        go_net_pnl=round(go_net, 2),
        go_avg_pnl=round(go_avg, 2),
        go_profit_factor=round(go_pf, 2),
        go_best_trade=round(go_best, 2),
        go_worst_trade=round(go_worst, 2),
        avoided_losses_count=avoided_cnt,
        avoided_losses_rupees=round(avoided_rup, 2),
        missed_winners_count=missed_cnt,
        missed_winners_rupees=round(missed_rup, 2),
        filter_efficiency_pct=round(filter_eff, 1),
        by_direction=by_dir,
        by_regime=by_regime,
        by_score_tier=score_res,
    )
