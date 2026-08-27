"""Performance analytics for intraday confluence journal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from journal.confluence_db import ConfluenceJournal, ConfluenceRunRecord, shared_confluence_journal


@dataclass(frozen=True)
class ConfluencePerformanceSummary:
    total_runs: int
    go_count: int
    nogo_count: int
    watch_count: int
    settled_actual: int
    settled_hypothetical: int
    go_wins: int
    go_losses: int
    go_win_rate: float
    go_net_pnl: float
    go_avg_pnl: float
    by_setup: dict[str, dict[str, Any]] = field(default_factory=dict)


def compute_confluence_performance(
    records: list[ConfluenceRunRecord] | None = None,
    journal: ConfluenceJournal | None = None,
) -> ConfluencePerformanceSummary:
    if records is None:
        j = journal or shared_confluence_journal()
        records = j.list(limit=1000)

    go_recs = [r for r in records if r.decision == "GO"]
    nogo_recs = [r for r in records if r.decision == "NO-GO"]
    watch_recs = [r for r in records if r.decision == "WATCH"]
    actual = [r for r in records if r.is_actual_trade == 1 and r.actual_pnl is not None]
    hypo = [r for r in records if r.is_actual_trade == 0 and r.hypothetical_pnl is not None]

    go_pnls = [r.actual_pnl for r in actual if r.actual_pnl is not None]
    go_wins = sum(1 for p in go_pnls if p > 100)
    go_losses = sum(1 for p in go_pnls if p < -100)
    n_settled = len(go_pnls)
    go_wr = (go_wins / n_settled) if n_settled > 0 else 0.0
    go_net = sum(go_pnls) if go_pnls else 0.0
    go_avg = (go_net / n_settled) if n_settled > 0 else 0.0

    by_setup: dict[str, dict[str, Any]] = {}
    for sid, label in (("A", "Trend Momentum"), ("B", "ORB"), ("C", "Reversal")):
        sub = [r for r in records if r.setup_id == sid]
        p_act = [r.actual_pnl for r in sub if r.actual_pnl is not None]
        by_setup[sid] = {
            "label": label,
            "runs": len(sub),
            "go_count": sum(1 for r in sub if r.decision == "GO"),
            "nogo_count": sum(1 for r in sub if r.decision == "NO-GO"),
            "watch_count": sum(1 for r in sub if r.decision == "WATCH"),
            "win_rate": (sum(1 for p in p_act if p > 100) / len(p_act)) if p_act else 0.0,
            "net_pnl": sum(p_act) if p_act else 0.0,
        }

    return ConfluencePerformanceSummary(
        total_runs=len(records),
        go_count=len(go_recs),
        nogo_count=len(nogo_recs),
        watch_count=len(watch_recs),
        settled_actual=n_settled,
        settled_hypothetical=len(hypo),
        go_wins=go_wins,
        go_losses=go_losses,
        go_win_rate=round(go_wr, 3),
        go_net_pnl=round(go_net, 2),
        go_avg_pnl=round(go_avg, 2),
        by_setup=by_setup,
    )
