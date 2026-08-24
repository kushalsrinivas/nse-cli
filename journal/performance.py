"""Performance analytics over the trade journal.

Answers: which setups actually make money? Computes aggregate stats plus
breakdowns by strategy, direction, calendar period, and indicator setup.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from journal.db import Trade


@dataclass
class Stats:
    total: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float | None = None
    gross_profit: float = 0.0
    gross_loss: float = 0.0          # positive number magnitude of losses
    net_pnl: float = 0.0
    avg_win: float | None = None
    avg_loss: float | None = None    # positive magnitude
    risk_reward: float | None = None
    profit_factor: float | None = None
    max_drawdown: float = 0.0
    avg_duration_min: float | None = None
    largest_winner: float = 0.0
    largest_loser: float = 0.0       # negative
    max_consec_wins: int = 0
    max_consec_losses: int = 0

    @property
    def has_data(self) -> bool:
        return self.total > 0


def summarize(trades: list[Trade]) -> Stats:
    s = Stats()
    if not trades:
        return s

    pnls = [t.pnl or 0.0 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    s.total = len(trades)
    s.wins, s.losses = len(wins), len(losses)
    s.win_rate = round(s.wins / s.total * 100, 1)
    s.gross_profit = round(sum(wins), 2)
    s.gross_loss = round(abs(sum(losses)), 2)
    s.net_pnl = round(sum(pnls), 2)
    s.avg_win = round(s.gross_profit / s.wins, 2) if wins else None
    s.avg_loss = round(s.gross_loss / s.losses, 2) if losses else None
    if s.avg_loss:
        s.risk_reward = round(s.avg_win / s.avg_loss, 2) if s.avg_win else 0.0
    s.profit_factor = (
        round(s.gross_profit / s.gross_loss, 2) if s.gross_loss else None
    )
    s.largest_winner = round(max(pnls), 2) if pnls else 0.0
    s.largest_loser = round(min(pnls), 2) if pnls else 0.0

    durations = [t.duration_minutes for t in trades if t.duration_minutes is not None]
    s.avg_duration_min = round(sum(durations) / len(durations), 1) if durations else None

    # Max drawdown on cumulative P&L curve (chronological).
    cum = peak = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        s.max_drawdown = min(s.max_drawdown, cum - peak)
    s.max_drawdown = round(s.max_drawdown, 2)

    # Streaks.
    streak_w = streak_l = 0
    for p in pnls:
        if p > 0:
            streak_w += 1; streak_l = 0
        elif p < 0:
            streak_l += 1; streak_w = 0
        s.max_consec_wins = max(s.max_consec_wins, streak_w)
        s.max_consec_losses = max(s.max_consec_losses, streak_l)

    return s


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------

def _bucket_key(t: Trade, key: str):
    ts = datetime.fromisoformat(t.timestamp)
    match key:
        case "strategy":
            return t.strategy or "manual"
        case "direction":
            return t.direction
        case "setup":
            # Indicator fingerprint: e.g. "M▲ E▲ S▲ V●"
            parts = [
                (t.macd_state or "-")[:1],
                (t.ema_state or "-")[:1],
                (t.sma_state or "-")[:1],
                (t.volume_state or "-")[:1],
            ]
            return " ".join(parts).strip() or "(unrecorded)"
        case "day":
            return ts.strftime("%Y-%m-%d")
        case "week":
            iso = ts.isocalendar()
            return f"{iso[0]}-W{iso[1]:02d}"
        case "month":
            return ts.strftime("%Y-%m")
        case _:
            raise ValueError(f"unknown breakdown {key!r}")


def breakdown(trades: list[Trade], key: str) -> dict[str, Stats]:
    buckets: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        buckets[str(_bucket_key(t, key))].append(t)
    return {k: summarize(v) for k, v in sorted(buckets.items())}


def all_breakdowns(trades: list[Trade]) -> dict[str, dict[str, Stats]]:
    keys = ("strategy", "direction", "setup", "day", "week", "month")
    return {k: breakdown(trades, k) for k in keys}
