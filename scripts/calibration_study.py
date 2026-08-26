#!/usr/bin/env python3
"""Calibration & Strategy Comparative Study for NIFTY Overnight Plays.

Evaluates:
1. Granular Score Bins (50-59, 60-64, 65-69, 70-74, 75-79, 80-84, 85+)
2. Strategy Performance Comparison: ITM vs ATM vs Debit Spread
3. Weekday & Holding Structure Breakdown
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table

from data import nifty
from model.magnitude import compute_distribution
from model.options_ev import (
    bs_greeks,
    bs_price,
    estimate_fees_per_lot,
    estimate_spread_cost,
)
from model.overnight import (
    apply_discipline,
    collect_overnight_signals,
    estimate_expected_iv_change,
)

console = Console()
SPOT = 24000.0
BASE_IV = 14.0
LOT_SIZE = 75


def score_bin_study(signals: list) -> None:
    bins = [
        ("50 - 59 (Weak)", lambda s: 50 <= s.score < 60),
        ("60 - 64 (Watch)", lambda s: 60 <= s.score < 65),
        ("65 - 69 (Valid Setup)", lambda s: 65 <= s.score < 70),
        ("70 - 74 (Solid Setup)", lambda s: 70 <= s.score < 75),
        ("75 - 79 (High Conviction)", lambda s: 75 <= s.score < 80),
        ("80 - 84 (Very High)", lambda s: 80 <= s.score < 85),
        ("85+     (Extreme)", lambda s: s.score >= 85),
    ]

    t = Table(title="1. Confidence Score Bin Calibration", expand=False)
    t.add_column("Score Bin", style="bold")
    t.add_column("n", justify="right")
    t.add_column("Win %", justify="right")
    t.add_column("Mean Gap", justify="right")
    t.add_column("Median Gap", justify="right")
    t.add_column("P10", justify="right", style="red")
    t.add_column("P90", justify="right", style="green")

    for label, pred in bins:
        subset = [s for s in signals if pred(s)]
        if not subset:
            continue
        dist = compute_distribution([s.gap_pct for s in subset])
        win_color = "green" if dist.p_positive >= 0.55 else "red" if dist.p_positive < 0.50 else "yellow"
        t.add_row(
            label,
            str(dist.n_samples),
            f"[{win_color}]{dist.p_positive * 100:.1f}%[/]",
            f"{dist.mean_pct:+.3f}%",
            f"{dist.median_pct:+.3f}%",
            f"{dist.p10_pct:+.2f}%",
            f"{dist.p90_pct:+.2f}%",
        )
    console.print(t)


def simulate_strategy_over_signals(signals: list, strat_type: str, dte: int = 7) -> dict:
    pnls = []
    lot_size = LOT_SIZE

    for s in signals:
        is_call = (s.direction.value == "bullish")
        spot_entry = s.entry_close
        spot_exit = s.next_open
        ds = spot_exit - spot_entry

        weekday = s.timestamp.weekday() if hasattr(s.timestamp, "weekday") else 0
        div = estimate_expected_iv_change(weekday, dte, s.regime)
        iv_entry = BASE_IV / 100.0
        iv_exit = max(0.01, (BASE_IV + div) / 100.0)

        if strat_type == "ATM":
            # Delta ~ 0.50 (Strike = spot)
            strike = round(spot_entry / 50.0) * 50.0
            p_entry = bs_price(spot_entry, strike, dte, iv_entry, is_call)
            p_exit = bs_price(spot_exit, strike, dte - 1.0, iv_exit, is_call)
            spread = estimate_spread_cost(None, None, p_entry)
            fees = estimate_fees_per_lot(p_entry, lot_size) / lot_size
            gross_pnl = max(-p_entry, p_exit - p_entry)
            net_lot = (gross_pnl - spread - fees) * lot_size
            pnls.append(net_lot)

        elif strat_type == "ITM":
            # Delta ~ 0.75 (Strike approx 250 pts ITM)
            strike = (spot_entry - 250.0) if is_call else (spot_entry + 250.0)
            strike = round(strike / 50.0) * 50.0
            p_entry = bs_price(spot_entry, strike, dte, iv_entry, is_call)
            p_exit = bs_price(spot_exit, strike, dte - 1.0, iv_exit, is_call)
            spread = estimate_spread_cost(None, None, p_entry)
            fees = estimate_fees_per_lot(p_entry, lot_size) / lot_size
            gross_pnl = max(-p_entry, p_exit - p_entry)
            net_lot = (gross_pnl - spread - fees) * lot_size
            pnls.append(net_lot)

        elif strat_type == "DEBIT_SPREAD":
            # Long strike ATM (Delta ~0.50), Short strike OTM 200 pts (Delta ~0.28)
            l_strike = round(spot_entry / 50.0) * 50.0
            s_strike = (l_strike + 200.0) if is_call else (l_strike - 200.0)
            p_l_entry = bs_price(spot_entry, l_strike, dte, iv_entry, is_call)
            p_s_entry = bs_price(spot_entry, s_strike, dte, iv_entry, is_call)
            net_debit = max(0.1, p_l_entry - p_s_entry)

            p_l_exit = bs_price(spot_exit, l_strike, dte - 1.0, iv_exit, is_call)
            p_s_exit = bs_price(spot_exit, s_strike, dte - 1.0, iv_exit, is_call)
            exit_val = max(0.0, p_l_exit - p_s_exit)

            gross_pnl = exit_val - net_debit
            gross_pnl = max(-net_debit, min(200.0 - net_debit, gross_pnl))

            spread = (estimate_spread_cost(None, None, p_l_entry)
                      + estimate_spread_cost(None, None, p_s_entry))
            fees = 2 * estimate_fees_per_lot(net_debit, lot_size) / lot_size
            net_lot = (gross_pnl - spread - fees) * lot_size
            pnls.append(net_lot)

    arr = np.array(pnls)
    n = len(arr)
    wins = (arr > 0).sum()
    win_rate = (wins / n) if n > 0 else 0.0
    tot_pnl = float(np.sum(arr))
    avg_pnl = float(np.mean(arr)) if n > 0 else 0.0

    pos_sum = arr[arr > 0].sum() if (arr > 0).any() else 0.0
    neg_sum = abs(arr[arr < 0].sum()) if (arr < 0).any() else 1.0
    pf = (pos_sum / neg_sum) if neg_sum > 0 else 0.0

    cum = np.cumsum(arr)
    dd = float((np.maximum.accumulate(cum) - cum).max()) if n > 0 else 0.0

    return {
        "n": n,
        "win_rate": win_rate,
        "avg_lot_pnl": avg_pnl,
        "total_pnl": tot_pnl,
        "profit_factor": pf,
        "max_drawdown": dd,
    }


def strategy_comparison_study(signals: list) -> None:
    t = Table(title="2. Option Strategy Comparison (Disciplined Sample)", expand=False)
    t.add_column("Strategy Structure", style="bold")
    t.add_column("Trades (n)", justify="right")
    t.add_column("Win Rate", justify="right")
    t.add_column("Avg PnL / Lot", justify="right")
    t.add_column("Total Net PnL", justify="right")
    t.add_column("Profit Factor", justify="right")
    t.add_column("Max DD (₹)", justify="right", style="red")

    strats = [
        ("ATM Single-Leg (Control, Δ ~0.50)", "ATM"),
        ("ITM Single-Leg (High Delta, Δ ~0.75)", "ITM"),
        ("Debit Vertical Spread (200w)", "DEBIT_SPREAD"),
    ]

    for label, code in strats:
        res = simulate_strategy_over_signals(signals, code)
        color = "green" if res["total_pnl"] > 0 else "red"
        t.add_row(
            label,
            str(res["n"]),
            f"{res['win_rate'] * 100:.1f}%",
            f"[{color}]₹{res['avg_lot_pnl']:+,.0f}[/]",
            f"[{color}]₹{res['total_pnl']:+,.0f}[/]",
            f"{res['profit_factor']:.2f}",
            f"₹{res['max_drawdown']:,.0f}",
        )
    console.print(t)


def weekday_study(signals: list) -> None:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    t = Table(title="3. Weekday Breakdown", expand=False)
    t.add_column("Weekday", style="bold")
    t.add_column("n", justify="right")
    t.add_column("Win %", justify="right")
    t.add_column("Mean Gap", justify="right")
    t.add_column("ATM Avg PnL/Lot", justify="right")
    t.add_column("ITM Avg PnL/Lot", justify="right")

    for wd_idx, name in enumerate(days):
        subset = [s for s in signals if hasattr(s.timestamp, "weekday") and s.timestamp.weekday() == wd_idx]
        if not subset:
            continue
        dist = compute_distribution([s.gap_pct for s in subset])
        atm_res = simulate_strategy_over_signals(subset, "ATM")
        itm_res = simulate_strategy_over_signals(subset, "ITM")

        atm_col = "green" if atm_res["avg_lot_pnl"] > 0 else "red"
        itm_col = "green" if itm_res["avg_lot_pnl"] > 0 else "red"

        t.add_row(
            name,
            str(len(subset)),
            f"{dist.p_positive * 100:.1f}%",
            f"{dist.mean_pct:+.3f}%",
            f"[{atm_col}]₹{atm_res['avg_lot_pnl']:+,.0f}[/]",
            f"[{itm_col}]₹{itm_res['avg_lot_pnl']:+,.0f}[/]",
        )
    console.print(t)


def main() -> None:
    console.print("[bold]Running NIFTY Overnight Calibration & Strategy Study (5y History)...[/]\n")
    res = nifty.fetch_history(period="5y")
    raw_signals = collect_overnight_signals(res.candles)
    disc_signals = apply_discipline(raw_signals)

    console.print(f"Total qualifying signals: {len(raw_signals)} raw -> {len(disc_signals)} disciplined\n")

    score_bin_study(disc_signals)
    console.print()
    strategy_comparison_study(disc_signals)
    console.print()
    weekday_study(raw_signals)


if __name__ == "__main__":
    main()
