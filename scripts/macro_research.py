#!/usr/bin/env python3
"""Macro research: do overnight macro moves predict NIFTY's next open?

Tests, over 5y of disciplined overnight signals AND all nights:
  1. signed correlation (directional edge)
  2. absolute correlation (straddle/magnitude edge)
  3. conditional buckets on S&P overnight move
  4. straddle filter: nights AFTER big |SPX| moves — bigger NIFTY |gap|?
"""

import numpy as np
import pandas as pd

from analysis.indicators import compute as ci
from data import nifty
from model.macro import fetch_macro_history, macro_feature_frame, research
from model.overnight import apply_discipline, collect_overnight_signals

PERIOD = "5y"


def main():
    res = nifty.fetch_history(period=PERIOD)
    f = ci(res.candles).frame
    dates = pd.DatetimeIndex(f.index)

    macro = fetch_macro_history(PERIOD)
    print("macro series loaded:", ", ".join(sorted(macro)))
    feats = macro_feature_frame(dates, macro)

    # Gap for every night: next_open vs today's close.
    gap = (f["open"].shift(-1) / f["close"] - 1) * 100
    gaps_by_date = pd.Series(gap.values[:-1], index=dates[:-1])

    print("\n--- ALL NIGHTS ---")
    r = research(gaps_by_date.iloc[:-1], feats.iloc[:-1])
    _print_research(r)

    # Disciplined signal nights only.
    sigs = apply_discipline(collect_overnight_signals(res.candles))
    sig_gaps = pd.Series(
        {pd.Timestamp(s.timestamp): s.gap_pct for s in sigs}).sort_index()
    print("\n--- DISCIPLINED SIGNAL NIGHTS ---")
    r2 = research(sig_gaps, feats)
    _print_research(r2)


def _print_research(r: dict) -> None:
    if not r.get("n"):
        print("no aligned data")
        return
    print(f"n={r['n']} aligned nights")
    print(f"\n{'feature':<14}{'corr(signed)':>13}{'corr(|gap|)':>12}")
    for col in sorted(r["corr_signed"], key=lambda c: -abs(r["corr_signed"][c])):
        print(f"{col:<14}{r['corr_signed'][col]:>+13.3f}{r['corr_abs'].get(col, float('nan')):>12.3f}")

    if "spx_buckets" in r:
        print("\nNIFTY next-open gap conditioned on S&P overnight session:")
        print(f"{'bucket':<18}{'n':>5}{'up%':>6}{'avgGap%':>9}{'avg|gap|%':>10}")
        for name, (n, up, avg, aavg) in r["spx_buckets"].items():
            print(f"{name:<18}{n:>5}{up:>6.1f}{avg:>+9.3f}{aavg:>10.3f}")


if __name__ == "__main__":
    main()
