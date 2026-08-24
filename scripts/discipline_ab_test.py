#!/usr/bin/env python3
"""A/B test: baseline vs disciplined (no weekends / expiry-adjacent nights)."""

import numpy as np

from data import nifty
from model.overnight import (
    apply_discipline,
    collect_overnight_signals,
    premium_outlook,
)
from model.overnight_card import _bucket_defs, signal_conditions

SPOT = 24000.0
IV = 13.0
DTE = 7


def simulate(subset):
    if len(subset) < 10:
        return None
    gaps = np.array([s.gap_pct for s in subset])
    o = premium_outlook(gaps, SPOT, IV, DTE)
    prem = np.array([
        min(o["delta_used"] * s.entry_close * s.gap_pct / 100 - o["theta_overnight"],
            o["est_atm_premium"])          # long option loss capped at premium
        / o["est_atm_premium"] * 100 for s in subset])
    cum = np.cumsum(prem)
    dd = (np.maximum.accumulate(cum) - cum).max()
    yrs = max((subset[-1].timestamp - subset[0].timestamp).days / 365.25, 0.5)
    return (len(prem), len(prem) / yrs, 100 * (gaps > 0).mean(), gaps.mean(),
            prem.mean(), 100 * (prem > 0).mean(), prem.sum(), dd)


def main():
    res = nifty.fetch_history(period="5y")
    base = collect_overnight_signals(res.candles)
    disc = apply_discipline(base)

    print(f"5y qualifying closes: {len(base)} baseline -> {len(disc)} after "
          f"discipline ({len(base) - len(disc)} dropped)\n")

    buckets = [("ALL", lambda c, s: True)] + [
        (name, lambda c, s, p=p: p(c)) for name, p in _bucket_defs()
    ]
    buckets += [
        ("regime=high_vol", lambda c, s: s.regime == "high_volatility"),
        ("regime=trending_bear", lambda c, s: s.regime == "trending_bear"),
    ]

    hdr = (f"{'bucket':<34}{'n':>5}{'tr/yr':>6}{'gapW%':>7}{'avgGap%':>9}"
           f"{'prem/nt':>9}{'premW%':>8}{'sumPP':>8}{'maxDD':>7}")
    for title, sigs in (("=== BASELINE (every night) ===", base),
                        ("=== DISCIPLINED ===", disc)):
        print(title)
        print(hdr)
        print("-" * len(hdr))
        for name, pred in buckets:
            subset = [s for s in sigs if pred(signal_conditions(s), s)]
            r = simulate(subset)
            if r:
                n, tpy, gw, ag, pa, pw, tot, dd = r
                print(f"{name:<34}{n:>5}{tpy:>6.0f}{gw:>7.1f}{ag:>+9.3f}"
                      f"{pa:>+9.1f}{pw:>8.1f}{tot:>+8.0f}{dd:>7.0f}")
        print()


if __name__ == "__main__":
    main()
