#!/usr/bin/env python3
"""Overnight LONG STRADDLE test: buy ATM CE + PE at close, exit next open.

Direction doesn't matter here — P&L is driven by |gap| vs double theta.
    pnl ≈ (delta × spot × |gap| − 2×theta) / (2×premium)
Loss capped at −100% of total premium paid.

Compares every EOD condition bucket on ABSOLUTE gap, which is what a
straddle trades, alongside the directional strategy for reference.
"""

import numpy as np

from data import nifty
from model.overnight import (
    apply_discipline,
    collect_overnight_signals,
    premium_outlook,
)

SPOT = 24000.0
IV = 13.0
DTE = 7


def straddle_sim(subset):
    if len(subset < 1 if False else subset) < 10:
        return None
    o = premium_outlook(np.array([abs(s.gap_pct) for s in subset]), SPOT, IV, DTE)
    delta, theta, prem = o["delta_used"], o["theta_overnight"], o["est_atm_premium"]
    be_abs = 2 * theta / (delta * SPOT) * 100          # |gap| needed to break even
    pnls = []
    for s in subset:
        gross = delta * SPOT * abs(s.gap_pct) / 100 - 2 * theta
        pnls.append(max(gross, -2 * prem) / (2 * prem) * 100)
    pnls = np.array(pnls)
    cum = np.cumsum(pnls)
    dd = (np.maximum.accumulate(cum) - cum).max()
    yrs = max((subset[-1].timestamp - subset[0].timestamp).days / 365.25, 0.5)
    return dict(n=len(pnls), tpy=len(pnls) / yrs,
                avg_abs=float(np.mean([abs(s.gap_pct) for s in subset])),
                be=be_abs, pavg=pnls.mean(), pwin=100 * (pnls > 0).mean(),
                total=pnls.sum(), dd=dd)


def main():
    res = nifty.fetch_history(period="5y")
    base = collect_overnight_signals(res.candles)
    disc = apply_discipline(base)

    def vol_ok(s):
        return True   # placeholder to keep bucket tuple shapes aligned

    buckets = [
        ("ALL", lambda s: True),
        ("regime=high_vol", lambda s: s.regime == "high_volatility"),
        ("regime=sideways", lambda s: s.regime == "sideways"),
        ("regime=trending_bull", lambda s: s.regime == "trending_bull"),
        ("strong close", lambda s: (s.close_pos > 0.7) or (s.close_pos < 0.3)),
        ("weak close (faded)", lambda s: 0.35 < s.close_pos < 0.65),
        ("rel vol >= 1.3x", lambda s: np.isfinite(s.rel_volume) and s.rel_volume >= 1.3),
        ("thin day (<0.8x)", lambda s: not np.isfinite(s.rel_volume) or s.rel_volume < 0.8),
        ("score >= 75", lambda s: s.score >= 75),
        ("high-vol + strong close", lambda s: s.regime == "high_volatility"
            and ((s.close_pos > 0.7) or (s.close_pos < 0.3))),
        ("high-vol + weak close", lambda s: s.regime == "high_volatility"
            and 0.35 < s.close_pos < 0.65),
    ]

    hdr = (f"{'bucket':<26}{'n':>5}{'tr/yr':>6}{'avg|gap|%':>10}"
           f"{'BE|gap|%':>10}{'prem/nt%':>10}{'win%':>7}{'sumPP':>8}{'maxDD':>7}")
    for title, sigs in (("=== BASELINE ===", base),
                        ("=== DISCIPLINED (no weekend/expiry) ===", disc)):
        print(title)
        print(hdr)
        print("-" * len(hdr))
        rows = []
        for name, pred in buckets:
            r = straddle_sim([s for s in sigs if pred(s)])
            if r:
                rows.append((r["total"], name, r))
        # sort by total P&L so winners surface first
        for _, name, r in sorted(rows, reverse=True):
            print(f"{name:<26}{r['n']:>5}{r['tpy']:>6.0f}{r['avg_abs']:>10.3f}"
                  f"{r['be']:>10.3f}{r['pavg']:>+10.1f}{r['pwin']:>7.1f}"
                  f"{r['total']:>+8.0f}{r['dd']:>7.0f}")
        print()

    # What share of nights moved enough to beat the straddle breakeven?
    o = premium_outlook(np.array([abs(s.gap_pct) for s in disc]), SPOT, IV, DTE)
    be = 2 * o["theta_overnight"] / (o["delta_used"] * SPOT) * 100
    absg = np.array([abs(s.gap_pct) for s in disc])
    print(f"Disciplined nights beating straddle BE ({be:.3f}%): "
          f"{100 * (absg > be).mean():.1f}%")
    for q in (50, 75, 90, 95):
        print(f"  |gap| p{q}: {np.percentile(absg, q):.3f}%")


if __name__ == "__main__":
    main()
