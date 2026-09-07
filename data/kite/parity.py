"""Parity checks: Kite data vs the incumbent sources (pure functions).

Thresholds are generous on purpose — the harness catches wiring errors
(wrong token, bad timezone, misaligned dates), not market microstructure.
Corporate actions (splits/bonus) legitimately break close-parity; the
report surfaces worst offenders for eyeballing instead of failing blindly.
"""

from __future__ import annotations

import pandas as pd

CLOSE_TOL_PCT = 0.5
CHAIN_LTP_TOL_PCT = 5.0
CHAIN_MATCH_MIN_PCT = 80.0


def compare_closes(a: pd.Series, b: pd.Series,
                   tol_pct: float = CLOSE_TOL_PCT) -> dict:
    """Align two close series (date-normalized inner join) and diff them."""
    left = a.copy()
    left.index = pd.DatetimeIndex(left.index).normalize()
    right = b.copy()
    right.index = pd.DatetimeIndex(right.index).normalize()
    joined = pd.concat([left.rename("a"), right.rename("b")],
                       axis=1, join="inner").dropna()
    if joined.empty:
        return {"n": 0, "pass": False, "reason": "no overlapping dates"}
    diff = (joined["a"] - joined["b"]).abs() / joined["b"].abs().replace(0, float("nan"))
    diff = diff.dropna() * 100.0
    worst = diff.sort_values(ascending=False).head(3)
    return {
        "n": int(len(diff)),
        "max_abs_pct": round(float(diff.max()), 3),
        "mean_abs_pct": round(float(diff.mean()), 4),
        "worst": [(d.strftime("%Y-%m-%d"), round(float(v), 3))
                  for d, v in worst.items()],
        "pass": bool(float(diff.max()) <= tol_pct),
    }


def compare_chain_ltps(expected: dict[float, float],
                       actual: dict[float, float],
                       tol_pct: float = CHAIN_LTP_TOL_PCT) -> dict:
    """Strike -> LTP maps (NSE scrape vs Kite quotes). Returns match stats."""
    strikes = sorted(set(expected) & set(actual))
    if not strikes:
        return {"matched": 0, "pass": False, "reason": "no common strikes"}
    diffs = []
    for s in strikes:
        e, a = expected[s], actual[s]
        if e and a and e > 0:
            diffs.append(abs(e - a) / e * 100.0)
    if not diffs:
        return {"matched": len(strikes), "pass": False,
                "reason": "no quotable LTPs"}
    return {
        "matched": len(strikes),
        "max_abs_pct": round(float(max(diffs)), 2),
        "mean_abs_pct": round(float(sum(diffs) / len(diffs)), 3),
        "pass": bool(max(diffs) <= tol_pct),
    }
