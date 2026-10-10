"""Canonical NIFTY 50 universe: symbols, index weights, sectors.

Source of truth for membership: NSE `ind_nifty50list.csv`
(https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv),
verified 2026-09-05. Notable: post-demerger Tata Motors is represented by
Tata Motors Passenger Vehicles (TMPV); the old TATAMOTORS symbol is delisted
and must NOT be used.

Weights are free-float weights from the NSE factsheet (~11-Jun-2026;
WIPRO approx). They drift with prices and quarterly rebalances — treat
them as priors, not truth. Everything downstream normalizes weights to sum
to 1.0, so absolute scale does not matter; only relative size does.

Refresh checklist (quarterly): re-pull the NSE CSV, update weights from
the latest factsheet, verify every Yahoo symbol still resolves. If a
symbol fails, the fetcher marks it missing and proceeds on weight
coverage, so a stale list degrades gracefully instead of crashing.

Membership is now dynamic: when the platform universe (data_store/app.db,
`platform_cli.py universe refresh`) holds a current NIFTY 50 membership,
`get_universe()` returns *that* list. The table below then only supplies
weight and sector priors; names it does not know get the median weight and
their NSE Industry as sector. Without a platform snapshot the table is
used as before.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Constituent:
    symbol: str          # Yahoo symbol, e.g. "HDFCBANK.NS"
    short: str           # NSE symbol, e.g. "HDFCBANK"
    weight: float        # approx index weight in percent (normalized at load)
    sector: str


# fmt: off
_UNIVERSE: tuple[Constituent, ...] = (
    Constituent("HDFCBANK.NS",   "HDFCBANK",   10.74, "Financials"),
    Constituent("ICICIBANK.NS",  "ICICIBANK",   8.88, "Financials"),
    Constituent("RELIANCE.NS",   "RELIANCE",    8.04, "Energy"),
    Constituent("BHARTIARTL.NS", "BHARTIARTL",  5.16, "Telecom"),
    Constituent("LT.NS",         "LT",          4.28, "Infra"),
    Constituent("SBIN.NS",       "SBIN",        3.92, "Financials"),
    Constituent("INFY.NS",       "INFY",        3.68, "IT"),
    Constituent("AXISBANK.NS",   "AXISBANK",    3.56, "Financials"),
    Constituent("KOTAKBANK.NS",  "KOTAKBANK",   2.73, "Financials"),
    Constituent("ITC.NS",        "ITC",         2.57, "FMCG"),
    Constituent("M&M.NS",        "M&M",         2.53, "Auto"),
    Constituent("BAJFINANCE.NS", "BAJFINANCE",  2.20, "Financials"),
    Constituent("TCS.NS",        "TCS",         2.06, "IT"),
    Constituent("HINDUNILVR.NS", "HINDUNILVR",  1.79, "FMCG"),
    Constituent("SUNPHARMA.NS",  "SUNPHARMA",   1.79, "Pharma"),
    Constituent("MARUTI.NS",     "MARUTI",      1.62, "Auto"),
    Constituent("ETERNAL.NS",    "ETERNAL",     1.60, "Consumer"),
    Constituent("NTPC.NS",       "NTPC",        1.57, "Power"),
    Constituent("TITAN.NS",      "TITAN",       1.57, "Consumer"),
    Constituent("TATASTEEL.NS",  "TATASTEEL",   1.54, "Metals"),
    Constituent("HINDALCO.NS",   "HINDALCO",    1.40, "Metals"),
    Constituent("BEL.NS",        "BEL",         1.36, "Industrials"),
    Constituent("ADANIPORTS.NS", "ADANIPORTS",  1.24, "Infra"),
    Constituent("POWERGRID.NS",  "POWERGRID",   1.22, "Power"),
    Constituent("ULTRACEMCO.NS", "ULTRACEMCO",  1.21, "Cement"),
    Constituent("SHRIRAMFIN.NS", "SHRIRAMFIN",  1.17, "Financials"),
    Constituent("ASIANPAINT.NS", "ASIANPAINT",  1.15, "Consumer"),
    Constituent("JSWSTEEL.NS",   "JSWSTEEL",    1.13, "Metals"),
    Constituent("GRASIM.NS",     "GRASIM",      1.11, "Cement"),
    Constituent("HCLTECH.NS",    "HCLTECH",     1.10, "IT"),
    Constituent("BAJAJ-AUTO.NS", "BAJAJ-AUTO",  1.06, "Auto"),
    Constituent("COALINDIA.NS",  "COALINDIA",   0.96, "Energy"),
    Constituent("NESTLEIND.NS",  "NESTLEIND",   0.96, "FMCG"),
    Constituent("INDIGO.NS",     "INDIGO",      0.96, "Services"),
    Constituent("EICHERMOT.NS",  "EICHERMOT",   0.93, "Auto"),
    Constituent("ONGC.NS",       "ONGC",        0.93, "Energy"),
    Constituent("BAJAJFINSV.NS", "BAJAJFINSV",  0.90, "Financials"),
    Constituent("TECHM.NS",      "TECHM",       0.88, "IT"),
    Constituent("WIPRO.NS",      "WIPRO",       0.85, "IT"),
    Constituent("TRENT.NS",      "TRENT",       0.85, "Consumer"),
    Constituent("APOLLOHOSP.NS", "APOLLOHOSP",  0.82, "Pharma"),
    Constituent("ADANIENT.NS",   "ADANIENT",    0.79, "Metals"),
    Constituent("TMPV.NS",       "TMPV",        0.74, "Auto"),
    Constituent("CIPLA.NS",      "CIPLA",       0.73, "Pharma"),
    Constituent("DRREDDY.NS",    "DRREDDY",     0.73, "Pharma"),
    Constituent("SBILIFE.NS",    "SBILIFE",     0.73, "Financials"),
    Constituent("JIOFIN.NS",     "JIOFIN",      0.71, "Financials"),
    Constituent("MAXHEALTH.NS",  "MAXHEALTH",   0.71, "Pharma"),
    Constituent("TATACONSUM.NS", "TATACONSUM",  0.68, "FMCG"),
    Constituent("HDFCLIFE.NS",   "HDFCLIFE",    0.55, "Financials"),
)
# fmt: on

WEIGHT_SNAPSHOT = "NSE factsheet 11-Jun-2026 (WIPRO approx 0.85)"

# Minimum weight coverage for a breadth snapshot to be trustworthy.
MIN_WEIGHT_COVERAGE = 0.70
# Minimum names covered.
MIN_NAMES_COVERED = 35


PLATFORM_APP_DB = Path(__file__).resolve().parents[2] / "data_store" / "app.db"


@lru_cache(maxsize=1)
def _platform_members() -> tuple[Constituent, ...] | None:
    """Current NIFTY 50 membership from the platform universe, or None."""
    path = Path(os.environ.get("NSE_PLATFORM_APP_DB", PLATFORM_APP_DB))
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT m.symbol, c.industry FROM index_membership m LEFT JOIN companies c "
            "ON c.isin=m.isin WHERE m.index_id='NSE:NIFTY 50' AND m.valid_to IS NULL").fetchall()
        conn.close()
    except sqlite3.Error:
        return None
    if len(rows) < MIN_NAMES_COVERED:
        return None
    prior = {c.short: c for c in _UNIVERSE}
    ws = sorted(c.weight for c in _UNIVERSE)
    median = ws[len(ws) // 2]
    out = []
    for sym, industry in sorted(rows):
        p = prior.get(sym)
        out.append(Constituent(f"{sym}.NS", sym, p.weight if p else median,
                               p.sector if p else (industry or "Other")))
    return tuple(sorted(out, key=lambda c: -c.weight))


def get_universe() -> tuple[Constituent, ...]:
    return _platform_members() or _UNIVERSE


def symbols() -> list[str]:
    return [c.symbol for c in get_universe()]


def weights_normalized(subset: list[str] | None = None) -> dict[str, float]:
    """Index weights normalized to sum to 1.0 over `subset` (or full universe)."""
    members = [c for c in get_universe() if subset is None or c.symbol in subset]
    total = sum(c.weight for c in members)
    if total <= 0:
        return {}
    return {c.symbol: c.weight / total for c in members}


def full_weights_normalized() -> dict[str, float]:
    return weights_normalized(None)


def heavyweights(top_n: int = 8) -> list[Constituent]:
    return sorted(get_universe(), key=lambda c: -c.weight)[:top_n]


def sectors() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in get_universe():
        out.setdefault(c.sector, []).append(c.symbol)
    return out
