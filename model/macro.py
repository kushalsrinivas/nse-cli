"""Overnight macro intelligence layer.

While NIFTY sleeps, these markets move — and they're all known by the
time NIFTY opens:

    US equities      ^GSPC ^IXIC ^DJI    (close ~01:30-02:30 IST)
    Crude oil        BZ=F  CL=F             (~24h trading)
    Dollar-INR       INR=X                  (global FX session)
    India VIX        ^INDIAVIX              (previous NSE close)
    Asia early       ^N225 ^KS11 ^HSI       (Japan/Korea/HK)

Alignment rule: a foreign bar labelled date D completes during the IST
morning of D+1, so for a NIFTY decision on date D we use every series'
bar labelled strictly BEFORE D (`merge_asof`, no exact matches). No
look-ahead: everything in the feature row was printd before the open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import SETTINGS
from data.cache import shared_cache

log = logging.getLogger(__name__)

MACRO_TICKERS: dict[str, str] = {
    "spx": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "brent": "BZ=F",
    "wti": "CL=F",
    "usdinr": "INR=X",
    "indiavix": "^INDIAVIX",
    "nikkei": "^N225",
    "kospi": "^KS11",
    "hangseng": "^HSI",
}

RET_FEATURES = ("spx", "nasdaq", "brent", "wti", "usdinr",
                "nikkei", "kospi", "hangseng")


def fetch_macro_history(period: str = "5y") -> dict[str, pd.Series]:
    """Daily closes per macro series, cached."""
    cache = shared_cache()
    cached = cache.get("macro_history", {"period": period}, ttl=3600)
    if cached is not None:
        return cached

    import yfinance as yf
    out: dict[str, pd.Series] = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d",
                                           auto_adjust=False)
            if df is None or df.empty:
                continue
            s = df["Close"].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            out[name] = s[~s.index.duplicated(keep="last")]
        except Exception as exc:
            log.warning("macro fetch failed for %s: %s", ticker, exc)
    cache.set(out, "macro_history", {"period": period})
    return out


def macro_feature_frame(nifty_dates: pd.DatetimeIndex,
                        macro: dict[str, pd.Series]) -> pd.DataFrame:
    """Feature rows indexed by NIFTY date, each using only prior-day data.

    For each series: percent change of its latest bar STRICTLY BEFORE the
    NIFTY date (the session that finished overnight). indiavix contributes
    level + change; others contribute the overnight return.
    """
    idx = pd.DatetimeIndex(nifty_dates)
    feats = pd.DataFrame(index=idx)

    for name in RET_FEATURES:
        s = macro.get(name)
        if s is None or len(s) < 2:
            feats[name + "_ret"] = np.nan
            continue
        ret = s.pct_change() * 100
        aligned = _last_before(ret, idx)
        feats[name + "_ret"] = aligned

    vix = macro.get("indiavix")
    if vix is not None and len(vix) > 1:
        feats["vix_level"] = _last_before(vix, idx)
        feats["vix_chg"] = _last_before(vix.pct_change() * 100, idx)
    else:
        feats["vix_level"] = np.nan
        feats["vix_chg"] = np.nan

    # Composite risk sentiment: average of equity overnight moves.
    eq_cols = [c for c in ("spx_ret", "nasdaq_ret") if c in feats]
    feats["risk_pulse"] = feats[eq_cols].mean(axis=1, skipna=True)
    return feats


def _last_before(series: pd.Series, targets: pd.DatetimeIndex) -> pd.Series:
    """Value of the latest bar strictly before each target date."""
    s = series.dropna().sort_index()
    pos = s.index.searchsorted(targets.values, side="left")
    vals = np.where(pos > 0, s.values[np.maximum(pos - 1, 0)], np.nan)
    ok = pos > 0
    return pd.Series(np.where(ok, vals, np.nan), index=targets)


# ---------------------------------------------------------------------------
# Live snapshot for the nightly card
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MacroSnapshot:
    rows: list[tuple[str, float, float]]   # (label, overnight % change, level)
    risk_pulse: float                      # mean equity move overnight
    notes: list[str]

    @property
    def headline(self) -> str:
        tone = "RISK-OFF" if self.risk_pulse < -0.4 else \
               "RISK-ON" if self.risk_pulse > 0.4 else "NEUTRAL"
        return f"{tone} ({self.risk_pulse:+.2f}%)"


# Display precision + unit suffix per series.
_PRICE_FMT = {
    "S&P 500": ("{:.2f}", "pts"), "Nasdaq": ("{:.2f}", "pts"),
    "Dow": ("{:.2f}", "pts"), "Brent": ("${:.2f}", "/bbl"),
    "WTI": ("${:.2f}", "/bbl"), "USD/INR": ("{:.2f}", "₹/$"),
    "Nikkei*": ("{:.0f}", "pts"), "KOSPI*": ("{:.1f}", "pts"),
    "Hang Seng*": ("{:.0f}", "pts"),
}


def live_snapshot(period: str = "5mo") -> MacroSnapshot | None:
    """What the world did while NIFTY was closed — usable right now."""
    try:
        macro = fetch_macro_history(period)
    except Exception as exc:
        log.warning("macro snapshot unavailable: %s", exc)
        return None
    labels = {"spx": "S&P 500", "nasdaq": "Nasdaq", "dow": "Dow",
              "brent": "Brent", "wti": "WTI", "usdinr": "USD/INR",
              "nikkei": "Nikkei*", "kospi": "KOSPI*", "hangseng": "Hang Seng*"}
    rows, notes = [], []
    for name in RET_FEATURES:
        s = macro.get(name)
        if s is None or len(s) < 2:
            continue
        chg = (s.iloc[-1] / s.iloc[-2] - 1) * 100
        rows.append((labels.get(name, name),
                     round(float(chg), 2), float(s.iloc[-1])))
    if macro.get("nikkei") is not None:
        notes.append("*Asia figures may be mid-session")
    pulse_vals = [v for k, v, _ in rows if k in ("S&P 500", "Nasdaq")]
    pulse = float(np.mean(pulse_vals)) if pulse_vals else 0.0
    return MacroSnapshot(rows=rows, risk_pulse=pulse, notes=notes)


def format_level(label: str, level: float) -> str:
    fmt, suffix = _PRICE_FMT.get(label, ("{:.2f}", ""))
    return f"{fmt.format(level)} {suffix}".strip()


# ---------------------------------------------------------------------------
# Research: do macros predict the next-open gap?
# ---------------------------------------------------------------------------

def research(gaps_by_date: pd.Series, feats: pd.DataFrame) -> dict:
    """Correlations + conditional stats linking overnight macro to NIFTY gap.

    gaps_by_date: signed next-open gap (%) indexed by ENTRY date.
    """
    joined = pd.DataFrame({"gap": gaps_by_date}).join(feats, how="inner").dropna(
        subset=["gap"])
    out: dict = {"n": len(joined)}

    mag = joined["gap"].abs()
    corr_signed, corr_mag = {}, {}
    for col in feats.columns:
        c = joined[col].astype(float)
        valid = c.notna()
        if valid.sum() < 50:
            continue
        corr_signed[col] = round(float(corr(c[valid], joined.loc[valid, "gap"])), 3)
        corr_mag[col] = round(float(corr(c[valid], mag[valid])), 3)
    out["corr_signed"] = corr_signed
    out["corr_abs"] = corr_mag

    # Conditional buckets on the strongest known driver: S&P overnight.
    spx = joined.get("spx_ret")
    if spx is not None:
        b = {
            "SPX <-1%": (spx < -1.0), "SPX -1..-0.4%": (spx >= -1.0) & (spx < -0.4),
            "SPX -0.4..+0.4%": (spx.abs() <= 0.4),
            "SPX +0.4..+1%": (spx > 0.4) & (spx <= 1.0), "SPX >+1%": (spx > 1.0),
        }
        out["spx_buckets"] = {}
        for name, mask in b.items():
            sub = joined[mask.fillna(False)]
            if len(sub) >= 15:
                g = sub["gap"]
                out["spx_buckets"][name] = (
                    len(sub), round(float((g > 0).mean()) * 100, 1),
                    round(float(g.mean()), 3),
                    round(float(sub["gap"].abs().mean()), 3))
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
