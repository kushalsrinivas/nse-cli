"""L1 — the feature store: one row per NIFTY trade date.

Two things make this different from the ad-hoc feature extraction it
replaces.

**The decision point is explicit.** A feature is legal only if it is
observable when the decision is actually made, and that is not the same
moment for the two products this repo supports:

    EOD      15:25 IST on day t-1. You buy at the close and exit at the
             next open. Tonight's US session has NOT happened yet, so the
             global block is lagged one extra day. Target: gap[t], c2c[t].

    PREOPEN  08:30 IST on day t. Wall Street closed ~01:30, Asia is
             trading. The global block is fresh — this is where its
             information lives. But the gap is no longer tradeable (you
             enter at the 09:15 open, after it), so the target is the
             session, open to close.

Getting this wrong is the difference between a real edge and a look-ahead
artefact, so `decision_point` is a required argument with no default.

**Everything domestic is shifted.** Indicator value at bar i uses only
bars <= i, so one `.shift(1)` makes the whole block legal for predicting
day t. No feature here reads bar t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EOD = "eod"
PREOPEN = "preopen"
DECISION_POINTS = (EOD, PREOPEN)

#: Targets, all in percent. `up_*` are the binary versions.
TARGETS = (
    "gap_pct", "session_pct", "c2c_pct", "range_pct",
    "mfe_from_open_pct", "mae_from_open_pct",
    "up_gap", "up_session", "up_c2c",
)

#: The tradeable target at each decision point. The others stay available
#: as context (a PREOPEN gap forecast is useful even though you cannot
#: trade it) but must never be scored as if they were the product.
TRADEABLE_TARGET = {EOD: "gap_pct", PREOPEN: "session_pct"}


@dataclass
class Dataset:
    frame: pd.DataFrame
    features: list[str]
    decision_point: str
    macro_cols: list[str] = field(default_factory=list)
    domestic_cols: list[str] = field(default_factory=list)
    breadth_cols: list[str] = field(default_factory=list)

    def xy(self, target: str, features: list[str] | None = None
           ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Aligned (X, y, index) with rows missing anything dropped."""
        cols = list(features or self.features)
        sub = self.frame[cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
        return (sub[cols].to_numpy(dtype=float),
                sub[target].to_numpy(dtype=float),
                pd.DatetimeIndex(sub.index))

    @property
    def tradeable_target(self) -> str:
        return TRADEABLE_TARGET[self.decision_point]

    def describe(self) -> str:
        return (f"{len(self.frame)} rows "
                f"{self.frame.index[0]:%Y-%m-%d} → {self.frame.index[-1]:%Y-%m-%d}, "
                f"{len(self.features)} features "
                f"({len(self.domestic_cols)}d/{len(self.macro_cols)}g/"
                f"{len(self.breadth_cols)}x), point={self.decision_point}")


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def _targets(frame: pd.DataFrame) -> pd.DataFrame:
    o, h, low, c = (frame["open"], frame["high"], frame["low"], frame["close"])
    prev_c = c.shift(1)
    out = pd.DataFrame(index=frame.index)
    out["gap_pct"] = (o - prev_c) / prev_c * 100
    out["session_pct"] = (c - o) / o * 100
    out["c2c_pct"] = (c - prev_c) / prev_c * 100
    out["range_pct"] = (h - low) / o * 100
    out["mfe_from_open_pct"] = (h - o) / o * 100
    out["mae_from_open_pct"] = (low - o) / o * 100
    for name, base in (("up_gap", "gap_pct"), ("up_session", "session_pct"),
                       ("up_c2c", "c2c_pct")):
        out[name] = (out[base] > 0).astype(float).where(out[base].notna())
    return out


# ---------------------------------------------------------------------------
# Domestic block (prefix d_) — state of the NIFTY tape through bar t-1
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50.0)


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - frame["close"].shift()).abs(),
        (frame["low"] - frame["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    up = frame["high"].diff()
    dn = -frame["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=frame.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=frame.index)
    atr = _atr(frame, period).replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1 / period, adjust=False).mean() / atr
    mdi = 100 * minus.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _domestic(frame: pd.DataFrame) -> pd.DataFrame:
    """Tape state AT each bar. The caller shifts; nothing here peeks ahead."""
    c, o, h, low = frame["close"], frame["open"], frame["high"], frame["low"]
    f = pd.DataFrame(index=frame.index)

    f["d_ret1"] = c.pct_change() * 100
    f["d_ret5"] = c.pct_change(5) * 100
    f["d_ret20"] = c.pct_change(20) * 100

    rng = (h - low).replace(0, np.nan)
    f["d_close_pos"] = (c - low) / rng                       # 0..1 in own bar
    hi20, lo20 = h.rolling(20).max(), low.rolling(20).min()
    f["d_close_pos20"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)

    f["d_rsi"] = _rsi(c)
    f["d_adx"] = _adx(frame)

    atr = _atr(frame)
    f["d_atr_pct"] = atr / c * 100
    # MACD histogram normalised by ATR, not by the MACD level. Dividing by
    # the level makes the signal loudest exactly at the zero crossing,
    # which is where it means least (audit F-20).
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["d_macd_hist_atr"] = (macd - macd.ewm(span=9, adjust=False).mean()) / atr.replace(0, np.nan)

    for n in (20, 50, 200):
        f[f"d_dist_sma{n}"] = (c / c.rolling(n).mean() - 1) * 100

    ret = c.pct_change()
    rv20 = ret.rolling(20).std() * np.sqrt(252) * 100
    f["d_rv20"] = rv20
    f["d_rv_ratio"] = (ret.rolling(5).std() * np.sqrt(252) * 100) / rv20.replace(0, np.nan)

    prev_c = c.shift(1)
    f["d_gap"] = (o - prev_c) / prev_c * 100                 # this bar's own gap
    f["d_session"] = (c - o) / o * 100
    return f


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_dataset(candles, macro: dict | None = None, *,
                  decision_point: str,
                  breadth_frame: pd.DataFrame | None = None) -> Dataset:
    """Assemble the feature/target frame for one decision point.

    `macro` is `model.macro.fetch_macro_history()` output; omit it to build
    a domestic-only dataset (useful for ablation). `breadth_frame` is an
    optional per-date constituent block, already lagged by the caller.
    """
    if decision_point not in DECISION_POINTS:
        raise ValueError(f"decision_point must be one of {DECISION_POINTS}")

    from model.backtest import _base_frame
    frame = _base_frame(candles)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    out = _targets(frame)

    # Domestic state is known through bar t-1 at either decision point.
    dom = _domestic(frame).shift(1)
    domestic_cols = list(dom.columns)
    out = out.join(dom)

    # Day-of-week of the target session: known arbitrarily far in advance.
    out["d_dow"] = pd.DatetimeIndex(out.index).dayofweek.astype(float)
    domestic_cols.append("d_dow")

    macro_cols: list[str] = []
    if macro:
        from model.macro import macro_feature_frame
        mf = macro_feature_frame(pd.DatetimeIndex(frame.index), macro)
        # macro_feature_frame row t already uses only bars strictly before
        # t -- i.e. the session that finished overnight before NIFTY opens
        # on t. That is exactly the PREOPEN information set. At EOD the
        # decision is made the previous afternoon, before that session has
        # even started, so it gets one more lag.
        if decision_point == EOD:
            mf = mf.shift(1)
        mf = mf.add_prefix("g_")
        macro_cols = list(mf.columns)
        out = out.join(mf)

    breadth_cols: list[str] = []
    if breadth_frame is not None and len(breadth_frame):
        bf = breadth_frame.reindex(out.index).add_prefix("x_")
        breadth_cols = list(bf.columns)
        out = out.join(bf)

    features = domestic_cols + macro_cols + breadth_cols
    # Drop rows with no target at all; per-model dropna happens in xy().
    out = out.dropna(subset=["gap_pct", "session_pct"])
    return Dataset(frame=out, features=features, decision_point=decision_point,
                   macro_cols=macro_cols, domestic_cols=domestic_cols,
                   breadth_cols=breadth_cols)
