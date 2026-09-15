"""L8 — the evaluation harness. Everything else in this package answers to it.

The audit's root cause was not a bad model, it was the absence of a
scoreboard: 21,000 lines accumulated with nothing that could say whether
any of it helped. So this module exists before the models do, and the rule
is simple — a model ships only if it beats a *named* baseline out of
sample, by more than the bootstrap interval on the difference.

Three disciplines are non-negotiable here:

**Purged, embargoed walk-forward.** Train strictly before, predict strictly
after, with a gap between. The overnight target for day t is realised at
the open of t, and day t-1's features were computed from its close, so an
embargo of one session removes the overlap.

**Named baselines, always reported.** "Brier 0.19" means nothing on its
own. `P(up) = 0.622` is the number to beat for direction; unconditional
quantiles for the distribution. A model that cannot beat a constant is a
constant with extra steps.

**Block bootstrap.** Daily signals are autocorrelated, so i.i.d. intervals
are too narrow. Everything uncertain gets a moving-block interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

EPS = 1e-12


class Forecaster(Protocol):
    """Anything the harness can score. `predict` returns probabilities for
    binary targets, a point forecast for continuous ones."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Forecaster: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class FoldSpec:
    min_train: int = 400        # ~1.6y before the first prediction
    step: int = 21              # refit monthly
    embargo: int = 1            # sessions dropped between train and test


@dataclass
class Predictions:
    """Out-of-sample predictions, in chronological order."""

    index: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray
    name: str = ""
    folds: int = 0
    kind: str = "binary"        # "binary" | "point" | "quantile"

    def __len__(self) -> int:
        return len(self.y_true)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def walk_forward(X: np.ndarray, y: np.ndarray, index: pd.DatetimeIndex,
                 factory, *, spec: FoldSpec | None = None,
                 name: str = "", kind: str = "binary") -> Predictions:
    """Refit every `spec.step` rows; predict the block that follows.

    `factory` is called fresh per fold, so no state leaks between them.
    """
    spec = spec or FoldSpec()
    n = len(y)
    preds, trues, idx = [], [], []
    folds = 0
    start = spec.min_train
    while start < n:
        stop = min(start + spec.step, n)
        train_end = start - spec.embargo
        if train_end < 50:
            start = stop
            continue
        model = factory()
        model.fit(X[:train_end], y[:train_end])
        block = model.predict(X[start:stop])
        preds.append(np.asarray(block, dtype=float).reshape(len(block), -1))
        trues.append(y[start:stop])
        idx.append(index[start:stop])
        folds += 1
        start = stop
    if not preds:
        return Predictions(pd.DatetimeIndex([]), np.array([]), np.array([]),
                           name=name, folds=0, kind=kind)
    stacked = np.vstack(preds)
    return Predictions(
        index=pd.DatetimeIndex(np.concatenate([i.values for i in idx])),
        y_true=np.concatenate(trues),
        y_pred=stacked[:, 0] if stacked.shape[1] == 1 else stacked,
        name=name, folds=folds, kind=kind)


# ---------------------------------------------------------------------------
# Baselines — the numbers every model is measured against
# ---------------------------------------------------------------------------

class ConstantProbability:
    """P(up) = the training base rate. The bar for any direction model."""

    def __init__(self, p: float | None = None) -> None:
        self.p = p

    def fit(self, X, y):
        if self.p is None:
            self._fitted = float(np.mean(y))
        else:
            self._fitted = self.p
        return self

    def predict(self, X):
        return np.full(len(X), self._fitted)


class ConstantMean:
    """Point forecast = the training mean. The bar for any magnitude model."""

    def fit(self, X, y):
        self._mu = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mu)


class ZeroForecast:
    """Predict no move. Beating this on MAE is a low bar that the audited
    cohort engine still failed."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class UnconditionalQuantiles:
    """Training-sample quantiles, ignoring every feature."""

    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)) -> None:
        self.quantiles = tuple(quantiles)

    def fit(self, X, y):
        self._q = np.percentile(y, [q * 100 for q in self.quantiles])
        return self

    def predict(self, X):
        return np.tile(self._q, (len(X), 1))


class RandomWalkVol:
    """Tomorrow's vol = trailing realised vol. The bar for L2."""

    def __init__(self, col: int = 0) -> None:
        self.col = col

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.abs(X[:, self.col])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y, p, threshold: float = 0.5) -> float:
    return float(np.mean((np.asarray(p) >= threshold) == (np.asarray(y) > 0.5)))


def auc(y, p) -> float:
    """Rank AUC via the Mann-Whitney identity (no sklearn)."""
    y = np.asarray(y) > 0.5
    p = np.asarray(p, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks within ties
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier_skill(y, p, p_base) -> float:
    """1 - Brier/Brier_base. Positive means the model added something."""
    b, bb = brier(y, p), brier(y, p_base)
    return float(1 - b / bb) if bb > EPS else 0.0


def mae(y, f) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f))))


def rmse(y, f) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(f)) ** 2)))


def sign_hit(y, f) -> float:
    y, f = np.asarray(y), np.asarray(f)
    live = y != 0
    return float(np.mean(np.sign(f[live]) == np.sign(y[live]))) if live.any() else float("nan")


def corr(y, f) -> float:
    y, f = np.asarray(y, dtype=float), np.asarray(f, dtype=float)
    if np.std(y) < EPS or np.std(f) < EPS:
        return 0.0
    return float(np.corrcoef(y, f)[0, 1])


def pinball(y, q_pred, quantiles) -> float:
    """Mean pinball loss across quantile levels — the distributional score."""
    y = np.asarray(y, dtype=float)
    q_pred = np.atleast_2d(np.asarray(q_pred, dtype=float))
    losses = []
    for k, tau in enumerate(quantiles):
        d = y - q_pred[:, k]
        losses.append(np.mean(np.maximum(tau * d, (tau - 1) * d)))
    return float(np.mean(losses))


def qlike(realized_var, forecast_var) -> float:
    """QLIKE — the standard volatility loss, robust to noisy proxies."""
    rv = np.clip(np.asarray(realized_var, dtype=float), EPS, None)
    fv = np.clip(np.asarray(forecast_var, dtype=float), EPS, None)
    return float(np.mean(np.log(fv) + rv / fv))


def pit_uniformity(y, q_pred, quantiles) -> float:
    """Max deviation of realised coverage from nominal. 0 = perfect."""
    y = np.asarray(y, dtype=float)
    q_pred = np.atleast_2d(np.asarray(q_pred, dtype=float))
    return float(max(abs(np.mean(y <= q_pred[:, k]) - tau)
                     for k, tau in enumerate(quantiles)))


def calibration_table(y, p, bins=(0.0, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 1.01)
                      ) -> list[dict]:
    """Predicted vs realised, bucketed. The reliability curve as a table."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     "predicted": float(p[m].mean()),
                     "realized": float(y[m].mean()),
                     "error": float(p[m].mean() - y[m].mean())})
    return rows


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(total)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_ci(values: np.ndarray, statistic=np.mean, *,
                       block: int = 10, draws: int = 2000,
                       alpha: float = 0.05, seed: int = 7
                       ) -> tuple[float, float]:
    """Moving-block bootstrap CI. Daily series are autocorrelated; i.i.d.
    resampling would report intervals that are too narrow to be honest."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    stats = np.empty(draws)
    for i in range(draws):
        sample = np.concatenate([v[s:s + block] for s in starts[i]])[:n]
        stats[i] = statistic(sample)
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))


def paired_delta_ci(loss_a: np.ndarray, loss_b: np.ndarray, **kw
                    ) -> tuple[float, float, float]:
    """CI on mean(loss_a - loss_b), paired per observation.

    This is the decision rule for shipping: the interval must exclude 0.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    lo, hi = block_bootstrap_ci(d, **kw)
    return float(d.mean()), lo, hi


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Score:
    name: str
    n: int
    folds: int
    kind: str
    metrics: dict = field(default_factory=dict)
    calibration: list[dict] = field(default_factory=list)

    def row(self, keys) -> list[str]:
        out = []
        for k in keys:
            v = self.metrics.get(k)
            out.append("—" if v is None or (isinstance(v, float) and np.isnan(v))
                       else f"{v:.4f}" if isinstance(v, float) else str(v))
        return out


def score_binary(pred: Predictions, baseline: Predictions | None = None) -> Score:
    y, p = pred.y_true, pred.y_pred
    m = {"brier": brier(y, p), "log_loss": log_loss(y, p),
         "accuracy": accuracy(y, p), "auc": auc(y, p),
         "ece": expected_calibration_error(y, p), "base_rate": float(np.mean(y))}
    if baseline is not None and len(baseline) == len(pred):
        m["brier_skill"] = brier_skill(y, p, baseline.y_pred)
        d, lo, hi = paired_delta_ci((p - y) ** 2, (baseline.y_pred - y) ** 2)
        m["delta_brier"] = d
        m["delta_lo"], m["delta_hi"] = lo, hi
        m["beats_baseline"] = bool(hi < 0)      # lower Brier is better
    return Score(pred.name, len(pred), pred.folds, "binary", m,
                 calibration_table(y, p))


def score_point(pred: Predictions, baseline: Predictions | None = None) -> Score:
    y, f = pred.y_true, pred.y_pred
    m = {"mae": mae(y, f), "rmse": rmse(y, f), "corr": corr(y, f),
         "sign_hit": sign_hit(y, f)}
    if baseline is not None and len(baseline) == len(pred):
        d, lo, hi = paired_delta_ci(np.abs(y - f), np.abs(y - baseline.y_pred))
        m["delta_mae"] = d
        m["delta_lo"], m["delta_hi"] = lo, hi
        m["beats_baseline"] = bool(hi < 0)
    return Score(pred.name, len(pred), pred.folds, "point", m)


def score_quantile(pred: Predictions, quantiles,
                   baseline: Predictions | None = None) -> Score:
    y, q = pred.y_true, pred.y_pred
    m = {"pinball": pinball(y, q, quantiles),
         "pit_max_dev": pit_uniformity(y, q, quantiles)}
    for k, tau in enumerate(quantiles):
        m[f"cover_{int(tau * 100)}"] = float(np.mean(y <= np.atleast_2d(q)[:, k]))
    if baseline is not None and len(baseline) == len(pred):
        def per_obs(qp):
            qp = np.atleast_2d(qp)
            return np.mean([np.maximum(tau * (y - qp[:, k]),
                                       (tau - 1) * (y - qp[:, k]))
                            for k, tau in enumerate(quantiles)], axis=0)
        d, lo, hi = paired_delta_ci(per_obs(q), per_obs(baseline.y_pred))
        m["delta_pinball"] = d
        m["delta_lo"], m["delta_hi"] = lo, hi
        m["beats_baseline"] = bool(hi < 0)
    return Score(pred.name, len(pred), pred.folds, "quantile", m)


# ---------------------------------------------------------------------------
# Slicing — a model that only works in 2021 is not a model
# ---------------------------------------------------------------------------

def slice_scores(pred: Predictions, by: pd.Series, scorer=score_binary,
                 min_n: int = 30, **kw) -> dict[str, Score]:
    """Re-score within each level of `by` (aligned on the prediction index)."""
    labels = by.reindex(pred.index)
    out: dict[str, Score] = {}
    for level in pd.unique(labels.dropna()):
        m = (labels == level).to_numpy()
        if m.sum() < min_n:
            continue
        sub = Predictions(pred.index[m], pred.y_true[m],
                          pred.y_pred[m] if pred.y_pred.ndim == 1 else pred.y_pred[m, :],
                          name=f"{pred.name}[{level}]", folds=pred.folds,
                          kind=pred.kind)
        out[str(level)] = scorer(sub, **kw)
    return out


def by_year(pred: Predictions) -> pd.Series:
    return pd.Series(pred.index.year.astype(str), index=pred.index)


def by_tercile(values: pd.Series, labels=("low", "mid", "high")) -> pd.Series:
    """Terciles computed on the full series — for slicing, never for fitting."""
    return pd.Series(pd.qcut(values, 3, labels=list(labels)), index=values.index)
