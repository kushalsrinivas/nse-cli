"""Estimators, in numpy.

This repo depends on numpy and pandas and nothing heavier, which suits the
model class the data actually supports: ~1,200 daily observations with a
best single feature around r = 0.5. A regularised linear model is the right
tool here, and pulling in a tree-ensemble stack to fit eight coefficients
would be the same mistake the audit found — sophistication ahead of
evidence. Stage 6 tests a gradient-boosted alternative against these and
keeps it only if it wins.

Every estimator standardises inside `fit`, so the scaler is fitted on the
training fold only and the harness's walk-forward stays honest.
"""

from __future__ import annotations

import numpy as np


class _Standardizer:
    def fit(self, X: np.ndarray) -> _Standardizer:
        self.mu_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd_ = np.where(sd < 1e-9, 1.0, sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) / self.sd_


class RidgeLogistic:
    """L2-penalised logistic regression via Newton-IRLS.

    `l2` is applied to the standardised coefficients and never to the
    intercept, so the penalty does not fight the base rate.
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 60, tol: float = 1e-8) -> None:
        self.l2, self.max_iter, self.tol = l2, max_iter, tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeLogistic:
        self.scaler_ = _Standardizer().fit(X)
        Z = np.c_[np.ones(len(X)), self.scaler_.transform(X)]
        y = np.asarray(y, dtype=float)
        w = np.zeros(Z.shape[1])
        # Start at the base-rate intercept so the first Newton step is sane.
        p0 = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        w[0] = np.log(p0 / (1 - p0))
        penalty = np.full(Z.shape[1], self.l2)
        penalty[0] = 0.0

        for _ in range(self.max_iter):
            eta = Z @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
            grad = Z.T @ (p - y) + penalty * w
            s = np.clip(p * (1 - p), 1e-8, None)
            H = (Z * s[:, None]).T @ Z + np.diag(penalty)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            w_new = w - step
            if np.max(np.abs(w_new - w)) < self.tol:
                w = w_new
                break
            w = w_new
        self.coef_ = w
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        Z = np.c_[np.ones(len(X)), self.scaler_.transform(X)]
        return Z @ self.coef_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.decision(X), -35, 35)))

    def coefficients(self, names: list[str]) -> dict[str, float]:
        """Standardised coefficients — directly comparable across features."""
        return dict(zip(names, self.coef_[1:], strict=False))


class RidgeRegression:
    """L2-penalised least squares. The point-forecast workhorse."""

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeRegression:
        self.scaler_ = _Standardizer().fit(X)
        Z = np.c_[np.ones(len(X)), self.scaler_.transform(X)]
        penalty = np.full(Z.shape[1], self.l2)
        penalty[0] = 0.0
        A = Z.T @ Z + np.diag(penalty)
        self.coef_ = np.linalg.solve(A, Z.T @ np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones(len(X)), self.scaler_.transform(X)] @ self.coef_

    def coefficients(self, names: list[str]) -> dict[str, float]:
        return dict(zip(names, self.coef_[1:], strict=False))


class QuantileRegression:
    """Linear quantile regression by IRLS on the smoothed pinball loss.

    Fits every level jointly-shaped but independently solved, then sorts the
    predicted quantiles per row so the emitted distribution is monotone —
    crossing quantiles are a real artefact of fitting levels separately and
    would otherwise produce a P25 above a P75.
    """

    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                 l2: float = 1.0, max_iter: int = 80, eps: float = 1e-4) -> None:
        self.quantiles = tuple(quantiles)
        self.l2, self.max_iter, self.eps = l2, max_iter, eps

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantileRegression:
        self.scaler_ = _Standardizer().fit(X)
        Z = np.c_[np.ones(len(X)), self.scaler_.transform(X)]
        y = np.asarray(y, dtype=float)
        penalty = np.full(Z.shape[1], self.l2)
        penalty[0] = 0.0
        self.coefs_ = []
        for tau in self.quantiles:
            w = np.zeros(Z.shape[1])
            w[0] = float(np.quantile(y, tau))
            for _ in range(self.max_iter):
                r = y - Z @ w
                # IRLS weights for the check function: |r| in the denominator,
                # floored so a residual at exactly zero cannot blow up.
                wt = np.where(r >= 0, tau, 1 - tau) / np.maximum(np.abs(r), self.eps)
                A = (Z * wt[:, None]).T @ Z + np.diag(penalty)
                b = (Z * wt[:, None]).T @ y
                try:
                    w_new = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    w_new = np.linalg.lstsq(A, b, rcond=None)[0]
                if np.max(np.abs(w_new - w)) < 1e-7:
                    w = w_new
                    break
                w = w_new
            self.coefs_.append(w)
        self.coefs_ = np.array(self.coefs_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = np.c_[np.ones(len(X)), self.scaler_.transform(X)]
        q = Z @ self.coefs_.T                      # (n, n_quantiles)
        return np.sort(q, axis=1)                  # enforce monotonicity


class HARVolatility:
    """HAR-RV: tomorrow's variance from daily, weekly and monthly realised.

    The canonical volatility baseline-beater. Fitted in log-variance so the
    forecast cannot go negative and the errors are closer to symmetric.
    """

    def __init__(self, l2: float = 0.1) -> None:
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> HARVolatility:
        # y is realised |move|; model log of its square.
        ly = np.log(np.maximum(np.asarray(y, dtype=float) ** 2, 1e-10))
        self.inner_ = RidgeRegression(self.l2).fit(np.log(np.maximum(X, 1e-6)), ly)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        var = np.exp(np.clip(self.inner_.predict(np.log(np.maximum(X, 1e-6))), -30, 10))
        return np.sqrt(var)


def logistic_factory(l2: float = 1.0):
    return lambda: RidgeLogistic(l2=l2)


def ridge_factory(l2: float = 1.0):
    return lambda: RidgeRegression(l2=l2)


def quantile_factory(quantiles=(0.1, 0.25, 0.5, 0.75, 0.9), l2: float = 1.0):
    return lambda: QuantileRegression(quantiles=quantiles, l2=l2)


class GradientBoostedStumps:
    """Small gradient-boosted tree ensemble, logistic loss.

    Present so that "we tested nonlinearity" is a statement of fact rather
    than an assumption. Depth is deliberately tiny and the tree count low:
    with ~600 usable observations and eight features, a deep ensemble would
    fit noise, and the point of the ablation is to give nonlinearity a fair
    chance, not to manufacture a winner.
    """

    def __init__(self, n_trees: int = 60, depth: int = 2, lr: float = 0.06,
                 min_leaf: int = 25, l2: float = 1.0) -> None:
        self.n_trees, self.depth, self.lr = n_trees, depth, lr
        self.min_leaf, self.l2 = min_leaf, l2

    # -- one regression tree on the gradient ------------------------------
    def _split(self, X, g, h, idx, depth):
        if depth == 0 or len(idx) < 2 * self.min_leaf:
            return {"leaf": -g[idx].sum() / (h[idx].sum() + self.l2)}
        best = None
        parent = g[idx].sum() ** 2 / (h[idx].sum() + self.l2)
        for j in range(X.shape[1]):
            v = X[idx, j]
            order = np.argsort(v, kind="mergesort")
            sv, sg, sh = v[order], g[idx][order], h[idx][order]
            cg, ch = np.cumsum(sg), np.cumsum(sh)
            tot_g, tot_h = cg[-1], ch[-1]
            for k in range(self.min_leaf - 1, len(idx) - self.min_leaf):
                if sv[k] == sv[k + 1]:
                    continue
                gain = (cg[k] ** 2 / (ch[k] + self.l2)
                        + (tot_g - cg[k]) ** 2 / (tot_h - ch[k] + self.l2) - parent)
                if best is None or gain > best[0]:
                    best = (gain, j, (sv[k] + sv[k + 1]) / 2)
        if best is None or best[0] <= 0:
            return {"leaf": -g[idx].sum() / (h[idx].sum() + self.l2)}
        _, j, thr = best
        left = idx[X[idx, j] <= thr]
        right = idx[X[idx, j] > thr]
        if len(left) < self.min_leaf or len(right) < self.min_leaf:
            return {"leaf": -g[idx].sum() / (h[idx].sum() + self.l2)}
        return {"j": j, "thr": thr,
                "l": self._split(X, g, h, left, depth - 1),
                "r": self._split(X, g, h, right, depth - 1)}

    @staticmethod
    def _apply(node, X):
        out = np.empty(len(X))
        stack = [(node, np.arange(len(X)))]
        while stack:
            nd, idx = stack.pop()
            if "leaf" in nd:
                out[idx] = nd["leaf"]
                continue
            m = X[idx, nd["j"]] <= nd["thr"]
            stack.append((nd["l"], idx[m]))
            stack.append((nd["r"], idx[~m]))
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostedStumps:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        p0 = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        self.base_ = np.log(p0 / (1 - p0))
        self.trees_ = []
        f = np.full(len(y), self.base_)
        for _ in range(self.n_trees):
            p = 1.0 / (1.0 + np.exp(-np.clip(f, -35, 35)))
            g, h = p - y, np.clip(p * (1 - p), 1e-6, None)
            tree = self._split(X, g, h, np.arange(len(y)), self.depth)
            self.trees_.append(tree)
            f += self.lr * self._apply(tree, X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        f = np.full(len(X), self.base_)
        for t in self.trees_:
            f += self.lr * self._apply(t, X)
        return 1.0 / (1.0 + np.exp(-np.clip(f, -35, 35)))


def gbm_factory(**kw):
    return lambda: GradientBoostedStumps(**kw)
