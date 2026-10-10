"""Correlation clusters and NIFTY-beta from daily returns (plan §7.3).

Rolling 60-session log-return correlation; instruments are linked when
their correlation is ≥ threshold (default 0.7) and clusters are the
connected components (single-linkage agglomeration at that threshold).
Recomputed weekly; stored in `correlation_clusters` with the as-of date,
so a backtest uses the clusters known at the time.
"""

from __future__ import annotations

import math
from datetime import date


def _returns(closes: list[float]) -> list[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            if closes[i - 1] > 0 and closes[i] > 0]


def _corr(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n < 20:
        return None
    a, b = a[-n:], b[-n:]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / math.sqrt(va * vb)


def beta(a: list[float], m: list[float]) -> float | None:
    n = min(len(a), len(m))
    if n < 20:
        return None
    a, m = a[-n:], m[-n:]
    mm = sum(m) / n
    var = sum((x - mm) ** 2 for x in m)
    if var <= 0:
        return None
    ma = sum(a) / n
    return round(sum((x - ma) * (y - mm) for x, y in zip(a, m, strict=True)) / var, 3)


def clusters(closes: dict[str, list[float]], *, threshold: float = 0.7,
             window: int = 60) -> dict[str, str]:
    """instrument_key → cluster label (the smallest key in its component)."""
    rets = {k: _returns(v[-(window + 1):]) for k, v in closes.items()}
    keys = sorted(k for k, r in rets.items() if len(r) >= 20)
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            c = _corr(rets[a], rets[b])
            if c is not None and c >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return {k: find(k) for k in keys}


def compute_and_store(market_conn, app_conn, keys: list[str], as_of: date, *,
                      benchmark: str = "NSE:NIFTY 50", threshold: float = 0.7,
                      window: int = 60) -> dict[str, str]:
    closes = {}
    for k in set(keys) | {benchmark}:
        rows = market_conn.execute("SELECT close FROM bars_1d WHERE instrument_key=? AND date<? "
                                   "ORDER BY date DESC LIMIT ?",
                                   (k, as_of.isoformat(), window + 1)).fetchall()
        if rows:
            closes[k] = [r[0] for r in reversed(rows)]
    cl = clusters({k: v for k, v in closes.items() if k != benchmark}, threshold=threshold,
                  window=window)
    mret = _returns(closes.get(benchmark, []))
    app_conn.executemany(
        "INSERT OR REPLACE INTO correlation_clusters (as_of, instrument_key, cluster, beta) "
        "VALUES (?,?,?,?)",
        [(as_of.isoformat(), k, c, beta(_returns(closes[k]), mret)) for k, c in cl.items()])
    app_conn.commit()
    return cl


def load(app_conn, as_of: date) -> dict[str, str]:
    row = app_conn.execute("SELECT MAX(as_of) FROM correlation_clusters WHERE as_of<=?",
                           (as_of.isoformat(),)).fetchone()
    if not row or not row[0]:
        return {}
    return {r[0]: r[1] for r in app_conn.execute(
        "SELECT instrument_key, cluster FROM correlation_clusters WHERE as_of=?", (row[0],))}
