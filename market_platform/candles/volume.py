"""Index volume proxy: an index has no traded volume, so the order-block
core uses its front-month future's 1m volume (exactly what the NIFTY system
did with FUT1). One join function serves live and replay, so they cannot
diverge:

    bars = join_index_volume(bars_of_one_minute, proxy)   # {key: Bar}

`proxy` maps an index key to its future key for the session. Live, the
future is the planner's T1 contract; in replay it is the nearest-month
future of that underlying present in `bars_1m` for the session
(`front_future_keys`). If the future has no bar for the minute, the index
bar's volume is None (rvol not applicable) — never a guess.
"""

from __future__ import annotations

import re
from dataclasses import replace

_MON = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
                                    "OCT", "NOV", "DEC"), 1)}
_FUT = re.compile(r"^(?:NFO|BFO):(?P<u>[A-Z&-]+?)(?P<yy>\d{2})(?P<mon>[A-Z]{3})FUT$")


def fut_sort_key(key: str) -> tuple[str, int, int] | None:
    m = _FUT.match(key)
    if not m or m.group("mon") not in _MON:
        return None
    return m.group("u"), int(m.group("yy")), _MON[m.group("mon")]


def front_future_keys(keys: list[str], underlyings: dict[str, str]) -> dict[str, str]:
    """index_key → nearest-month future key among `keys` for each underlying
    (`underlyings`: index_key → derivative underlying, e.g. 'NIFTY')."""
    by_u: dict[str, list[tuple[int, int, str]]] = {}
    for k in keys:
        p = fut_sort_key(k)
        if p:
            by_u.setdefault(p[0], []).append((p[1], p[2], k))
    out = {}
    for idx, u in underlyings.items():
        if u in by_u:
            out[idx] = min(by_u[u])[2]
    return out


def join_index_volume(bars: dict, proxy: dict[str, str]) -> dict:
    if not proxy:
        return bars
    out = dict(bars)
    for idx, fut in proxy.items():
        b = out.get(idx)
        if b is None:
            continue
        f = bars.get(fut)
        out[idx] = replace(b, volume=f.volume if f is not None and f.ts == b.ts else None)
    return out
