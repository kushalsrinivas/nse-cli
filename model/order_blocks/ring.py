"""Absolute-index ring buffer (fixes platform audit A5: unbounded engine state).

The order-block core addresses bars, ATR and rvol by *absolute* index
(a zone remembers `bos_index`, a swing its `pivot_index`). `RingList`
keeps those indices valid while storing only the last `maxlen` items:

    r = RingList(3); for x in "abcde": r.append(x)
    len(r) == 5          # absolute length
    r[4] == "e"; r[2] == "c"; r[1]  -> IndexError (evicted)
    r[0:5] == ["c", "d", "e"]        # slices are clamped to what is held

Clamping a slice means a look-back that reaches past the ring sees the
data as if history started there — exactly what a fresh engine would see.
The ring is sized well beyond every look-back the core uses (leg ≤ 6 bars,
source/sweep ≤ 10, swing references ≤ 60), so the result is identical
except for swings older than the ring, which are dropped anyway.
"""

from __future__ import annotations

from collections import deque


class RingList:
    __slots__ = ("_d", "_start", "maxlen")

    def __init__(self, maxlen: int, items=()) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")
        self.maxlen = maxlen
        self._d: deque = deque(maxlen=maxlen)
        self._start = 0                      # absolute index of _d[0]
        for x in items:
            self.append(x)

    def append(self, x) -> None:
        if len(self._d) == self.maxlen:
            self._start += 1
        self._d.append(x)

    def __len__(self) -> int:
        return self._start + len(self._d)

    @property
    def first_index(self) -> int:
        return self._start

    def held(self) -> int:
        return len(self._d)

    def _abs(self, i: int) -> int:
        return len(self) + i if i < 0 else i

    def __getitem__(self, i):
        if isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            if step != 1:
                raise ValueError("RingList slices support step 1 only")
            start = max(start, self._start)
            if stop <= start:
                return []
            return [self._d[j - self._start] for j in range(start, stop)]
        j = self._abs(i)
        if j < self._start or j >= len(self):
            raise IndexError(f"index {i} not held (ring holds {self._start}..{len(self) - 1})")
        return self._d[j - self._start]

    def __iter__(self):
        return iter(self._d)

    def __bool__(self) -> bool:
        return len(self._d) > 0

    def __repr__(self) -> str:
        return f"RingList(maxlen={self.maxlen}, held={len(self._d)}, len={len(self)})"
