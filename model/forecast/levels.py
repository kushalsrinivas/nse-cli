"""L5 — support and resistance as probabilities.

The audited engine treated the 60-day high and low as "levels". On a
breakdown day that put resistance 7.2% away and support at `None`, and the
card announced "breakdown into open space" — the level engine going silent
exactly when levels matter.

Here a level is scored, not asserted. Two distinct questions get two
distinct answers, which is the distinction the old output never made:

    P(touch)        does price trade through it at any point?
    P(close beyond) does it still hold at the close?

P(touch) is always the larger of the two, and the gap between them is the
rejection case — the level is reached and defended. Both come from the
forecast distribution and an empirical touch curve measured from history,
rather than from a Brownian assumption the tape does not obey.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TouchCurve:
    """P(the session reaches d sigma away) as measured, by direction.

    Estimated from (open, high, low, close) history: for each session,
    how far did price travel in sigma units, and how often did it get
    at least d sigma away? A Brownian reflection argument gives 2*P(close
    beyond) for a driftless walk; real sessions do not match that, so this
    measures it instead.
    """

    grid: tuple[float, ...]
    up: tuple[float, ...]
    down: tuple[float, ...]
    n: int

    @classmethod
    def from_history(cls, mfe_pct, mae_pct, sigma_pct,
                     grid=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
                     ) -> TouchCurve:
        mfe = np.asarray(mfe_pct, dtype=float)
        mae = np.asarray(mae_pct, dtype=float)
        s = np.asarray(sigma_pct, dtype=float)
        ok = np.isfinite(mfe) & np.isfinite(mae) & np.isfinite(s) & (s > 1e-9)
        up_z, dn_z = mfe[ok] / s[ok], np.abs(mae[ok]) / s[ok]
        return cls(grid=tuple(grid),
                   up=tuple(float((up_z >= d).mean()) for d in grid),
                   down=tuple(float((dn_z >= d).mean()) for d in grid),
                   n=int(ok.sum()))

    def p_touch(self, distance_sigma: float, upward: bool) -> float:
        d = abs(distance_sigma)
        curve = self.up if upward else self.down
        if d <= self.grid[0]:
            return float(curve[0])
        if d >= self.grid[-1]:
            # Exponential decay past the measured grid rather than a hard 0.
            last, prev = curve[-1], curve[-2]
            rate = max(prev, 1e-6) / max(last, 1e-9)
            steps = (d - self.grid[-1]) / (self.grid[-1] - self.grid[-2])
            return float(max(last / max(rate ** steps, 1e-9), 0.0))
        return float(np.interp(d, self.grid, curve))


@dataclass
class LevelAssessment:
    """One price level, scored against the forecast."""

    level: float
    label: str
    distance_pct: float
    distance_sigma: float
    upward: bool
    p_touch: float
    p_close_beyond: float
    p_reject: float                  # touched but not held through the close
    sources: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.p_close_beyond >= 0.40:
            return "likely to break"
        if self.p_touch >= 0.50 and self.p_reject >= 0.55:
            return "likely to reject"
        if self.p_touch < 0.25:
            return "out of reach today"
        return "contested"

    def line(self) -> str:
        return (f"{self.level:>9,.0f}  {self.label:<22} "
                f"{self.distance_pct:+6.2f}% ({self.distance_sigma:+.2f}σ)  "
                f"touch {self.p_touch:>5.0%}  break {self.p_close_beyond:>5.0%}  "
                f"reject {self.p_reject:>5.0%}  {self.verdict}")


def assess_level(level: float, spot: float, dist, curve: TouchCurve,
                 label: str = "", sources: list[str] | None = None
                 ) -> LevelAssessment:
    """Score one level against a `ConditionalDistribution` and touch curve."""
    dist_pct = (level - spot) / spot * 100.0
    upward = dist_pct >= 0
    sigma = max(dist.scale_pct, 1e-9)
    d_sigma = dist_pct / sigma
    p_close = dist.p_beyond(dist_pct)
    p_touch = max(curve.p_touch(d_sigma, upward), p_close)   # touch >= close
    return LevelAssessment(
        level=float(level), label=label or "level",
        distance_pct=round(dist_pct, 3), distance_sigma=round(d_sigma, 2),
        upward=upward, p_touch=round(p_touch, 4),
        p_close_beyond=round(p_close, 4),
        p_reject=round(max(p_touch - p_close, 0.0) / p_touch, 4) if p_touch > 1e-9 else 0.0,
        sources=list(sources or []))


def candidate_levels(frame, chain=None, spot: float | None = None) -> list[dict]:
    """Assemble levels worth scoring, each tagged with where it came from.

    Structure the old engine ignored entirely: the previous session's own
    high/low and close, the opening reference, round numbers traders
    actually watch, and — when a chain is available — the strikes carrying
    the most open interest, which is where dealer hedging concentrates.
    """
    out: list[dict] = []
    if frame is None or not len(frame):
        return out
    last = frame.iloc[-1]
    spot = float(spot if spot is not None else last["close"])

    out.append({"level": float(last["high"]), "label": "prev day high",
                "sources": ["PDH"]})
    out.append({"level": float(last["low"]), "label": "prev day low",
                "sources": ["PDL"]})
    out.append({"level": float(last["close"]), "label": "prev close",
                "sources": ["PDC"]})

    for n, tag in ((5, "5d"), (20, "20d"), (60, "60d")):
        if len(frame) >= n:
            w = frame.iloc[-n:]
            out.append({"level": float(w["high"].max()),
                        "label": f"{tag} high", "sources": [f"swing-{tag}"]})
            out.append({"level": float(w["low"].min()),
                        "label": f"{tag} low", "sources": [f"swing-{tag}"]})

    step = 100.0 if spot < 30_000 else 500.0
    for k in (-1, 1):
        out.append({"level": float(round(spot / step) * step + k * step),
                    "label": "round number", "sources": ["psychological"]})

    if chain is not None and getattr(chain, "rows", None):
        try:
            rows = sorted(chain.rows,
                          key=lambda r: -((r.call.open_interest or 0)
                                          + (r.put.open_interest or 0)))
            for r in rows[:3]:
                side = ("call wall" if (r.call.open_interest or 0)
                        >= (r.put.open_interest or 0) else "put wall")
                out.append({"level": float(r.strike), "label": f"OI {side}",
                            "sources": ["open-interest"]})
        except (AttributeError, TypeError):
            pass

    # Merge levels within 0.1% of each other, keeping every source tag.
    merged: list[dict] = []
    for cand in sorted(out, key=lambda d: d["level"]):
        if merged and abs(cand["level"] - merged[-1]["level"]) / spot < 0.001:
            merged[-1]["sources"].extend(cand["sources"])
            merged[-1]["label"] = f"{merged[-1]['label']} / {cand['label']}"
        else:
            merged.append(dict(cand))
    return merged
