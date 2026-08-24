"""Dynamic indicator weighting.

Three layers compose the effective weight of each indicator group:

1. Baseline weights — sensible priors per information type (trend 25%,
   MACD 15%, momentum/RSI 10%, Bollinger 10%, volume 15%, VWAP 10%, S/R 15%).
2. Regime multipliers — trending regimes lean on trend/MACD; sideways leans
   on mean-reversion tools; low volume penalizes everything via confidence.
3. Learned reliability — persisted from backtests: groups whose signals
   historically led to winning trades gain share, losers lose it.

Effective weights are always renormalized to 100%. If an indicator is
missing (e.g. VWAP on daily data) its weight redistributes to survivors.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from config import SETTINGS
from model.regime import MarketRegime

BASELINE_WEIGHTS: dict[str, float] = {
    "trend": 0.25,
    "macd": 0.15,
    "momentum": 0.10,
    "bollinger": 0.10,
    "volume": 0.15,
    "vwap": 0.10,
    "sr": 0.15,
}

REGIME_MULTIPLIERS: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.TRENDING_BULL: {"trend": 1.35, "macd": 1.30, "volume": 1.10,
                                 "bollinger": 0.70, "sr": 0.85},
    MarketRegime.TRENDING_BEAR: {"trend": 1.35, "macd": 1.30, "volume": 1.10,
                                 "bollinger": 0.70, "sr": 0.85},
    MarketRegime.SIDEWAYS: {"bollinger": 1.45, "sr": 1.40, "vwap": 1.20,
                            "trend": 0.75, "macd": 0.80},
    MarketRegime.HIGH_VOLATILITY: {"bollinger": 1.25, "sr": 1.15,
                                   "trend": 0.85, "macd": 0.90},
    MarketRegime.LOW_VOLATILITY: {"bollinger": 1.20, "sr": 1.10, "volume": 0.80},
}


@dataclass(frozen=True)
class GroupReliability:
    """Backtest-derived performance of one indicator group."""

    group: str
    win_rate: float = 0.50          # fraction of agreeing setups that won
    avg_return_pct: float = 0.0     # mean forward return when it voted
    samples: int = 0

    @property
    def reliability(self) -> str:
        if self.samples < 30 or abs(self.win_rate - 0.5) < 0.03:
            return "LOW"
        if self.win_rate >= 0.58 and self.avg_return_pct > 0:
            return "HIGH"
        if self.win_rate >= 0.53:
            return "MEDIUM"
        return "LOW"

    def multiplier(self, max_shift: float = 0.35) -> float:
        """Reliability → weight multiplier in [1-max_shift, 1+max_shift].

        Scales with sqrt(samples): a 65% hit rate over 12 trades should move
        weights far less than the same rate over 300 trades.
        """
        if self.samples <= 0:
            return 1.0
        edge = (self.win_rate - 0.5) * 2 + math.copysign(
            min(abs(self.avg_return_pct), 0.5), self.avg_return_pct
        )
        evidence = min(1.0, math.sqrt(self.samples / 200))
        shift = max(-max_shift, min(max_shift, edge * max_shift)) * evidence
        return 1.0 + shift


def load_learned_weights(path: Path | None = None) -> dict[str, GroupReliability] | None:
    path = Path(path or SETTINGS.learned_weights_path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return {
            g: GroupReliability(group=g, win_rate=v["win_rate"],
                                avg_return_pct=v["avg_return"],
                                samples=int(v["samples"]))
            for g, v in raw.get("groups", {}).items()
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_learned_weights(reliabilities: dict[str, GroupReliability],
                         meta: dict | None = None,
                         path: Path | None = None) -> None:
    path = Path(path or SETTINGS.learned_weights_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta or {},
        "groups": {
            r.group: {"win_rate": round(r.win_rate, 4),
                      "avg_return": round(r.avg_return_pct, 4),
                      "samples": r.samples}
            for r in reliabilities.values()
        },
    }
    path.write_text(json.dumps(payload, indent=2))


@dataclass(frozen=True)
class WeightSet:
    weights: dict[str, float]
    regime: MarketRegime | None = None
    learned: bool = False

    def for_group(self, group: str) -> float:
        return self.weights.get(group, 0.0)


def compute_effective_weights(
    regime: MarketRegime | None = None,
    available_groups: set[str] | None = None,
    learned: dict[str, GroupReliability] | None = None,
) -> WeightSet:
    available_groups = available_groups or set(BASELINE_WEIGHTS)
    weights = {g: w for g, w in BASELINE_WEIGHTS.items()}

    if regime is not None:
        mults = REGIME_MULTIPLIERS.get(regime, {})
        for g in weights:
            weights[g] *= mults.get(g, 1.0)

    used_learning = False
    if learned:
        for g, rel in learned.items():
            if g in weights:
                weights[g] *= rel.multiplier()
                used_learning = True

    # Drop unavailable groups, renormalize to exactly 1.0 over what survives.
    weights = {g: w for g, w in weights.items() if g in available_groups}
    total = sum(weights.values())
    if total <= 0:
        weights = {g: BASELINE_WEIGHTS[g] for g in weights}
        total = sum(weights.values())
    normalized = {g: round(w / total, 4) for g, w in weights.items()}
    return WeightSet(weights=normalized, regime=regime, learned=used_learning)
