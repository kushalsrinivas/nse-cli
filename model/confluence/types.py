"""Datatypes for intraday confluence setup evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ConditionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NA = "n/a"


@dataclass(frozen=True)
class ConditionCheck:
    name: str
    status: ConditionStatus
    detail: str = ""


@dataclass
class SuggestedContract:
    symbol: str
    strike: float
    is_call: bool
    expiry: str
    entry_price: float | None
    delta: float | None
    dte: int


@dataclass
class ConfluenceSetupResult:
    setup_id: str                       # A, B, C
    title: str
    decision: str                       # GO, NO-GO, WATCH
    direction: str                        # bullish, bearish, neutral
    confidence_score: float             # pct conditions passed
    conditions: list[ConditionCheck] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    decision_rationale: str = ""
    suggested: SuggestedContract | None = None
    notes: str = ""

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.conditions if c.status is ConditionStatus.PASS)

    @property
    def applicable_count(self) -> int:
        return sum(1 for c in self.conditions if c.status is not ConditionStatus.NA)

    @property
    def go(self) -> bool:
        return self.decision == "GO"


@dataclass
class ConfluenceReport:
    run_id: str
    timestamp: str
    trade_date: str
    spot: float
    setups: list[ConfluenceSetupResult] = field(default_factory=list)
    vix_level: float | None = None
    vix_change: float | None = None
    error: str | None = None
