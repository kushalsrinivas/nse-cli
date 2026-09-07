"""Types for the overnight setup engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SetupConditionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NA = "n/a"


class SetupDecision(str, Enum):
    GO = "GO"
    NO_GO = "NO-GO"
    WATCH = "WATCH"


@dataclass(frozen=True)
class SetupCondition:
    name: str
    status: SetupConditionStatus
    detail: str = ""


@dataclass
class OvernightSetupResult:
    setup_id: str                    # "ON-A" ..
    name: str
    direction: str                   # "bullish" | "bearish" | "neutral"
    decision: SetupDecision
    confidence: float                # 0-100: PASS share of evaluated conditions
    conditions: list[SetupCondition] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    rationale: str = ""
    suggested: str = "none"          # structure hint; EV engine still decides
    blocks: bool = False             # True: a NO-GO here should block the night
    kind: str = "candidate"          # "candidate" | "filter"

    @property
    def short_label(self) -> str:
        if self.decision is SetupDecision.GO:
            return "GO"
        if self.blocks:
            return "BLOCKS"
        if self.decision is SetupDecision.WATCH and self.kind == "filter":
            return "clear"
        return self.decision.value.lower()


@dataclass
class OvernightSetupsReport:
    results: list[OvernightSetupResult] = field(default_factory=list)

    def by_id(self, setup_id: str) -> OvernightSetupResult | None:
        return next((r for r in self.results if r.setup_id == setup_id), None)

    @property
    def blockers(self) -> list[OvernightSetupResult]:
        return [r for r in self.results
                if r.blocks and r.decision is SetupDecision.NO_GO]


def make_condition(name: str, ok: bool | None, detail: str = "") -> SetupCondition:
    """None = not evaluable (missing data) -> N/A, never a FAIL."""
    status = SetupConditionStatus.PASS if ok is True else (
        SetupConditionStatus.FAIL if ok is False else SetupConditionStatus.NA)
    return SetupCondition(name, status, detail)


def confidence_of(conditions: list[SetupCondition]) -> float:
    evaluated = [c for c in conditions
                 if c.status is not SetupConditionStatus.NA]
    if not evaluated:
        return 0.0
    passed = sum(1 for c in evaluated
                 if c.status is SetupConditionStatus.PASS)
    return round(passed / len(evaluated) * 100, 1)
