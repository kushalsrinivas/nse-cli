"""Run Laya over setup states and turn its answers into veto verdicts.

`laya` (and its torch/transformers stack) is an optional dependency: it is
imported lazily, and a missing install or failed checkpoint download raises
LayaUnavailable once, which callers turn into a notice. A per-setup
inference error degrades into a verdict with `error` set and no veto —
the filter can never block a trade by failing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace

from model.laya_filter.questions import SETUP_QUESTIONS
from model.laya_filter.state import SetupState

#: English root checkpoint. States here are English, so the Router's
#: script/language detection buys nothing. Override with LAYA_MODEL /
#: LAYA_SUBFOLDER to drop in a checkpoint fine-tuned on journal outcomes.
DEFAULT_MODEL = "convaiinnovations/laya"


class LayaUnavailable(RuntimeError):
    """laya is not installed, or the checkpoint could not be loaded."""


@dataclass(frozen=True)
class VetoPolicy:
    """Thresholds on Laya's answers. Any tripped rule vetoes a GO.

    Defaults are deliberately loose — the veto should fire only when the
    model is emphatic — and are NOT fitted. Tighten them only after
    `laya-eval` has enough settled rows to say which rules carry signal.
    """
    max_fail: float = 0.70           # P(likely_to_fail) above this vetoes
    min_execute: float = 0.30        # P(warrants_execution) below this vetoes
    max_conflict: float = 0.75       # P(conflicting_signal) above this vetoes
    direction_min_conf: float = 0.70  # opposite direction at >= this vetoes
    min_quality: float | None = None  # expected setup_quality (0-3); off

    def reasons(self, answers: dict, direction: str) -> list[str]:
        out: list[str] = []
        p_fail = _noul(answers, "likely_to_fail")
        if p_fail is not None and p_fail > self.max_fail:
            out.append(f"P(fail)={p_fail:.2f} > {self.max_fail:.2f}")
        p_exec = _noul(answers, "warrants_execution")
        if p_exec is not None and p_exec < self.min_execute:
            out.append(f"P(execute)={p_exec:.2f} < {self.min_execute:.2f}")
        p_conf = _noul(answers, "conflicting_signal")
        if p_conf is not None and p_conf > self.max_conflict:
            out.append(f"P(conflict)={p_conf:.2f} > {self.max_conflict:.2f}")
        d = answers.get("direction") or {}
        opposite = {"bullish": "bearish", "bearish": "bullish"}.get(direction)
        if opposite and d.get("choice") == opposite:
            conf = float(d.get("confidence") or 0.0)
            if conf >= self.direction_min_conf:
                out.append(f"reads {opposite} ({conf:.2f}) against a "
                           f"{direction} setup")
        if self.min_quality is not None:
            q = (answers.get("setup_quality") or {}).get("score")
            if q is not None and q < self.min_quality:
                out.append(f"quality {q:.2f} < {self.min_quality:.2f}")
        return out


@dataclass
class LayaVerdict:
    source: str
    setup_id: str
    decision: str                    # the engine's decision, before Laya
    direction: str
    answers: dict = field(default_factory=dict)
    would_veto: bool = False         # policy tripped (computed for every setup)
    reasons: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    model: str = ""
    error: str | None = None

    @property
    def veto(self) -> bool:
        """Only a GO can be vetoed; NO-GO/WATCH are already not trades."""
        return self.would_veto and self.decision == "GO" and not self.error

    @property
    def p_fail(self) -> float | None:
        return _noul(self.answers, "likely_to_fail")

    @property
    def p_execute(self) -> float | None:
        return _noul(self.answers, "warrants_execution")

    @property
    def quality(self) -> float | None:
        return (self.answers.get("setup_quality") or {}).get("score")

    @property
    def laya_direction(self) -> str | None:
        return (self.answers.get("direction") or {}).get("choice")

    @property
    def regime(self) -> str | None:
        return (self.answers.get("market_regime") or {}).get("choice")


class LayaFilter:
    """Lazy-loading wrapper: one agent, one forward pass per setup.

    Pass `agent` (anything with `predict(state, questions) -> {"answers":
    ...}`) to skip loading — tests inject a fake; an app that already built
    a Laya agent can share it.
    """

    def __init__(self, agent=None, *, model_id: str | None = None,
                 subfolder: str | None = None, device: str | None = None,
                 policy: VetoPolicy | None = None,
                 questions: dict | None = None) -> None:
        self._agent = agent
        self.model_id = model_id or os.environ.get("LAYA_MODEL") or DEFAULT_MODEL
        self.subfolder = subfolder or os.environ.get("LAYA_SUBFOLDER") or None
        self.device = device or os.environ.get("LAYA_DEVICE") or None
        self.policy = policy or VetoPolicy()
        self.questions = questions or SETUP_QUESTIONS

    @property
    def model_label(self) -> str:
        return self.model_id + (f"/{self.subfolder}" if self.subfolder else "")

    @property
    def agent(self):
        if self._agent is None:
            try:
                import laya
            except ImportError as exc:
                raise LayaUnavailable(
                    "laya is not installed — pip install -e '.[laya]' "
                    "(pulls torch + transformers)") from exc
            try:
                self._agent = laya.load(self.model_id, device=self.device,
                                        subfolder=self.subfolder)
            except Exception as exc:
                raise LayaUnavailable(
                    f"could not load {self.model_label}: {exc}") from exc
        return self._agent

    def judge(self, state: SetupState) -> LayaVerdict:
        verdict = LayaVerdict(source=state.source, setup_id=state.setup_id,
                              decision=state.decision,
                              direction=state.direction,
                              model=self.model_label)
        agent = self.agent            # LayaUnavailable propagates: fail once, loudly
        t0 = time.perf_counter()
        try:
            out = agent.predict(state.as_text(), self.questions)
        except Exception as exc:     # one bad setup must not sink the batch
            verdict.error = str(exc)
            return verdict
        verdict.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        verdict.answers = dict(out.get("answers") or {})
        verdict.reasons = self.policy.reasons(verdict.answers, state.direction)
        verdict.would_veto = bool(verdict.reasons)
        return verdict

    def judge_many(self, states: list[SetupState]) -> list[LayaVerdict]:
        return [self.judge(s) for s in states]


def apply_verdicts(results: list, verdicts: list[LayaVerdict], *,
                   enforce: bool = False) -> list:
    """Return results with vetoed GOs downgraded to WATCH (copies, not mutation).

    Shadow mode (`enforce=False`, the default) returns `results` unchanged:
    verdicts are information only. Nothing is ever upgraded.
    """
    if not enforce:
        return list(results)
    by_id = {v.setup_id: v for v in verdicts}
    out = []
    for r in results:
        v = by_id.get(str(r.setup_id))
        if v is None or not v.veto:
            out.append(r)
            continue
        out.append(_downgrade(r, "laya veto: " + "; ".join(v.reasons)))
    return out


def _downgrade(result, reason: str):
    dec = result.decision
    watch = "WATCH" if isinstance(dec, str) and not hasattr(dec, "value") \
        else type(dec)("WATCH")
    changes = {"decision": watch,
               "blocked_reasons": [reason, *list(result.blocked_reasons)]}
    if hasattr(result, "decision_rationale"):
        changes["decision_rationale"] = reason
    elif hasattr(result, "rationale"):
        changes["rationale"] = reason
    return replace(result, **changes)


def _noul(answers: dict, key: str) -> float | None:
    v = (answers.get(key) or {}).get("noul")
    return None if v is None else float(v)
