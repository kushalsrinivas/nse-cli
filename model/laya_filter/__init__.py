"""Laya trade filter: a veto-only second opinion on rule-generated setups.

Laya (convaiinnovations/laya) is a non-autoregressive encoder that answers
typed questions (choice / score / noul) about a text state in one forward
pass. Here it reads a setup's own condition trail and market context and
answers "should this setup be trusted?" style questions.

Guardrails, same as the breadth layer:
- It can only veto a GO, never create or upgrade a trade.
- It ships in SHADOW mode: verdicts are computed and journaled but do not
  change any decision unless the caller passes `enforce=True`.
- The base checkpoints are near chance zero-shot on out-of-domain decisions
  (Laya's own README). Treat every verdict as unvalidated until
  `model_cli.py laya-eval` shows vetoed setups underperform kept ones.
"""

from model.laya_filter.filter import (
    LayaFilter,
    LayaUnavailable,
    LayaVerdict,
    VetoPolicy,
    apply_verdicts,
)
from model.laya_filter.questions import SETUP_QUESTIONS
from model.laya_filter.state import SetupState, setup_state

__all__ = [
    "SETUP_QUESTIONS",
    "LayaFilter",
    "LayaUnavailable",
    "LayaVerdict",
    "SetupState",
    "VetoPolicy",
    "apply_verdicts",
    "setup_state",
]
