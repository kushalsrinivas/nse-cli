"""Overnight setup engine: concrete, auditable pre-close checklists.

Mirrors the intraday confluence format (named setups, PASS/FAIL/NA
conditions, GO/NO-GO/WATCH) but every condition is overnight-native and
breadth-aware. Two setups are directional candidates (ON-A, ON-C), two are
stand-aside filters (ON-B, ON-D) that can only veto, never create, a trade
— same philosophy as `model/breadth/integration.py`.
"""

from __future__ import annotations
