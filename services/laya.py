"""Laya filter workflow: judge a confluence report's setups, optionally
enforce vetoes and journal the verdicts. No console output; callers render.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class LayaRun:
    report: object = None                 # possibly with vetoed GOs downgraded
    verdicts: list = field(default_factory=list)
    enforced: bool = False
    notices: list[tuple[str, str]] = field(default_factory=list)


def judge_confluence(report, *, enforce: bool = False, journal=False,
                     laya_filter=None, events: list[str] | None = None) -> LayaRun:
    """Run Laya over every setup in `report`.

    `journal`: False (dry-run), True (shared journal.db) or a LayaJournal.
    Laya being unavailable is a notice, never an exception: the report
    comes back untouched.
    """
    from model.laya_filter import (
        LayaFilter,
        LayaUnavailable,
        apply_verdicts,
        setup_state,
    )

    run = LayaRun(report=report, enforced=enforce)
    if getattr(report, "error", None) or not getattr(report, "setups", None):
        run.notices.append(("warn", "laya: no setups to judge"))
        return run

    flt = laya_filter or LayaFilter()
    ctx = {"spot": report.spot, "vix": report.vix_level,
           "vix_change": report.vix_change, "events": list(events or [])}
    states = [setup_state(s, source="confluence", context=ctx)
              for s in report.setups]
    try:
        run.verdicts = flt.judge_many(states)
    except LayaUnavailable as exc:
        run.notices.append(("warn", f"laya unavailable: {exc}"))
        return run

    errors = [v for v in run.verdicts if v.error]
    if errors:
        run.notices.append(("warn", f"laya failed on {len(errors)} setup(s); "
                                    "those are kept (a filter never blocks by failing)"))
    run.report = replace(report, setups=apply_verdicts(
        report.setups, run.verdicts, enforce=enforce))
    if not enforce and any(v.veto for v in run.verdicts):
        run.notices.append(("info", "laya: shadow mode — vetoes shown, not "
                                    "applied (pass --enforce to apply)"))

    if journal is not False:
        from journal.laya_db import LayaJournal
        lj = LayaJournal() if journal is True else journal
        for v in run.verdicts:
            lj.add(report.run_id, v, enforced=enforce)
    return run


def judge_card(state, *, enforce: bool = False, journal=False, run_id: str,
               apply_veto=None, laya_filter=None) -> LayaRun:
    """Judge one whole decision card (overnight / premarket).

    `apply_veto(verdict)` enforces a veto on the caller's card; it runs only
    when `enforce` is set. Unavailable Laya is a notice, never an exception.
    """
    from model.laya_filter import LayaFilter, LayaUnavailable

    run = LayaRun(enforced=enforce)
    flt = laya_filter or LayaFilter()
    try:
        verdict = flt.judge(state)
    except LayaUnavailable as exc:
        run.notices.append(("warn", f"laya unavailable: {exc}"))
        return run
    run.verdicts = [verdict]
    if verdict.error:
        run.notices.append(("warn", f"laya failed ({verdict.error}); card kept"))
    elif verdict.veto and enforce and apply_veto is not None:
        apply_veto(verdict)
        run.notices.append(("warn", "laya veto ENFORCED: " + "; ".join(verdict.reasons)))
    elif verdict.veto:
        run.notices.append(("info", "laya would veto (shadow mode, not applied): "
                                    + "; ".join(verdict.reasons)))
    if journal is not False:
        from journal.laya_db import LayaJournal
        lj = LayaJournal() if journal is True else journal
        lj.add(run_id, verdict, enforced=enforce)
    return run
