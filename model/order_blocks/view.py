"""Rich renderers for order-block cards, zones, journal and backtests."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_DECISION = {"GO": "bold green", "WATCH": "yellow", "NO-GO": "red", "SHADOW": "cyan"}
_STATUS = {"pass": "green", "fail": "red", "n/a": "dim"}


def zones_table(zones, title: str = "Live order-block zones") -> Table:
    t = Table(title=title)
    for col in ("TF", "Dir", "Kind", "Zone", "BOS bar", "Status", "Age", "rvol",
                "FVG", "Sweep"):
        t.add_column(col)
    for z in zones:
        t.add_row(z.timeframe, z.direction, z.kind,
                  f"{z.zone_low:,.1f}–{z.zone_high:,.1f}",
                  z.bos_bar_ts.strftime("%m-%d %H:%M"), z.status, str(z.bars_alive),
                  f"{z.rvol:.2f}" if z.rvol is not None else "n/a",
                  "yes" if z.fvg_low is not None else "—",
                  f"{z.swept_level:,.1f}" if z.swept_level is not None else "—")
    return t


def setup_card(ev) -> Panel:
    s, sig = ev.setup, ev.signal
    style = _DECISION.get(ev.decision, "white")
    head = Text.assemble(
        (f"{ev.decision}", style), "  ",
        f"{s.horizon} · {s.plan.direction} · {s.zone.timeframe} {s.zone.kind} · "
        f"score {s.score.total:.0f}/100")
    plan = Table.grid(padding=(0, 2))
    plan.add_row("Underlying", f"entry {s.plan.u_entry:,.1f}  stop {s.plan.u_stop:,.1f}  "
                 f"target {s.plan.u_target:,.1f} ({s.plan.target_source})  R:R {s.plan.u_rr:.2f}")
    plan.add_row("Zone", f"{s.zone.zone_low:,.1f}–{s.zone.zone_high:,.1f}  "
                 f"disp {s.zone.disp_body_atr:.2f} ATR  HTF {s.htf_trend}")
    plan.add_row("Score", "  ".join(f"{k} {v:g}" for k, v in s.score.components.items())
                 + (f"  (N/A: {', '.join(s.score.na)})" if s.score.na else ""))
    choice = ev.selection.choice
    if choice is not None:
        plan.add_row("Structure", f"{choice.name}  net {choice.o_entry:+.2f}/unit  "
                     f"EV ₹{choice.ev.ev_model:,.0f}/lot  stop {choice.o_stop}  "
                     f"target {choice.o_target}  lot {choice.lot_size}")
        for q, side in zip(choice.legs, choice.sides, strict=True):
            plan.add_row("", f"{'BUY ' if side > 0 else 'SELL'} {q.tradingsymbol}  "
                         f"{q.bid}/{q.ask}  iv {q.iv}")
    elif ev.selection.rejected:
        plan.add_row("Structure", "none — " + "; ".join(ev.selection.rejected[:3]))
    if ev.sizing is not None:
        z = ev.sizing
        plan.add_row("Risk", (f"{z.lots} lots · risk ₹{z.risk_rupees:,.0f} · stress "
                              f"₹{z.stress_loss_per_lot:,.0f}/lot · {z.bound_by}")
                     if z.allowed else f"blocked: {z.reason}")
    if sig.p_win_calibrated is not None:
        plan.add_row("P(win)", f"{sig.p_win_calibrated:.2f} (calibrated)")
    else:
        plan.add_row("P(win)", "not calibrated — score is a ranking only")
    gates = Table(show_header=False, box=None)
    for c in ev.checks:
        st = c.status.value
        gates.add_row(Text(st.upper(), _STATUS.get(st, "white")), c.name, c.detail or "")
    parts = [head, plan, gates]
    if ev.reasons:
        parts.append(Text("Blocked: " + "; ".join(ev.reasons), "red" if ev.decision == "NO-GO"
                          else "yellow"))
    if ev.laya:
        parts.append(Text(f"Laya (shadow): {ev.laya}", "dim"))
    return Panel(Group(*parts), title=f"Order block {sig.signal_id}", border_style=style)


def signals_table(rows, title: str = "Order-block signals") -> Table:
    t = Table(title=title)
    for col in ("Trigger", "Hz", "Dir", "Score", "Decision", "Structure", "Lots",
                "EV ₹/lot", "Blocked"):
        t.add_column(col, overflow="fold")
    for r in rows:
        t.add_row(r.trigger_ts[5:16], r.horizon[:5], r.direction[:4], f"{r.score:.0f}",
                  Text(r.decision, _DECISION.get(r.decision, "white")), r.structure or "—",
                  str(r.lots), f"{r.ev_rupees:,.0f}" if r.ev_rupees is not None else "—",
                  (r.blocked_reasons or "")[:60])
    return t


def positions_table(rows, title: str = "Paper positions") -> Table:
    t = Table(title=title)
    for col in ("ID", "Opened", "Hz", "Structure", "Lots", "Entry", "Status", "Exit",
                "Reason", "Net ₹", "R"):
        t.add_column(col)
    for p in rows:
        t.add_row(p.position_id, p.opened_at[5:16], p.horizon[:5], p.structure, str(p.lots),
                  f"{p.entry_net:+.2f}", p.status,
                  f"{p.exit_net:+.2f}" if p.exit_net is not None else "—",
                  p.exit_reason or "—",
                  f"{p.net_pnl:+,.0f}" if p.net_pnl is not None else "—",
                  f"{p.r_multiple:+.2f}" if p.r_multiple is not None else "—")
    return t


def backtest_panels(summary: dict) -> list:
    out = []
    for h, block in summary.get("horizons", {}).items():
        t = Table(title=f"{h} — underlying layer (R, after {summary.get('cost_points')} pt costs)")
        for col in ("Arm", "n", "E[R]", "95% CI", "Win%", "PF", "MaxDD R", "Total R"):
            t.add_column(col)
        for arm in ("OB", "B0", "B1", "B2", "B3"):
            m = block.get(arm, {})
            if not m.get("n"):
                t.add_row(arm, "0", "—", "—", "—", "—", "—", "—")
                continue
            ci = m.get("expectancy_ci", [None, None])
            t.add_row(arm, str(m["n"]), f"{m['expectancy_r']:+.3f}",
                      f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", f"{m['win_rate']:.0%}",
                      str(m.get("profit_factor")), str(m.get("max_dd_r")), str(m.get("total_r")))
        out.append(t)
        v = block.get("verdict", {})
        vt = Table(title=f"{h} promotion verdict: "
                         + ("PROMOTED" if v.get("promoted") else "NOT PROMOTED (stays shadow)"))
        vt.add_column("Check")
        vt.add_column("Pass")
        vt.add_column("Detail", overflow="fold")
        for c in v.get("checks", []):
            vt.add_row(c["name"], Text("yes" if c["pass"] else "no",
                                       "green" if c["pass"] else "red"), c["detail"])
        out.append(vt)
    opt = summary.get("options")
    if opt:
        t = Table(title="Option layer (₹ per trade, 1 lot)")
        for col in ("Layer", "Label", "n", "E[₹]", "Total ₹", "Win%", "Charges", "Premium"):
            t.add_column(col)
        for name, m in opt.items():
            if not m.get("n"):
                t.add_row(name, m.get("label", ""), "0", "—", "—", "—", "—", "—")
                continue
            t.add_row(name, m["label"], str(m["n"]), f"{m['expectancy_rupees']:+,.0f}",
                      f"{m['total_rupees']:+,.0f}", f"{m['win_rate']:.0%}",
                      f"{m['avg_charges']:,.0f}", f"{m['avg_premium']:,.0f}")
        out.append(t)
    cal = summary.get("calibration")
    if cal:
        out.append(Text(f"Calibration: {cal.get('status')} · Brier {cal.get('brier')} vs base "
                        f"{cal.get('brier_base')} · skill {cal.get('brier_skill')}"))
    return out
