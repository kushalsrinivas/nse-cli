"""Rich renderers for the breadth snapshot and scenario set.

Builders (`*_panel`) return Rich renderables shared by the CLI and the TUI;
`render_*` print them (CLI behaviour unchanged).
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from model.breadth.aggregate import BreadthSnapshot
from model.breadth.divergence import DivergenceSignal
from model.breadth.features import ConstituentFeatures
from model.breadth.scenarios import ScenarioSet, structure_view
from model.breadth.universe import get_universe


def breadth_panel(snap: BreadthSnapshot) -> Panel:
    t = Table(title=f"NIFTY 50 Breadth — {snap.date or 'latest'} "
                    f"({snap.n_covered} names, {snap.weight_coverage * 100:.0f}% weight)")
    t.add_column("Metric")
    t.add_column("Equal-wtd", justify="right")
    t.add_column("Cap-wtd", justify="right")
    adv = f"{snap.adv_pct:.0f}%" if snap.adv_pct is not None else "—"
    dec = f"{snap.dec_pct:.0f}%" if snap.dec_pct is not None else "—"
    t.add_row("Advancing / declining", f"{adv} / {dec}",
              f"{snap.adv_weight_pct:.0f}% adv" if snap.adv_weight_pct is not None else "—")
    t.add_row("> SMA20 / SMA50",
              f"{_f(snap.above_sma20_pct)} / {_f(snap.above_sma50_pct)}", "—")
    t.add_row("New highs / lows (20d)",
              f"{_f(snap.new_high_pct)} / {_f(snap.new_low_pct)}", "—")
    t.add_row("Confirming NIFTY", f"{_f(snap.confirming_pct)}",
              f"{_f(snap.weighted_confirm_pct)}")
    t.add_row("Adv volume share",
              f"{_f(snap.adv_volume_share)}",
              f"UDVR {snap.up_down_volume_ratio}" if snap.up_down_volume_ratio else "—")
    t.add_row("Concentration",
              f"top5 {snap.top5_contrib_share}%" if snap.top5_contrib_share is not None else "—",
              f"eff N {snap.effective_n}" if snap.effective_n else "—")
    t.add_row("Heavyweights (top-8)",
              f"{_f(snap.heavy_adv_pct)} adv",
              f"{snap.heavy_avg_ret:+.2f}%" if snap.heavy_avg_ret is not None else "—")
    score_line = (f"breadth {snap.breadth_score:+.0f}  "
                  f"participation={snap.participation}  "
                  f"diverging={snap.diverging_n}")
    if snap.breadth_accel is not None:
        score_line += f"  accel={snap.breadth_accel:+.0f}pp"
    return Panel(t, title="[bold]Market Breadth[/bold]",
                 subtitle=score_line, box=box.ROUNDED)


def sectors_panel(snap: BreadthSnapshot) -> Panel | Text:
    if not snap.sectors:
        return Text("No sector data.", style="yellow")
    s = Table(box=box.SIMPLE, expand=True)
    s.add_column("Sector")
    s.add_column("Adv%", justify="right")
    s.add_column("Avg 1d%", justify="right")
    s.add_column("Idx wt%", justify="right")
    for row in snap.sectors:
        style = "green" if (row.avg_ret or 0) > 0 else "red" if (row.avg_ret or 0) < 0 else ""
        s.add_row(row.sector,
                  f"{row.adv_pct:.0f}%" if row.adv_pct is not None else "—",
                  f"{row.avg_ret:+.2f}%" if row.avg_ret is not None else "—",
                  f"{row.weight_share * 100:.1f}%",
                  style=style)
    return Panel(s, title="[bold]Sector Breadth[/bold]", box=box.ROUNDED)


def _pos_mark(level: float | None, ref: float | None) -> tuple[str, str]:
    """▲/▼/● vs a reference level with a matching style."""
    if level is None or ref is None:
        return "—", ""
    if level > ref:
        return "▲", "green"
    if level < ref:
        return "▼", "red"
    return "●", "grey50"


def stocks_panel(feats: dict[str, ConstituentFeatures],
                 weights: dict[str, float] | None = None) -> Panel:
    """All covered constituents: return + MACD/EMA/SMA/volume/trend.

    One row per stock, sorted by 1d% descending — no truncation, so narrow
    leadership vs broad participation is visible stock by stock.
    """
    rows = [(s, f) for s, f in feats.items() if f.ret_1d is not None]
    if not rows:
        return Panel(Text("No constituent data.", style="yellow"),
                     title="[bold]All Constituents[/bold]", box=box.ROUNDED)
    rows.sort(key=lambda kv: kv[1].ret_1d or 0.0, reverse=True)
    t = Table(box=box.SIMPLE, expand=True)
    for name, kw in (("Stock", {}), ("1d%", {"justify": "right"}),
                     ("MACD", {"justify": "center"}),
                     ("EMA9", {"justify": "center"}),
                     ("EMA21", {"justify": "center"}),
                     ("SMA20", {"justify": "center"}),
                     ("SMA50", {"justify": "center"}),
                     ("Vol", {"justify": "right"}),
                     ("Trend", {"justify": "center"})):
        t.add_column(name, **kw)
    short = {c.symbol: c.short for c in get_universe()}
    for sym, f in rows:
        ret_style = "green" if (f.ret_1d or 0) > 0.05 else \
            "red" if (f.ret_1d or 0) < -0.05 else ""
        cells: list = [
            Text(short.get(sym, sym)),
            Text(f"{f.ret_1d:+.2f}%", style=ret_style),
        ]
        macd_m, macd_s = _pos_mark(f.macd_hist, 0.0) \
            if f.macd_hist is not None else ("—", "")
        cells.append(Text(macd_m, style=macd_s))
        for lvl, ref in ((f.close, f.ema9), (f.close, f.ema21),
                         (f.close, f.sma20), (f.close, f.sma50)):
            m, s = _pos_mark(lvl, ref)
            cells.append(Text(m, style=s))
        vol = f"{f.volume_ratio:.1f}×" if f.volume_ratio is not None else "—"
        if f.volume_anomaly:
            vol += " ●"
        cells.append(Text(vol, style="yellow" if f.volume_anomaly else ""))
        trend_style = {"UP": "bold green", "DN": "bold red",
                       "FLAT": "yellow"}.get(f.trend or "", "")
        cells.append(Text(f.trend or "—", style=trend_style))
        t.add_row(*cells)
    return Panel(t, title=f"[bold]All Constituents[/bold] — {len(rows)} stocks "
                          f"by 1d% (● = volume anomaly)",
                 box=box.ROUNDED)


def divergence_panel(flags: list[DivergenceSignal]) -> Panel:
    if not flags:
        return Panel(Text("no NIFTY-vs-breadth divergence detected", style="green"),
                     title="[bold]Divergence[/bold]", box=box.ROUNDED)
    t = Table(box=box.SIMPLE, expand=True)
    t.add_column("Flag")
    t.add_column("Sev", justify="right")
    t.add_column("Lean")
    t.add_column("Rationale")
    for fl in flags:
        style = "red" if fl.severity >= 3 else "yellow" if fl.severity == 2 else "dim"
        t.add_row(fl.flag, str(fl.severity), fl.direction, fl.rationale, style=style)
    return Panel(t, title="[bold]Constituent Divergence[/bold]", box=box.ROUNDED)


def tape_summary(snap: BreadthSnapshot, flags: list[DivergenceSignal]) -> str:
    """One-line bottom-up read of the tape (no decision-model input needed)."""
    bits = [f"breadth {snap.breadth_score:+.0f} ({snap.participation.lower()})"]
    if snap.confirming_pct is not None:
        bits.append(f"{snap.confirming_pct:.0f}% confirm")
    if snap.sectors:
        bits.append(f"leaders: {snap.sectors[0].sector} "
                    f"({snap.sectors[0].avg_ret:+.2f}%); "
                    f"laggards: {snap.sectors[-1].sector} "
                    f"({snap.sectors[-1].avg_ret:+.2f}%)")
    sev = max((f.severity for f in flags), default=0)
    if sev >= 2:
        worst = max(flags, key=lambda f: f.severity)
        bits.append(f"⚠ {worst.flag} ({worst.severity}/3)")
    return " · ".join(bits)


# -- CLI printers (behaviour unchanged) --------------------------------------

def render_breadth(snap: BreadthSnapshot, console: Console | None = None) -> None:
    console = console or Console()
    console.print(breadth_panel(snap))
    if snap.sectors:
        console.print(sectors_panel(snap))
    for note in snap.notes:
        console.print(f"[dim]{note}[/]")


def render_divergence(flags: list[DivergenceSignal],
                      console: Console | None = None) -> None:
    console = console or Console()
    console.print(divergence_panel(flags))


def render_scenarios(scen: ScenarioSet, console: Console | None = None) -> None:
    console = console or Console()
    t = Table(title=f"Overnight scenarios — posture {scen.posture}")
    t.add_column("Scenario")
    t.add_column("P", justify="right")
    for k, v in sorted(scen.probs.items(), key=lambda kv: -kv[1]):
        t.add_row(k, f"{v:.0%}")
    console.print(t)
    st = structure_view(scen)
    u = Table(title="Structure screen (needs EV confirmation)")
    u.add_column("Structure")
    u.add_column("View")
    for k, v in st.items():
        u.add_row(k, v)
    console.print(u)
    for e in scen.evidence:
        console.print(f"[dim]{e}[/]")


def _f(v: float | None, suffix: str = "%") -> str:
    return f"{v:.0f}{suffix}" if v is not None else "—"
