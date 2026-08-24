"""Rich renderables for each TUI tab."""

from __future__ import annotations

import pandas as pd
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from analysis.signals import Direction, SignalState, fmt_volume
from config import SETTINGS
from data.nifty import HistoryResult, summarize
from data.options import OptionChain
from journal.db import Trade
from journal.performance import Stats


DIR_STYLE = {"bullish": "bright_green", "bearish": "bright_red", "neutral": "yellow"}


def _fmt(v, fmt="{:,.2f}", na="—") -> str:
    return fmt.format(v) if v is not None else na


def _signed(v) -> str:
    return f"{v:+,.2f}" if v is not None else "—"


# ---------------------------------------------------------------------------
# Market tab
# ---------------------------------------------------------------------------

def market_view(result: HistoryResult) -> list[Panel]:
    q = result.quote
    up = (q.change or 0) >= 0
    color = GREEN = "bright_green" if up else "bright_red"
    arrow = "▲" if up else "▼"

    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="grey50")
    grid.add_column(justify="left")
    grid.add_row("LTP", Text(f"{_fmt(q.price)}", style=f"bold {color}"))
    grid.add_row("Change", Text.assemble(
        (_signed(q.change), f"bold {color}"),
        (f" {arrow} {_signed(q.change_pct)}%", color) if q.change_pct is not None else ("", ""),
    ))
    grid.add_row("Prev Close", _fmt(q.previous_close))
    grid.add_row("Day Range", f"{_fmt(q.day_low)} – {_fmt(q.day_high)}")
    grid.add_row("Open", _fmt(q.day_open))

    s = summarize(result)
    sgrid = Table.grid(padding=(0, 2))
    sgrid.add_column(justify="right", style="grey50")
    sgrid.add_column(justify="left")
    sgrid.add_row("Bars loaded", str(s["bars"]))
    sgrid.add_row("Range", f'{s["first_date"]} → {s["last_date"]}')
    sgrid.add_row("Period High", _fmt(s["high"]))
    sgrid.add_row("Period Low", _fmt(s["low"]))
    sgrid.add_row("Avg Volume", fmt_volume(s["avg_volume"]))

    subtitle = (
        f"period={result.period} interval={result.interval} · "
        f"fetched {result.quote.fetched_at:%d %b %H:%M:%S}"
        + (" · cached" if result.from_cache else "")
    )
    return [
        Panel(grid, title=f"[bold]{SETTINGS.display_name}[/bold]", subtitle=subtitle,
              box=box.ROUNDED),
        Panel(sgrid, title="[bold]Statistics[/bold]", box=box.ROUNDED),
    ]


def ohlcv_table(result: HistoryResult, rows: int | None = None) -> Table:
    n = rows or SETTINGS.table_rows
    candles = result.candles[-n:]
    table = Table(box=box.SIMPLE_HEAVY, expand=True, header_style="bold cyan")
    table.add_column("Date", justify="left")
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right")
    table.add_column("Low", justify="right")
    table.add_column("Close", justify="right")
    table.add_column("Chg%", justify="right")
    table.add_column("Volume", justify="right")

    intraday = result.interval != "1d"
    for i, c in enumerate(candles):
        prev = candles[i - 1].close if i > 0 else None
        pct = ((c.close - prev) / prev * 100) if prev else None
        col = "bright_green" if pct is None or pct >= 0 else "bright_red"
        table.add_row(
            c.timestamp.strftime("%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d"),
            _fmt(c.open), _fmt(c.high), _fmt(c.low),
            Text(f"{c.close:,.2f}", style=f"bold {col}"),
            Text(_signed(pct) if pct is not None else "-", style=col),
            fmt_volume(c.volume),
        )
    return table


# ---------------------------------------------------------------------------
# Technicals tab
# ---------------------------------------------------------------------------

def technicals_view(states: list[SignalState]) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold", min_width=14)
    table.add_column(min_width=20)
    table.add_column()

    detail_order = {
        "MACD": ["macd", "signal", "histogram"],
    }
    for st in states:
        style = DIR_STYLE[st.direction.value]
        if st.name == "Trend":
            table.add_row("", "")
            table.add_row("TREND", Text(st.status, style=f"bold {style}"))
            continue
        details = st.detail
        ordered = detail_order.get(st.name, list(details.keys()))
        first = True
        for key in ordered:
            val = details.get(key, "—")
            colored = Text(val, style="bright_green" if str(val).startswith("+") else
                           ("bright_red" if str(val).startswith("-") else ""))
            table.add_row(
                st.name if first else "",
                Text(st.status, style=f"{style}") if first else "",
                f"{key.replace('_', ' ').title():<12} {colored}",
            )
            first = False
    return Panel(table, title="[bold]Technical Indicators[/bold]",
                 subtitle="latest bar", box=box.ROUNDED)


# ---------------------------------------------------------------------------
# Signals tab
# ---------------------------------------------------------------------------

def signals_view(states: list[SignalState], events, limit: int = 25) -> list[Panel]:
    lines = Text()
    for i, st in enumerate(states):
        if st.name == "Trend":
            continue
        style = DIR_STYLE[st.direction.value]
        lines.append(f"{st.name:<12}", style="bold white")
        lines.append(f"  {st.status}\n", style=style)

    panels = [Panel(lines, title="[bold]Live Signals[/bold]", box=box.ROUNDED)]

    table = Table(box=box.SIMPLE, expand=True, header_style="bold magenta")
    table.add_column("Time", style="grey50")
    table.add_column("Signal", style="bold")
    table.add_column("Dir", justify="center")
    table.add_column("Detail")
    for ev in events[:limit]:
        d = {"bullish": "▲", "bearish": "▼", "neutral": "●"}[ev.direction.value]
        table.add_row(
            pd.Timestamp(ev.timestamp).strftime("%Y-%m-%d %H:%M"),
            ev.kind,
            Text(d, style=DIR_STYLE[ev.direction.value]),
            ev.description,
        )
    panels.append(Panel(table, title="[bold]Signal History[/bold] (newest first)",
                        box=box.ROUNDED))
    return panels


# ---------------------------------------------------------------------------
# Options tab
# ---------------------------------------------------------------------------

def expiries_line(expiries: tuple[str, ...], selected: str) -> Panel:
    line = Text()
    shown = expiries[:8]
    for i, e in enumerate(shown):
        style = "bold black on yellow" if e == selected else "grey50"
        marker = "▶ " if e == selected else f"{i + 1}. "
        line.append(marker + e + "  ", style=style)
    if len(expiries) > len(shown):
        line.append(f"  (+{len(expiries) - len(shown)} more)", style="grey50")
    return Panel(line, title="[bold]Expiries[/bold]  ([italic]e[/italic] = cycle)",
                 box=box.ROUNDED)


def options_table(chain: OptionChain, expiry: str) -> Panel:
    spot = chain.underlying_value or 0.0
    all_rows = chain.for_expiry(expiry)
    if not all_rows:
        return Panel(Text("No strikes returned for this expiry.", style="yellow"),
                     title="[bold]Option Chain[/bold]", box=box.ROUNDED)

    atm_idx = min(range(len(all_rows)), key=lambda i: abs(all_rows[i].strike - spot))
    lo = max(0, atm_idx - SETTINGS.strike_window)
    hi = min(len(all_rows), atm_idx + SETTINGS.strike_window + 1)
    window = all_rows[lo:hi]

    def cell(leg, field: str) -> Text:
        value = getattr(leg, field)
        if value is None:
            return Text("—", style="grey50")
        text = f"{value:,.2f}" if field in ("ltp", "iv", "bid", "ask") else f"{value:,.0f}"
        style = ""
        if field == "change_in_oi":
            style = "bright_green" if value >= 0 else "bright_red"
        return Text(text, style=style or "")

    table = Table(box=box.SIMPLE_HEAVY, expand=True, header_style="bold magenta")
    for name, just in (("OI", "right"), ("Chg OI", "right"), ("Vol", "right"),
                       ("IV", "right"), ("Bid", "right")):
        table.add_column(name, justify=just)
    table.add_column("CE LTP", justify="right", style="green")
    table.add_column("Strike", justify="center", style="bold white")
    table.add_column("PE LTP", justify="right", style="red")
    for name in ("Bid", "IV", "Vol", "Chg OI", "OI"):
        table.add_column(name, justify="right")

    min_dist = min(abs(r.strike - spot) for r in window)
    for row in window:
        strike_cell = Text(f"{row.strike:,.0f}")
        if abs(row.strike - spot) == min_dist:
            strike_cell.style = "bold black on yellow"
        table.add_row(
            cell(row.call, "open_interest"), cell(row.call, "change_in_oi"),
            cell(row.call, "volume"), cell(row.call, "iv"), cell(row.call, "bid"),
            cell(row.call, "ltp"), strike_cell, cell(row.put, "ltp"),
            cell(row.put, "bid"), cell(row.put, "iv"), cell(row.put, "volume"),
            cell(row.put, "change_in_oi"), cell(row.put, "open_interest"),
        )

    subtitle = (f"spot {spot:,.2f} · source={chain.source} · "
                f"fetched {chain.fetched_at:%H:%M:%S}")
    return Panel(table, title=f"[bold]Option Chain[/bold] — {expiry}",
                 subtitle=subtitle, box=box.ROUNDED)


# ---------------------------------------------------------------------------
# Journal tab
# ---------------------------------------------------------------------------

def trades_table(trades: list[Trade], title: str) -> Panel:
    table = Table(box=box.SIMPLE, expand=True, header_style="bold cyan")
    for name, kw in (("ID", {}), ("Time", {"style": "grey50"}),
                     ("Dir", {"justify": "center"}), ("Entry", {"justify": "right"}),
                     ("Exit", {"justify": "right"}), ("Qty", {"justify": "right"}),
                     ("P&L", {"justify": "right"}), ("P&L %", {"justify": "right"}),
                     ("Strategy", {}), ("Status", {"justify": "center"}),
                     ("Reason/Notes", {})):
        table.add_column(name, **kw)

    for t in trades:
        pnl_col = "bright_green" if (t.pnl or 0) >= 0 else "bright_red"
        dir_mark = "L" if t.direction == "long" else "S"
        reason = t.exit_reason if t.status == "closed" and t.exit_reason else (
            t.entry_reason or t.notes or "")[:40]
        table.add_row(
            str(t.id),
            t.timestamp[:16].replace("T", " "),
            Text(dir_mark, style="green" if dir_mark == "L" else "red"),
            f"{t.entry_price:,.2f}",
            f"{t.exit_price:,.2f}" if t.exit_price else "—",
            f"{t.quantity:g}",
            Text(_signed(t.pnl), style=pnl_col) if t.pnl is not None else "—",
            f"{t.pnl_pct:+.2f}%" if t.pnl_pct is not None else "—",
            t.strategy,
            Text("OPEN", style="yellow") if t.status == "open" else "closed",
            reason,
        )
    return Panel(table, title=title, box=box.ROUNDED)


JOURNAL_HELP = (
    "[bold]add[/bold] long|short [qty] [strategy] · "
    "[bold]close[/bold] id [price] · [bold]note[/bold] id text · "
    "[bold]del[/bold] id · [bold]filter[/bold] open|closed|all · "
    "[bold]search[/bold] text · [bold]show[/bold] id"
)


# ---------------------------------------------------------------------------
# Performance tab
# ---------------------------------------------------------------------------

def performance_summary(stats: Stats) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="grey50", justify="right")
    grid.add_column()
    grid.add_column(style="grey50", justify="right")
    grid.add_column()

    pnl_style = "bright_green" if stats.net_pnl >= 0 else "bright_red"
    rows = [
        ("Total trades", str(stats.total), "Win rate", f"{stats.win_rate}%" if stats.win_rate is not None else "—"),
        ("Wins", str(stats.wins), "Net P&L", Text(f"{stats.net_pnl:+,.2f}", style=f"bold {pnl_style}")),
        ("Losses", str(stats.losses), "Profit factor", str(stats.profit_factor) if stats.profit_factor else "∞"),
        ("Gross profit", f"+{stats.gross_profit:,.2f}", "Risk/reward", str(stats.risk_reward or "—")),
        ("Gross loss", f"-{stats.gross_loss:,.2f}", "Max drawdown",
         Text(f"{stats.max_drawdown:,.2f}", style="bright_red" if stats.max_drawdown < 0 else "")),
        ("Avg win", f"+{stats.avg_win:,.2f}" if stats.avg_win else "—", "Avg duration",
         _duration(stats.avg_duration_min)),
        ("Avg loss", f"-{stats.avg_loss:,.2f}" if stats.avg_loss else "—", "Streaks",
         f"{stats.max_consec_wins}W / {stats.max_consec_losses}L"),
        ("Best trade", f"+{stats.largest_winner:,.2f}" if stats.total else "—", "Worst trade",
         f"{stats.largest_loser:,.2f}" if stats.total else "—"),
    ]
    for l1, v1, l2, v2 in rows:
        grid.add_row(l1, v1, l2, v2)
    return Panel(grid, title="[bold]Performance Summary[/bold] (closed trades)",
                 box=box.ROUNDED)


def breakdown_tables(breakdowns: dict) -> Panel:
    table = Table(box=box.SIMPLE, expand=True, header_style="bold cyan")
    table.add_column("Bucket", style="bold")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("Net P&L", justify="right")
    table.add_column("PF", justify="right")

    for kind in ("strategy", "direction", "setup", "month"):
        buckets = breakdowns.get(kind, {})
        for name, s in list(buckets.items())[:10]:
            pnl_col = "bright_green" if s.net_pnl >= 0 else "bright_red"
            table.add_row(
                f"[dim]{kind}[/dim] {name}",
                str(s.total),
                f"{s.win_rate}%" if s.win_rate is not None else "—",
                Text(f"{s.net_pnl:+,.2f}", style=pnl_col),
                str(s.profit_factor) if s.profit_factor else "∞",
            )
    return Panel(table, title="[bold]Breakdowns[/bold]", box=box.ROUNDED)


def _duration(minutes: float | None) -> str:
    if minutes is None:
        return "—"
    if minutes < 60 * 24:
        return f"{minutes / 60:.1f}h" if minutes >= 60 else f"{minutes:.0f}m"
    return f"{minutes / (60 * 24):.1f}d"
