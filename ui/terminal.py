"""Rich-based terminal trading dashboard."""

from __future__ import annotations

from datetime import datetime

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from config import SETTINGS
from data.nifty import HistoryResult, summarize
from data.options import ChainRow, OptionChain

console = Console()

GREEN = "bright_green"
RED = "bright_red"
DIM = "grey50"


def _fmt(value, fmt="{:,.2f}", na="—") -> str:
    return fmt.format(value) if value is not None else na


def _signed(value, fmt="{:+,.2f}") -> str:
    return fmt.format(value) if value is not None else "—"


def header() -> None:
    now = datetime.now().strftime("%a %d %b %Y · %H:%M:%S")
    console.print(
        Panel(
            Align.center(
                Text.assemble(
                    (f"  {SETTINGS.display_name} TERMINAL  ", "bold white on dark_blue"),
                    ("   ", ""),
                    (now, DIM),
                )
            ),
            box=box.HEAVY,
            border_style="dark_blue",
            padding=(0, 0),
        )
    )


def price_panel(result: HistoryResult) -> Panel:
    q = result.quote
    up = (q.change or 0) >= 0
    color = GREEN if up else RED
    arrow = "▲" if up else "▼"

    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style=DIM)
    grid.add_column(justify="left")

    grid.add_row("LTP", Text(f"{_fmt(q.price)}", style=f"bold {color}"))
    grid.add_row(
        "Change",
        Text.assemble(
            (_signed(q.change), f"bold {color}"),
            (f" {arrow} {_signed(q.change_pct)}%", f"{color}") if q.change_pct is not None else ("", ""),
        ),
    )
    grid.add_row("Prev Close", Text(_fmt(q.previous_close)))
    grid.add_row("Day Range", Text(f"{_fmt(q.day_low)} – {_fmt(q.day_high)}"))
    if result.interval in ("1m", "5m", "15m", "30m", "60m", "1h"):
        grid.add_row("Volume", Text(_fmt(q.volume, "{:,}")))

    subtitle = (
        f"period={result.period} interval={result.interval} · "
        f"fetched {q.fetched_at:%H:%M:%S}"
        + (" [cached]" if result.from_cache else "")
    )
    return Panel(grid, title=f"[bold]{SETTINGS.display_name}[/bold]", subtitle=subtitle, box=box.ROUNDED)


def stats_panel(result: HistoryResult) -> Panel:
    s = summarize(result)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style=DIM)
    grid.add_column(justify="left")
    grid.add_row("Bars loaded", str(s["bars"]))
    grid.add_row("Range", f'{s["first_date"]} → {s["last_date"]}')
    grid.add_row("Period High", _fmt(s["high"]))
    grid.add_row("Period Low", _fmt(s["low"]))
    grid.add_row(
        "Span Δ",
        Text(_signed(s["pct_change_over_span"]) + "%",
             style=GREEN if (s["pct_change_over_span"] or 0) >= 0 else RED),
    )
    return Panel(grid, title="[bold]Statistics[/bold]", box=box.ROUNDED)


def history_table(result: HistoryResult, rows: int | None = None) -> Panel:
    n = rows or SETTINGS.table_rows
    candles = result.candles[-n:]

    table = Table(box=box.SIMPLE_HEAVY, expand=True, header_style="bold cyan")
    table.add_column("Date", justify="left")
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right")
    table.add_column("Low", justify="right")
    table.add_column("Close", justify="right")
    table.add_column("Chg%", justify="right")
    if result.interval != "1d":
        table.add_column("Vol", justify="right")

    for i, c in enumerate(candles):
        prev = candles[i - 1].close if i > 0 else None
        pct = ((c.close - prev) / prev * 100) if prev else None
        color = GREEN if pct is None or pct >= 0 else RED
        cells = [
            c.timestamp.strftime("%Y-%m-%d %H:%M" if result.interval != "1d" else "%Y-%m-%d"),
            _fmt(c.open), _fmt(c.high), _fmt(c.low),
            Text(f"{c.close:,.2f}", style=f"bold {color}"),
            Text(_signed(pct) if pct is not None else "-", style=color),
        ]
        if result.interval != "1d":
            cells.append(_fmt(c.volume, "{:,}", na="-"))
        table.add_row(*cells)

    return Panel(table, title=f"[bold]Recent OHLCV[/bold] (last {len(candles)})", box=box.ROUNDED)


# ---------------------------------------------------------------------------
# Options chain
# ---------------------------------------------------------------------------

def _leg_cell(leg, field: str) -> Text:
    value = getattr(leg, field)
    if value is None:
        return Text("—", style=DIM)
    text = f"{value:,.2f}" if field in ("ltp", "iv", "bid", "ask") else f"{value:,.0f}"
    style = ""
    if field == "change_in_oi":
        style = GREEN if value >= 0 else RED
    return Text(text, style=style or "")


def options_table(chain: OptionChain, expiry: str) -> Panel:
    spot = chain.underlying_value or 0.0
    all_rows = chain.for_expiry(expiry)
    if not all_rows:
        return Panel(Text("No strikes returned for this expiry.", style="yellow"),
                     title="[bold]Option Chain[/bold]", box=box.ROUNDED)

    # Keep a window of strikes around ATM.
    atm_idx = min(range(len(all_rows)), key=lambda i: abs(all_rows[i].strike - spot))
    lo = max(0, atm_idx - SETTINGS.strike_window)
    hi = min(len(all_rows), atm_idx + SETTINGS.strike_window + 1)
    window = all_rows[lo:hi]

    table = Table(box=box.SIMPLE_HEAVY, expand=True, header_style="bold magenta")
    table.add_column("OI", justify="right", style="cyan")
    table.add_column("Chg OI", justify="right")
    table.add_column("Vol", justify="right")
    table.add_column("IV", justify="right")
    table.add_column("Bid", justify="right")
    table.add_column("CE LTP", justify="right", style="green")
    table.add_column("Strike", justify="center", style="bold white")
    table.add_column("PE LTP", justify="right", style="red")
    table.add_column("Bid", justify="right")
    table.add_column("IV", justify="right")
    table.add_column("Vol", justify="right")
    table.add_column("Chg OI", justify="right")
    table.add_column("OI", justify="right", style="cyan")

    for row in window:
        strike_cell = Text(f"{row.strike:,.0f}")
        if abs(row.strike - spot) == min(abs(r.strike - spot) for r in window):
            strike_cell.style = f"bold black on yellow"

        table.add_row(
            _leg_cell(row.call, "open_interest"),
            _leg_cell(row.call, "change_in_oi"),
            _leg_cell(row.call, "volume"),
            _leg_cell(row.call, "iv"),
            _leg_cell(row.call, "bid"),
            _leg_cell(row.call, "ltp"),
            strike_cell,
            _leg_cell(row.put, "ltp"),
            _leg_cell(row.put, "bid"),
            _leg_cell(row.put, "iv"),
            _leg_cell(row.put, "volume"),
            _leg_cell(row.put, "change_in_oi"),
            _leg_cell(row.put, "open_interest"),
        )

    subtitle = (
        f"spot {spot:,.2f} · ATM ≈ {all_rows[atm_idx].strike:,.0f} · "
        f"source={chain.source} · fetched {chain.fetched_at:%H:%M:%S}"
    )
    return Panel(table, title=f"[bold]Option Chain[/bold] — expiry {expiry}",
                 subtitle=subtitle, box=box.ROUNDED)


def expiries_panel(expiries: tuple[str, ...], selected: str) -> Panel:
    line = Text()
    for i, e in enumerate(expiries):
        style = "bold black on yellow" if e == selected else DIM
        marker = "▶ " if e == selected else f"{i + 1}. "
        line.append(marker + e + "  ", style=style)
    return Panel(line, title="[bold]Expiries[/bold]", box=box.ROUNDED)


def error_panel(message: str) -> Panel:
    return Panel(Text(message, style="bold red"), title="[bold red]Error[/bold red]",
                 border_style="red", box=box.HEAVY)


def status_line(message: str, ok: bool = True) -> None:
    icon = "[bright_green]✔[/bright_green]" if ok else "[yellow]…[/yellow]"
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(f" {icon} [{DIM}]{ts}[/{DIM}] {message}")


def render_dashboard(result: HistoryResult, chain: OptionChain | None,
                     selected_expiry: str | None, error: str | None = None) -> None:
    console.clear()
    header()
    console.print()

    top = Table.grid(padding=(0, 1), expand=True)
    top.add_column(ratio=1)
    top.add_column(ratio=1)
    top.add_row(price_panel(result), stats_panel(result))
    console.print(top)
    from ui.chart import candle_panel
    console.print(candle_panel(result))
    console.print(history_table(result))

    if chain:
        expiry = selected_expiry or (chain.expiries[0] if chain.expiries else None)
        if expiry:
            console.print()
            console.print(expiries_panel(chain.expiries, expiry))
            console.print(options_table(chain, expiry))
    elif error:
        console.print()
        console.print(error_panel(error))

    console.print()
    console.print(
        Align.center(Text("[grey50]commands:[/] [bold]r[/bold] refresh · "
                          "[bold]e[/bold] next expiry · [bold]c[/bold] clear cache · "
                          "[bold]q[/bold] quit"), vertical="middle")
    )
