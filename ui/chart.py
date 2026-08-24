"""Terminal candlestick charts — no external charting dependency.

Renders recent OHLCV as Unicode candles (wick │, body ██) inside a Rich
Panel, sized to the terminal so both the classic dashboard and the TUI
can drop it in wherever a price chart belongs.
"""

from __future__ import annotations

from rich import box
from rich.panel import Panel
from rich.text import Text

UP = "bright_green"
DOWN = "bright_red"
WICK_DIM = "grey58"


def render_candles(candles, height: int = 16,
                   max_candles: int = 32) -> Text:
    """Unicode candlestick chart as a styled Rich Text block."""
    candles = list(candles)[-max_candles:]
    if len(candles) < 2:
        return Text("not enough bars to chart", style="yellow")

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    hi, lo = max(highs), min(lows)
    span = (hi - lo) or 1.0
    pad = span * 0.05
    top, bottom = hi + pad, lo - pad
    scale = height / (top - bottom)

    def row_of(price: float) -> int:
        return int(round((top - price) * scale))

    n_cols = len(candles) * 3 + 1      # 2-char body + gap, +1 for close marker
    grid: list[list[str]] = [[" "] * n_cols for _ in range(height)]
    styles: list[list[str | None]] = [[None] * n_cols for _ in range(height)]

    last_close_row = None
    for i, c in enumerate(candles):
        base = i * 3
        color = UP if c.close >= c.open else DOWN
        r_hi, r_lo = row_of(c.high), row_of(c.low)
        r_top, r_bot = sorted((row_of(max(c.open, c.close)),
                               row_of(min(c.open, c.close))))
        # Wick
        for r in range(r_hi, r_lo + 1):
            grid[r][base] = "│"
            styles[r][base] = WICK_DIM
        # Body (min 1 cell tall)
        if r_bot == r_top:
            for cc in (base, base + 1):
                grid[r_top][cc] = "▄"
                styles[r_top][cc] = color
        else:
            for r in range(r_top, r_bot + 1):
                for cc in (base, base + 1):
                    grid[r][cc] = "█"
                    styles[r][cc] = color
        last_close_row = row_of(c.close)

    if last_close_row is not None:
        grid[last_close_row][-1] = "◀"
        styles[last_close_row][-1] = "bold cyan"

    out = Text()
    label_w = len(f"{hi:,.2f}")
    axis_rows = {0: f"{hi:,.2f}", height // 2: f"{(hi + lo) / 2:,.2f}",
                 height - 1: f"{lo:,.2f}"}
    for r in range(height):
        lab = axis_rows.get(r, "")
        out.append(f"{lab:>{label_w}} ", style="grey50")
        for cc in range(n_cols):
            out.append(grid[r][cc], style=styles[r][cc])
        out.append("\n")

    intraday = candles[0].timestamp.hour or candles[0].timestamp.minute
    fmt = "%d %b %H:%M" if intraday else "%d %b"
    first_ts = candles[0].timestamp.strftime(fmt)
    last_ts = candles[-1].timestamp.strftime(fmt)
    footer = f"{'':>{label_w + 1}}{first_ts}"
    footer = footer.ljust(label_w + 1 + n_cols - len(last_ts)) + last_ts
    out.append(footer, style="grey50")
    return out


def candle_panel(result, rows: int = 32, height: int = 16) -> Panel:
    """Chart panel for a HistoryResult — drop-in next to the OHLCV table."""
    import shutil
    window = result.candles[-rows:]
    label_overhead = len(f"{max(c.high for c in window):,.2f}") + 1
    term_w = shutil.get_terminal_size((120, 24)).columns - 8   # borders + padding
    fit = max(10, (term_w - label_overhead) // 3)
    candles = window[-fit:]
    chart = render_candles(candles, height=height)
    intraday = result.interval != "1d"
    subtitle = (f"interval={result.interval} · last {len(candles)} bars · "
                f"{'intraday' if intraday else 'daily'}")
    return Panel(chart, title="[bold]Price Chart[/bold]",
                 subtitle=subtitle, box=box.ROUNDED)
