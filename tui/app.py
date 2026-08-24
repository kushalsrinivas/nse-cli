"""Textual trading terminal for nifty-strats."""

from __future__ import annotations

import shlex
import threading

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static, TabbedContent, TabPane

from analysis import indicators as ta
from analysis.signals import scan_events, snapshot
from config import SETTINGS
from data import nifty
from data import options as opts
from journal.db import shared_journal
from journal.performance import all_breakdowns, summarize
from strategies.base import DEFAULT_STRATEGY
from tui import views


class TerminalState:
    """Shared snapshot of market + analysis data."""

    def __init__(self) -> None:
        self.history: nifty.HistoryResult | None = None
        self.chain: opts.OptionChain | None = None
        self.expiry: str | None = None
        self.indicators: ta.IndicatorSet | None = None
        self.states: list = []
        self.events: list = []
        self.error: str | None = None
        self.lock = threading.Lock()


class NiftyTerminal(App):
    TITLE = "NIFTY 50 • TRADING TERMINAL"
    CSS = """
    Screen { background: $surface; }
    #status { dock: top; height: 1; padding: 0 1; color: $text-muted; }
    TabbedContent { height: 1fr; }
    Static.panel-holder { padding: 0 1; }
    #journal-input { dock: bottom; }
    """
    BINDINGS = [
        Binding("1", "tab('market')", "Market", show=False),
        Binding("2", "tab('technicals')", "Technicals", show=False),
        Binding("3", "tab('options')", "Options", show=False),
        Binding("4", "tab('signals')", "Signals", show=False),
        Binding("5", "tab('journal')", "Journal", show=False),
        Binding("6", "tab('performance')", "Performance", show=False),
        Binding("r", "refresh_data", "Refresh"),
        Binding("e", "next_expiry", "Expiry"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = TerminalState()
        self.journal_filter = "all"
        self.journal_search: str | None = None

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="status")
        with TabbedContent(initial="market"):
            for tab_id, title in (
                ("market", "1 Market"), ("technicals", "2 Technicals"),
                ("options", "3 Options"), ("signals", "4 Signals"),
                ("journal", "5 Journal"), ("performance", "6 Performance"),
            ):
                with TabPane(title=title, id=tab_id):
                    if tab_id == "journal":
                        yield Input(placeholder=views.JOURNAL_HELP.replace("[bold]", "").replace("[/bold]", ""), id="journal-input")
                    yield VerticalScroll(Static("", classes="panel-holder"), id=f"scroll-{tab_id}")
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh_data()
        self.set_interval(SETTINGS.refresh_seconds, self.action_refresh_data)

    # -- data loading (background worker) --------------------------------------

    @work(exclusive=True, thread=True)
    def action_refresh_data(self) -> None:
        try:
            history = nifty.fetch_history()
            ind = ta.compute(history.candles)
            states = snapshot(ind)
            events = scan_events(ind)
            try:
                chain = opts.fetch_chain(expiry=self.state.expiry)
                self.state.expiry = chain.expiries[0] if not self.state.expiry else self.state.expiry
            except Exception as exc:
                chain = None
                self.state.error = f"Option chain unavailable: {exc}"
            with self.state.lock:
                self.state.history = history
                self.state.indicators = ind
                self.state.states = states
                self.state.events = events
                self.state.chain = chain
            self.call_from_thread(self._render_all)
        except Exception as exc:
            self.call_from_thread(self._show_error, str(exc))

    # -- rendering ---------------------------------------------------------------

    def _holder(self, tab_id: str) -> Static:
        return self.query_one(f"#scroll-{tab_id} Static.panel-holder", Static)

    @staticmethod
    def _set(holder: Static, *renderables) -> None:
        holder.update(Group(*renderables))

    def _render_all(self) -> None:
        st = self.state
        hist = st.history
        status = Text()
        if hist:
            q = hist.quote
            col = "bright_green" if (q.change or 0) >= 0 else "bright_red"
            status.append(
                f"{SETTINGS.display_name} {q.price:,.2f} "
                f"{q.change:+.2f} ({q.change_pct:+.2f}%)  ·  fetched {hist.quote.fetched_at:%H:%M:%S}"
                + (" · cached" if hist.from_cache else ""),
                style=f"bold {col}",
            )
        if st.states:
            trend = next((s for s in st.states if s.name == "Trend"), None)
            if trend:
                style = views.DIR_STYLE[trend.direction.value]
                status.append(f"  ·  Trend ", style="grey50").append(trend.status, style=f"bold {style}")
        self.query_one("#status", Static).update(status)
        self._render_market()
        self._render_technicals()
        self._render_options()
        self._render_signals()
        self._render_journal()
        self._render_performance()

    def _render_market(self) -> None:
        hist = self.state.history
        holder = self._holder("market")
        if not hist:
            holder.update(Text("No data yet…", style="yellow"))
            return
        parts = views.market_view(hist)
        parts.append(Panel(views.ohlcv_table(hist),
                           title=f"[bold]Recent OHLCV[/bold] (last {SETTINGS.table_rows})",
                           box=box.ROUNDED))
        self._set(holder, *parts)

    def _render_technicals(self) -> None:
        holder = self._holder("technicals")
        if not self.state.states:
            holder.update(Text("No data yet…", style="yellow"))
            return
        panels = [views.technicals_view(self.state.states)]
        if self.state.history:
            panels.append(Panel(views.ohlcv_table(self.state.history, rows=8),
                                title="[bold]Recent Candles[/bold]", box=box.ROUNDED))
        self._set(holder, *panels)

    def _render_options(self) -> None:
        holder = self._holder("options")
        chain = self.state.chain
        if not chain or not chain.expiries:
            msg = self.state.error or "No option-chain data."
            self._set(holder, views.expiries_line((), ""),
                      Panel(Text(msg, style="yellow"), box=box.ROUNDED))
            return
        expiry = self.state.expiry or chain.expiries[0]
        self._set(holder,
                  views.expiries_line(chain.expiries, expiry),
                  views.options_table(chain, expiry))

    def _render_signals(self) -> None:
        holder = self._holder("signals")
        if not self.state.states:
            holder.update(Text("No data yet…", style="yellow"))
            return
        self._set(holder, *views.signals_view(self.state.states, self.state.events))

    def _render_journal(self) -> None:
        holder = self._holder("journal")
        j = shared_journal()
        trades = j.list(status=self.journal_filter, search=self.journal_search)
        open_count = len(j.list(status="open"))
        title = (f"[bold]Trade Journal[/bold] — filter={self.journal_filter}"
                 f"{' search=' + repr(self.journal_search) if self.journal_search else ''}"
                 f" · {open_count} open")
        holder.update(views.trades_table(trades, title))

    def _render_performance(self) -> None:
        holder = self._holder("performance")
        j = shared_journal()
        closed = j.all_closed()
        stats = summarize(closed)
        breakdowns = all_breakdowns(closed)
        if not stats.has_data:
            self._set(holder, views.performance_summary(stats),
                      Panel(Text("No closed trades yet — take a paper trade from the "
                                 "Journal tab.", style="yellow"), box=box.ROUNDED))
        else:
            self._set(holder, views.performance_summary(stats), views.breakdown_tables(breakdowns))

    def _show_error(self, message: str) -> None:
        self.query_one("#status", Static).update(Text(message, style="bold red"))

    # -- actions -----------------------------------------------------------------

    def action_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    def action_next_expiry(self) -> None:
        chain = self.state.chain
        if not chain or len(chain.expiries) < 2:
            return
        current = self.state.expiry or chain.expiries[0]
        idx = chain.expiries.index(current)
        self.state.expiry = chain.expiries[(idx + 1) % len(chain.expiries)]
        self._render_options()

    # -- journal input handling -----------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "journal-input":
            return
        raw = event.value.strip()
        event.input.clear()
        if not raw:
            return
        feedback = self._run_journal_command(raw)
        if feedback:
            self.notify(feedback, severity="information", timeout=6)
        self._render_journal()
        self._render_performance()

    def _current_states(self) -> dict[str, str]:
        out = {}
        for s in self.state.states:
            key = {"MACD": "macd", "Volume": "volume"}.get(s.name)
            if s.name.startswith("EMA"):
                key = "ema"
            elif s.name.startswith("SMA"):
                key = "sma"
            if key:
                out[key] = s.status.strip("▲▼● ")
        return out

    def _run_journal_command(self, raw: str) -> str | None:
        j = shared_journal()
        parts = shlex.split(raw)
        cmd, args = parts[0].lower(), parts[1:]

        try:
            match cmd:
                case "add" | "take":
                    direction = args[0]
                    qty = float(args[1]) if len(args) > 1 else 75.0
                    strategy = args[2] if len(args) > 2 else DEFAULT_STRATEGY.name
                    price = self.state.history.quote.price
                    decision = DEFAULT_STRATEGY.decide(self.state.indicators, self.state.events)
                    trade = j.open_trade(
                        instrument=SETTINGS.option_symbol,
                        direction=direction,
                        entry_price=price,
                        quantity=qty,
                        stop_loss=decision.stop_loss,
                        target=decision.target,
                        strategy=strategy,
                        entry_reason=decision.reason,
                        states=self._current_states(),
                    )
                    return f"opened #{trade.id} {trade.direction.upper()} @{price:,.2f}"
                case "close":
                    trade_id = int(args[0])
                    price = float(args[1]) if len(args) > 1 else self.state.history.quote.price
                    closed = j.close_trade(trade_id, price,
                                           exit_reason="manual close")
                    return (f"closed #{trade_id} P&L {closed.pnl:+,.2f}"
                            if closed else f"cannot close #{trade_id}")
                case "note":
                    trade_id, text = int(args[0]), " ".join(args[1:])
                    return "saved" if j.set_notes(trade_id, text) else f"no trade #{trade_id}"
                case "del" | "delete":
                    return f"deleted #{args[0]}" if j.delete(int(args[0])) else "not found"
                case "filter":
                    if args[0] in ("open", "closed", "all"):
                        self.journal_filter = args[0]
                        return None
                    return "filter must be open|closed|all"
                case "search":
                    self.journal_search = " ".join(args) or None
                    return None
                case "show":
                    t = j.get(int(args[0]))
                    return str(t.__dict__) if t else "not found"
                case "help":
                    return views.JOURNAL_HELP
                case _:
                    return f"unknown command {cmd!r} — try help"
        except (IndexError, ValueError) as exc:
            return f"bad command: {exc}"


def run() -> None:
    NiftyTerminal().run()
