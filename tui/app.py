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
from journal.overnight_db import shared_overnight_journal
from journal.overnight_perf import compute_overnight_performance
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
        Binding("7", "tab('overnight')", "Overnight", show=False),
        Binding("r", "force_refresh", "Refresh"),
        Binding("e", "next_expiry", "Expiry"),
        Binding("c", "copy_context", "Copy"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = TerminalState()
        self.journal_filter = "all"
        self.journal_search: str | None = None
        self.oj_filter = "all"
        self.oj_search: str | None = None

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="status")
        with TabbedContent(initial="market"):
            for tab_id, title in (
                ("market", "1 Market"), ("technicals", "2 Technicals"),
                ("options", "3 Options"), ("signals", "4 Signals"),
                ("journal", "5 Journal"), ("performance", "6 Performance"),
                ("overnight", "7 Overnight"),
            ):
                with TabPane(title=title, id=tab_id):
                    if tab_id == "journal":
                        yield Input(placeholder=views.JOURNAL_HELP.replace("[bold]", "").replace("[/bold]", ""), id="journal-input")
                    elif tab_id == "overnight":
                        yield Input(placeholder=views.OVERNIGHT_JOURNAL_HELP.replace("[bold]", "").replace("[/bold]", ""), id="overnight-input")
                    yield VerticalScroll(Static("", classes="panel-holder"), id=f"scroll-{tab_id}")
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh_data()
        self.set_interval(SETTINGS.refresh_seconds, self.action_refresh_data)

    # -- data loading (background worker) --------------------------------------

    def action_force_refresh(self) -> None:
        """'r' — bypass all caches and pull fresh data."""
        self._refresh_worker(True)

    @work(exclusive=True, thread=True)
    def action_refresh_data(self, force: bool = False) -> None:
        try:
            history = nifty.fetch_history(use_cache=not force)
            ind = ta.compute(history.candles)
            states = snapshot(ind)
            events = scan_events(ind)
            try:
                chain = opts.fetch_chain(expiry=self.state.expiry,
                                         use_cache=not force)
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
            if force:
                self.call_from_thread(
                    lambda: self.notify("refreshed from source", timeout=3))
        except Exception as exc:
            self.call_from_thread(self._show_error, str(exc))

    def _refresh_worker(self, force: bool) -> None:
        # Re-enter through the worker so the UI never blocks on I/O.
        self.action_refresh_data(force)

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
        self._render_overnight_journal()

    def _render_overnight_journal(self) -> None:
        holder = self._holder("overnight")
        oj = shared_overnight_journal()
        
        # Determine filter params
        decision_filter = "all"
        trade_type = "all"
        dir_filter = "all"
        if self.oj_filter == "go":
            decision_filter = "GO"
        elif self.oj_filter == "no-go":
            decision_filter = "NO-GO"
        elif self.oj_filter == "actual":
            trade_type = "actual"
        elif self.oj_filter == "hypo":
            trade_type = "hypothetical"
        elif self.oj_filter in ("ce", "bullish"):
            dir_filter = "bullish"
        elif self.oj_filter in ("pe", "bearish"):
            dir_filter = "bearish"

        records = oj.list(
            decision=decision_filter,
            direction=dir_filter,
            trade_type=trade_type,
            search=self.oj_search,
            limit=100,
        )
        perf = compute_overnight_performance(journal=oj)
        
        title = (
            f"[bold]OVERNIGHT TRADE JOURNAL[/bold] — filter={self.oj_filter}"
            f"{' search=' + repr(self.oj_search) if self.oj_search else ''} · {len(records)} runs"
        )
        parts = [
            views.overnight_performance_panel(perf),
            views.overnight_journal_table(records, title=title),
        ]
        self._set(holder, *parts)

    def _render_market(self) -> None:
        hist = self.state.history
        holder = self._holder("market")
        if not hist:
            holder.update(Text("No data yet…", style="yellow"))
            return
        parts = views.market_view(hist)
        from ui.chart import candle_panel
        parts.append(candle_panel(hist))
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

    def action_copy_context(self) -> None:
        """'c' — copy the useful text for the active tab to the system
        clipboard."""
        tab = self.query_one(TabbedContent).active
        text = ""
        if tab == "journal":
            trades = shared_journal().list(limit=1)
            if trades:
                t = trades[0]
                parts = [t.contract_name]
                if t.option_type:
                    parts.append(f"expiry {t.expiry}")
                    parts.append(f"{t.lots}L @ ₹{t.entry_price:,.2f}")
                    if t.delta_entry is not None:
                        parts.append(f"Δ {t.delta_entry:+.2f}")
                else:
                    parts.append(f"{t.direction} @ {t.entry_price:,.2f}")
                parts.append(t.status)
                text = " | ".join(parts)
        elif not text:
            hist = self.state.history
            if hist:
                q = hist.quote
                text = (f"NIFTY {q.price:,.2f} {q.change:+.2f} "
                        f"({q.change_pct:+.2f}%)")
        if not text:
            self.notify("nothing to copy", severity="warning", timeout=3)
            return
        self.copy_to_clipboard(text)
        self.notify(f"copied → {text}", timeout=4)

    # -- journal input handling -----------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        event.input.clear()
        if not raw:
            return

        if event.input.id == "journal-input":
            feedback = self._run_journal_command(raw)
            if feedback:
                try:
                    self.copy_to_clipboard(feedback)
                    feedback += "   [copied to clipboard]"
                except Exception:
                    pass
                self.notify(feedback, severity="information", timeout=6)
            self._render_journal()
            self._render_performance()
        elif event.input.id == "overnight-input":
            feedback = self._run_overnight_journal_command(raw)
            if feedback:
                try:
                    self.copy_to_clipboard(feedback)
                    feedback += "   [copied to clipboard]"
                except Exception:
                    pass
                self.notify(feedback, severity="information", timeout=6)
            self._render_overnight_journal()

    def _run_overnight_journal_command(self, raw: str) -> str | None:
        oj = shared_overnight_journal()
        parts = shlex.split(raw)
        cmd, args = parts[0].lower(), parts[1:]
        if cmd == "oj" and args:
            cmd, args = args[0].lower(), args[1:]

        try:
            match cmd:
                case "filter":
                    if args and args[0] in ("all", "go", "no-go", "actual", "hypo", "ce", "pe", "bullish", "bearish"):
                        self.oj_filter = args[0]
                        return f"filtered overnight journal → {args[0]}"
                    return "filter must be all|go|no-go|actual|hypo|ce|pe"
                case "search":
                    self.oj_search = " ".join(args) or None
                    return f"search overnight journal → {self.oj_search}" if self.oj_search else "cleared search"
                case "settle":
                    if len(args) < 2:
                        return "usage: settle <id> <exit_price>"
                    rec_id = int(args[0])
                    exit_p = float(args[1])
                    rec = oj.settle(rec_id, exit_p)
                    if rec:
                        return f"settled #{rec_id} ({rec.contract_name}) @ ₹{exit_p:.2f} → P&L {rec.pnl_display} ({rec.outcome})"
                    return f"record #{rec_id} not found"
                case "run":
                    from model.overnight_card import build_overnight_setup
                    candles = self.state.history.candles if self.state.history else []
                    if not candles:
                        return "no market history loaded"
                    setup = build_overnight_setup(candles, self.state.chain)
                    return f"evaluated overnight setup → {setup.verdict} ({setup.composite.direction.value.upper()}, score {setup.composite.score:.0f})"
                case "show":
                    if not args:
                        return "usage: show <id>"
                    rec = oj.get(int(args[0]))
                    if not rec:
                        return f"record #{args[0]} not found"
                    return f"#{rec.id} | {rec.trade_date} | {rec.decision} {rec.contract_name} | EV lot ₹{rec.expected_value_lot or 0:,.0f} | P&L: {rec.pnl_display} | Gates: {rec.blocked_reasons or 'None'}"
                case "help":
                    return views.OVERNIGHT_JOURNAL_HELP
                case _:
                    return f"unknown overnight command {cmd!r} — try help"
        except (IndexError, ValueError) as exc:
            return f"bad overnight command: {exc}"

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

    # -- options journaling helpers -------------------------------------------

    def _resolve_expiry(self, token: str) -> str | None:
        """Expiry token: 1-based index into the chain, ISO date, or unique
        substring like 'aug'."""
        chain = self.state.chain
        if not chain or not chain.expiries:
            return None
        if token.isdigit():
            idx = int(token) - 1
            return chain.expiries[idx] if 0 <= idx < len(chain.expiries) else None
        for e in chain.expiries:
            if e == token:
                return e
        matches = [e for e in chain.expiries if token.lower() in e.lower()]
        return matches[0] if len(matches) == 1 else None

    def _fresh_leg(self, expiry: str, strike: float, opt_type: str):
        """Live leg lookup — bypasses the cache so trade prices are never
        stale."""
        from data import options as opts
        try:
            chain = opts.fetch_chain(expiry=expiry, use_cache=False)
        except Exception:
            return None
        for row in chain.for_expiry(expiry):
            if abs(row.strike - strike) < 0.51:
                return row.call if opt_type == "ce" else row.put
        return None

    def _open_option_trade(self, j, opt_type: str, args: list[str]) -> str:
        from datetime import datetime
        from model.options_scan import bs_greeks

        if not args:
            return "usage: add ce|pe <strike> <expiry#|date> [lots] [@premium]"
        strike = float(args[0])
        expiry = self._resolve_expiry(args[1]) if len(args) > 1 else (
            self.state.chain.expiries[0] if self.state.chain
            and self.state.chain.expiries else None)
        if not expiry:
            return (f"cannot resolve expiry {args[1]!r} — use a chain index "
                    f"(1, 2…), an ISO date, or a month fragment")
        lots = int(args[2]) if len(args) > 2 and args[2].isdigit() else 1
        premium = next((float(a[1:]) for a in args if a.startswith("@")), None)

        leg = self._fresh_leg(expiry, strike, opt_type)
        if premium is None:
            premium = leg.ltp if leg else None
        if not premium:
            return f"no LTP found for {strike:g} {opt_type.upper()} {expiry} — pass @premium"

        spot = (self.state.chain.underlying_value if self.state.chain else None) \
            or (self.state.history.quote.price if self.state.history else None)
        dte = max((datetime.strptime(expiry, "%Y-%m-%d")
                   - datetime.now()).days, 0.25)
        greeks = bs_greeks(spot or strike, strike, dte,
                           ((leg.iv if leg else None) or 14.0) / 100.0,
                           is_call=(opt_type == "ce"))
        lot_size = SETTINGS.lot_size
        trade = j.open_trade(
            instrument=SETTINGS.option_symbol,
            direction="long",
            entry_price=premium,
            quantity=lots * lot_size,
            stop_loss=round(premium * (1 - SETTINGS.default_stop_pct / 100), 2),
            target=round(premium * (1 + SETTINGS.default_stop_pct / 200), 2),
            strategy="option_buy",
            entry_reason=f"manual option entry ({lots} lot{'s' if lots != 1 else ''})",
            states=self._current_states(),
            option_type=opt_type,
            strike=strike,
            expiry=expiry,
            lots=lots,
            lot_size=lot_size,
            delta_entry=greeks["delta"],
        )
        warn = ""
        if datetime.now().strftime("%Y-%m-%d") == expiry:
            warn = "  ⚠ EXPIRY DAY"
        return (f"opened #{trade.id} {trade.contract_name} @ ₹{premium:,.2f} "
                f"(Δ {greeks['delta']:+.2f}, {lots}L = ₹{premium * lots * lot_size:,.0f})"
                f"{warn}")

    def _default_exit_price(self, j, trade_id: int) -> float | None:
        """Exit at the contract's current LTP when available."""
        t = j.get(trade_id)
        if t and t.option_type and t.strike is not None:
            leg = self._fresh_leg(t.expiry, t.strike, t.option_type.lower())
            if leg and leg.ltp:
                return float(leg.ltp)
        return self.state.history.quote.price if self.state.history else None

    def _run_journal_command(self, raw: str) -> str | None:
        j = shared_journal()
        parts = shlex.split(raw)
        cmd, args = parts[0].lower(), parts[1:]

        try:
            match cmd:
                case "add" | "take":
                    kind = args[0].lower()
                    if kind in ("ce", "pe"):
                        return self._open_option_trade(j, kind, args[1:])
                    direction = kind
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
                    price = float(args[1]) if len(args) > 1 \
                        else self._default_exit_price(j, trade_id)
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
