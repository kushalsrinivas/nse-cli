# nifty-strats

NIFTY 50 terminal trading dashboard: a technical decision model, an
overnight options engine with distributional EV, NIFTY 50 constituent
breadth intelligence, and a Kite market-data service — with paper journals
for everything. **Paper trading only: this codebase never places orders.**

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # add ruff via pip install -e ".[dev]" for lint
python main.py                         # Textual TUI (tabs 1-8, `b` = breadth refresh)
python main.py --tonight               # one EOD verdict card (dry-run default)
python main.py --tonight --verbose --journal   # full audit trail + record it
```

Python ≥ 3.10. Market data needs network (yfinance/NSE by default; Kite
with a session — see below).

## Daily workflow

| Command | What |
|---|---|
| `main.py` | TUI: Market · Technicals · Options · Signals · Journal · Performance · Overnight · **Breadth (8)** |
| `main.py --tonight [--verbose] [--journal]` | One EOD run: fetch once, verdict first. `--journal` records |
| `model_cli.py stock-overnight [--symbol X] [--lots N]` | Naked CE/PE overnight per stock, ranked GO table (separate journal) |
| `model_cli.py evaluate / backtest / optimize / journal` | Score now · walk-forward report · fit weights · review setups |
| `model_cli.py breadth / breadth-backtest` | Constituent snapshot · NIFTY-only vs NIFTY+breadth ablation |
| `model_cli.py kite-login / kite-master / kite-parity / kite-live` | Kite session · instrument master · source parity · WS streaming |
| `main.py -oj` , `--settle-overnight ID PX` , `--settle-confluence ID PX` , `--settle-stock ID PX` | Journals + manual settlement |

All decision commands are **dry-run by default** (`--journal` records).
`tonight` never lifts a sub-threshold setup on breadth alone, and breadth
filters can only veto, never create, a trade.

## Architecture

```
ingestion            intelligence              decisions               records
─────────            ────────────              ─────────               ───────
data/nifty.py ─┐
data/options.py ─┼─▶ analysis/ ─▶ model/ ─▶ services/ ─▶ main.py / model_cli / tui
data/kite/* ───┘   indicators    regime/composite   (workflows:        paper journals
                   signals       EV/overnight/      tonight, stock,    (journal.db,
                                 breadth/setups     kite ops)          ignored by git)
```

- **`services/`** — workflows shared by every frontend (fetch bundles,
  tonight run, stock screen, kite ops). Returns data + notices; no UI code.
- **`model/`** — regime, composite scoring, options EV, overnight card,
  `breadth/`, `overnight_setups/` (ON-A..ON-D), backtest, risk.
- **`data/kite/`** — paper-only Kite service: auth, instrument master,
  rate-limited REST, own asyncio WS client + binary parser, tick
  aggregator, candle store. Secrets via `KITE_API_KEY`/`KITE_API_SECRET`
  env vars only — never in the repo.
- **`strategies/`** — seam used by the TUI to pre-fill paper trades
  (nothing auto-trades).

## Testing & lint

```bash
python -m unittest discover -s tests          # 141 tests, network-free
python -m ruff check .                        # curated rules, see pyproject.toml
```

Conventions: naive-local timestamps in journals; `x == x` idioms are
deliberate NaN guards (excluded from lint); broad `except Exception` in
fetchers is the degrade-don't-crash design.

## Docs

- `docs/BREADTH_LAYER.md` — breadth design, guardrails, ablation evidence
- `docs/STOCK_OVERNIGHT.md` — per-stock engine, lots, journal, caveats
- `docs/KITE.md` — Kite service: commands, data audit, production rules

## Known limitations (honest)

- Backtests simulate the underlying (ATR stops), not option fills; no
  slippage/fees in most paths — paper-trade before trusting anything.
- NIFTY volume feeds are unreliable (`^NSEI` ships none; index WS packets
  have none) — volume gates use proxies, disclosed where applied.
- Learned weights and single-leg pricing carry documented biases; see docs.

MIT — see LICENSE.
