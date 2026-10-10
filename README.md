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

Python ≥ 3.10 (the multi-market platform needs ≥ 3.11 for `tomllib`). Market
data needs network (yfinance/NSE by default; Kite with a session — see below).

### Multi-market order-block platform (paper)

`platform_cli.py` runs the order-block methodology across indices, sectors
and stocks (NIFTY 50 … 500, sector indices, SENSEX), with separate bullish and
bearish pipelines, a central risk governor, paper execution for cash/futures/
options, replay research and a read-only web dashboard. Start with
[`docs/PLATFORM.md`](docs/PLATFORM.md) (run instructions, results, limitations):

```bash
python platform_cli.py init && python platform_cli.py universe refresh
python platform_cli.py backtest run --from 2025-10-01 --to 2026-09-30
python platform_cli.py run          # live paper session
python platform_cli.py dashboard    # http://127.0.0.1:8765
```

## Daily workflow

| Command | What |
|---|---|
| `main.py` | TUI: Market · Technicals · Options · Signals · Journal · Performance · Overnight · **Breadth (8)** |
| `main.py --tonight [--verbose] [--journal]` | One EOD run: fetch once, verdict first. `--journal` records |
| `model_cli.py stock-overnight [--symbol X] [--lots N]` | Naked CE/PE overnight per stock, ranked GO table (separate journal) |
| `model_cli.py evaluate / backtest / optimize / journal` | Score now · walk-forward report · fit weights · review setups |
| `model_cli.py breadth / breadth-backtest` | Constituent snapshot · NIFTY-only vs NIFTY+breadth ablation |
| `model_cli.py premarket` | 08:30 IST card; always journals a paper candidate separately from the model verdict |
| `model_cli.py premarket-journal` | Review forced/model paper candidates and data-blocked runs |
| `model_cli.py premarket --exit 23100CE@223.55` | Price a position you already hold: exit at the open, or hold? |
| `model_cli.py forecast-eval` | Score the forecast stack + the incumbent engine on the harness |
| `tonight --laya` · `premarket --laya` · `laya` · `laya-eval` | Laya veto filter on the overnight/premarket cards and Setups A/B/C (shadow default; `--laya-enforce` applies) |
| `model_cli.py archive-chain [--expiries N]` | Persist tonight's option chain — **cron this at ~15:25 IST** |
| `model_cli.py kite-login / kite-master / kite-parity / kite-live` | Kite session · instrument master · source parity · WS streaming |
| `model_cli.py ob-audit / ob-backfill / ob-record` | Order-block data: availability report · REST minute archive · WS leg recorder |
| `model_cli.py ob-quality / ob-coverage / ob-reconcile` | Order-block gates: data quality (exit 3 on CRITICAL) · option-quote coverage · live paper vs backtest |
| `model_cli.py ob / ob-paper / ob-backtest / ob-journal / ob-settle / ob-kill` | Order-block system (paper only): scan · live paper session · backtest + verdict · review · manual close · kill switch — see `docs/ORDER_BLOCKS.md` §10 |
| `main.py -oj` , `--settle-overnight ID PX` , `--settle-confluence ID PX` , `--settle-stock ID PX` | Journals + manual settlement |

All decision commands — `tonight`, `overnight`, `confluence` — are
**dry-run by default** (`--journal` records).
`tonight` never lifts a sub-threshold setup on breadth alone, and breadth
filters can only veto, never create, a trade.

`premarket` records one paper-run row on every invocation. Its forced paper
candidate is not a model GO: the genuine verdict and any failed gates remain
visible, and stale data, missing structures, or a risk-budget failure are
journaled without opening a paper position. The premarket maximum modeled loss
is ₹20,000 per position (4% of the default ₹500,000 account setting). This is
paper-only; no order is placed.

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
python -m unittest discover -s tests          # 215 tests, network-free
python -m ruff check .                        # curated rules, see pyproject.toml
```

Conventions: naive-local timestamps in journals; `x == x` idioms are
deliberate NaN guards (excluded from lint); broad `except Exception` in
fetchers is the degrade-don't-crash design.

## Docs

- `docs/BREADTH_LAYER.md` — breadth design, guardrails, ablation evidence
- `docs/STOCK_OVERNIGHT.md` — per-stock engine, lots, journal, caveats
- `docs/KITE.md` — Kite service: commands, data audit, production rules
- `docs/LAYA.md` — Laya veto filter: questions, policy, shadow mode, evaluation
- `docs/PLATFORM.md` — multi-market platform as built: architecture, methodology, run/configure/monitor, test/load/recovery results, limitations, baseline comparison
- `docs/PLATFORM_PLAN.md` — the platform's audit, design rationale, schemas and phased plan
- `docs/ORDER_BLOCKS.md` — order-block system (NIFTY options, paper-only): schemas, detection, scoring, backtest, risk, Kite wiring, runbook (§10)

## Known limitations (honest)

**There is no measured directional edge — at either decision point, under
any model tested.** Five years, purged walk-forward, named baselines:
EOD gap direction AUC 0.551, PREOPEN session direction AUC 0.539, neither
beating a constant. Gradient boosting does not help; no regime slice
survives multiple comparisons. The legacy composite score claims 70.4% at
80+ and delivers 61.6%.

**Two things do beat their baselines.** The PREOPEN gap direction (AUC
0.748, Brier 0.210 vs 0.248) — but it is known at 08:30 and you enter at
the 09:15 open, so it is context, not a trade. And the session range
forecast, which beats trailing realised vol though not India VIX.

**The one robust tradeable edge is short volatility.** Implied/realised =
1.19x over 1,221 sessions (95% CI [+0.107, +0.194]); P(|move| > 1 implied
sigma) = 0.218 against 0.317 if fair. Long-premium structures start ~19%
behind, and the legacy engine only ever bought premium.

Run `model_cli.py forecast-eval` to reproduce all of it. Details in
`docs/AUDIT.md`.

- **No historical option chain exists**, so the EV engine cannot be
  validated at all. `archive-chain` starts fixing this from today forward;
  it needs ~40 sessions before an options backtest is worth running.
- Backtests simulate the underlying (ATR stops), not option fills — and
  resolve same-bar stop/target optimistically, so win rates are flattered.
- NIFTY volume feeds are unreliable (`^NSEI` ships none; index WS packets
  have none) — volume gates use proxies, disclosed where applied. The
  thin-volume gate has no measured relationship to next-day outcomes.
- The forecast ignores the overnight global session, which is the dominant
  driver of the gap (S&P overnight return: r = +0.53, 69.6% sign accuracy).
- No learned weights ship. The previous artefact was fitted on its own
  validation split and recorded −0.451 R held-out expectancy; it was
  removed, and `optimize` now refuses to persist a non-positive fit.

MIT — see LICENSE.
