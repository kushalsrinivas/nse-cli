# Stock Overnight Engine — per-name naked CE/PE

Same overnight play as NIFTY (regime → composite → cohort gap distribution
→ Greek EV → fail-closed gates, 0.75d hold, Friday/weekend blocks), run
independently per constituent on its own tape, equity chain, lot size and
monthly expiry calendar. Fixed lots per GO; dry-run default.

## Commands

- `model_cli.py stock-overnight [--symbol X] [--lots N] [--journal] [--event ...]`
  Full 50 in one evening batch (~5–10 min; per-symbol progress lines,
  one failure never stops the batch). `--journal` records to the separate
  stock journal; re-runs skip symbols already recorded today (resumable).
- `main.py --stock-overnight [--symbol X] [--lots N] [--journal]`
- `main.py --settle-stock ID EXIT_PRICE`

## Files

- `data/equity_lots.py` — 50 lot sizes + freeze qty, pinned to
  upstox-instrument-master 2026-09-06. `lot_for()` raises on unknown
  symbols (never falls back to 75). Refresh quarterly.
- `data/options.py` — `type=Indices|Equities` routing (`chain_kind()`),
  threaded through provider ABC + `fetch_chain`/`fetch_expiries` (cache keys
  include kind; NIFTY paths unchanged).
- `model/overnight.py` — `apply_discipline(..., expiry_dates=)`: exact ISO
  matching when provided, else legacy NIFTY weekday heuristic.
- `model/overnight_card.py` — `build_overnight_setup(..., expiry_dates=,
  underlying=)`; Gate 3 exact-matches stock expiries; `OvernightSetup.lot_size`
  replaces two `* 75` hardcodes; candidate symbols use the underlying name.
- `model/options_ev.py` — `generate_strategy_candidates(..., underlying="NIFTY")`
  (default preserves NIFTY behavior exactly).
- `model/stock_overnight.py` — `evaluate_stock` / `evaluate_all`
  (settings = `replace(SETTINGS, lot_size=...)`; naked legs only).
- `journal/stock_overnight_db.py` + `stock_overnight_perf.py` — separate
  `stock_overnight_journal` table (symbol/lot_size/lots; NO-GO settles
  per-lot for avoided/missed measurement); perf overall + by symbol.

## Evidence (2026-09-06, Friday tape — expect NO-GO dominance)

- RELIANCE E2E: score 64.8 bullish, 30-signal cohort, Friday + Wilson +
  negative-EV gates engaged; contract correctly named `RELIANCE 1290 CE`.
- Full 50 dry-run: 50 evaluated, 0 errors, 0 skips, 0 GO (Friday block
  vetoes everything by design — the run proves robustness, not edge).
- Resume verified: second `--journal` pass SKIPs already-recorded symbols.

## Known approximations (do not trade real size on these)

1. NIFTY-trained learned weights score stock tapes (no per-stock training).
2. European BS on American-style equity options; no auto ex-div detection —
   pass known events via `--event`.
3. Equity-option fees reuse the index charge model (same STT structure).
4. Fixed *lots* concentrates rupee risk in high-premium names; consider
   fixed-*notional* later. 50 chains ≈ 100 NSE calls (sequential with the
   provider's pacing, resumable).
5. ON-A..D setups and breadth adjustment do NOT run per stock yet — the
   per-name card is NIFTY-logic on stock data; wiring the breadth overlay
   per name is the natural next step after the base proves out.
