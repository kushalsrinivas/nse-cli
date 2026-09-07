# NIFTY 50 Constituent Breadth Layer — change log, rationale, evidence

Branch: `feature/constituent-breadth-layer` (from `main @ ced5f14`).
`main` is untouched and clean; all work below lives on the feature branch.
Status: **advisory-only. Do NOT enable by default until the evidence changes.**

## 1. What changed (files)

| File | Change |
|---|---|
| `model/breadth/universe.py` (new) | Canonical 50: NSE `ind_nifty50list.csv` membership (verified 2026-09-05), Jun-2026 factsheet weights, sector tags. TMPV replaces delisted TATAMOTORS; entrants ASIANPAINT / INDIGO / JIOFIN / MAXHEALTH replace HEROMOTOCO / BRITANNIA / INDUSINDBK / DIVISLAB. |
| `data/constituents.py` (new) | One batched `yf.download` for the universe, `data/nifty.py`-style normalize, TTL cache (`v2` key), empty-frame → missing, weight-coverage accounting. |
| `model/breadth/features.py` (new) | Pure per-stock features: 1d/5d/20d/60d returns, overnight gap vs intraday split, 20d rel-strength vs NIFTY, volume ratio + anomaly, realized vol, ROC-10, SMA20/50 position, 20d range position + highs/lows, beta/corr-60, abnormal-move flag. Walk-forward safe. |
| `model/breadth/aggregate.py` (new) | `BreadthSnapshot`: equal- AND cap-weighted breadth (adv/dec, SMA position, highs/lows, volume, confirming/diverging, heavyweight top-8, sector breadth, top-5 share / HHI / effective-N concentration, breadth acceleration, −100…+100 composite, BROAD/LEAN/CONCENTRATED/NARROW/DIVERGENT participation). |
| `model/breadth/divergence.py` (new) | bull/bear-trap, concentrated-rally, breakout/breakdown confirm-vs-divergence, resistance-weakness, heavyweight-divergence; severity 0–3 + rationale. |
| `model/breadth/scenarios.py` (new) | 5 overnight scenarios (continuation / flat / reversal / gap-against / event-vol) from posture priors + bounded log-odds shifts (breadth, divergence, VIX, global, events); structure screen (CE/PE/spreads/hedged/none); EOD narrative. |
| `model/breadth/integration.py` (new) | Guardrails: bounded ±8pts, **never lifts sub-65 across the gate** (clamps at 64.9), downgrades always allowed, coverage-gated, regime-weighted, never flips direction, never touches sizing directly. |
| `model/breadth/backtest.py` (new) | Ablation replay (baseline vs breadth arms, identical dates/signals/outcomes) + model-free evidence: IC (next-day + gap), breadth quintiles, participation split, divergence-filter P&L. |
| `model/breadth/live.py`, `view.py` (new) | Shared snapshot assembly; Rich renderers. |
| `model/pipeline.py` | `evaluate(..., breadth=None, use_breadth=True, persist=True)`: opt-in adjustment, breadth fields persisted into `setups.indicator_scores` JSON + `notes` (no schema migration). Default `None` = byte-identical NIFTY-only baseline. |
| `model/overnight_card.py` | `build_overnight_setup(..., breadth=None)`: divergence caution gate (sev-3 opposing blocks; sev-2 blocks marginal <72), scenarios attached, rationale + `signal_scores` enriched (no DB migration). |
| `model_cli.py` | `breadth` (snapshot + scenarios + narrative), `breadth-backtest` (ablation report), `--breadth` flags on `evaluate` / `overnight`. |
| `tests/test_breadth.py` (new) | 22 tests, synthetic only: features, broad vs concentrated tapes, equal-vs-weighted, divergence flags, guardrails, scenario math, live assembly, ablation determinism. |

## 2. Why this shape

- **Opt-in, not default.** The brief demands proof of incremental edge before complexity ships. Defaults reproduce the baseline exactly.
- **Evidence before sizing.** Breadth moves the score within ±8pts; sizing tiers, EV math, and risk gates are untouched.
- **No migrations.** All breadth state rides in existing JSON columns (`indicator_scores`, `signal_scores`) + rationale text.
- **Fail-closed inherits.** Coverage <70% weight or <35 names → adjustment is exactly 0.

## 3. Evidence (real data, 2026-09-05)

Ablation `breadth-backtest --period 2y` (497 NIFTY bars, 203 replay days,
50/50 constituents, loop cross-checked to match canonical `backtest`
trade-for-trade: 65 trades, exp −0.417R, PF 0.377, DD 30.66R):

- **Δ expectancy +0.000 R/trade, Δ win rate +0.00pp** — identical trade sets; no score sat close enough to a gate for ±8pts to matter.
- **IC breadth→next-day −0.042, IC breadth→gap −0.004** — no rank edge.
- **Quintiles** (next-day mean): Q1 −0.050%, Q2 +0.148%, Q3 +0.004%, Q4 +0.103%, Q5 −0.085% — hump-shaped, extremes revert; no tradeable monotonicity.
- **Participation split** (baseline expectancy): BROAD (37) −0.312R, CONCENTRATED (6) −0.667R, DIVERGENT (18) −0.531R, LEAN (4) −0.500R. Directionally consistent with "concentrated moves fail" but n=6 and everything is negative.
- **Divergence filter**: 0 skips on the corrected universe (1 skip / +1.0R net on the stale universe) — too rare to matter.

**Verdict: no incremental edge. Keep breadth advisory-only** (the harness prints exactly this). Contributing context: the NIFTY-only baseline itself is −0.417R on this window (matches the −0.451R validation expectancy already encoded in `learned_weights.json`), so there is no working model for breadth to improve yet — fix the base model first.

## 4. Data-quality findings (fixed)

1. `TATAMOTORS.NS` is delisted (Oct-2025 demerger) — replaced with `TMPV.NS` per NSE; universe re-verified against the official CSV (5 substitutions total).
2. Empty frames were counted as covered (coverage showed 100% with a dead symbol) — empty frames now go to `missing`; cache key bumped to `v2`.
3. Yahoo gap: `M&M.NS`, `BAJAJ-AUTO.NS`, all 5 new symbols verified resolvable.

## 6. The `tonight` command (recommended EOD path)

`model_cli.py tonight` (or `main.py --tonight`) replaces the two-command
`evaluate --breadth` + `overnight --period 2y --breadth` dance:

- **Fetch once**: NIFTY history + chain + constituents share one bundle.
- **Lazy breadth**: NIFTY-only screen first; the 50-ticker fetch happens
  only if the base score clears 50 (`TONIGHT_SCREEN_SCORE`).
- **Dry run by default**: nothing is journaled unless `--journal` is passed
  (`build_overnight_setup(..., record=False)` also suppresses the inner
  pipeline persist — verified by test with journal sentinels).
- **Verdict first**: one `TONIGHT — GO/NO-GO` card (score walk, breadth
  line, scenario odds, structures or top-3 blockers); full EV bridge +
  breadth + macro behind `--verbose`.
- Flags: `--period` (default 2y), `--cperiod` (default 6mo),
  `--event` (repeatable), `--no-breadth`, `--journal`, `--verbose`.

The older `evaluate --breadth` / `overnight --breadth` paths are unchanged
for backward compatibility, but they always fetch breadth and always
journal — prefer `tonight`.

## 7. What would change the verdict (follow-ups, not done)

1. Fix the negative-expectancy base model first (micro-stop bug `config.py:64`, inverted `analysis/signals.py` crosses, `learned_weights.json` trained on −0.45R) — re-run ablation after.
2. 5y window + per-regime IC (breadth may work at breakouts; current harness pools all regimes).
3. Count clamp frequency (base in [57,65) + confirming breadth) to size the "almost matters" set.
4. Replace INDA proxy with real GIFT futures for the global leg of scenarios.
5. Quote-level constituent volumes are equity volumes, not index flow — consider futures/OI signing for buildup features.

## 8. TUI Breadth tab (key `8`)

The terminal (`main.py`) has an 8th tab: tape read + breadth panel + all
50 constituents (1d%, MACD histogram sign, close vs EMA9/EMA21/SMA20/SMA50,
volume multiple + anomaly flag, UP/DN/FLAT trend vote) + sector breadth +
divergence. `c` on this tab copies the one-line tape read.

Fetch policy (deliberate): **not** on the 60s auto-refresh. Loads lazily
on first visit (cached) or via `b` (force refresh), in an isolated worker
group so it never blocks or cancels the market refresh. If the tab is
opened before NIFTY history arrives, the load is deferred until it lands.
Stale snapshots show their fetch time; scenarios/EV stay in `tonight` —
the tab is intelligence, not a decision.

## 9. Overnight setups ON-A..ON-D (breadth-aware checklists)

`model/overnight_setups/` mirrors the intraday confluence format (named
setups, PASS/FAIL/NA conditions, GO/NO-GO/WATCH) but every condition is
overnight-native and breadth-aware. Missing inputs read N/A, never FAIL.

- **ON-A Broad Trend Hold** (candidate, bull/bear): score≥65 + BROAD
  participation + ≥55% cap-weighted confirmation + divergence sev≤1 +
  cohort n≥10 + VIX<22 + P(cont)≥28%. GO needs zero FAIL and ≥4 PASS.
- **ON-B Narrow Thrust Stand-Aside** (filter): ≥2 of narrow participation,
  top-5 ≥60%, trap flag sev≥2, confirmation <45%, heavyweight drag ->
  NO-GO + **blocks the night**; else WATCH (clear).
- **ON-C Exhaustion Reversal** (candidate, counter-trend): |day|≥0.4% +
  relief flag + confirm <45% + P(reversal)≥18% + VIX<25 + no events.
- **ON-D Event & Volatility Stand-Down** (filter): events or VIX≥22 or
  P(event-vol)≥25% -> NO-GO + **blocks**; else WATCH (clear).

Filters can only veto, never create, a trade. Candidates annotate; the EV
engine still decides. Wired into `build_overnight_setup` (Gate 2c +
rationale `setups:` bit), `tonight` verdict line + `--verbose` checklists,
and the TUI Breadth tab (VIX/cohort read N/A there).

Evidence, same 2y replay (baseline trades split by setup state; VIX/cohort
N/A historically, so ON-A GO rests on 5/7 legs):

- **ON-A GO: 32 trades, exp −0.250R, win 38% vs ON-A not GO: 33 trades,
  exp −0.579R, win 21%.** First split in this project that separates
  "less bad" from "terrible" at n≈30/side: breadth confirmation halves
  losses and nearly doubles win rate. Still negative — the base model is
  broken — but ON-A is measuring something real.
- ON-B triggered once in 2y (too strict to matter historically).
- ON-C never GO'd on score≥65 days (exhaustion rarely coincides with
  trend-scores; expect it to matter as WATCH/qualifier, not trigger).