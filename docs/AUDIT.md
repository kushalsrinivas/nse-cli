# Quantitative audit — findings and Stage 0 remediation

Full report: the "Overnight Engine Teardown" artifact (24 ranked findings,
measured skill vs. baselines, proposed architecture, staged plan).

Everything below was measured on this repo's own cached data by replaying
this repo's own functions. Sample: 1,236 daily bars, 2021-08-26 → 2026-09-15.

## The headline

| Metric | This model | Baseline |
|---|---|---|
| Directional hit rate on the overnight gap | **59.7%** (n=578) | 62.2% — always predict "up" |
| Brier, walk-forward cohort engine | **0.2318** (n=201) | 0.2218 — constant P(up)=0.622 |
| corr(cohort magnitude forecast, realized gap) | **−0.116** | — |
| Hit rate by score: 65-70 / 70-75 / 75-80 / 80+ | 0.562 / 0.587 / 0.596 / 0.582 | flat, non-monotonic |

The composite score carries no information about outcome. The cohort engine
has negative skill against a single hard-coded number.

The overnight gap is mostly driven by the global overnight session, which
this repo already fetches and then discards:

| Input (all known by 08:30 IST) | corr with gap | sign hit |
|---|---|---|
| S&P 500 overnight return | +0.533 | 69.6% |
| risk_pulse composite | +0.517 | 68.2% |
| Nasdaq | +0.496 | 68.1% |
| INDA | +0.458 | 67.1% |
| Walk-forward OLS on top 5 | +0.513 | **72.6%** (Brier skill +0.206) |

`model/overnight_card.py` passes `global_pulse=None` into the scenario
engine. GIFT Nifty is not fetched at all. Nikkei/KOSPI/Hang Seng correlate
~0 because the features use *yesterday's* close rather than the live
morning session.

## Stage 0 — completed

| ID | Defect | Fix |
|---|---|---|
| F-01 | `default_stop_pct = 0.30` meant 30%, every call site divided by 100 again → stop 0.3% wide → **74 lots, ₹832,500 premium on ₹500,000 equity** | Renamed `default_stop_frac` (a fraction, never a percent); fixed 5 call sites; added `max_premium_deploy_pct` ceiling in `RiskManager`, since the affordability check only ever tested one lot |
| F-02 | `optimize_weights` searched on the validation split and reported it as out-of-sample; shipped artefact recorded −0.451 R held-out and loaded on every run | Search moved to the train split; `optimize` refuses to persist a fit without > 0 R over ≥ 30 held-out trades; `learned_weights.json` deleted |
| F-03 | `structure_view()` never received the direction — recommended CE/bull_spread into a 65%-probability bearish continuation | Direction threaded through `structure_view` / `render_scenarios` and all call sites; verdicts expressed as with-view / against-view |
| F-08 | Breadth adjustment applied after the composite clamp → `confidence_score` of 101.0 on a "/100" card | Clamped in `compute_adjustment` and defensively in `pipeline.evaluate` |
| F-12 | On expiry evening every delta is ~0 or ~1, so "closest to 0.50" returned a δ = −1.00, γ = 0 contract labelled ATM, reported at **+₹4,595/lot EV on a ₹31.5 premium** | `generate_strategy_candidates` returns no candidates unless some strike sits in 0.35 ≤ \|δ\| ≤ 0.65 |
| F-14 | Three invariants were bare `assert`s in a live path — vanish under `-O`, otherwise crash the command | Replaced with fail-closed validation that marks the candidate untradeable and names the violation |
| F-23 | `overnight` journaled unconditionally (2026-09-10 held nine rows); `tonight` honoured dry-run | `overnight --journal` added, dry-run default; `OvernightJournal.add` is idempotent per (trade_date, engine_version) and never overwrites a settled row; existing duplicates collapsed 22 → 7 |
| — | No historical option chain, so the EV engine is unfalsifiable | New `journal/chain_archive.py` + `model_cli.py archive-chain`; round-trips to a real `OptionChain` |

22 regression tests in `tests/test_stage0_fixes.py`. Suite: 215 passing.

## Not fixed — decisions for the owner

- **Breadth ships enabled by default** while `docs/BREADTH_LAYER.md` says
  "advisory-only. Do NOT enable by default until the evidence changes"
  (IC −0.042 next-day, −0.004 gap). Flipping the default is a behaviour
  change, so it is left alone pending a call.
- **Sizing now blocks more often, correctly.** With a real 30% stop, one
  NIFTY lot at a ₹150 premium risks ₹3,375 — above the ₹2,500 that 0.5% of
  a ₹500,000 account allows. The engine says so instead of returning 74
  lots. Either raise `account_equity`, raise the risk tier, or accept that
  NIFTY lots are large for this account size.

## Next — Stage 1

Build the evaluation harness (purged walk-forward, named baselines,
Brier / log-loss / pinball / calibration, block-bootstrap intervals), then
run the *existing* engine through it and publish the result as the standing
baseline. Nothing after that ships without beating a named number.
