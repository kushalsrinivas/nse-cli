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

---

# Stages 1-6 — the forecast stack, and what the evidence actually says

Stage 0 fixed defects. Stages 1-6 asked the prior question: **is there
anything to forecast?** The answer changed the product.

## Stage 1 — the harness (`model/forecast/evaluate.py`)

Purged, embargoed walk-forward; named baselines; block-bootstrap intervals
on every paired loss difference. A model "ships" only when the 95% CI on
the difference excludes zero. Run it with:

    model_cli.py forecast-eval

It scores the incumbent engine and the new stack through identical folds,
so any later claim of improvement is like-for-like.

## The decision-point problem

The single most important structural finding, and it is not a bug — it is
a property of the trade:

| Decision point | When | Global session known? | Tradeable target |
|---|---|---|---|
| **EOD** | 15:25 IST | **No** — tonight's US session hasn't happened | gap into tomorrow |
| **PREOPEN** | 08:30 IST | **Yes** — Wall St closed ~01:30 | today's session (you enter at the 09:15 open) |

The overnight gap is highly forecastable *from the global session*, but
only at 08:30 — by which time you can no longer trade it. At 15:25, when
`overnight` actually decides, that information does not exist yet.
`features.py` makes `decision_point` a required argument for this reason
and lags the global block an extra day at EOD.

## Stage 2-3 — direction and distribution

Feature sets pre-registered by hypothesis, `l2=1.0` fixed in advance, all
results reported:

| Decision point | Target | Tradeable | Best Brier | Baseline | AUC | Ships? |
|---|---|---|---|---|---|---|
| EOD | gap direction | **yes** | 0.2536 | 0.2479 | 0.551 | **no** |
| PREOPEN | gap direction | no (context) | **0.2104** | 0.2479 | **0.748** | **YES** |
| PREOPEN | session direction | **yes** | 0.2556 | 0.2494 | 0.539 | **no** |
| PREOPEN | session range | yes, via vol | MAE 0.3348 | 0.3826 | r +0.347 | **YES** |

Quantile regression on either tradeable target failed to beat unconditional
quantiles (pinball delta +0.002 to +0.003, CIs straddling zero). So the
conditional *scale* is forecastable and the conditional *location* is not.

## Stage 4 — the volatility edge

| Question | Answer |
|---|---|
| Is there a volatility risk premium? | **Yes.** Implied/realised = **1.190x**, 95% CI on the difference [+0.107, +0.194], n=1,221. P(\|move\| > 1 implied sigma) = **0.218** against 0.317 if fairly priced. |
| Does our range forecast beat trailing realised vol? | **Yes** — MAE 0.335 vs 0.383, CI excludes zero. |
| Does it beat India VIX? | **No** — delta -0.004, CI [-0.022, +0.014]. VIX already knows what our features know. |
| Does (forecast - implied) predict the vol trade's P&L? | **No** — corr -0.028, quintiles non-monotonic. |

Horizon discipline matters here and nearly caught us out a second time: an
early run compared VIX against a Parkinson range estimate, which covers
only the intraday session and excludes the gap, and produced a fake 1.89x
premium. Matched to close-to-close it is 1.19x. This is the same class of
error as F-09.

## Stage 6 — ablations (run early, because they decide what to build)

- **Nonlinearity.** Gradient-boosted stumps vs ridge logistic vs constant,
  both decision points: no configuration beat the constant. CIs straddle
  zero throughout.
- **Regime conditioning.** Twelve slices (vol, trend, ADX, gap size). The
  best — ADX mid-tercile, AUC 0.634 — is what one expects to find by
  chance when examining twelve slices. Not pursued without pre-registration
  and confirmation on a different period.

## What this means

**There is no directional edge at either tradeable decision point**, under
any model class tested, in any regime tested. The old engine's composite
score claims 70.4% at scores of 80+ and delivers 61.6%.

**There is one robust edge, and it is short volatility.** Options have run
~19% rich over five years. Every long-premium structure starts that far
behind. The audited engine only ever bought premium.

## What was built (`model/forecast/`)

    features.py       L1  decision-point-aware feature rows
    evaluate.py       L8  the harness, built first
    models.py             ridge logistic / ridge / quantile / HAR / GBM, numpy only
    volatility.py     L2  VIX-anchored, every quantity carries its horizon
    distribution.py   L3  location + scale x empirical (fat-tailed) shape
    levels.py         L5  P(touch) vs P(close beyond), from a measured touch curve
    options_edge.py   L6  edge = EV(our distribution) - EV(implied distribution)
    decision.py       L7  five views; NO TRADE is the default
    engine.py             orchestration
    report.py             the pre-market card
    legacy.py             adapters so the incumbent is scored identically

Commands: `model_cli.py premarket` and `model_cli.py forecast-eval`.

`options_edge.evaluate_structure` computes P&L as `exit_value - price_paid
- friction`, so **EV now moves when you overpay** — verified: paying 115%
of quoted changes the number, where the audited engine's premium term
cancelled algebraically (F-10). The structure menu includes defined-risk
short-volatility structures, and on live data the iron butterfly and iron
condor rank first on edge while every long-premium structure is negative
after the hurdle — the measured premium reproducing itself in the ranking
without being hard-coded there.

## Honest limits

- **No historical option chain**, so L6 is validated on logic and on
  VIX-as-implied, never on real fills. `archive-chain` is accumulating the
  data; ~40 sessions are needed.
- **The edge is real but may be out of reach.** The short-vol structures
  that rank first risk ~₹23,000/lot, against a ₹2,500 budget at 0.5% of
  ₹500,000. The engine now says exactly what equity or risk tolerance the
  trade would need rather than just "too small".
- **`up_session` base rate is 0.479** — the session is down more often than
  up, while the gap is up 62% of the time. All the drift is overnight, and
  none of it is capturable by an intraday entry.
- GIFT Nifty, the most direct read of the NIFTY open, is still not fetched.
  It would sharpen the gap forecast further but cannot make it tradeable.

---

## Follow-on: `premarket --exit` — closing the loop on the overnight trade

The measured edge (PREOPEN gap, AUC 0.748) cannot be used to *enter* an
overnight trade, because at 15:25 the global session has not happened. It
can be used to *exit* one you already hold. `--exit` marks the position
across the gap distribution and asks the only question left at 08:30:

    model_cli.py premarket --exit 23100CE@223.55 --expiry 2026-09-22
    model_cli.py premarket --exit 23100CE@223.55 --exit -23300CE@120.10

Two corrections came out of building it.

**Overnight IV change is measured, not assumed.** The audited engine
hard-coded a Friday hold at **-0.8** VIX points. Measured over 1,229
sessions the Friday-to-Monday change is **+0.507** (95% CI [+0.363,
+0.651]) — the opposite sign, on the case where the assumption mattered
most. Weekday entries run -0.08 to -0.17, and the unconditional mean is
indistinguishable from zero (95% CI [-0.055, +0.054]). Caveat stated in
the code: VIX is close-to-close, so this spans the next session rather than
only the move to the open — an upper bound, and the only proxy this repo
has data for.

**The entry must be marked at what you paid.** Marking a 23100CE filled at
223.55 using the chain's quoted IV of 14.93% reprices it at 215.12 — an
instant -₹632/lot the position never lost. `attach_ivs` now solves for the
vol implied by the fill, so the only P&L drivers left are the move, the
decay and the IV change.

The verdict carries a materiality band: when holding changes expected value
by less than friction plus forecast error, it says **TOO CLOSE TO CALL —
take the open** rather than manufacturing a decision from noise. On the
worked example the difference was ₹64 on a ₹16,766 position.

## Also fixed here (pre-existing, unrelated to the audit)

`ensure_master` compared a DATE stamp parsed to midnight against `now`, so
after 20:00 local a master refreshed that morning read as 20 hours old and
was refetched every evening. `tests/test_source.py` failed for the same
reason whenever the suite ran late. Now compares dates: today's master is
fresh all day, yesterday's stays usable through the early morning before
today's is published.
