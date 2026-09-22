# Laya veto filter

[Laya](https://huggingface.co/convaiinnovations/laya) is an encoder that
answers typed questions (`choice` / `score` / `noul`) about a text state
in one forward pass. It generates no text. Here it acts as a **second
opinion on setups the rule engines already produced**. It does not
generate trades.

```
tonight card / premarket card / Setups A/B/C ──▶ Laya: 9 questions, 1 pass ──▶ keep │ veto
```

Laya judges the **whole decision card**: the overnight `tonight` verdict
(score, bucket history, best structure and EV, breadth, ON-A..D, blocking
reasons) and the 08:30 premarket card (gap model, vol edge, best
structure, risk, invalidation). For overnight it runs after the gates and
**before journaling**, so an enforced veto is what gets recorded.

## Commands

| Command | What |
|---|---|
| `model_cli.py tonight --laya` | Overnight card + Laya verdict (shadow: shown, not applied) |
| `model_cli.py tonight --laya-enforce --journal` | A vetoed GO night is recorded as NO-GO with `laya veto: ...` |
| `model_cli.py premarket --laya [--journal]` | Premarket card + Laya verdict; `--journal` records the verdict |
| `model_cli.py premarket --laya-enforce` | A vetoed TRADE becomes NO TRADE |
| `model_cli.py laya [--enforce] [--journal]` | Intraday Setups A/B/C, one verdict each |
| `model_cli.py laya-eval` | Kept vs would-veto on settled confluence rows, plus AUC of P(fail) |

Install: `.venv/bin/pip install laya` (pulls torch + transformers; the
repo is not pip-installable itself). The ~1.7 GB checkpoint downloads on
first use to the HF cache. Without laya, `--laya` prints a notice and the
card is unchanged.

Measured on this machine (M-series, laya 0.3.6, torch 2.14): model load
~55 s from cache, **3–8 s per card on CPU**; MPS was *slower* (4.6–7 s).
The README's 33 ms is a CUDA T4 figure. `LAYA_DEVICE=cpu` is recommended.
Cards are ~380 tokens; with these short questions laya leaves ~460 tokens
for state, so nothing is truncated.

Environment overrides: `LAYA_MODEL`, `LAYA_SUBFOLDER`, `LAYA_DEVICE`.

## Questions (`model/laya_filter/questions.py`)

direction (choice), market_regime (choice), setup_quality (score 0-3),
legitimate, regime_compatible, volatility_abnormal, likely_to_fail,
conflicting_signal, warrants_execution (noul).

The state is the setup's own condition trail rendered as words, plus spot,
VIX and events. It is not a numeric feature vector, because an encoder
reads words better than numbers.

## Veto policy (`VetoPolicy`, NOT fitted)

A GO is vetoed if any of these hold: P(fail) > 0.70, P(execute) < 0.30,
P(conflict) > 0.75, or Laya reads the opposite direction with confidence
≥ 0.70. Only a GO can be vetoed. An inference error never vetoes.

## Guardrails / honest status

- **Shadow mode is the default.** The breadth layer follows the same rule:
  a filter can veto a trade but never create or upgrade one.
- By Laya's own README, the base checkpoints score near chance zero-shot
  on out-of-domain decisions (0.36 vs 0.32 random). They were not trained
  on market data. Assume the verdicts carry **no signal** until
  `laya-eval` shows ≥30 settled rows per arm and an AUC of P(fail) above
  0.5.
- Laya's shipped temperatures are fitted on other domains, so its
  probabilities are not calibrated for this one.
- `laya-eval` joins only confluence outcomes today. Overnight verdicts are
  journaled as `ON-<date>` (joinable to `overnight_trade_journal.trade_date`)
  and premarket as `PM-<date>`, which has no outcome journal yet.
- The realistic path to value is to fine-tune on journal outcomes (Laya's
  RLCD notebook) and load the result via `LAYA_MODEL`. The labels come
  from the settled confluence journal.
