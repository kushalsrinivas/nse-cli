# Kite Market-Data Foundation (Phases 0–2)

Paper-only. No order placement anywhere in this package.

## Commands

- `model_cli.py kite-login [--request-token T] [--logout]` — session setup.
  Login lasts till 6 AM IST; credentials from `KITE_API_KEY`/`KITE_API_SECRET`
  only; token cache `~/.config/nifty-strats/kite_session.json` (0600).
- `model_cli.py kite-master [--exchange X]` — daily master refresh.
- `model_cli.py kite-parity [--days 60] [--all]` — Kite vs yfinance closes
  + NSE-scrape vs Kite chain LTPs. Must be clean before trusting Kite data.
- `model_cli.py kite-live [--minutes 375] [--tape-every 300]` — stream
  NIFTY + 50 stocks (quote mode), aggregate 1m candles, live tape print,
  EOD 1d flush. Ctrl-C flushes cleanly.

## Architecture

```
WS (own asyncio client, own binary parser)
  -> TickAggregator (exchange-ts 1m bins, grace-held emission)
  -> CandleStore (1m 90d, 1d 10y, 5m derived) + latest-tick registry
REST (kiteconnect, per-bucket limiters) -> master / quotes / historical
Strategy consumes: Candle frames, OptionChain dataclass, BreadthSnapshot
```

## Data audit (against the official spec)

| # | Question | Answer |
|---|---|---|
| 1 | Obtainable | LTP/OHLC/prev-close/change; equity vol + VWAP; buy/sell pending qty; 5×2 depth; option/future LTP/OHLC/vol/VWAP; OI + OI day H/L; trade + exchange timestamps; circuits; lot/tick sizes |
| 2 | Not obtainable | Index volume/depth/avg (index packets lack them); IV/Greeks (compute via BS inversion); turnover; change-in-OI (derive vs stored); expired tokens (archive master daily); US/Asia bench (Yahoo stays) |
| 3 | Derived internally | Chg-OI, PCR splits, turnover (vol×avg), chain snapshots, ATM/OTM, breadth, gap cohorts, scenarios, 5m rollups, 1d EOD |
| 4 | Realtime | WS ticks (NIFTY quote, 50 stocks quote, legs/futures full in Phase 3); live tape; provisional bins (never stored) |
| 5 | Historical | REST day/minute (+OI) → same Candle/frame dataclasses as yfinance |
| 6 | Persisted | Master (expired rows kept), settled 1m (90d), 1d (10y). Ticks: ring buffer only |
| 7 | Feeds strategy | Frames → indicators/regime/composite/breadth (unchanged); chain builder + EOD-from-store in Phase 3 |
| 8 | Bottlenecks | 50-chain evening assembly (~100 NSE-quote calls at 1/s in Phase 3); historical backfill 3/s |
| 9 | Rate-limit risks | Buckets enforced in-client with sleep (quote 1/s, hist 3/s, other 10/s); 429 only on bugs; 3 WS conns/key, we use 1 |
| 10 | Weaknesses | Daily re-login is manual (6 AM expiry); LTP-mode ticks lack exchange ts (arrival bucketing); index has no volume (volume gate must migrate to futures/breadth); single WS connection (no failover conn yet) |
| 11 | For overnight | Futures basis + OI distribution → ON-A/ON-D inputs; EOD chain snapshot 15:30 replaces scrape; breadth EOD from store instead of yfinance batch |

## Production rules enforced

Reconnect exp-backoff (1s→60s) + full resubscribe/mode restore; 30s
heartbeat watchdog; staleness judged on exchange ts (inhibits GO in
Phase 3); (token, ts, ltp, vol) dedup; volume-monotonic candle math;
5s grace-held emission with late-drop counters; session filter
09:15–15:30 IST + weekends; unique (exchange, tradingsymbol) keys with
token remap on master refresh; 403 → explicit relogin (no silent retry);
per-module logging; health via client/store/aggregator counters;
graceful shutdown flushes store.

## Phase 4 — Kite-first routing (`data/source.py`)

Every NSE-sourced feed resolves through one router: **Kite when a session
is valid, legacy Yahoo/NSE-scrape fallback otherwise.** Tri-state
`--source auto|yahoo|kite` (default `auto`; explicit `kite` fails loud
without a session). Macro/global feeds (`model/macro.py`: US indices,
crude, FX, Asia) are deliberately untouched, as are the parity baselines
in `services/kite_ops.py` and `scripts/*` research tools.

- NIFTY history / chains, constituent bundles, stock chains+expiries,
  5m intraday (`fetch_intraday`), India VIX — all Kite-first.
- Tonight EOD proxies NIFTY volume from the front future (index packets
  carry none); verdict card shows `source: kite` + basis/OI provenance.
- `get_india_vix()` (120s TTL) feeds the EV benchmark, scenario engine
  and confluence VIX gates; each keeps its Yahoo-macro fallback.

## Phase 3 — chain builder, EOD-from-Kite, futures positioning

- `data/kite/chain.py`: chains assembled from master legs + one /quote
  batch; per-leg IV recovered by bisecting our BS pricer (None when no
  time value → EV engine's 14% fallback). `KiteChainProvider.chain_for`
  is primary; NSE scrape is fallback on any failure.
- `tonight --source kite` (default still yahoo): NIFTY history + breadth
  frames + chain all from Kite. Index volume is proxied from the front
  future (logged on the verdict card) — this retires the `^NSEI` NaN
  volume hack on the Kite path.
- `stock-overnight --source kite`: per-stock Kite history + assembled
  chain + master expiries, same flags otherwise.
- `data/kite/eod.py`: one `EodContext` per underlying (candles, spot,
  futures basis bps, futures OI day-change %).
- ON-A gains "futures orderly (|basis|≤40bp)"; ON-D gains "no OI shock
  (|ΔOI|<25%)". Both read N/A without data — never FAIL on absence.
- Audit-table updates: chain snapshots now obtainable (assembled, not
  native); index volume still unobtainable (proxied, disclosed);
  IV/Greeks derived via inversion (was: unavailable).
