# Order-block system for NIFTY options — design spec

Status: **design only, nothing in this document is implemented yet.**
Execution: **paper only.** No module described here calls `place_order`.
Horizons: intraday (MIS-equivalent, flat by 15:15) and overnight
(NRML-equivalent, decided ~15:20, exited at or after the next open).

This spec is written against `feature/laya-filter`. It reuses the Kite
stack (`data/kite/*`, `data/source.py`), the chain archive, `RiskManager`,
the forecast harness (`model/forecast/evaluate.py`), `options_edge`, and
the Laya shadow filter. It adds an order-block package, a few tables, a
paper broker and a backtester, and nothing else.

---

## 0. Read this first: what the repo already knows

`docs/AUDIT.md` measured **no directional edge at either tradeable
decision point** (EOD gap AUC 0.551, PREOPEN session AUC 0.539). It also
found that long premium starts **~19% behind**, because implied/realised
vol is 1.19x. An order-block system is a directional, usually
long-premium strategy. So the design commits to three things up front:

1. **Overnight OB trades run as shadow signals by default.** They are
   journaled but get no paper position until the harness shows they beat
   the named baselines (§6.5). The EOD decision point is exactly where the
   audit found nothing. A pattern-based entry has to clear the same bar.
2. **The option leg is chosen by `options_edge`, not hard-coded to a
   naked CE/PE.** A bullish block can be expressed as a long CE, a bull
   call spread, or a **bull put credit spread**. The credit spread is the
   only one that sides with the measured vol premium. Each structure must
   clear `LONG_PREMIUM_HURDLE` (or its short-vol analogue) after friction.
3. **Score is a ranking, not a probability**, until isotonic calibration
   on ≥ 100 settled out-of-sample signals exists (§4.4).

None of this blocks building the system. It decides what "GO" is allowed
to mean while there is no evidence yet.

---

## 1. Module layout

```
data/kite/
  instruments.py        (exists)  + snapshot_history() writer, §2.1
  legs.py               NEW  LegRecorder: ATM±N option legs + front fut on WS
  backfill.py           NEW  REST minute backfill → ob_series_1m / option_candles_1m
model/order_blocks/
  __init__.py
  types.py              Bar, Swing, Zone, Signal, TradePlan dataclasses
  swings.py             confirmed k-left/k-right pivots, no look-ahead
  structure.py          trend state, BOS / CHoCH events
  detect.py             displacement leg → source candle → zone (+FVG, sweep)
  lifecycle.py          ACTIVE → TOUCHED → TRIGGERED/INVALIDATED/EXPIRED
  score.py              0–100 component score + hard gates
  contract.py           underlying plan → option structure (via options_edge)
  engine.py             bar-driven orchestrator, identical for live and backtest
  backtest.py           event simulator on 1m bars + harness integration
  view.py               rich card (matches confluence/overnight views)
journal/
  ob_db.py              zones, signals, paper orders/fills/positions, events
services/
  order_blocks.py       ob_scan / ob_paper_loop / ob_settle workflows
execution/
  paper_broker.py       NEW  kiteconnect-shaped order API, simulated fills
  kill_switch.py        NEW  file/env kill switch checked before every order
```

`engine.py` takes completed bars one at a time and returns events. The
live loop and the backtester call **the same function** on the same bar
objects. That is the main defence against look-ahead: there is no
separate vectorised detector that could peek.

CLI (added to `model_cli.py`, dry-run by default like every other
decision command):

| Command | What |
|---|---|
| `ob-audit` | Data-availability report (§2.6): what Kite gives for the window you ask about |
| `ob-backfill --days 365` | REST minute backfill of NIFTY spot + front futures into `ob_series_1m` |
| `ob [--tf 15m] [--journal] [--laya]` | Live scan: active zones, any setups, contract plan |
| `ob-paper [--minutes 375]` | Live loop on kite-live ticks: zones → signals → paper broker |
| `ob-journal` / `ob-settle ID PX` | Review / manual settle (mirrors `cj`, `oj`) |
| `ob-backtest [--from --to] [--grid]` | Underlying walk-forward; option layer where archive exists |
| `ob-kill [--off]` | Engage / release the kill switch |

---

## 2. Data

### 2.1 What Kite gives, and what this design does about each gap

| Need | Kite reality | Design response |
|---|---|---|
| NIFTY spot 1m/5m/15m/60m OHLC | Historical API, minute data in ≤ 60-day requests, years deep; WS index packets | `ob-backfill` → `ob_series_1m`; live from `TickAggregator` |
| Volume for displacement/rvol | **Index packets carry no volume** (KITE.md audit row 2) | Front-month NIFTY future 1m volume, rolled by rule (§2.3) |
| Futures OI | Historical with `oi=1`, WS full/quote packets | Stored on fut series; context only, not scored in v1 |
| Option history for **expired** contracts | **Not served.** Tokens are reused after expiry | Forward-collect from now on (`LegRecorder`, chain archive). Nothing fixes the past |
| Option bid/ask history | Not in historical candles at all | `option_quotes` captured from WS `full` depth at 1/min (+ on every signal) |
| IV / Greeks | Not provided | `data/kite/chain.py` BS inversion (exists); Greeks from the same pricer |
| Lot size per contract and date | In master, but `kite_instruments` **overwrites** on refresh | New append-only `kite_instrument_history` (§2.2) |
| Live, session-aware candles | `TickAggregator` (exchange-ts bins, grace-held) | Reuse; resample 1m → 5m/15m/60m on read |

Consequence for backtests: **the underlying signal can be backtested over
years; the option P&L can only be validated on data captured from the day
`LegRecorder` starts.** Anything earlier is a synthetic reprice (§6.3).
Reports label it `PROVISIONAL`.

### 2.2 Schemas (SQLite, shared `journal.db`)

Conventions follow the repo: naive IST wall-clock text timestamps
(`"%Y-%m-%d %H:%M"` for bars, ISO seconds for events), `REAL` prices,
`TEXT` ids, `CHECK` constraints on enums, `CREATE … IF NOT EXISTS` in the
owning module. Options are keyed by **(exchange, tradingsymbol)**, never
by token, because tokens are reused.

```sql
-- Append-only master history. kite_instruments keeps only the latest
-- attributes per symbol; this keeps every day's view, so a backtest on
-- date D resolves lot/tick/expiry as they were on D.
CREATE TABLE IF NOT EXISTS kite_instrument_history (
    exchange TEXT NOT NULL,
    tradingsymbol TEXT NOT NULL,
    as_of TEXT NOT NULL,                 -- master download date
    instrument_token INTEGER NOT NULL,
    name TEXT DEFAULT '',
    instrument_type TEXT NOT NULL,       -- EQ / FUT / CE / PE / INDEX
    expiry TEXT DEFAULT '',
    strike REAL,
    lot_size INTEGER,
    tick_size REAL,
    UNIQUE (exchange, tradingsymbol, as_of)
);
CREATE INDEX IF NOT EXISTS idx_kih_lookup
    ON kite_instrument_history(exchange, instrument_type, expiry, as_of);

-- Long-lived underlying series. kite_candles_1m prunes at 90 days and is
-- token-keyed; backtests need years and stable names.
CREATE TABLE IF NOT EXISTS ob_series_1m (
    series TEXT NOT NULL,                -- 'NIFTY_SPOT' | 'NIFTY_FUT1'
    ts TEXT NOT NULL,                    -- bar start, IST, minute floor
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume INTEGER,                      -- NULL for spot (index has none)
    oi INTEGER,
    contract TEXT DEFAULT '',            -- tradingsymbol behind FUT1 on this bar
    source TEXT NOT NULL CHECK(source IN ('kite_hist', 'kite_ws')),
    UNIQUE (series, ts)
);

CREATE TABLE IF NOT EXISTS option_candles_1m (
    exchange TEXT NOT NULL,
    tradingsymbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume INTEGER NOT NULL DEFAULT 0,
    oi INTEGER,
    source TEXT NOT NULL CHECK(source IN ('kite_hist', 'kite_ws')),
    UNIQUE (exchange, tradingsymbol, ts)
);

-- Top of book. Candles say what traded; this says what was executable.
CREATE TABLE IF NOT EXISTS option_quotes (
    exchange TEXT NOT NULL,
    tradingsymbol TEXT NOT NULL,
    captured_at TEXT NOT NULL,           -- local receipt, ISO seconds
    exchange_ts TEXT,                    -- from packet; NULL if absent
    spot REAL,
    ltp REAL,
    bid REAL, bid_qty INTEGER,
    ask REAL, ask_qty INTEGER,
    depth_json TEXT DEFAULT '',          -- 5x2 levels, only on signal/fill snapshots
    volume INTEGER, oi INTEGER,
    iv REAL,                             -- BS inversion of mid, NULL if no time value
    reason TEXT NOT NULL DEFAULT 'periodic'
        CHECK(reason IN ('periodic', 'signal', 'fill', 'exit', 'open_snapshot')),
    UNIQUE (exchange, tradingsymbol, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_oq_sym_ts ON option_quotes(tradingsymbol, captured_at);
```

Order-block domain tables (`journal/ob_db.py`):

```sql
CREATE TABLE IF NOT EXISTS ob_zones (
    zone_id TEXT PRIMARY KEY,            -- sha1(series|tf|source_bar_ts|direction|params_hash)[:16]
    series TEXT NOT NULL,                -- structure series, normally 'NIFTY_SPOT'
    timeframe TEXT NOT NULL CHECK(timeframe IN ('5m','15m','60m','1d')),
    direction TEXT NOT NULL CHECK(direction IN ('bullish','bearish')),
    source_bar_ts TEXT NOT NULL,         -- the order-block candle
    bos_bar_ts TEXT NOT NULL,            -- candle whose CLOSE broke structure
    first_eligible_ts TEXT NOT NULL,     -- bos_bar_ts + one bar (bar it can first be traded on)
    zone_low REAL NOT NULL,
    zone_high REAL NOT NULL,
    zone_mid REAL NOT NULL,              -- 50% "mean threshold"
    broken_swing REAL NOT NULL,          -- swing level the BOS closed through
    broken_swing_ts TEXT NOT NULL,
    leg_origin REAL NOT NULL,            -- extreme where the impulse started
    atr_at_bos REAL NOT NULL,
    disp_body_atr REAL NOT NULL,         -- largest body in impulse leg / ATR
    disp_range_atr REAL NOT NULL,        -- leg range / ATR
    rvol REAL,                           -- NULL when no futures volume
    fvg_low REAL, fvg_high REAL,         -- NULL when no FVG in leg
    swept_level REAL,                    -- liquidity taken before the move, else NULL
    kind TEXT NOT NULL DEFAULT 'BOS' CHECK(kind IN ('BOS','CHOCH')),
    status TEXT NOT NULL CHECK(status IN
        ('ACTIVE','TOUCHED','TRIGGERED','INVALIDATED','EXPIRED','CONSUMED')),
    touched_ts TEXT, closed_ts TEXT,
    close_reason TEXT DEFAULT '',
    bars_alive INTEGER NOT NULL DEFAULT 0,
    params_hash TEXT NOT NULL,           -- detector config fingerprint
    engine_version TEXT NOT NULL DEFAULT 'ob-v1',
    mode TEXT NOT NULL CHECK(mode IN ('live','backtest')),
    run_id TEXT NOT NULL,                -- backtest run or live session
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obz_status ON ob_zones(status, timeframe);
CREATE INDEX IF NOT EXISTS idx_obz_run ON ob_zones(run_id);

-- Immutable. A changed plan is a NEW row that references the old one.
CREATE TABLE IF NOT EXISTS ob_signals (
    signal_id TEXT PRIMARY KEY,          -- sha1(zone_id|trigger_ts|horizon|engine_version)[:16]
    zone_id TEXT NOT NULL REFERENCES ob_zones(zone_id),
    supersedes TEXT,                     -- prior signal_id, if revised
    horizon TEXT NOT NULL CHECK(horizon IN ('intraday','overnight')),
    trigger_ts TEXT NOT NULL,            -- close of the trigger bar
    decided_at TEXT NOT NULL,            -- wall clock of decision
    direction TEXT NOT NULL CHECK(direction IN ('bullish','bearish')),
    -- underlying plan
    u_entry REAL NOT NULL, u_stop REAL NOT NULL, u_target REAL NOT NULL,
    u_rr REAL NOT NULL,
    -- score
    score REAL NOT NULL,
    score_json TEXT NOT NULL,            -- {component: points, ...}
    p_win_calibrated REAL,               -- NULL until calibration ships (§4.4)
    -- option plan
    structure TEXT DEFAULT '',           -- long_call / bull_call_spread / bull_put_spread / ...
    legs_json TEXT DEFAULT '[]',         -- [{tradingsymbol, qty, side, bid, ask, iv, delta}]
    o_entry REAL, o_stop REAL, o_target REAL,   -- net premium per unit
    ev_rupees REAL,                      -- options_edge EV per lot after friction
    stress_loss_per_lot REAL,            -- §7.2
    lots INTEGER NOT NULL DEFAULT 0,
    lot_size INTEGER,
    risk_rupees REAL,
    -- verdict
    decision TEXT NOT NULL CHECK(decision IN ('GO','WATCH','NO-GO','SHADOW')),
    gates_json TEXT NOT NULL,            -- [{name, status, detail}], same shape as ConditionCheck
    blocked_reasons TEXT DEFAULT '',
    laya_json TEXT DEFAULT '',
    data_age_sec REAL,                   -- staleness at decision time
    vix REAL,
    engine_version TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('live','backtest')),
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_decision ON ob_signals(decision, horizon);

CREATE TABLE IF NOT EXISTS ob_paper_orders (
    order_id TEXT PRIMARY KEY,           -- 'PO-' + uuid4 hex[:12]
    signal_id TEXT NOT NULL REFERENCES ob_signals(signal_id),
    tag TEXT NOT NULL,                   -- ≤20 chars, = signal_id[:12]+leg; kite-compatible
    leg_index INTEGER NOT NULL,
    exchange TEXT NOT NULL DEFAULT 'NFO',
    tradingsymbol TEXT NOT NULL,
    transaction_type TEXT NOT NULL CHECK(transaction_type IN ('BUY','SELL')),
    product TEXT NOT NULL CHECK(product IN ('MIS','NRML')),
    order_type TEXT NOT NULL CHECK(order_type IN ('MARKET','LIMIT','SL','SL-M')),
    quantity INTEGER NOT NULL,           -- units, multiple of lot_size
    price REAL, trigger_price REAL,
    purpose TEXT NOT NULL CHECK(purpose IN ('entry','stop','target','time_exit','gap_exit','manual','kill')),
    status TEXT NOT NULL CHECK(status IN ('OPEN','COMPLETE','CANCELLED','REJECTED')),
    filled_qty INTEGER NOT NULL DEFAULT 0,
    avg_price REAL,
    status_message TEXT DEFAULT '',
    placed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (tag, purpose, leg_index)     -- idempotency: replays cannot double-enter
);

CREATE TABLE IF NOT EXISTS ob_paper_fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES ob_paper_orders(order_id),
    filled_at TEXT NOT NULL,
    qty INTEGER NOT NULL,
    price REAL NOT NULL,
    book_bid REAL, book_ask REAL,        -- book used for the fill
    book_age_sec REAL,
    fill_model TEXT NOT NULL,            -- 'touch_ask', 'walk_depth', 'gap_open', 'ltp_fallback'
    charges REAL NOT NULL                -- §6.4 cost model, per fill
);

CREATE TABLE IF NOT EXISTS ob_paper_positions (
    position_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL UNIQUE REFERENCES ob_signals(signal_id),
    horizon TEXT NOT NULL,
    structure TEXT NOT NULL,
    lots INTEGER NOT NULL, lot_size INTEGER NOT NULL,
    entry_net REAL NOT NULL,             -- net premium per unit (debit +, credit −)
    u_stop REAL NOT NULL, u_target REAL NOT NULL,
    o_stop REAL, o_target REAL,
    opened_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('OPEN','CLOSED')),
    closed_at TEXT, exit_net REAL,
    exit_reason TEXT CHECK(exit_reason IN
        ('u_stop','o_stop','target','time','gap','eod','expiry_guard','kill','manual')),
    trigger_side TEXT,                   -- which rule fired: 'underlying' | 'option'
    gross_pnl REAL, charges REAL, net_pnl REAL,
    r_multiple REAL,                     -- net_pnl / risk_rupees
    mae_rupees REAL, mfe_rupees REAL,
    gap_pnl REAL                         -- overnight only: mark at 09:15 vs prior 15:30
);

-- Append-only event log. Recovery replays this.
CREATE TABLE IF NOT EXISTS ob_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,                  -- zone_new, zone_touch, signal, order, fill, exit, gate_block, kill, stale, reconnect
    ref_id TEXT,
    payload_json TEXT NOT NULL,
    run_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ob_backtest_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    git_sha TEXT NOT NULL,
    params_json TEXT NOT NULL,
    data_from TEXT NOT NULL, data_to TEXT NOT NULL,
    fold_spec_json TEXT NOT NULL,
    option_layer TEXT NOT NULL CHECK(option_layer IN ('none','archived','synthetic')),
    summary_json TEXT NOT NULL
);
```

Backtest trades reuse `ob_signals` + `ob_paper_positions` with
`mode='backtest'`. Live and simulated results are then read by the same
queries and the same performance code.

### 2.3 Futures-volume roll rule

`NIFTY_FUT1` = nearest monthly future **until the close of T−2 trading
days before its expiry**, then the next. The `contract` column records
which. rvol is computed only within one contract. A 20-bar window that
spans a roll is marked `rvol=NULL`, and the volume component then reads
N/A (scored 0, never FAIL). This matches how ON-A/ON-D treat missing
data.

### 2.4 Bars and timeframes

Base: 1m spot (`ob_series_1m` or live aggregator). Derived on read with
`origin="start_day"` so that 09:15 anchors every bin:

| TF | Role | Bars/session |
|---|---|---|
| 60m | Higher-timeframe bias (HTF) | 7 (last bin 15:15–15:30 is partial; it is **dropped**, not used) |
| 15m | Zone detection (intraday + overnight) | 25 |
| 5m | Entry trigger (intraday) | 75 |
| 1m | Simulation only (stop/target sequencing) | 375 |

A bar is **complete** only once a 1m bar with `ts ≥ bin_end` has
settled. That is the aggregator's grace-held emission, so it is
5 s + one tick late in live, and exact in backtest. Engine inputs are
complete bars only. The in-flight bar never reaches `engine.py`.

### 2.5 Python types (`model/order_blocks/types.py`)

```python
@dataclass(frozen=True)
class Bar:
    ts: datetime          # bin start, IST naive
    tf: str               # '5m' | '15m' | '60m'
    open: float; high: float; low: float; close: float
    volume: int | None    # FUT1 volume aligned by ts; None if unavailable

@dataclass(frozen=True)
class Swing:
    kind: Literal['high', 'low']
    price: float
    pivot_ts: datetime        # the extreme bar
    confirmed_ts: datetime    # close of pivot+k bar — the earliest it is knowable
    broken: bool = False

@dataclass
class Zone:                   # mirrors ob_zones 1:1
    ...

@dataclass(frozen=True)
class TradePlan:
    direction: str
    u_entry: float; u_stop: float; u_target: float
    horizon: Literal['intraday', 'overnight']

@dataclass
class ObSignal:               # mirrors ob_signals 1:1; gates reuse ConditionCheck
    ...
```

### 2.6 `ob-audit`: the first deliverable

Before any detector code, run the audit and commit its output to
`docs/ORDER_BLOCKS_DATA_AUDIT.md`. It answers each of these with a
number, not a yes/no:

1. NIFTY spot 1m coverage via historical API: first date, gaps per
   session, bars ≠ 375 per session.
2. Front-future 1m coverage, volume populated %, OI populated %.
3. Kite 15m vs Yahoo 15m closes (reuse `kite-parity` code); max diff.
4. For today's NFO master: NIFTY expiries listed, strikes per expiry, the
   **lot_size of each live NIFTY contract** (see §7.1 on `config.lot_size`).
5. Historical minute candles for a *live* NIFTY option: how far back from
   listing they go. This sets how much option history `ob-backfill` can
   still rescue for unexpired series.
6. `option_chain_snapshots` coverage so far (`ChainArchive.coverage()`).
7. WS `full` mode on 80 option tokens for 10 minutes: depth present %,
   exchange_ts present %, median quote age, median spread by moneyness.

---

## 3. Detection rules (deterministic, ordered)

All parameters live in one frozen `ObParams` dataclass. `params_hash` is
its fingerprint, and every zone and signal records it.

```python
@dataclass(frozen=True)
class ObParams:
    pivot_k: int = 3               # left = right = k
    atr_n: int = 14
    disp_body_atr: float = 1.0     # largest body in impulse leg ≥ this × ATR
    disp_range_atr: float = 1.5    # OR leg range ≥ this × ATR
    max_leg_bars: int = 6          # impulse must break structure within N bars
    rvol_n: int = 20
    rvol_min: float = 1.2          # gate only when volume is present
    max_zone_atr: float = 1.0      # wider zones trimmed to body, then rejected if still wider
    zone_age_bars: int = 20        # on detection TF
    stop_buffer_atr: float = 0.10
    min_u_rr: float = 1.5
    sweep_lookback: int = 10
```

### 3.1 ATR and relative volume

- `TR_t = max(H_t − L_t, |H_t − C_{t−1}|, |L_t − C_{t−1}|)`; ATR is
  Wilder's RMA over `atr_n` **completed** bars on the same TF. ATR is
  undefined for the first `atr_n` bars of history, but is carried across
  sessions, so the overnight gap enters TR exactly once.
- `rvol_t = V_t / mean(V_{t−rvol_n} … V_{t−1})`. The current bar is
  excluded from its own baseline. Time-of-day matters (09:15 bars are
  always heavy), so v1 also stores
  `rvol_tod_t = V_t / median(V at the same bin over the prior 20 sessions)`.
  The score uses `rvol_tod` when ≥ 10 sessions exist, else `rvol`.

### 3.2 Confirmed swings

Bar `i` is a swing high iff `H_i > max(H_{i−k..i−1})` and
`H_i ≥ max(H_{i+1..i+k})`. Strict on the left and non-strict on the right,
so equal highs resolve to the earlier bar. Swing lows mirror this. The
swing's `confirmed_ts` is the **close of bar `i+k`**. Before that, no code
path may read it. `swings.py` exposes only
`confirmed_swings(as_of: datetime)`, and the engine passes the current
bar's close time.

The existing `latest_swing_pivot` in `model/confluence/indicators.py` uses
`i+1` and does not track confirmation time. It is **not** reused.

### 3.3 Structure: BOS vs CHoCH

Track `last_high` / `last_low`: the most recent confirmed swing of each
kind not yet broken. Trend state is `up` once a confirmed higher-high and
higher-low pair exists, `down` for the mirror, `none` otherwise.

On each completed bar `t`:

- **Bullish break**: `C_t > last_high.price`, and `last_high` is not
  broken. It is **BOS** if trend is `up` or `none`, and **CHoCH** if trend
  was `down`. Then mark `last_high.broken = True`. Wicks through the level
  without a close beyond it do nothing.
- Bearish mirrors this with `C_t < last_low.price`.

### 3.4 Displacement leg and the source candle (bullish shown)

Given a bullish break at bar `b` of swing high `S`:

1. **Leg origin** `o`: the bar with the lowest low in
   `(S.pivot_ts, b]`. If the leg from `o` to `b` exceeds `max_leg_bars`,
   reject. That is a grind, not displacement.
2. **Displacement test**: over bars `o..b`, either
   `max(|C−O|) ≥ disp_body_atr × ATR_b` or
   `(H_b − L_o) ≥ disp_range_atr × ATR_b`. Record both ratios.
3. **Source candle** `s`: the **last bearish candle** (`C < O`) in
   `[o − 3, b − 1]`, searched backward from `b − 1`. If none is bearish,
   use bar `o` itself. Doji (`|C−O| < 0.1 × ATR`) count as opposing.
4. **Zone** `= [L_s, H_s]`. If `H_s − L_s > max_zone_atr × ATR_b`, drop
   the upper wick: `[L_s, max(O_s, C_s)]`. If that is still too wide,
   reject. `zone_mid = (low + high) / 2`.
5. **Volume gate** (only when FUT1 volume is present): `max(rvol)` over
   `o..b` must reach `rvol_min`. If volume is absent the gate is N/A, not
   FAIL.
6. **FVG** (optional confluence): any `j` in `o+1..b` with
   `L_{j+1} > H_{j−1}` (needs bar `b+1`, so FVG is evaluated one bar later
   and attached to the zone then). Store `[H_{j−1}, L_{j+1}]` of the
   largest one.
7. **Liquidity sweep**: in `[o − sweep_lookback, o]`, some bar took out a
   confirmed swing low (`L < swing_low.price`, swing confirmed before that
   bar) and closed back above it. Store the swept level.
8. `first_eligible_ts = close of bar b` (the zone can be traded from bar
   `b+1`). The zone row is written with `status='ACTIVE'`.

Bearish mirrors this throughout (last bullish candle before a downside
break below `last_low`).

### 3.5 Lifecycle

Evaluated on every completed detection-TF bar `t > b`, in this order:

1. **Invalidated** if `C_t < zone_low` (bullish) / `C_t > zone_high`
   (bearish). A close through, not a wick.
2. **Touched** (first time only) if `L_t ≤ zone_high` (bullish) /
   `H_t ≥ zone_low` (bearish). `touched_ts = t`. A zone first touched on
   bar `b+1` is legal. Being touched on bar `b` itself is impossible by
   construction.
3. **Expired** after `zone_age_bars` bars with no trigger. Intraday 15m
   zones also expire at **15:15 the same session**. 60m/overnight zones
   carry across sessions up to the age limit.
4. **Superseded**: a newer same-direction zone whose range overlaps by
   > 50% replaces the older one (older → `EXPIRED`,
   `close_reason='superseded'`). This keeps stacked zones from multiplying
   signals off one move.
5. **TRIGGERED → CONSUMED**: one trade attempt per zone, ever.

### 3.6 Entry triggers

**Intraday** (detection 15m, trigger 5m):

- After `touched_ts`, look at 5m bars. The trigger is the first 5m
  **close above the most recent 5m swing high confirmed after the
  touch** (a 5m CHoCH back in the zone's direction). The 5m swing must
  itself be confirmed (k=2 on 5m), with the confirmation rule from §3.2.
- The 5m trigger bar's low must not have closed below `zone_low`.
- Window: triggers between **09:30 and 14:45** only. 09:15–09:30 is
  dominated by the gap auction, and after 14:45 there isn't enough time to
  reach a target before 15:15.
- Entry = next 5m bar's open (backtest), or market at decision (live/paper).

**Overnight** (detection 15m/60m, decision at the 15:15 bar close):

- An ACTIVE or TOUCHED bullish zone where the 15:00–15:15 15m bar closes
  **inside or above the zone**, and the 60m HTF bias is not `down`.
- The decision is made at 15:16–15:20 and the paper entry is filled at
  15:20–15:25 quotes. The chain archive cron at 15:25 then captures the
  same book.
- Exit at the next session. The rules for this are in §5.3.

### 3.7 Underlying plan

- `u_stop = zone_low − stop_buffer_atr × ATR` (bullish).
- `u_target` = the nearest opposing liquidity beyond entry: the next
  unbroken confirmed swing high on 15m, else the prior-day high, else
  `entry + 2 × (entry − u_stop)`.
- `u_rr = (u_target − u_entry) / (u_entry − u_stop)`. Must be
  `≥ min_u_rr` or the signal is NO-GO.

---

## 4. Setup scoring (0–100)

### 4.1 Components: exact formulas

`clip01(x) = min(1, max(0, x))`. Each component is a piecewise-linear
ramp between a floor (0 points) and a full-credit point.

| # | Component | Max | Points |
|---|---|---|---|
| 1 | Structure | 20 | BOS with trend = 20; CHoCH = 12; plus nothing else |
| 2 | Displacement | 20 | `20 × clip01((disp_body_atr_obs − 0.8) / (1.8 − 0.8))` |
| 3 | Volume | 15 | `15 × clip01((rvol_tod − 1.0) / (2.0 − 1.0))`; volume absent → 0 and component marked N/A |
| 4 | Liquidity sweep | 15 | sweep present = 15, else 0 |
| 5 | FVG | 10 | FVG overlaps zone = 10; FVG in leg but not overlapping = 5; none = 0 |
| 6 | HTF alignment | 10 | 60m trend agrees = 10; `none` = 5; opposes = 0 |
| 7 | Freshness | 5 | first touch within 6 bars of BOS = 5; linear to 0 at `zone_age_bars` |
| 8 | Zone tightness | 5 | `5 × clip01((1.0 − zone_width_atr) / (1.0 − 0.3))` |
| | **Total** | **100** | |

Bands: **< 60 reject · 60–74 WATCH · ≥ 75 eligible.** These are
pre-registered. They do not move after looking at backtest results. If
they turn out wrong, that is a finding to report, not a parameter to tune.

### 4.2 Hard gates (override any score)

Each gate becomes a `ConditionCheck` in `gates_json`, the same type the
confluence engine uses, so the existing views can render it.

| Gate | FAIL when |
|---|---|
| Zone state | not ACTIVE/TOUCHED at trigger |
| Underlying R:R | `u_rr < 1.5` |
| Data fresh | newest spot tick exchange_ts > 60 s old, or any leg quote > 10 s old |
| Feed healthy | WS state ≠ connected, or a reconnect within the last 2 min |
| Session window | outside the trigger windows in §3.6 |
| Event block | listed in `--events` (RBI, budget, results days, FOMC/CPI night for overnight) |
| Expiry guard | intraday: expiry day after 13:00. Overnight: would hold into expiry day (DTE at exit < 1) |
| Contract liquidity | §5.2 filters fail for every candidate |
| Option EV | best structure's `ev_rupees ≤ 0` after friction + hurdle |
| Risk | `RiskManager.size()` blocked (reason copied verbatim) |
| Kill switch | engaged |
| Evidence gate (overnight) | harness has not passed §6.5 → decision becomes **SHADOW**, not GO |

### 4.3 Decision

```
score < 60                         → NO-GO
any gate FAIL                      → NO-GO (reasons listed)
60 ≤ score < 75                    → WATCH (journaled, no paper order)
score ≥ 75, all gates PASS/N/A     → GO (intraday) | SHADOW (overnight, until §6.5 passes)
Laya --laya-enforce veto           → NO-GO with 'laya veto: ...' (can only veto, never upgrade)
```

### 4.4 From score to probability

`p_win_calibrated` stays NULL until there are ≥ 100 settled
**out-of-sample** signals per horizon. After that it is set by isotonic
regression of `win = r_multiple > 0` on `score`, fitted on walk-forward
training folds only. Its quality is reported as Brier vs the constant
base rate, using `score_binary` and `calibration_table` from
`model/forecast/evaluate.py`. If Brier skill ≤ 0, the score stays a
ranking and the card says so.

---

## 5. Underlying signal → NIFTY option trade

### 5.1 Sequence

1. Underlying plan (§3.7) passes.
2. Build the candidate chain: `KiteChainProvider.chain_for("NIFTY", spot,
   expiry, wings=10)` for each allowed expiry. IVs are attached by BS
   inversion (exists).
3. Build structures for the direction (bullish shown):

   | Structure | Legs | Why it's on the menu |
   |---|---|---|
   | `long_call` | +1 CE, \|δ\| 0.45–0.60 | simplest; pays the full vol premium |
   | `bull_call_spread` | +1 CE δ≈0.50, −1 CE at/just past `u_target` | caps vega/theta drag; target is already known |
   | `bull_put_spread` | −1 PE below `u_stop`, +1 PE one-two strikes lower | the only structure on the side of the measured vol premium; risk defined |

4. Price each one with `options_edge.evaluate_structure` over a
   **horizon-matched** distribution: intraday = remaining-session
   distribution, overnight = gap distribution. Neither is the audit's
   location-free one; the OB signal supplies the location and the harness
   decides whether that location is real. Pick the max-EV structure that
   passes §5.2. If none has EV > 0, the gate fails.
5. Map stops: `o_stop` = structure value repriced at `u_stop` (same IV,
   DTE reduced by expected hold). The option's own stop is the looser of
   that value and a 40% premium loss on debit structures. **The
   underlying stop is primary.** The option stop only exists to catch IV
   collapse while the underlying sits still. The position records which
   one fired.

### 5.2 Contract filters

- Expiry: intraday = nearest weekly with DTE ≥ 1 (DTE 0 allowed only
  before 13:00). Overnight = nearest weekly with **DTE ≥ 2 at entry**.
  Expiry weekday comes from the master, never hard-coded.
- Delta (long legs): 0.45 ≤ |δ| ≤ 0.60. This matches the F-12 band
  0.35–0.65 already enforced in the repo, but tighter.
- Spread: `(ask − bid) ≤ max(2 × tick, 1.5% × mid)` on every leg.
- Depth: `ask_qty` (buys) / `bid_qty` (sells) ≥ order quantity at top of
  book, else the fill walks depth (§6.4) and the slippage is priced in.
- OI ≥ 50,000 units on each leg (filters out stale far strikes).
- Quote age ≤ 10 s.

### 5.3 Exits

| Horizon | Exit rule (first to fire) |
|---|---|
| Intraday | `u_stop` hit (1m close through, or 1m low ≤ stop when within 0.1 ATR. See §6.2 sequencing) · `u_target` touched · `o_stop` · **15:15 time exit** · kill switch |
| Overnight | At 09:15–09:20 next session: if the open is beyond `u_stop` → **gap exit at the first executable bid**. Else the position becomes intraday-managed with the same stop/target, and a hard time exit at 10:30 (max hold ≈ 1 session). `premarket --exit` runs at 08:30 as advice and is journaled; v1 does not auto-act on it |

---

## 6. Backtesting

### 6.1 Two layers, labelled honestly

| Layer | Data | Covers | Label |
|---|---|---|---|
| Underlying | `ob_series_1m` (years) | Zones, triggers, R-multiples on NIFTY | `VALIDATED` once §6.5 passes |
| Options, archived | `option_candles_1m` + `option_quotes` + chain archive (from start date forward) | Actual contract, actual book | `VALIDATED` once n ≥ 40 overnight / 100 intraday |
| Options, synthetic | BS repricing with VIX-as-IV + `MEASURED_IV_CHANGE` | Anything before archives exist | `PROVISIONAL`, never used for promotion |

### 6.2 Event simulator rules (`model/order_blocks/backtest.py`)

- Same `engine.py` and the same bar objects as live. The simulator only
  supplies complete bars in time order and a `PaperBroker` with
  `fill_source=historical`.
- **Intrabar sequencing on 1m**: when stop and target fall in the same 1m
  bar, **stop first**. (Note: the existing `model/backtest._simulate_outcome`
  checks target before stop despite its comment. Do not copy it.)
- Stops on the underlying fire on 1m data. Fill = stop price (intraday)
  or **the open of the gapping bar** if the bar opens beyond the stop.
- Option fills: buy at ask, sell at bid from the nearest `option_quotes`
  row at or before the decision time (≤ 10 s old), else from the 1m
  option candle with a half-spread penalty: `median_spread(moneyness, DTE
  bucket)` measured from the quotes table. With neither available, the
  trade is **excluded**, not filled at LTP.
- Overnight gap: exit priced from the 09:15–09:16 `open_snapshot` quotes.
  `gap_pnl` is stored separately.
- One position per zone. The concurrency caps from §7 apply inside the
  backtest exactly as live.

### 6.3 Synthetic option layer (exploratory only)

Price the chosen structure with `bs_price` at entry using VIX/100 ×
term-structure factor 1.0 as IV, plus friction from §6.4. At exit,
reprice at the simulated spot with IV shifted by `MEASURED_IV_CHANGE`
(weekday-specific, from `model/forecast/position.py`) for overnight, and
unchanged for intraday. The report always prints the synthetic and
archived numbers side by side when both exist, so the bias of the
synthetic layer is measured, not assumed.

### 6.4 Costs (per executed order, configurable `CostModel`)

Rates change, so verify them against Zerodha's brokerage calculator when
implementing and pin them in config with a `costs_as_of` date:

| Item | Options rate (store in config) |
|---|---|
| Brokerage | ₹20 per executed order |
| STT | sell side, % of premium |
| Exchange txn charge (NSE) | % of premium |
| SEBI fee | ₹ per crore of premium |
| Stamp duty | buy side, % of premium |
| GST | 18% on brokerage + txn + SEBI |
| Slippage | buys at ask, sells at bid; walk depth beyond top-of-book qty |

The cost model reconciles with `estimate_fees_per_lot` in
`model/options_ev.py`. One function, not two.

### 6.5 Validation protocol: what "beats" means

Use `walk_forward` from `model/forecast/evaluate.py`: expanding train,
**one month test folds, 5-session purge + embargo**. Unit of observation
is the **session**, not the trade, because same-move signals are not
independent. Daily R is summed and bootstrapped in blocks of 5 sessions.

Pre-registered baselines (same direction, timing, stop width and target
rule, so only the zone logic differs):

| Baseline | Definition |
|---|---|
| B0 random | Same count of entries per session at random 5m bars in the trigger window, direction from a fair coin |
| B1 drift | Same entry times, always long |
| B2 all-zones | Every zone that touches, no score/gates (tests whether the scoring adds anything) |
| B3 HTF-only | Enter on 60m trend direction at the same times (tests whether OB adds anything over trend) |

Promotion of a horizon from SHADOW → paper GO requires **all** of:

1. Expectancy in R after costs > 0, with the 95% block-bootstrap CI
   excluding 0, on ≥ 150 OOS trades (intraday) / ≥ 60 (overnight).
2. `paired_delta_ci` vs B0, B2 and B3 excludes 0 in the strategy's favour.
3. Robustness: on the pre-registered grid
   `pivot_k ∈ {2,3,4} × disp_body_atr ∈ {0.8,1.0,1.25} × rvol_min ∈
   {1.0,1.2,1.5} × zone_age ∈ {12,20,30}` (81 configs), ≥ 70% of configs
   are positive OOS. The default config is chosen **before** the run and
   reported whatever its rank.
4. No single calendar year contributes > 50% of total R.
5. Score bands are monotonic: mean R of 75+ > 60–74 > < 60.

If (1)–(5) fail, the system stays a shadow journal. That is a legitimate
outcome and the report says it plainly, the way `docs/AUDIT.md` does.

### 6.6 Metrics reported

Expectancy (R and ₹), profit factor, win rate, avg win/loss, max
drawdown (₹ and R) and duration, MAE/MFE distribution, gap P&L
distribution (overnight), trades/session, exposure time, results by
year / VIX tercile / DTE bucket / structure / score band, calibration
table, and the baseline deltas with CIs.

`E = p_w·W̄ − (1 − p_w)·L̄ − C̄`, reported per trade in ₹ and in R.

---

## 7. Risk controls

### 7.1 Before anything ships: `config.lot_size`

`Settings.lot_size = 75`. NSE revised NIFTY's lot size during 2025–26,
and the master carries the current value per contract. The OB system
**must take lot size from the contract's master row** (via
`kite_instrument_history` for the trade date) and never from config. The
`ob-audit` step prints the live value. The existing overnight/premarket
engines still read `config.lot_size`, which is a separate fix worth
making on its own.

### 7.2 Sizing

Reuse `RiskManager.size()` with OB-specific inputs:

- `R` = `equity × tier`. Tier from score: 75–84 → `risk_normal` (0.5%),
  85+ → `risk_high` (0.75%). `risk_exceptional` is **never** used by this
  system.
- `L_unit` = per-unit loss under **stress**, not at the planned stop:
  - Intraday: value at `u_stop` + 1 tick of slippage per leg + exit half-spread.
  - Overnight: value at `spot × (1 ∓ 1.5 σ_gap)` (σ_gap from the forecast
    vol layer) with IV shifted by the measured weekday change
    (+0.507 vol pts on Friday entries), floored at the gap-exit bid model.
    Capped at net debit for long premium; at spread width − credit for
    credit spreads.
- `lots = floor(R / (L_unit × lot_size))`, then the existing deploy
  ceiling (`max_premium_deploy_pct`).

**Account-size reality check.** At ₹500,000 equity, 0.5% is ₹2,500. One
NIFTY ATM call at ₹150 × 65 units = ₹9,750 premium; a stop costing ~30% of premium is about
₹2,900 a lot, so **most naked-premium signals will size to zero lots**. The
engine reports this as `Risk: budget too small` (it already does). The
spreads exist partly for this reason. Raising equity or the tier is a
user decision, never an automatic fallback.

### 7.3 Hard limits (all checked pre-order, all journaled when they block)

| Limit | Value (v1) |
|---|---|
| Daily loss (realised + MTM of open positions) | 2% equity → halt new entries for the day |
| Consecutive losing trades | 3 → halt new entries for the day |
| Concurrent OB positions | intraday ≤ 2, overnight ≤ 1; total ≤ `max_open_setups` across all engines |
| Directional exposure | ≤ `max_direction_exposure_risk` (1.5%) summed over all open risk |
| Per expiry | ≤ 1% equity stress-loss on any single expiry |
| Entries per session | ≤ 3 |
| Zone reuse | 1 attempt per zone, ever |
| Averaging / adding | forbidden; no order may increase an open position |
| Expiry | no overnight hold into expiry day; intraday flat by 15:15, expiry day by 15:00 |
| Events | event list blocks overnight entries the evening before |
| Stale data | §4.2 gate. Also auto-flatten intraday paper positions if data is stale > 3 min while open |
| Kill switch | `~/.config/nifty-strats/KILL` exists or `OB_KILL=1` → no orders; open paper positions flattened at next quote |

The risk governor is a separate object from the engine. The engine
proposes and the governor decides. Its block reason is copied verbatim
into the signal.

---

## 8. Kite integration

### 8.1 Data flow

```
             ┌───────────── KiteWS (exists, 1 connection) ─────────────┐
             │ NIFTY 50 (quote) · FUT1 (full) · ATM±10 CE/PE × 2 expiries (full) │
             └───────────────┬───────────────────────────┬─────────────┘
                             │ ticks                     │ depth
                  TickAggregator (exists)          LegRecorder (NEW)
                   1m settled bars                 · re-centres ladder when
                             │                       spot moves > 2 strikes
                ob_series_1m / kite_candles_1m     · 1/min option_quotes rows
                             │                     · option_candles_1m
                   resample 5m/15m/60m                       │
                             └──────► ObEngine (same code as backtest)
                                         │ events
                                    RiskGovernor ──► PaperBroker ──► ob_* tables
                                         │                │
                                     Laya (shadow)     ob_events (append-only)
```

Token budget: 1 + 1 + 21 strikes × 2 sides × 2 expiries = 86 tokens.
That is far under the per-connection limit, so it fits on the existing
single connection alongside `kite-live`'s 51.

### 8.2 PaperBroker interface (kiteconnect-shaped)

```python
class Broker(Protocol):
    def place_order(self, *, variety: str, exchange: str, tradingsymbol: str,
                    transaction_type: str, quantity: int, product: str,
                    order_type: str, price: float | None = None,
                    trigger_price: float | None = None, tag: str | None = None) -> str: ...
    def cancel_order(self, variety: str, order_id: str) -> str: ...
    def orders(self) -> list[dict]: ...
    def positions(self) -> dict: ...
```

`PaperBroker` implements this against `option_quotes` (live) or historical
books (backtest). The signature mirrors `KiteConnect.place_order` so a
future live adapter is mechanical. **No live adapter is built in this
plan.** `execution/` contains no import of `kiteconnect`, and a test
asserts that.

Fill model (`PaperBroker._fill`):

1. MARKET buy: book age ≤ 2 s → fill at ask for `min(qty, ask_qty)`, walk
   remaining depth levels. Book older or depth insufficient → fill at the
   worst level touched + 1 tick, and mark `fill_model='walk_depth'`.
2. LIMIT buy: fills when a later book has `ask ≤ price`.
3. SL/SL-M exits are simulated by the engine from the **underlying** stop,
   then sent as MARKET sells. This mirrors how you'd run it live, since
   option SL orders on illiquid strikes slip badly.
4. Gap exit: first book with `exchange_ts ≥ 09:15:00`.
5. Freeze-quantity slicing: orders above the exchange freeze limit are
   split. The limit is in config with an as-of date.

### 8.3 Operational rules

- **Idempotency**: `zone_id`, `signal_id` and order `tag` are
  deterministic hashes. A replay after a WS reconnect or a process restart
  hits `UNIQUE` and is a no-op. Logged as a `dup` event, not an error.
- **Restart recovery**: on start, read `ob_paper_positions WHERE
  status='OPEN'` and the event tail, rebuild zone state by replaying the
  day's completed bars from storage through `engine.py`, then resume. Bars
  missed while down are backfilled from REST before live ticks are
  processed.
- **Session expiry**: the Kite session dies at 6 AM IST. The overnight
  exit runs at 09:15, so `ob-paper` checks `session_valid()` at startup
  and refuses to run (loudly) without a fresh login. This is the existing
  `KiteAuthError` path.
- **Persistence order**: bar → store → engine. A bar is never processed
  that is not already on disk.
- **Audit**: every signal row carries `params_hash`, `engine_version`,
  `data_age_sec`, `vix`, gate trail and Laya output. Rebuilding a decision
  from storage must reproduce it bit-for-bit, and a test does exactly that.
- Secrets stay as today: env/`.env` only, never logged.

### 8.4 Cron (IST, trading days)

| Time | Job |
|---|---|
| 08:00 | `kite-login` reminder / check; `kite-master` (also writes `kite_instrument_history`) |
| 08:30 | `premarket --exit` for any open overnight OB position (advisory, journaled) |
| 09:10 | `ob-paper` start (backfills, recovers, subscribes) |
| 09:15–09:20 | open snapshot of held legs (`reason='open_snapshot'`) |
| 15:25 | `archive-chain --expiries 2` (exists) |
| 15:35 | `ob-paper` stops; EOD flush; daily summary row |
| 16:00 | `ob-backfill --days 3` (fills any WS gaps from REST) |

---

## 9. Build sequence and acceptance criteria

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | `config.lot_size` audit + `kite_instrument_history` + `ob-audit` | Audit doc committed; lot size read from master in a test |
| 1 | `ob-backfill`, `LegRecorder`, `option_quotes` | 5 sessions recorded; spread/age stats in the audit doc. **Start this first: every day it doesn't run is option data lost** |
| 2 | `swings` / `structure` / `detect` / `lifecycle` | Property tests: (a) appending future bars never changes past zones/swings; (b) no zone is eligible before `bos_bar_ts` close; (c) hand-labelled fixture of 20 sessions matches |
| 3 | `score` + gates + `contract` + card | Every signal has a full gate trail; golden-file test of one GO and one NO-GO |
| 4 | Underlying backtest + harness + baselines | `ob-backtest` reproduces; report shows all of §6.6 and the §6.5 verdict |
| 5 | `PaperBroker` + `ob-paper` live loop + recovery | Kill-process-mid-trade test recovers with no duplicate orders; no `kiteconnect` import in `execution/` |
| 6 | Option-layer backtest on archived data + TUI tab | Synthetic vs archived side by side; promotion verdict per horizon |

Live order submission is out of scope. Promotion to real money is a
separate decision that needs: §6.5 passing on archived option data, ≥ 3
months of paper results within the backtest's CI, and a written
operational review.
