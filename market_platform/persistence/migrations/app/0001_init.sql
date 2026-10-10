-- app.db: platform state (low volume, consistency-critical). See docs/PLATFORM_PLAN.md §5.2.

CREATE TABLE config_versions (
  config_hash TEXT PRIMARY KEY,
  body_json   TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  note        TEXT DEFAULT ''
);

CREATE TABLE runs (
  run_id            TEXT PRIMARY KEY,
  kind              TEXT NOT NULL CHECK(kind IN ('paper','backtest','shadow','import','tool')),
  config_hash       TEXT NOT NULL REFERENCES config_versions(config_hash),
  strategy_version  TEXT NOT NULL,
  universe_snapshot TEXT NOT NULL,
  data_version      TEXT NOT NULL,
  started_at        TEXT NOT NULL,
  ended_at          TEXT,
  status            TEXT NOT NULL CHECK(status IN ('running','completed','failed','aborted')),
  notes             TEXT DEFAULT ''
);

-- universe -------------------------------------------------------------------
CREATE TABLE indices (
  index_id         TEXT PRIMARY KEY,
  name             TEXT NOT NULL,
  exchange         TEXT NOT NULL,
  category         TEXT NOT NULL CHECK(category IN ('broad','sector','thematic','strategy')),
  sector           TEXT DEFAULT '',
  kite_symbol      TEXT DEFAULT '',
  kite_token       INTEGER,
  deriv_underlying TEXT,
  constituents_url TEXT DEFAULT '',
  source           TEXT NOT NULL,
  updated_at       TEXT NOT NULL
);

CREATE TABLE universe_snapshots (
  snapshot_id   TEXT PRIMARY KEY,
  as_of         TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  sources_json  TEXT NOT NULL,
  n_indices     INTEGER NOT NULL,
  n_companies   INTEGER NOT NULL,
  n_instruments INTEGER NOT NULL,
  sha256        TEXT NOT NULL,
  survivorship  TEXT NOT NULL DEFAULT 'snapshot'   -- 'snapshot' | 'imported_history'
);

CREATE TABLE companies (
  isin              TEXT PRIMARY KEY,
  symbol            TEXT NOT NULL,
  name              TEXT DEFAULT '',
  industry          TEXT DEFAULT '',
  sector            TEXT DEFAULT '',
  updated_at        TEXT NOT NULL
);
CREATE INDEX companies_symbol ON companies(symbol);

CREATE TABLE symbol_history (
  isin TEXT NOT NULL, symbol TEXT NOT NULL, valid_from TEXT NOT NULL, valid_to TEXT,
  PRIMARY KEY (isin, valid_from)
);

CREATE TABLE index_membership (
  index_id       TEXT NOT NULL,
  isin           TEXT NOT NULL,
  symbol         TEXT NOT NULL,
  weight         REAL,
  valid_from     TEXT NOT NULL,
  valid_to       TEXT,
  source         TEXT NOT NULL,
  first_snapshot TEXT NOT NULL,
  PRIMARY KEY (index_id, isin, valid_from)
);
CREATE INDEX membership_current ON index_membership(index_id, valid_to);
CREATE INDEX membership_isin ON index_membership(isin);

CREATE TABLE instrument_eligibility (
  snapshot_id       TEXT NOT NULL,
  instrument_key    TEXT NOT NULL,
  isin              TEXT,
  symbol            TEXT NOT NULL,
  exchange          TEXT NOT NULL,
  kind              TEXT NOT NULL CHECK(kind IN ('equity','index')),
  token             INTEGER,
  sector            TEXT DEFAULT '',
  industry          TEXT DEFAULT '',
  indices_json      TEXT NOT NULL DEFAULT '[]',
  lot_size          INTEGER,
  tick_size         REAL,
  fno_eligible      INTEGER NOT NULL,
  weekly_options    INTEGER NOT NULL,
  deriv_underlying  TEXT,
  liquidity_tier    TEXT DEFAULT 'unknown',
  adv_value_cr      REAL,
  median_spread_bps REAL,
  data_status       TEXT NOT NULL,
  reasons           TEXT NOT NULL DEFAULT '[]',
  PRIMARY KEY (snapshot_id, instrument_key)
);

CREATE TABLE trading_calendar (
  exchange   TEXT NOT NULL,
  date       TEXT NOT NULL,
  is_trading INTEGER NOT NULL,
  open_time  TEXT,
  close_time TEXT,
  note       TEXT DEFAULT '',
  source     TEXT NOT NULL,
  PRIMARY KEY (exchange, date)
);

CREATE TABLE corporate_actions (
  isin    TEXT NOT NULL,
  symbol  TEXT NOT NULL,
  ex_date TEXT NOT NULL,
  kind    TEXT NOT NULL CHECK(kind IN ('split','bonus','dividend','symbol_change','merger','demerger','rights','other')),
  ratio   REAL,
  detail  TEXT DEFAULT '',
  source  TEXT NOT NULL,
  PRIMARY KEY (isin, ex_date, kind)
);

-- structure & signals -----------------------------------------------------------
CREATE TABLE zones (
  zone_id           TEXT PRIMARY KEY,
  instrument_key    TEXT NOT NULL,
  timeframe         TEXT NOT NULL,
  direction         TEXT NOT NULL CHECK(direction IN ('bullish','bearish')),
  kind              TEXT NOT NULL,
  source_bar_ts     TEXT NOT NULL,
  bos_bar_ts        TEXT NOT NULL,
  first_eligible_ts TEXT NOT NULL,
  zone_low          REAL NOT NULL,
  zone_high         REAL NOT NULL,
  features_json     TEXT NOT NULL,
  status            TEXT NOT NULL,
  status_ts         TEXT,
  close_reason      TEXT DEFAULT '',
  params_hash       TEXT NOT NULL,
  run_id            TEXT NOT NULL
);
CREATE INDEX zones_live ON zones(instrument_key, status);

CREATE TABLE signals (
  signal_id          TEXT PRIMARY KEY,
  run_id             TEXT NOT NULL,
  pipeline           TEXT NOT NULL CHECK(pipeline IN ('bullish','bearish')),
  instrument_key     TEXT NOT NULL,
  underlying         TEXT NOT NULL,
  direction          TEXT NOT NULL CHECK(direction IN ('bullish','bearish')),
  strategy           TEXT NOT NULL,
  setup_type         TEXT NOT NULL,
  timeframe          TEXT NOT NULL,
  horizon            TEXT NOT NULL CHECK(horizon IN ('intraday','overnight')),
  zone_id            TEXT NOT NULL,
  zone_low           REAL,
  zone_high          REAL,
  detected_at        TEXT NOT NULL,
  available_at       TEXT NOT NULL,
  session            TEXT NOT NULL,
  entry              REAL NOT NULL,
  invalidation       REAL NOT NULL,
  stop               REAL NOT NULL,
  targets_json       TEXT NOT NULL,
  rr                 REAL NOT NULL,
  score              REAL,
  score_json         TEXT NOT NULL DEFAULT '{}',
  p_calibrated       REAL,
  confirmations_json TEXT NOT NULL DEFAULT '[]',
  context_json       TEXT NOT NULL DEFAULT '{}',
  data_quality       TEXT NOT NULL,
  liquidity_json     TEXT NOT NULL DEFAULT '{}',
  est_costs          REAL,
  proposal_json      TEXT,
  cluster_id         TEXT,
  status             TEXT NOT NULL CHECK(status IN ('QUALIFIED','WATCH','REJECTED','SUPPRESSED',
                                                    'APPROVED','EXECUTED','EXPIRED','NOT_EXECUTABLE')),
  qualify_reasons    TEXT NOT NULL DEFAULT '[]',
  reject_reasons     TEXT NOT NULL DEFAULT '[]',
  config_hash        TEXT NOT NULL,
  universe_snapshot  TEXT NOT NULL,
  created_at         TEXT NOT NULL
);
CREATE INDEX signals_board ON signals(pipeline, status, detected_at);
CREATE INDEX signals_instr ON signals(instrument_key, detected_at);

CREATE TABLE signal_status_history (
  signal_id TEXT NOT NULL, ts TEXT NOT NULL, status TEXT NOT NULL, reason TEXT DEFAULT ''
);
CREATE INDEX ssh_signal ON signal_status_history(signal_id);

-- risk, execution, portfolio --------------------------------------------------------
CREATE TABLE risk_decisions (
  decision_id          TEXT PRIMARY KEY,
  signal_id            TEXT NOT NULL UNIQUE,
  run_id               TEXT NOT NULL,
  ts                   TEXT NOT NULL,
  approved             INTEGER NOT NULL,
  reason_codes         TEXT NOT NULL,
  instrument_key       TEXT,
  product              TEXT,
  route                TEXT,
  lots                 INTEGER,
  lot_size             INTEGER,
  quantity             INTEGER,
  risk_rupees          REAL,
  stress_json          TEXT,
  exposure_before_json TEXT NOT NULL,
  limits_json          TEXT NOT NULL
);

CREATE TABLE orders (
  order_id         TEXT PRIMARY KEY,
  run_id           TEXT NOT NULL,
  signal_id        TEXT NOT NULL,
  tag              TEXT NOT NULL,
  leg_index        INTEGER NOT NULL,
  attempt          INTEGER NOT NULL DEFAULT 0,
  instrument_key   TEXT NOT NULL,
  segment          TEXT NOT NULL CHECK(segment IN ('equity','futures','options')),
  transaction_type TEXT NOT NULL CHECK(transaction_type IN ('BUY','SELL')),
  product          TEXT NOT NULL CHECK(product IN ('MIS','CNC','NRML')),
  order_type       TEXT NOT NULL CHECK(order_type IN ('MARKET','LIMIT','SL','SL-M')),
  quantity         INTEGER NOT NULL,
  price            REAL,
  trigger_price    REAL,
  purpose          TEXT NOT NULL,
  status           TEXT NOT NULL CHECK(status IN ('OPEN','COMPLETE','CANCELLED','REJECTED')),
  filled_qty       INTEGER NOT NULL DEFAULT 0,
  avg_price        REAL,
  status_message   TEXT DEFAULT '',
  placed_at        TEXT NOT NULL,
  updated_at       TEXT NOT NULL,
  UNIQUE (run_id, tag, purpose, leg_index, attempt)
);
CREATE INDEX orders_signal ON orders(signal_id);

CREATE TABLE fills (
  fill_id      TEXT PRIMARY KEY,
  order_id     TEXT NOT NULL,
  filled_at    TEXT NOT NULL,
  qty          INTEGER NOT NULL,
  price        REAL NOT NULL,
  book_bid     REAL,
  book_ask     REAL,
  book_age_sec REAL,
  fill_model   TEXT NOT NULL,
  charges      REAL NOT NULL
);
CREATE INDEX fills_order ON fills(order_id);

CREATE TABLE positions (
  position_id    TEXT PRIMARY KEY,
  run_id         TEXT NOT NULL,
  signal_id      TEXT NOT NULL UNIQUE,
  instrument_key TEXT NOT NULL,
  underlying     TEXT NOT NULL,
  sector         TEXT DEFAULT '',
  cluster_id     TEXT,
  direction      TEXT NOT NULL,
  horizon        TEXT NOT NULL,
  segment        TEXT NOT NULL,
  product        TEXT NOT NULL,
  structure      TEXT NOT NULL,
  legs_json      TEXT NOT NULL,
  quantity       INTEGER NOT NULL,
  lot_size       INTEGER NOT NULL,
  entry_net      REAL NOT NULL,
  u_entry        REAL,
  u_stop         REAL NOT NULL,
  u_target       REAL NOT NULL,
  risk_rupees    REAL NOT NULL,
  opened_at      TEXT NOT NULL,
  status         TEXT NOT NULL CHECK(status IN ('OPEN','CLOSED')),
  closed_at      TEXT,
  exit_net       REAL,
  exit_reason    TEXT,
  gross_pnl      REAL,
  charges        REAL,
  net_pnl        REAL,
  r_multiple     REAL,
  mae_rupees     REAL,
  mfe_rupees     REAL,
  gap_pnl        REAL
);
CREATE INDEX positions_open ON positions(run_id, status);

CREATE TABLE portfolio_snapshots (
  run_id TEXT NOT NULL, ts TEXT NOT NULL, body_json TEXT NOT NULL, PRIMARY KEY (run_id, ts)
);

-- audit, recovery, health -----------------------------------------------------------------
CREATE TABLE events (
  seq          INTEGER PRIMARY KEY AUTOINCREMENT,
  ts           TEXT NOT NULL,
  run_id       TEXT NOT NULL,
  kind         TEXT NOT NULL,
  key          TEXT,
  payload_json TEXT NOT NULL
);
CREATE INDEX events_run ON events(run_id, seq);

CREATE TABLE checkpoints (
  consumer TEXT PRIMARY KEY, position TEXT NOT NULL, ts TEXT NOT NULL
);

CREATE TABLE health_events (
  ts TEXT NOT NULL, component TEXT NOT NULL, state TEXT NOT NULL,
  metric TEXT DEFAULT '', value REAL, detail TEXT DEFAULT ''
);
CREATE INDEX health_recent ON health_events(component, ts);
