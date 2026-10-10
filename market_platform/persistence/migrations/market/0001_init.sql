-- market.db: high-volume market data (append-mostly). See docs/PLATFORM_PLAN.md §5.3.
-- 5m/15m/60m bars are NOT stored: they are derived from bars_1m with the
-- same BarBuilder the live path uses, so live and replay cannot diverge.

CREATE TABLE bars_1m (
  instrument_key TEXT NOT NULL,          -- 'NSE:HDFCBANK', 'NSE:NIFTY 50', 'NFO:NIFTY26OCTFUT'
  ts             TEXT NOT NULL,          -- IST minute start 'YYYY-MM-DD HH:MM'
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
  volume  INTEGER,
  oi      INTEGER,
  n_ticks INTEGER,
  source  TEXT NOT NULL CHECK(source IN ('kite_ws','kite_hist','repair','legacy')),
  PRIMARY KEY (instrument_key, ts)
) WITHOUT ROWID;

CREATE TABLE bars_1d (
  instrument_key TEXT NOT NULL,
  date           TEXT NOT NULL,
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
  volume   INTEGER,
  oi       INTEGER,
  adjusted INTEGER NOT NULL DEFAULT 0,
  source   TEXT NOT NULL,
  PRIMARY KEY (instrument_key, date)
) WITHOUT ROWID;

CREATE TABLE option_quotes (
  exchange      TEXT NOT NULL,
  tradingsymbol TEXT NOT NULL,
  captured_at   TEXT NOT NULL,
  exchange_ts   TEXT,
  underlying    TEXT DEFAULT '',
  spot REAL, ltp REAL, bid REAL, bid_qty INTEGER, ask REAL, ask_qty INTEGER,
  depth_json TEXT DEFAULT '', volume INTEGER, oi INTEGER, iv REAL,
  reason      TEXT NOT NULL DEFAULT 'periodic',
  expiry      TEXT DEFAULT '',
  strike      REAL,
  option_type TEXT DEFAULT '',
  PRIMARY KEY (exchange, tradingsymbol, captured_at)
) WITHOUT ROWID;

CREATE TABLE option_candles_1m (
  exchange TEXT NOT NULL, tradingsymbol TEXT NOT NULL, ts TEXT NOT NULL,
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
  volume INTEGER NOT NULL DEFAULT 0, oi INTEGER,
  source TEXT NOT NULL, expiry TEXT DEFAULT '', strike REAL, option_type TEXT DEFAULT '',
  underlying TEXT DEFAULT '',
  PRIMARY KEY (exchange, tradingsymbol, ts)
) WITHOUT ROWID;

CREATE TABLE quality_events (
  ts TEXT NOT NULL, instrument_key TEXT, kind TEXT NOT NULL, severity TEXT NOT NULL, detail TEXT DEFAULT ''
);
CREATE INDEX quality_recent ON quality_events(ts);
CREATE INDEX quality_instr ON quality_events(instrument_key, ts);

CREATE TABLE backfill_watermarks (
  instrument_key TEXT NOT NULL, timeframe TEXT NOT NULL, last_ts TEXT NOT NULL,
  source TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY (instrument_key, timeframe)
);
