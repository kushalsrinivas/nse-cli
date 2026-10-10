-- Run-scoped identity (Phase 6).
-- signal_id is deterministic (same bars + strategy version → same id), which is
-- what makes runs comparable. It must therefore be unique per RUN, not
-- globally: a second backtest over the same dates produces the same ids.
-- signals / risk_decisions / positions are rebuilt keyed on (run_id, signal_id);
-- signal_status_history gains run_id.

CREATE TABLE signals_v3 (
  signal_id          TEXT NOT NULL,
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
  created_at         TEXT NOT NULL,
  PRIMARY KEY (run_id, signal_id)
);
INSERT INTO signals_v3 SELECT * FROM signals;
DROP TABLE signals;
ALTER TABLE signals_v3 RENAME TO signals;
CREATE INDEX signals_board ON signals(pipeline, status, detected_at);
CREATE INDEX signals_instr ON signals(instrument_key, detected_at);
CREATE INDEX signals_run ON signals(run_id, detected_at);

CREATE TABLE risk_decisions_v3 (
  decision_id          TEXT PRIMARY KEY,
  signal_id            TEXT NOT NULL,
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
  limits_json          TEXT NOT NULL,
  UNIQUE (run_id, signal_id)
);
INSERT INTO risk_decisions_v3 SELECT * FROM risk_decisions;
DROP TABLE risk_decisions;
ALTER TABLE risk_decisions_v3 RENAME TO risk_decisions;

CREATE TABLE positions_v3 (
  position_id    TEXT PRIMARY KEY,
  run_id         TEXT NOT NULL,
  signal_id      TEXT NOT NULL,
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
  gap_pnl        REAL,
  UNIQUE (run_id, signal_id)
);
INSERT INTO positions_v3 SELECT * FROM positions;
DROP TABLE positions;
ALTER TABLE positions_v3 RENAME TO positions;
CREATE INDEX positions_open ON positions(run_id, status);

ALTER TABLE signal_status_history ADD COLUMN run_id TEXT NOT NULL DEFAULT '';
CREATE INDEX ssh_run ON signal_status_history(run_id, signal_id);
