-- Market context snapshots (Phase 4). One row per recompute; the signal
-- rows reference `snapshot_id` inside context_json, so every signal can
-- show exactly which market/sector state it was scored against.
CREATE TABLE context_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  ts          TEXT NOT NULL,
  run_id      TEXT NOT NULL,
  regime      TEXT NOT NULL,
  regime_conf REAL NOT NULL,
  vol_regime  TEXT NOT NULL,
  quality     TEXT NOT NULL,
  body_json   TEXT NOT NULL
);
CREATE INDEX context_recent ON context_snapshots(run_id, ts);

-- Correlation clusters of instruments (recomputed weekly from daily returns).
CREATE TABLE correlation_clusters (
  as_of TEXT NOT NULL, instrument_key TEXT NOT NULL, cluster TEXT NOT NULL, beta REAL,
  PRIMARY KEY (as_of, instrument_key)
);
