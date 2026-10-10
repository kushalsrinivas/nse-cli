-- Run-scoped zones (V5). Zone ids are deterministic like signal ids, so the
-- same zone appears in every run over the same bars; its lifecycle (status,
-- close reason) belongs to the run that observed it.
CREATE TABLE zones_v5 (
  zone_id           TEXT NOT NULL,
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
  run_id            TEXT NOT NULL,
  PRIMARY KEY (run_id, zone_id)
);
INSERT INTO zones_v5 SELECT * FROM zones;
DROP TABLE zones;
ALTER TABLE zones_v5 RENAME TO zones;
CREATE INDEX zones_live ON zones(instrument_key, status);
CREATE INDEX zones_run ON zones(run_id, instrument_key);
