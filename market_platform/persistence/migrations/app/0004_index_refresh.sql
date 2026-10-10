-- Per-index constituent refresh state (validation, freshness). A live session
-- refuses to start when a configured index is rejected, missing or older than
-- universe.max_membership_age_days (see universe/validate.py, service.readiness).
CREATE TABLE index_refresh (
  index_id    TEXT PRIMARY KEY,
  last_ok     TEXT,              -- as_of of the last validated, applied file
  last_try    TEXT NOT NULL,
  last_status TEXT NOT NULL,     -- ok | rejected | missing
  source      TEXT DEFAULT '',
  issues_json TEXT NOT NULL DEFAULT '[]'
);
