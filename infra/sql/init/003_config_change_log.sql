-- Config change log table for tracking gating threshold changes.
--
-- Each PUT to /config/gating-thresholds logs one row per changed field,
-- capturing who changed what, the old and new values, and when.
-- Separate from audit_log because the schemas are fundamentally different.

CREATE TABLE IF NOT EXISTS config_change_log (
    id              SERIAL PRIMARY KEY,
    changed_by_role TEXT NOT NULL,
    product         TEXT NOT NULL,
    field_name      TEXT NOT NULL,
    old_value       TEXT NOT NULL,
    new_value       TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_config_change_log_created
    ON config_change_log (created_at DESC);
