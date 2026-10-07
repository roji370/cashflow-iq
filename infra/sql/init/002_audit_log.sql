-- Audit log table for scoring request traceability.
--
-- Every /score/{customer_id} call logs a row here with model versions,
-- gate result, and a SHA-256 hash of the input features (NOT raw values,
-- to avoid PII in the queryable audit trail).

CREATE TABLE IF NOT EXISTS audit_log (
    id                      SERIAL PRIMARY KEY,
    customer_id             TEXT NOT NULL,
    product                 TEXT NOT NULL,
    capacity_model_version  TEXT NOT NULL,
    intent_model_version    TEXT NOT NULL,
    gate_result             TEXT NOT NULL,  -- 'pass', 'fail', or 'manual_review'
    requesting_role         TEXT NOT NULL,
    input_feature_hash      TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for the admin audit trail endpoint (GET /audit/{customer_id})
CREATE INDEX IF NOT EXISTS idx_audit_log_customer_id
    ON audit_log (customer_id, created_at DESC);
