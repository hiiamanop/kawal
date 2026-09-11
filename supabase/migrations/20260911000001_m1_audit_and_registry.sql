ALTER TABLE outbox
    ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    ADD COLUMN lease_expires_at TIMESTAMPTZ;

CREATE INDEX ix_outbox_dispatchable
    ON outbox (created_at)
    WHERE published_at IS NULL;

CREATE TABLE audit_traces (
    trace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    case_id VARCHAR(128) NOT NULL REFERENCES cases(case_id),
    decision_id UUID REFERENCES decisions(decision_id),
    event_type VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_audit_trace_event UNIQUE (decision_id, event_type)
);

CREATE INDEX ix_audit_traces_case ON audit_traces (case_id, created_at);

CREATE TABLE policy_registry (
    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(128) NOT NULL,
    bundle_version VARCHAR(64) NOT NULL,
    artifact_hash CHAR(64) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_policy_registry_version UNIQUE (name, bundle_version)
);

CREATE UNIQUE INDEX uq_policy_registry_active
    ON policy_registry (name) WHERE is_active;

CREATE TABLE model_registry (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(128) NOT NULL,
    version VARCHAR(64) NOT NULL,
    artifact_hash CHAR(64) NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_model_registry_version UNIQUE (name, version)
);

INSERT INTO policy_registry (name, bundle_version, artifact_hash, is_active)
VALUES (
    'kawal.ticket-creation',
    'm1.v1',
    'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
    TRUE
);
