CREATE TABLE tenants (
    tenant_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE authority_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version VARCHAR(32) NOT NULL,
    jurisdiction_id VARCHAR(64) NOT NULL,
    category VARCHAR(32) NOT NULL,
    unit_id VARCHAR(64) NOT NULL,
    effective_from TIMESTAMPTZ NOT NULL,
    effective_to TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_authority_entry UNIQUE (version, jurisdiction_id, category, unit_id),
    CONSTRAINT ck_authority_effective_range CHECK (
        effective_to IS NULL OR effective_to > effective_from
    )
);

CREATE INDEX ix_authority_lookup
    ON authority_entries (jurisdiction_id, category, version);

CREATE TABLE cases (
    case_id VARCHAR(128) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    conversation_id VARCHAR(128) NOT NULL,
    revision BIGINT NOT NULL DEFAULT 1 CHECK (revision >= 1),
    processing_state VARCHAR(32) NOT NULL CHECK (
        processing_state IN (
            'ASSEMBLING', 'READY', 'ANALYZING', 'WAITING_RESULTS',
            'WAITING_CLARIFICATION', 'EXECUTING', 'TICKETED', 'REJECTED',
            'WAITING_DEPENDENCY', 'BLOCKED', 'UNRESOLVED'
        )
    ),
    category VARCHAR(32),
    risk VARCHAR(32),
    sensitivity VARCHAR(32) NOT NULL DEFAULT 'NORMAL' CHECK (
        sensitivity IN ('NORMAL', 'RESTRICTED')
    ),
    ticket_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX ix_cases_conversation ON cases (tenant_id, conversation_id);

CREATE TABLE raw_messages (
    message_id VARCHAR(128) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    conversation_id VARCHAR(128) NOT NULL,
    source_message_id VARCHAR(128) NOT NULL,
    text TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND 64000),
    received_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_raw_messages_source UNIQUE (tenant_id, source_message_id)
);

CREATE INDEX ix_raw_messages_conversation
    ON raw_messages (tenant_id, conversation_id, received_at);

CREATE TABLE case_snapshots (
    case_id VARCHAR(128) NOT NULL REFERENCES cases(case_id) ON DELETE RESTRICT,
    revision BIGINT NOT NULL CHECK (revision >= 1),
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    conversation_id VARCHAR(128) NOT NULL,
    state VARCHAR(32) NOT NULL,
    evidence_hash CHAR(64) NOT NULL,
    messages_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (case_id, revision)
);

CREATE TABLE decisions (
    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id VARCHAR(128) NOT NULL REFERENCES cases(case_id),
    input_revision BIGINT NOT NULL CHECK (input_revision >= 1),
    sequence INTEGER NOT NULL DEFAULT 1 CHECK (sequence >= 1),
    mode VARCHAR(32) NOT NULL CHECK (
        mode IN ('EXECUTE', 'RE_EVALUATE', 'REQUEST_CLARIFICATION', 'REJECT_IGNORE')
    ),
    reason_codes TEXT[] NOT NULL DEFAULT '{}',
    policy_result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_decision_sequence UNIQUE (case_id, input_revision, sequence)
);

CREATE TABLE ticket_commands (
    command_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    case_id VARCHAR(128) NOT NULL REFERENCES cases(case_id),
    decision_id UUID NOT NULL REFERENCES decisions(decision_id),
    idempotency_key VARCHAR(255) NOT NULL,
    payload_hash CHAR(64) NOT NULL,
    request_payload JSONB NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'DISPATCHED', 'CONFIRMED', 'FAILED')
    ),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_ticket_command_idempotency UNIQUE (tenant_id, idempotency_key),
    CONSTRAINT ck_ticket_idempotency_format CHECK (
        idempotency_key = tenant_id || ':' || case_id || ':ticket:create:v1'
    )
);

CREATE TABLE inbox (
    consumer_name VARCHAR(128) NOT NULL,
    event_id UUID NOT NULL,
    aggregate_id VARCHAR(128) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (consumer_name, event_id)
);

CREATE TABLE outbox (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    aggregate_type VARCHAR(64) NOT NULL,
    aggregate_id VARCHAR(128) NOT NULL,
    revision BIGINT NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    partition_key VARCHAR(128) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    published_at TIMESTAMPTZ
);

CREATE INDEX ix_outbox_unpublished ON outbox (created_at) WHERE published_at IS NULL;

CREATE FUNCTION enforce_append_only()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'append-only table % rejects %', TG_TABLE_NAME, TG_OP;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER raw_messages_immutable
    BEFORE UPDATE OR DELETE ON raw_messages
    FOR EACH ROW EXECUTE FUNCTION enforce_append_only();

CREATE TRIGGER case_snapshots_immutable
    BEFORE UPDATE OR DELETE ON case_snapshots
    FOR EACH ROW EXECUTE FUNCTION enforce_append_only();

CREATE TRIGGER decisions_immutable
    BEFORE UPDATE OR DELETE ON decisions
    FOR EACH ROW EXECUTE FUNCTION enforce_append_only();

CREATE TRIGGER inbox_immutable
    BEFORE UPDATE OR DELETE ON inbox
    FOR EACH ROW EXECUTE FUNCTION enforce_append_only();

INSERT INTO tenants (tenant_id, name) VALUES ('research', 'KAWAL Research');

INSERT INTO authority_entries (
    version, jurisdiction_id, category, unit_id, effective_from
) VALUES
    ('m1.v1', 'JUR-FICT-01', 'ROAD', 'UNIT-BINA-MARGA-01', '2026-01-01T00:00:00Z'),
    ('m1.v1', 'JUR-FICT-02', 'ROAD', 'UNIT-BINA-MARGA-02', '2026-01-01T00:00:00Z'),
    ('m1.v1', 'JUR-FICT-03', 'ROAD', 'UNIT-BINA-MARGA-03', '2026-01-01T00:00:00Z');
