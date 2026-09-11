ALTER TABLE raw_messages
    ADD COLUMN quoted_source_message_id VARCHAR(128),
    ADD COLUMN case_key VARCHAR(128),
    ADD COLUMN connector_id VARCHAR(64) NOT NULL DEFAULT 'replay',
    ADD COLUMN account_id VARCHAR(128) NOT NULL DEFAULT 'research';

ALTER TABLE raw_messages
    ADD CONSTRAINT uq_raw_messages_source_identity
    UNIQUE (tenant_id, connector_id, account_id, source_message_id);

CREATE INDEX ix_raw_messages_conversation_received
    ON raw_messages (tenant_id, connector_id, account_id, conversation_id, received_at);

CREATE TABLE attachments (
    attachment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id VARCHAR(128) NOT NULL REFERENCES raw_messages(message_id),
    object_key VARCHAR(512) NOT NULL,
    mime_type VARCHAR(128) NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes > 0 AND size_bytes <= 10485760),
    content_hash CHAR(64) NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_attachment_object UNIQUE (object_key)
);

CREATE INDEX ix_attachments_message ON attachments (message_id);

CREATE TABLE message_case_links (
    message_id VARCHAR(128) PRIMARY KEY REFERENCES raw_messages(message_id),
    case_id VARCHAR(128) NOT NULL REFERENCES cases(case_id),
    linked_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX ix_message_case_links_case ON message_case_links (case_id);

CREATE TABLE assembly_timers (
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    connector_id VARCHAR(64) NOT NULL,
    account_id VARCHAR(128) NOT NULL,
    conversation_id VARCHAR(128) NOT NULL,
    due_at TIMESTAMPTZ NOT NULL,
    hard_due_at TIMESTAMPTZ NOT NULL,
    generation BIGINT NOT NULL DEFAULT 1 CHECK (generation >= 1),
    fired_at TIMESTAMPTZ,
    PRIMARY KEY (tenant_id, connector_id, account_id, conversation_id)
);

CREATE INDEX ix_assembly_timers_due
    ON assembly_timers (due_at) WHERE fired_at IS NULL;

ALTER TABLE attachments ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_case_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE assembly_timers ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON attachments, message_case_links, assembly_timers FROM anon, authenticated;
