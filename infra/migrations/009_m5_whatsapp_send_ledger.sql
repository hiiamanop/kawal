CREATE TABLE IF NOT EXISTS whatsapp_send_ledger (
    send_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    conversation_id VARCHAR(128) NOT NULL,
    case_id VARCHAR(128) REFERENCES cases(case_id),
    idempotency_key VARCHAR(256) NOT NULL,
    recipient_phone VARCHAR(64) NOT NULL,
    purpose VARCHAR(32) NOT NULL DEFAULT 'CLARIFICATION' CHECK (
        purpose IN ('CLARIFICATION', 'STATUS_UPDATE', 'RECEIPT')
    ),
    text TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND 4096),
    quoted_source_message_id VARCHAR(128),
    payload_hash CHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'SENT', 'DELIVERED', 'DELIVERY_UNKNOWN', 'FAILED')
    ),
    source_message_id VARCHAR(128),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_whatsapp_send_idempotency UNIQUE (idempotency_key)
);

CREATE INDEX IF NOT EXISTS ix_whatsapp_send_ledger_case ON whatsapp_send_ledger (case_id);
CREATE INDEX IF NOT EXISTS ix_whatsapp_send_ledger_status ON whatsapp_send_ledger (status) WHERE status IN ('PENDING', 'DELIVERY_UNKNOWN');
