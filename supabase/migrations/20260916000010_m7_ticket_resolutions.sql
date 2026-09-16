CREATE TABLE IF NOT EXISTS ticket_resolutions (
    ticket_id VARCHAR(128) PRIMARY KEY,
    case_id VARCHAR(128) REFERENCES cases(case_id) ON DELETE SET NULL,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'research',
    status VARCHAR(32) NOT NULL DEFAULT 'SUBMITTED' CHECK (
        status IN ('SUBMITTED', 'IN_PROGRESS', 'RESOLVED', 'CLOSED')
    ),
    authority_unit_id VARCHAR(128),
    category VARCHAR(64),
    risk VARCHAR(32),
    location TEXT,
    citizen_text TEXT,
    citizen_photo_url TEXT,
    admin_notes TEXT,
    resolution_photo_url TEXT,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS ix_ticket_resolutions_status ON ticket_resolutions (status);
CREATE INDEX IF NOT EXISTS ix_ticket_resolutions_unit ON ticket_resolutions (authority_unit_id);
CREATE INDEX IF NOT EXISTS ix_ticket_resolutions_case ON ticket_resolutions (case_id);
