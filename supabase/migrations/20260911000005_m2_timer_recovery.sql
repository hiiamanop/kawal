ALTER TABLE assembly_timers
    ADD COLUMN lease_token UUID,
    ADD COLUMN lease_expires_at TIMESTAMPTZ,
    ADD COLUMN assembled_through TIMESTAMPTZ,
    ADD COLUMN assembled_through_message_id VARCHAR(128);

CREATE INDEX ix_assembly_timers_claimable
    ON assembly_timers (due_at, lease_expires_at)
    WHERE fired_at IS NULL;

ALTER TABLE assembly_timers
    ADD CONSTRAINT ck_assembly_timer_lease_pair CHECK (
        (lease_token IS NULL AND lease_expires_at IS NULL)
        OR (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
    );
