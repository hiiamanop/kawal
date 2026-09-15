from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import pytest

psycopg = pytest.importorskip("psycopg")

from contracts.models import Category, RawMessage
from infra.db import TransactionRunner
from scripts.run_all_workers import WorkerSupervisor, build_worker_commands
from services.intake.assembly import ConversationAssembler
from services.intake.service import IntakeService

DATABASE_URL = os.getenv("KAWAL_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres")
REDPANDA_BOOTSTRAP = os.getenv("KAWAL_REDPANDA_BOOTSTRAP", "127.0.0.1:19092")


@pytest.mark.integration
def test_full_concurrency_e2e_pipeline() -> None:
    # 1. Ensure test tenant
    tenant_id = f"tenant-conc-e2e-{uuid4().hex[:6]}"
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tenants (tenant_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (tenant_id, "Concurrency E2E Tenant"),
            )

    runner = TransactionRunner(DATABASE_URL)
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)

    # 2. Inject 3 distinct complaints with past received_at so assembly timers are due
    now_past = datetime.now(timezone.utc) - timedelta(seconds=15)
    complaints = [
        (
            "road",
            "halo min punten mau lapor, ini jalan terusan buah batu no 120 rt 02 rw 05 kelurahan kujangsari kecamatan bandung kidul kota bandung jalannya ancur berlubang bgt pas dket borma, kemarin ada pemotor jatuh bahaya bgt tolong perbaiki",
            Category.ROAD,
        ),
        (
            "waste",
            "min tolong bgt ini tumpukan sampah liar di jalan cibadak no 88 rt 01 rw 03 kelurahan karanganyar kecamatan astanaanyar kota bandung udh seminggu ga diangkut bau busuk laler kemana2 ganggu warga",
            Category.WASTE,
        ),
        (
            "drainage",
            "siang admin, lapor gorong-gorong mampet air meluap banjir di jalan kliningan no 15 rt 03 rw 04 kelurahan turangga kecamatan lengkong kota bandung pas ujan deres air masuk rumah warga",
            Category.DRAINAGE_FLOOD,
        ),
    ]

    for key, text, _ in complaints:
        msg = RawMessage(
            message_id=f"{tenant_id}:msg-{key}",
            tenant_id=tenant_id,
            conversation_id=f"{tenant_id}-{key}@c.us",
            source_message_id=f"src-{key}-{uuid4().hex[:6]}",
            text=text,
            received_at=now_past,
        )
        assert intake.accept(msg, connector_id="openwa", account_id="bot") is True

    # 3. Launch unified worker supervisor
    cmds = build_worker_commands(sys.executable, DATABASE_URL, REDPANDA_BOOTSTRAP)
    env = os.environ.copy()
    env["KAWAL_DATABASE_URL"] = DATABASE_URL
    env["REDPANDA_BOOTSTRAP_SERVERS"] = REDPANDA_BOOTSTRAP

    supervisor = WorkerSupervisor(cmds, env)
    supervisor.start()

    try:
        # 4. Wait for all 3 cases to be assembled, analyzed with ONNX, checked with OPA, and ticketed
        deadline = time.time() + 45.0
        success = False

        while time.time() < deadline:
            time.sleep(1.0)
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT case_id, processing_state, category, ticket_id
                        FROM cases
                        WHERE tenant_id = %s
                        """,
                        (tenant_id,),
                    )
                    rows = cur.fetchall()
                    if len(rows) == 3 and all(r[1] == "TICKETED" and r[3] is not None for r in rows):
                        success = True
                        break

        assert success is True, "All 3 concurrent cases should reach TICKETED status within deadline"

        # 5. Verify database invariants: audit traces and unique ticket IDs
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_type, payload->>'ticket_id'
                    FROM audit_traces
                    WHERE tenant_id = %s AND event_type = 'ticket.executed.v1'
                    """,
                    (tenant_id,),
                )
                executed_traces = cur.fetchall()
                assert len(executed_traces) == 3
                ticket_ids = [row[1] for row in executed_traces]
                assert len(set(ticket_ids)) == 3, "Each ticket must have a unique ID"

                cur.execute(
                    """
                    SELECT count(*) FROM cases WHERE tenant_id = %s AND processing_state = 'TICKETED'
                    """,
                    (tenant_id,),
                )
                assert cur.fetchone()[0] == 3

    finally:
        supervisor.stop(timeout_seconds=3.0)
