import os
from pathlib import Path
from uuid import uuid5

import pytest

psycopg = pytest.importorskip("psycopg")

from infra.db import TransactionRunner
from services.core.persistence import M1Store
from services.intake.replay import (
    EVENT_NAMESPACE,
    build_replay_case,
    load_fixture,
    replay_to_durable_ticket,
)
from services.outbox.relay import OutboxRelay
from services.core.orchestrator import build_ticket_command
from services.core.policy import evaluate_ticket_creation
from contracts.models import PolicyInput
from services.simulator.store import TicketSimulator


DATABASE_URL = os.getenv("KAWAL_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set KAWAL_TEST_DATABASE_URL to run PostgreSQL integration tests",
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "three_bubble_road_complaint.json"
MIGRATIONS = tuple(sorted((Path(__file__).parents[1] / "supabase" / "migrations").glob("*.sql")))


@pytest.fixture
def runner() -> TransactionRunner:
    assert DATABASE_URL is not None
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DROP SCHEMA public CASCADE")
            cursor.execute("CREATE SCHEMA public")
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
            for migration in MIGRATIONS:
                cursor.execute(migration.read_text(encoding="utf-8"))
    return TransactionRunner(DATABASE_URL)


def policy_input(fixture) -> PolicyInput:
    _, snapshot, analysis = build_replay_case(fixture)
    return PolicyInput(
        tenant_id=snapshot.tenant_id,
        case_id=snapshot.case_id,
        revision=snapshot.revision,
        category=analysis.category,
        jurisdiction_id=analysis.jurisdiction_id,
        authority_unit_id=analysis.authority_unit_id,
        missing_fields=analysis.missing_fields,
        has_mandatory_evidence=analysis.has_mandatory_evidence,
        idempotency_key=f"{snapshot.tenant_id}:{snapshot.case_id}:ticket:create:v1",
    )


def test_durable_replay_creates_one_ticket_and_persists_the_vertical_slice(
    runner: TransactionRunner,
) -> None:
    simulator = TicketSimulator()
    relay = OutboxRelay(runner, M1Store(), simulator)
    fixture = load_fixture(FIXTURE_PATH)

    first = replay_to_durable_ticket(fixture, runner, M1Store(), relay)
    second = replay_to_durable_ticket(fixture, runner, M1Store(), relay)

    assert second.ticket_id == first.ticket_id
    assert simulator.ticket_count == 1
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            for table, expected in (
                ("raw_messages", 3),
                ("cases", 1),
                ("case_snapshots", 1),
                ("decisions", 1),
                ("ticket_commands", 1),
                ("inbox", 1),
                ("audit_traces", 1),
            ):
                cursor.execute(f"SELECT count(*) FROM {table}")
                assert cursor.fetchone()[0] == expected
            cursor.execute("SELECT count(*) FROM outbox WHERE published_at IS NOT NULL")
            assert cursor.fetchone()[0] == 1


def test_committed_event_survives_a_crash_before_relay_dispatch(runner: TransactionRunner) -> None:
    store = M1Store()
    fixture = load_fixture(FIXTURE_PATH)
    messages, snapshot, analysis = build_replay_case(fixture)
    command = build_ticket_command(snapshot, analysis)
    assert command is not None
    policy = evaluate_ticket_creation(policy_input(fixture))

    with pytest.raises(RuntimeError, match="simulated crash"):
        runner.run(
            lambda connection: store.persist_case(
                connection,
                event_id=uuid5(EVENT_NAMESPACE, f"{fixture.tenant_id}:{fixture.scenario_id}"),
                snapshot=snapshot,
                messages=messages,
                command=command,
                policy=policy,
            ),
            after_commit=lambda: (_ for _ in ()).throw(RuntimeError("simulated crash")),
        )

    simulator = TicketSimulator()
    recovered_relay = OutboxRelay(TransactionRunner(DATABASE_URL), M1Store(), simulator)
    receipt = recovered_relay.dispatch_next()

    assert receipt is not None
    assert receipt.status == "SUBMITTED"
    assert simulator.ticket_count == 1
