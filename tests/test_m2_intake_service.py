import pytest
from datetime import datetime, timezone

from contracts.models import RawMessage
from services.intake.openwa import OpenWAConnector, build_normalized_message_id
from services.intake.service import IntakeService, TARGET_PERSIST_LATENCY_MS


class FakeAssembler:
    def __init__(self) -> None:
        self.calls = 0

    def ingest(self, *args: object) -> bool:
        self.calls += 1
        return True


class FakeRunner:
    def __init__(self) -> None:
        self.committed = False

    def run(self, operation: object) -> bool:
        result = operation(object())
        self.committed = True
        return result


class RecordingIntake:
    def __init__(self) -> None:
        self.persisted = False
        self.assembly_calls = 0
        self.calls: list[tuple[object, ...]] = []
        self.seen_persisted_when_returning: list[bool] = []
        self.message_ids: set[str] = set()

    def accept(
        self,
        message: RawMessage,
        quoted_source_message_id: str | None = None,
        case_key: str | None = None,
        connector_id: str = "replay",
        account_id: str = "research",
    ) -> bool:
        self.calls.append(
            (message, quoted_source_message_id, case_key, connector_id, account_id)
        )
        duplicate = message.message_id in self.message_ids
        self.message_ids.add(message.message_id)
        self.persisted = True
        accepted = not duplicate
        self.seen_persisted_when_returning.append(self.persisted)
        return accepted


def raw_message() -> RawMessage:
    return RawMessage(
        message_id="research:wa-1",
        tenant_id="research",
        conversation_id="chat-1",
        source_message_id="wa-1",
        text="laporan",
        received_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
    )


def openwa_payload() -> dict[str, str]:
    return {
        "id": "wa-2",
        "tenant_id": "research",
        "chat_id": "chat-2",
        "body": "laporan",
        "timestamp": "2026-09-11T00:00:00+00:00",
        "quotedMsgId": "wa-1",
        "case_key": "case-1",
    }


def test_accept_commits_before_returning_and_runs_assembly_in_service_boundary() -> None:
    runner = FakeRunner()
    assembler = FakeAssembler()
    service = IntakeService(runner, assembler)

    assert service.accept(raw_message()) is True
    assert runner.committed is True
    assert assembler.calls == 1


def test_openwa_callback_acks_after_persistence_without_assembly_or_openwa_dependency() -> None:
    intake = RecordingIntake()
    connector = OpenWAConnector(intake, "openwa", "account-1")

    assert connector.callback(openwa_payload()) is True

    message, quoted_ref, case_key, connector_id, account_id = intake.calls[0]
    assert intake.seen_persisted_when_returning == [True]
    assert intake.assembly_calls == 0
    assert isinstance(message, RawMessage)
    assert message.message_id == "research:openwa:account-1:wa-2"
    assert message.source_message_id == "wa-2"
    assert quoted_ref == "wa-1"
    assert case_key == "case-1"
    assert connector_id == "openwa"
    assert account_id == "account-1"
    assert connector.health_check() is True


def test_openwa_callback_forwards_duplicate_semantics() -> None:
    intake = RecordingIntake()
    connector = OpenWAConnector(intake, "openwa", "account-1")

    assert connector.callback(openwa_payload()) is True
    assert connector.callback(openwa_payload()) is False
    assert len(intake.calls) == 2
    assert intake.seen_persisted_when_returning == [True, True]


def test_openwa_multi_connector_isolates_message_identity_and_preserves_uniqueness() -> None:
    intake = RecordingIntake()
    connector_a1 = OpenWAConnector(intake, "openwa-a", "account-1")
    connector_a2 = OpenWAConnector(intake, "openwa-a", "account-2")
    connector_b1 = OpenWAConnector(intake, "openwa-b", "account-1")

    payload = openwa_payload()

    msg_a1 = connector_a1.normalize(payload)
    assert msg_a1.message_id == "research:openwa-a:account-1:wa-2"
    assert msg_a1.source_message_id == "wa-2"

    assert connector_a1.callback(payload) is True
    assert connector_a2.callback(payload) is True
    assert connector_b1.callback(payload) is True

    assert len(intake.message_ids) == 3
    assert "research:openwa-a:account-1:wa-2" in intake.message_ids
    assert "research:openwa-a:account-2:wa-2" in intake.message_ids
    assert "research:openwa-b:account-1:wa-2" in intake.message_ids

    assert connector_a1.callback(payload) is False
    assert connector_a2.callback(payload) is False
    assert connector_b1.callback(payload) is False


def test_persist_latency_measurement_is_deterministic_and_within_target(monkeypatch) -> None:
    ticks = iter((10.0, 10.125))
    monkeypatch.setattr("services.intake.service.perf_counter", lambda: next(ticks))

    accepted, latency_ms = IntakeService(FakeRunner(), FakeAssembler()).accept_with_latency(
        raw_message()
    )

    assert accepted is True
    assert latency_ms == 125.0
    assert latency_ms <= TARGET_PERSIST_LATENCY_MS


def test_openwa_filters_and_rejects_self_sent_events() -> None:
    intake = RecordingIntake()
    connector = OpenWAConnector(intake, "openwa", "account-1")

    base = openwa_payload()
    cases = [
        {**base, "fromMe": True},
        {**base, "from_me": True},
        {**base, "isMe": True},
        {**base, "self_sent": True},
        {**base, "sender": {"isMe": True}},
        {**base, "id": "true_628123456789@c.us_3EB0123456"},
        {**base, "from": "account-1"},
    ]

    for payload in cases:
        assert connector.is_self_sent(payload) is True
        assert connector.should_filter(payload) is True
        assert connector.is_filtered(payload) is True
        assert connector.callback(payload) is False
        with pytest.raises(ValueError, match="self-sent"):
            connector.normalize(payload)

    assert len(intake.calls) == 0


def test_openwa_filters_and_rejects_receipt_only_events() -> None:
    intake = RecordingIntake()
    connector = OpenWAConnector(intake, "openwa", "account-1")

    base = openwa_payload()
    cases = [
        {**base, "event": "onAck", "ack": 2},
        {**base, "event": "ack"},
        {**base, "type": "receipt"},
        {**base, "type": "delivery"},
        {**base, "receipt_only": True},
        {**base, "isReceipt": True},
        {"id": "wa-receipt-1", "ack": 1, "status": "delivered"},
    ]

    for payload in cases:
        assert connector.is_receipt_only(payload) is True
        assert connector.should_filter(payload) is True
        assert connector.callback(payload) is False
        with pytest.raises(ValueError, match="receipt-only"):
            connector.normalize(payload)

    assert len(intake.calls) == 0


def test_openwa_filters_and_rejects_group_chat_events() -> None:
    intake = RecordingIntake()
    connector = OpenWAConnector(intake, "openwa", "account-1")

    base = openwa_payload()
    cases = [
        {**base, "isGroupMsg": True},
        {**base, "isGroup": True},
        {**base, "group_chat": True},
        {**base, "chat_id": "120363024847291@g.us"},
        {**base, "from": "6281234567-1600000000@g.us"},
        {**base, "chat": {"isGroup": True}},
        {**base, "chat": {"id": "120363024847291@g.us"}},
    ]

    for payload in cases:
        assert connector.is_group_chat(payload) is True
        assert connector.should_filter(payload) is True
        assert connector.callback(payload) is False
        with pytest.raises(ValueError, match="group chat"):
            connector.normalize(payload)

    assert len(intake.calls) == 0


def test_openwa_normalized_message_id_fits_128_max_with_long_identifiers() -> None:
    intake = RecordingIntake()
    long_connector = "openwa-whatsapp-gateway-cluster-production-primary-node"
    long_account = "account-customer-service-department-jakarta-selatan-unit-1"
    connector = OpenWAConnector(intake, long_connector, long_account)

    payload = {
        "id": "3EB0" + "F" * 80,
        "tenant_id": "tenant-provinsi-dki-jakarta-indonesia",
        "chat_id": "chat-2",
        "body": "laporan infrastruktur jalan rusak",
        "timestamp": "2026-09-11T00:00:00+00:00",
    }

    message = connector.normalize(payload)
    assert len(message.message_id) <= 128
    assert message.message_id.startswith("tenant-provinsi")
    assert "openwa-whatsapp-gate" in message.message_id
    assert "account-customer-ser" in message.message_id
    assert message.source_message_id == payload["id"]

    assert connector.callback(payload) is True
    assert len(intake.calls) == 1
    persisted_msg = intake.calls[0][0]
    assert isinstance(persisted_msg, RawMessage)
    assert len(persisted_msg.message_id) <= 128


def test_openwa_long_id_collision_resistance_and_deduplication() -> None:
    intake = RecordingIntake()
    long_connector = "openwa-whatsapp-gateway-cluster-production-primary-node"
    long_account = "account-customer-service-department-jakarta-selatan-unit-1"
    connector = OpenWAConnector(intake, long_connector, long_account)

    payload_a = {
        "id": "wa-source-" + "A" * 70,
        "tenant_id": "research",
        "chat_id": "chat-1",
        "body": "laporan a",
        "timestamp": "2026-09-11T00:00:00+00:00",
    }
    payload_b = {
        "id": "wa-source-" + "B" * 70,
        "tenant_id": "research",
        "chat_id": "chat-1",
        "body": "laporan b",
        "timestamp": "2026-09-11T00:00:00+00:00",
    }

    msg_a = connector.normalize(payload_a)
    msg_b = connector.normalize(payload_b)
    msg_a_repeat = connector.normalize(payload_a)

    assert len(msg_a.message_id) <= 128
    assert len(msg_b.message_id) <= 128
    # Collision resistance
    assert msg_a.message_id != msg_b.message_id
    # Deterministic identity
    assert msg_a.message_id == msg_a_repeat.message_id

    # Intake deduplication semantics
    assert connector.callback(payload_a) is True
    assert connector.callback(payload_a) is False
    assert connector.callback(payload_b) is True
