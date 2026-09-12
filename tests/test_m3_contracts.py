from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import unittest
from pydantic import ValidationError

from contracts.models import (
    AgentEvidence,
    AgentResult,
    AgentResultStatus,
    AgentResultV1,
    AgentTelemetry,
    AgentUncertainty,
    AgentVersions,
    AnalysisResult,
    AttachmentMetadata,
    CanonicalSpan,
    CanonicalSpanLabel,
    CaseSnapshot,
    Category,
    ComplaintTrajectory,
    Completeness,
    CompletenessLevel,
    CostKind,
    DatasetSplit,
    DecisionMode,
    DecisionRecord,
    EvidenceItem,
    EvidenceKind,
    FamilySplitAuditResult,
    MessageInput,
    OperationReceipt,
    PolicyInput,
    PolicyResult,
    ProcessingState,
    RawMessage,
    ReplayFixture,
    ResultEvidence,
    ResultStatus,
    ResultTelemetry,
    ResultUncertainty,
    ResultVersions,
    RiskLevel,
    Sensitivity,
    TelemetryCostKind,
    TelemetryInfo,
    TicketCommand,
    TicketCreateRequest,
    TicketPriority,
    TicketReceipt,
    TicketStatus,
    TicketVisibility,
    TrajectoryBubble,
    TrajectoryTurn,
    TurnExpectedAction,
    UncertaintyInfo,
    UncertaintyMethod,
    VersionInfo,
)


def _make_valid_telemetry() -> TelemetryInfo:
    return TelemetryInfo(
        latency_ms=15.4,
        input_tokens=120,
        output_tokens=35,
        cost_usd=0.0002,
        cost_kind=CostKind.ACTUAL,
        attempt=1,
    )


def _make_valid_versions() -> VersionInfo:
    return VersionInfo(
        model="indobert-base-p1",
        preprocess="v1.0.0",
        calibration="temp-scaling-v1",
        prompt="prompt-v1",
    )


def _make_valid_uncertainty() -> UncertaintyInfo:
    return UncertaintyInfo(
        method=UncertaintyMethod.ENTROPY,
        value=0.042,
        calibrated=True,
    )


def _make_valid_evidence() -> tuple[EvidenceItem, ...]:
    return (
        EvidenceItem(
            kind=EvidenceKind.TEXT_SPAN,
            ref="msg-001",
            claim="jalan berlubang parah",
            start=0,
            end=21,
        ),
    )


class TestM3Contracts(unittest.TestCase):
    def test_category_enum_six_p0_categories(self) -> None:
        expected = {
            "ROAD",
            "DRAINAGE_FLOOD",
            "WASTE",
            "CLEAN_WATER",
            "CIVIL_ADMIN",
            "HEALTH_SERVICE",
        }
        actual = {c.value for c in Category}
        self.assertEqual(actual, expected)
        self.assertEqual(len(Category), 6)
        self.assertEqual(Category.ROAD, "ROAD")
        self.assertEqual(Category.DRAINAGE_FLOOD, "DRAINAGE_FLOOD")
        self.assertEqual(Category.WASTE, "WASTE")
        self.assertEqual(Category.CLEAN_WATER, "CLEAN_WATER")
        self.assertEqual(Category.CIVIL_ADMIN, "CIVIL_ADMIN")
        self.assertEqual(Category.HEALTH_SERVICE, "HEALTH_SERVICE")

    def test_risk_level_enum(self) -> None:
        expected = {"LOW", "MEDIUM", "HIGH", "URGENT"}
        actual = {r.value for r in RiskLevel}
        self.assertEqual(actual, expected)
        self.assertEqual(RiskLevel.LOW, "LOW")
        self.assertEqual(RiskLevel.MEDIUM, "MEDIUM")
        self.assertEqual(RiskLevel.HIGH, "HIGH")
        self.assertEqual(RiskLevel.URGENT, "URGENT")

    def test_completeness_enum_and_alias(self) -> None:
        expected = {"SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"}
        actual = {c.value for c in Completeness}
        self.assertEqual(actual, expected)
        self.assertEqual(Completeness.SUFFICIENT, "SUFFICIENT")
        self.assertEqual(Completeness.INCOMPLETE, "INCOMPLETE")
        self.assertEqual(Completeness.AMBIGUOUS, "AMBIGUOUS")
        self.assertIs(CompletenessLevel, Completeness)

    def test_agent_result_enums(self) -> None:
        self.assertEqual(
            {s.value for s in ResultStatus},
            {"SUCCEEDED", "UNAVAILABLE", "INVALID_OUTPUT", "TIMED_OUT"},
        )
        self.assertIs(AgentResultStatus, ResultStatus)

        self.assertEqual(
            {m.value for m in UncertaintyMethod},
            {"ENTROPY", "MARGIN", "INTERVAL", "RULE", "UNKNOWN"},
        )

        self.assertEqual(
            {k.value for k in EvidenceKind},
            {"TEXT_SPAN", "IMAGE", "DATABASE", "RULE"},
        )

        self.assertEqual(
            {c.value for c in CostKind},
            {"ACTUAL", "ESTIMATED", "UNKNOWN"},
        )
        self.assertIs(TelemetryCostKind, CostKind)

    def test_uncertainty_info(self) -> None:
        u = UncertaintyInfo(
            method=UncertaintyMethod.MARGIN,
            value=0.15,
            calibrated=True,
        )
        self.assertEqual(u.method, UncertaintyMethod.MARGIN)
        self.assertEqual(u.value, 0.15)
        self.assertTrue(u.calibrated)
        self.assertIs(ResultUncertainty, UncertaintyInfo)
        self.assertIs(AgentUncertainty, UncertaintyInfo)

        u_rule = UncertaintyInfo(
            method=UncertaintyMethod.RULE,
            value=None,
            calibrated=False,
        )
        self.assertIsNone(u_rule.value)
        self.assertFalse(u_rule.calibrated)

        with self.assertRaises(ValidationError):
            UncertaintyInfo(method="INVALID_METHOD", value=0.1, calibrated=True)  # type: ignore[arg-type]

        with self.assertRaises(ValidationError):
            UncertaintyInfo(
                method=UncertaintyMethod.MARGIN,
                value=0.1,
                calibrated=True,
                extra_field="disallowed",  # type: ignore[call-arg]
            )

    def test_evidence_item(self) -> None:
        item = EvidenceItem(
            kind=EvidenceKind.TEXT_SPAN,
            ref="raw_messages:001",
            claim="terjadi banjir 50cm",
            start=10,
            end=30,
        )
        self.assertEqual(item.kind, EvidenceKind.TEXT_SPAN)
        self.assertEqual(item.ref, "raw_messages:001")
        self.assertEqual(item.claim, "terjadi banjir 50cm")
        self.assertEqual(item.start, 10)
        self.assertEqual(item.end, 30)
        self.assertIs(ResultEvidence, EvidenceItem)
        self.assertIs(AgentEvidence, EvidenceItem)

        item_no_offsets = EvidenceItem(
            kind=EvidenceKind.IMAGE,
            ref="attachments:img-01",
            claim="foto genangan air",
        )
        self.assertIsNone(item_no_offsets.start)
        self.assertIsNone(item_no_offsets.end)

        with self.assertRaises(ValidationError):
            EvidenceItem(
                kind=EvidenceKind.TEXT_SPAN,
                ref="msg-1",
                claim="test",
                start=20,
                end=10,
            )

        with self.assertRaises(ValidationError):
            EvidenceItem(
                kind=EvidenceKind.TEXT_SPAN,
                ref="msg-1",
                claim="test",
                start=-1,
                end=10,
            )

    def test_version_info(self) -> None:
        v = VersionInfo(
            model="indobert-p0",
            preprocess="clean-v2",
        )
        self.assertEqual(v.model, "indobert-p0")
        self.assertEqual(v.preprocess, "clean-v2")
        self.assertIsNone(v.calibration)
        self.assertIsNone(v.prompt)
        self.assertIs(ResultVersions, VersionInfo)
        self.assertIs(AgentVersions, VersionInfo)

        v_full = VersionInfo(
            model="indobert-p0",
            preprocess="clean-v2",
            calibration="temp-scaling-0.8",
            prompt="system-prompt-v3",
        )
        self.assertEqual(v_full.calibration, "temp-scaling-0.8")
        self.assertEqual(v_full.prompt, "system-prompt-v3")

        with self.assertRaises(ValidationError):
            VersionInfo(model="indobert-p0")  # type: ignore[call-arg]

    def test_telemetry_info(self) -> None:
        t = TelemetryInfo(latency_ms=45.2)
        self.assertEqual(t.latency_ms, 45.2)
        self.assertIsNone(t.input_tokens)
        self.assertIsNone(t.output_tokens)
        self.assertIsNone(t.cost_usd)
        self.assertEqual(t.cost_kind, CostKind.UNKNOWN)
        self.assertEqual(t.attempt, 1)
        self.assertIs(ResultTelemetry, TelemetryInfo)
        self.assertIs(AgentTelemetry, TelemetryInfo)

        with self.assertRaises(ValidationError):
            TelemetryInfo(latency_ms=-1.0)

        with self.assertRaises(ValidationError):
            TelemetryInfo(latency_ms=10.0, attempt=0)

    def test_agent_result_v1_succeeded(self) -> None:
        result = AgentResultV1(
            result_id="res-001",
            task_id="task-clf-01",
            case_id="case-123",
            input_revision=1,
            input_hash="a" * 64,
            agent="classifier-local",
            task="category",
            status=ResultStatus.SUCCEEDED,
            data={"category": "DRAINAGE_FLOOD", "confidence": 0.94},
            confidence=0.94,
            uncertainty=_make_valid_uncertainty(),
            evidence=_make_valid_evidence(),
            versions=_make_valid_versions(),
            telemetry=_make_valid_telemetry(),
            error=None,
        )
        self.assertEqual(result.result_id, "res-001")
        self.assertEqual(result.status, ResultStatus.SUCCEEDED)
        self.assertIsNotNone(result.data)
        self.assertEqual(result.data["category"], "DRAINAGE_FLOOD")
        self.assertEqual(result.confidence, 0.94)
        self.assertIsNone(result.error)
        self.assertIs(AgentResult, AgentResultV1)

    def test_agent_result_v1_succeeded_without_confidence(self) -> None:
        result = AgentResultV1(
            result_id="res-002",
            task_id="task-rule-01",
            case_id="case-123",
            input_revision=1,
            input_hash="b" * 64,
            agent="rule-engine",
            task="category",
            status=ResultStatus.SUCCEEDED,
            data={"category": "ROAD"},
            confidence=None,
            uncertainty=UncertaintyInfo(
                method=UncertaintyMethod.RULE,
                value=None,
                calibrated=False,
            ),
            evidence=(),
            versions=VersionInfo(model="rule-v1", preprocess="raw"),
            telemetry=TelemetryInfo(latency_ms=0.5),
            error=None,
        )
        self.assertIsNone(result.confidence)
        self.assertEqual(result.data, {"category": "ROAD"})

    def test_agent_result_v1_succeeded_rejects_missing_data_or_present_error(self) -> None:
        with self.assertRaises(ValidationError) as ctx1:
            AgentResultV1(
                result_id="res-003",
                task_id="task-01",
                case_id="case-123",
                input_revision=1,
                input_hash="c" * 64,
                agent="clf",
                task="category",
                status=ResultStatus.SUCCEEDED,
                data=None,
                confidence=0.9,
                uncertainty=_make_valid_uncertainty(),
                versions=_make_valid_versions(),
                telemetry=_make_valid_telemetry(),
                error=None,
            )
        self.assertIn("data must be non-null when status is SUCCEEDED", str(ctx1.exception))

        with self.assertRaises(ValidationError) as ctx2:
            AgentResultV1(
                result_id="res-004",
                task_id="task-01",
                case_id="case-123",
                input_revision=1,
                input_hash="d" * 64,
                agent="clf",
                task="category",
                status=ResultStatus.SUCCEEDED,
                data={"category": "ROAD"},
                confidence=0.9,
                uncertainty=_make_valid_uncertainty(),
                versions=_make_valid_versions(),
                telemetry=_make_valid_telemetry(),
                error={"message": "something unexpected"},
            )
        self.assertIn("error must be null when status is SUCCEEDED", str(ctx2.exception))

    def test_agent_result_v1_failure_states(self) -> None:
        failure_statuses = [
            ResultStatus.UNAVAILABLE,
            ResultStatus.INVALID_OUTPUT,
            ResultStatus.TIMED_OUT,
        ]
        for status in failure_statuses:
            with self.subTest(status=status):
                result = AgentResultV1(
                    result_id="res-fail",
                    task_id="task-01",
                    case_id="case-123",
                    input_revision=2,
                    input_hash="e" * 64,
                    agent="clf",
                    task="category",
                    status=status,
                    data=None,
                    confidence=None,
                    uncertainty=UncertaintyInfo(
                        method=UncertaintyMethod.UNKNOWN,
                        value=None,
                        calibrated=False,
                    ),
                    evidence=(),
                    versions=VersionInfo(model="indobert", preprocess="v1"),
                    telemetry=TelemetryInfo(latency_ms=105.0),
                    error={"code": status.value, "detail": "failed as expected"},
                )
                self.assertEqual(result.status, status)
                self.assertIsNone(result.data)
                self.assertIsNone(result.confidence)
                self.assertIsNotNone(result.error)
                self.assertEqual(result.error["code"], status.value)

    def test_agent_result_v1_failure_states_reject_data_confidence_or_missing_error(self) -> None:
        failure_statuses = [
            ResultStatus.UNAVAILABLE,
            ResultStatus.INVALID_OUTPUT,
            ResultStatus.TIMED_OUT,
        ]
        for status in failure_statuses:
            with self.subTest(status=status):
                with self.assertRaises(ValidationError) as ctx1:
                    AgentResultV1(
                        result_id="res-fail-bad-data",
                        task_id="task-01",
                        case_id="case-123",
                        input_revision=1,
                        input_hash="f" * 64,
                        agent="clf",
                        task="category",
                        status=status,
                        data={"category": "ROAD"},
                        confidence=None,
                        uncertainty=_make_valid_uncertainty(),
                        versions=_make_valid_versions(),
                        telemetry=_make_valid_telemetry(),
                        error={"code": "ERR"},
                    )
                self.assertIn("data must be null when status is not SUCCEEDED", str(ctx1.exception))

                with self.assertRaises(ValidationError) as ctx2:
                    AgentResultV1(
                        result_id="res-fail-bad-conf",
                        task_id="task-01",
                        case_id="case-123",
                        input_revision=1,
                        input_hash="0" * 64,
                        agent="clf",
                        task="category",
                        status=status,
                        data=None,
                        confidence=0.5,
                        uncertainty=_make_valid_uncertainty(),
                        versions=_make_valid_versions(),
                        telemetry=_make_valid_telemetry(),
                        error={"code": "ERR"},
                    )
                self.assertIn("confidence must be null when status is not SUCCEEDED", str(ctx2.exception))

                with self.assertRaises(ValidationError) as ctx3:
                    AgentResultV1(
                        result_id="res-fail-no-err",
                        task_id="task-01",
                        case_id="case-123",
                        input_revision=1,
                        input_hash="1" * 64,
                        agent="clf",
                        task="category",
                        status=status,
                        data=None,
                        confidence=None,
                        uncertainty=_make_valid_uncertainty(),
                        versions=_make_valid_versions(),
                        telemetry=_make_valid_telemetry(),
                        error=None,
                    )
                self.assertIn("error must be non-null when status is not SUCCEEDED", str(ctx3.exception))

    def test_agent_result_v1_validation_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            AgentResultV1(
                result_id="res-bad-rev",
                task_id="task-01",
                case_id="case-123",
                input_revision=0,
                input_hash="2" * 64,
                agent="clf",
                task="category",
                status=ResultStatus.SUCCEEDED,
                data={"k": "v"},
                confidence=0.5,
                uncertainty=_make_valid_uncertainty(),
                versions=_make_valid_versions(),
                telemetry=_make_valid_telemetry(),
                error=None,
            )

        with self.assertRaises(ValidationError):
            AgentResultV1(
                result_id="res-bad-conf",
                task_id="task-01",
                case_id="case-123",
                input_revision=1,
                input_hash="3" * 64,
                agent="clf",
                task="category",
                status=ResultStatus.SUCCEEDED,
                data={"k": "v"},
                confidence=1.05,
                uncertainty=_make_valid_uncertainty(),
                versions=_make_valid_versions(),
                telemetry=_make_valid_telemetry(),
                error=None,
            )

    def test_agent_result_v1_immutability_and_coercion(self) -> None:
        result = AgentResultV1(
            result_id="res-coercion",
            task_id="task-coercion",
            case_id="case-001",
            input_revision=1,
            input_hash="4" * 64,
            agent="classifier",
            task="intent",
            status="SUCCEEDED",  # type: ignore[arg-type]
            data={"intent": "COMPLAINT"},
            confidence=0.88,
            uncertainty={"method": "MARGIN", "value": 0.2, "calibrated": True},  # type: ignore[arg-type]
            evidence=[{"kind": "TEXT_SPAN", "ref": "msg-1", "claim": "keluhan", "start": 0, "end": 7}],  # type: ignore[arg-type]
            versions={"model": "indobert", "preprocess": "v1"},  # type: ignore[arg-type]
            telemetry={"latency_ms": 22.0, "cost_kind": "ACTUAL", "attempt": 1},  # type: ignore[arg-type]
            error=None,
        )
        self.assertIsInstance(result.uncertainty, UncertaintyInfo)
        self.assertIsInstance(result.evidence, tuple)
        self.assertEqual(len(result.evidence), 1)
        self.assertIsInstance(result.evidence[0], EvidenceItem)
        self.assertIsInstance(result.versions, VersionInfo)
        self.assertIsInstance(result.telemetry, TelemetryInfo)
        self.assertEqual(result.status, ResultStatus.SUCCEEDED)

        with self.assertRaises(ValidationError):
            result.status = ResultStatus.UNAVAILABLE  # type: ignore[misc]

    def test_preserved_existing_contracts(self) -> None:
        analysis = AnalysisResult(
            category=Category.DRAINAGE_FLOOD,
            risk=RiskLevel.HIGH,
            sensitivity=Sensitivity.NORMAL,
            jurisdiction_id="JUR-01",
            authority_unit_id="UNIT-WATER-01",
            missing_fields=(),
            has_mandatory_evidence=True,
        )
        self.assertEqual(analysis.category, Category.DRAINAGE_FLOOD)
        self.assertEqual(analysis.risk, RiskLevel.HIGH)

        msg = RawMessage(
            message_id="msg-1",
            tenant_id="tenant-1",
            conversation_id="conv-1",
            source_message_id="src-1",
            text="Lapor sampah menumpuk",
            received_at=datetime.now(timezone.utc),
        )
        self.assertEqual(msg.text, "Lapor sampah menumpuk")

        policy_in = PolicyInput(
            tenant_id="tenant-1",
            case_id="case-1",
            revision=1,
            category=Category.WASTE,
            jurisdiction_id="JUR-01",
            authority_unit_id="UNIT-WASTE-01",
            missing_fields=(),
            has_mandatory_evidence=True,
            idempotency_key="idemp-01",
        )
        self.assertEqual(policy_in.category, Category.WASTE)

        policy_res = PolicyResult(
            decision="ALLOW",
            rule_ids=("rule-01",),
            reason_codes=("ok",),
        )
        self.assertEqual(policy_res.decision, "ALLOW")

        traj_bubble = TrajectoryBubble(source_message_id="b-1", text="halo")
        self.assertEqual(traj_bubble.offset_seconds, 0)
        self.assertEqual(traj_bubble.canonical_spans, ())

        self.assertEqual(DatasetSplit.TRAIN, "train")
        self.assertEqual(DatasetSplit.DEV, "dev")
        self.assertEqual(DatasetSplit.TEST, "test")

    def test_canonical_span_contract(self) -> None:
        span1 = CanonicalSpan(0, 5, "OBJ")
        self.assertEqual(span1.start, 0)
        self.assertEqual(span1.end, 5)
        self.assertEqual(span1.label, CanonicalSpanLabel.OBJ)
        self.assertEqual(span1[0], 0)
        self.assertEqual(span1[1], 5)
        self.assertEqual(span1[2], "OBJ")
        self.assertTrue(isinstance(span1, tuple))
        s, e, l = span1
        self.assertEqual((s, e, l), (0, 5, "OBJ"))
        self.assertEqual(span1, (0, 5, "OBJ"))
        self.assertEqual(span1._asdict(), {"start": 0, "end": 5, "label": "OBJ"})

        # Alternate instantiation forms
        span_kw = CanonicalSpan(start=10, end=20, label=CanonicalSpanLabel.LOC)
        self.assertEqual(span_kw, (10, 20, "LOC"))
        span_seq = CanonicalSpan([30, 40, "TIME"])
        self.assertEqual(span_seq, (30, 40, "TIME"))
        span_dict = CanonicalSpan({"start": 50, "end": 60, "label": "OBJ"})
        self.assertEqual(span_dict, (50, 60, "OBJ"))

        # Validation: offsets must be non-negative
        with self.assertRaises(ValueError):
            CanonicalSpan(-1, 5, "OBJ")
        with self.assertRaises(ValueError):
            CanonicalSpan(0, -5, "OBJ")

        # Validation: start must be strictly less than end
        with self.assertRaises(ValueError):
            CanonicalSpan(5, 5, "OBJ")
        with self.assertRaises(ValueError):
            CanonicalSpan(6, 5, "OBJ")

        # Validation: allowed labels are OBJ, LOC, TIME
        with self.assertRaises(ValueError):
            CanonicalSpan(0, 5, "INVALID")
        with self.assertRaises(ValueError):
            CanonicalSpan(0, 5, "B-OBJ")

        # Validation: extra fields forbidden
        with self.assertRaises(ValueError):
            CanonicalSpan(0, 5, "OBJ", extra_field="bad")
        with self.assertRaises(ValueError):
            CanonicalSpan({"start": 0, "end": 5, "label": "OBJ", "extra": 1})

    def test_trajectory_bubble_canonical_spans(self) -> None:
        # Default empty tuple preserves M1/M2 backward compatibility
        bubble_default = TrajectoryBubble(source_message_id="b-0", text="jalan rusak")
        self.assertEqual(bubble_default.canonical_spans, ())

        # HiFi list-of-lists format
        bubble_hifi = TrajectoryBubble.model_validate({
            "source_message_id": "b-1",
            "text": "Jalan rusak di Kaliurang kemarin",
            "offset_seconds": 5,
            "canonical_spans": [
                [0, 11, "OBJ"],
                [15, 24, "LOC"],
                [25, 32, "TIME"],
            ],
        })
        self.assertEqual(len(bubble_hifi.canonical_spans), 3)
        self.assertIsInstance(bubble_hifi.canonical_spans[0], CanonicalSpan)
        self.assertEqual(bubble_hifi.canonical_spans[0], (0, 11, "OBJ"))
        self.assertEqual(bubble_hifi.canonical_spans[1], (15, 24, "LOC"))
        self.assertEqual(bubble_hifi.canonical_spans[2], (25, 32, "TIME"))

        # Dict format
        bubble_dict = TrajectoryBubble.model_validate({
            "source_message_id": "b-2",
            "text": "Lampu merah padam",
            "canonical_spans": [{"start": 0, "end": 11, "label": "OBJ"}],
        })
        self.assertEqual(bubble_dict.canonical_spans[0], (0, 11, "OBJ"))

        # extra='forbid' is strictly preserved
        with self.assertRaises(ValidationError):
            TrajectoryBubble(
                source_message_id="b-3",
                text="valid text",
                extra_unsupported_field="disallowed",  # type: ignore[call-arg]
            )

        # Invalid span offsets trigger ValidationError
        with self.assertRaises(ValidationError):
            TrajectoryBubble.model_validate({
                "source_message_id": "b-4",
                "text": "valid text",
                "canonical_spans": [[-1, 5, "OBJ"]],
            })

        with self.assertRaises(ValidationError):
            TrajectoryBubble.model_validate({
                "source_message_id": "b-5",
                "text": "valid text",
                "canonical_spans": [[5, 5, "OBJ"]],
            })

        with self.assertRaises(ValidationError):
            TrajectoryBubble.model_validate({
                "source_message_id": "b-6",
                "text": "valid text",
                "canonical_spans": [[0, 5, "DISALLOWED_LABEL"]],
            })

    def test_complaint_trajectory_with_canonical_spans(self) -> None:
        bubble = TrajectoryBubble(
            source_message_id="b-1",
            text="Jalan rusak di Kaliurang",
            canonical_spans=(CanonicalSpan(0, 11, "OBJ"), CanonicalSpan(15, 24, "LOC")),
        )
        turn_action = TurnExpectedAction(turn=1, allowed_actions=(DecisionMode.EXECUTE,))
        turn = TrajectoryTurn(
            turn=1,
            bubbles=(bubble,),
            expected_action=turn_action,
        )
        traj = ComplaintTrajectory(
            scenario_id="scen-001",
            family_id="fam-001",
            split=DatasetSplit.TRAIN,
            category=Category.ROAD,
            turns=(turn,),
            canonical_spans=(CanonicalSpan(0, 11, "OBJ"),),
        )
        self.assertEqual(len(traj.canonical_spans), 1)
        self.assertEqual(traj.canonical_spans[0], (0, 11, "OBJ"))
        self.assertEqual(len(traj.turns[0].bubbles[0].canonical_spans), 2)
        self.assertEqual(traj.turns[0].bubbles[0].canonical_spans[0], (0, 11, "OBJ"))

        # extra='forbid' preserved on ComplaintTrajectory
        with self.assertRaises(ValidationError):
            ComplaintTrajectory(
                scenario_id="scen-002",
                family_id="fam-002",
                split=DatasetSplit.TRAIN,
                category=Category.ROAD,
                turns=(turn,),
                unsupported_extra_field="bad",  # type: ignore[call-arg]
            )

    def test_migration_files_integrity(self) -> None:
        root = Path(__file__).resolve().parent.parent
        supa_migration = root / "supabase" / "migrations" / "20260911000007_m3_model_artifacts.sql"
        infra_migration = root / "infra" / "migrations" / "008_m3_model_artifacts.sql"

        self.assertTrue(supa_migration.is_file(), f"{supa_migration} missing")
        self.assertTrue(infra_migration.is_file(), f"{infra_migration} missing")

        supa_content = supa_migration.read_text(encoding="utf-8")
        infra_content = infra_migration.read_text(encoding="utf-8")

        self.assertEqual(supa_content, infra_content, "Migration files must be identical")

        required_columns = ["task", "tokenizer", "calibration", "dataset_split", "eval_summary"]
        for col in required_columns:
            self.assertTrue(
                f"ADD COLUMN {col}" in supa_content or f"{col} " in supa_content,
                f"Missing column {col}",
            )

        self.assertIn("ENABLE ROW LEVEL SECURITY", supa_content)
        self.assertIn("service_role", supa_content)
        self.assertIn("authenticated", supa_content)
        self.assertIn("anon", supa_content)


if __name__ == "__main__":
    unittest.main()
