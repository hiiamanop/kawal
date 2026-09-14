from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from contracts.models import (
    AnalysisResult,
    CaseSnapshot,
    Category,
    DecisionMode,
    OutboundMessageCommand,
    OutboundSendReceipt,
    ProcessingState,
    RawMessage,
    RiskLevel,
    Sensitivity,
    TicketCommand,
    TicketReceipt,
)
from services.clarification.dispatcher import ClarificationDispatcher
from services.clarification.engine import (
    ClarificationSession,
    start_clarification_session,
)
from services.core.decision import (
    BudgetState,
    DecisionInput,
    DecisionPlan,
    TrustContext,
    TrustObservation,
    decide,
    estimate_contextual_trust,
)
from services.core.orchestrator import build_ticket_command
from services.core.policy import AUTHORITY_DIRECTORY
from services.intelligence.completeness import resolve_location_completeness
from services.ml.runtime import (
    ClassificationPrediction,
    LocalMLRuntime,
    SpanNER,
)
from services.reliability.client import ReliableTicketClient


class PipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    tenant_id: str
    conversation_id: str
    revision: int
    decision_mode: DecisionMode
    processing_state: ProcessingState
    category: Category
    risk: RiskLevel
    completeness: str
    ticket_receipt: TicketReceipt | None = None
    clarification_session: ClarificationSession | None = None
    outbound_receipt: OutboundSendReceipt | None = None
    ticket_command: TicketCommand | None = None
    outbound_command: OutboundMessageCommand | None = None
    reason_codes: tuple[str, ...] = ()
    audit_trace: dict[str, Any] = Field(default_factory=dict)


def _map_risk(raw: str) -> RiskLevel:
    clean = raw.upper().strip()
    for level in RiskLevel:
        if level.value == clean:
            return level
    return RiskLevel.MEDIUM


def _map_category(raw: str) -> Category:
    clean = raw.upper().strip()
    for cat in Category:
        if cat.value == clean:
            return cat
    return Category.ROAD


class CaseProcessingPipeline:
    def __init__(
        self,
        ml_runtime: LocalMLRuntime | None = None,
        ticket_client: ReliableTicketClient | None = None,
        clarification_dispatcher: ClarificationDispatcher | None = None,
        default_jurisdiction_id: str = "JUR-FICT-01",
        defer_execution: bool = False,
    ) -> None:
        self._ml_runtime = ml_runtime or LocalMLRuntime()
        self._ticket_client = ticket_client
        self._clarification_dispatcher = clarification_dispatcher or ClarificationDispatcher()
        self._default_jurisdiction_id = default_jurisdiction_id
        self._defer_execution = defer_execution

    @property
    def ml_runtime(self) -> LocalMLRuntime:
        return self._ml_runtime

    @property
    def ticket_client(self) -> ReliableTicketClient | None:
        return self._ticket_client

    @property
    def clarification_dispatcher(self) -> ClarificationDispatcher:
        return self._clarification_dispatcher

    def process_messages(
        self,
        tenant_id: str,
        conversation_id: str,
        messages: Sequence[RawMessage | str],
        case_id: str | None = None,
        revision: int = 1,
        jurisdiction_id: str | None = None,
    ) -> PipelineResult:
        """Helper to process a list of raw messages or text strings directly."""
        now = datetime.now(timezone.utc)
        resolved_case_id = case_id or f"case-{uuid4().hex[:10]}"
        raw_msgs: list[RawMessage] = []

        for idx, m in enumerate(messages, 1):
            if isinstance(m, RawMessage):
                raw_msgs.append(m)
            else:
                raw_msgs.append(
                    RawMessage(
                        message_id=f"{tenant_id}:{conversation_id}:msg-{idx}",
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        source_message_id=f"src-{idx}-{uuid4().hex[:6]}",
                        text=str(m),
                        received_at=now,
                    )
                )

        combined_text = " ".join(m.text for m in raw_msgs)
        evidence_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()

        snapshot = CaseSnapshot(
            case_id=resolved_case_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            revision=revision,
            state=ProcessingState.READY,
            messages=tuple(raw_msgs),
            evidence_hash=evidence_hash,
            created_at=now,
        )

        return self.process_snapshot(snapshot, jurisdiction_id=jurisdiction_id)

    def process_snapshot(
        self,
        snapshot: CaseSnapshot,
        jurisdiction_id: str | None = None,
    ) -> PipelineResult:
        """Run complete end-to-end KAWAL decision and execution cycle on a case snapshot."""
        jur_id = jurisdiction_id or self._default_jurisdiction_id
        text = " ".join(m.text for m in snapshot.messages)

        # 1. Local ML Runtime Inference (IndoBERT + NER + Hybrid Completeness)
        ml_res = self._ml_runtime.predict(text, context={"jurisdiction_id": jur_id})
        pred = ml_res.prediction

        if pred is not None:
            category = _map_category(pred.category)
            risk = _map_risk(pred.risk)
            completeness = pred.completeness
            intent = pred.intent
        else:
            # Deterministic local fallback
            completeness = resolve_location_completeness(text, ml_res.entities)
            risk = RiskLevel.MEDIUM
            intent = "COMPLAINT"
            lower = text.lower()
            if "sampah" in lower or "limbah" in lower:
                category = Category.WASTE
            elif "banjir" in lower or "drainase" in lower or "gorong" in lower:
                category = Category.DRAINAGE_FLOOD
            elif "air" in lower or "pdam" in lower or "keruh" in lower:
                category = Category.CLEAN_WATER
            elif "ktp" in lower or "dukcapil" in lower or "kk" in lower:
                category = Category.CIVIL_ADMIN
            elif "puskesmas" in lower or "bpjs" in lower or "faskes" in lower:
                category = Category.HEALTH_SERVICE
            elif "satpol" in lower or "tertib" in lower or "pkl" in lower:
                category = Category.PUBLIC_ORDER
            elif "dishub" in lower or "rambu" in lower or "angkot" in lower or "lampu merah" in lower:
                category = Category.TRANSPORTATION
            elif "kebakaran" in lower or "damkar" in lower or "api" in lower:
                category = Category.FIRE_RESCUE
            elif "bansos" in lower or "telantar" in lower or "dinsos" in lower:
                category = Category.SOCIAL_AFFAIRS
            elif "sekolah" in lower or "disdik" in lower or "guru" in lower:
                category = Category.EDUCATION
            elif "taman" in lower or "pohon" in lower or "perumahan" in lower:
                category = Category.PARKS_HOUSING
            else:
                category = Category.ROAD

        # 2. Authority Routing Verification
        expected_unit = AUTHORITY_DIRECTORY.get((jur_id, category))
        authority_valid = expected_unit is not None
        authority_unit_id = expected_unit or "UNIT-UNASSIGNED"

        # 3. Contextual Trust Estimation
        trust = estimate_contextual_trust(
            observation=None,
            parent_successes=95,
            parent_total=100,
            evidence_quality=0.95,
        )

        # 4. Pure 4-Mode Decision Evaluation
        is_complaint = (intent == "COMPLAINT" or "lapor" in text.lower() or "aduan" in text.lower())
        if "opini" in text.lower() or "sekadar salam" in text.lower():
            is_complaint = False

        required_complete = (completeness == "SUFFICIENT")
        evidence_covered = (len(ml_res.entities) > 0 or len(text.strip()) > 20)

        decision_input = DecisionInput(
            case_id=snapshot.case_id,
            revision=snapshot.revision,
            required_fields_complete=required_complete,
            evidence_covered=evidence_covered,
            authority_valid=authority_valid,
            policy_allowed=authority_valid,
            is_complaint=is_complaint,
            conflicts=(),
            escalation_candidates=(),
            budget=BudgetState(remaining_usd=1.0, remaining_latency_ms=5000.0, remaining_egress_bytes=1000000),
            clarification_available=True,
        )
        plan: DecisionPlan = decide(decision_input)

        # 5. Build commands and optionally execute inline if not deferred
        ticket_receipt: TicketReceipt | None = None
        ticket_command: TicketCommand | None = None
        outbound_command: OutboundMessageCommand | None = None
        outbound_receipt: OutboundSendReceipt | None = None
        clarification_session: ClarificationSession | None = None

        if plan.mode == DecisionMode.EXECUTE:
            analysis = AnalysisResult(
                category=category,
                risk=risk,
                sensitivity=Sensitivity.NORMAL,
                jurisdiction_id=jur_id,
                authority_unit_id=authority_unit_id,
                missing_fields=(),
                has_mandatory_evidence=True,
            )
            ticket_command = build_ticket_command(snapshot, analysis)
            if ticket_command is not None:
                if not self._defer_execution and self._ticket_client is not None:
                    ticket_receipt = self._ticket_client.create_ticket(
                        ticket_command.request, ticket_command.idempotency_key
                    )
                proc_state = ProcessingState.TICKETED
            else:
                proc_state = ProcessingState.WAITING_RESULTS

        elif plan.mode == DecisionMode.REQUEST_CLARIFICATION:
            missing_fields = ("location",) if completeness in ("INCOMPLETE", "AMBIGUOUS") else ("issue",)
            session = start_clarification_session(
                case_id=snapshot.case_id,
                tenant_id=snapshot.tenant_id,
                conversation_id=snapshot.conversation_id,
                category=category,
                missing_fields=missing_fields,
            )
            clarification_session = session
            last_msg_src = snapshot.messages[-1].source_message_id if snapshot.messages else None
            outbound_command = self._clarification_dispatcher.build_clarification_command(
                session=session,
                quoted_source_message_id=last_msg_src,
            )
            if not self._defer_execution and self._clarification_dispatcher is not None:
                session, outbound_receipt = self._clarification_dispatcher.dispatch_clarification_request(
                    session=session,
                    quoted_source_message_id=last_msg_src,
                )
                clarification_session = session
            proc_state = ProcessingState.WAITING_CLARIFICATION

        elif plan.mode == DecisionMode.REJECT_IGNORE:
            proc_state = ProcessingState.REJECTED

        else:  # RE_EVALUATE
            proc_state = ProcessingState.WAITING_RESULTS

        return PipelineResult(
            case_id=snapshot.case_id,
            tenant_id=snapshot.tenant_id,
            conversation_id=snapshot.conversation_id,
            revision=snapshot.revision,
            decision_mode=plan.mode,
            processing_state=proc_state,
            category=category,
            risk=risk,
            completeness=completeness,
            ticket_receipt=ticket_receipt,
            ticket_command=ticket_command,
            clarification_session=clarification_session,
            outbound_receipt=outbound_receipt,
            outbound_command=outbound_command,
            reason_codes=plan.reason_codes,
            audit_trace={
                "plan": plan.model_dump(mode="json"),
                "trust": trust.model_dump(mode="json"),
                "ml_status": ml_res.status,
            },
        )
