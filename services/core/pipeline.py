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
    ConflictInput,
    ConflictType,
    DecisionInput,
    DecisionPlan,
    EscalationCandidate,
    TrustContext,
    TrustObservation,
    decide,
    diagnose_conflict,
    estimate_contextual_trust,
)
from services.core.escalation import EscalationCommand, build_anonymized_escalation_command
from services.core.orchestrator import build_ticket_command
from services.core.policy import AUTHORITY_DIRECTORY
from services.intelligence.causality import analyze_causality
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
    escalation_command: EscalationCommand | None = None
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
        previous_category: Category | None = None,
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

        return self.process_snapshot(
            snapshot,
            jurisdiction_id=jurisdiction_id,
            previous_category=previous_category,
        )

    def process_snapshot(
        self,
        snapshot: CaseSnapshot,
        jurisdiction_id: str | None = None,
        previous_category: Category | None = None,
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

        # Causal & Cross-Domain Disentanglement
        causal_res = analyze_causality(text, ml_runtime=self._ml_runtime)
        if causal_res.hazard_escalation_category is not None:
            category = causal_res.hazard_escalation_category
            risk = RiskLevel.HIGH
        elif causal_res.has_causal_relation and causal_res.root_cause_category is not None:
            category = causal_res.root_cause_category

        # Multi-Turn Category Anchoring & Conflict Diagnosis (Solusi 1 & 3)
        conflicts: list[Any] = []
        if previous_category is not None and previous_category != category:
            conflict = diagnose_conflict(
                ConflictInput(
                    type=ConflictType.CLASSIFICATION,
                    field="category",
                    left_value=previous_category.value,
                    right_value=category.value,
                    left_trust=0.9,
                    right_trust=float(pred.confidence if pred else 0.5),
                    consequence_weight=1.0,
                    evidence_refs=(f"rev_{snapshot.revision-1}", f"rev_{snapshot.revision}"),
                    route_changes=True,
                )
            )
            if conflict is not None:
                conflicts.append(conflict)

            # Solusi 1: When clarifying details without explicit category correction, anchor to prior category
            cancel_cues = ("bukan", "salah lapor", "ralat aduan", "keliru")
            explicit_correction = any(cue in text.lower() for cue in cancel_cues)
            if not explicit_correction and snapshot.revision > 1:
                category = previous_category

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

        # 4. Pure 4-Mode Decision Evaluation (Hazard-aware and robust against false rejections)
        HAZARD_DISTRESS_CUES = (
            "bahaya", "tolong", "bantu", "darurat", "urgent",
            "rusak", "roboh", "tumbang", "ambles", "ambruk", "jebol", "hancur", "pecah", "bocor",
            "meletup", "percikan api", "kebakaran", "hangus", "asap",
            "banjir", "tergenang", "meluap", "mampet", "tersumbat",
            "sampah", "bau", "busuk", "limbah",
            "jatoh", "jatuh", "kepleset", "korban", "celaka", "tabrakan", "macet parah",
            "pungli", "pemerasan", "tawuran",
        )
        text_lower = text.lower()
        has_hazard_signal = any(cue in text_lower for cue in HAZARD_DISTRESS_CUES)
        is_high_risk = (risk in (RiskLevel.HIGH, RiskLevel.URGENT))
        is_emergency_cat = (category in (Category.FIRE_RESCUE, Category.ROAD, Category.DRAINAGE_FLOOD, Category.CLEAN_WATER))
        has_complaint_explicit = ("lapor" in text_lower or "aduan" in text_lower or "keluhan" in text_lower)

        is_complaint = (
            intent == "COMPLAINT"
            or has_complaint_explicit
            or (has_hazard_signal and (is_high_risk or is_emergency_cat or "tolong" in text_lower or "min" in text_lower))
        )

        pure_opinion_cues = ("sekadar opini", "hanya pendapat", "sekadar saran", "sekadar salam", "cuma nanya opini")
        if any(cue in text_lower for cue in pure_opinion_cues) and not has_hazard_signal:
            is_complaint = False

        required_complete = (completeness == "SUFFICIENT")
        evidence_covered = (len(ml_res.entities) > 0 or len(text.strip()) > 20)

        escalation_candidates = ()
        if self._defer_execution and is_complaint and authority_valid and (not required_complete or not evidence_covered):
            escalation_candidates = (
                EscalationCandidate(
                    task_id="local-clarification-check",
                    expected_utility_gain=5.0,
                    cost_usd=0.0,
                    latency_ms=300.0,
                    egress_bytes=0,
                    queue_delay_ms=0.0,
                    policy_allowed=True,
                ),
            )

        decision_input = DecisionInput(
            case_id=snapshot.case_id,
            revision=snapshot.revision,
            required_fields_complete=required_complete,
            evidence_covered=evidence_covered,
            authority_valid=authority_valid,
            policy_allowed=authority_valid,
            is_complaint=is_complaint,
            conflicts=tuple(conflicts),
            escalation_candidates=escalation_candidates,
            budget=BudgetState(remaining_usd=1.0, remaining_latency_ms=5000.0, remaining_egress_bytes=1000000),
            clarification_available=True,
        )
        plan: DecisionPlan = decide(decision_input)

        # 5. Build commands and optionally execute inline if not deferred
        ticket_receipt: TicketReceipt | None = None
        ticket_command: TicketCommand | None = None
        outbound_command: OutboundMessageCommand | None = None
        escalation_command: EscalationCommand | None = None
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

        elif plan.mode == DecisionMode.RE_EVALUATE and plan.selected_task_id is not None:
            escalation_command = build_anonymized_escalation_command(
                tenant_id=snapshot.tenant_id,
                case_id=snapshot.case_id,
                revision=snapshot.revision,
                task_id=plan.selected_task_id,
                category=category.value,
                risk=risk.value,
                completeness=completeness,
            )
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
            escalation_command=escalation_command,
            reason_codes=plan.reason_codes,
            audit_trace={
                "plan": plan.model_dump(mode="json"),
                "trust": trust.model_dump(mode="json"),
                "ml_status": ml_res.status,
                "ml_latency_ms": ml_res.latency_ms,
            },
        )
