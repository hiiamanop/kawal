from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from contracts.harness import (
    ArbitrationOutput,
    ComplaintCase,
    ExtractionOutput,
    HarnessRun,
    SocialPost,
    TriageOutput,
    VerificationOutput,
    deterministic_case_id,
)
from services.harness.budget import BudgetExceededError, BudgetTracker
from services.harness.location import assess_location
from services.harness.ports import Agent, AgentTask
from services.harness.source_intent import assess_source_intent
from services.harness.trace import TraceRecorder


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _hash_output(value: dict[str, Any] | None) -> str:
    if value is None:
        return "none"
    return _hash_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


class HarnessOrchestrator:
    def __init__(
        self,
        triage: Agent,
        extraction: Agent,
        verifier: Agent,
        arbiter: Agent | None = None,
        max_cost_usd: float = 0.05,
    ) -> None:
        self._triage = triage
        self._extraction = extraction
        self._verifier = verifier
        self._arbiter = arbiter
        self._max_cost_usd = max_cost_usd

    def run(self, post: SocialPost, run_id: str | None = None) -> HarnessRun:
        resolved_run_id = run_id or uuid4().hex
        trace = TraceRecorder(resolved_run_id)
        budget = BudgetTracker(self._max_cost_usd)
        trace.add("run.started", post_id=post.post_id, platform=post.platform)

        if not post.text.strip():
            trace.add("run.failed", reason="EMPTY_TEXT")
            return self._finish(resolved_run_id, post, "FAILED", None, trace, budget)

        case_id = deterministic_case_id(post)
        trace.add("case.identified", case_id=case_id)

        triage_result, exceeded = self._call_agent(
            self._triage,
            AgentTask(
                role="triage",
                source_text=post.text,
                source_media=post.media_urls,
                context={"allowed_categories": ["ROAD"], "city": "Bandung"},
                budget_remaining_usd=budget.remaining_usd,
            ),
            trace,
            budget,
            post,
        )
        if exceeded:
            return self._finish(
                resolved_run_id,
                post,
                "BUDGET_EXCEEDED",
                None,
                trace,
                budget,
                budget_status="EXCEEDED",
            )
        if triage_result is None or triage_result.status != "OK":
            trace.add("run.failed", reason="TRIAGE_UNAVAILABLE")
            return self._finish(resolved_run_id, post, "FAILED", None, trace, budget)

        triage = self._parse_triage(triage_result.output, trace)
        if triage is None:
            trace.add("run.failed", reason="TRIAGE_INVALID")
            return self._finish(resolved_run_id, post, "FAILED", None, trace, budget)

        extraction_result, exceeded = self._call_agent(
            self._extraction,
            AgentTask(
                role="extraction",
                source_text=post.text,
                source_media=post.media_urls,
                context={"city": "Bandung"},
                budget_remaining_usd=budget.remaining_usd,
            ),
            trace,
            budget,
            post,
        )
        if exceeded:
            return self._finish(
                resolved_run_id,
                post,
                "BUDGET_EXCEEDED",
                None,
                trace,
                budget,
                budget_status="EXCEEDED",
            )
        if extraction_result is None or extraction_result.status != "OK":
            trace.add("run.failed", reason="EXTRACTION_UNAVAILABLE")
            return self._finish(resolved_run_id, post, "FAILED", None, trace, budget)

        extraction = self._parse_extraction(extraction_result.output, trace)
        if extraction is None:
            trace.add("run.failed", reason="EXTRACTION_INVALID")
            return self._finish(resolved_run_id, post, "FAILED", None, trace, budget)

        location_assessment = assess_location(
            extraction.location,
            extraction.evidence_spans,
            post.text,
        )
        trace.add(
            "location.assessed",
            case_id=case_id,
            quality=location_assessment.quality,
            reason_codes=list(location_assessment.reason_codes),
        )
        source_assessment = assess_source_intent(post.text, post.author_handle)
        trace.add(
            "source.assessed",
            case_id=case_id,
            intent=source_assessment.intent,
            reason_codes=list(source_assessment.reason_codes),
        )

        verifier_result, exceeded = self._call_agent(
            self._verifier,
            AgentTask(
                role="verifier",
                source_text=post.text,
                source_media=post.media_urls,
                context={
                    "extraction": extraction.model_dump(mode="json"),
                    "allowed_categories": ["ROAD"],
                },
                budget_remaining_usd=budget.remaining_usd,
            ),
            trace,
            budget,
            post,
        )
        if exceeded:
            return self._finish(
                resolved_run_id,
                post,
                "BUDGET_EXCEEDED",
                None,
                trace,
                budget,
                budget_status="EXCEEDED",
            )
        if verifier_result is None or verifier_result.status != "OK":
            trace.add("run.review_required", reason="VERIFIER_UNAVAILABLE")
            return self._finish(
                resolved_run_id,
                post,
                "NEEDS_REVIEW",
                None,
                trace,
                budget,
            )

        verification = self._parse_verification(verifier_result.output, trace)
        if verification is None:
            trace.add("run.review_required", reason="VERIFICATION_INVALID")
            return self._finish(
                resolved_run_id,
                post,
                "NEEDS_REVIEW",
                None,
                trace,
                budget,
            )

        arbitration: ArbitrationOutput | None = None
        arbiter_calls = 0
        agreed = (
            triage.is_complaint == verification.is_complaint
            and triage.category == verification.category
        )
        reason_codes: list[str] = []

        non_actionable_sources = {
            "NEWS_REPORT",
            "OPINION",
            "QUESTION",
            "ANNOUNCEMENT",
        }
        if source_assessment.intent in non_actionable_sources and triage.is_complaint:
            reason_codes.extend(
                ("SOURCE_NOT_ACTIONABLE", f"SOURCE_{source_assessment.intent}")
            )
            trust_decision = "REJECTED"
            status = "REJECTED"
        elif not agreed:
            reason_codes.append("TRIAGE_VERIFIER_DISAGREEMENT")
            if self._arbiter is not None:
                arbiter_result, exceeded = self._call_agent(
                    self._arbiter,
                    AgentTask(
                        role="arbiter",
                        source_text=post.text,
                        context={
                            "triage": triage.model_dump(mode="json"),
                            "verification": verification.model_dump(mode="json"),
                        },
                        budget_remaining_usd=budget.remaining_usd,
                    ),
                    trace,
                    budget,
                    post,
                )
                arbiter_calls = 1
                if exceeded:
                    return self._finish(
                        resolved_run_id,
                        post,
                        "BUDGET_EXCEEDED",
                        None,
                        trace,
                        budget,
                        budget_status="EXCEEDED",
                    )
                if arbiter_result is not None and arbiter_result.status == "OK":
                    arbitration = self._parse_arbitration(arbiter_result.output, trace)
            trust_decision = "NEEDS_REVIEW"
            status = "NEEDS_REVIEW"
        elif not triage.is_complaint:
            reason_codes.append("NON_COMPLAINT")
            trust_decision = "REJECTED"
            status = "REJECTED"
        else:
            location_specific = location_assessment.quality == "SPECIFIC"
            if not verification.evidence_supported:
                reason_codes.append("EVIDENCE_NOT_SUPPORTED")
            if not location_specific:
                reason_codes.extend(location_assessment.reason_codes)
            if reason_codes:
                trust_decision = "NEEDS_REVIEW"
                status = "NEEDS_REVIEW"
            else:
                reason_codes.append("AGENTS_AGREE")
                trust_decision = "AGREED"
                status = "ACCEPTED"

        case = ComplaintCase(
            case_id=case_id,
            post=post,
            triage=triage,
            extraction=extraction,
            verification=verification,
            arbitration=arbitration,
            location_quality=location_assessment.quality,
            source_intent=source_assessment.intent,
            trust_decision=trust_decision,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            reason_codes=tuple(reason_codes),
        )
        trace.add(
            "case.written",
            case_id=case_id,
            status=status,
            trust_decision=trust_decision,
            reason_codes=list(reason_codes),
        )
        run_status = "COMPLETED" if status == "ACCEPTED" else status
        return self._finish(
            resolved_run_id,
            post,
            run_status,  # type: ignore[arg-type]
            case,
            trace,
            budget,
            arbiter_calls=arbiter_calls,
        )

    def _call_agent(
        self,
        agent: Agent,
        task: AgentTask,
        trace: TraceRecorder,
        budget: BudgetTracker,
        post: SocialPost,
    ) -> tuple[Any | None, bool]:
        estimate = agent.estimate_cost(task)
        if not budget.can_afford(estimate):
            trace.add(
                "budget.exceeded",
                role=task.role,
                estimate_usd=estimate,
                remaining_usd=budget.remaining_usd,
            )
            return None, True

        try:
            result = agent.run(task)
        except Exception as exc:
            trace.add(
                "agent.error",
                role=task.role,
                agent_id=getattr(agent, "agent_id", "unknown"),
                error_type=type(exc).__name__,
            )
            return None, False

        try:
            budget.record(task.role, result.usage)
        except BudgetExceededError:
            trace.add(
                "budget.exceeded",
                role=task.role,
                actual_cost_usd=result.usage.cost_usd,
                remaining_usd=budget.remaining_usd,
            )
            return None, True

        trace.add(
            "agent.completed",
            role=task.role,
            agent_id=result.agent_id,
            status=result.status,
            model=result.usage.model,
            provider=result.usage.provider,
            cost_usd=result.usage.cost_usd,
            latency_ms=result.usage.latency_ms,
            attempts=result.attempts,
            input_hash=_hash_text(post.text),
            output_hash=_hash_output(result.output),
        )
        return result, False

    @staticmethod
    def _parse_triage(
        output: dict[str, Any] | None, trace: TraceRecorder
    ) -> TriageOutput | None:
        try:
            return TriageOutput.model_validate(output)
        except Exception:
            trace.add("agent.invalid_output", role="triage")
            return None

    @staticmethod
    def _parse_extraction(
        output: dict[str, Any] | None, trace: TraceRecorder
    ) -> ExtractionOutput | None:
        try:
            return ExtractionOutput.model_validate(output)
        except Exception:
            trace.add("agent.invalid_output", role="extraction")
            return None

    @staticmethod
    def _parse_verification(
        output: dict[str, Any] | None, trace: TraceRecorder
    ) -> VerificationOutput | None:
        try:
            return VerificationOutput.model_validate(output)
        except Exception:
            trace.add("agent.invalid_output", role="verifier")
            return None

    @staticmethod
    def _parse_arbitration(
        output: dict[str, Any] | None, trace: TraceRecorder
    ) -> ArbitrationOutput | None:
        try:
            return ArbitrationOutput.model_validate(output)
        except Exception:
            trace.add("agent.invalid_output", role="arbiter")
            return None

    @staticmethod
    def _finish(
        run_id: str,
        post: SocialPost,
        status: str,
        case: ComplaintCase | None,
        trace: TraceRecorder,
        budget: BudgetTracker,
        *,
        arbiter_calls: int = 0,
        budget_status: str = "OK",
    ) -> HarnessRun:
        return HarnessRun(
            run_id=run_id,
            post=post,
            status=status,  # type: ignore[arg-type]
            case=case,
            trace=trace.events,
            cost=budget.report(
                arbiter_calls=arbiter_calls, budget_status=budget_status
            ),
        )
