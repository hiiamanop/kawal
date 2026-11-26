from __future__ import annotations

import re
from typing import NamedTuple, Protocol, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from contracts.harness import (
    CacheProvenance,
    ComplaintCase,
    CostReport,
    ExtractionOutput,
    HarnessRun,
    SocialPost,
    TraceEvent,
    TriageOutput,
    VerificationOutput,
    deterministic_case_id,
)
from contracts.models import RiskLevel
from services.harness.agents import (
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.cache import CacheEntry, HarnessAnalysisCache
from services.harness.location import assess_location
from services.harness.ports import Agent, AgentResult, AgentTask
from services.harness.source_intent import assess_source_intent
from services.harness.trace import TraceRecorder


_LOCATION_PREFIXES = frozenset({"jl", "jalan", "gg", "gang"})
_GENERIC_LOCATION_TOKENS = frozenset(
    {
        "stasiun",
        "terminal",
        "pasar",
        "kecamatan",
        "kelurahan",
        "kota",
        "kabupaten",
        "daerah",
        "dekat",
        "no",
        "nomor",
        "rt",
        "rw",
        "kel",
        "kec",
        "jl",
        "jalan",
        "gg",
        "gang",
        "raya",
    }
)
_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.URGENT: 3,
}


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def normalize_location(value: str) -> str:
    tokens = [
        token
        for token in _tokens(value)
        if token not in _LOCATION_PREFIXES and token not in _GENERIC_LOCATION_TOKENS
    ]
    return " ".join(tokens)


def locations_overlap(current: str | None, cached: Sequence[str]) -> bool:
    if not current:
        return False
    current_normalized = normalize_location(current)
    if not current_normalized:
        return False
    current_tokens = set(current_normalized.split())
    for candidate in cached:
        candidate_normalized = normalize_location(candidate)
        if not candidate_normalized:
            continue
        if current_normalized == candidate_normalized:
            return True
        overlap = current_tokens.intersection(candidate_normalized.split())
        if len(overlap) >= 2:
            return True
        if any(len(token) >= 5 for token in overlap):
            return True
    return False


def _max_risk(first: RiskLevel, second: RiskLevel) -> RiskLevel:
    return first if _RISK_ORDER[first] >= _RISK_ORDER[second] else second


class CacheReuseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    reason_codes: tuple[str, ...] = ()


class _PrefilterResult(NamedTuple):
    triage: TriageOutput
    extraction: ExtractionOutput
    verification: VerificationOutput
    latency_ms: float


class RunOrchestrator(Protocol):
    def run(self, post: SocialPost, run_id: str | None = None) -> HarnessRun: ...


class CacheFirstRunner:
    """Run deterministic guards, then reuse a safe cache entry or delegate."""

    def __init__(
        self,
        *,
        inner: RunOrchestrator,
        cache: HarnessAnalysisCache,
        rule_triage: Agent | None = None,
        rule_extraction: Agent | None = None,
        rule_verifier: Agent | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._rule_triage = rule_triage or RuleTriageAgent()
        self._rule_extraction = rule_extraction or RuleExtractionAgent()
        self._rule_verifier = rule_verifier or RuleVerifierAgent()

    def run(self, post: SocialPost, run_id: str | None = None) -> HarnessRun:
        resolved_run_id = run_id or uuid4().hex
        trace = TraceRecorder(resolved_run_id)
        trace.add("run.started", post_id=post.post_id, platform=post.platform)

        if not post.text.strip():
            inner_run = self._inner.run(post, run_id=resolved_run_id)
            return inner_run.model_copy(
                update={"trace": tuple(trace.events) + inner_run.trace}
            )

        source_assessment = assess_source_intent(post.text, post.author_handle)
        prefilter = self._prefilter(post, trace)
        if prefilter is None:
            provenance = CacheProvenance(
                attempted=False,
                reason_codes=("CACHE_PREFILTER_UNAVAILABLE",),
            )
            inner_run = self._inner.run(post, run_id=resolved_run_id)
            return inner_run.model_copy(
                update={
                    "trace": tuple(trace.events) + inner_run.trace,
                    "cache": provenance,
                }
            )

        location = assess_location(
            prefilter.extraction.location,
            prefilter.extraction.evidence_spans,
            post.text,
        )
        lookup = self._cache.lookup(
            post.text,
            prefilter.triage.category,
            source_assessment.intent,
        )
        trace.add(
            "cache.lookup",
            hit=lookup.hit,
            match_type=lookup.match_type,
            similarity=lookup.similarity,
            reason_codes=list(lookup.reason_codes),
        )
        source_case_id = self._source_case_id(lookup.entry)
        provenance = CacheProvenance(
            attempted=True,
            hit=lookup.hit,
            reused=False,
            match_type=lookup.match_type,
            similarity=lookup.similarity,
            source_case_id=source_case_id,
            reason_codes=lookup.reason_codes,
        )

        if lookup.hit and lookup.entry is not None:
            decision = self._reuse_decision(
                post=post,
                entry=lookup.entry,
                source_intent=source_assessment.intent,
                location_quality=location.quality,
                rule_triage=prefilter.triage,
                rule_extraction=prefilter.extraction,
                rule_verification=prefilter.verification,
                match_type=lookup.match_type,
            )
            if decision.allowed:
                self._cache.increment_hit(lookup.entry)
                case = self._cached_case(
                    post=post,
                    entry=lookup.entry,
                    source_intent=source_assessment.intent,
                    location_quality=location.quality,
                    rule_triage=prefilter.triage,
                    rule_extraction=prefilter.extraction,
                    rule_verification=prefilter.verification,
                    match_type=lookup.match_type,
                )
                trace.add(
                    "cache.hit",
                    match_type=lookup.match_type,
                    similarity=lookup.similarity,
                    source_case_id=source_case_id,
                )
                trace.add("cache.reused", case_id=case.case_id)
                trace.add(
                    "case.written",
                    case_id=case.case_id,
                    status=case.status,
                    trust_decision=case.trust_decision,
                    reason_codes=list(case.reason_codes),
                )
                provenance = provenance.model_copy(update={"reused": True})
                return HarnessRun(
                    run_id=resolved_run_id,
                    post=post,
                    status="COMPLETED",
                    case=case,
                    trace=trace.events,
                    cost=CostReport(
                        total_cost_usd=0.0,
                        total_latency_ms=prefilter.latency_ms,
                        calls=0,
                        by_role={
                            "triage": 0.0,
                            "extraction": 0.0,
                            "verifier": 0.0,
                        },
                    ),
                    cache=provenance,
                )

            trace.add("cache.rejected", reason_codes=list(decision.reason_codes))
            provenance = provenance.model_copy(
                update={
                    "reason_codes": tuple(
                        dict.fromkeys(provenance.reason_codes + decision.reason_codes)
                    )
                }
            )
        else:
            trace.add("cache.miss", reason_codes=list(lookup.reason_codes))

        inner_run = self._inner.run(post, run_id=resolved_run_id)
        events = list(trace.events) + list(inner_run.trace)
        recorded = False
        record_reasons: tuple[str, ...] = ()
        if inner_run.case is not None:
            record_location = inner_run.case.extraction.location
            record_result = self._cache.record(
                case=inner_run.case,
                rule_triage=prefilter.triage,
                rule_verification=prefilter.verification,
                run_id=resolved_run_id,
                location_entities=(record_location,)
                if record_location
                else (prefilter.extraction.location,)
                if prefilter.extraction.location
                else (),
            )
            recorded = record_result.stored
            record_reasons = record_result.reason_codes
            events.append(
                TraceEvent(
                    event="cache.recorded",
                    data={
                        "run_id": resolved_run_id,
                        "stored": recorded,
                        "reason_codes": list(record_reasons),
                    },
                )
            )

        provenance = provenance.model_copy(
            update={
                "recorded": recorded,
                "reason_codes": tuple(
                    dict.fromkeys(provenance.reason_codes + record_reasons)
                ),
            }
        )
        return inner_run.model_copy(
            update={"trace": tuple(events), "cache": provenance}
        )

    def _prefilter(
        self, post: SocialPost, trace: TraceRecorder
    ) -> _PrefilterResult | None:
        triage_result = self._run_rule(
            self._rule_triage,
            AgentTask(
                role="triage",
                source_text=post.text,
                source_media=post.media_urls,
                context={"allowed_categories": ["ROAD"], "city": "Bandung"},
            ),
            trace,
        )
        extraction_result = self._run_rule(
            self._rule_extraction,
            AgentTask(
                role="extraction",
                source_text=post.text,
                source_media=post.media_urls,
                context={"city": "Bandung"},
            ),
            trace,
        )
        if triage_result is None or extraction_result is None:
            return None
        try:
            triage = TriageOutput.model_validate(triage_result.output)
            extraction = ExtractionOutput.model_validate(extraction_result.output)
        except Exception:
            trace.add("prefilter.invalid_output")
            return None

        verifier_result = self._run_rule(
            self._rule_verifier,
            AgentTask(
                role="verifier",
                source_text=post.text,
                source_media=post.media_urls,
                context={
                    "extraction": extraction.model_dump(mode="json"),
                    "allowed_categories": ["ROAD"],
                },
            ),
            trace,
        )
        if verifier_result is None:
            return None
        try:
            verification = VerificationOutput.model_validate(verifier_result.output)
        except Exception:
            trace.add("prefilter.invalid_output", role="verifier")
            return None

        latency_ms = sum(
            result.usage.latency_ms
            for result in (triage_result, extraction_result, verifier_result)
        )
        return _PrefilterResult(
            triage=triage,
            extraction=extraction,
            verification=verification,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _run_rule(
        agent: Agent, task: AgentTask, trace: TraceRecorder
    ) -> AgentResult | None:
        try:
            result = agent.run(task)
        except Exception as exc:
            trace.add(
                "prefilter.error",
                role=task.role,
                error_type=type(exc).__name__,
            )
            return None
        trace.add(
            "prefilter.completed",
            role=task.role,
            status=result.status,
            latency_ms=result.usage.latency_ms,
        )
        return result if result.status == "OK" else None

    def _reuse_decision(
        self,
        *,
        post: SocialPost,
        entry: CacheEntry,
        source_intent: str,
        location_quality: str,
        rule_triage: TriageOutput,
        rule_extraction: ExtractionOutput,
        rule_verification: VerificationOutput,
        match_type: str,
    ) -> CacheReuseDecision:
        reasons: list[str] = []
        if (
            source_intent != "CITIZEN_COMPLAINT"
            or entry.source_intent != "CITIZEN_COMPLAINT"
        ):
            reasons.append("CACHE_SOURCE_INTENT_MISMATCH")
        if not rule_triage.is_complaint or not rule_verification.is_complaint:
            reasons.append("CACHE_RULE_NOT_COMPLAINT")
        if (
            rule_triage.category != entry.category
            or rule_verification.category != entry.category
        ):
            reasons.append("CACHE_CATEGORY_MISMATCH")
        if location_quality != "SPECIFIC":
            reasons.append("CACHE_LOCATION_NOT_SPECIFIC")
        if not locations_overlap(rule_extraction.location, entry.location_entities):
            reasons.append("CACHE_LOCATION_MISMATCH")
        if not rule_verification.evidence_supported:
            reasons.append("CACHE_EVIDENCE_UNSUPPORTED")
        if match_type == "EXACT" and (
            not entry.evidence_spans
            or not all(span in post.text for span in entry.evidence_spans)
        ):
            reasons.append("CACHE_EVIDENCE_MISMATCH")
        if match_type == "SEMANTIC" and not rule_extraction.evidence_spans:
            reasons.append("CACHE_EVIDENCE_MISMATCH")
        return CacheReuseDecision(
            allowed=not reasons,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _source_case_id(entry: CacheEntry | None) -> str | None:
        if entry is None:
            return None
        value = entry.record_guards.get("source_case_id")
        return str(value) if value else None

    @staticmethod
    def _cached_case(
        *,
        post: SocialPost,
        entry: CacheEntry,
        source_intent: str,
        location_quality: str,
        rule_triage: TriageOutput,
        rule_extraction: ExtractionOutput,
        rule_verification: VerificationOutput,
        match_type: str,
    ) -> ComplaintCase:
        cached = entry.analysis
        triage = TriageOutput(
            is_complaint=True,
            category=cached.triage.category,
            risk=_max_risk(cached.triage.risk, rule_triage.risk),
            confidence=cached.triage.confidence,
            reason_codes=tuple(
                dict.fromkeys(cached.triage.reason_codes + ("CACHE_REUSED",))
            ),
        )
        return ComplaintCase(
            case_id=deterministic_case_id(post),
            post=post,
            triage=triage,
            extraction=rule_extraction,
            verification=rule_verification,
            arbitration=None,
            location_quality=location_quality,  # type: ignore[arg-type]
            source_intent=source_intent,  # type: ignore[arg-type]
            trust_decision="AGREED",
            status="ACCEPTED",
            reason_codes=(
                "CACHE_REUSED",
                f"CACHE_{match_type}",
                "CACHE_GUARDS_PASSED",
            ),
        )
