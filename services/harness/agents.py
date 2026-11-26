from __future__ import annotations

import re
import time
from typing import Any, Mapping

from contracts.harness import (
    AgentResult,
    AgentUsage,
    ArbitrationOutput,
    ExtractionOutput,
    TriageOutput,
    VerificationOutput,
)
from contracts.models import Category, RiskLevel
from services.harness.category_lexicon import match_category, match_object
from services.harness.ports import AgentTask


def _usage(model: str, started: float) -> AgentUsage:
    return AgentUsage(
        model=model,
        provider="local",
        cost_source="local",
        cost_usd=0.0,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def _result(
    agent_id: str,
    role: str,
    output: Any,
    model: str,
    started: float,
) -> AgentResult:
    return AgentResult(
        agent_id=agent_id,
        role=role,
        status="OK",
        output=output.model_dump(mode="json"),
        usage=_usage(model, started),
        prompt_version="rule-v1",
    )


class RuleTriageAgent:
    agent_id = "rule-triage-v1"
    role = "triage"

    _COMPLAINT_CUES = (
        "lapor",
        "aduan",
        "keluhan",
        "tolong",
        "mohon",
        "rusak",
        "lubang",
        "berlubang",
        "banjir",
        "sampah",
        "macet",
        "bahaya",
        "bocor",
        "mati",
        "mampet",
        "menumpuk",
        "kekurangan",
        "terhambat",
        "belum",
        "antrean",
        "mahal",
        "kotor",
        "terlantar",
        "tumbang",
        "pungli",
        "parkir liar",
    )
    _ROAD_CUES = ("jalan", "jl", "lubang", "aspal", "trotoar", "berlubang")
    _WASTE_CUES = ("sampah", "limbah", "tps")
    _DRAINAGE_CUES = ("banjir", "drainase", "gorong", "selokan", "tergenang")
    _HAZARD_CUES = ("bahaya", "parah", "korban", "kecelakaan", "besar", "dalam")

    def estimate_cost(self, task: AgentTask) -> float:
        del task
        return 0.0

    def run(self, task: AgentTask) -> AgentResult:
        started = time.perf_counter()
        text = task.source_text.strip()
        lower = text.lower()

        complaint_hit = any(cue in lower for cue in self._COMPLAINT_CUES)
        is_complaint = complaint_hit and bool(text)
        if not text:
            is_complaint = False

        category, ambiguous = match_category(lower)
        category_unknown = category is None
        if category is None:
            category = Category.ROAD

        hazard_hit = any(cue in lower for cue in self._HAZARD_CUES)
        risk = RiskLevel.HIGH if hazard_hit and is_complaint else RiskLevel.MEDIUM
        confidence = (
            0.55
            if is_complaint and (category_unknown or ambiguous)
            else 0.9
            if is_complaint
            else 0.2
        )

        reason_codes: list[str] = []
        if is_complaint:
            reason_codes.append("COMPLAINT_CUE")
        if category_unknown:
            reason_codes.append("CATEGORY_UNKNOWN")
        if ambiguous:
            reason_codes.append("AMBIGUOUS_CATEGORY")
        if category == Category.ROAD:
            reason_codes.append("ROAD_DAMAGE")
        if hazard_hit:
            reason_codes.append("HAZARD_CUE")
        if not is_complaint:
            reason_codes.append("NO_COMPLAINT_CUE")

        output = TriageOutput(
            is_complaint=is_complaint,
            category=category,
            risk=risk,
            confidence=confidence,
            reason_codes=tuple(reason_codes),
        )
        return _result(self.agent_id, self.role, output, "rule-triage", started)


class RuleExtractionAgent:
    agent_id = "rule-extraction-v1"
    role = "extraction"

    _STREET_PATTERN = re.compile(
        r"\b(?:Jl\.?|Jalan)\s+(?!(?:di|ke|yang|itu|ini)\b)"
        r"([A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+){0,1})",
        re.IGNORECASE,
    )
    _LOCATION_LABEL_PATTERN = re.compile(
        r"\b(?:Kelurahan|Kecamatan|Kota)\s+([A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+){0,1})",
        re.IGNORECASE,
    )
    _OBJECT_CUES = ("lubang", "aspal", "trotoar", "jalan rusak", "sampah", "banjir")
    _TIME_CUES = ("kemarin", "pagi", "siang", "sore", "malam", "minggu", "bulan")
    _STREET_STOPWORDS = {
        "rusak",
        "parah",
        "ada",
        "lubang",
        "besar",
        "tolong",
        "bandung",
        "kota",
        "min",
    }

    def estimate_cost(self, task: AgentTask) -> float:
        del task
        return 0.0

    def run(self, task: AgentTask) -> AgentResult:
        started = time.perf_counter()
        text = task.source_text
        lower = text.lower()
        evidence: list[str] = []

        street_match = self._STREET_PATTERN.search(text)
        location: str | None = None
        if street_match is not None:
            words = street_match.group(1).split()
            while words and words[-1].lower() in self._STREET_STOPWORDS:
                words.pop()
            if words:
                location = " ".join(words)
                evidence.append(street_match.group(0))

        if location is None:
            label_match = self._LOCATION_LABEL_PATTERN.search(text)
            if label_match is not None:
                location = label_match.group(0)
                evidence.append(label_match.group(0))

        object_value, _ = match_object(lower)
        if object_value is not None:
            evidence.append(object_value)

        time_value = next((cue for cue in self._TIME_CUES if cue in lower), None)
        if time_value is not None:
            evidence.append(time_value)

        confidence = 0.85 if location and object_value else 0.6 if object_value else 0.2
        output = ExtractionOutput(
            location=location,
            object=object_value,
            time=time_value,
            evidence_spans=tuple(dict.fromkeys(evidence)),
            confidence=confidence,
        )
        return _result(self.agent_id, self.role, output, "rule-extraction", started)


class RuleVerifierAgent:
    agent_id = "rule-verifier-v1"
    role = "verifier"

    _COMPLAINT_VERBS = ("lapor", "aduan", "keluhan", "tolong", "mohon")
    _DAMAGE_CUES = (
        "rusak",
        "lubang",
        "berlubang",
        "ambles",
        "hancur",
        "banjir",
        "sampah",
        "bocor",
        "mati",
        "mampet",
        "menumpuk",
        "kekurangan",
        "terhambat",
        "belum",
        "antrean",
        "mahal",
        "kotor",
        "terlantar",
        "tumbang",
        "pungli",
        "parkir liar",
    )

    def estimate_cost(self, task: AgentTask) -> float:
        del task
        return 0.0

    def run(self, task: AgentTask) -> AgentResult:
        started = time.perf_counter()
        text = task.source_text
        lower = text.lower()
        extraction_data = task.context.get("extraction")
        extraction = (
            ExtractionOutput.model_validate(extraction_data)
            if isinstance(extraction_data, Mapping)
            else ExtractionOutput(confidence=0.0)
        )

        has_verb = any(cue in lower for cue in self._COMPLAINT_VERBS)
        has_damage = any(cue in lower for cue in self._DAMAGE_CUES)
        is_complaint = bool(text.strip()) and (has_verb or has_damage)

        category, ambiguous = match_category(lower)
        _, object_category = match_object(lower)

        issues: list[str] = []
        if category is None:
            category = object_category or Category.ROAD
            issues.append("CATEGORY_UNKNOWN")
        elif object_category is not None and object_category != category:
            category = object_category
            issues.append("CATEGORY_CONFLICT")
        if ambiguous:
            issues.append("AMBIGUOUS_CATEGORY")

        evidence_supported = True
        if not extraction.evidence_spans:
            evidence_supported = False
            issues.append("NO_EVIDENCE_SPANS")
        for span in extraction.evidence_spans:
            if span and span not in text:
                evidence_supported = False
                issues.append("EVIDENCE_SPAN_NOT_IN_SOURCE")
                break
        if extraction.location is not None and extraction.location not in text:
            evidence_supported = False
            issues.append("LOCATION_NOT_IN_SOURCE")
        if extraction.object is not None and extraction.object not in text:
            evidence_supported = False
            issues.append("OBJECT_NOT_IN_SOURCE")

        confidence = 0.85 if is_complaint and evidence_supported else 0.5 if is_complaint else 0.2
        output = VerificationOutput(
            is_complaint=is_complaint,
            category=category,
            evidence_supported=evidence_supported,
            issues=tuple(dict.fromkeys(issues)),
            confidence=confidence,
        )
        return _result(self.agent_id, self.role, output, "rule-verifier", started)


class RuleArbiterAgent:
    agent_id = "rule-arbiter-v1"
    role = "arbiter"

    def estimate_cost(self, task: AgentTask) -> float:
        del task
        return 0.0

    def run(self, task: AgentTask) -> AgentResult:
        started = time.perf_counter()
        triage_data = task.context.get("triage")
        verification_data = task.context.get("verification")
        triage = (
            TriageOutput.model_validate(triage_data)
            if isinstance(triage_data, Mapping)
            else None
        )
        verification = (
            VerificationOutput.model_validate(verification_data)
            if isinstance(verification_data, Mapping)
            else None
        )
        category = (
            triage.category
            if triage is not None
            else verification.category
            if verification is not None
            else Category.ROAD
        )
        is_complaint = bool(
            (triage.is_complaint if triage is not None else False)
            or (verification.is_complaint if verification is not None else False)
        )
        output = ArbitrationOutput(
            is_complaint=is_complaint,
            category=category,
            rationale="Triage and Verifier disagree; human review is required.",
            confidence=0.5,
        )
        return _result(self.agent_id, self.role, output, "rule-arbiter", started)


class PricedStubAgent:
    """Deterministic agent with explicit cost, used by budget tests."""

    def __init__(
        self,
        role: str,
        output: Mapping[str, Any] | None = None,
        estimate_usd: float = 0.0,
        cost_usd: float = 0.0,
        status: str = "OK",
    ) -> None:
        self.agent_id = f"priced-stub-{role}"
        self.role = role
        self._output = dict(output) if output is not None else None
        self._estimate_usd = estimate_usd
        self._cost_usd = cost_usd
        self._status = status

    def estimate_cost(self, task: AgentTask) -> float:
        del task
        return self._estimate_usd

    def run(self, task: AgentTask) -> AgentResult:
        started = time.perf_counter()
        usage = AgentUsage(
            model="priced-stub",
            provider="test",
            input_tokens=1,
            output_tokens=1,
            cost_usd=self._cost_usd,
            cost_source="estimated",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        return AgentResult(
            agent_id=self.agent_id,
            role=self.role,
            status=self._status,  # type: ignore[arg-type]
            output=self._output,
            usage=usage,
            prompt_version="stub-v1",
        )
