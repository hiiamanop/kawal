from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from contracts.harness import CostReport, HarnessRun, deterministic_case_id
from services.harness.agents import (
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.llm import (
    LLMArbiterAgent,
    LLMExtractionAgent,
    LLMTriageAgent,
    LLMVerifierAgent,
    OpenRouterChatClient,
)
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.ports import RunOrchestrator, SourcePort
from services.harness.sources import AgentReachXSource, ReplayXSource
from services.llm_config import openrouter_api_key
from services.monitor.models import AnalysisStatus, MonitorCase, ScanRequest, ScanResult
from services.monitor.routing import recommend_department
from services.monitor.scoring import viral_score
from services.monitor.store import MonitorStore


_ANALYSIS_STATUS: dict[str, AnalysisStatus] = {
    "COMPLETED": "OK",
    "NEEDS_REVIEW": "OK",
    "REJECTED": "OK",
    "FAILED": "FAILED",
    "BUDGET_EXCEEDED": "BUDGET_EXCEEDED",
}


def build_source(request: ScanRequest) -> SourcePort:
    if request.source_mode == "live":
        return AgentReachXSource()
    if not request.fixture_dir:
        raise ValueError("fixture_dir is required for replay scans")
    return ReplayXSource(request.fixture_dir)


def build_orchestrator(analysis_mode: str) -> HarnessOrchestrator:
    if analysis_mode == "rule":
        return HarnessOrchestrator(
            triage=RuleTriageAgent(),
            extraction=RuleExtractionAgent(),
            verifier=RuleVerifierAgent(),
            arbiter=RuleArbiterAgent(),
        )
    if analysis_mode == "llm":
        if not openrouter_api_key():
            raise ValueError("OPENROUTER_API_KEY is required for LLM scans")
        client = OpenRouterChatClient()
        return HarnessOrchestrator(
            triage=LLMTriageAgent(client),
            extraction=LLMExtractionAgent(client),
            verifier=LLMVerifierAgent(client),
            arbiter=LLMArbiterAgent(client),
        )
    raise ValueError(f"unsupported analysis_mode: {analysis_mode}")


def run_scan(
    request: ScanRequest,
    store: MonitorStore,
    *,
    source: SourcePort | None = None,
    orchestrator: RunOrchestrator | None = None,
) -> ScanResult:
    started_at = datetime.now(timezone.utc)
    scan_id = uuid4().hex
    selected_source = source or build_source(request)
    selected_orchestrator = orchestrator or build_orchestrator(request.analysis_mode)
    posts = selected_source.fetch(request.query, request.limit)

    cases: list[MonitorCase] = []
    errors: list[str] = []
    new_cases = 0
    updated_cases = 0
    failed = 0

    for post in posts:
        existing = store.get_case(deterministic_case_id(post))
        if existing is not None and existing.analysis_status == "OK":
            now = datetime.now(timezone.utc)
            refreshed = existing.model_copy(
                update={
                    "post": post,
                    "viral_score": viral_score(post, existing.run.case, now),
                    "last_seen_at": now,
                    "updated_at": now,
                }
            )
            store.upsert_case(refreshed, request.query)
            updated_cases += 1
            cases.append(refreshed)
            continue

        error_reason: str | None = None
        try:
            run = selected_orchestrator.run(post)
        except Exception as exc:
            error_reason = type(exc).__name__
            run = HarnessRun(
                run_id=f"failed-{uuid4().hex[:8]}",
                post=post,
                status="FAILED",
                cost=CostReport(),
            )

        if run.case is None:
            if run.status == "BUDGET_EXCEEDED":
                analysis_status = "BUDGET_EXCEEDED"
            else:
                analysis_status = "FAILED"
                error_reason = error_reason or "ANALYSIS_INCOMPLETE"
        else:
            analysis_status = _ANALYSIS_STATUS.get(run.status, "FAILED")

        if analysis_status != "OK":
            failed += 1
            errors.append(f"{post.post_id}: {error_reason or run.status}")

        now = datetime.now(timezone.utc)
        if run.case is None:
            case = MonitorCase(
                case_id=deterministic_case_id(post),
                post=post,
                run=run,
                viral_score=None,
                department=None,
                analysis_status=analysis_status,
                first_seen_at=now,
                last_seen_at=now,
                updated_at=now,
            )
        else:
            case = MonitorCase(
                case_id=run.case.case_id,
                post=post,
                run=run,
                viral_score=viral_score(post, run.case, now),
                department=recommend_department(run.case),
                analysis_status=analysis_status,
                first_seen_at=now,
                last_seen_at=now,
                updated_at=now,
            )

        created = store.upsert_case(case, request.query)
        if created:
            new_cases += 1
        else:
            updated_cases += 1
        cases.append(case)

    result = ScanResult(
        scan_id=scan_id,
        query=request.query,
        source_mode=request.source_mode,
        analysis_mode=request.analysis_mode,
        new_cases=new_cases,
        updated_cases=updated_cases,
        failed=failed,
        cases=tuple(cases),
        errors=tuple(errors),
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
    )
    store.record_scan(result)
    return result
