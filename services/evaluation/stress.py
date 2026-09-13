from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from contracts.models import (
    Category,
    DecisionMode,
    ProcessingState,
    TicketStatus,
)
from services.clarification.dispatcher import ClarificationDispatcher
from services.core.pipeline import CaseProcessingPipeline, PipelineResult
from services.intake.openwa import OpenWAConnector
from services.reliability.client import ReliableTicketClient
from services.simulator.store import TicketSimulator


class LoadProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    target_cases: int
    concurrent_workers: int
    burst_multiplier: float = 1.0


class LatencyProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    min_ms: float


class StressResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_name: str
    total_cases: int
    total_messages: int
    successful_cases: int
    failed_cases: int
    tickets_created: int
    clarifications_sent: int
    rejections: int
    duplicate_tickets: int
    elapsed_seconds: float
    cases_per_second: float
    messages_per_second: float
    latency: LatencyProfile
    decision_distribution: dict[str, int]
    zero_duplicates_invariant: bool
    all_cases_accounted: bool


STANDARD_LOAD_PROFILES: tuple[LoadProfile, ...] = (
    LoadProfile(name="25%", target_cases=25, concurrent_workers=2),
    LoadProfile(name="50%", target_cases=50, concurrent_workers=4),
    LoadProfile(name="75%", target_cases=75, concurrent_workers=6),
    LoadProfile(name="100%", target_cases=100, concurrent_workers=8),
    LoadProfile(name="125%", target_cases=125, concurrent_workers=10),
    LoadProfile(name="burst_2x", target_cases=200, concurrent_workers=16, burst_multiplier=2.0),
)

CATEGORIES = list(Category)
COMPLAINT_TEMPLATES = [
    "Lapor, {issue} di Jl. Merdeka No. {num}, RT {rt} RW {rw}, Kelurahan Babakan Ciamis, Kecamatan Sumur Bandung, Kota Bandung.",
    "Halo petugas, aduan {issue} dekat pasar tetapi belum jelas alamat nomor rumahnya.",
    "Sekadar opini dan salam sapa untuk rekan-rekan dinas.",
]
ISSUE_BY_CATEGORY: dict[Category, str] = {
    Category.ROAD: "jalan berlubang parah",
    Category.DRAINAGE_FLOOD: "saluran drainase tersumbat berat",
    Category.WASTE: "sampah menumpuk tak diangkut",
    Category.CLEAN_WATER: "pipa air keran keruh dan berbau",
    Category.CIVIL_ADMIN: "KTP warga belum selesai di dukcapil",
    Category.HEALTH_SERVICE: "jadwal puskesmas tidak pasti",
    Category.PUBLIC_ORDER: "PKL memblokir trotoar tertib umum",
    Category.TRANSPORTATION: "rambu rambu lalu lintas dishub hilang",
    Category.FIRE_RESCUE: "kebakaran rumah tadi subuh damkar",
    Category.SOCIAL_AFFAIRS: "anak telantar di bawah jembatan bansos dinsos",
    Category.EDUCATION: "sekolah disdik atap bocor",
    Category.PARKS_HOUSING: "taman umum pohon tumbang perumahan",
}


def _generate_case_messages(case_idx: int) -> tuple[str, list[str]]:
    cat = CATEGORIES[case_idx % len(CATEGORIES)]
    issue = ISSUE_BY_CATEGORY[cat]
    profile = case_idx % 3

    if profile == 0:
        num = (case_idx % 50) + 1
        rt = (case_idx % 20) + 1
        rw = (case_idx % 10) + 1
        text = COMPLAINT_TEMPLATES[0].format(issue=issue, num=num, rt=f"{rt:02}", rw=f"{rw:02}")
        return f"case-stress-{case_idx:05d}", [text]
    elif profile == 1:
        text = COMPLAINT_TEMPLATES[1].format(issue=issue)
        return f"case-stress-{case_idx:05d}", [text]
    else:
        return f"case-stress-{case_idx:05d}", [COMPLAINT_TEMPLATES[2]]


def run_stress_profile(
    profile: LoadProfile,
    pipeline: CaseProcessingPipeline | None = None,
) -> StressResult:
    simulator = TicketSimulator()
    connector = OpenWAConnector()
    pipe = pipeline or CaseProcessingPipeline(
        ticket_client=ReliableTicketClient(simulator=simulator),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=connector),
        default_jurisdiction_id="JUR-FICT-01",
    )

    latencies: list[float] = []
    results: list[PipelineResult] = []
    errors: list[Exception] = []
    lock = Lock()
    total_msgs = 0

    def process_case(case_idx: int) -> None:
        nonlocal total_msgs
        case_id, messages = _generate_case_messages(case_idx)
        start = time.perf_counter()
        try:
            result = pipe.process_messages(
                tenant_id="tenant-stress",
                conversation_id=f"stress-conv-{case_idx}@c.us",
                messages=messages,
                case_id=case_id,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            with lock:
                latencies.append(elapsed)
                results.append(result)
                total_msgs += len(messages)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            with lock:
                latencies.append(elapsed)
                errors.append(exc)

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=profile.concurrent_workers) as executor:
        futures = [
            executor.submit(process_case, i)
            for i in range(profile.target_cases)
        ]
        for future in as_completed(futures):
            future.result()

    wall_elapsed = time.perf_counter() - wall_start

    lat_arr = np.array(latencies) if latencies else np.array([0.0])
    decision_dist: dict[str, int] = {}
    tickets_created = 0
    clarifications_sent = 0
    rejections = 0
    for r in results:
        mode = r.decision_mode.value
        decision_dist[mode] = decision_dist.get(mode, 0) + 1
        if r.processing_state == ProcessingState.TICKETED:
            tickets_created += 1
        elif r.processing_state == ProcessingState.WAITING_CLARIFICATION:
            clarifications_sent += 1
        elif r.processing_state == ProcessingState.REJECTED:
            rejections += 1

    successful = len(results)
    failed = len(errors)
    all_accounted = (successful + failed) == profile.target_cases

    ticket_ids = [r.ticket_receipt.ticket_id for r in results if r.ticket_receipt is not None]
    unique_tickets = len(set(ticket_ids))
    duplicate_tickets = len(ticket_ids) - unique_tickets

    return StressResult(
        profile_name=profile.name,
        total_cases=profile.target_cases,
        total_messages=total_msgs,
        successful_cases=successful,
        failed_cases=failed,
        tickets_created=tickets_created,
        clarifications_sent=clarifications_sent,
        rejections=rejections,
        duplicate_tickets=duplicate_tickets,
        elapsed_seconds=round(wall_elapsed, 4),
        cases_per_second=round(successful / wall_elapsed, 2) if wall_elapsed > 0 else 0.0,
        messages_per_second=round(total_msgs / wall_elapsed, 2) if wall_elapsed > 0 else 0.0,
        latency=LatencyProfile(
            p50_ms=round(float(np.percentile(lat_arr, 50)), 2),
            p95_ms=round(float(np.percentile(lat_arr, 95)), 2),
            p99_ms=round(float(np.percentile(lat_arr, 99)), 2),
            mean_ms=round(float(np.mean(lat_arr)), 2),
            max_ms=round(float(np.max(lat_arr)), 2),
            min_ms=round(float(np.min(lat_arr)), 2),
        ),
        decision_distribution=decision_dist,
        zero_duplicates_invariant=(duplicate_tickets == 0),
        all_cases_accounted=all_accounted,
    )


def run_full_stress_suite(
    profiles: Sequence[LoadProfile] | None = None,
) -> list[StressResult]:
    active_profiles = profiles or STANDARD_LOAD_PROFILES
    return [run_stress_profile(p) for p in active_profiles]
