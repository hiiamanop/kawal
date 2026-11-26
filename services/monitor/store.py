from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from contracts.models import Category
from services.monitor.models import (
    AnalysisStatus,
    CaseUpdate,
    MonitorCase,
    MonitorStats,
    OperatorStatus,
    ScanResult,
)


class InvalidTransition(ValueError):
    def __init__(self, case_id: str, old: str, new: str) -> None:
        super().__init__(
            f"Invalid monitor status transition for {case_id}: {old} -> {new}"
        )
        self.case_id = case_id
        self.old = old
        self.new = new


_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "NEW": {"MONITORING", "READY_TO_FORWARD", "DISMISSED"},
    "MONITORING": {"READY_TO_FORWARD", "DISMISSED"},
    "READY_TO_FORWARD": {"FORWARDED", "MONITORING", "DISMISSED"},
    "FORWARDED": set(),
    "DISMISSED": set(),
}


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


class MonitorStore:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def create_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS monitor_cases (
                    case_id TEXT PRIMARY KEY,
                    post_json TEXT NOT NULL,
                    run_json TEXT NOT NULL,
                    case_json TEXT,
                    viral_score INTEGER,
                    department TEXT,
                    category TEXT,
                    trust_status TEXT,
                    operator_status TEXT NOT NULL,
                    analysis_status TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS monitor_case_queries (
                    case_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    seen_at TEXT NOT NULL,
                    PRIMARY KEY (case_id, query)
                );
                CREATE TABLE IF NOT EXISTS monitor_scans (
                    scan_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    source_mode TEXT NOT NULL,
                    analysis_mode TEXT NOT NULL,
                    new_cases INTEGER NOT NULL,
                    updated_cases INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    errors_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS monitor_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    old_status TEXT NOT NULL,
                    new_status TEXT NOT NULL,
                    note TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    at TEXT NOT NULL
                );
                """
            )

    def upsert_case(self, case: MonitorCase, query: str) -> bool:
        case_json = (
            json.dumps(case.run.case.model_dump(mode="json"), ensure_ascii=False)
            if case.run.case is not None
            else None
        )
        post_json = json.dumps(case.post.model_dump(mode="json"), ensure_ascii=False)
        run_json = json.dumps(case.run.model_dump(mode="json"), ensure_ascii=False)
        category = (
            case.run.case.triage.category.value if case.run.case is not None else None
        )
        trust = case.run.case.status if case.run.case is not None else None
        seen_at = _iso(case.last_seen_at)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT operator_status, notes, first_seen_at FROM monitor_cases WHERE case_id = ?",
                (case.case_id,),
            ).fetchone()
            created = existing is None
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO monitor_cases (
                        case_id, post_json, run_json, case_json, viral_score,
                        department, category, trust_status, operator_status,
                        analysis_status, notes, first_seen_at, last_seen_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case.case_id,
                        post_json,
                        run_json,
                        case_json,
                        case.viral_score,
                        case.department,
                        category,
                        trust,
                        case.operator_status,
                        case.analysis_status,
                        case.notes,
                        _iso(case.first_seen_at),
                        seen_at,
                        _iso(case.updated_at),
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE monitor_cases
                    SET post_json = ?, run_json = ?, case_json = ?, viral_score = ?,
                        department = ?, category = ?, trust_status = ?,
                        analysis_status = ?, last_seen_at = ?, updated_at = ?
                    WHERE case_id = ?
                    """,
                    (
                        post_json,
                        run_json,
                        case_json,
                        case.viral_score,
                        case.department,
                        category,
                        trust,
                        case.analysis_status,
                        seen_at,
                        _iso(case.updated_at),
                        case.case_id,
                    ),
                )
            connection.execute(
                """
                INSERT INTO monitor_case_queries (case_id, query, seen_at)
                VALUES (?, ?, ?)
                ON CONFLICT(case_id, query) DO UPDATE SET seen_at = excluded.seen_at
                """,
                (case.case_id, query, seen_at),
            )
            connection.commit()
        return created

    def get_case(self, case_id: str) -> MonitorCase | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM monitor_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        return self._row_to_case(row) if row is not None else None

    def list_cases(
        self,
        *,
        status: OperatorStatus | None = None,
        trust: str | None = None,
        category: Category | str | None = None,
        query: str | None = None,
        analysis_status: AnalysisStatus | None = None,
        limit: int = 100,
    ) -> list[MonitorCase]:
        clauses: list[str] = []
        params: list[Any] = []
        if analysis_status is None:
            clauses.append("analysis_status = 'OK'")
            if trust is None:
                clauses.append("trust_status IN ('ACCEPTED', 'NEEDS_REVIEW')")
            if status is None:
                clauses.append("operator_status != 'DISMISSED'")
        if analysis_status is not None:
            clauses.append("analysis_status = ?")
            params.append(analysis_status)
        if trust is not None:
            clauses.append("trust_status = ?")
            params.append(trust)
        if status is not None:
            clauses.append("operator_status = ?")
            params.append(status)
        if category is not None:
            clauses.append("category = ?")
            params.append(getattr(category, "value", category))
        if query is not None:
            clauses.append(
                "case_id IN (SELECT case_id FROM monitor_case_queries WHERE query = ?)"
            )
            params.append(query)
        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM monitor_cases WHERE {where} "
                "ORDER BY viral_score IS NULL, viral_score DESC, last_seen_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._row_to_case(row) for row in rows]

    def update_case(self, case_id: str, update: CaseUpdate) -> MonitorCase:
        current = self.get_case(case_id)
        if current is None:
            raise KeyError(case_id)
        new_status = update.status or current.operator_status
        new_note = current.notes if update.note is None else update.note
        if new_status != current.operator_status:
            allowed = _ALLOWED_TRANSITIONS.get(current.operator_status, set())
            if new_status not in allowed:
                raise InvalidTransition(case_id, current.operator_status, new_status)
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE monitor_cases
                SET operator_status = ?, notes = ?, updated_at = ?
                WHERE case_id = ?
                """,
                (new_status, new_note, _iso(now), case_id),
            )
            if new_status != current.operator_status:
                connection.execute(
                    """
                    INSERT INTO monitor_actions (
                        case_id, old_status, new_status, note, actor, at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case_id,
                        current.operator_status,
                        new_status,
                        new_note,
                        "local-operator",
                        _iso(now),
                    ),
                )
            connection.commit()
        updated = self.get_case(case_id)
        assert updated is not None
        return updated

    def record_scan(self, result: ScanResult) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO monitor_scans (
                    scan_id, query, source_mode, analysis_mode, new_cases,
                    updated_cases, failed, errors_json, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.scan_id,
                    result.query,
                    result.source_mode,
                    result.analysis_mode,
                    result.new_cases,
                    result.updated_cases,
                    result.failed,
                    json.dumps(list(result.errors), ensure_ascii=False),
                    _iso(result.started_at),
                    _iso(result.finished_at),
                ),
            )
            connection.commit()

    def stats(self) -> MonitorStats:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN trust_status IN ('ACCEPTED', 'NEEDS_REVIEW') THEN 1 ELSE 0 END) AS complaints,
                    SUM(CASE WHEN trust_status = 'NEEDS_REVIEW' THEN 1 ELSE 0 END) AS needs_review,
                    SUM(CASE WHEN operator_status = 'READY_TO_FORWARD' THEN 1 ELSE 0 END) AS ready_to_forward,
                    SUM(CASE WHEN operator_status = 'FORWARDED' THEN 1 ELSE 0 END) AS forwarded
                FROM monitor_cases
                WHERE analysis_status = 'OK'
                """
            ).fetchone()
            last = connection.execute(
                "SELECT MAX(finished_at) AS finished_at FROM monitor_scans"
            ).fetchone()
        return MonitorStats(
            total=int(row["total"] or 0),
            complaints=int(row["complaints"] or 0),
            needs_review=int(row["needs_review"] or 0),
            ready_to_forward=int(row["ready_to_forward"] or 0),
            forwarded=int(row["forwarded"] or 0),
            last_scan_at=_parse(last["finished_at"]) if last else None,
        )

    def list_actions(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM monitor_actions WHERE case_id = ? ORDER BY id ASC",
                (case_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_case_queries(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM monitor_case_queries WHERE case_id = ? ORDER BY seen_at ASC",
                (case_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _row_to_case(row: sqlite3.Row) -> MonitorCase:
        from contracts.harness import HarnessRun, SocialPost

        return MonitorCase(
            case_id=row["case_id"],
            post=SocialPost.model_validate(json.loads(row["post_json"])),
            run=HarnessRun.model_validate(json.loads(row["run_json"])),
            viral_score=row["viral_score"],
            department=row["department"],
            operator_status=row["operator_status"],
            analysis_status=row["analysis_status"],
            notes=row["notes"],
            first_seen_at=_parse(row["first_seen_at"]),
            last_seen_at=_parse(row["last_seen_at"]),
            updated_at=_parse(row["updated_at"]),
        )
