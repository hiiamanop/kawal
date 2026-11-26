from __future__ import annotations

from pathlib import Path
import shutil
import threading

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.harness.sources import SourceError
from services.monitor.config import DEFAULT_DB_PATH
from services.monitor.forwarding import write_forward_packet
from services.monitor.models import (
    AnalysisStatus,
    CaseUpdate,
    MonitorCase,
    OperatorStatus,
    ScanRequest,
)
from services.monitor.scan import run_scan
from services.monitor.store import InvalidTransition, MonitorStore


class ForwardRequest(BaseModel):
    note: str = ""


def _case_payload(case: MonitorCase) -> dict:
    return case.model_dump(mode="json")


def create_monitor_app(db_path: str | Path | None = None) -> FastAPI:
    resolved_db = Path(db_path) if db_path is not None else Path(DEFAULT_DB_PATH)
    store = MonitorStore(resolved_db)
    store.create_schema()
    forward_dir = resolved_db.parent / "forwards"

    app = FastAPI(title="KAWAL Monitor")
    app.state.monitor_store = store
    app.state.scan_lock = threading.Lock()
    app.state.forward_dir = forward_dir

    package_dir = Path(__file__).parent
    app.mount(
        "/static",
        StaticFiles(directory=package_dir / "static"),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((package_dir / "templates" / "index.html").read_text())

    @app.get("/api/monitor/cases")
    def list_cases(
        status: OperatorStatus | None = Query(default=None),
        trust: str | None = Query(default=None),
        category: str | None = Query(default=None),
        query: str | None = Query(default=None),
        analysis_status: AnalysisStatus | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict:
        cases = store.list_cases(
            status=status,
            trust=trust,
            category=category,
            query=query,
            analysis_status=analysis_status,
            limit=limit,
        )
        return {"cases": [_case_payload(case) for case in cases]}

    @app.get("/api/monitor/cases/{case_id}")
    def case_detail(case_id: str) -> dict:
        case = store.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found")
        return {
            "case": _case_payload(case),
            "actions": store.list_actions(case_id),
            "queries": store.list_case_queries(case_id),
        }

    @app.patch("/api/monitor/cases/{case_id}")
    def update_case(case_id: str, update: CaseUpdate) -> dict:
        try:
            case = store.update_case(case_id, update)
        except KeyError:
            raise HTTPException(status_code=404, detail="Case not found") from None
        except InvalidTransition as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "INVALID_TRANSITION",
                    "message": str(exc),
                },
            ) from None
        return {"case": _case_payload(case)}

    @app.post("/api/monitor/scan")
    def scan(request: ScanRequest) -> dict:
        if not app.state.scan_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "SCAN_IN_PROGRESS",
                    "message": "A scan is already running",
                },
            )
        try:
            result = run_scan(request, store)
        except SourceError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": exc.code, "message": str(exc)},
            ) from None
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_SCAN", "message": str(exc)},
            ) from None
        finally:
            app.state.scan_lock.release()
        return result.model_dump(mode="json")

    @app.post("/api/monitor/cases/{case_id}/forward")
    def forward(case_id: str, request: ForwardRequest) -> dict:
        case = store.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found")
        try:
            updated = store.update_case(case_id, CaseUpdate(status="READY_TO_FORWARD"))
        except InvalidTransition as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "INVALID_TRANSITION", "message": str(exc)},
            ) from None
        packet = write_forward_packet(updated, request.note, app.state.forward_dir)
        return {
            **packet.model_dump(mode="json"),
            "case": _case_payload(updated),
        }

    @app.get("/api/monitor/stats")
    def stats() -> dict:
        return store.stats().model_dump(mode="json")

    @app.get("/api/monitor/health")
    def health() -> dict:
        try:
            store.stats()
            database = "ok"
        except Exception:
            database = "error"
        return {
            "app": "ok",
            "database": database,
            "agent_reach": "available" if shutil.which("agent-reach") else "missing",
        }

    return app
