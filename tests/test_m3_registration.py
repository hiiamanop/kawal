from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock
import uuid
import pytest

from scripts.evaluate_m3 import M3EvaluationReport
from scripts.register_m3_artifacts import (
    DuplicateModelError,
    INSERT_MODEL_SQL,
    RegistrationConfig,
    RegistrationError,
    RegistrationRecord,
    SELECT_MODEL_SQL,
    UPDATE_MODEL_SQL,
    ValidationError,
    build_arg_parser,
    get_database_connection,
    load_and_validate_eval_report,
    load_and_validate_manifest,
    main,
    register_artifacts,
    sanitize_secret,
)
from services.ml.manifest import (
    ArtifactManifest,
    ArtifactManifestItem,
    compute_file_sha256,
)

try:
    import psycopg
    from psycopg.errors import UniqueViolation
    from psycopg.types.json import Jsonb
except ImportError:
    psycopg = None  # type: ignore[assignment]
    UniqueViolation = Exception  # type: ignore[assignment,misc]
    Jsonb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Live DB Safety Guard Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def guard_against_live_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no test can ever accidentally mutate or connect to a real live database."""
    # Ensure DATABASE_URL is cleared by default unless explicitly set in a mock test
    monkeypatch.delenv("DATABASE_URL", raising=False)


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_workspace() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()


def create_dummy_artifact(
    directory: Path,
    filename: str = "model.onnx",
    content: bytes = b"dummy model weights 12345",
    name: str = "indobert-multitask-fp32",
    version: str = "v1.0.0",
    task: str = "classification",
    precision: str = "fp32",
) -> tuple[Path, ArtifactManifestItem]:
    target_path = directory / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(content)

    sha = compute_file_sha256(target_path)
    size = target_path.stat().st_size
    item = ArtifactManifestItem(
        name=name,
        version=version,
        sha256=sha,
        path=filename,
        size_bytes=size,
        task=task,
        precision=precision,
    )
    return target_path, item


def create_dummy_manifest(
    directory: Path,
    items: list[ArtifactManifestItem],
    filename: str = "manifest.json",
    environment: str = "local-cpu",
) -> Path:
    manifest_dict = {
        "manifest_version": "1.0.0",
        "environment": environment,
        "artifacts": {item.name: item.model_dump() for item in items},
    }
    manifest_path = directory / filename
    manifest_path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    return manifest_path


def create_dummy_eval_report(
    directory: Path,
    filename: str = "evaluation_results.json",
    status: str = "PASSED",
    all_checks_passed: bool = True,
    sla_conformance: bool = True,
    metrics: dict | None = None,
    manifest_audit: dict | None = None,
) -> Path:
    report = M3EvaluationReport(
        evaluation_version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=42,
        status=status,
        split_audit={"passed": True, "total_trajectories": 10},
        benchmark={"sla_met": sla_conformance, "p95_latency_ms": 45.0},
        metrics=metrics
        if metrics is not None
        else {
            "accuracy": 0.94,
            "calibration": {"temperature": 1.15, "calibrated_ece": 0.025},
        },
        manifest_audit=manifest_audit if manifest_audit is not None else {"all_valid": True},
        sla_conformance=sla_conformance,
        all_checks_passed=all_checks_passed,
    )
    report_path = directory / filename
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report_path


class MockDbCursor:
    def __init__(self, existing_rows: dict[tuple[str, str], tuple[uuid.UUID, str]] | None = None) -> None:
        self.existing_rows = existing_rows or {}
        self.executed_queries: list[tuple[str, tuple]] = []
        self._last_result: list | None = None

    def execute(self, query: str, params: tuple = ()) -> None:
        self.executed_queries.append((query.strip(), params))
        norm_query = " ".join(query.split())

        if "SELECT model_id, artifact_hash FROM model_registry" in norm_query:
            name, version = params[0], params[1]
            if (name, version) in self.existing_rows:
                model_id, sha = self.existing_rows[(name, version)]
                self._last_result = [(model_id, sha)]
            else:
                self._last_result = []

        elif "INSERT INTO model_registry" in norm_query:
            name, version, sha = params[0], params[1], params[2]
            new_id = uuid.uuid4()
            now = datetime.now(timezone.utc)
            self.existing_rows[(name, version)] = (new_id, sha)
            self._last_result = [(new_id, now)]

        elif "UPDATE model_registry" in norm_query:
            name = params[7]
            version = params[8]
            sha = params[0]
            if (name, version) in self.existing_rows:
                model_id = self.existing_rows[(name, version)][0]
            else:
                model_id = uuid.uuid4()
            now = datetime.now(timezone.utc)
            self.existing_rows[(name, version)] = (model_id, sha)
            self._last_result = [(model_id, now)]

        else:
            self._last_result = []

    def fetchone(self) -> tuple | None:
        if self._last_result and len(self._last_result) > 0:
            return self._last_result.pop(0)
        return None

    def fetchall(self) -> list:
        res = self._last_result or []
        self._last_result = []
        return res

    def __enter__(self) -> MockDbCursor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class MockDbConnection:
    def __init__(self, cursor: MockDbCursor | None = None) -> None:
        self.cursor_obj = cursor or MockDbCursor()
        self.closed = False
        self.committed = False
        self.rolled_back = False

    def cursor(self) -> MockDbCursor:
        return self.cursor_obj

    def transaction(self) -> MockDbConnection:
        return self

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> MockDbConnection:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()


# ---------------------------------------------------------------------------
# Unit Tests: Manifest and File Validation
# ---------------------------------------------------------------------------

def test_manifest_valid_and_checksum_match(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])

    manifest, base_dir, items = load_and_validate_manifest(manifest_path)
    assert len(items) == 1
    assert items[0].name == "indobert-multitask-fp32"
    assert items[0].sha256 == item.sha256
    assert base_dir == temp_workspace


def test_manifest_missing_file_raises_validation_error(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    # Delete the artifact file
    (temp_workspace / "model.onnx").unlink()

    with pytest.raises(ValidationError, match="file does not exist"):
        load_and_validate_manifest(manifest_path)


def test_manifest_hash_mismatch_raises_validation_error(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    # Corrupt the artifact file
    (temp_workspace / "model.onnx").write_bytes(b"tampered content")

    with pytest.raises(ValidationError, match="SHA256 checksum mismatch"):
        load_and_validate_manifest(manifest_path)


def test_manifest_size_mismatch_raises_validation_error(temp_workspace: Path) -> None:
    model_file, item = create_dummy_artifact(temp_workspace, "model.onnx")
    # Change size_bytes in manifest manually
    bad_item = item.model_copy(update={"size_bytes": item.size_bytes + 50})
    manifest_path = create_dummy_manifest(temp_workspace, [bad_item])

    with pytest.raises(ValidationError, match="Size mismatch"):
        load_and_validate_manifest(manifest_path)


def test_manifest_path_traversal_parent_rejected(temp_workspace: Path) -> None:
    item = ArtifactManifestItem(
        name="traversal-test",
        version="v1.0.0",
        sha256="a" * 64,
        path="subdir/subfile.bin",
        size_bytes=10,
        task="classification",
        precision="fp32",
    )
    manifest_dict = {
        "manifest_version": "1.0.0",
        "environment": "local-cpu",
        "artifacts": {
            "traversal-test": {
                "name": "traversal-test",
                "version": "v1.0.0",
                "sha256": "a" * 64,
                "path": "../secret.bin",
                "size_bytes": 10,
                "task": "classification",
                "precision": "fp32",
            }
        },
    }
    manifest_path = temp_workspace / "bad_manifest.json"
    manifest_path.write_text(json.dumps(manifest_dict), encoding="utf-8")

    with pytest.raises(ValidationError, match="traversal|relative without '\.\.'|Invalid artifact manifest"):
        load_and_validate_manifest(manifest_path)


def test_manifest_empty_artifacts_rejected(temp_workspace: Path) -> None:
    manifest_path = temp_workspace / "empty_manifest.json"
    manifest_path.write_text(json.dumps({"manifest_version": "1.0.0", "artifacts": {}}), encoding="utf-8")

    with pytest.raises(ValidationError, match="Manifest contains no artifacts"):
        load_and_validate_manifest(manifest_path)


def test_manifest_filter_specific_artifact(temp_workspace: Path) -> None:
    _, item1 = create_dummy_artifact(temp_workspace, "m1.onnx", name="model-1")
    _, item2 = create_dummy_artifact(temp_workspace, "m2.onnx", name="model-2")
    manifest_path = create_dummy_manifest(temp_workspace, [item1, item2])

    _, _, items = load_and_validate_manifest(manifest_path, artifact_name="model-2")
    assert len(items) == 1
    assert items[0].name == "model-2"

    with pytest.raises(ValidationError, match="not found in manifest"):
        load_and_validate_manifest(manifest_path, artifact_name="unknown-model")


# ---------------------------------------------------------------------------
# Unit Tests: Evaluation Report Validation
# ---------------------------------------------------------------------------

def test_evaluation_report_valid_passed(temp_workspace: Path) -> None:
    eval_path = create_dummy_eval_report(temp_workspace, status="PASSED", all_checks_passed=True)
    report = load_and_validate_eval_report(eval_path)
    assert report.status == "PASSED"
    assert report.all_checks_passed is True


def test_evaluation_report_failed_status_rejected_unless_allowed(temp_workspace: Path) -> None:
    eval_path = create_dummy_eval_report(temp_workspace, status="FAILED", all_checks_passed=False)

    with pytest.raises(ValidationError, match="status is 'FAILED'"):
        load_and_validate_eval_report(eval_path, allow_failed=False)

    # Allowed with flag
    report = load_and_validate_eval_report(eval_path, allow_failed=True)
    assert report.status == "FAILED"


def test_evaluation_report_sla_failed_rejected_unless_allowed(temp_workspace: Path) -> None:
    eval_path = create_dummy_eval_report(
        temp_workspace, status="PASSED", all_checks_passed=True, sla_conformance=False
    )

    with pytest.raises(ValidationError, match="latency SLA conformance"):
        load_and_validate_eval_report(eval_path, allow_failed=False)

    report = load_and_validate_eval_report(eval_path, allow_failed=True)
    assert report.sla_conformance is False


def test_evaluation_report_manifest_audit_invalid_item_rejected(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx", name="model-a")
    eval_path = create_dummy_eval_report(
        temp_workspace,
        status="PASSED",
        all_checks_passed=True,
        manifest_audit={"all_valid": False, "items_checked": {"model-a": False}},
    )

    with pytest.raises(ValidationError, match="marked artifact 'model-a' as invalid in manifest_audit"):
        load_and_validate_eval_report(eval_path, allow_failed=False, items_to_check=[item])


def test_evaluation_report_missing_file_rejected(temp_workspace: Path) -> None:
    with pytest.raises(ValidationError, match="file not found"):
        load_and_validate_eval_report(temp_workspace / "non_existent.json")


def test_evaluation_report_malformed_json_rejected(temp_workspace: Path) -> None:
    bad_file = temp_workspace / "bad.json"
    bad_file.write_text("{ incomplete json", encoding="utf-8")
    with pytest.raises(ValidationError, match="Invalid evaluation report"):
        load_and_validate_eval_report(bad_file)


# ---------------------------------------------------------------------------
# Unit Tests: DB Connection & Secret Safety
# ---------------------------------------------------------------------------

def test_database_url_derived_only_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure DATABASE_URL is not set -> get_database_connection raises
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RegistrationError, match="DATABASE_URL environment variable is not set"):
        get_database_connection()


def test_cli_parser_has_no_database_url_flag() -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--database-url", "postgresql://user:pass@localhost/db"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--db-url", "postgresql://user:pass@localhost/db"])


def test_database_credentials_never_printed_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
    temp_workspace: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    secret_pw = "SuperSecret_P@ssword_98765"
    sensitive_url = f"postgresql://myuser:{secret_pw}@127.0.0.1:54329/kawal_db"
    monkeypatch.setenv("DATABASE_URL", sensitive_url)

    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    config = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
    )

    with pytest.raises(RegistrationError) as exc_info:
        register_artifacts(config)

    err_str = str(exc_info.value)
    assert secret_pw not in err_str
    assert "myuser" not in err_str or "[REDACTED]" in err_str or "[REDACTED_DATABASE_URL]" in err_str

    # Also test main CLI execution
    exit_code = main(["-m", str(manifest_path), "-e", str(eval_path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert secret_pw not in captured.out
    assert secret_pw not in captured.err


def test_sanitize_secret_utility() -> None:
    raw = "Failed to connect to postgresql://admin:secret123@db.example.com:5432/production_db"
    sanitized = sanitize_secret(raw)
    assert "secret123" not in sanitized
    assert "admin" not in sanitized
    assert "[REDACTED_DATABASE_URL]" in sanitized

    raw2 = "Connection failed: postgres://localhost:5432/mydb"
    sanitized2 = sanitize_secret(raw2)
    assert "[REDACTED_DATABASE_URL]" in sanitized2

    raw3 = "psycopg.OperationalError: password='super_secret_password' failed"
    sanitized3 = sanitize_secret(raw3)
    assert "super_secret_password" not in sanitized3
    assert "password='***'" in sanitized3


# ---------------------------------------------------------------------------
# Unit Tests: Validate-Only & Dry-Run Modes (Zero Live DB Mutation)
# ---------------------------------------------------------------------------

def test_validate_only_mode_offline(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    config = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        validate_only=True,
    )

    records = register_artifacts(config)
    assert len(records) == 1
    assert records[0].status == "VALIDATED"
    assert records[0].name == item.name
    assert records[0].version == item.version


def test_dry_run_without_database_url(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    config = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        dry_run=True,
    )

    records = register_artifacts(config)
    assert len(records) == 1
    assert records[0].status == "DRY_RUN_WOULD_REGISTER"


def test_dry_run_with_mock_db_does_not_mutate(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    mock_cursor = MockDbCursor()
    mock_conn = MockDbConnection(cursor=mock_cursor)

    config = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        dry_run=True,
    )

    records = register_artifacts(config, connection=mock_conn)
    assert len(records) == 1
    assert records[0].status == "DRY_RUN_WOULD_REGISTER"

    # Verify no INSERT or UPDATE queries executed
    for query, _ in mock_cursor.executed_queries:
        assert "INSERT" not in query.upper()
        assert "UPDATE" not in query.upper()


def test_dry_run_detects_existing_model_policies(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    existing_id = uuid.uuid4()
    mock_cursor = MockDbCursor(existing_rows={(item.name, item.version): (existing_id, item.sha256)})
    mock_conn = MockDbConnection(cursor=mock_cursor)

    # 1. Error policy
    cfg_error = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        dry_run=True,
        duplicate_policy="error",
    )
    with pytest.raises(DuplicateModelError, match="Dry-run detected existing model"):
        register_artifacts(cfg_error, connection=mock_conn)

    # 2. Skip policy
    cfg_skip = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        dry_run=True,
        duplicate_policy="skip",
    )
    records_skip = register_artifacts(cfg_skip, connection=mock_conn)
    assert records_skip[0].status == "DRY_RUN_WOULD_SKIP"

    # 3. Update policy
    cfg_update = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        dry_run=True,
        duplicate_policy="update",
    )
    records_update = register_artifacts(cfg_update, connection=mock_conn)
    assert records_update[0].status == "DRY_RUN_WOULD_UPDATE"


# ---------------------------------------------------------------------------
# Unit Tests: Parameterized SQL & Duplicate Protection
# ---------------------------------------------------------------------------

def test_parameterized_psycopg_insert(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx", task="ner", precision="fp32")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    mock_cursor = MockDbCursor()
    mock_conn = MockDbConnection(cursor=mock_cursor)

    config = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        tokenizer="custom/tokenizer-base",
        dataset_split="validation",
    )

    records = register_artifacts(config, connection=mock_conn)
    assert len(records) == 1
    assert records[0].status == "REGISTERED"
    assert records[0].model_id is not None

    # Inspect executed queries and verify parameterization (%s)
    insert_queries = [
        (q, p) for q, p in mock_cursor.executed_queries if "INSERT INTO model_registry" in " ".join(q.split())
    ]
    assert len(insert_queries) == 1
    query_str, params = insert_queries[0]

    # Verify query string contains only %s placeholders, no direct SQL injection
    assert "%s" in query_str
    assert item.name not in query_str
    assert item.sha256 not in query_str

    # Verify parameter values
    assert params[0] == item.name
    assert params[1] == item.version
    assert params[2] == item.sha256.lower()
    assert params[4] == "ner"
    assert params[5] == "custom/tokenizer-base"
    assert params[7] == "validation"


def test_duplicate_policy_error_raises_duplicate_model_error(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    existing_id = uuid.uuid4()
    mock_cursor = MockDbCursor(existing_rows={(item.name, item.version): (existing_id, item.sha256)})
    mock_conn = MockDbConnection(cursor=mock_cursor)

    config = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        duplicate_policy="error",
    )

    with pytest.raises(DuplicateModelError, match="already exists"):
        register_artifacts(config, connection=mock_conn)

    # Verify rollback was called or transaction failed safely
    assert mock_conn.committed is False


def test_duplicate_policy_skip(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    existing_id = uuid.uuid4()
    mock_cursor = MockDbCursor(existing_rows={(item.name, item.version): (existing_id, item.sha256)})
    mock_conn = MockDbConnection(cursor=mock_cursor)

    config = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        duplicate_policy="skip",
    )

    records = register_artifacts(config, connection=mock_conn)
    assert len(records) == 1
    assert records[0].status == "SKIPPED"
    assert records[0].model_id == str(existing_id)

    # Verify no INSERT query was performed
    insert_queries = [
        q for q, _ in mock_cursor.executed_queries if "INSERT INTO model_registry" in " ".join(q.split())
    ]
    assert len(insert_queries) == 0


def test_duplicate_policy_update(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    existing_id = uuid.uuid4()
    mock_cursor = MockDbCursor(existing_rows={(item.name, item.version): (existing_id, "oldhash" * 8)})
    mock_conn = MockDbConnection(cursor=mock_cursor)

    config = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        duplicate_policy="update",
    )

    records = register_artifacts(config, connection=mock_conn)
    assert len(records) == 1
    assert records[0].status == "UPDATED"
    assert records[0].model_id == str(existing_id)

    # Verify UPDATE query was executed with parameters
    update_queries = [
        (q, p) for q, p in mock_cursor.executed_queries if "UPDATE model_registry" in " ".join(q.split())
    ]
    assert len(update_queries) == 1
    _, params = update_queries[0]
    assert params[0] == item.sha256.lower()
    assert params[7] == item.name
    assert params[8] == item.version


def test_concurrent_unique_violation_handling(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    # Cursor that simulates race condition: SELECT says row doesn't exist, but INSERT raises UniqueViolation
    class RaceConditionCursor(MockDbCursor):
        def execute(self, query: str, params: tuple = ()) -> None:
            norm = " ".join(query.split())
            if "INSERT INTO model_registry" in norm:
                raise UniqueViolation("duplicate key value violates unique constraint")
            super().execute(query, params)

    mock_cursor = RaceConditionCursor()
    mock_conn = MockDbConnection(cursor=mock_cursor)

    # Error policy on concurrent duplicate
    cfg_error = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        duplicate_policy="error",
    )
    with pytest.raises(DuplicateModelError, match="Concurrent duplicate detected"):
        register_artifacts(cfg_error, connection=mock_conn)

    # Skip policy on concurrent duplicate
    mock_cursor2 = RaceConditionCursor()
    mock_conn2 = MockDbConnection(cursor=mock_cursor2)
    cfg_skip = RegistrationConfig(
        manifest_path=str(manifest_path),
        eval_report_path=str(eval_path),
        duplicate_policy="skip",
    )
    records = register_artifacts(cfg_skip, connection=mock_conn2)
    assert records[0].status == "SKIPPED"


# ---------------------------------------------------------------------------
# Unit Tests: CLI Execution and Main Entry Point
# ---------------------------------------------------------------------------

def test_cli_main_validate_only(temp_workspace: Path, capsys: pytest.CaptureFixture) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--eval-report",
            str(eval_path),
            "--validate-only",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "M3 Registration Complete" in captured.out
    assert "[VALIDATED]" in captured.out


def test_cli_main_json_output(temp_workspace: Path, capsys: pytest.CaptureFixture) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--eval-report",
            str(eval_path),
            "--validate-only",
            "--json",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == item.name
    assert data[0]["status"] == "VALIDATED"


def test_cli_subprocess_execution(temp_workspace: Path) -> None:
    _, item = create_dummy_artifact(temp_workspace, "model.onnx")
    manifest_path = create_dummy_manifest(temp_workspace, [item])
    eval_path = create_dummy_eval_report(temp_workspace)

    # 1. Module invocation (-m scripts.register_m3_artifacts)
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.register_m3_artifacts",
            "--manifest",
            str(manifest_path),
            "--eval-report",
            str(eval_path),
            "--validate-only",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "[VALIDATED]" in res.stdout

    # 2. Direct script invocation (python scripts/register_m3_artifacts.py)
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "register_m3_artifacts.py"
    res2 = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--manifest",
            str(manifest_path),
            "--eval-report",
            str(eval_path),
            "--validate-only",
        ],
        capture_output=True,
        text=True,
    )
    assert res2.returncode == 0
    assert "[VALIDATED]" in res2.stdout
