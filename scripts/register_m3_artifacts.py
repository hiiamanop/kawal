"""M3 Artifact Registration CLI.

Safely registers validated M3 model artifacts and evaluation reports
into the `model_registry` PostgreSQL table.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Literal, Sequence

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pydantic import BaseModel, ConfigDict, Field

from scripts.evaluate_m3 import M3EvaluationReport
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
# SQL Queries (Parameterized)
# ---------------------------------------------------------------------------

SELECT_MODEL_SQL = """
SELECT model_id, artifact_hash
FROM model_registry
WHERE name = %s AND version = %s
"""

INSERT_MODEL_SQL = """
INSERT INTO model_registry (
    name,
    version,
    artifact_hash,
    parameters,
    task,
    tokenizer,
    calibration,
    dataset_split,
    eval_summary
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING model_id, created_at
"""

UPDATE_MODEL_SQL = """
UPDATE model_registry
SET artifact_hash = %s,
    parameters = %s,
    task = %s,
    tokenizer = %s,
    calibration = %s,
    dataset_split = %s,
    eval_summary = %s
WHERE name = %s AND version = %s
RETURNING model_id, created_at
"""


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class RegistrationError(Exception):
    """Base exception for artifact registration."""

    pass


class ValidationError(RegistrationError):
    """Raised when manifest, artifact files, or evaluation report fail validation."""

    pass


class DuplicateModelError(RegistrationError):
    """Raised when an artifact model version already exists and duplicate_policy is 'error'."""

    pass


class DatabaseConnectionError(RegistrationError):
    """Raised when database connection fails, ensuring credentials are not leaked."""

    pass


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class RegistrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_path: str = Field(default="artifacts/manifest.json", min_length=1)
    eval_report_path: str = Field(default="evaluation_results.json", min_length=1)
    artifacts_dir: str | None = None
    artifact_name: str | None = None
    tokenizer: str = Field(default="indobenchmark/indobert-base-p1", min_length=1, max_length=128)
    dataset_split: str = Field(default="train", min_length=1, max_length=64)
    duplicate_policy: Literal["error", "skip", "update"] = "error"
    dry_run: bool = False
    validate_only: bool = False
    allow_failed_eval: bool = False


class RegistrationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str | None = None
    name: str
    version: str
    artifact_hash: str
    task: str
    status: Literal[
        "REGISTERED",
        "UPDATED",
        "SKIPPED",
        "VALIDATED",
        "DRY_RUN_WOULD_REGISTER",
        "DRY_RUN_WOULD_UPDATE",
        "DRY_RUN_WOULD_SKIP",
    ]
    message: str = ""
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_secret(text: str) -> str:
    """Redact passwords and database connection strings from text."""
    if not text:
        return text
    sanitized = re.sub(
        r"postgres(?:ql)?:\/\/[^\s\'\"]+",
        "[REDACTED_DATABASE_URL]",
        text,
    )
    sanitized = re.sub(
        r"(password\s*=\s*['\"]?)[^'\";\s]+(['\"]?)",
        r"\1***\2",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


def get_database_connection() -> Any:
    """Derive DB connection ONLY from DATABASE_URL environment variable.

    Never prints or leaks the DATABASE_URL.
    """
    if psycopg is None:
        raise RegistrationError("psycopg library is required for database operations.")
    url = os.environ.get("DATABASE_URL")
    if not url or not url.strip():
        raise RegistrationError(
            "DATABASE_URL environment variable is not set. "
            "Database connection can only be derived from DATABASE_URL."
        )
    try:
        return psycopg.connect(url)
    except Exception as exc:
        sanitized = sanitize_secret(str(exc))
        raise DatabaseConnectionError(f"Database connection failed: {sanitized}") from None


def _to_json_param(data: Any) -> Any:
    """Wrap dict in Jsonb if psycopg is available, otherwise json string."""
    if Jsonb is not None:
        return Jsonb(data)
    return json.dumps(data)


def load_and_validate_manifest(
    manifest_path: str | Path,
    artifacts_dir: str | Path | None = None,
    artifact_name: str | None = None,
) -> tuple[ArtifactManifest, Path, list[ArtifactManifestItem]]:
    """Load and strictly validate ArtifactManifest, path containment, and file checksums."""
    m_path = Path(manifest_path).resolve()
    if m_path.is_dir():
        m_path = m_path / "manifest.json"
    if not m_path.is_file():
        raise ValidationError(f"Manifest file not found: {m_path}")

    try:
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = ArtifactManifest.model_validate(data)
    except Exception as exc:
        raise ValidationError(f"Invalid artifact manifest at {m_path}: {exc}") from exc

    base_dir = Path(artifacts_dir).resolve() if artifacts_dir else m_path.parent.resolve()

    if not manifest.artifacts:
        raise ValidationError(f"Manifest contains no artifacts: {m_path}")

    if artifact_name is not None:
        if artifact_name not in manifest.artifacts:
            available = list(manifest.artifacts.keys())
            raise ValidationError(
                f"Artifact '{artifact_name}' not found in manifest. Available artifacts: {available}"
            )
        items_to_validate = [manifest.artifacts[artifact_name]]
    else:
        items_to_validate = list(manifest.artifacts.values())

    for item in items_to_validate:
        item_p = Path(item.path)
        if item_p.is_absolute() or ".." in item_p.parts:
            raise ValidationError(
                f"Artifact '{item.name}' path '{item.path}' is invalid: must be relative without '..'"
            )
        target_file = (base_dir / item_p).resolve()
        if not target_file.is_relative_to(base_dir):
            raise ValidationError(
                f"Artifact '{item.name}' path '{item.path}' escapes base directory '{base_dir}'"
            )
        if not target_file.is_file():
            raise ValidationError(
                f"Artifact '{item.name}' file does not exist at '{target_file}'"
            )

        actual_sha = compute_file_sha256(target_file)
        if actual_sha.lower() != item.sha256.lower():
            raise ValidationError(
                f"SHA256 checksum mismatch for artifact '{item.name}': "
                f"expected {item.sha256.lower()}, got {actual_sha.lower()}"
            )

        actual_size = target_file.stat().st_size
        if actual_size != item.size_bytes:
            raise ValidationError(
                f"Size mismatch for artifact '{item.name}': "
                f"expected {item.size_bytes} bytes, got {actual_size} bytes"
            )

    return manifest, base_dir, items_to_validate


def load_and_validate_eval_report(
    eval_report_path: str | Path,
    allow_failed: bool = False,
    items_to_check: list[ArtifactManifestItem] | None = None,
) -> M3EvaluationReport:
    """Load and validate M3EvaluationReport schema, status, and SLA conformance."""
    r_path = Path(eval_report_path).resolve()
    if not r_path.is_file():
        raise ValidationError(f"Evaluation report file not found: {r_path}")

    try:
        with open(r_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        report = M3EvaluationReport.model_validate(data)
    except Exception as exc:
        raise ValidationError(f"Invalid evaluation report at {r_path}: {exc}") from exc

    if not allow_failed:
        if report.status != "PASSED":
            raise ValidationError(
                f"Evaluation report status is '{report.status}' (expected 'PASSED'). "
                f"Use --allow-failed-eval to register anyway."
            )
        if not report.all_checks_passed:
            raise ValidationError(
                "Evaluation report has failed checks (all_checks_passed=False). "
                "Use --allow-failed-eval to register anyway."
            )
        if not report.sla_conformance:
            raise ValidationError(
                "Evaluation report failed latency SLA conformance. "
                "Use --allow-failed-eval to register anyway."
            )

    if items_to_check and report.manifest_audit and isinstance(report.manifest_audit, dict):
        items_checked = report.manifest_audit.get("items_checked")
        if isinstance(items_checked, dict) and not allow_failed:
            for item in items_to_check:
                if item.name in items_checked and items_checked[item.name] is False:
                    raise ValidationError(
                        f"Evaluation report marked artifact '{item.name}' as invalid in manifest_audit"
                    )

    return report


def extract_calibration(report: M3EvaluationReport) -> dict[str, Any]:
    """Extract calibration metadata from evaluation report."""
    metrics = report.metrics or {}
    if "calibration" in metrics and isinstance(metrics["calibration"], dict):
        return metrics["calibration"]
    cal: dict[str, Any] = {}
    for key in ("temperature", "calibrated_ece", "uncalibrated_ece", "ece"):
        if key in metrics:
            cal[key] = metrics[key]
    return cal


def extract_eval_summary(report: M3EvaluationReport) -> dict[str, Any]:
    """Extract structured evaluation summary for model_registry."""
    return {
        "evaluation_version": report.evaluation_version,
        "timestamp": report.timestamp,
        "seed": report.seed,
        "status": report.status,
        "sla_conformance": report.sla_conformance,
        "all_checks_passed": report.all_checks_passed,
        "metrics": report.metrics,
        "split_audit": report.split_audit,
        "manifest_audit": report.manifest_audit,
    }


# ---------------------------------------------------------------------------
# Registration Logic
# ---------------------------------------------------------------------------

def register_artifacts(
    config: RegistrationConfig,
    connection: Any | None = None,
) -> list[RegistrationRecord]:
    """Execute artifact validation and safe PostgreSQL registration."""
    # 1. Validate manifest and verify hashes and relative paths
    manifest, base_dir, items = load_and_validate_manifest(
        manifest_path=config.manifest_path,
        artifacts_dir=config.artifacts_dir,
        artifact_name=config.artifact_name,
    )

    # 2. Validate evaluation report
    report = load_and_validate_eval_report(
        eval_report_path=config.eval_report_path,
        allow_failed=config.allow_failed_eval,
        items_to_check=items,
    )

    calibration_data = extract_calibration(report)
    eval_summary_data = extract_eval_summary(report)

    # 3. Handle validate-only mode (offline, zero DB interaction)
    if config.validate_only:
        records: list[RegistrationRecord] = []
        for item in items:
            records.append(
                RegistrationRecord(
                    name=item.name,
                    version=item.version,
                    artifact_hash=item.sha256.lower(),
                    task=item.task,
                    status="VALIDATED",
                    message="Validated successfully (validate-only mode, zero DB interaction)",
                )
            )
        return records

    # 4. Handle dry-run mode (read-only duplicate check, zero mutation)
    if config.dry_run:
        records = []
        conn = None
        should_close = False
        if connection is not None:
            conn = connection
        elif os.environ.get("DATABASE_URL") and psycopg is not None:
            try:
                conn = get_database_connection()
                should_close = True
            except Exception:
                conn = None

        try:
            for item in items:
                status: Literal[
                    "DRY_RUN_WOULD_REGISTER",
                    "DRY_RUN_WOULD_UPDATE",
                    "DRY_RUN_WOULD_SKIP",
                ] = "DRY_RUN_WOULD_REGISTER"
                msg = "Dry-run: would insert into model_registry"

                if conn is not None:
                    with conn.cursor() as cur:
                        cur.execute(SELECT_MODEL_SQL, (item.name, item.version))
                        row = cur.fetchone()
                        if row:
                            if config.duplicate_policy == "error":
                                raise DuplicateModelError(
                                    f"Dry-run detected existing model '{item.name}' version '{item.version}' (model_id={row[0]})"
                                )
                            elif config.duplicate_policy == "skip":
                                status = "DRY_RUN_WOULD_SKIP"
                                msg = f"Dry-run: existing model '{item.name}' version '{item.version}' would be skipped"
                            elif config.duplicate_policy == "update":
                                status = "DRY_RUN_WOULD_UPDATE"
                                msg = f"Dry-run: existing model '{item.name}' version '{item.version}' would be updated"

                records.append(
                    RegistrationRecord(
                        name=item.name,
                        version=item.version,
                        artifact_hash=item.sha256.lower(),
                        task=item.task,
                        status=status,
                        message=msg,
                    )
                )
        finally:
            if should_close and conn is not None:
                conn.close()

        return records

    # 5. Live Registration: connection derived strictly from DATABASE_URL env
    conn = connection if connection is not None else get_database_connection()
    should_close = connection is None
    records = []

    try:
        with conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    for item in items:
                        parameters = {
                            "precision": item.precision,
                            "size_bytes": item.size_bytes,
                            "relative_path": item.path,
                            "manifest_version": manifest.manifest_version,
                            "environment": manifest.environment,
                        }

                        # Parameterized duplicate check
                        cur.execute(SELECT_MODEL_SQL, (item.name, item.version))
                        existing = cur.fetchone()

                        if existing is not None:
                            existing_id, existing_hash = existing
                            if config.duplicate_policy == "error":
                                raise DuplicateModelError(
                                    f"Model '{item.name}' version '{item.version}' already exists "
                                    f"(model_id={existing_id}, hash={existing_hash})"
                                )
                            elif config.duplicate_policy == "skip":
                                records.append(
                                    RegistrationRecord(
                                        model_id=str(existing_id),
                                        name=item.name,
                                        version=item.version,
                                        artifact_hash=str(existing_hash).strip(),
                                        task=item.task,
                                        status="SKIPPED",
                                        message=f"Model already registered (model_id={existing_id}); skipped.",
                                    )
                                )
                                continue
                            elif config.duplicate_policy == "update":
                                cur.execute(
                                    UPDATE_MODEL_SQL,
                                    (
                                        item.sha256.lower(),
                                        _to_json_param(parameters),
                                        item.task,
                                        config.tokenizer,
                                        _to_json_param(calibration_data),
                                        config.dataset_split,
                                        _to_json_param(eval_summary_data),
                                        item.name,
                                        item.version,
                                    ),
                                )
                                upd_row = cur.fetchone()
                                model_id_val = str(upd_row[0]) if upd_row else str(existing_id)
                                created_at_val = (
                                    upd_row[1].isoformat()
                                    if upd_row and len(upd_row) > 1 and hasattr(upd_row[1], "isoformat")
                                    else str(upd_row[1])
                                    if upd_row and len(upd_row) > 1
                                    else None
                                )
                                records.append(
                                    RegistrationRecord(
                                        model_id=model_id_val,
                                        name=item.name,
                                        version=item.version,
                                        artifact_hash=item.sha256.lower(),
                                        task=item.task,
                                        status="UPDATED",
                                        message="Model registry record updated.",
                                        created_at=created_at_val,
                                    )
                                )
                                continue

                        # Parameterized INSERT
                        try:
                            cur.execute(
                                INSERT_MODEL_SQL,
                                (
                                    item.name,
                                    item.version,
                                    item.sha256.lower(),
                                    _to_json_param(parameters),
                                    item.task,
                                    config.tokenizer,
                                    _to_json_param(calibration_data),
                                    config.dataset_split,
                                    _to_json_param(eval_summary_data),
                                ),
                            )
                            ins_row = cur.fetchone()
                            model_id_val = str(ins_row[0]) if ins_row else None
                            created_at_val = (
                                ins_row[1].isoformat()
                                if ins_row and len(ins_row) > 1 and hasattr(ins_row[1], "isoformat")
                                else str(ins_row[1])
                                if ins_row and len(ins_row) > 1
                                else None
                            )
                            records.append(
                                RegistrationRecord(
                                    model_id=model_id_val,
                                    name=item.name,
                                    version=item.version,
                                    artifact_hash=item.sha256.lower(),
                                    task=item.task,
                                    status="REGISTERED",
                                    message="Model registered successfully.",
                                    created_at=created_at_val,
                                )
                            )
                        except UniqueViolation:
                            # Handle concurrent race condition
                            if config.duplicate_policy == "error":
                                raise DuplicateModelError(
                                    f"Concurrent duplicate detected for model '{item.name}' version '{item.version}'"
                                )
                            elif config.duplicate_policy == "skip":
                                records.append(
                                    RegistrationRecord(
                                        name=item.name,
                                        version=item.version,
                                        artifact_hash=item.sha256.lower(),
                                        task=item.task,
                                        status="SKIPPED",
                                        message="Model concurrent registration duplicate; skipped.",
                                    )
                                )
                            elif config.duplicate_policy == "update":
                                cur.execute(
                                    UPDATE_MODEL_SQL,
                                    (
                                        item.sha256.lower(),
                                        _to_json_param(parameters),
                                        item.task,
                                        config.tokenizer,
                                        _to_json_param(calibration_data),
                                        config.dataset_split,
                                        _to_json_param(eval_summary_data),
                                        item.name,
                                        item.version,
                                    ),
                                )
                                upd_row = cur.fetchone()
                                records.append(
                                    RegistrationRecord(
                                        model_id=str(upd_row[0]) if upd_row else None,
                                        name=item.name,
                                        version=item.version,
                                        artifact_hash=item.sha256.lower(),
                                        task=item.task,
                                        status="UPDATED",
                                        message="Model record updated after concurrent conflict.",
                                        created_at=str(upd_row[1]) if upd_row and len(upd_row) > 1 else None,
                                    )
                                )
    finally:
        if should_close and conn is not None:
            conn.close()

    return records


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safe registration CLI for M3 model artifacts and evaluation reports into model_registry."
    )
    parser.add_argument(
        "--manifest",
        "-m",
        dest="manifest_path",
        type=str,
        default="artifacts/manifest.json",
        help="Path to manifest.json or directory containing manifest.json (default: artifacts/manifest.json)",
    )
    parser.add_argument(
        "--eval-report",
        "-e",
        dest="eval_report_path",
        type=str,
        default="evaluation_results.json",
        help="Path to evaluation results JSON file (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--artifacts-dir",
        "-d",
        type=str,
        default=None,
        help="Base directory for artifact files (defaults to manifest's parent directory)",
    )
    parser.add_argument(
        "--artifact-name",
        "-n",
        type=str,
        default=None,
        help="Optional specific artifact name to register (defaults to all artifacts in manifest)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="indobenchmark/indobert-base-p1",
        help="Tokenizer name for model_registry (default: indobenchmark/indobert-base-p1)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split name for model_registry (default: train)",
    )
    parser.add_argument(
        "--on-duplicate",
        choices=["error", "skip", "update"],
        default="error",
        dest="duplicate_policy",
        help="Action when model (name, version) already exists: error, skip, or update (default: error)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate artifacts and check registration status without mutating the database",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest, checksums, and evaluation report offline without connecting to DB",
    )
    parser.add_argument(
        "--allow-failed-eval",
        "--force",
        action="store_true",
        dest="allow_failed_eval",
        help="Allow registering artifacts even if evaluation report failed quality/SLA checks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print registration results as JSON to stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = RegistrationConfig(
        manifest_path=args.manifest_path,
        eval_report_path=args.eval_report_path,
        artifacts_dir=args.artifacts_dir,
        artifact_name=args.artifact_name,
        tokenizer=args.tokenizer,
        dataset_split=args.dataset_split,
        duplicate_policy=args.duplicate_policy,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        allow_failed_eval=args.allow_failed_eval,
    )

    try:
        records = register_artifacts(config)
        if args.json:
            print(json.dumps([r.model_dump() for r in records], indent=2))
        else:
            print(f"M3 Registration Complete ({len(records)} artifact(s) processed):")
            for r in records:
                print(f"  - [{r.status}] {r.name}:{r.version} (hash: {r.artifact_hash[:12]}...) {r.message}")
        return 0
    except RegistrationError as exc:
        sanitized_msg = sanitize_secret(str(exc))
        sys.stderr.write(f"Registration Error: {sanitized_msg}\n")
        return 1
    except Exception as exc:
        sanitized_msg = sanitize_secret(str(exc))
        sys.stderr.write(f"Unexpected Error: {sanitized_msg}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
