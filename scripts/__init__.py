from __future__ import annotations

import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from contracts.models import ComplaintTrajectory, FamilySplitAuditResult
from services.dataset.splits import audit_family_splits
from services.ml.manifest import (
    ArtifactManifest,
    ArtifactManifestItem,
    compute_file_sha256,
)


class OptionalDependencyError(ImportError):
    """Raised when an optional ML library is required for external execution but missing."""

    def __init__(self, package_name: str, purpose: str = "ML execution") -> None:
        self.package_name = package_name
        self.purpose = purpose
        super().__init__(
            f"Optional ML dependency '{package_name}' is required for {purpose}, "
            f"but is not installed. Install dependencies via `pip install -r requirements-ml.txt`."
        )


class DatasetAuditError(ValueError):
    """Raised when dataset split audit fails the zero-leakage guarantee."""

    pass


def check_ml_dependencies(*packages: str) -> dict[str, bool]:
    """Check availability of optional packages without raising exceptions."""
    results: dict[str, bool] = {}
    for pkg in packages:
        try:
            __import__(pkg)
            results[pkg] = True
        except ImportError:
            results[pkg] = False
    return results


def require_ml_dependencies(*packages: str, purpose: str = "ML execution") -> None:
    """Ensure required optional packages are installed, or raise OptionalDependencyError."""
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError as exc:
            raise OptionalDependencyError(pkg, purpose=purpose) from exc


def set_deterministic_seed(seed: int) -> None:
    """Pin seeds across Python standard library and available ML engines."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate_dataset_splits(
    trajectories: Sequence[ComplaintTrajectory | dict[str, Any]],
    allow_violations: bool = False,
) -> FamilySplitAuditResult:
    """Validate zero-leakage guarantee across dataset family splits."""
    audit = audit_family_splits(trajectories)
    if not audit.passed and not allow_violations:
        violations_str = "; ".join(audit.violations) if audit.violations else "audit.passed is False"
        raise DatasetAuditError(
            f"Dataset split audit failed zero-leakage guarantee: {violations_str}"
        )
    return audit


def create_artifact_item(
    file_path: Path | str,
    name: str,
    version: str = "v1.0.0",
    task: str = "general",
    precision: Literal["fp32", "fp16", "int8"] = "fp32",
    relative_to: Path | str | None = None,
) -> ArtifactManifestItem:
    """Create an ArtifactManifestItem from an existing file on disk."""
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Artifact file does not exist: {p}")

    if relative_to is not None:
        rel_dir = Path(relative_to).resolve()
        rel_path = p.relative_to(rel_dir).as_posix()
    else:
        rel_path = p.name

    sha256_hash = compute_file_sha256(p)
    size_bytes = p.stat().st_size

    return ArtifactManifestItem(
        name=name,
        version=version,
        sha256=sha256_hash,
        path=rel_path,
        size_bytes=size_bytes,
        task=task,
        precision=precision,
    )


def create_or_update_artifact_manifest(
    output_dir: Path | str,
    items: Sequence[ArtifactManifestItem],
    manifest_version: str = "1.0.0",
    environment: str = "local-cpu",
    manifest_filename: str = "manifest.json",
) -> ArtifactManifest:
    """Create or update manifest.json in output_dir with verified artifact items."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = out_dir / manifest_filename

    existing_artifacts: dict[str, ArtifactManifestItem] = {}
    if manifest_file.is_file():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_manifest = ArtifactManifest.model_validate(data)
            existing_artifacts.update(loaded_manifest.artifacts)
        except Exception:
            existing_artifacts = {}

    for item in items:
        target_file = out_dir / item.path
        if not target_file.is_file():
            raise FileNotFoundError(f"Artifact file '{item.path}' not found in '{out_dir}'")
        actual_sha = compute_file_sha256(target_file)
        if actual_sha.lower() != item.sha256.lower():
            raise ValueError(
                f"Checksum mismatch for '{item.path}': expected {item.sha256}, got {actual_sha}"
            )
        existing_artifacts[item.name] = item

    manifest = ArtifactManifest(
        manifest_version=manifest_version,
        environment=environment,
        artifacts=existing_artifacts,
    )

    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    return manifest


from scripts.evaluate_m3 import EvaluationConfig, M3EvaluationReport, run_evaluate_m3
from scripts.export_onnx import ONNXExportConfig, run_export_onnx
from scripts.train_dapt import DAPTConfig, run_train_dapt
from scripts.train_multitask import MultitaskTrainConfig, run_train_multitask
from scripts.train_ner import NERTrainConfig, run_train_ner

__all__ = [
    "OptionalDependencyError",
    "DatasetAuditError",
    "check_ml_dependencies",
    "require_ml_dependencies",
    "set_deterministic_seed",
    "validate_dataset_splits",
    "create_artifact_item",
    "create_or_update_artifact_manifest",
    "DAPTConfig",
    "run_train_dapt",
    "MultitaskTrainConfig",
    "run_train_multitask",
    "NERTrainConfig",
    "run_train_ner",
    "ONNXExportConfig",
    "run_export_onnx",
    "EvaluationConfig",
    "M3EvaluationReport",
    "run_evaluate_m3",
]
