from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

SHA256_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")


def compute_file_sha256(path: Path | str) -> str:
    hasher = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class ArtifactManifestItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    sha256: str = Field(min_length=64, max_length=64)
    path: str = Field(min_length=1, max_length=512)
    size_bytes: int = Field(ge=1)
    task: str = Field(default="general", min_length=1, max_length=64)
    precision: Literal["fp32", "fp16", "int8"] = "fp32"

    @field_validator("sha256")
    @classmethod
    def validate_sha256_format(cls, value: str) -> str:
        if not SHA256_REGEX.match(value):
            raise ValueError("sha256 must be a valid 64-character hexadecimal string")
        return value.lower()

    @field_validator("path")
    @classmethod
    def validate_path_safety(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("path must be a relative path and cannot contain parent directory traversal '..'")
        return value


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_name: str
    is_valid: bool
    expected_sha256: str
    actual_sha256: str | None = None
    file_exists: bool
    error_message: str | None = None


class ManifestValidationError(Exception):
    pass


class ArtifactNotFoundError(ManifestValidationError):
    pass


class ArtifactIntegrityError(ManifestValidationError):
    pass


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str = Field(default="1.0.0", min_length=1, max_length=32)
    environment: str = Field(default="local-cpu", min_length=1, max_length=64)
    artifacts: dict[str, ArtifactManifestItem] = Field(default_factory=dict)
    _validation_cache: dict[str, ValidationResult] = PrivateAttr(default_factory=dict)

    def clear_validation_cache(self) -> None:
        self._validation_cache.clear()

    def get_checkpoint_version(self, name: str | None = None) -> str | None:
        if name is not None:
            if name in self.artifacts:
                return self.artifacts[name].version
            for item in self.artifacts.values():
                if item.name == name:
                    return item.version
            return None
        if len(self.artifacts) == 1:
            return next(iter(self.artifacts.values())).version
        for item in self.artifacts.values():
            if item.task in ("classification", "multitask", "general"):
                return item.version
        if self.artifacts:
            return next(iter(self.artifacts.values())).version
        return None

    def get_artifact(self, name: str) -> ArtifactManifestItem:
        if name not in self.artifacts:
            raise ArtifactNotFoundError(f"Artifact '{name}' not found in manifest")
        return self.artifacts[name]

    def validate_artifact_file(
        self,
        name: str,
        base_dir: Path | str,
        raise_on_error: bool = False,
        use_cache: bool = True,
    ) -> ValidationResult:
        cache_key = f"{Path(base_dir).resolve()}:{name}"
        if use_cache and cache_key in self._validation_cache:
            cached = self._validation_cache[cache_key]
            if raise_on_error and not cached.is_valid:
                if not cached.file_exists:
                    raise ArtifactNotFoundError(cached.error_message or f"Artifact not found: {name}")
                raise ArtifactIntegrityError(cached.error_message or f"Artifact integrity error: {name}")
            return cached

        item = self.get_artifact(name)
        base = Path(base_dir).resolve()
        target_path = (base / item.path).resolve()

        if not target_path.is_relative_to(base):
            msg = f"Artifact path '{item.path}' resolves outside base directory '{base}'"
            if raise_on_error:
                raise ManifestValidationError(msg)
            res = ValidationResult(
                artifact_name=name,
                is_valid=False,
                expected_sha256=item.sha256,
                file_exists=False,
                error_message=msg,
            )
            self._validation_cache[cache_key] = res
            return res

        if not target_path.is_file():
            msg = f"Artifact file not found at {target_path}"
            if raise_on_error:
                raise ArtifactNotFoundError(msg)
            res = ValidationResult(
                artifact_name=name,
                is_valid=False,
                expected_sha256=item.sha256,
                file_exists=False,
                error_message=msg,
            )
            self._validation_cache[cache_key] = res
            return res

        computed_hash = compute_file_sha256(target_path)
        if computed_hash != item.sha256.lower():
            msg = f"SHA256 mismatch for artifact '{name}': expected {item.sha256}, got {computed_hash}"
            if raise_on_error:
                raise ArtifactIntegrityError(msg)
            res = ValidationResult(
                artifact_name=name,
                is_valid=False,
                expected_sha256=item.sha256,
                actual_sha256=computed_hash,
                file_exists=True,
                error_message=msg,
            )
            self._validation_cache[cache_key] = res
            return res

        res = ValidationResult(
            artifact_name=name,
            is_valid=True,
            expected_sha256=item.sha256,
            actual_sha256=computed_hash,
            file_exists=True,
        )
        self._validation_cache[cache_key] = res
        return res

    def validate_all(
        self,
        base_dir: Path | str,
        raise_on_error: bool = False,
        use_cache: bool = True,
    ) -> dict[str, ValidationResult]:
        results: dict[str, ValidationResult] = {}
        for name in self.artifacts:
            results[name] = self.validate_artifact_file(
                name,
                base_dir,
                raise_on_error=raise_on_error,
                use_cache=use_cache,
            )
        return results

    @classmethod
    def from_file(cls, file_path: Path | str) -> ArtifactManifest:
        path = Path(file_path)
        if not path.is_file():
            raise ManifestValidationError(f"Manifest file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
