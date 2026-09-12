from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")
PrecisionType = Literal["fp32", "fp16", "bf16", "amp"]
DeviceType = Literal["cpu", "cuda", "gpu"]
TaskType = Literal["dapt", "multitask", "ner", "classification"]

ALLOWED_CPU_PRECISIONS: tuple[PrecisionType, ...] = ("fp32",)
ALLOWED_GPU_PRECISIONS: tuple[PrecisionType, ...] = ("fp32", "fp16", "bf16", "amp")


def compute_content_sha256(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return sha256(content).hexdigest().lower()


def compute_file_sha256(path: Path | str) -> str:
    hasher = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def compute_split_hash(items: Iterable[Any], salt: str = "") -> str:
    extracted: list[str] = []
    for item in items:
        if isinstance(item, str):
            extracted.append(item.strip())
        elif isinstance(item, dict):
            key = item.get("id") or item.get("scenario_id") or item.get("family_id")
            if key is not None:
                extracted.append(str(key).strip())
            else:
                extracted.append(json.dumps(item, sort_keys=True, separators=(",", ":")))
        elif hasattr(item, "scenario_id"):
            extracted.append(str(getattr(item, "scenario_id")).strip())
        elif hasattr(item, "family_id"):
            extracted.append(str(getattr(item, "family_id")).strip())
        elif hasattr(item, "id"):
            extracted.append(str(getattr(item, "id")).strip())
        else:
            extracted.append(str(item).strip())

    extracted.sort()
    prefix = f"{salt}\n" if salt else ""
    payload = prefix + "\n".join(extracted)
    return sha256(payload.encode("utf-8")).hexdigest().lower()


def compute_split_map_hashes(split_map: dict[str, Any]) -> dict[str, str]:
    splits: dict[str, list[str]] = {}
    for item_id, split_val in split_map.items():
        s_val = split_val.value if hasattr(split_val, "value") else str(split_val)
        splits.setdefault(s_val, []).append(item_id)
    return {s: compute_split_hash(ids) for s, ids in sorted(splits.items())}


def is_precision_allowed(device: str, precision: str) -> bool:
    dev = device.lower().strip()
    prec = precision.lower().strip()
    if dev == "cpu":
        return prec in ALLOWED_CPU_PRECISIONS
    if dev in ("cuda", "gpu"):
        return prec in ALLOWED_GPU_PRECISIONS
    return False


def validate_precision_constraint(device: str, precision: str) -> None:
    if not is_precision_allowed(device, precision):
        allowed = ALLOWED_CPU_PRECISIONS if device.lower().strip() == "cpu" else ALLOWED_GPU_PRECISIONS
        raise PrecisionConstraintError(
            f"Precision '{precision}' is not permitted on device '{device}'. Allowed precisions: {list(allowed)}"
        )


class TrainingConfigError(ValueError):
    pass


class PrecisionConstraintError(TrainingConfigError):
    pass


class SplitHashMismatchError(TrainingConfigError):
    pass


class HeldoutValidationError(TrainingConfigError):
    pass


class DataLeakageError(HeldoutValidationError):
    pass


class ModelVersionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str = Field(min_length=1, max_length=256)
    model_version: str = Field(min_length=1, max_length=64)
    tokenizer_name: str | None = Field(default=None, max_length=256)
    tokenizer_version: str = Field(min_length=1, max_length=64)
    max_sequence_length: int = Field(default=448, ge=16, le=4096)
    vocab_size: int | None = Field(default=None, ge=1)
    tokenizer_sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @field_validator("tokenizer_sha256")
    @classmethod
    def validate_tokenizer_sha256(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_REGEX.match(value):
            raise ValueError("tokenizer_sha256 must be a valid 64-character hexadecimal string")
        return value.lower() if value else None

    @property
    def effective_tokenizer_name(self) -> str:
        return self.tokenizer_name or self.base_model


class HyperparametersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    learning_rate: float = Field(gt=0.0, le=1.0)
    batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    num_train_epochs: int = Field(ge=1)
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    warmup_steps: int | None = Field(default=None, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    optimizer: str = Field(default="adamw", min_length=1, max_length=64)
    lr_scheduler: str = Field(default="linear", min_length=1, max_length=64)
    early_stopping_patience: int | None = Field(default=None, ge=1)
    task_weights: dict[str, float] = Field(default_factory=dict)
    temperature: float | None = Field(default=None, gt=0.0)
    masking_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_weights")
    @classmethod
    def validate_task_weights(cls, values: dict[str, float]) -> dict[str, float]:
        for k, v in values.items():
            if v < 0.0:
                raise ValueError(f"Task weight for '{k}' must be non-negative, got {v}")
        return values

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


class PrecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device: DeviceType = "cpu"
    training_precision: PrecisionType = "fp32"
    serving_precision: Literal["fp32"] = "fp32"

    @model_validator(mode="after")
    def check_precision_rules(self) -> PrecisionConfig:
        validate_precision_constraint(self.device, self.training_precision)
        return self


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=42, ge=0)
    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    dev_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    expected_split_hashes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ratio_sum(self) -> SplitConfig:
        total = self.train_ratio + self.dev_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Sum of split ratios must equal 1.0, got {total:.6f}")
        return self

    @field_validator("expected_split_hashes")
    @classmethod
    def validate_split_hash_hex(cls, values: dict[str, str]) -> dict[str, str]:
        validated: dict[str, str] = {}
        for k, v in values.items():
            if not SHA256_REGEX.match(v):
                raise ValueError(f"Split hash for '{k}' must be a 64-character hexadecimal string")
            validated[k.lower()] = v.lower()
        return validated

    def verify_split_hashes(self, actual_splits: dict[str, Iterable[Any]]) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for split_name, expected_hash in self.expected_split_hashes.items():
            if split_name not in actual_splits:
                results[split_name] = False
                continue
            actual_hash = compute_split_hash(actual_splits[split_name])
            results[split_name] = actual_hash == expected_hash.lower()
        return results

    def assert_valid_split_hashes(self, actual_splits: dict[str, Iterable[Any]]) -> None:
        for split_name, expected_hash in self.expected_split_hashes.items():
            if split_name not in actual_splits:
                raise SplitHashMismatchError(f"Split '{split_name}' missing from actual splits")
            actual_hash = compute_split_hash(actual_splits[split_name])
            if actual_hash != expected_hash.lower():
                raise SplitHashMismatchError(
                    f"Split hash mismatch for '{split_name}': expected {expected_hash}, got {actual_hash}"
                )


class HeldoutManifestItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    split: str = Field(default="dev", min_length=1, max_length=32)
    path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(min_length=64, max_length=64)
    expected_count: int = Field(ge=1)
    split_hash: str | None = Field(default=None, min_length=64, max_length=64)
    size_bytes: int | None = Field(default=None, ge=1)

    @field_validator("sha256", "split_hash")
    @classmethod
    def validate_hash_format(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_REGEX.match(value):
            raise ValueError("hash must be a valid 64-character hexadecimal string")
        return value.lower() if value else None

    @field_validator("path")
    @classmethod
    def validate_path_safety(cls, value: str) -> str:
        p = Path(value)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("path must be relative and cannot contain parent traversal '..'")
        return value


class HeldoutValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    path: str
    is_valid: bool
    file_exists: bool
    sha256_matches: bool
    count_matches: bool
    split_hash_matches: bool | None = None
    zero_leakage: bool = True
    actual_count: int | None = None
    actual_sha256: str | None = None
    error_message: str | None = None


class HeldoutDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str = Field(default="1.0.0", min_length=1, max_length=32)
    datasets: dict[str, HeldoutManifestItem] = Field(default_factory=dict)


def validate_heldout_dataset(
    item: HeldoutManifestItem,
    base_dir: Path | str,
    train_sample_ids: set[str] | Sequence[str] | None = None,
    raise_on_error: bool = False,
) -> HeldoutValidationResult:
    target_path = Path(base_dir) / item.path
    if not target_path.exists() or not target_path.is_file():
        msg = f"Heldout file does not exist: {target_path}"
        if raise_on_error:
            raise HeldoutValidationError(msg)
        return HeldoutValidationResult(
            name=item.name,
            path=item.path,
            is_valid=False,
            file_exists=False,
            sha256_matches=False,
            count_matches=False,
            error_message=msg,
        )

    actual_hash = compute_file_sha256(target_path)
    sha256_matches = actual_hash == item.sha256.lower()
    if not sha256_matches and raise_on_error:
        raise HeldoutValidationError(
            f"SHA256 mismatch for heldout dataset '{item.name}': expected {item.sha256}, got {actual_hash}"
        )

    raw_text = target_path.read_text(encoding="utf-8")
    actual_count = 0
    extracted_ids: list[str] = []

    if item.path.endswith(".jsonl") or not (raw_text.strip().startswith("[") or raw_text.strip().startswith("{")):
        for line in raw_text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            actual_count += 1
            try:
                parsed = json.loads(line_str)
                if isinstance(parsed, dict):
                    rec_id = parsed.get("id") or parsed.get("scenario_id") or parsed.get("family_id")
                    if rec_id is not None:
                        extracted_ids.append(str(rec_id))
            except Exception:
                pass
    else:
        try:
            parsed_json = json.loads(raw_text)
            if isinstance(parsed_json, list):
                actual_count = len(parsed_json)
                for entry in parsed_json:
                    if isinstance(entry, dict):
                        rec_id = entry.get("id") or entry.get("scenario_id") or entry.get("family_id")
                        if rec_id is not None:
                            extracted_ids.append(str(rec_id))
                    elif isinstance(entry, str):
                        extracted_ids.append(entry)
            elif isinstance(parsed_json, dict):
                entries = parsed_json.get("items") or parsed_json.get("data") or parsed_json.get("trajectories") or []
                actual_count = len(entries)
                for entry in entries:
                    if isinstance(entry, dict):
                        rec_id = entry.get("id") or entry.get("scenario_id") or entry.get("family_id")
                        if rec_id is not None:
                            extracted_ids.append(str(rec_id))
        except Exception:
            actual_count = len(raw_text.splitlines())

    count_matches = actual_count == item.expected_count
    if not count_matches and raise_on_error:
        raise HeldoutValidationError(
            f"Record count mismatch for '{item.name}': expected {item.expected_count}, got {actual_count}"
        )

    split_hash_matches: bool | None = None
    if item.split_hash is not None and extracted_ids:
        actual_split_hash = compute_split_hash(extracted_ids)
        split_hash_matches = actual_split_hash == item.split_hash.lower()
        if not split_hash_matches and raise_on_error:
            raise SplitHashMismatchError(
                f"Split hash mismatch for heldout '{item.name}': expected {item.split_hash}, got {actual_split_hash}"
            )

    zero_leakage = True
    if train_sample_ids is not None and extracted_ids:
        train_set = set(train_sample_ids) if not isinstance(train_sample_ids, set) else train_sample_ids
        heldout_set = set(extracted_ids)
        intersection = heldout_set.intersection(train_set)
        if intersection:
            zero_leakage = False
            msg = f"Data leakage detected in heldout '{item.name}': {len(intersection)} overlapping IDs"
            if raise_on_error:
                raise DataLeakageError(msg)

    is_valid = (
        sha256_matches
        and count_matches
        and (split_hash_matches is not False)
        and zero_leakage
    )
    error_msg = None
    if not is_valid:
        reasons: list[str] = []
        if not sha256_matches:
            reasons.append(f"hash mismatch ({actual_hash} != {item.sha256})")
        if not count_matches:
            reasons.append(f"count mismatch ({actual_count} != {item.expected_count})")
        if split_hash_matches is False:
            reasons.append("split hash mismatch")
        if not zero_leakage:
            reasons.append("data leakage detected")
        error_msg = "; ".join(reasons)

    return HeldoutValidationResult(
        name=item.name,
        path=item.path,
        is_valid=is_valid,
        file_exists=True,
        sha256_matches=sha256_matches,
        count_matches=count_matches,
        split_hash_matches=split_hash_matches,
        zero_leakage=zero_leakage,
        actual_count=actual_count,
        actual_sha256=actual_hash,
        error_message=error_msg,
    )


def validate_heldout_manifest(
    manifest: HeldoutDatasetManifest,
    base_dir: Path | str,
    train_sample_ids: set[str] | Sequence[str] | None = None,
    raise_on_error: bool = False,
) -> list[HeldoutValidationResult]:
    results: list[HeldoutValidationResult] = []
    for item in manifest.datasets.values():
        res = validate_heldout_dataset(
            item=item,
            base_dir=base_dir,
            train_sample_ids=train_sample_ids,
            raise_on_error=raise_on_error,
        )
        results.append(res)
    return results


def evaluate_metric_thresholds(
    metrics: dict[str, float],
    thresholds: dict[str, float | tuple[Literal[">=", "<=", ">", "<", "=="], float]],
) -> tuple[bool, dict[str, bool]]:
    details: dict[str, bool] = {}
    lower_is_better_keywords = ("loss", "latency", "error", "cost", "ece", "time_ms")

    for metric_name, rule in thresholds.items():
        if metric_name not in metrics:
            details[metric_name] = False
            continue

        val = metrics[metric_name]
        if isinstance(rule, tuple):
            op, target = rule
        else:
            target = rule
            name_lower = metric_name.lower()
            op = "<=" if any(kw in name_lower for kw in lower_is_better_keywords) else ">="

        if op == ">=":
            details[metric_name] = val >= target
        elif op == "<=":
            details[metric_name] = val <= target
        elif op == ">":
            details[metric_name] = val > target
        elif op == "<":
            details[metric_name] = val < target
        elif op == "==":
            details[metric_name] = abs(val - target) < 1e-6
        else:
            details[metric_name] = False

    passed = len(details) > 0 and all(details.values())
    return passed, details


class MetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    split: str = Field(min_length=1, max_length=32)
    epoch: int | None = Field(default=None, ge=0)
    step: int | None = Field(default=None, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    is_best: bool = False
    passed_thresholds: bool = True
    threshold_details: dict[str, bool] = Field(default_factory=dict)
    checkpoint_version: str | None = None
    checkpoint_sha256: str | None = None
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_checkpoint_sha256(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_REGEX.match(value):
            raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal string")
        return value.lower() if value else None

    def evaluate_thresholds(
        self,
        thresholds: dict[str, float | tuple[Literal[">=", "<=", ">", "<", "=="], float]],
    ) -> MetricResult:
        passed, details = evaluate_metric_thresholds(self.metrics, thresholds)
        return self.model_copy(update={"passed_thresholds": passed, "threshold_details": details})


class M3TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: str = Field(default="1.0.0", min_length=1, max_length=32)
    task: TaskType
    seed: int = Field(default=42, ge=0)
    model_version: ModelVersionConfig
    split_config: SplitConfig
    hyperparameters: HyperparametersConfig
    precision_config: PrecisionConfig = Field(default_factory=PrecisionConfig)
    heldout_manifest: HeldoutDatasetManifest | None = None
    target_metric_thresholds: dict[str, float] = Field(default_factory=dict)
    description: str | None = None

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_json(cls, json_str: str) -> M3TrainingConfig:
        return cls.model_validate_json(json_str)

    @classmethod
    def from_file(cls, path: Path | str) -> M3TrainingConfig:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def compute_config_hash(self) -> str:
        return sha256(self.model_dump_json().encode("utf-8")).hexdigest().lower()

    def validate_runtime_compatibility(self, target_device: str) -> None:
        validate_precision_constraint(target_device, self.precision_config.training_precision)
