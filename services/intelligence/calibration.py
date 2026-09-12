from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Literal, Sequence
from pydantic import BaseModel, ConfigDict, Field


def _softmax_with_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not logits:
        return []

    scaled = [float(z) / temperature for z in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(v - max_val) for v in scaled]
    sum_exp = sum(exp_vals)
    if sum_exp <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [v / sum_exp for v in exp_vals]


def _compute_nll(
    logits_batch: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float,
) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    n = len(logits_batch)
    if n == 0:
        return 0.0

    total_loss = 0.0
    for logits, y in zip(logits_batch, labels):
        y_int = int(y)
        if not (0 <= y_int < len(logits)):
            raise ValueError(f"Label {y_int} is out of bounds for logits of length {len(logits)}")
        scaled = [float(z) / temperature for z in logits]
        max_val = max(scaled)
        sum_exp = sum(math.exp(v - max_val) for v in scaled)
        if sum_exp <= 0.0:
            total_loss += math.log(len(logits))
        else:
            log_sum_exp = max_val + math.log(sum_exp)
            total_loss += log_sum_exp - scaled[y_int]

    return total_loss / n


def compute_ece(
    probabilities_batch: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    if not probabilities_batch or len(probabilities_batch) != len(labels):
        raise ValueError("probabilities_batch and labels must be non-empty and of equal length")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    n_samples = len(probabilities_batch)
    bin_boundaries = [i / float(n_bins) for i in range(n_bins + 1)]

    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin_confs: list[float] = []
        in_bin_correct: list[int] = []

        for probs, y in zip(probabilities_batch, labels):
            y_int = int(y)
            max_conf = max(probs)
            pred_y = probs.index(max_conf)

            if (i == 0 and bin_lower <= max_conf <= bin_upper) or (bin_lower < max_conf <= bin_upper):
                in_bin_confs.append(max_conf)
                in_bin_correct.append(1 if pred_y == y_int else 0)

        bin_count = len(in_bin_confs)
        if bin_count > 0:
            avg_conf = sum(in_bin_confs) / bin_count
            accuracy = sum(in_bin_correct) / bin_count
            ece += (bin_count / n_samples) * abs(accuracy - avg_conf)

    return round(ece, 6)


def fit_temperature(
    logits_batch: Sequence[Sequence[float]],
    labels: Sequence[int],
    min_temp: float = 0.05,
    max_temp: float = 10.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    if len(logits_batch) != len(labels):
        raise ValueError("logits_batch and labels must have the same length")
    if not logits_batch:
        return 1.0

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a = min_temp
    b = max_temp

    c = b - phi * (b - a)
    d = a + phi * (b - a)

    fc = _compute_nll(logits_batch, labels, c)
    fd = _compute_nll(logits_batch, labels, d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = _compute_nll(logits_batch, labels, c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = _compute_nll(logits_batch, labels, d)

    optimal_t = (a + b) / 2.0
    optimal_t = max(min_temp, min(max_temp, optimal_t))

    uncal_nll = _compute_nll(logits_batch, labels, 1.0)
    opt_nll = _compute_nll(logits_batch, labels, optimal_t)
    if opt_nll > uncal_nll:
        optimal_t = 1.0

    return round(optimal_t, 4)


def load_temperatures(path_or_dict: str | Path | dict[str, Any]) -> dict[str, float]:
    """Load per-head temperatures from temperatures.json or dict format."""
    if isinstance(path_or_dict, dict):
        raw = path_or_dict
    else:
        p = Path(path_or_dict)
        if not p.is_file():
            raise FileNotFoundError(f"Temperatures file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict of temperatures, got {type(raw).__name__}")

    # Unwrap if wrapped in "temperatures" or "heads"
    if "temperatures" in raw and isinstance(raw["temperatures"], dict):
        raw = raw["temperatures"]
    elif "heads" in raw and isinstance(raw["heads"], dict):
        raw = raw["heads"]

    temperatures: dict[str, float] = {}
    for k, v in raw.items():
        try:
            val = float(v)
            if val <= 0.0:
                raise ValueError(f"Temperature for {k} must be positive, got {val}")
            temperatures[str(k)] = round(val, 4)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid temperature value for head '{k}': {v}") from exc

    return temperatures


def discover_temperatures_path(
    explicit_path: str | Path | None = None,
    candidate_paths: Sequence[str | Path | None] | None = None,
) -> Path | None:
    """Discover a valid temperatures.json file from explicit path or candidate search paths."""
    if explicit_path is not None:
        p = Path(explicit_path).resolve()
        if p.is_file():
            return p
        if p.is_dir():
            for fname in ("temperatures.json", "multitask_temperatures.json"):
                cand = p / fname
                if cand.is_file():
                    return cand
        raise FileNotFoundError(f"Explicit temperature artifact not found at: {explicit_path}")

    if not candidate_paths:
        return None

    for cand in candidate_paths:
        if cand is None:
            continue
        p = Path(cand).resolve()
        if p.is_file():
            if p.name.endswith(".json") and ("temp" in p.name.lower() or p.name == "temperatures.json"):
                return p
        search_dirs = [p] if p.is_dir() else [p.parent]
        if not p.is_dir() and p.parent.parent.is_dir():
            search_dirs.append(p.parent.parent)
        for sdir in search_dirs:
            for fname in ("temperatures.json", "multitask_temperatures.json", "calibration/temperatures.json"):
                target = sdir / fname
                if target.is_file():
                    return target
    return None


def save_temperatures(temperatures: dict[str, float], path: str | Path) -> Path:
    """Save per-head temperatures in canonical temperatures.json format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: round(float(v), 4) for k, v in temperatures.items()}
    with p.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return p


def fit_multitask_temperatures(
    logits_by_head: dict[str, Sequence[Sequence[float]]],
    labels_by_head: dict[str, Sequence[int]],
    min_temp: float = 0.05,
    max_temp: float = 10.0,
) -> dict[str, float]:
    """Fit per-head temperature scaling on dev logits and labels using robust NLL minimization."""
    temperatures: dict[str, float] = {}
    for head, logits in logits_by_head.items():
        labels = labels_by_head.get(head)
        if logits and labels is not None and len(labels) == len(logits):
            t = fit_temperature(logits, labels, min_temp=min_temp, max_temp=max_temp)
            temperatures[head] = t
    return temperatures


def apply_temperature(
    logits: Sequence[float],
    temperature: float = 1.0,
) -> list[float]:
    return _softmax_with_temperature(logits, temperature)


def apply_temperature_batch(
    logits_batch: Sequence[Sequence[float]],
    temperature: float = 1.0,
) -> list[list[float]]:
    return [_softmax_with_temperature(logits, temperature) for logits in logits_batch]


class CalibrationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    uncalibrated_nll: float = Field(ge=0.0)
    calibrated_nll: float = Field(ge=0.0)
    uncalibrated_ece: float = Field(ge=0.0, le=1.0)
    calibrated_ece: float = Field(ge=0.0, le=1.0)
    optimal_temperature: float = Field(gt=0.0)
    head: str | None = Field(default=None)


class TemperatureCalibrator:
    def __init__(
        self,
        temperature: float | dict[str, float] | str | Path = 1.0,
        temperatures: dict[str, float] | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        self.head_temperatures: dict[str, float] = {}
        self.temperature: float = 1.0
        self.source_path: Path | None = Path(source_path).resolve() if source_path else None

        if temperatures is not None:
            self.head_temperatures = {str(k): round(float(v), 4) for k, v in temperatures.items()}
            for k, v in self.head_temperatures.items():
                if v <= 0.0:
                    raise ValueError(f"temperature for {k} must be positive")
            if self.head_temperatures:
                self.temperature = float(next(iter(self.head_temperatures.values())))
        elif isinstance(temperature, (str, Path)):
            p = Path(temperature).resolve()
            self.source_path = p
            self.head_temperatures = load_temperatures(p)
            if self.head_temperatures:
                self.temperature = float(next(iter(self.head_temperatures.values())))
        elif isinstance(temperature, dict):
            self.head_temperatures = load_temperatures(temperature)
            if self.head_temperatures:
                self.temperature = float(next(iter(self.head_temperatures.values())))
        else:
            t = float(temperature)
            if t <= 0.0:
                raise ValueError("temperature must be positive")
            self.temperature = round(t, 4)

    @classmethod
    def from_artifact(
        cls,
        explicit_path: str | Path | None = None,
        candidate_paths: Sequence[str | Path | None] | None = None,
        default_temperature: float = 1.0,
    ) -> TemperatureCalibrator:
        discovered = discover_temperatures_path(
            explicit_path=explicit_path,
            candidate_paths=candidate_paths,
        )
        if discovered is not None:
            temps = load_temperatures(discovered)
            return cls(temperatures=temps, source_path=discovered)
        return cls(temperature=default_temperature)

    def get_temperature(self, head: str | None = None) -> float:
        if head is not None and head in self.head_temperatures:
            return self.head_temperatures[head]
        return self.temperature

    def set_temperature(self, temperature: float, head: str | None = None) -> None:
        t = float(temperature)
        if t <= 0.0:
            raise ValueError("temperature must be positive")
        val = round(t, 4)
        if head is not None:
            self.head_temperatures[head] = val
        else:
            self.temperature = val

    def fit(
        self,
        logits_batch: Sequence[Sequence[float]],
        labels: Sequence[int],
        min_temp: float = 0.05,
        max_temp: float = 10.0,
        head: str | None = None,
    ) -> float:
        t = fit_temperature(
            logits_batch=logits_batch,
            labels=labels,
            min_temp=min_temp,
            max_temp=max_temp,
        )
        if head is not None:
            self.head_temperatures[head] = t
        else:
            self.temperature = t
        return t

    def fit_head(
        self,
        head: str,
        logits_batch: Sequence[Sequence[float]],
        labels: Sequence[int],
        min_temp: float = 0.05,
        max_temp: float = 10.0,
    ) -> float:
        return self.fit(logits_batch, labels, min_temp=min_temp, max_temp=max_temp, head=head)

    def fit_multitask(
        self,
        logits_by_head: dict[str, Sequence[Sequence[float]]],
        labels_by_head: dict[str, Sequence[int]],
        min_temp: float = 0.05,
        max_temp: float = 10.0,
    ) -> dict[str, float]:
        fitted = fit_multitask_temperatures(
            logits_by_head=logits_by_head,
            labels_by_head=labels_by_head,
            min_temp=min_temp,
            max_temp=max_temp,
        )
        self.head_temperatures.update(fitted)
        if self.head_temperatures:
            self.temperature = float(next(iter(self.head_temperatures.values())))
        return dict(self.head_temperatures)

    def calibrate(self, logits: Sequence[float], head: str | None = None) -> list[float]:
        t = self.get_temperature(head)
        return apply_temperature(logits, t)

    def calibrate_head(self, head: str, logits: Sequence[float]) -> list[float]:
        return self.calibrate(logits, head=head)

    def calibrate_batch(
        self,
        logits_batch: Sequence[Sequence[float]],
        head: str | None = None,
    ) -> list[list[float]]:
        t = self.get_temperature(head)
        return apply_temperature_batch(logits_batch, t)

    def calibrate_head_batch(
        self,
        head: str,
        logits_batch: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        return self.calibrate_batch(logits_batch, head=head)

    def evaluate(
        self,
        logits_batch: Sequence[Sequence[float]],
        labels: Sequence[int],
        n_bins: int = 10,
        head: str | None = None,
    ) -> CalibrationMetrics:
        t = self.get_temperature(head)
        uncal_probs = apply_temperature_batch(logits_batch, 1.0)
        cal_probs = apply_temperature_batch(logits_batch, t)

        uncal_nll = _compute_nll(logits_batch, labels, 1.0)
        cal_nll = _compute_nll(logits_batch, labels, t)

        uncal_ece = compute_ece(uncal_probs, labels, n_bins=n_bins)
        cal_ece = compute_ece(cal_probs, labels, n_bins=n_bins)

        return CalibrationMetrics(
            uncalibrated_nll=round(uncal_nll, 6),
            calibrated_nll=round(cal_nll, 6),
            uncalibrated_ece=uncal_ece,
            calibrated_ece=cal_ece,
            optimal_temperature=t,
            head=head,
        )

    def evaluate_head(
        self,
        head: str,
        logits_batch: Sequence[Sequence[float]],
        labels: Sequence[int],
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        return self.evaluate(logits_batch, labels, n_bins=n_bins, head=head)

    def evaluate_multitask(
        self,
        logits_by_head: dict[str, Sequence[Sequence[float]]],
        labels_by_head: dict[str, Sequence[int]],
        n_bins: int = 10,
    ) -> dict[str, CalibrationMetrics]:
        results: dict[str, CalibrationMetrics] = {}
        for head, logits in logits_by_head.items():
            labels = labels_by_head.get(head)
            if logits and labels is not None and len(labels) == len(logits):
                results[head] = self.evaluate(logits, labels, n_bins=n_bins, head=head)
        return results

    def to_dict(self) -> dict[str, float]:
        if self.head_temperatures:
            return dict(self.head_temperatures)
        return {"temperature": self.temperature}

    def save_json(self, path: str | Path) -> Path:
        return save_temperatures(self.to_dict(), path)

    def load_temperatures(self, path: str | Path) -> dict[str, float]:
        self.head_temperatures = load_temperatures(path)
        if self.head_temperatures:
            self.temperature = float(next(iter(self.head_temperatures.values())))
        return dict(self.head_temperatures)

    @classmethod
    def from_json(cls, path: str | Path) -> TemperatureCalibrator:
        temps = load_temperatures(path)
        return cls(temperatures=temps)


MultiHeadTemperatureCalibrator = TemperatureCalibrator


class CpuBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=1, ge=1)
    sequence_length: int = Field(default=448, ge=1)
    overlap_tokens: int = Field(default=64, ge=0)
    num_threads: int = Field(default=1, ge=1)
    warmup_runs: int = Field(default=3, ge=0)
    benchmark_runs: int = Field(default=10, ge=1)
    target_p95_latency_ms: float = Field(default=100.0, gt=0.0)
    precision: Literal["fp32"] = "fp32"
    device: Literal["cpu"] = "cpu"


class CpuBenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: CpuBenchmarkConfig
    mean_latency_ms: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    p99_latency_ms: float = Field(ge=0.0)
    min_latency_ms: float = Field(ge=0.0)
    max_latency_ms: float = Field(ge=0.0)
    throughput_qps: float = Field(ge=0.0)
    samples_count: int = Field(ge=1)
    sla_met: bool
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _calculate_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    return sorted_values[int(lower)] * (upper - index) + sorted_values[int(upper)] * (index - lower)


def _default_fp32_workload(sequence_length: int, batch_size: int) -> float:
    acc = 0.0
    limit = min(sequence_length * batch_size, 512)
    for i in range(limit):
        v = float(i) * 0.001
        acc += v * v
    return acc


def run_cpu_benchmark(
    config: CpuBenchmarkConfig,
    runner: Callable[[], Any] | None = None,
) -> CpuBenchmarkResult:
    target_runner = runner or (
        lambda: _default_fp32_workload(config.sequence_length, config.batch_size)
    )

    for _ in range(config.warmup_runs):
        target_runner()

    latencies_ms: list[float] = []
    for _ in range(config.benchmark_runs):
        t0 = time.perf_counter()
        target_runner()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    sorted_latencies = sorted(latencies_ms)
    total_ms = sum(latencies_ms)
    mean_ms = total_ms / len(latencies_ms)
    p50_ms = _calculate_percentile(sorted_latencies, 50.0)
    p95_ms = _calculate_percentile(sorted_latencies, 95.0)
    p99_ms = _calculate_percentile(sorted_latencies, 99.0)
    min_ms = sorted_latencies[0]
    max_ms = sorted_latencies[-1]

    total_sec = total_ms / 1000.0
    throughput = (len(latencies_ms) * config.batch_size) / total_sec if total_sec > 0.0 else 0.0
    sla_met = p95_ms <= config.target_p95_latency_ms

    return CpuBenchmarkResult(
        config=config,
        mean_latency_ms=round(mean_ms, 4),
        p50_latency_ms=round(p50_ms, 4),
        p95_latency_ms=round(p95_ms, 4),
        p99_latency_ms=round(p99_ms, 4),
        min_latency_ms=round(min_ms, 4),
        max_latency_ms=round(max_ms, 4),
        throughput_qps=round(throughput, 4),
        samples_count=len(latencies_ms),
        sla_met=sla_met,
    )
