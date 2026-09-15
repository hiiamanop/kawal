from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import threading
import time
from typing import Iterable


@dataclass
class Histogram:
    count: int = 0
    total: float = 0.0


class MetricsRegistry:
    def __init__(self) -> None:
        self._counters: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: defaultdict[tuple[str, tuple[tuple[str, str], ...]], Histogram] = defaultdict(Histogram)
        self._lock = threading.Lock()

    @staticmethod
    def _labels(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((labels or {}).items()))

    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._counters[(name, self._labels(labels))] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            histogram = self._histograms[(name, self._labels(labels))]
            histogram.count += 1
            histogram.total += value

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"{name}{self._render_labels(labels)} {value}")
            for (name, labels), histogram in sorted(self._histograms.items()):
                suffix = self._render_labels(labels)
                lines.append(f"{name}_count{suffix} {histogram.count}")
                lines.append(f"{name}_sum{suffix} {histogram.total}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_labels(labels: Iterable[tuple[str, str]]) -> str:
        labels = tuple(labels)
        if not labels:
            return ""
        escaped = [f'{key}="{value.replace(chr(34), chr(92) + chr(34))}"' for key, value in labels]
        return "{" + ",".join(escaped) + "}"


metrics = MetricsRegistry()


def measure(name: str, labels: dict[str, str] | None = None):
    started = time.perf_counter()

    def finish() -> None:
        metrics.observe(name, (time.perf_counter() - started) * 1000.0, labels)

    return finish
