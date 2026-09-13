from __future__ import annotations

from enum import StrEnum
from threading import Lock
import time
from typing import Any, Callable


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._time_fn = time_fn
        self._lock = Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_in_flight = False

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._evaluate_state_locked()
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def _evaluate_state_locked(self) -> None:
        if self._state == CircuitState.OPEN:
            now = self._time_fn()
            if self._last_failure_time is not None:
                elapsed = now - self._last_failure_time
                if elapsed >= self.recovery_timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_in_flight = False

    def allow_request(self) -> bool:
        with self._lock:
            self._evaluate_state_locked()
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                if not self._half_open_in_flight:
                    self._half_open_in_flight = True
                    return True
                return False
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._half_open_in_flight = False
            self._last_failure_time = None

    def record_failure(self) -> None:
        with self._lock:
            now = self._time_fn()
            self._last_failure_time = now
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_in_flight = False
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self.allow_request():
            raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_in_flight = False
