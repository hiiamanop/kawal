from __future__ import annotations

from contracts.harness import TraceEvent


class TraceRecorder:
    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._events: list[TraceEvent] = []

    def add(self, event: str, **data: object) -> None:
        self._events.append(
            TraceEvent(
                event=event,
                data={"run_id": self._run_id, **data},
            )
        )

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)
