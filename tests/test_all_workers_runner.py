from __future__ import annotations

import os
import sys
import time

from scripts.run_all_workers import WorkerSupervisor, build_worker_commands


def test_build_worker_commands_defaults() -> None:
    cmds = build_worker_commands(
        python_bin=sys.executable,
        database_url="postgresql://localhost/test",
        redpanda_bootstrap="127.0.0.1:19092",
        include_intake=False,
    )
    assert "outbox-relay" in cmds
    assert "intake-assembly" in cmds
    assert "case-ready" in cmds
    assert "tool-gateway" in cmds
    assert "openwa-intake" not in cmds


def test_build_worker_commands_with_intake() -> None:
    cmds = build_worker_commands(
        python_bin=sys.executable,
        database_url="postgresql://localhost/test",
        redpanda_bootstrap="127.0.0.1:19092",
        include_intake=True,
    )
    assert "openwa-intake" in cmds


def test_supervisor_start_and_graceful_stop() -> None:
    # Use lightweight sleep commands to verify process lifecycle
    dummy_cmds = {
        "dummy-1": [sys.executable, "-c", "import time; time.sleep(10)"],
        "dummy-2": [sys.executable, "-c", "import time; time.sleep(10)"],
    }
    supervisor = WorkerSupervisor(dummy_cmds)
    supervisor.start()

    assert len(supervisor.processes) == 2
    for proc in supervisor.processes.values():
        assert proc.poll() is None

    # Stop supervisor gracefully
    supervisor.stop(timeout_seconds=2.0)

    for proc in supervisor.processes.values():
        assert proc.poll() is not None
