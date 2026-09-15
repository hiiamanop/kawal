#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [supervisor] %(message)s",
)
logger = logging.getLogger("supervisor")


def stream_pipe(pipe, prefix: str) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            clean_line = line.rstrip("\r\n")
            print(f"[{prefix}] {clean_line}", flush=True)
    except Exception:
        pass
    finally:
        pipe.close()


def build_worker_commands(
    python_bin: str,
    database_url: str,
    redpanda_bootstrap: str,
    include_intake: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    repo_root = Path(__file__).resolve().parent.parent

    cmds: dict[str, list[str]] = {
        "outbox-relay": [
            python_bin,
            str(repo_root / "scripts" / "run_outbox_relay.py"),
            "--database-url",
            database_url,
            "--bootstrap-servers",
            redpanda_bootstrap,
            "--batch-size",
            "20",
            "--poll-seconds",
            "0.5",
        ],
        "intake-assembly": [
            python_bin,
            str(repo_root / "scripts" / "run_intake_assembly_worker.py"),
            "--database-url",
            database_url,
            "--bootstrap-servers",
            redpanda_bootstrap,
            "--poll-interval",
            "0.5",
        ],
        "case-ready": [
            python_bin,
            str(repo_root / "scripts" / "run_case_ready_worker.py"),
            "--database-url",
            database_url,
            "--bootstrap-servers",
            redpanda_bootstrap,
            "--poll-seconds",
            "0.5",
        ],
        "tool-gateway": [
            python_bin,
            str(repo_root / "scripts" / "run_gateway_worker.py"),
            "--bootstrap-servers",
            redpanda_bootstrap,
            "--poll-interval",
            "0.5",
        ],
    }

    if include_intake:
        cmds["openwa-intake"] = [
            python_bin,
            str(repo_root / "scripts" / "run_openwa_intake.py"),
        ]

    return cmds


class WorkerSupervisor:
    def __init__(self, commands: dict[str, list[str]], env: dict[str, str] | None = None) -> None:
        self._commands = commands
        self._env = env or os.environ.copy()
        self._processes: dict[str, subprocess.Popen] = {}
        self._threads: list[threading.Thread] = []
        self._stopping = False

    @property
    def processes(self) -> dict[str, subprocess.Popen]:
        return self._processes

    def start(self) -> None:
        logger.info("Starting %d background worker services...", len(self._commands))
        for name, cmd in self._commands.items():
            logger.info("Launching service [%s] -> %s", name, " ".join(cmd[:3]))
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=self._env,
                text=True,
                bufsize=1,
            )
            self._processes[name] = proc

            th = threading.Thread(
                target=stream_pipe,
                args=(proc.stdout, name),
                daemon=True,
                name=f"stream-{name}",
            )
            th.start()
            self._threads.append(th)

    def wait(self, poll_interval: float = 0.5) -> int:
        while not self._stopping:
            for name, proc in list(self._processes.items()):
                ret = proc.poll()
                if ret is not None:
                    logger.warning("Service [%s] exited unexpectedly with code %d", name, ret)
                    self.stop()
                    return ret
            time.sleep(poll_interval)
        return 0

    def stop(self, timeout_seconds: float = 5.0) -> None:
        if self._stopping:
            return
        self._stopping = True
        logger.info("Shutting down all worker services gracefully...")

        for name, proc in self._processes.items():
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                except OSError:
                    pass

        deadline = time.time() + timeout_seconds
        for name, proc in self._processes.items():
            remaining = max(0.1, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("Service [%s] did not terminate in time; killing SIGKILL", name)
                proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="KAWAL Unified Background Workers Supervisor")
    parser.add_argument(
        "--database-url",
        default=os.getenv("KAWAL_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres"),
        help="PostgreSQL DSN",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAWAL_REDPANDA_BOOTSTRAP", "127.0.0.1:19092"),
        help="Redpanda bootstrap address",
    )
    parser.add_argument(
        "--with-intake",
        action="store_true",
        help="Also run the OpenWA webhook intake server",
    )
    args = parser.parse_args()

    python_bin = sys.executable
    commands = build_worker_commands(
        python_bin=python_bin,
        database_url=args.database_url,
        redpanda_bootstrap=args.bootstrap_servers,
        include_intake=args.with_intake,
    )

    env = os.environ.copy()
    env["KAWAL_DATABASE_URL"] = args.database_url
    env["REDPANDA_BOOTSTRAP_SERVERS"] = args.bootstrap_servers

    supervisor = WorkerSupervisor(commands=commands, env=env)

    def handle_signal(sig, frame):
        logger.info("Caught signal %d; initiating shutdown sequence", sig)
        supervisor.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    supervisor.start()
    return supervisor.wait()


if __name__ == "__main__":
    raise SystemExit(main())
