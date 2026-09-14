from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.db import TransactionRunner
from services.clarification.dispatcher import ClarificationDispatcher
from services.core.pipeline import CaseProcessingPipeline
from services.core.worker import CaseReadyWorker
from services.intake.openwa import OpenWAConnector
from services.ml import LocalMLRuntime
from services.outbox.broker import RedpandaEventBroker
from services.outbox.consumer import AtomicInboxConsumer
from services.reliability.client import ReliableTicketClient
from services.simulator.store import TicketSimulator


def main() -> int:
    parser = argparse.ArgumentParser(description="KAWAL cases.ready.v1 live worker")
    parser.add_argument("--database-url", default=os.getenv("KAWAL_DATABASE_URL"))
    parser.add_argument("--bootstrap-servers", default=os.getenv("KAWAL_REDPANDA_BOOTSTRAP", "127.0.0.1:19092"))
    parser.add_argument("--group-id", default="kawal-case-ready-worker-v1")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="Disable ONNX neural model and run with fallback mode",
    )
    args = parser.parse_args()
    if not args.database_url:
        parser.error("--database-url or KAWAL_DATABASE_URL is required")

    broker = RedpandaEventBroker(bootstrap_servers=args.bootstrap_servers)
    consumer = AtomicInboxConsumer(args.group_id, TransactionRunner(args.database_url), broker)

    ml_runtime = (
        LocalMLRuntime()
        if args.no_onnx
        else LocalMLRuntime.live_from_artifacts(allow_fallback=True)
    )

    pipeline = CaseProcessingPipeline(
        ml_runtime=ml_runtime,
        defer_execution=True,
    )
    worker = CaseReadyWorker(consumer, pipeline)
    try:
        while True:
            count = worker.consume_once()
            if count:
                print(f"Processed {count} case-ready event(s).")
            if args.once:
                return 0
            if not count:
                time.sleep(args.poll_seconds)
    finally:
        broker.close()


if __name__ == "__main__":
    raise SystemExit(main())
