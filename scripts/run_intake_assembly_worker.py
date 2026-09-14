#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra.db import TransactionRunner
from services.intake.worker import IntakeAssemblyWorker
from services.outbox.broker import RedpandaEventBroker
from services.outbox.consumer import AtomicInboxConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("intake-assembly-worker")


def main() -> int:
    parser = argparse.ArgumentParser(description="KAWAL Intake Assembly Worker CLI")
    parser.add_argument(
        "--database-url",
        default=os.getenv("KAWAL_DATABASE_URL"),
        help="PostgreSQL connection DSN",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAWAL_REDPANDA_BOOTSTRAP", "127.0.0.1:19092"),
        help="Redpanda bootstrap servers",
    )
    parser.add_argument(
        "--group-id",
        default="kawal-intake-assembly-v1",
        help="Consumer group ID for intake.messages.v1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size per poll cycle",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Sleep interval in seconds between polls",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll and timer cycle and exit",
    )
    args = parser.parse_args()

    if not args.database_url:
        parser.error("--database-url or KAWAL_DATABASE_URL is required")

    runner = TransactionRunner(args.database_url)
    broker = RedpandaEventBroker(bootstrap_servers=args.bootstrap_servers)
    consumer = AtomicInboxConsumer(
        transaction_runner=runner,
        broker=broker,
        consumer_name=args.group_id,
    )

    worker = IntakeAssemblyWorker(
        transaction_runner=runner,
        consumer=consumer,
    )

    if args.once:
        msg_count, assembled = worker.step(max_records=args.batch_size)
        print(f"Processed {msg_count} intake message(s), assembled {len(assembled)} conversation(s).")
        return 0

    logger.info("Starting IntakeAssemblyWorker on %s with group %s...", args.bootstrap_servers, args.group_id)
    try:
        while True:
            msg_count, assembled = worker.step(max_records=args.batch_size)
            if msg_count == 0 and len(assembled) == 0:
                time.sleep(args.poll_interval)
            else:
                logger.info("Processed %d intake messages, assembled conversations: %s", msg_count, assembled)
    except KeyboardInterrupt:
        logger.info("Stopping IntakeAssemblyWorker on interrupt.")
        return 0
    finally:
        broker.close()


if __name__ == "__main__":
    raise SystemExit(main())
