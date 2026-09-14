#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra.db import TransactionRunner
from services.core.gateway_worker import ToolGatewayWorker
from services.intake.openwa import OpenWAConnector
from services.outbox.broker import RedpandaEventBroker
from services.outbox.consumer import AtomicInboxConsumer
from services.reliability.client import ReliableTicketClient
from services.simulator.store import TicketSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gateway-worker")


def main() -> int:
    parser = argparse.ArgumentParser(description="KAWAL Tool Gateway Worker CLI")
    parser.add_argument(
        "--group-id",
        default="kawal-gateway-worker-v1",
        help="Consumer group ID for Redpanda",
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
        help="Sleep interval in seconds between empty polls",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll cycle across commands and exit",
    )
    args = parser.parse_args()

    dsn = os.environ.get("KAWAL_DATABASE_URL")
    if not dsn:
        logger.error("KAWAL_DATABASE_URL environment variable is required")
        return 1

    bootstrap = os.environ.get("REDPANDA_BOOTSTRAP_SERVERS", "127.0.0.1:19092")
    runner = TransactionRunner(dsn)
    broker = RedpandaEventBroker(bootstrap_servers=bootstrap)
    consumer = AtomicInboxConsumer(
        transaction_runner=runner,
        broker=broker,
        consumer_name=args.group_id,
    )

    ticket_client = ReliableTicketClient(simulator=TicketSimulator())
    messaging_connector = OpenWAConnector()

    worker = ToolGatewayWorker(
        consumer=consumer,
        ticket_client=ticket_client,
        messaging_connector=messaging_connector,
    )

    if args.once:
        t_count = worker.consume_ticket_once(max_records=args.batch_size)
        m_count = worker.consume_message_once(max_records=args.batch_size)
        print(f"Processed {t_count} ticket command(s), {m_count} message command(s).")
        return 0

    logger.info("Starting ToolGatewayWorker loop on %s with group %s...", bootstrap, args.group_id)
    try:
        while True:
            t_count = worker.consume_ticket_once(max_records=args.batch_size)
            m_count = worker.consume_message_once(max_records=args.batch_size)
            if t_count == 0 and m_count == 0:
                time.sleep(args.poll_interval)
            else:
                logger.info("Processed %d ticket and %d message command(s)", t_count, m_count)
    except KeyboardInterrupt:
        logger.info("Stopping ToolGatewayWorker on interrupt.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
