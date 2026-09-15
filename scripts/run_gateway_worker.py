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
from services.core.model_gateway import ModelGateway, OllamaLocalAdapter, OmniRouteAdapter
from services.core.opa import OpaPolicyClient
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
        "--bootstrap-servers",
        default=os.getenv("KAWAL_REDPANDA_BOOTSTRAP", os.getenv("REDPANDA_BOOTSTRAP_SERVERS", "127.0.0.1:19092")),
        help="Redpanda bootstrap servers",
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
    parser.add_argument(
        "--opa-url",
        default=os.getenv("KAWAL_OPA_URL", "http://127.0.0.1:8181"),
        help="OPA Data API base URL used as the ticket hard gate",
    )
    parser.add_argument(
        "--no-opa",
        action="store_true",
        help="Use the in-process policy evaluator instead of OPA",
    )
    parser.add_argument(
        "--omniroute-model",
        default=os.getenv("KAWAL_OMNIROUTE_MODEL"),
        help="Optional OmniRoute model (e.g. gpt-4o-mini) for gated escalation commands",
    )
    parser.add_argument(
        "--omniroute-url",
        default=os.getenv("KAWAL_OMNIROUTE_URL", "http://localhost:20128/v1"),
        help="OmniRoute base URL",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("KAWAL_OLLAMA_MODEL"),
        help="Optional local Ollama model for gated escalation commands",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("KAWAL_OLLAMA_URL", "http://127.0.0.1:11434"),
        help="Local Ollama API base URL",
    )
    args = parser.parse_args()

    dsn = os.environ.get("KAWAL_DATABASE_URL")
    if not dsn:
        logger.error("KAWAL_DATABASE_URL environment variable is required")
        return 1

    bootstrap = args.bootstrap_servers
    runner = TransactionRunner(dsn)
    broker = RedpandaEventBroker(bootstrap_servers=bootstrap)
    consumer = AtomicInboxConsumer(
        transaction_runner=runner,
        broker=broker,
        consumer_name=args.group_id,
    )

    ticket_client = ReliableTicketClient(simulator=TicketSimulator())
    messaging_connector = OpenWAConnector()

    model_gateway = None
    if args.omniroute_model:
        model_gateway = ModelGateway(OmniRouteAdapter(args.omniroute_model, base_url=args.omniroute_url))
    elif args.ollama_model:
        model_gateway = ModelGateway(OllamaLocalAdapter(args.ollama_model, base_url=args.ollama_url))

    worker = ToolGatewayWorker(
        consumer=consumer,
        ticket_client=ticket_client,
        messaging_connector=messaging_connector,
        policy_client=None if args.no_opa else OpaPolicyClient(base_url=args.opa_url),
        model_gateway=model_gateway,
    )

    if args.once:
        t_count = worker.consume_ticket_once(max_records=args.batch_size)
        m_count = worker.consume_message_once(max_records=args.batch_size)
        e_count = worker.consume_escalation_once(max_records=args.batch_size)
        print(f"Processed {t_count} ticket, {m_count} message, {e_count} escalation command(s).")
        return 0

    logger.info("Starting ToolGatewayWorker loop on %s with group %s...", bootstrap, args.group_id)
    try:
        while True:
            t_count = worker.consume_ticket_once(max_records=args.batch_size)
            m_count = worker.consume_message_once(max_records=args.batch_size)
            e_count = worker.consume_escalation_once(max_records=args.batch_size)
            if t_count == 0 and m_count == 0 and e_count == 0:
                time.sleep(args.poll_interval)
            else:
                logger.info("Processed %d ticket, %d message, and %d escalation command(s)", t_count, m_count, e_count)
    except KeyboardInterrupt:
        logger.info("Stopping ToolGatewayWorker on interrupt.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
