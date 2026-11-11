#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra.db import TransactionRunner
from services.core.gateway_worker import ToolGatewayWorker
from services.core.model_gateway import ModelGateway, OpenRouterAdapter
from services.core.opa import OpaPolicyClient
from services.intake.openwa import OpenWAConnector, OpenWAHttpTransport
from services.intake.send_ledger import SendLedgerStore
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
        "--openrouter-model",
        default=os.getenv("KAWAL_OPENROUTER_MODEL") or os.getenv("OPENROUTER_MODEL"),
        help="OpenRouter model used for gated escalation commands (e.g. google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--openrouter-url",
        default=os.getenv("KAWAL_OPENROUTER_BASE_URL") or os.getenv("OPENROUTER_BASE_URL"),
        help="OpenRouter API base URL",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key (defaults to OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--omniroute-model",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--omniroute-url",
        default=None,
        help=argparse.SUPPRESS,
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
    if os.getenv("KAWAL_OPENWA_BASE_URL") and os.getenv("KAWAL_OPENWA_API_KEY") and os.getenv("KAWAL_OPENWA_SESSION_ID"):
        logger.info("Initializing live OpenWAConnector with HTTP transport...")
        messaging_connector = OpenWAConnector(
            account_id=os.getenv("KAWAL_OPENWA_ACCOUNT_ID", "research"),
            send_ledger=SendLedgerStore(),
            transport=OpenWAHttpTransport.from_environment().send_text,
        )
    else:
        logger.info("Initializing mock OpenWAConnector...")
        messaging_connector = OpenWAConnector()

    model_gateway = None
    openrouter_model_id = args.openrouter_model or args.omniroute_model
    if openrouter_model_id:
        model_gateway = ModelGateway(
            OpenRouterAdapter(
                openrouter_model_id,
                base_url=args.openrouter_url or args.omniroute_url,
                api_key=args.openrouter_api_key,
            )
        )

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
