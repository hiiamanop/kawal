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
from services.core.persistence import M1Store
from services.outbox.broker import (
    RedpandaBrokerUnavailableError,
    RedpandaEventBroker,
)
from services.outbox.relay import EVENT_TYPE_TO_TOPIC, OutboxRelayDaemon


def main() -> int:
    parser = argparse.ArgumentParser(description="KAWAL transactional outbox relay daemon for Redpanda")
    parser.add_argument(
        "--database-url",
        default=os.getenv("KAWAL_DATABASE_URL"),
        help="PostgreSQL/Supabase connection URL; defaults to KAWAL_DATABASE_URL",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAWAL_REDPANDA_BOOTSTRAP", "127.0.0.1:19092"),
        help="Redpanda Kafka bootstrap server",
    )
    parser.add_argument("--once", action="store_true", help="Relay current batch once and exit")
    parser.add_argument("--poll-seconds", type=float, default=0.5, help="Idle polling interval")
    parser.add_argument("--batch-size", type=int, default=50, help="Maximum events relayed per iteration")
    args = parser.parse_args()

    if not args.database_url:
        parser.error("--database-url or KAWAL_DATABASE_URL is required")

    broker = RedpandaEventBroker(bootstrap_servers=args.bootstrap_servers)
    try:
        broker.ensure_topics(tuple(sorted(set(EVENT_TYPE_TO_TOPIC.values()))))
    except RedpandaBrokerUnavailableError as exc:
        print(f"Redpanda unavailable: {exc}", file=sys.stderr)
        return 2

    relay = OutboxRelayDaemon(
        transaction_runner=TransactionRunner(args.database_url),
        store=M1Store(),
        broker=broker,
    )

    try:
        while True:
            count = relay.relay_batch(max_batch=args.batch_size)
            if count:
                print(f"Relayed {count} event(s) to Redpanda.")
            if args.once:
                return 0
            if not count:
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        broker.close()


if __name__ == "__main__":
    raise SystemExit(main())
