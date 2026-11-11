#!/usr/bin/env python3
"""Collect public social-media conversations through Agent Reach.

Examples:
    python scripts/run_social_collector.py --once
    python scripts/run_social_collector.py --channels twitter,facebook --query "banjir bandung" --limit 5
    python scripts/run_social_collector.py --dry-run --query "jalan rusak bandung"
"""

from __future__ import annotations

import argparse
import json
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

from services.intake.social_reach import (
    AgentReachClient,
    AgentReachError,
    PostgresSocialPostStore,
    parse_channels,
    parse_queries,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION = REPO_ROOT / "infra" / "migrations" / "013_m9_social_media_posts.sql"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [social-collector] %(message)s",
)
logger = logging.getLogger("social-collector")


def ensure_schema(database_url: str) -> None:
    import psycopg

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = 'social_media_posts'"
            )
            if cursor.fetchone():
                return
            if not MIGRATION.is_file():
                raise RuntimeError(f"Migration not found: {MIGRATION}")
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        connection.commit()
    logger.info("Applied %s", MIGRATION.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="KAWAL Agent Reach social-media collector")
    parser.add_argument(
        "--database-url",
        default=os.getenv("KAWAL_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5432/postgres"),
        help="PostgreSQL DSN used to persist collected posts",
    )
    parser.add_argument(
        "--channels",
        default=os.getenv("KAWAL_SOCIAL_CHANNELS", "twitter,facebook,instagram"),
        help="Comma-separated Agent Reach channels",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        default=None,
        help="Search query; repeat for multiple queries (default: KAWAL_SOCIAL_QUERIES)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max items per query/channel")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=0.0,
        help="Run continuously with this delay; 0 runs one collection cycle",
    )
    parser.add_argument(
        "--agent-reach-bin",
        default=os.getenv("KAWAL_AGENT_REACH_BIN"),
        help="Path to the agent-reach executable",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("KAWAL_AGENT_REACH_TIMEOUT_SECONDS", "60")),
        help="Timeout per agent-reach invocation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print posts without persisting")
    args = parser.parse_args()

    channels = parse_channels(args.channels)
    queries = parse_queries(args.queries or os.getenv("KAWAL_SOCIAL_QUERIES"))
    if not queries:
        parser.error("At least one --query or KAWAL_SOCIAL_QUERIES is required")

    store = None
    if not args.dry_run:
        ensure_schema(args.database_url)
        store = PostgresSocialPostStore(args.database_url)

    client = AgentReachClient(
        binary=args.agent_reach_bin,
        timeout_seconds=args.timeout_seconds,
    )

    while True:
        started = time.time()
        try:
            result = client.collect(queries=queries, channels=channels, limit=args.limit)
            logger.info(
                "Collected %d post(s) from %d channel(s) in %.1fs",
                len(result.posts),
                len(channels),
                time.time() - started,
            )
            for error in result.errors:
                logger.warning("Agent Reach source failed: %s", error)

            if args.dry_run:
                print(json.dumps([post.model_dump() for post in result.posts], indent=2, ensure_ascii=False, default=str))
            elif store is not None and result.posts:
                written = store.upsert_many(result.posts)
                logger.info("Persisted %d post(s) into social_media_posts", written)
        except AgentReachError as exc:
            logger.error("Agent Reach collector failed: %s", exc)
            if args.interval_seconds <= 0:
                return 1
        except Exception as exc:  # noqa: BLE001 - long-running collector must survive one bad cycle
            logger.exception("Unexpected collector failure: %s", exc)
            if args.interval_seconds <= 0:
                return 1

        if args.interval_seconds <= 0:
            return 0
        time.sleep(max(1.0, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
