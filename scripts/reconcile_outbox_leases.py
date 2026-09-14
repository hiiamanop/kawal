from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.db import TransactionRunner
from services.core.persistence import M1Store


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Release invalid far-future transactional outbox leases safely"
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("KAWAL_DATABASE_URL"),
        help="PostgreSQL/Supabase URL; defaults to KAWAL_DATABASE_URL",
    )
    parser.add_argument(
        "--max-lease-seconds",
        type=float,
        default=600.0,
        help="Release only unpublished leases farther than this duration (default: 600)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply reconciliation; without this flag only prints the intended action",
    )
    args = parser.parse_args()

    if not args.database_url:
        parser.error("--database-url or KAWAL_DATABASE_URL is required")
    if args.max_lease_seconds <= 0:
        parser.error("--max-lease-seconds must be positive")

    if not args.apply:
        print(
            "Dry run: no leases changed. Re-run with --apply to release unpublished "
            f"leases farther than {args.max_lease_seconds:.0f} seconds in the future."
        )
        return 0

    released = TransactionRunner(args.database_url).run(
        lambda conn: M1Store().reconcile_stuck_outbox_leases(
            conn,
            max_lease_duration=timedelta(seconds=args.max_lease_seconds),
        )
    )
    print(f"Released {len(released)} stuck outbox lease(s): {', '.join(map(str, released)) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
