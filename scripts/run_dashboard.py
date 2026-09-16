#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from services.dashboard.app import create_dashboard_app


def main() -> int:
    parser = argparse.ArgumentParser(description="KAWAL Live Web Inspector & Dashboard")
    parser.add_argument(
        "--host",
        default=os.getenv("KAWAL_DASHBOARD_HOST", "0.0.0.0"),
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("KAWAL_DASHBOARD_PORT", "8088")),
        help="Port to listen on (default: 8088)",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("KAWAL_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres"),
        help="PostgreSQL connection DSN",
    )
    args = parser.parse_args()

    app = create_dashboard_app(database_url=args.database_url)
    print(f"Starting KAWAL Case Inspector on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
