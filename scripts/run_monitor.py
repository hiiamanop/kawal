#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from services.monitor.app import create_monitor_app
from services.monitor.config import DEFAULT_DB_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="KAWAL Harness Monitor Console")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Port to listen on (default: 8088)",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="SQLite database path (default: runs/monitor/monitor.db)",
    )
    args = parser.parse_args()

    app = create_monitor_app(db_path=args.db_path)
    print(f"Starting KAWAL Monitor on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
