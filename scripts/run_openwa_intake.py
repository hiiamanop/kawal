from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import uvicorn

from infra.db import TransactionRunner
from services.intake.api import create_live_intake_app
from services.intake.assembly import ConversationAssembler
from services.intake.openwa import OpenWAConnector
from services.intake.send_ledger import SendLedgerStore
from services.intake.service import IntakeService


def main() -> int:
    database_url = os.environ.get("KAWAL_DATABASE_URL")
    if not database_url:
        print("KAWAL_DATABASE_URL is required", file=sys.stderr)
        return 2
    account_id = os.environ.get("KAWAL_OPENWA_ACCOUNT_ID")
    if not account_id:
        print("KAWAL_OPENWA_ACCOUNT_ID is required", file=sys.stderr)
        return 2

    intake_service = IntakeService(
        transaction_runner=TransactionRunner(database_url),
        assembler=ConversationAssembler(),
    )
    connector = OpenWAConnector.live_from_environment(
        intake_service=intake_service,
        account_id=account_id,
        send_ledger=SendLedgerStore(),
    )
    app = create_live_intake_app(connector)
    host = os.environ.get("KAWAL_INTAKE_HOST", "0.0.0.0")
    port = int(os.environ.get("KAWAL_INTAKE_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
