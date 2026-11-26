from __future__ import annotations

DEFAULT_DB_PATH = "runs/monitor/monitor.db"

DEFAULT_QUERIES = (
    "jalan rusak bandung",
    "trotoar rusak bandung",
    "sampah menumpuk bandung",
    "banjir bandung",
    "lampu jalan mati bandung",
    "pungli bandung",
)

ENGAGEMENT_WEIGHT = 0.45
RECENCY_WEIGHT = 0.25
RISK_WEIGHT = 0.20
TRUST_WEIGHT = 0.10
RECENCY_WINDOW_HOURS = 72.0
