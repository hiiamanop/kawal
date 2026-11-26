from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INFRA_MIGRATION = (
    REPO_ROOT / "infra" / "migrations" / "014_m6_harness_semantic_cache.sql"
)
SUPABASE_MIGRATION = (
    REPO_ROOT
    / "supabase"
    / "migrations"
    / "20261126000000_m6_harness_semantic_cache.sql"
)
COMPOSE = REPO_ROOT / "docker-compose.yml"


def test_migration_exists_in_both_trees() -> None:
    assert INFRA_MIGRATION.is_file()
    assert SUPABASE_MIGRATION.is_file()


def test_migration_uses_pgvector_and_hnsw() -> None:
    sql = INFRA_MIGRATION.read_text(encoding="utf-8").lower()
    assert "harness_case_cache" in sql
    assert "embedding vector(768)" in sql
    assert "using hnsw" in sql
    assert "vector_cosine_ops" in sql
    assert "record_guards jsonb" in sql


def test_migrations_are_equivalent() -> None:
    assert INFRA_MIGRATION.read_text(encoding="utf-8") == SUPABASE_MIGRATION.read_text(
        encoding="utf-8"
    )


def test_compose_uses_pgvector_image() -> None:
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "pgvector/pgvector:pg16" in compose
