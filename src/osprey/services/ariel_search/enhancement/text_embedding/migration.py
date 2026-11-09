"""ARIEL text embedding migration.

This module provides the database migration for the text embedding
enhancement module.
"""

import re
from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    MigrationSkippedError,
    model_to_table_name,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection


def vector_index_name(table_name: str) -> str:
    """Return the name of the HNSW embedding index on ``table_name``."""
    return f"idx_{table_name}_hnsw"


def legacy_vector_index_name(table_name: str) -> str:
    """Return the name the superseded IVFFlat index was created under."""
    return f"idx_{table_name}_vector"


def create_vector_index_sql(table_name: str) -> str:
    """Return the DDL creating the embedding index on ``table_name``.

    The single producer of that statement: the creating migration and the
    reconcile migration must write the same index, so neither spells it out.
    """
    return (
        f"CREATE INDEX IF NOT EXISTS {vector_index_name(table_name)} "
        f"ON {table_name} USING hnsw (embedding vector_cosine_ops)"
    )


#: The first pgvector release with the HNSW access method every embedding
#: index here is built with.
HNSW_MIN_PGVECTOR = (0, 5, 0)


def _version_key(version: str) -> tuple[int, ...]:
    """``"0.4.4"`` → ``(0, 4, 4)``; a non-numeric tail (``"0.5.0-dev"``) is ignored."""
    parts: list[int] = []
    for part in version.split("."):
        digits = re.match(r"\d+", part)
        if digits is None:
            break
        parts.append(int(digits.group()))
    return tuple(parts)


async def require_pgvector_hnsw(conn: "AsyncConnection", unavailable: str) -> None:
    """Refuse, as a skip, unless pgvector is available at a version with HNSW.

    The version judged is the installed one, else the one ``CREATE EXTENSION``
    would install. Below 0.5 the ``CREATE INDEX ... USING hnsw`` would fail
    with a bare "access method does not exist"; the skip names the version
    instead, and the migration retries once pgvector is upgraded.

    Args:
        conn: The migration's connection.
        unavailable: The skip message for a server with no pgvector at all.

    Raises:
        MigrationSkippedError: With `unavailable` when pgvector is absent, or
            naming the version found and the one needed when it is too old.
    """
    result = await conn.execute(
        "SELECT installed_version, default_version FROM pg_available_extensions "
        "WHERE name = 'vector'"
    )
    row = await result.fetchone()
    if not row:
        raise MigrationSkippedError(unavailable)
    installed, default = row[0], row[1]
    version = installed or default
    if isinstance(version, str) and _version_key(version) < HNSW_MIN_PGVECTOR:
        needed = ".".join(str(n) for n in HNSW_MIN_PGVECTOR)
        how = (
            "upgrade the pgvector package, then run ALTER EXTENSION vector UPDATE"
            if installed
            else "upgrade the pgvector package"
        )
        raise MigrationSkippedError(
            f"pgvector {version} is {'installed' if installed else 'available'} in this "
            f"PostgreSQL installation, but ARIEL's embedding indexes use HNSW, which "
            f"needs pgvector {needed} or newer: {how}. Until then ARIEL falls back to "
            "keyword-only search."
        )


#: The skip message of a text-embedding migration on a server without pgvector.
PGVECTOR_UNAVAILABLE_MESSAGE = (
    "pgvector extension is not available in this PostgreSQL installation. "
    "Install pgvector to enable semantic search. "
    "ARIEL will fall back to keyword-only search."
)


class TextEmbeddingMigration(BaseMigration):
    """Text embedding enhancement migration.

    Creates:
    - pgvector extension
    - text_embeddings_<model_name> table for each configured model
    - HNSW vector indexes
    """

    def __init__(self, models: list[tuple[str, int]] | None = None) -> None:
        """Initialize the migration.

        Args:
            models: List of (model_name, dimension) tuples to create tables for.
                   If None, uses a default for testing.
        """
        super().__init__()
        self._models = models

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "text_embedding"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema."""
        return ["core_schema"]

    def _get_models(self) -> list[tuple[str, int]]:
        """Get the list of models to create tables for.

        Returns:
            List of (model_name, dimension) tuples
        """
        if self._models:
            return self._models
        # Default: nomic-embed-text (most common)
        return [("nomic-embed-text", 768)]

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the text embedding migration."""
        await require_pgvector_hnsw(conn, PGVECTOR_UNAVAILABLE_MESSAGE)

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        models = self._get_models()
        for model_name, dimension in models:
            table_name = model_to_table_name(model_name)

            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id              SERIAL PRIMARY KEY,
                    entry_id        TEXT NOT NULL REFERENCES enhanced_entries(entry_id) ON DELETE CASCADE,
                    embedding       vector({dimension}),
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    input_key       TEXT NULL,
                    UNIQUE(entry_id)
                )
                """
            )

            await conn.execute(create_vector_index_sql(table_name))

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the text embedding migration."""
        models = self._get_models()
        for model_name, _dimension in models:
            table_name = model_to_table_name(model_name)
            await conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

        # Note: We don't drop the vector extension as other things may use it
