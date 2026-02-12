"""ARIEL text embedding migration.

This module provides the database migration for the text embedding
enhancement module.

See 01_DATA_LAYER.md Section 2.5 for SQL specification.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migration import BaseMigration, model_to_table_name

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class TextEmbeddingMigration(BaseMigration):
    """Text embedding enhancement migration.

    Creates:
    - pgvector extension
    - text_embeddings_<model_name> table for each configured model
    - IVFFlat vector indexes
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

        In production, this would be populated from config. For MVP,
        we use a default model list.

        Returns:
            List of (model_name, dimension) tuples
        """
        if self._models:
            return self._models
        # Default: nomic-embed-text (most common)
        return [("nomic-embed-text", 768)]

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the text embedding migration."""
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        models = self._get_models()
        for model_name, dimension in models:
            table_name = model_to_table_name(model_name)

            # Create embedding table
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id              SERIAL PRIMARY KEY,
                    entry_id        TEXT NOT NULL REFERENCES enhanced_entries(entry_id) ON DELETE CASCADE,
                    embedding       vector({dimension}),
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(entry_id)
                )
                """  # noqa: S608
            )

            # Create IVFFlat index for vector similarity search
            # MVP default: lists=224 (sqrt of 50K entries)
            index_name = f"idx_{table_name}_vector"
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 224)
                """  # noqa: S608
            )

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the text embedding migration."""
        models = self._get_models()
        for model_name, _dimension in models:
            table_name = model_to_table_name(model_name)
            await conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")  # noqa: S608

        # Note: We don't drop the vector extension as other things may use it
