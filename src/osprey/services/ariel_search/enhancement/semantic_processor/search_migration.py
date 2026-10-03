"""Reconcile semantic processor keyword-search indexes.

Deployments that applied the original semantic processor migration already have
its migration record, so they need a separate additive migration to rebuild the
keyword FTS expression and remove the unqueried keyword-array index.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import BaseMigration
from osprey.services.ariel_search.database.search_fts import (
    SEMANTIC_FTS_EXPRESSION_V1_FROZEN,
    SEMANTIC_FTS_EXPRESSION_V2,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class SemanticProcessorSearchMigration(BaseMigration):
    """Rebuilds semantic keyword-search indexes for summary and keywords."""

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "semantic_processor_search_index"

    @property
    def depends_on(self) -> list[str]:
        """Depends on semantic processor columns being present."""
        return ["semantic_processor"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the enriched semantic FTS index migration."""
        await conn.execute(
            """
            CREATE OR REPLACE FUNCTION osprey_text_array_to_string(TEXT[])
            RETURNS TEXT
            LANGUAGE sql
            IMMUTABLE
            PARALLEL SAFE
            RETURNS NULL ON NULL INPUT
            AS $$ SELECT array_to_string($1, ' ') $$
            """
        )
        await conn.execute("DROP INDEX IF EXISTS idx_entries_keywords")
        await conn.execute("DROP INDEX IF EXISTS idx_entries_text_search")
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_entries_text_search
            ON enhanced_entries
            USING GIN({SEMANTIC_FTS_EXPRESSION_V1_FROZEN})
            """
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the enriched semantic FTS index migration."""
        await conn.execute("DROP INDEX IF EXISTS idx_entries_text_search")


class SemanticProcessorSearchIndexV2Migration(BaseMigration):
    """Builds the v2 semantic keyword-search index over ``attachment_text``.

    Creates ``idx_entries_text_search_v2`` over ``SEMANTIC_FTS_EXPRESSION_V2``;
    the v1 ``idx_entries_text_search`` is kept. Built like the raw-text v2
    index: under a non-queueing ``SHARE`` lock, so reads proceed during the
    build, with ``maintenance_work_mem`` raised for its transaction.
    """

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "semantic_processor_search_index_v2"

    @property
    def depends_on(self) -> list[str]:
        """Depends on the fold and on the v1 index migration's helper function."""
        return ["attachment_text_upstream_fold", "semantic_processor_search_index"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Create the v2 index.

        Raises:
            MigrationBusyError: If ``enhanced_entries`` stayed busy for every
                lock attempt; the runner leaves the migration unapplied.
        """
        from osprey.services.ariel_search.database.attachment_text_migration import (
            build_kept_indexes,
        )

        await build_kept_indexes(
            conn,
            [
                f"""
                CREATE INDEX IF NOT EXISTS idx_entries_text_search_v2
                ON enhanced_entries
                USING GIN({SEMANTIC_FTS_EXPRESSION_V2})
                """
            ],
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Drop the v2 index; the v1 index stays."""
        await conn.execute("DROP INDEX IF EXISTS idx_entries_text_search_v2")
