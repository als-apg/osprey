"""ARIEL repository for database operations.

This module provides the ARIELRepository class with async CRUD operations
for the ARIEL database.

See 04_OSPREY_INTEGRATION.md Section 6 and 01_DATA_LAYER.md Section 4
for specification.
"""

import functools
import json
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from osprey.services.ariel_search.exceptions import (
    DatabaseQueryError,
    ModuleNotEnabledError,
)
from osprey.services.ariel_search.models import (
    EmbeddingTableInfo,
    EnhancedLogbookEntry,
    enhanced_entry_from_row,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from osprey.services.ariel_search.config import ARIELConfig

logger = get_logger("ariel")

F = TypeVar("F", bound=Callable[..., Any])


def requires_module(module_type: str, module_name: str) -> Callable[[F], F]:
    """Decorator that checks if required module is enabled.

    Args:
        module_type: Type of module ('search' or 'enhancement')
        module_name: Name of the module

    Returns:
        Decorated function that raises ModuleNotEnabledError if module disabled
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: "ARIELRepository", *args: Any, **kwargs: Any) -> Any:
            if module_type == "search":
                enabled = self.config.is_search_module_enabled(module_name)
            elif module_type == "enhancement":
                enabled = self.config.is_enhancement_module_enabled(module_name)
            else:
                enabled = False

            if not enabled:
                raise ModuleNotEnabledError(
                    f"Module '{module_name}' is not enabled. "
                    f"Enable it in config.yml under {module_type}_modules.{module_name}",
                    module_name=module_name,
                )
            return func(self, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


class ARIELRepository:
    """Repository for ARIEL database operations.

    Provides async CRUD operations for enhanced logbook entries.
    Methods are available based on enabled modules in config.

    Attributes:
        pool: Database connection pool
        config: ARIEL configuration
    """

    def __init__(self, pool: "AsyncConnectionPool", config: "ARIELConfig") -> None:
        """Initialize the repository.

        Args:
            pool: Database connection pool
            config: ARIEL configuration
        """
        self.pool = pool
        self.config = config

    # === Core Methods (always available) ===

    async def get_entry(self, entry_id: str) -> EnhancedLogbookEntry | None:
        """Get a single entry by ID.

        Args:
            entry_id: The entry ID

        Returns:
            EnhancedLogbookEntry or None if not found
        """
        from psycopg.rows import dict_row

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "SELECT * FROM enhanced_entries WHERE entry_id = %s",
                        [entry_id],
                    )
                    row = await cur.fetchone()
                    return enhanced_entry_from_row(row) if row else None
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get entry {entry_id}: {e}",
                query=f"SELECT entry_id={entry_id}",
            ) from e

    async def get_entries_by_ids(self, entry_ids: list[str]) -> list[EnhancedLogbookEntry]:
        """Get multiple entries by their IDs.

        Args:
            entry_ids: List of entry IDs

        Returns:
            List of EnhancedLogbookEntry (may be fewer than requested if some not found)
        """
        from psycopg.rows import dict_row

        if not entry_ids:
            return []

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "SELECT * FROM enhanced_entries WHERE entry_id = ANY(%s)",
                        [entry_ids],
                    )
                    rows = await cur.fetchall()
                    return [enhanced_entry_from_row(row) for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get entries by IDs: {e}",
                query=f"SELECT entry_ids=ANY([{len(entry_ids)} ids])",
            ) from e

    async def upsert_entry(self, entry: EnhancedLogbookEntry) -> None:
        """Insert or update an entry.

        Args:
            entry: The entry to upsert
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO enhanced_entries (
                        entry_id, source_system, timestamp, author, raw_text,
                        attachments, metadata, enhancement_status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (entry_id) DO UPDATE SET
                        source_system = EXCLUDED.source_system,
                        timestamp = EXCLUDED.timestamp,
                        author = EXCLUDED.author,
                        raw_text = EXCLUDED.raw_text,
                        attachments = EXCLUDED.attachments,
                        metadata = EXCLUDED.metadata,
                        enhancement_status = EXCLUDED.enhancement_status
                    """,
                    [
                        entry["entry_id"],
                        entry["source_system"],
                        entry["timestamp"],
                        entry.get("author", ""),
                        entry["raw_text"],
                        json.dumps(entry.get("attachments", [])),
                        json.dumps(entry.get("metadata", {})),
                        json.dumps(entry.get("enhancement_status", {})),
                    ],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to upsert entry {entry['entry_id']}: {e}",
                query=f"UPSERT entry_id={entry['entry_id']}",
            ) from e

    async def search_by_time_range(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[EnhancedLogbookEntry]:
        """Get entries within a time range.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (inclusive)
            limit: Maximum entries to return

        Returns:
            List of EnhancedLogbookEntry sorted by timestamp descending
        """
        from psycopg.rows import dict_row

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    conditions = []
                    params: list[Any] = []

                    if start is not None:
                        conditions.append("timestamp >= %s")
                        params.append(start)
                    if end is not None:
                        conditions.append("timestamp <= %s")
                        params.append(end)

                    where_clause = " AND ".join(conditions) if conditions else "TRUE"
                    params.append(limit)

                    await cur.execute(
                        f"""
                        SELECT * FROM enhanced_entries
                        WHERE {where_clause}
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        params,
                    )
                    rows = await cur.fetchall()
                    return [enhanced_entry_from_row(row) for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to search by time range: {e}",
                query=f"SELECT time_range=({start}, {end})",
            ) from e

    async def count_entries(self) -> int:
        """Count total entries in the database.

        Returns:
            Total number of entries
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute("SELECT COUNT(*) FROM enhanced_entries")
                row = await result.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to count entries: {e}",
                query="SELECT COUNT(*)",
            ) from e

    async def get_distinct_authors(self) -> list[str]:
        """Get distinct author values from the database.

        Returns:
            Sorted list of unique author names
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    "SELECT DISTINCT author FROM enhanced_entries "
                    "WHERE author IS NOT NULL AND author != '' "
                    "ORDER BY author"
                )
                rows = await result.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get distinct authors: {e}",
                query="SELECT DISTINCT author",
            ) from e

    async def get_distinct_source_systems(self) -> list[str]:
        """Get distinct source_system values from the database.

        Returns:
            Sorted list of unique source system names
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    "SELECT DISTINCT source_system FROM enhanced_entries "
                    "WHERE source_system IS NOT NULL AND source_system != '' "
                    "ORDER BY source_system"
                )
                rows = await result.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get distinct source systems: {e}",
                query="SELECT DISTINCT source_system",
            ) from e

    # === Enhancement Status Methods ===

    async def get_incomplete_entries(
        self,
        module_name: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[EnhancedLogbookEntry]:
        """Get entries with incomplete or failed enhancements.

        Args:
            module_name: Filter by specific module (optional)
            status: Filter by status ('failed', 'pending') (optional)
            limit: Maximum entries to return

        Returns:
            List of entries needing enhancement
        """
        from psycopg.rows import dict_row

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    if module_name and status:
                        await cur.execute(
                            """
                            SELECT * FROM enhanced_entries
                            WHERE enhancement_status->%s->>'status' = %s
                            ORDER BY created_at ASC
                            LIMIT %s
                            """,
                            [module_name, status, limit],
                        )
                    elif module_name:
                        await cur.execute(
                            """
                            SELECT * FROM enhanced_entries
                            WHERE NOT (enhancement_status ? %s)
                               OR enhancement_status->%s->>'status' IN ('failed', 'pending')
                            ORDER BY created_at ASC
                            LIMIT %s
                            """,
                            [module_name, module_name, limit],
                        )
                    else:
                        await cur.execute(
                            """
                            SELECT * FROM enhanced_entries
                            ORDER BY created_at ASC
                            LIMIT %s
                            """,
                            [limit],
                        )
                    rows = await cur.fetchall()
                    return [enhanced_entry_from_row(row) for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get incomplete entries: {e}",
                query=f"SELECT incomplete module={module_name}",
            ) from e

    async def get_enhancement_stats(self) -> dict[str, Any]:
        """Get statistics about enhancement completion.

        Returns:
            Dict with stats per module
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total_entries,
                        COUNT(*) FILTER (
                            WHERE enhancement_status->'text_embedding'->>'status' = 'complete'
                        ) AS text_embedding_complete,
                        COUNT(*) FILTER (
                            WHERE enhancement_status->'text_embedding'->>'status' = 'failed'
                        ) AS text_embedding_failed,
                        COUNT(*) FILTER (
                            WHERE enhancement_status->'text_embedding'->>'status' = 'pending'
                            OR NOT enhancement_status ? 'text_embedding'
                        ) AS text_embedding_pending,
                        COUNT(*) FILTER (
                            WHERE enhancement_status->'semantic_processor'->>'status' = 'complete'
                        ) AS semantic_processor_complete,
                        COUNT(*) FILTER (
                            WHERE enhancement_status->'semantic_processor'->>'status' = 'failed'
                        ) AS semantic_processor_failed,
                        COUNT(*) FILTER (
                            WHERE enhancement_status->'semantic_processor'->>'status' = 'pending'
                            OR NOT enhancement_status ? 'semantic_processor'
                        ) AS semantic_processor_pending
                    FROM enhanced_entries
                    """
                )
                row = await result.fetchone()
                if not row:
                    return {"total_entries": 0}

                return {
                    "total_entries": row[0],
                    "text_embedding": {
                        "complete": row[1],
                        "failed": row[2],
                        "pending": row[3],
                    },
                    "semantic_processor": {
                        "complete": row[4],
                        "failed": row[5],
                        "pending": row[6],
                    },
                }
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get enhancement stats: {e}",
                query="SELECT enhancement_stats",
            ) from e

    async def mark_enhancement_complete(self, entry_id: str, module_name: str) -> None:
        """Mark an enhancement as complete for an entry.

        Args:
            entry_id: The entry ID
            module_name: The enhancement module name
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    UPDATE enhanced_entries
                    SET enhancement_status = jsonb_set(
                        enhancement_status,
                        %s,
                        jsonb_build_object('status', 'complete', 'completed_at', NOW()::text)
                    )
                    WHERE entry_id = %s
                    """,
                    [[module_name], entry_id],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to mark enhancement complete: {e}",
                query=f"UPDATE entry_id={entry_id} module={module_name}",
            ) from e

    async def mark_enhancement_failed(
        self,
        entry_id: str,
        module_name: str,
        error: str,
    ) -> None:
        """Mark an enhancement as failed for an entry.

        Args:
            entry_id: The entry ID
            module_name: The enhancement module name
            error: Error message
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    UPDATE enhanced_entries
                    SET enhancement_status = jsonb_set(
                        enhancement_status,
                        %s::text[],
                        jsonb_build_object(
                            'status', 'failed',
                            'failed_at', NOW()::text,
                            'error', %s::text
                        )
                    )
                    WHERE entry_id = %s
                    """,
                    [[module_name], error[:500], entry_id],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to mark enhancement failed: {e}",
                query=f"UPDATE entry_id={entry_id} module={module_name}",
            ) from e

    # === Embedding Methods ===

    async def get_embedding_tables(self) -> list[EmbeddingTableInfo]:
        """Discover all embedding tables in the database.

        Returns:
            List of EmbeddingTableInfo for each embedding table
        """
        try:
            async with self.pool.connection() as conn:
                # Find all embedding tables
                result = await conn.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name LIKE 'text_embeddings_%'
                    """
                )
                rows = await result.fetchall()

                tables: list[EmbeddingTableInfo] = []
                active_model = self.config.get_search_model()

                for row in rows:
                    table_name = row[0]

                    # Get entry count
                    count_result = await conn.execute(
                        f"SELECT COUNT(*) FROM {table_name}"  # noqa: S608
                    )
                    count_row = await count_result.fetchone()
                    entry_count = int(count_row[0]) if count_row else 0

                    # Get dimension from column type
                    dim_result = await conn.execute(
                        """
                        SELECT atttypmod
                        FROM pg_attribute
                        WHERE attrelid = %s::regclass
                        AND attname = 'embedding'
                        """,
                        [table_name],
                    )
                    dim_row = await dim_result.fetchone()
                    dimension = int(dim_row[0]) if dim_row and dim_row[0] > 0 else None

                    # Check if this is the active model
                    is_active = False
                    if active_model:
                        from osprey.services.ariel_search.database.migration import (
                            model_to_table_name,
                        )

                        is_active = table_name == model_to_table_name(active_model)

                    tables.append(
                        EmbeddingTableInfo(
                            table_name=table_name,
                            entry_count=entry_count,
                            dimension=dimension,
                            is_active=is_active,
                        )
                    )

                return tables
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get embedding tables: {e}",
                query="SELECT embedding tables",
            ) from e

    async def validate_search_model_table(self, model: str) -> None:
        """Validate that the embedding table for the model exists.

        Args:
            model: Model name to validate

        Raises:
            ConfigurationError: If table does not exist
        """
        from osprey.services.ariel_search.database.migration import model_to_table_name
        from osprey.services.ariel_search.exceptions import ConfigurationError

        table_name = model_to_table_name(model)

        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    )
                    """,
                    [table_name],
                )
                row = await result.fetchone()
                if not row or not row[0]:
                    raise ConfigurationError(
                        f"Embedding table '{table_name}' does not exist. "
                        f"The model '{model}' is configured for semantic search "
                        "but migrations have not been run. Run 'osprey ariel migrate'.",
                        config_key="search_modules.semantic.model",
                    )
        except ConfigurationError:
            raise
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to validate embedding table: {e}",
                query=f"SELECT table exists {table_name}",
            ) from e

    @requires_module("enhancement", "text_embedding")
    async def store_text_embedding(
        self,
        entry_id: str,
        embedding: list[float],
        model_name: str,
    ) -> None:
        """Store a text embedding for an entry.

        Args:
            entry_id: The entry ID
            embedding: The embedding vector
            model_name: The model name (determines table)
        """
        from osprey.services.ariel_search.database.migration import model_to_table_name

        table_name = model_to_table_name(model_name)

        try:
            async with self.pool.connection() as conn:
                # Format embedding as PostgreSQL array
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                await conn.execute(
                    f"""
                    INSERT INTO {table_name} (entry_id, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (entry_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        created_at = NOW()
                    """,  # noqa: S608
                    [entry_id, embedding_str],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to store text embedding: {e}",
                query=f"INSERT {table_name} entry_id={entry_id}",
            ) from e

    # === Search Methods ===

    @requires_module("search", "keyword")
    async def keyword_search(
        self,
        where_clauses: list[str],
        params: list[Any],
        search_text: str,
        max_results: int = 10,
        include_highlights: bool = True,
    ) -> list[tuple[EnhancedLogbookEntry, float, list[str]]]:
        """Execute keyword search using full-text search.

        Args:
            where_clauses: SQL WHERE conditions
            params: Query parameters
            search_text: Original search text for highlighting
            max_results: Maximum results to return
            include_highlights: Include highlighted snippets

        Returns:
            List of (entry, score, highlights) tuples
        """
        from psycopg.rows import dict_row

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

                    # Build query with FTS ranking
                    # Note: raw_text contains subject + details merged by adapter
                    if include_highlights:
                        query = f"""
                            SELECT e.*,
                                   ts_rank(
                                       to_tsvector('english', raw_text),
                                       plainto_tsquery('english', %s)
                                   ) AS rank,
                                   ts_headline('english', raw_text, plainto_tsquery('english', %s),
                                       'StartSel=<b>, StopSel=</b>, MaxFragments=3'
                                   ) AS headline
                            FROM enhanced_entries e
                            WHERE {where_sql}
                            ORDER BY rank DESC
                            LIMIT %s
                        """  # noqa: S608
                        all_params = [search_text, search_text] + params + [max_results]
                    else:
                        query = f"""
                            SELECT e.*,
                                   ts_rank(
                                       to_tsvector('english', raw_text),
                                       plainto_tsquery('english', %s)
                                   ) AS rank,
                                   NULL AS headline
                            FROM enhanced_entries e
                            WHERE {where_sql}
                            ORDER BY rank DESC
                            LIMIT %s
                        """  # noqa: S608
                        all_params = [search_text] + params + [max_results]

                    await cur.execute(query, all_params)
                    rows = await cur.fetchall()

                    results: list[tuple[EnhancedLogbookEntry, float, list[str]]] = []
                    for row in rows:
                        # Row is now a dict, extract rank and headline, pass rest to factory
                        rank = float(row.pop("rank", 0.0) or 0.0)
                        headline = row.pop("headline", "") or ""

                        entry = enhanced_entry_from_row(row)
                        highlights = [headline] if headline else []
                        results.append((entry, rank, highlights))

                    return results

        except Exception as e:
            raise DatabaseQueryError(
                f"Keyword search failed: {e}",
                query=f"KEYWORD SEARCH: {search_text}",
            ) from e

    @requires_module("search", "keyword")
    async def fuzzy_search(
        self,
        search_text: str,
        threshold: float = 0.3,
        max_results: int = 10,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[tuple[EnhancedLogbookEntry, float, list[str]]]:
        """Execute fuzzy search using pg_trgm similarity.

        Args:
            search_text: Text to search for
            threshold: Minimum similarity threshold (0-1)
            max_results: Maximum results to return
            start_date: Filter entries after this time
            end_date: Filter entries before this time

        Returns:
            List of (entry, score, highlights) tuples
        """
        from psycopg.rows import dict_row

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    where_clauses = ["similarity(raw_text, %s) >= %s"]
                    params: list[Any] = [search_text, threshold]

                    if start_date:
                        where_clauses.append("timestamp >= %s")
                        params.append(start_date)
                    if end_date:
                        where_clauses.append("timestamp <= %s")
                        params.append(end_date)

                    where_sql = " AND ".join(where_clauses)

                    query = f"""
                        SELECT e.*, similarity(raw_text, %s) AS sim
                        FROM enhanced_entries e
                        WHERE {where_sql}
                        ORDER BY sim DESC
                        LIMIT %s
                    """  # noqa: S608
                    all_params = [search_text] + params + [max_results]

                    await cur.execute(query, all_params)
                    rows = await cur.fetchall()

                    results: list[tuple[EnhancedLogbookEntry, float, list[str]]] = []
                    for row in rows:
                        sim = float(row.pop("sim", 0.0) or 0.0)
                        entry = enhanced_entry_from_row(row)
                        results.append((entry, sim, []))

                    return results

        except Exception as e:
            raise DatabaseQueryError(
                f"Fuzzy search failed: {e}",
                query=f"FUZZY SEARCH: {search_text}",
            ) from e

    @requires_module("search", "semantic")
    async def semantic_search(
        self,
        query_embedding: list[float],
        model_name: str,
        max_results: int = 10,
        similarity_threshold: float = 0.7,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        author: str | None = None,
        source_system: str | None = None,
    ) -> list[tuple[EnhancedLogbookEntry, float]]:
        """Execute semantic similarity search using pgvector.

        Args:
            query_embedding: Query embedding vector
            model_name: Model name for table lookup
            max_results: Maximum results to return
            similarity_threshold: Minimum similarity threshold
            start_date: Filter entries after this time
            end_date: Filter entries before this time
            author: Filter by author name (ILIKE match)
            source_system: Filter by source system (exact match)

        Returns:
            List of (entry, similarity) tuples
        """
        from psycopg.rows import dict_row

        from osprey.services.ariel_search.database.migration import model_to_table_name

        table_name = model_to_table_name(model_name)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    where_clauses = ["1 - (emb.embedding <=> %s::vector) >= %s"]
                    params: list[Any] = [embedding_str, similarity_threshold]

                    if start_date:
                        where_clauses.append("e.timestamp >= %s")
                        params.append(start_date)
                    if end_date:
                        where_clauses.append("e.timestamp <= %s")
                        params.append(end_date)
                    if author:
                        where_clauses.append("e.author ILIKE %s")
                        params.append(f"%{author}%")
                    if source_system:
                        where_clauses.append("e.source_system = %s")
                        params.append(source_system)

                    where_sql = " AND ".join(where_clauses)

                    query = f"""
                        SELECT e.*, 1 - (emb.embedding <=> %s::vector) AS similarity
                        FROM enhanced_entries e
                        JOIN {table_name} emb ON e.entry_id = emb.entry_id
                        WHERE {where_sql}
                        ORDER BY similarity DESC
                        LIMIT %s
                    """  # noqa: S608

                    all_params = [embedding_str] + params + [max_results]

                    await cur.execute(query, all_params)
                    rows = await cur.fetchall()

                    results: list[tuple[EnhancedLogbookEntry, float]] = []
                    for row in rows:
                        similarity = float(row.pop("similarity", 0.0) or 0.0)
                        entry = enhanced_entry_from_row(row)
                        results.append((entry, similarity))

                    return results

        except Exception as e:
            raise DatabaseQueryError(
                f"Semantic search failed: {e}",
                query=f"SEMANTIC SEARCH model={model_name}",
            ) from e

    # === Ingestion Run Tracking ===

    async def start_ingestion_run(self, source_system: str) -> int:
        """Record the start of an ingestion run.

        Args:
            source_system: Source system identifier

        Returns:
            The ingestion run ID
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    INSERT INTO ingestion_runs (started_at, source_system, status)
                    VALUES (NOW(), %s, 'running')
                    RETURNING id
                    """,
                    [source_system],
                )
                row = await result.fetchone()
                if not row:
                    raise DatabaseQueryError(
                        "Failed to start ingestion run: no ID returned",
                        query="INSERT ingestion_runs",
                    )
                return int(row[0])
        except DatabaseQueryError:
            raise
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to start ingestion run: {e}",
                query="INSERT ingestion_runs",
            ) from e

    async def complete_ingestion_run(
        self,
        run_id: int,
        entries_added: int,
        entries_updated: int,
        entries_failed: int,
    ) -> None:
        """Mark an ingestion run as successfully completed.

        Args:
            run_id: The ingestion run ID
            entries_added: Number of new entries added
            entries_updated: Number of existing entries updated
            entries_failed: Number of entries that failed
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    UPDATE ingestion_runs
                    SET completed_at = NOW(),
                        status = 'success',
                        entries_added = %s,
                        entries_updated = %s,
                        entries_failed = %s
                    WHERE id = %s
                    """,
                    [entries_added, entries_updated, entries_failed, run_id],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to complete ingestion run {run_id}: {e}",
                query=f"UPDATE ingestion_runs id={run_id}",
            ) from e

    async def fail_ingestion_run(self, run_id: int, error_message: str) -> None:
        """Mark an ingestion run as failed.

        Args:
            run_id: The ingestion run ID
            error_message: Error description
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    UPDATE ingestion_runs
                    SET completed_at = NOW(),
                        status = 'failed',
                        error_message = %s
                    WHERE id = %s
                    """,
                    [error_message[:500], run_id],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to mark ingestion run {run_id} as failed: {e}",
                query=f"UPDATE ingestion_runs id={run_id}",
            ) from e

    async def get_last_successful_run(self, source_system: str) -> datetime | None:
        """Get the completion time of the last successful ingestion run.

        Args:
            source_system: Source system identifier

        Returns:
            Completion timestamp of last successful run, or None if no runs found
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT MAX(completed_at) FROM ingestion_runs
                    WHERE source_system = %s AND status = 'success'
                    """,
                    [source_system],
                )
                row = await result.fetchone()
                if row and row[0]:
                    return row[0]
                return None
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get last successful run: {e}",
                query=f"SELECT MAX(completed_at) source_system={source_system}",
            ) from e

    # === Health Check ===

    async def health_check(self) -> tuple[bool, str]:
        """Check database connectivity and basic health.

        Returns:
            Tuple of (healthy, message)
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute("SELECT 1")
            return (True, "Database connected")
        except Exception as e:
            return (False, f"Database unreachable: {e}")
