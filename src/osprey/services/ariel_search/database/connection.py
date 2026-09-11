"""ARIEL database connection management.

This module provides async connection pool management for the ARIEL database.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from osprey.services.ariel_search.config import DatabaseConfig


async def create_connection_pool(
    config: "DatabaseConfig",
    *,
    uri: str | None = None,
    max_size: int = 10,
) -> "AsyncConnectionPool":
    """Create async connection pool for ARIEL repository.

    Args:
        config: Database configuration with connection URI
        uri: Connect with this DSN instead of ``config.uri``. The one caller
            that passes it is the read-only pool the agent's raw-SQL path uses,
            which reaches the SAME store as a different role, so everything
            else about the pool is unchanged.
        max_size: Upper bound on pooled connections. The default sizes the
            ingestion and search pool; a pool serving one tool wants far fewer.

    Returns:
        Configured AsyncConnectionPool ready for use

    Raises:
        ImportError: If psycopg_pool is not installed
    """
    try:
        from psycopg_pool import AsyncConnectionPool
    except ImportError as e:
        raise ImportError(
            "psycopg[pool] is required for ARIEL database support. "
            "Install with: pip install 'psycopg[pool]'"
        ) from e

    pool = AsyncConnectionPool(
        conninfo=uri if uri is not None else config.uri,
        min_size=1,
        max_size=max_size,
        kwargs={"autocommit": True},
        open=False,  # Don't open immediately
        reconnect_timeout=0,  # Don't retry on failure
    )
    await pool.open(wait=True, timeout=5.0)
    return pool
