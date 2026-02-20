"""ARIEL database connection management.

This module provides async connection pool management for the ARIEL database.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from osprey.services.ariel_search.config import DatabaseConfig


async def create_connection_pool(config: "DatabaseConfig") -> "AsyncConnectionPool":
    """Create async connection pool for ARIEL repository.

    Args:
        config: Database configuration with connection URI

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
        conninfo=config.uri,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True},
        open=False,  # Don't open immediately
    )
    await pool.open()
    return pool
