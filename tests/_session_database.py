"""A PostgreSQL database a test session owns, on a server it did not start.

A server the session found (one ``ARIEL_TEST_DATABASE_URL`` names, or a running
dev database) is shared with whatever else reaches it -- another session,
another worktree, a stopped run's leftovers -- so the session writes only to a
database it created, named for it alone, and drops it at session end.
"""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

logger = logging.getLogger(__name__)

# Dev database servers - try port 5432 (ariel-postgres), then 5433 (ariel-dev-db).
# Each URL names the server's maintenance database; a test session never writes
# there, it creates a database of its own beside it (``own_database``).
DEV_SERVER_URL_5432 = "postgresql://ariel:ariel@localhost:5432/ariel"
DEV_SERVER_URL_5433 = "postgresql://ariel:ariel@localhost:5433/ariel"


def dev_server_url() -> str | None:
    """The first running dev database server (5432, then 5433), or ``None``."""
    try:
        import psycopg
    except ImportError:
        logger.debug("psycopg not available (libpq missing?), skipping dev database check")
        return None

    for url in (DEV_SERVER_URL_5432, DEV_SERVER_URL_5433):
        try:
            with psycopg.connect(url, autocommit=True) as conn:
                conn.execute("SELECT 1")
            return url
        except Exception as e:
            logger.debug(f"Dev database server at {url.split('@')[-1]} not available: {e}")
    return None


def own_database(request: pytest.FixtureRequest, server_url: str, prefix: str) -> str:
    """Create a database for the fixture's scope on *server_url*'s server.

    The name is *prefix*, the xdist worker and a random suffix; the drop is
    registered with ``request.addfinalizer``. ``vector`` is created when the
    server has pgvector; the migrations skip the vector tables otherwise.

    Returns:
        The new database's URL (*server_url* with its database replaced).
    """
    import psycopg
    from psycopg import sql

    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    name = f"{prefix}_{worker}_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(server_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))

    def _drop() -> None:
        with psycopg.connect(server_url, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )

    request.addfinalizer(_drop)
    url = urlunsplit(urlsplit(server_url)._replace(path=f"/{name}"))
    with psycopg.connect(url, autocommit=True) as conn:
        # no pgvector on this server; the vector migrations skip
        with contextlib.suppress(psycopg.Error):
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    logger.info(f"Own database {name} on {server_url.split('@')[-1].rsplit('/', 1)[0]}")
    return url
