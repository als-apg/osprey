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
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest

logger = logging.getLogger(__name__)

# Dev database servers - try port 5432 (ariel-postgres), then 5433 (ariel-dev-db).
# Each URL names the server's maintenance database; a test session never writes
# there, it creates a database of its own beside it (``own_database``). The host
# is the loopback address, not ``localhost``: psycopg resolves a name with a
# blocking ``getaddrinfo`` that no connect timeout bounds, and a loaded host's
# resolver can hold it past the test's own timeout.
DEV_SERVER_URL_5432 = "postgresql://ariel:ariel@127.0.0.1:5432/ariel"
DEV_SERVER_URL_5433 = "postgresql://ariel:ariel@127.0.0.1:5433/ariel"

#: Seconds each connect to a session database server may take (libpq's
#: ``connect_timeout``). Well under the test timeout, so a server that does not
#: answer is reported as itself; without it psycopg waits 130 s per address.
CONNECT_TIMEOUT_S = 15


def _connect(url: str) -> Any:
    import psycopg

    return psycopg.connect(url, autocommit=True, connect_timeout=CONNECT_TIMEOUT_S)


def _not_answering(url: str, what: str, exc: BaseException) -> str:
    """One line: which server, what was tried, under which bound, and its answer."""
    parts = urlsplit(url)
    lines = str(exc).strip().splitlines()
    detail = lines[0] if lines else type(exc).__name__
    return (
        f"database server {parts.hostname}:{parts.port or 5432} did not answer "
        f"({what}, connect_timeout {CONNECT_TIMEOUT_S} s): {detail}"
    )


def dev_server_url() -> str | None:
    """The first running dev database server (5432, then 5433), or ``None``."""
    try:
        import psycopg  # noqa: F401
    except ImportError:
        logger.debug("psycopg not available (libpq missing?), skipping dev database check")
        return None

    for url in (DEV_SERVER_URL_5432, DEV_SERVER_URL_5433):
        try:
            with _connect(url) as conn:
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
    try:
        with _connect(server_url) as conn:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    except psycopg.OperationalError as exc:
        pytest.fail(_not_answering(server_url, "CREATE DATABASE", exc), pytrace=False)

    def _drop() -> None:
        with _connect(server_url) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )

    request.addfinalizer(_drop)
    url = urlunsplit(urlsplit(server_url)._replace(path=f"/{name}"))
    with _connect(url) as conn:
        # no pgvector on this server; the vector migrations skip
        with contextlib.suppress(psycopg.Error):
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    logger.info(f"Own database {name} on {server_url.split('@')[-1].rsplit('/', 1)[0]}")
    return url
