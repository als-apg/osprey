"""PostgreSQL for a test session: databases it owns, and the server it starts.

A server the session found (one ``ARIEL_TEST_DATABASE_URL`` names, or a running
dev database) is shared with whatever else reaches it -- another session,
another worktree, a stopped run's leftovers -- so the session writes only to a
database it created, named for it alone, and drops it at session end.

A session that finds no server, or must not use one it finds, starts one:
:func:`pgvector_container_url`, the image the framework deploys, once per
session whichever suites ask for it. ``tests/conftest.py`` exports the fixture
so suites in different directories share the one container; it starts only
when a test requests it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest

from tests._container_support import (
    CONTAINER_START_ATTEMPTS,
    CONTAINER_START_BACKOFF_S,
    CONTAINER_STATE_TIMEOUT,
    DOCKER_CLI_PROBE_TIMEOUT,
    READINESS_CEILING,
    is_docker_available,
    require_container_runtime,
    start_or_fail,
    stop_quietly,
)

logger = logging.getLogger(__name__)

#: The Postgres image the framework deploys (``upstream_images.py``); the
#: session's own server runs it so the suites test what ships.
SHIPPED_POSTGRES_IMAGE = "pgvector/pgvector:pg16"

#: Postgres' own port inside the container, resolved during the start retry.
POSTGRES_CONTAINER_PORT = 5432

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


def answers(url: str) -> None:
    """Ask the server at *url* for ``SELECT 1``, raising while it does not answer.

    The probe a readiness wait polls once the server's container has started.
    """
    with _connect(url) as conn:
        conn.execute("SELECT 1")


def dev_server_url() -> str | None:
    """The first running dev database server (5432, then 5433), or ``None``."""
    try:
        import psycopg  # noqa: F401
    except ImportError:
        logger.debug("psycopg not available (libpq missing?), skipping dev database check")
        return None

    for url in (DEV_SERVER_URL_5432, DEV_SERVER_URL_5433):
        try:
            answers(url)
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


def pgvector_container_start_bound() -> float:
    """Seconds :func:`pgvector_container_url` may take to start before it fails.

    :func:`~tests._container_support.start_or_fail` retries the start, and each
    attempt runs testcontainers' own readiness wait, bounded by its configured
    timeout, with a pause between attempts. Then
    :func:`~tests._container_support.require_container_runtime` asks the runtime
    (``docker info``), waits for the server to answer from the host -- up to its
    ceiling plus one round of probe, pause and liveness read -- and asks the
    runtime again when it never did.
    """
    from testcontainers.core.config import testcontainers_config

    start = (
        CONTAINER_START_ATTEMPTS * testcontainers_config.timeout
        + (CONTAINER_START_ATTEMPTS - 1) * CONTAINER_START_BACKOFF_S
    )
    readiness_round = CONNECT_TIMEOUT_S + 0.5 + 2 * CONTAINER_STATE_TIMEOUT
    return start + 2 * DOCKER_CLI_PROBE_TIMEOUT + READINESS_CEILING + readiness_round


def database_required() -> bool:
    """Whether ``ARIEL_TEST_REQUIRE_DB=1`` asks for a missing database to fail.

    A gate that runs an integration file sets it, so the gate cannot pass with
    every database test skipped.
    """
    return os.environ.get("ARIEL_TEST_REQUIRE_DB") == "1"


def skip_or_fail(reason: str) -> None:
    """Skip for want of a database -- or fail, when one is required."""
    if database_required():
        pytest.fail(f"{reason} (ARIEL_TEST_REQUIRE_DB=1)")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def pgvector_container_url(request: pytest.FixtureRequest) -> str:
    """The maintenance URL of a Postgres container this session started and owns.

    Runs :data:`SHIPPED_POSTGRES_IMAGE`. Session-scoped and not autouse: the
    container starts at the first test that asks for it, and every later
    request in the session -- from any suite -- gets the same server. Under
    xdist each worker is a session of its own; the container-starting files sit
    in one ``xdist_group("docker")``, so a run starts one.

    No suite writes to the database this URL names: each creates databases of
    its own beside it, so a destructive test in one suite cannot reach another's
    rows.

    Returns:
        A ``postgresql://`` URL (psycopg 3 form) to the container's ``ariel``
        database.

    Skips:
        When Docker is not reachable or testcontainers is not installed -- or
        fails instead under ``ARIEL_TEST_REQUIRE_DB=1``. Past that, a container
        that will not start fails (:func:`~tests._container_support.start_or_fail`).
    """
    if not is_docker_available():
        skip_or_fail("docker daemon is not reachable")
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        skip_or_fail("testcontainers[postgres] not installed")

    # The port goes through start_or_fail rather than being read afterwards:
    # `get_connection_url()` resolves the published port too, so it races the
    # same way. Resolving it inside the retry means a fresh container is built
    # when the mapping never appears.
    container, _port = start_or_fail(
        lambda: PostgresContainer(
            image=SHIPPED_POSTGRES_IMAGE,
            username="ariel",
            password="ariel",
            dbname="ariel",
        ),
        "postgres",
        POSTGRES_CONTAINER_PORT,
    )
    request.addfinalizer(lambda: stop_quietly(container))

    url = container.get_connection_url()
    if url.startswith("postgresql+psycopg2://"):
        url = url.replace("postgresql+psycopg2://", "postgresql://")

    require_container_runtime(lambda: answers(url), "postgres", container=container)
    logger.info(f"Session Postgres container started: {url.split('@')[-1]}")
    return url
