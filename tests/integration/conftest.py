"""Fixtures for the qmd/ARIEL integration lane.

Nothing here is autouse, so the other modules under ``tests/integration/`` are
unaffected by this file's existence.

This lane starts its own Postgres container and hands each module its own
database inside it. One behaviour proven here is ``osprey ariel purge``, which
issues ``TRUNCATE enhanced_entries CASCADE`` and drops every embedding table, so
no two modules may share a database; ``tests/services/ariel_search/conftest.py``
is not a conftest of this directory, so its ``database_url`` is not in reach.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable

import pytest

from tests._container_support import is_docker_available, start_or_fail, stop_quietly

#: Database created with the container; also the admin database the factory
#: connects to in order to create the per-module ones.
LANE_DATABASE = "qmd_ariel_lane"

#: Postgres' own port inside the container, resolved during the start retry.
POSTGRES_CONTAINER_PORT = 5432


def _skip_or_fail(reason: str) -> None:
    """Skip for want of Postgres -- or fail under ``ARIEL_TEST_REQUIRE_DB=1``.

    A gate running this lane sets the variable so it cannot pass on skips.
    """
    if os.environ.get("ARIEL_TEST_REQUIRE_DB") == "1":
        pytest.fail(f"{reason} (ARIEL_TEST_REQUIRE_DB=1)")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def qmd_ariel_postgres(request: pytest.FixtureRequest):
    """A Postgres container owned by this lane alone.

    Returns:
        The started ``PostgresContainer``.
    """
    if not is_docker_available():
        _skip_or_fail("docker daemon is not reachable")
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        _skip_or_fail("testcontainers[postgres] not installed")

    # The port goes through start_or_fail rather than being read afterwards:
    # `get_connection_url()` resolves the published port too, so it races the
    # same way the sidecar's does. Resolving it inside the retry means a fresh
    # container is built when the mapping never appears.
    container, _port = start_or_fail(
        lambda: PostgresContainer(
            image="ankane/pgvector:latest",
            username="ariel",
            password="ariel",
            dbname=LANE_DATABASE,
        ),
        "postgres (qmd/ariel lane)",
        POSTGRES_CONTAINER_PORT,
    )
    request.addfinalizer(lambda: stop_quietly(container))
    return container


@pytest.fixture(scope="session")
def qmd_ariel_database_factory(qmd_ariel_postgres) -> Callable[[str], str]:
    """Return a factory creating a named database and handing back its DSN.

    One database per test module, so the destructive purge coverage cannot reach
    another module's seeded rows whatever order the modules run in.
    """
    import psycopg

    base = qmd_ariel_postgres.get_connection_url()
    if base.startswith("postgresql+psycopg2://"):
        base = base.replace("postgresql+psycopg2://", "postgresql://")
    prefix = base.rsplit("/", 1)[0]

    def make(name: str) -> str:
        with psycopg.connect(f"{prefix}/{LANE_DATABASE}", autocommit=True) as conn:
            with contextlib.suppress(psycopg.errors.DuplicateDatabase):
                conn.execute(f'CREATE DATABASE "{name}"')
        return f"{prefix}/{name}"

    return make
