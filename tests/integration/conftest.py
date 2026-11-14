"""Fixtures for the qmd/ARIEL integration lane.

Nothing here is autouse, so the other modules under ``tests/integration/`` are
unaffected by this file's existence.

This lane works on the session's own Postgres container
(``pgvector_container_url``, from ``tests/conftest.py``), the one the ARIEL
search suite starts too, and hands each module its own database inside it. One
behaviour proven here is ``osprey ariel purge``, which issues ``TRUNCATE
enhanced_entries CASCADE`` and drops every embedding table, so no two modules
-- and no other suite on the server -- may share a database.

The lane never takes ``ARIEL_TEST_DATABASE_URL`` or a running dev database, as
the ARIEL search suite's ``database_url`` does: only a server the session
started is safe for the purge.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import pytest


@pytest.fixture(scope="session")
def qmd_ariel_database_factory(pgvector_container_url: str) -> Callable[[str], str]:
    """Return a factory creating a named database and handing back its DSN.

    One database per test module, so the destructive purge coverage cannot reach
    another module's seeded rows whatever order the modules run in.
    """
    import psycopg

    prefix = pgvector_container_url.rsplit("/", 1)[0]

    def make(name: str) -> str:
        with psycopg.connect(pgvector_container_url, autocommit=True) as conn:
            with contextlib.suppress(psycopg.errors.DuplicateDatabase):
                conn.execute(f'CREATE DATABASE "{name}"')
        return f"{prefix}/{name}"

    return make
