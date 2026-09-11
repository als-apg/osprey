"""The service opens a second, SELECT-only pool for the agent's raw-SQL path.

``sql_query`` is auto-allowed and, before the read-only role existed, ran on the
pool ingestion writes with — a superuser on every stack that deployed the
bundled Postgres. ``BEGIN READ ONLY`` and the statement allowlist stay, but
neither covers a server-side read such as ``pg_read_file()``; the privileges the
connection holds do.

The fallback is the part that has to keep working: a data volume older than the
role has no ``_ro`` login, and a project pointing at a database osprey did not
provision has no read-only identity at all. Both keep the tool on the ingestion
pool — where it already was — and say so once, at start-up.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from osprey.services.ariel_search.config import ARIELConfig

ARIEL_LOGGER = "ariel"


def _config(**database: Any) -> ARIELConfig:
    """An ARIEL config whose database block is whatever the test needs."""
    return ARIELConfig.from_dict(
        {"database": database, "ingestion": {"adapter": "generic_json"}},
        {"username": "ariel", "database_name": "ariel", "port_host": 5432},
    )


@pytest.fixture
def pool_factory(monkeypatch):
    """Record every ``create_connection_pool`` call and hand back a stand-in.

    Returns the list of ``(uri, max_size)`` pairs the service asked for, in
    order, so a test can say which DSNs were dialed without a database.
    """
    calls: list[tuple[str, int]] = []

    async def _fake_create_pool(config, *, uri=None, max_size=10):
        calls.append((uri if uri is not None else config.uri, max_size))
        return object()

    import osprey.services.ariel_search.database.connection as conn_mod

    monkeypatch.setattr(conn_mod, "create_connection_pool", _fake_create_pool)
    monkeypatch.setattr(
        "osprey.services.ariel_search.database.repository.ARIELRepository",
        lambda pool, config: object(),
    )
    return calls


@pytest.fixture(autouse=True)
def pinned_passwords(monkeypatch):
    """Neither password may come from the developer's own shell."""
    monkeypatch.setenv("ARIEL_DB_PASSWORD", "owner-secret")
    monkeypatch.setenv("ARIEL_DB_READONLY_PASSWORD", "ro-secret")
    for name in ("ARIEL_DATABASE_HOST", "ARIEL_DATABASE_PORT"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_derived_dsn_opens_both_pools(pool_factory):
    """Two logins on one store, and the read-only one is the smaller pool: it
    serves a single tool, not the search and ingestion traffic."""
    from osprey.services.ariel_search.service import create_ariel_service

    service = await create_ariel_service(_config())

    assert pool_factory == [
        ("postgresql://ariel:owner-secret@localhost:5432/ariel", 10),
        ("postgresql://ariel_ro:ro-secret@localhost:5432/ariel", 3),
    ]
    assert service.readonly_pool is not None
    assert service.readonly_pool is not service.pool


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_explicit_dsn_leaves_one_pool_and_one_warning(pool_factory, caplog):
    """A database osprey did not provision has no ``_ro`` role to open."""
    from osprey.services.ariel_search.service import create_ariel_service

    explicit = "postgresql://someone:else@logbook-db.example.org:5432/ariel"
    with caplog.at_level(logging.WARNING, logger=ARIEL_LOGGER):
        service = await create_ariel_service(_config(uri=explicit))

    assert pool_factory == [(explicit, 10)]
    assert service.readonly_pool is None
    assert "SQL tool is running on the ingestion role" in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_refused_readonly_login_falls_back_and_says_so(monkeypatch, caplog):
    """The shape an existing data volume has: the DSN resolves, the login does
    not exist. The tool already ran on the ingestion connection before the role
    existed, so this is a warning rather than a refusal to start.
    """
    calls: list[str] = []

    async def _fake_create_pool(config, *, uri=None, max_size=10):
        dialed = uri if uri is not None else config.uri
        calls.append(dialed)
        if uri is not None:
            raise OSError('password authentication failed for user "ariel_ro"')
        return object()

    import osprey.services.ariel_search.database.connection as conn_mod

    monkeypatch.setattr(conn_mod, "create_connection_pool", _fake_create_pool)
    monkeypatch.setattr(
        "osprey.services.ariel_search.database.repository.ARIELRepository",
        lambda pool, config: object(),
    )

    from osprey.services.ariel_search.service import create_ariel_service

    with caplog.at_level(logging.WARNING, logger=ARIEL_LOGGER):
        service = await create_ariel_service(_config())

    assert len(calls) == 2
    assert service.readonly_pool is None
    assert "SQL tool is running on the ingestion role" in caplog.text
    assert "ariel_ro" in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_readonly_pool_is_closed_with_the_service(pool_factory):
    """Both pools are the service's to close; leaking the smaller one would
    hold idle connections on a store the process is done with."""

    class _Pool:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    ingest, readonly = _Pool(), _Pool()

    from osprey.services.ariel_search.service import ARIELSearchService

    service = ARIELSearchService(
        config=_config(),
        pool=ingest,  # type: ignore[arg-type]
        repository=object(),  # type: ignore[arg-type]
        readonly_pool=readonly,  # type: ignore[arg-type]
    )
    async with service:
        pass

    assert ingest.closed
    assert readonly.closed


# ---------------------------------------------------------------------------
# The two consumers that close the service by hand
#
# `async with service` closes both pools, but neither long-lived consumer uses
# it: the MCP server builds the service directly so it can outlive one request,
# and the capability path caches a module singleton. Whatever the context
# manager would have done is theirs to do instead.
# ---------------------------------------------------------------------------


class _RecordingPool:
    """Stands in for an ``AsyncConnectionPool`` that only has to be closed."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _service_with_two_pools() -> tuple[Any, _RecordingPool, _RecordingPool]:
    from osprey.services.ariel_search.service import ARIELSearchService

    ingest, readonly = _RecordingPool(), _RecordingPool()
    service = ARIELSearchService(
        config=_config(),
        pool=ingest,  # type: ignore[arg-type]
        repository=object(),  # type: ignore[arg-type]
        readonly_pool=readonly,  # type: ignore[arg-type]
    )
    return service, ingest, readonly


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_mcp_context_shutdown_closes_both_pools():
    """The server that hosts ``sql_query`` is the readonly pool's one consumer."""
    from osprey.mcp_server.ariel.server_context import ARIELContext

    service, ingest, readonly = _service_with_two_pools()
    context = ARIELContext.__new__(ARIELContext)
    context._service = service

    await context.shutdown()

    assert ingest.closed
    assert readonly.closed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_closing_the_capability_singleton_closes_both_pools(monkeypatch):
    """Same hand-rolled teardown, one module along."""
    from osprey.services.ariel_search import capability

    service, ingest, readonly = _service_with_two_pools()
    monkeypatch.setattr(capability, "_ariel_service_instance", service)

    await capability.close_ariel_service()

    assert ingest.closed
    assert readonly.closed
    assert capability._ariel_service_instance is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_single_pool_service_still_closes_cleanly(monkeypatch):
    """A stack on the fallback has no readonly pool; neither teardown may trip."""
    from osprey.mcp_server.ariel.server_context import ARIELContext
    from osprey.services.ariel_search import capability
    from osprey.services.ariel_search.service import ARIELSearchService

    for close in ("context", "singleton"):
        pool = _RecordingPool()
        service = ARIELSearchService(
            config=_config(),
            pool=pool,  # type: ignore[arg-type]
            repository=object(),  # type: ignore[arg-type]
        )
        if close == "context":
            context = ARIELContext.__new__(ARIELContext)
            context._service = service
            await context.shutdown()
        else:
            monkeypatch.setattr(capability, "_ariel_service_instance", service)
            await capability.close_ariel_service()
        assert pool.closed
