"""Shared test doubles and seam patchers for the ``cli_operations`` test modules.

``test_cli_operations_pipeline.py`` (ingest-side operations) and
``test_cli_operations_maintenance.py`` (database-reshaping operations) drive the
same collaborators through the same seams, so the stand-ins and the
monkeypatching live here once.

``cli_operations`` imports every collaborator lazily, inside the function that
uses it, so a patch must name the module that *owns* the symbol rather than
``cli_operations`` itself: ``create_ariel_service`` on the ``ariel_search``
package, ``create_connection_pool`` on ``database.connection``, ``run_migrations``
on ``database.migrations``, ``get_adapter`` on ``ingestion`` and
``create_enhancers_from_config`` on ``enhancement``. Each helper below patches at
that source and returns the arguments it observed, so a test can assert on what
the operation passed down without reaching into the real module.

The module name is underscore-prefixed so pytest does not collect it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StubService:
    """Async-context-manager stand-in for ``ARIELSearchService``.

    ``pool`` defaults to the repository's :class:`_FakePool` -- the real service
    and its repository share one pool -- so anything the code under test runs
    through ``service.pool.connection()`` is recorded on ``repo.pool.calls``.
    ``enters``/``exits`` prove the ``async with service`` block was left even on
    the error paths.
    """

    def __init__(self, repository: Any, pool: Any = None) -> None:
        self.repository = repository
        self.pool = repository.pool if pool is None else pool
        self.enters = 0
        self.exits = 0

    async def __aenter__(self) -> _StubService:
        self.enters += 1
        return self

    async def __aexit__(self, *exc: object) -> bool:
        self.exits += 1
        return False


class _Adapter:
    """Ingestion adapter yielding scripted entries from an async generator.

    Args:
        entries: Entries to yield, in order.
        raise_at: Index at which ``fetch_entries`` raises instead of yielding,
            for the "stream dies mid-ingest" path.
    """

    def __init__(
        self,
        entries: Iterable[dict[str, Any]] = (),
        source_system_name: str = "TestSource",
        raise_at: int | None = None,
    ) -> None:
        self.entries = list(entries)
        self.source_system_name = source_system_name
        self.raise_at = raise_at
        self.fetch_calls: list[dict[str, Any]] = []

    async def fetch_entries(self, since: Any = None, limit: int | None = None):
        self.fetch_calls.append({"since": since, "limit": limit})
        for index, entry in enumerate(self.entries):
            if self.raise_at is not None and index == self.raise_at:
                raise RuntimeError("adapter stream failed")
            yield entry


class _Enhancer:
    """Enhancement module recording every ``enhance`` call.

    Failures drive the ``mark_enhancement_failed`` branch, and the two knobs are
    independent:

    Args:
        fails_on: Entry ids for which ``enhance`` raises a generated
            ``RuntimeError`` naming the entry.
        error: Exception raised for *every* entry, when the operation under test
            cares about the message that reaches ``mark_enhancement_failed``.

    ``seen`` records the entry ids in order; ``conns`` records the connection
    each call was handed, positionally aligned with ``seen``.
    """

    def __init__(
        self,
        name: str = "text_embedding",
        fails_on: Iterable[str] = (),
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.fails_on = set(fails_on)
        self.error = error
        self.seen: list[str] = []
        self.conns: list[Any] = []

    async def enhance(self, entry: dict[str, Any], conn: Any) -> None:
        entry_id = entry["entry_id"]
        self.seen.append(entry_id)
        self.conns.append(conn)
        if self.error is not None:
            raise self.error
        if entry_id in self.fails_on:
            raise RuntimeError(f"enhance failed on {entry_id}")


# ---------------------------------------------------------------------------
# Patch helpers -- each names the module that owns the symbol
# ---------------------------------------------------------------------------


def _patch_service(monkeypatch: pytest.MonkeyPatch, service: Any) -> list[Any]:
    """Route ``create_ariel_service`` to *service*; returns the configs it saw."""
    import osprey.services.ariel_search as ariel_pkg

    configs: list[Any] = []

    async def _fake_create(config):
        configs.append(config)
        return service

    monkeypatch.setattr(ariel_pkg, "create_ariel_service", _fake_create)
    return configs


def _forbid_service(
    monkeypatch: pytest.MonkeyPatch, reason: str = "service should not be created"
) -> None:
    """Make service creation an error, for paths that must not reach it."""
    import osprey.services.ariel_search as ariel_pkg

    async def _fake_create(config):
        raise AssertionError(reason)

    monkeypatch.setattr(ariel_pkg, "create_ariel_service", _fake_create)


def _patch_adapter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Route ``get_adapter`` to *adapter*, or raise it if it is an exception."""
    import osprey.services.ariel_search.ingestion as ing

    def _fake_get(config):
        if isinstance(adapter, Exception):
            raise adapter
        return adapter

    monkeypatch.setattr(ing, "get_adapter", _fake_get)


def _patch_enhancers(monkeypatch: pytest.MonkeyPatch, enhancers: Iterable[Any]) -> None:
    """Route ``create_enhancers_from_config`` to *enhancers*."""
    import osprey.services.ariel_search.enhancement as enh

    listed = list(enhancers)
    monkeypatch.setattr(enh, "create_enhancers_from_config", lambda config: list(listed))


def _patch_pool(monkeypatch: pytest.MonkeyPatch, pool: Any) -> list[Any]:
    """Route ``create_connection_pool`` to *pool*; returns the db configs it saw."""
    import osprey.services.ariel_search.database.connection as conn_mod

    seen: list[Any] = []

    async def _fake_create_pool(db_config):
        seen.append(db_config)
        return pool

    monkeypatch.setattr(conn_mod, "create_connection_pool", _fake_create_pool)
    return seen


def _patch_migrations(
    monkeypatch: pytest.MonkeyPatch,
    applied: list[str] | None = None,
    error: Exception | None = None,
) -> list[Any]:
    """Route ``run_migrations``; returns the ``(pool, config)`` pairs it saw."""
    import osprey.services.ariel_search.database.migrations as mig_mod

    seen: list[Any] = []

    async def _fake_run_migrations(pool, config):
        seen.append((pool, config))
        if error is not None:
            raise error
        return list(applied or [])

    monkeypatch.setattr(mig_mod, "run_migrations", _fake_run_migrations)
    return seen
