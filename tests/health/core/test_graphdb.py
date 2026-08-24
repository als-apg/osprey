"""Tests for the core ``graphdb`` health category.

Drives the category against a fake neo4j driver injected through
``neo4j.GraphDatabase.driver``, exercising the presence gate (a
``services.graphdb`` config block), the two emitted rows (bolt connectivity and
the ``(:Resource)`` count), and every degradation path — a missing driver
package, an unreachable store, and a rejected credential — none of which may
produce an ``error`` row or raise.
"""

from __future__ import annotations

import sys

import pytest

from osprey.health.core.graphdb import graphdb
from osprey.health.models import CheckResult, Status

# --------------------------------------------------------------------------- #
# Fake driver
# --------------------------------------------------------------------------- #


class _FakeRecord(dict):
    """A driver record: mapping access by result key."""


class _FakeEagerResult:
    """The ``EagerResult`` shape ``Driver.execute_query`` returns."""

    def __init__(self, records: list[_FakeRecord]) -> None:
        self.records = records


class _FakeDriver:
    """Records the queries it was asked to run and answers them from a script."""

    def __init__(
        self,
        *,
        count: int = 7,
        connect_error: Exception | None = None,
        count_error: Exception | None = None,
    ) -> None:
        self.count = count
        self.connect_error = connect_error
        self.count_error = count_error
        self.queries: list[str] = []
        self.closed = False

    def execute_query(self, query: str, *args, **kwargs) -> _FakeEagerResult:
        self.queries.append(query)
        if len(self.queries) == 1:
            if self.connect_error is not None:
                raise self.connect_error
            return _FakeEagerResult([_FakeRecord(ok=1)])
        if self.count_error is not None:
            raise self.count_error
        return _FakeEagerResult([_FakeRecord(count=self.count)])

    def close(self) -> None:
        self.closed = True


def _install_driver(monkeypatch: pytest.MonkeyPatch, driver: _FakeDriver) -> list[tuple]:
    """Point ``GraphDatabase.driver`` at ``driver``; return captured call args."""
    import neo4j

    calls: list[tuple] = []

    def _factory(uri, *, auth=None, **config):
        calls.append((uri, auth))
        return driver

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", _factory)
    return calls


def _cfg(**graphdb_block) -> dict:
    """A config carrying a non-empty ``services.graphdb`` block (the gate)."""
    block = {"path": "./services/graphdb"}
    block.update(graphdb_block)
    return {"services": {"graphdb": block}}


async def _run(config) -> dict[str, CheckResult]:
    results = await graphdb(config)()
    assert isinstance(results, list)
    assert all(isinstance(r, CheckResult) for r in results)
    return {r.name: r for r in results}


@pytest.fixture(autouse=True)
def _no_ambient_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the resolver off whatever ``GRAPHDB_PASSWORD`` the host carries."""
    monkeypatch.delenv("GRAPHDB_PASSWORD", raising=False)


# --------------------------------------------------------------------------- #
# Presence gate
# --------------------------------------------------------------------------- #


async def test_no_rows_when_no_services_block() -> None:
    assert await _run({"deployment": {"bind_address": "127.0.0.1"}}) == {}


async def test_no_rows_when_no_graphdb_block() -> None:
    assert await _run({"services": {"qmd": {"port_host": 4444}}}) == {}


async def test_no_rows_when_graphdb_block_empty() -> None:
    assert await _run({"services": {"graphdb": {}}}) == {}


async def test_no_rows_when_graphdb_block_is_none() -> None:
    assert await _run({"services": {"graphdb": None}}) == {}


async def test_no_rows_when_config_none() -> None:
    assert await _run(None) == {}


async def test_gate_never_opens_a_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _FakeDriver()
    calls = _install_driver(monkeypatch, driver)
    assert await _run(None) == {}
    assert calls == []


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


async def test_configured_emits_both_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_driver(monkeypatch, _FakeDriver())
    by_name = await _run(_cfg())
    assert set(by_name) == {"graphdb_connection", "graphdb_resources"}
    assert all(r.category == "graphdb" for r in by_name.values())


async def test_connection_row_ok_with_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_driver(monkeypatch, _FakeDriver())
    row = (await _run(_cfg()))["graphdb_connection"]
    assert row.status is Status.OK
    assert "bolt://localhost:7687" in row.message
    assert row.latency_ms > 0.0


async def test_connection_uses_resolved_uri_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_driver(monkeypatch, _FakeDriver())
    monkeypatch.setenv("GRAPHDB_PASSWORD", "s3cret")
    await _run(_cfg(port_host=17687))
    assert calls == [("bolt://localhost:17687", ("neo4j", "s3cret"))]


async def test_external_uri_is_dialed_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_driver(monkeypatch, _FakeDriver())
    await _run(_cfg(uri="bolt://graph.example.org:7687", username="reader"))
    assert calls[0][0] == "bolt://graph.example.org:7687"
    assert calls[0][1][0] == "reader"


async def test_resource_row_counts_resource_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _FakeDriver(count=8431)
    _install_driver(monkeypatch, driver)
    row = (await _run(_cfg()))["graphdb_resources"]
    assert row.status is Status.OK
    assert row.value == "8,431 Resource nodes"
    # The row must say WHICH nodes it counted: n10s bookkeeping nodes are not data.
    assert "Resource" in row.message
    assert ":Resource" in driver.queries[1]


async def test_driver_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _FakeDriver()
    _install_driver(monkeypatch, driver)
    await _run(_cfg())
    assert driver.closed is True


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


async def test_missing_driver_warns_and_names_the_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "neo4j", None)
    by_name = await _run(_cfg())
    assert set(by_name) == {"graphdb_connection"}
    row = by_name["graphdb_connection"]
    assert row.status is Status.WARNING
    assert "neo4j" in row.message


async def test_unreachable_store_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    from neo4j.exceptions import ServiceUnavailable

    _install_driver(
        monkeypatch, _FakeDriver(connect_error=ServiceUnavailable("Connection refused"))
    )
    by_name = await _run(_cfg())
    assert set(by_name) == {"graphdb_connection"}
    row = by_name["graphdb_connection"]
    assert row.status is Status.WARNING
    assert "unreachable" in row.message
    assert row.details


async def test_driver_construction_failure_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    import neo4j

    def _boom(uri, *, auth=None, **config):
        raise ValueError(f"Unsupported URI scheme: {uri}")

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", _boom)
    by_name = await _run(_cfg(uri="wat://nowhere"))
    assert set(by_name) == {"graphdb_connection"}
    assert by_name["graphdb_connection"].status is Status.WARNING


async def test_auth_failure_names_the_password_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neo4j.exceptions import AuthError

    _install_driver(monkeypatch, _FakeDriver(connect_error=AuthError("unauthorized")))
    by_name = await _run(_cfg())
    assert set(by_name) == {"graphdb_connection"}
    row = by_name["graphdb_connection"]
    assert row.status is Status.WARNING
    assert "GRAPHDB_PASSWORD" in f"{row.message} {row.details}"


async def test_empty_graph_warns_and_names_the_seed_verb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_driver(monkeypatch, _FakeDriver(count=0))
    by_name = await _run(_cfg())
    assert by_name["graphdb_connection"].status is Status.OK
    row = by_name["graphdb_resources"]
    assert row.status is Status.WARNING
    assert "osprey knowledge seed-graph" in f"{row.message} {row.details}"


async def test_count_failure_warns_but_keeps_the_connection_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neo4j.exceptions import Neo4jError

    _install_driver(monkeypatch, _FakeDriver(count_error=Neo4jError("procedure not found")))
    by_name = await _run(_cfg())
    assert by_name["graphdb_connection"].status is Status.OK
    assert by_name["graphdb_resources"].status is Status.WARNING


async def test_malformed_block_warns_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_driver(monkeypatch, _FakeDriver())
    by_name = await _run(_cfg(port_host="seven-thousand"))
    assert set(by_name) == {"graphdb_connection"}
    row = by_name["graphdb_connection"]
    assert row.status is Status.WARNING
    assert "services.graphdb.port_host" in f"{row.message} {row.details}"


async def test_no_row_is_ever_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from neo4j.exceptions import ServiceUnavailable

    for driver in (
        _FakeDriver(),
        _FakeDriver(count=0),
        _FakeDriver(connect_error=ServiceUnavailable("down")),
        _FakeDriver(count_error=RuntimeError("boom")),
    ):
        _install_driver(monkeypatch, driver)
        by_name = await _run(_cfg())
        assert all(r.status is not Status.ERROR for r in by_name.values())


# --------------------------------------------------------------------------- #
# Registry surface
# --------------------------------------------------------------------------- #


def test_registered_as_a_core_category() -> None:
    from osprey.health.core import CORE_CATEGORY_NAMES, get_core_category_factory

    assert "graphdb" in CORE_CATEGORY_NAMES
    assert get_core_category_factory("graphdb") is graphdb


def test_registered_as_config_dependent() -> None:
    from osprey.health.records import CONFIG_DEPENDENT

    assert "graphdb" in CONFIG_DEPENDENT
