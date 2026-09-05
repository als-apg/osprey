"""Tests for the core ``graphdb`` health category.

Drives the category against a fake neo4j driver injected through
``neo4j.GraphDatabase.driver``, exercising the presence gate (a
``services.graphdb`` config block), the three emitted rows (bolt connectivity,
the ``(:Resource)`` count and the ``(:_OspreySeed)`` provenance marker), and
every degradation path — a missing driver package, an unreachable store, a
rejected credential, an unreadable count or marker — none of which may produce
an ``error`` row or raise.
"""

from __future__ import annotations

import sys

import pytest

from osprey.health.core.graphdb import SEED_DIGEST_DETAIL_PREFIX, graphdb, seed_digest
from osprey.health.models import CheckResult, Status
from osprey.port_layout import default_port

# --------------------------------------------------------------------------- #
# Fake driver
# --------------------------------------------------------------------------- #


class _FakeRecord(dict):
    """A driver record: mapping access by result key."""


class _FakeEagerResult:
    """The ``EagerResult`` shape ``Driver.execute_query`` returns."""

    def __init__(self, records: list[_FakeRecord]) -> None:
        self.records = records


#: A plausible corpus digest. Only its prefix and its full text matter here.
DIGEST = "9f2c1ab34de5670089abcdef0123456789abcdef0123456789abcdef01234567"


class _FakeDriver:
    """Records the queries it was asked to run and answers them from a script.

    The ping is the first query; everything after it is answered on what the
    query text asks for, so the probe may read the count and the marker in
    either order without the fake having to be rewritten.
    """

    def __init__(
        self,
        *,
        count: int = 7,
        sha256: str | None = DIGEST,
        direction_source: str | None = "grammar",
        null_marker: bool = False,
        connect_error: Exception | None = None,
        count_error: Exception | None = None,
        marker_error: Exception | None = None,
    ) -> None:
        self.count = count
        self.sha256 = sha256
        self.direction_source = direction_source
        self.null_marker = null_marker
        self.connect_error = connect_error
        self.count_error = count_error
        self.marker_error = marker_error
        self.queries: list[str] = []
        self.closed = False

    def execute_query(self, query: str, *args, **kwargs) -> _FakeEagerResult:
        self.queries.append(query)
        if len(self.queries) == 1:
            if self.connect_error is not None:
                raise self.connect_error
            return _FakeEagerResult([_FakeRecord(ok=1)])
        if "_OspreySeed" in query:
            if self.marker_error is not None:
                raise self.marker_error
            if self.null_marker:
                return _FakeEagerResult([_FakeRecord(sha256=None, direction_source=None)])
            if self.sha256 is None:
                return _FakeEagerResult([])
            return _FakeEagerResult(
                [_FakeRecord(sha256=self.sha256, direction_source=self.direction_source)]
            )
        if self.count_error is not None:
            raise self.count_error
        return _FakeEagerResult([_FakeRecord(count=self.count)])

    def close(self) -> None:
        self.closed = True

    @property
    def marker_queries(self) -> list[str]:
        """Every query this driver was asked that reads the seed marker."""
        return [q for q in self.queries if "_OspreySeed" in q]


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


async def test_configured_emits_all_three_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_driver(monkeypatch, _FakeDriver())
    by_name = await _run(_cfg())
    assert set(by_name) == {"graphdb_connection", "graphdb_resources", "graphdb_seed"}
    assert all(r.category == "graphdb" for r in by_name.values())


async def test_connection_row_ok_with_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_driver(monkeypatch, _FakeDriver())
    row = (await _run(_cfg()))["graphdb_connection"]
    assert row.status is Status.OK
    # `services.graphdb` sets no port_host, so the store is on the deployment's
    # `graphdb_bolt` layout slot.
    assert f"bolt://localhost:{default_port('graphdb_bolt')}" in row.message
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
# Seed marker
# --------------------------------------------------------------------------- #


async def test_seed_row_reports_the_marker_digest_and_direction_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marked store is ``ok``, showing the digest prefix and where directions came from."""
    driver = _FakeDriver()
    _install_driver(monkeypatch, driver)
    row = (await _run(_cfg()))["graphdb_seed"]
    assert row.status is Status.OK
    assert row.value.startswith(DIGEST[:12])
    assert DIGEST[:13] not in row.value, "the row must abbreviate, not print the whole digest"
    assert "grammar" in row.value
    # The marker is read off the SAME driver the other two rows used, matched on
    # the label and the kind the seeder MERGEs it under.
    assert len(driver.marker_queries) == 1
    assert "kind" in driver.marker_queries[0]


async def test_seed_row_carries_the_full_digest_for_a_later_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``details`` holds the whole sha256, so no caller has to re-query the store."""
    _install_driver(monkeypatch, _FakeDriver())
    row = (await _run(_cfg()))["graphdb_seed"]
    assert row.details.startswith(SEED_DIGEST_DETAIL_PREFIX)
    assert DIGEST in row.details
    assert seed_digest(row) == DIGEST


async def test_seed_row_omits_a_direction_source_the_corpus_did_not_declare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corpus with no direction header leaves the value as the digest prefix alone."""
    _install_driver(monkeypatch, _FakeDriver(direction_source=None))
    row = (await _run(_cfg()))["graphdb_seed"]
    assert row.status is Status.OK
    assert row.value == DIGEST[:12]


async def test_corpus_without_a_marker_warns_that_it_cannot_be_identified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resources but no marker: a crashed import or a store seeded outside osprey."""
    _install_driver(monkeypatch, _FakeDriver(count=7, sha256=None))
    row = (await _run(_cfg()))["graphdb_seed"]
    assert row.status is Status.WARNING
    assert "no seed marker" in row.message
    assert "unseeded" not in row.message, "a store holding a corpus is not unseeded"
    assert "osprey knowledge seed-graph" in row.details
    assert seed_digest(row) == ""


async def test_empty_store_without_a_marker_warns_unseeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No resources and no marker: nothing has been imported yet."""
    _install_driver(monkeypatch, _FakeDriver(count=0, sha256=None))
    row = (await _run(_cfg()))["graphdb_seed"]
    assert row.status is Status.WARNING
    assert "unseeded" in row.message
    assert "osprey knowledge seed-graph" in row.details


async def test_marker_without_a_digest_reads_as_no_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker node carrying a null sha256 identifies nothing, so it warns."""
    _install_driver(monkeypatch, _FakeDriver(count=7, null_marker=True))
    row = (await _run(_cfg()))["graphdb_seed"]
    assert row.status is Status.WARNING
    assert "no seed marker" in row.message


async def test_unreadable_marker_warns_but_keeps_the_other_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed marker read degrades its own row and leaves the other two standing."""
    from neo4j.exceptions import Neo4jError

    _install_driver(monkeypatch, _FakeDriver(marker_error=Neo4jError("boom")))
    by_name = await _run(_cfg())
    assert by_name["graphdb_connection"].status is Status.OK
    assert by_name["graphdb_resources"].status is Status.OK
    assert by_name["graphdb_seed"].status is Status.WARNING
    assert "seed marker" in by_name["graphdb_seed"].message


async def test_unmarked_store_with_an_unreadable_count_claims_neither_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no count to go on, the row must not assert that the store is unseeded."""
    _install_driver(monkeypatch, _FakeDriver(sha256=None, count_error=RuntimeError("boom")))
    row = (await _run(_cfg()))["graphdb_seed"]
    assert row.status is Status.WARNING
    assert "no seed marker" in row.message
    assert "unseeded" not in row.message
    assert "unknown" in row.details


def test_seed_digest_is_empty_for_a_row_that_carries_none() -> None:
    """The accessor answers "" rather than guessing at unrelated detail text."""
    assert seed_digest(CheckResult("graphdb_seed", "graphdb", Status.WARNING, "unseeded")) == ""
    assert (
        seed_digest(
            CheckResult(
                "graphdb_resources",
                "graphdb",
                Status.OK,
                "counted",
                details="Import the TTL corpus.",
            )
        )
        == ""
    )


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
        _FakeDriver(sha256=None),
        _FakeDriver(count=0, sha256=None),
        _FakeDriver(null_marker=True),
        _FakeDriver(marker_error=RuntimeError("boom")),
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
