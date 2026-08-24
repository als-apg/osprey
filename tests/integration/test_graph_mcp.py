"""The graph MCP tools against a real Neo4j 5.20 + neosemantics store.

Every other graph test mocks the driver.  That proves the call shapes and the
envelope wiring, but it cannot prove the four things this feature actually
rests on:

* that ``Session.execute_read`` really refuses a write, and really refuses it
  with the status code :class:`~osprey.mcp_server.graph.server_context.GraphReadOnlyViolation`
  keys off — a mocked driver can only assert we *believe* it does;
* that the gate's refuse/pass matrix lines up with what the server would have
  done, i.e. that everything it passes is valid Cypher a read transaction
  accepts, and everything it refuses never reaches the wire;
* that the curated examples in
  :mod:`osprey.mcp_server.graph.tools.examples_data` are runnable Cypher which
  reproduces, on the shipped demo corpus, the verified counts — and returns
  rows with the parameter values shipped beside them;
* that the row cap and the query timeout are enforced by the *store* rather
  than by a client-side slice.

One store, seeded once per module with ``demo_machine.ttl`` (generated from
the control_assistant channel database) through the same primitives
``osprey knowledge seed-graph`` uses.

Skips are loud and only ever about the host.  If Docker is not reachable the
whole module skips with that reason; if Docker *is* reachable every test here
runs, because a graph test that quietly passes without a graph is worse than
no test at all.

The container recipe — pinned n10s jar bind-mounted at ``/plugins`` instead of
``NEO4J_PLUGINS``, APOC copied out of the image — is the shared one in
``tests/_graphdb_container.py``; see that module's docstring for why
``NEO4J_PLUGINS`` is deliberately unset.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from tests._container_support import start_or_skip, stop_quietly
from tests._graphdb_container import (
    GRAPHDB_TEST_PASSWORD,
    GRAPHDB_TEST_USERNAME,
    NEO4J_IMAGE,
)

logger = logging.getLogger(__name__)

# xdist_group("docker"): this module starts real containers, and the docker
# group is what keeps every such file on one xdist worker.
pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("docker")]


# --- Verified counts for the shipped demo_machine.ttl -----------------------
# Identical to tests/integration/test_graphdb_store.py; re-stated rather than
# imported so this module says what it is asserting, and so a change to the
# corpus has to be acknowledged in both places.

EXPECTED_DEVICES = 512
EXPECTED_BINDINGS = 2908
EXPECTED_WRITE_ONLY = 396
EXPECTED_READ_ONLY = 2512
EXPECTED_MAGNETS = 382

_SEM = "https://narad.example.org/schema/shared_semantics/"
MAGNET_CLASS_URI = _SEM + "Magnet"

#: Corpus key in :data:`~osprey.mcp_server.graph.tools.examples_data.EXAMPLE_QUERIES`
#: parameter sets.
DEMO = "demo"

#: Labels and properties ``get_schema`` must never surface.
BOOKKEEPING_LABELS = ("_OspreySeed", "_GraphConfig", "_NsPrefDef")
BOOKKEEPING_PROPERTIES = ("sha256", "seededAt", "kind")

#: Relationship types neosemantics writes for the NARAD predicates under
#: ``applyNeo4jNaming``.  A schema payload missing one of these means the
#: projection changed shape and every curated example is now wrong.
NARAD_RELATIONSHIP_TYPES = ("HASBINDING", "READSSIGNAL", "WRITESSIGNAL", "SUBCLASSOF", "TYPE")


# ---------------------------------------------------------------------------
# Plugins + container
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph_mcp_plugin_dir(graphdb_plugin_dir: Path) -> Path:
    """n10s + APOC, from the session-wide resolver (tests/_graphdb_container).

    The loud-skip gate for the whole module lives in that resolver: it is the
    first thing every store fixture depends on, so an unreachable daemon is
    reported once, with its reason, instead of as a wall of unexplained skips.
    """
    return graphdb_plugin_dir


def _seeded_store(plugin_dir: Path, ttl_text: str, label: str) -> Iterator[str]:
    """Start a store, seed it through the real seeder, and yield its bolt URI.

    Seeding goes through :mod:`~osprey.services.facility_knowledge.seeder.graph_seeder`
    rather than raw Cypher because that is the path ``osprey knowledge
    seed-graph`` takes — including ``write_marker``, which is what puts the
    ``_OspreySeed`` bookkeeping node in the store that ``get_schema`` then has
    to hide.
    """
    try:
        from testcontainers.community.neo4j import Neo4jContainer
    except ImportError:  # pragma: no cover - depends on the installed extras
        pytest.skip("testcontainers' neo4j module is not installed")

    def _build() -> Neo4jContainer:
        container = Neo4jContainer(image=NEO4J_IMAGE, password=GRAPHDB_TEST_PASSWORD)
        container.with_volume_mapping(str(plugin_dir), "/plugins", "rw")
        container.with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*,n10s.*")
        container.with_env("NEO4J_dbms_security_procedures_allowlist", "apoc.*,n10s.*")
        return container

    container = start_or_skip(_build, label=f"graphdb for {label}")
    try:
        uri = container.get_connection_url()

        from osprey.services.facility_knowledge.seeder import graph_seeder

        with graph_seeder.open_session(
            uri, GRAPHDB_TEST_USERNAME, GRAPHDB_TEST_PASSWORD
        ) as session:
            bootstrap = graph_seeder.bootstrap(session)
            assert bootstrap.ok, bootstrap.message
            imported = graph_seeder.import_ttl(session, ttl_text)
            assert imported.termination_status == graph_seeder.TERMINATION_OK, imported.extra_info
            graph_seeder.write_marker(session, graph_seeder.ttl_sha256(ttl_text))
            logger.info(
                f"{label}: seeded {imported.triples_loaded} triples, "
                f"{graph_seeder.resource_count(session)} Resource nodes"
            )
        yield uri
    finally:
        stop_quietly(container)


@pytest.fixture(scope="module")
def demo_store(graph_mcp_plugin_dir: Path) -> Iterator[str]:
    """A store seeded with the generated demo-machine corpus."""
    resource = (
        files("osprey.templates")
        .joinpath("apps")
        .joinpath("control_assistant")
        .joinpath("data")
        .joinpath("demo_machine.ttl")
    )
    yield from _seeded_store(graph_mcp_plugin_dir, resource.read_text(encoding="utf-8"), "demo")


# ---------------------------------------------------------------------------
# The server context, wired to a container
# ---------------------------------------------------------------------------


@contextmanager
def _installed_context(uri: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Install a GraphContext singleton pointed at *uri*.

    The two config seams ``initialize()`` reads through are replaced rather
    than a config file written: the point of this module is the store, and a
    fabricated ``config.yml`` would only be re-testing
    ``resolve_graphdb_connection``.  The password travels the way a deployment
    delivers it — through ``GRAPHDB_PASSWORD`` in the environment — and is
    never logged or interpolated into the URI.

    The singleton is module-global, so it is installed per test rather than per
    module, which is also what keeps one test's cap or timeout override from
    leaking into the next.
    """
    from osprey.deployment.graphdb_service import GRAPHDB_PASSWORD_ENV
    from osprey.mcp_server.graph import server_context as graph_ctx

    monkeypatch.setenv(GRAPHDB_PASSWORD_ENV, GRAPHDB_TEST_PASSWORD)
    monkeypatch.setattr(
        graph_ctx, "load_osprey_config", lambda: {"services": {"graphdb": {"uri": uri}}}
    )
    monkeypatch.setattr(graph_ctx, "get_config_value", lambda key, default=None: default)

    context = graph_ctx.initialize_server_context()
    try:
        # Inside the try: initialize() has already installed the singleton, so
        # a failed assertion here must still tear it down or every later test
        # inherits a context pointed at this lane.
        assert context.configured, (
            "the graph context did not read the container as a configured store"
        )
        yield context
    finally:
        graph_ctx.reset_server_context()


@pytest.fixture
def demo_ctx(demo_store: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The graph tools, live against the demo-seeded store."""
    with _installed_context(demo_store, monkeypatch) as context:
        yield context


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------


def _read_cypher(query: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call the ``read_cypher`` tool function and return its parsed payload."""
    from osprey.mcp_server.graph.tools.read_cypher import read_cypher

    fn = getattr(read_cypher, "fn", read_cypher)
    return json.loads(fn(query, params))


def _get_schema() -> dict[str, Any]:
    """Call the ``get_schema`` tool function and return its parsed payload."""
    from osprey.mcp_server.graph.tools.get_schema import get_schema

    fn = getattr(get_schema, "fn", get_schema)
    return json.loads(fn())


def _error_envelope(query: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run ``read_cypher`` expecting the standard error envelope, and return it.

    ``make_error`` raises a ``ToolError`` whose message *is* the envelope JSON,
    which is how fastmcp puts ``isError=True`` on the wire, so the envelope is
    read back off the exception rather than off a return value.
    """
    with pytest.raises(ToolError) as excinfo:
        _read_cypher(query, params)
    envelope = json.loads(str(excinfo.value))
    assert envelope.get("error") is True, envelope
    return envelope


@contextmanager
def _run_read_spy(context: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record every query that reaches ``GraphContext.run_read``.

    A refusal that never dials is the whole point of the gate, and "no rows
    came back" cannot distinguish it from a query the store ran and answered
    with nothing.
    """
    seen: list[str] = []
    original = context.run_read

    def _spy(cypher: str, params: Any = None, **kwargs: Any) -> Any:
        seen.append(cypher)
        return original(cypher, params, **kwargs)

    monkeypatch.setattr(context, "run_read", _spy)
    yield seen


def _example(key: str) -> Any:
    """Return the curated example with *key*."""
    from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES

    for example in EXAMPLE_QUERIES:
        if example.key == key:
            return example
    raise AssertionError(f"no curated example keyed {key!r}")


def _example_keys() -> list[str]:
    """Every curated example key, in shipped order."""
    from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES

    return [example.key for example in EXAMPLE_QUERIES]


def _run_example(key: str, corpus: str) -> dict[str, Any]:
    """Run one curated example with the parameter set for *corpus*."""
    example = _example(key)
    return _read_cypher(example.cypher, dict(example.parameters[corpus]))


def _store_state(uri: str) -> tuple[int, str | None]:
    """Read the store's corpus size and seed marker straight off the driver."""
    from osprey.services.facility_knowledge.seeder import graph_seeder

    with graph_seeder.open_session(uri, GRAPHDB_TEST_USERNAME, GRAPHDB_TEST_PASSWORD) as session:
        return graph_seeder.resource_count(session), graph_seeder.read_marker(session)


# ---------------------------------------------------------------------------
# 1. Read-only enforcement
# ---------------------------------------------------------------------------


def test_write_is_refused_by_the_read_transaction_and_changes_nothing(
    demo_ctx: Any, demo_store: str
) -> None:
    """A CREATE reaches the store, is refused there, and leaves no trace.

    The gate deliberately does not vet write keywords — write enforcement is
    the read transaction's job — so this is the only test that can prove the
    refusal is real rather than assumed.  The before/after read goes through a
    separate driver session so it cannot be fooled by the transaction the tool
    used.
    """
    before_count, before_marker = _store_state(demo_store)
    assert before_count > 0 and before_marker is not None, (
        "the fixture must leave a seeded, marked store for this test to mean anything"
    )

    envelope = _error_envelope(
        "CREATE (n:_OspreyIntegrationWriteProbe {marker: 'should-never-exist'}) RETURN n"
    )

    assert envelope["error_type"] == "validation_error", envelope
    assert envelope["details"]["kind"] == "read_only", envelope
    assert envelope["details"]["code"] == "Neo.ClientError.Statement.AccessMode", envelope

    after_count, after_marker = _store_state(demo_store)
    assert after_count == before_count
    assert after_marker == before_marker

    probe = _read_cypher("MATCH (n:_OspreyIntegrationWriteProbe) RETURN count(n) AS n")
    assert probe["rows"][0]["n"] == 0, "the refused CREATE left a node behind"


# ---------------------------------------------------------------------------
# 2. The gate, live
# ---------------------------------------------------------------------------

#: ``(id, query, substring the refusal must name)``.  Each is a shape the read
#: transaction alone would *not* stop: APOC and n10s procedures run happily in
#: read mode, and so does ``LOAD CSV`` from a URL.
_REFUSED: list[tuple[str, str, str]] = [
    (
        "apoc_procedure",
        "CALL apoc.load.json('http://example.org/x.json') YIELD value RETURN value",
        "apoc.load.json",
    ),
    (
        "apoc_procedure_backtick_segments",
        "CALL `apoc`.`load`.`json`('http://example.org/x.json') YIELD value RETURN value",
        "apoc.load.json",
    ),
    (
        "apoc_function_run_first_column_many",
        "RETURN apoc.cypher.runFirstColumnMany('MATCH (n) RETURN n', {}) AS smuggled",
        "apoc.cypher.runfirstcolumnmany",
    ),
    (
        "apoc_function_run_first_column_single",
        "RETURN apoc.cypher.runFirstColumnSingle('MATCH (n) RETURN n', {}) AS smuggled",
        "apoc.cypher.runfirstcolumnsingle",
    ),
    (
        "load_csv",
        'LOAD CSV FROM "http://example.org/rows.csv" AS row RETURN row LIMIT 1',
        "LOAD CSV",
    ),
    (
        "load_csv_split_by_comment",
        'LOAD /* sneaky */ CSV FROM "http://example.org/rows.csv" AS row RETURN row LIMIT 1',
        "LOAD CSV",
    ),
    (
        "comment_with_apostrophe_before_call",
        "/* it's only a comment */ CALL apoc.load.json('x') YIELD value RETURN value",
        "apoc.load.json",
    ),
    (
        "quote_inside_backtick_identifier",
        "CALL `apo'c`.load.json('x') YIELD value RETURN value",
        "apo'c.load.json",
    ),
    (
        "nested_in_subquery",
        "CALL { MATCH (n:Resource) CALL apoc.load.json('x') YIELD value RETURN value } "
        "RETURN 1 AS ok",
        "apoc.load.json",
    ),
    (
        "n10s_import",
        "CALL n10s.rdf.import.inline('<a> <b> <c> .', 'Turtle') YIELD terminationStatus "
        "RETURN terminationStatus",
        "n10s.rdf.import.inline",
    ),
    (
        "dbms_procedure",
        "CALL dbms.components() YIELD name RETURN name",
        "dbms.components",
    ),
]


@pytest.mark.parametrize(
    ("query", "named"),
    [pytest.param(query, named, id=case_id) for case_id, query, named in _REFUSED],
)
def test_gate_refuses_before_the_store_is_dialed(
    demo_ctx: Any, monkeypatch: pytest.MonkeyPatch, query: str, named: str
) -> None:
    """A refused query yields a validation_error naming it, and never dials."""
    with _run_read_spy(demo_ctx, monkeypatch) as seen:
        envelope = _error_envelope(query)

    assert envelope["error_type"] == "validation_error", envelope
    assert named in envelope["error_message"], envelope["error_message"]
    assert seen == [], f"a refused query still reached the store: {seen}"


#: Queries the gate must let through *and* the store must accept.  Two of them
#: (the string literal and the comment) contain the exact text the refusals
#: above trip on, which is what proves the scanner reads Cypher rather than
#: grepping it.
_PASSED: list[tuple[str, str]] = [
    ("call_subquery", "CALL { MATCH (r:Resource) RETURN count(r) AS n } RETURN n LIMIT 1"),
    ("allowlisted_procedure", "CALL db.labels() YIELD label RETURN label ORDER BY label LIMIT 1"),
    ("string_literal_mentions_apoc", "RETURN 'CALL apoc.load.json(1)' AS quoted LIMIT 1"),
    ("comment_mentions_apoc", "// CALL apoc.load.json(1)\nRETURN 1 AS ok"),
    ("db_as_a_variable_name", "MATCH (db:Resource) RETURN db.sourceName AS name LIMIT 1"),
    ("load_as_a_property_name", "MATCH (d:Resource) RETURN d.load AS load_prop LIMIT 1"),
]


@pytest.mark.parametrize("query", [pytest.param(query, id=case_id) for case_id, query in _PASSED])
def test_gate_passes_queries_the_store_then_runs(
    demo_ctx: Any, monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    """A passed query is not merely un-refused — it executes and returns a row.

    The spy runs here too, as the positive control for the refusal tests: those
    assert ``seen == []``, which an inert spy would satisfy just as well as a
    working gate.  Seeing the query arrive at ``run_read`` on this path is what
    makes the empty list over there mean something.
    """
    with _run_read_spy(demo_ctx, monkeypatch) as seen:
        payload = _read_cypher(query)

    assert seen == [query], f"the spy did not observe the read seam: {seen}"
    assert payload["row_count"] == 1, payload
    assert payload["truncated"] is False, payload


# ---------------------------------------------------------------------------
# 3. JSON round-trip of a whole node
# ---------------------------------------------------------------------------


def test_returning_a_node_yields_json_native_values(demo_ctx: Any) -> None:
    """A bare node comes back as plain JSON, not as driver objects.

    Two halves, and they prove different things.  The tool payload shows what
    an agent receives — a node flattened to a properties dict carrying its
    ``uri``.  But it cannot show *who* made it JSON: ``read_cypher`` serialises
    with ``default=str``, which would have quietly stringified anything the
    JSON-safety pass missed, so re-serialising its output proves nothing.  So
    the second half reads the same query through the context and serialises
    those rows with **no fallback**: that only succeeds if ``run_read`` handed
    back JSON-native values in the first place.
    """
    payload = _read_cypher("MATCH (d:Resource) RETURN d LIMIT 1")

    assert payload["row_count"] == 1, payload
    assert payload["columns"] == ["d"], payload
    node = payload["rows"][0]["d"]
    assert isinstance(node, dict) and node.get("uri"), node

    result = demo_ctx.run_read("MATCH (d:Resource) RETURN d LIMIT 1")
    assert result.rows, result
    json.dumps(result.rows)  # raises TypeError on anything not JSON-native


# ---------------------------------------------------------------------------
# 4. get_schema hygiene
# ---------------------------------------------------------------------------


def test_get_schema_reports_the_corpus_and_hides_the_bookkeeping(demo_ctx: Any) -> None:
    """The schema names the NARAD vocabulary and nothing the seeder wrote."""
    payload = _get_schema()

    assert "Resource" in payload["labels"]
    assert "ChannelBinding" in payload["labels"]
    assert "Class" in payload["labels"]
    for relationship_type in NARAD_RELATIONSHIP_TYPES:
        assert relationship_type in payload["relationship_types"], payload["relationship_types"]

    for label in BOOKKEEPING_LABELS:
        assert label not in payload["labels"], payload["labels"]
        assert label not in payload["properties_by_label"], list(payload["properties_by_label"])

    every_property = {name for names in payload["properties_by_label"].values() for name in names}
    for name in BOOKKEEPING_PROPERTIES:
        assert name not in every_property, sorted(every_property)

    # The seed marker really is in the store — otherwise the exclusions above
    # are vacuous.  Its label is queryable directly; it is only *listed*
    # nowhere.
    marker = _read_cypher("MATCH (m:_OspreySeed) RETURN count(m) AS n")
    assert marker["rows"][0]["n"] == 1, marker

    assert payload["properties_by_label"]["ChannelBinding"], payload["properties_by_label"]
    assert "fullPv" in payload["properties_by_label"]["ChannelBinding"]


# ---------------------------------------------------------------------------
# 5. The curated examples against the corpus
# ---------------------------------------------------------------------------


def test_example_q1a_counts_the_verified_devices(demo_ctx: Any) -> None:
    """The device census sums to the 512 verified devices."""
    payload = _run_example("q1a", DEMO)
    assert payload["truncated"] is False, payload
    assert sum(row["device_count"] for row in payload["rows"]) == EXPECTED_DEVICES


def test_example_q5_reproduces_the_verified_direction_split(demo_ctx: Any) -> None:
    """The binding rollup reproduces 396 write-only / 2512 read-only / 2908 total."""
    payload = _run_example("q5", DEMO)
    row = payload["rows"][0]
    assert row["write_only"] == EXPECTED_WRITE_ONLY, row
    assert row["read_only"] == EXPECTED_READ_ONLY, row
    assert row["readwrite"] == 0, row
    assert row["total"] == EXPECTED_BINDINGS, row


def test_example_q1c_rolls_up_the_verified_magnets(demo_ctx: Any) -> None:
    """Rolling ``Magnet`` up its subclasses lists exactly the 382 verified devices.

    The rollup is larger than the shipped row cap, so the cap is lifted to the
    example's own ``LIMIT`` for this test: the claim here is the count, and a
    capped answer would compare 200 against 382 and say nothing about the
    hierarchy.
    """
    assert _example("q1c").parameters[DEMO]["class_uri"] == MAGNET_CLASS_URI, (
        "the count below is the verified Magnet rollup, so it only means "
        "anything while the shipped parameter set still names Magnet"
    )
    demo_ctx.query_max_rows = 500

    payload = _run_example("q1c", DEMO)
    assert payload["truncated"] is False, payload
    assert payload["row_count"] == EXPECTED_MAGNETS, payload["row_count"]
    assert {row["device_class"] for row in payload["rows"]} > {"Quadrupole"}


def test_example_q1b_puts_the_magnets_under_one_branch(demo_ctx: Any) -> None:
    """The branch rollup reaches the same 382 without naming a magnet subclass."""
    payload = _run_example("q1b", DEMO)
    by_branch = {row["branch"]: row["device_count"] for row in payload["rows"]}
    assert by_branch.get("Magnet") == EXPECTED_MAGNETS, by_branch


@pytest.mark.parametrize("key", _example_keys())
def test_every_example_runs_on_the_demo_corpus(demo_ctx: Any, key: str) -> None:
    """Every shipped example returns usable rows with its demo parameter set."""
    payload = _run_example(key, DEMO)
    _assert_usable(key, DEMO, payload)


def _assert_usable(key: str, corpus: str, payload: dict[str, Any]) -> None:
    """Assert an example returned rows whose columns carry values.

    No null is tolerated: every device in the generated corpus carries a
    position and an ordinal, so a null column here is a projection or query
    fault rather than a corpus fact.

    Truncation is deliberately not asserted here: whether an example's own
    ``LIMIT`` lands above or below the row cap is a property of the corpus (the
    demo machine has 382 magnets to a 200-row cap), and the tests that need a
    *complete* answer to compare against a verified count say so themselves.
    """
    assert payload["row_count"] >= 1, f"{key} returned no rows: {payload}"

    for row in payload["rows"]:
        for column, value in row.items():
            assert value is not None, f"{key} returned a null {column} on {corpus}: {row}"


def test_example_q6_finds_one_owner_and_no_shared_endpoint(demo_ctx: Any) -> None:
    """The reverse PV lookup resolves an address back to exactly one device.

    The generated corpus mints one binding per channel, so an address maps to
    exactly one device: ``device_count`` of 1 is the *expected* answer there
    and is what says "no shared endpoints", not a failure to find any.
    """
    payload = _run_example("q6", DEMO)
    assert payload["row_count"] == 1, payload
    row = payload["rows"][0]
    assert row["pv"] == _example("q6").parameters[DEMO]["pv"]
    assert row["device_count"] == 1, row
    assert row["devices"], row


# ---------------------------------------------------------------------------
# 6. Bounds: the timeout and the row cap
# ---------------------------------------------------------------------------


def test_a_slow_query_comes_back_as_a_timeout_envelope(demo_ctx: Any) -> None:
    """The store terminates a runaway query and the tool reports it as such.

    A three-way cartesian product with a predicate: the predicate keeps the
    planner off the count store, so the query really does compare on the order
    of a billion pairs, and it streams rather than materialising, so the server
    hits the transaction timeout instead of the heap.
    """
    demo_ctx.query_timeout_s = 1

    envelope = _error_envelope(
        "MATCH (a:Resource), (b:Resource), (c:Resource) "
        "WHERE a.uri < b.uri AND b.uri < c.uri "
        "RETURN count(*) AS n"
    )

    assert envelope["error_type"] == "timeout_error", envelope
    assert envelope["suggestions"], envelope


def test_the_row_cap_truncates_and_says_so(demo_ctx: Any) -> None:
    """A query with more matches than the cap returns exactly the cap, flagged."""
    demo_ctx.query_max_rows = 5

    payload = _read_cypher("MATCH (b:ChannelBinding) RETURN b.fullPv AS pv")

    assert payload["row_count"] == 5, payload["row_count"]
    assert payload["truncated"] is True, payload
    assert len(payload["rows"]) == 5
    assert any("query_max_rows" in line for line in payload["guidance"]), payload["guidance"]
