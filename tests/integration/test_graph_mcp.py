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
* that the nine curated examples in
  :mod:`osprey.mcp_server.graph.tools.examples_data` are runnable Cypher which
  reproduces, on the shipped ALS corpus, the counts phase 1 verified — and
  returns rows on the generated demo corpus with the parameter values shipped
  beside them;
* that the row cap and the query timeout are enforced by the *store* rather
  than by a client-side slice.

Two lanes, two stores.  Neo4j Community serves exactly one database, so the
two shipped corpora cannot share a container: ``als_gtb.ttl`` (the ALS GTB
snapshot the verified counts come from) and ``demo_machine.ttl`` (generated
from the control_assistant channel database) each get their own, seeded once
per module through the same primitives ``osprey knowledge seed-graph`` uses.

Skips are loud and only ever about the host.  If Docker is not reachable the
whole module skips with that reason; if Docker *is* reachable every test here
runs, because a graph test that quietly passes without a graph is worse than
no test at all.

The container recipe — pinned n10s jar bind-mounted at ``/plugins`` instead of
``NEO4J_PLUGINS``, APOC copied out of the image — is the one
``tests/integration/test_graphdb_store.py`` established; see that module's
docstring for why ``NEO4J_PLUGINS`` is deliberately unset.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
import requests
from fastmcp.exceptions import ToolError

from tests._container_support import (
    is_docker_available,
    is_image_present,
    start_or_skip,
    stop_quietly,
)

logger = logging.getLogger(__name__)

# xdist_group("docker"): this module starts real containers, and the docker
# group is what keeps every such file on one xdist worker.
pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("docker")]


# ---------------------------------------------------------------------------
# Pins (same as tests/integration/test_graphdb_store.py — see its docstring)
# ---------------------------------------------------------------------------

NEO4J_IMAGE = "neo4j:5.20-community"
N10S_VERSION = "5.20.0"
N10S_JAR_URL = (
    f"https://github.com/neo4j-labs/neosemantics/releases/download/"
    f"{N10S_VERSION}/neosemantics-{N10S_VERSION}.jar"
)
N10S_JAR_ENV = "OSPREY_TEST_N10S_JAR"
IMAGE_LABS_DIR = "/var/lib/neo4j/labs"

#: Satisfies the deploy-time validator's rules for ``GRAPHDB_PASSWORD``
#: (>= 8 characters, no ``/``).  It is set into the environment for the
#: duration of a context fixture and never printed: the connection resolver
#: reads it from ``GRAPHDB_PASSWORD`` exactly as a deployment does.
GRAPHDB_TEST_PASSWORD = "ospreytest1234"
GRAPHDB_TEST_USERNAME = "neo4j"

# --- Counts verified in phase 1 for the shipped als_gtb.ttl ----------------
# Identical to tests/integration/test_graphdb_store.py; re-stated rather than
# imported so this module says what it is asserting, and so a change to either
# corpus has to be acknowledged in both places.

EXPECTED_DEVICES = 76
EXPECTED_BINDINGS = 964
EXPECTED_WRITE_ONLY = 260
EXPECTED_READ_ONLY = 704
EXPECTED_MAGNETS = 47

_SEM = "https://narad.example.org/schema/shared_semantics/"
MAGNET_CLASS_URI = _SEM + "Magnet"

#: Corpus keys in :data:`~osprey.mcp_server.graph.tools.examples_data.EXAMPLE_QUERIES`
#: parameter sets.
ALS = "als"
DEMO = "demo"

#: Columns one example is documented to return as ``null`` **on one corpus** —
#: a property of that corpus rather than a fault in the query.
#:
#: Keyed by ``(example, corpus)`` rather than by example alone, because a null
#: tolerated everywhere is a null nobody notices: q3's ``signal`` is non-null on
#: every row of both corpora today, so exempting it globally would retire
#: exactly the regression q3 is here to catch.  The one real entry is q2's
#: ``s_m`` on the ALS snapshot: the example deliberately drops the prototype's
#: ``sPositionM IS NOT NULL`` filter and falls back to ``ordinalInSection`` for
#: the ordering, so that a corpus recording only ordinals still walks in a
#: meaningful order — and the ALS corpus exercises exactly that path, carrying
#: 77 ``sectionCode`` values to 76 ``sPositionM`` ones.  Filtering the odd one
#: out would hide a device an operator can address, so
#: :func:`test_als_example_q2_has_exactly_one_device_without_a_position` pins
#: which device it is instead.
DOCUMENTED_NULL_COLUMNS: dict[tuple[str, str], frozenset[str]] = {
    ("q2", ALS): frozenset({"s_m"}),
}

#: The one ALS device with no ``sPositionM``.
DEVICE_WITHOUT_POSITION = "DCCT_0"

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


def _fetch_n10s_jar(dest_dir: Path) -> None:
    """Put the pinned neosemantics jar in *dest_dir*, or skip the module."""
    target = dest_dir / f"neosemantics-{N10S_VERSION}.jar"

    override = os.environ.get(N10S_JAR_ENV)
    if override:
        source = Path(override).expanduser()
        if not source.is_file():
            pytest.skip(f"{N10S_JAR_ENV} points at {source}, which is not a file")
        target.write_bytes(source.read_bytes())
        logger.info(f"n10s jar taken from {N10S_JAR_ENV}={source}")
        return

    try:
        with requests.get(N10S_JAR_URL, timeout=120, stream=True) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
    except requests.exceptions.RequestException as exc:
        pytest.skip(
            f"could not download the neosemantics jar from {N10S_JAR_URL} ({exc}); "
            f"set {N10S_JAR_ENV} to a local copy to run this test offline"
        )
    logger.info(f"n10s jar downloaded to {target} ({target.stat().st_size} bytes)")


def _copy_bundled_apoc(dest_dir: Path) -> None:
    """Copy the image's own APOC jar into *dest_dir* without starting anything."""
    import docker

    client = docker.from_env()
    container = client.containers.create(NEO4J_IMAGE)
    try:
        stream, _stat = container.get_archive(IMAGE_LABS_DIR)
        archive = io.BytesIO(b"".join(stream))
        copied = []
        with tarfile.open(fileobj=archive) as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if not (member.isfile() and name.startswith("apoc") and name.endswith(".jar")):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:  # pragma: no cover - defensive
                    continue
                (dest_dir / name).write_bytes(extracted.read())
                copied.append(name)
    finally:
        container.remove(force=True)

    if not copied:  # pragma: no cover - would mean the image changed shape
        pytest.skip(f"no APOC jar found under {IMAGE_LABS_DIR} in {NEO4J_IMAGE}")
    logger.info(f"APOC taken from the image: {', '.join(copied)}")


@pytest.fixture(scope="module")
def graph_mcp_plugin_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding n10s + APOC, resolved once for this module.

    The loud-skip gate for the whole module lives here: it is the first thing
    every store fixture depends on, so an unreachable daemon is reported once,
    with its reason, instead of as a wall of unexplained skips.
    """
    if not is_docker_available():
        pytest.skip(
            "docker daemon is not reachable — the graph MCP tools can only be "
            "verified against a real Neo4j + neosemantics store"
        )

    if not is_image_present(NEO4J_IMAGE):
        import docker

        logger.info(f"pulling {NEO4J_IMAGE} (not present locally)")
        try:
            docker.from_env().images.pull(NEO4J_IMAGE)
        except Exception as exc:
            pytest.skip(f"could not pull {NEO4J_IMAGE}: {exc}")

    plugin_dir = tmp_path_factory.mktemp("graph-mcp-plugins")
    _fetch_n10s_jar(plugin_dir)
    _copy_bundled_apoc(plugin_dir)
    for jar in plugin_dir.glob("*.jar"):
        jar.chmod(0o644)
    plugin_dir.chmod(0o755)
    return plugin_dir


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
def als_store(graph_mcp_plugin_dir: Path) -> Iterator[str]:
    """A store seeded with the shipped ALS corpus."""
    resource = (
        files("osprey.templates").joinpath("services").joinpath("graphdb").joinpath("als_gtb.ttl")
    )
    yield from _seeded_store(graph_mcp_plugin_dir, resource.read_text(encoding="utf-8"), "als")


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
    lane: two lanes cannot both be "the" context at once.
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
def als_ctx(als_store: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The graph tools, live against the ALS-seeded store."""
    with _installed_context(als_store, monkeypatch) as context:
        yield context


@pytest.fixture
def demo_ctx(demo_store: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The graph tools, live against the demo-seeded store."""
    with _installed_context(demo_store, monkeypatch) as context:
        yield context


@pytest.fixture
def lane_ctx(request: pytest.FixtureRequest) -> Any:
    """Resolve ``als_ctx``/``demo_ctx`` by name for lane-parametrized tests."""
    return request.getfixturevalue(request.param)


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
    als_ctx: Any, als_store: str
) -> None:
    """A CREATE reaches the store, is refused there, and leaves no trace.

    The gate deliberately does not vet write keywords — write enforcement is
    the read transaction's job — so this is the only test that can prove the
    refusal is real rather than assumed.  The before/after read goes through a
    separate driver session so it cannot be fooled by the transaction the tool
    used.
    """
    before_count, before_marker = _store_state(als_store)
    assert before_count > 0 and before_marker is not None, (
        "the fixture must leave a seeded, marked store for this test to mean anything"
    )

    envelope = _error_envelope(
        "CREATE (n:_OspreyIntegrationWriteProbe {marker: 'should-never-exist'}) RETURN n"
    )

    assert envelope["error_type"] == "validation_error", envelope
    assert envelope["details"]["kind"] == "read_only", envelope
    assert envelope["details"]["code"] == "Neo.ClientError.Statement.AccessMode", envelope

    after_count, after_marker = _store_state(als_store)
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
    als_ctx: Any, monkeypatch: pytest.MonkeyPatch, query: str, named: str
) -> None:
    """A refused query yields a validation_error naming it, and never dials."""
    with _run_read_spy(als_ctx, monkeypatch) as seen:
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
    als_ctx: Any, monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    """A passed query is not merely un-refused — it executes and returns a row.

    The spy runs here too, as the positive control for the refusal tests: those
    assert ``seen == []``, which an inert spy would satisfy just as well as a
    working gate.  Seeing the query arrive at ``run_read`` on this path is what
    makes the empty list over there mean something.
    """
    with _run_read_spy(als_ctx, monkeypatch) as seen:
        payload = _read_cypher(query)

    assert seen == [query], f"the spy did not observe the read seam: {seen}"
    assert payload["row_count"] == 1, payload
    assert payload["truncated"] is False, payload


# ---------------------------------------------------------------------------
# 3. JSON round-trip of a whole node
# ---------------------------------------------------------------------------


def test_returning_a_node_yields_json_native_values(als_ctx: Any) -> None:
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

    result = als_ctx.run_read("MATCH (d:Resource) RETURN d LIMIT 1")
    assert result.rows, result
    json.dumps(result.rows)  # raises TypeError on anything not JSON-native


# ---------------------------------------------------------------------------
# 4. get_schema hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane_ctx", ["als_ctx", "demo_ctx"], indirect=True)
def test_get_schema_reports_the_corpus_and_hides_the_bookkeeping(lane_ctx: Any) -> None:
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
# 5. The curated examples against both corpora
# ---------------------------------------------------------------------------


def test_als_example_q1a_counts_the_verified_devices(als_ctx: Any) -> None:
    """The device census sums to the 76 devices phase 1 verified."""
    payload = _run_example("q1a", ALS)
    assert payload["truncated"] is False, payload
    assert sum(row["device_count"] for row in payload["rows"]) == EXPECTED_DEVICES


def test_als_example_q5_reproduces_the_verified_direction_split(als_ctx: Any) -> None:
    """The binding rollup reproduces 260 write-only / 704 read-only / 964 total."""
    payload = _run_example("q5", ALS)
    row = payload["rows"][0]
    assert row["write_only"] == EXPECTED_WRITE_ONLY, row
    assert row["read_only"] == EXPECTED_READ_ONLY, row
    assert row["readwrite"] == 0, row
    assert row["total"] == EXPECTED_BINDINGS, row


def test_als_example_q1c_rolls_up_the_verified_magnets(als_ctx: Any) -> None:
    """Rolling ``Magnet`` up its subclasses lists exactly the 47 verified devices."""
    assert _example("q1c").parameters[ALS]["class_uri"] == MAGNET_CLASS_URI, (
        "the count below is the verified Magnet rollup, so it only means "
        "anything while the shipped parameter set still names Magnet"
    )

    payload = _run_example("q1c", ALS)
    assert payload["truncated"] is False, payload
    assert payload["row_count"] == EXPECTED_MAGNETS, payload["row_count"]
    assert {row["device_class"] for row in payload["rows"]} > {"Quadrupole"}


def test_als_example_q1b_puts_the_magnets_under_one_branch(als_ctx: Any) -> None:
    """The branch rollup reaches the same 47 without naming a magnet subclass."""
    payload = _run_example("q1b", ALS)
    by_branch = {row["branch"]: row["device_count"] for row in payload["rows"]}
    assert by_branch.get("Magnet") == EXPECTED_MAGNETS, by_branch


@pytest.mark.parametrize("key", _example_keys())
def test_every_example_runs_on_the_als_corpus(als_ctx: Any, key: str) -> None:
    """Every shipped example returns usable rows with its ALS parameter set."""
    payload = _run_example(key, ALS)
    _assert_usable(key, ALS, payload)


@pytest.mark.parametrize("key", _example_keys())
def test_every_example_runs_on_the_demo_corpus(demo_ctx: Any, key: str) -> None:
    """Every shipped example returns usable rows with its demo parameter set."""
    payload = _run_example(key, DEMO)
    _assert_usable(key, DEMO, payload)


def _assert_usable(key: str, corpus: str, payload: dict[str, Any]) -> None:
    """Assert an example returned rows whose columns carry values.

    The null exemption is per ``(example, corpus)`` so that a null which is a
    corpus fact on one lane is still a failure on the other.

    Truncation is deliberately not asserted here: whether an example's own
    ``LIMIT`` lands above or below the row cap is a property of the corpus (the
    demo machine has 382 magnets to the ALS snapshot's 47), and the tests that
    need a *complete* answer to compare against a verified count say so
    themselves.
    """
    assert payload["row_count"] >= 1, f"{key} returned no rows: {payload}"

    allowed_null = DOCUMENTED_NULL_COLUMNS.get((key, corpus), frozenset())
    for row in payload["rows"]:
        for column, value in row.items():
            if column in allowed_null:
                continue
            assert value is not None, f"{key} returned a null {column} on {corpus}: {row}"


def test_als_example_q2_has_exactly_one_device_without_a_position(als_ctx: Any) -> None:
    """Pin the single ALS device q2's ``s_m`` exemption exists for.

    The exemption in :data:`DOCUMENTED_NULL_COLUMNS` says "a null position is
    tolerated on this lane", which on its own would also tolerate a corpus that
    lost every ``sPositionM``.  This says which device, and how many.
    """
    payload = _run_example("q2", ALS)

    without_position = [row for row in payload["rows"] if row["s_m"] is None]
    assert len(without_position) == 1, without_position
    assert without_position[0]["device"] == DEVICE_WITHOUT_POSITION, without_position[0]
    assert without_position[0]["ordinal"] is not None, (
        "the ordinal is what q2 falls back to for ordering, so a device with "
        "neither position nor ordinal would sort arbitrarily"
    )


def test_demo_example_q6_finds_one_owner_and_no_shared_endpoint(demo_ctx: Any) -> None:
    """The reverse PV lookup answers on the demo corpus without error.

    The generated corpus mints one binding per channel, so an address maps to
    exactly one device: ``device_count`` of 1 is the *expected* answer there
    and is what says "no shared endpoints", not a failure to find any.
    """
    payload = _run_example("q6", DEMO)
    assert payload["row_count"] == 1, payload
    row = payload["rows"][0]
    assert row["pv"] == _example("q6").parameters[DEMO]["pv"]
    assert row["device_count"] == 1, row


def test_als_example_q6_reverse_lookup_names_its_device(als_ctx: Any) -> None:
    """The same lookup resolves a real ALS address back to its device."""
    payload = _run_example("q6", ALS)
    assert payload["row_count"] == 1, payload
    row = payload["rows"][0]
    assert row["pv"] == _example("q6").parameters[ALS]["pv"]
    assert row["device_count"] >= 1, row
    assert row["devices"], row


# ---------------------------------------------------------------------------
# 6. Bounds: the timeout and the row cap
# ---------------------------------------------------------------------------


def test_a_slow_query_comes_back_as_a_timeout_envelope(als_ctx: Any) -> None:
    """The store terminates a runaway query and the tool reports it as such.

    A three-way cartesian product with a predicate: the predicate keeps the
    planner off the count store, so the query really does compare on the order
    of a billion pairs, and it streams rather than materialising, so the server
    hits the transaction timeout instead of the heap.
    """
    als_ctx.query_timeout_s = 1

    envelope = _error_envelope(
        "MATCH (a:Resource), (b:Resource), (c:Resource) "
        "WHERE a.uri < b.uri AND b.uri < c.uri "
        "RETURN count(*) AS n"
    )

    assert envelope["error_type"] == "timeout_error", envelope
    assert envelope["suggestions"], envelope


def test_the_row_cap_truncates_and_says_so(als_ctx: Any) -> None:
    """A query with more matches than the cap returns exactly the cap, flagged."""
    als_ctx.query_max_rows = 5

    payload = _read_cypher("MATCH (b:ChannelBinding) RETURN b.fullPv AS pv")

    assert payload["row_count"] == 5, payload["row_count"]
    assert payload["truncated"] is True, payload
    assert len(payload["rows"]) == 5
    assert any("query_max_rows" in line for line in payload["guidance"]), payload["guidance"]
