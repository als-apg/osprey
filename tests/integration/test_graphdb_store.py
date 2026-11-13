"""End-to-end coverage of the graph store against a real Neo4j + neosemantics.

This is the only place the seeding primitives in
:mod:`osprey.services.graph_store.seeder` meet an actual
n10s plugin.  Every other graphdb test mocks the driver, which can prove the
call shapes but not the two things that decide whether the feature works: that
``n10s.graphconfig.init`` with :data:`~...graph_seeder.N10S_GRAPH_CONFIG`
produces a graph the operator queries can traverse, and that the graph view the
demo build writes imports into exactly the corpus those queries
were verified against.  So the assertions here are the four counts verified on
that view (1017 binding owners / 4731 bindings / 674 write-only + 4052 read-only
/ 480 magnets rolled up through the class hierarchy), plus n10s's own
``terminationStatus`` and its 43801 triples.

Beyond the counts, the seeded demo's ``PARTOF`` and ``LOCATEDIN`` edges must be
exactly the pairs the facility records imply, and the two match strings the
agent is handed — the record-name rule and the capability manifest's class
synonym match — must return exactly the nodes the records say they should.
Both strings are imported, never copied, so a drifted rule fails here. A small
named document seeds a second store (the session's shared one, wiped first,
never the demo store the counts read) to show a channel's names land as a list
on its ``:ChannelBinding``.

Counts only, never wall clock: seeding time depends on the host's disk and on
whether the image was cold, and a timing assertion here would fail on a loaded
CI runner while proving nothing about the graph.

The container recipe — the pinned image, the n10s jar (from the pinned release
or from ``OSPREY_TEST_N10S_JAR``), the APOC jar copied out of the image, and the
procedure allowlist that replaces ``NEO4J_PLUGINS`` — lives in
:mod:`tests._graphdb_container`, which every graph lane shares.  Without Docker,
without the image, or without the jar this module skips and says which.  A
store that stops answering after it started is not a skip: the read fails naming
the store.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from osprey.facility.records import FacilityRecords
from osprey.facility.views.graph_iri import iri
from tests._builds import BuiltProject
from tests._graphdb_container import (
    GRAPHDB_TEST_DATABASE,
    GRAPHDB_TEST_PASSWORD,
    GRAPHDB_TEST_USERNAME,
    WatchedSession,
    WatchedStore,
    watched_graphdb_store,
    wiped_graphdb_store,
)
from tests.facility._graph_relations import (
    NAMED_KINDS,
    expected_located_in,
    expected_name_holders,
    expected_part_of,
)

logger = logging.getLogger(__name__)

# xdist_group("docker"): this module starts a real container, and the docker
# group is what keeps every such file on one xdist worker — two testcontainers
# sessions starting at once race their own reapers over the port publisher.
pytestmark = [pytest.mark.xdist_group("docker")]


# ---------------------------------------------------------------------------
# Verified counts for the demo build's graph view
# ---------------------------------------------------------------------------
# The binding and direction counts are the Turtle census pinned in
# tests/templates/test_demo_ttl.py; the magnet rollup was verified against
# a live n10s import of the same file. The triple count is the view's rdflib
# triple count, which equalled n10s' count before the view gained partOf,
# locatedIn and record names (40928 then); this test is what confirms it holds.

#: Triples n10s reports loading from the view.
EXPECTED_TRIPLES_LOADED = 43801
#: Distinct resources carrying at least one channel binding: the 1016 devices
#: and the top place ``SR``, which carries the tune, chromaticity, user
#: beam mode and operation-mode channels.
EXPECTED_DEVICES = 1017
#: ``(:ChannelBinding)`` nodes.
EXPECTED_BINDINGS = 4731
EXPECTED_WRITE_ONLY = 674
EXPECTED_READ_ONLY = 4052
#: Devices whose type rolls up to ``Magnet`` through ``rdfs:subClassOf`` —
#: Dipole + Quadrupole + Sextupole + HCorrector + VCorrector + Septum + Kicker
#: + Solenoid + BuckingCoil + InsertionDevice.
#: This is the count that proves the *hierarchy* imported, not just the nodes.
EXPECTED_MAGNETS = 480

#: Ontology root the magnet rollup walks up to.  A driver parameter here, where
#: the prototype's Browser-oriented file inlines it as a literal.
MAGNET_CLASS_URI = "https://narad.example.org/schema/shared_semantics/Magnet"

#: What the throwaway store is called in its skip and failure messages.
SEEDER_STORE_LABEL = "graphdb (neo4j + n10s) for the seeder lifecycle"
#: The label of the session store this module wipes for its named document.
NAMED_STORE_LABEL = "graphdb (neo4j + n10s) seeded with a named facility document"

#: A record name that is a substring of other names in the demo (``U10``,
#: ``U100``, ``TC02U10`` ...): the name lookup must return only its own holders.
SUBSTRING_NAME = "U1"
#: A class synonym the vocabulary carries.
SYNONYM = "steerer"

_OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


# ---------------------------------------------------------------------------
# Queries (translated from the prototype's operator query file)
# ---------------------------------------------------------------------------

#: Q1a, reduced to its total: a device is a resource with a binding.
DEVICE_COUNT_CYPHER = """
MATCH (d:Resource)-[:HASBINDING]->()
RETURN count(DISTINCT d) AS n
"""

BINDING_COUNT_CYPHER = """
MATCH (b:ChannelBinding)
RETURN count(b) AS n
"""

#: Q5.  ``readwrite`` is carried through rather than dropped: a binding that is
#: both would mean the direction split silently stopped partitioning, which a
#: pair of equality assertions on the other two columns would not catch.
BINDING_DIRECTION_CYPHER = """
MATCH (b:ChannelBinding)
WITH b,
     EXISTS { (b)-[:WRITESSIGNAL]->() } AS is_write,
     EXISTS { (b)-[:READSSIGNAL]->()  } AS is_read
RETURN
  sum(CASE WHEN is_write AND NOT is_read THEN 1 ELSE 0 END) AS write_only,
  sum(CASE WHEN is_read  AND NOT is_write THEN 1 ELSE 0 END) AS read_only,
  sum(CASE WHEN is_read  AND is_write     THEN 1 ELSE 0 END) AS readwrite,
  count(b)                                                  AS total
"""

#: Q1c, counted rather than listed.  ``SUBCLASSOF*0..`` includes the root class
#: itself, and ``:TYPE`` is the edge n10s writes for ``rdf:type`` under
#: ``handleRDFTypes='LABELS_AND_NODES'`` — both halves of the canonical graph
#: config have to have taken effect for this to return anything but 0.
MAGNET_ROLLUP_CYPHER = """
MATCH (cls:Class)-[:SUBCLASSOF*0..]->(:Class {uri: $magnet_uri})
MATCH (d:Resource)-[:TYPE]->(cls)
RETURN count(DISTINCT d) AS n
"""

#: Every edge of one relationship type, as ``(source uri, target uri)`` rows.
#: Formatted with the type, which is one of the two constants below.
EDGES_CYPHER = "MATCH (a)-[:{rel}]->(b) RETURN a.uri AS src, b.uri AS dst"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def seeder_store(graphdb_plugin_dir: Path) -> Iterator[WatchedStore]:
    """Start the graph store and yield it watched.

    Session-scoped: this module reads the store rather than wiping it, so one
    seeded container serves every test here. A read that gets no answer fails
    naming this store and the state its container was in.

    A container of its own rather than the session's ``graphdb_session_store``:
    the test below bootstraps a store nothing has touched, the store a first
    deploy meets, before the wiped one a re-seed meets. On the shared store
    both bootstraps would meet a wiped one.
    """
    with watched_graphdb_store(graphdb_plugin_dir, label=SEEDER_STORE_LABEL) as store:
        yield store


@pytest.fixture(scope="session")
def demo_ttl(built_demo: BuiltProject) -> str:
    """The graph view the demo build writes, as the seeder reads it."""
    view = built_demo.build_dir / "data" / "graph" / "facility.ttl"
    return view.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def demo_records(built_demo: BuiltProject) -> FacilityRecords:
    """The records the demo build rendered the view from."""
    return FacilityRecords(built_demo.facility)


@pytest.fixture(scope="module")
def named_document() -> dict:
    """The named-record document the graph view's unit tests render."""
    from tests.facility.test_graph_view import _doc

    return _doc()


@pytest.fixture(scope="module")
def named_store(graphdb_session_store: WatchedStore, named_document: dict) -> WatchedStore:
    """The session's store, wiped, bootstrapped and seeded with the named document.

    Never the module's own ``seeder_store``: that one holds the demo the counts
    read, and seeding this document means wiping first.
    """
    from osprey.facility.views.graph import graph_text
    from osprey.services.graph_store import seeder as graph_seeder

    store = wiped_graphdb_store(graphdb_session_store, label=NAMED_STORE_LABEL)
    ttl = graph_text(FacilityRecords(named_document))
    with (
        store.reading(),
        graph_seeder.open_session(
            store.uri,
            GRAPHDB_TEST_USERNAME,
            GRAPHDB_TEST_PASSWORD,
            database=GRAPHDB_TEST_DATABASE,
        ) as session,
    ):
        boot = graph_seeder.bootstrap(session)
        assert boot.ok, boot.message
        imported = graph_seeder.import_ttl(session, ttl)
        assert imported.termination_status == graph_seeder.TERMINATION_OK, imported.extra_info
    return store


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _scalar(session: WatchedSession, cypher: str, **params) -> int:
    """Run a single-column count query and return the number."""
    record = session.single(cypher, params)
    assert record is not None, f"query returned no row: {cypher}"
    return int(record["n"])


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_bootstrap_seed_and_force_reseed(
    seeder_store: WatchedStore, demo_ttl: str, demo_records: FacilityRecords
) -> None:
    """Drive the real seeder through a store's whole life against real n10s.

    One test rather than five, deliberately: every step's precondition is the
    previous step's outcome (a re-bootstrap can only be proven a no-op against
    a store this test bootstrapped, and ``--force`` can only be proven to
    recover a *seeded* store), and splitting them would either re-seed the
    corpus per test or leave the order dependency implicit between functions.

    Every step runs inside the store's watch, the session's close included, so
    a store that stops answering fails this test naming the store rather than
    as a count that came out wrong.
    """
    from osprey.services.graph_store import seeder as graph_seeder

    expected_sha = graph_seeder.ttl_sha256(demo_ttl)

    with (
        seeder_store.reading(),
        graph_seeder.open_session(
            seeder_store.uri,
            GRAPHDB_TEST_USERNAME,
            GRAPHDB_TEST_PASSWORD,
            database=GRAPHDB_TEST_DATABASE,
        ) as session,
    ):
        watched = WatchedSession(session, seeder_store)
        # --- 1. Bootstrap a fresh store -------------------------------------
        first = graph_seeder.bootstrap(session)
        assert first.status is graph_seeder.BootstrapStatus.INITIALIZED, first.message
        assert first.ok
        assert graph_seeder.resource_count(session) == 0, (
            "a bootstrapped-but-unseeded store must read as empty: n10s' own "
            "_GraphConfig/_NsPrefDef nodes carry no :Resource label"
        )
        assert graph_seeder.read_marker(session) is None

        # --- 2. Seed the graph view -----------------------------------------
        result = graph_seeder.import_ttl(session, demo_ttl)
        assert result.termination_status == graph_seeder.TERMINATION_OK, (
            f"n10s refused the graph view: {result.extra_info}"
        )
        assert result.ok
        assert result.triples_loaded == EXPECTED_TRIPLES_LOADED
        graph_seeder.write_marker(session, expected_sha)
        assert graph_seeder.read_marker(session) == expected_sha

        # --- 3. The verified counts -----------------------------------------
        _assert_verified_counts(watched)
        _assert_view_relations_and_names(watched, demo_records, demo_ttl)
        seeded_resources = graph_seeder.resource_count(session)

        # --- 4. Re-bootstrap is a no-op -------------------------------------
        # The gate under test is graphconfig.show: n10s hard-refuses to
        # re-initialize a configured store, so a bootstrap that did not check
        # would raise here rather than report.
        second = graph_seeder.bootstrap(session)
        assert second.status is graph_seeder.BootstrapStatus.ALREADY_CANONICAL, second.message
        assert second.differing_keys == ()
        assert second.ok
        # Re-bootstrapping must not disturb what is already in the store.
        assert graph_seeder.resource_count(session) == seeded_resources
        assert graph_seeder.read_marker(session) == expected_sha

        # --- 5. The --force path --------------------------------------------
        # wipe() removes the graph config and the marker along with the data,
        # which is exactly why --force re-bootstraps before importing again.
        graph_seeder.wipe(session)
        assert graph_seeder.resource_count(session) == 0
        assert graph_seeder.read_marker(session) is None

        forced = graph_seeder.bootstrap(session)
        assert forced.status is graph_seeder.BootstrapStatus.INITIALIZED, forced.message

        reimport = graph_seeder.import_ttl(session, demo_ttl)
        assert reimport.termination_status == graph_seeder.TERMINATION_OK, reimport.extra_info
        assert reimport.triples_loaded == EXPECTED_TRIPLES_LOADED
        graph_seeder.write_marker(session, expected_sha)

        assert graph_seeder.read_marker(session) == expected_sha
        _assert_verified_counts(watched)
        _assert_view_relations_and_names(watched, demo_records, demo_ttl)


def _assert_verified_counts(session: WatchedSession) -> None:
    """Assert the four verified counts for this corpus.

    Factored out because ``--force`` has to land on the *same* graph the first
    seed did — a re-import that produced a different shape (duplicated nodes, a
    missing hierarchy) is the failure mode --force exists to avoid.
    """
    assert _scalar(session, DEVICE_COUNT_CYPHER) == EXPECTED_DEVICES
    assert _scalar(session, BINDING_COUNT_CYPHER) == EXPECTED_BINDINGS

    directions = session.single(BINDING_DIRECTION_CYPHER)
    assert directions is not None
    assert directions["write_only"] == EXPECTED_WRITE_ONLY
    assert directions["read_only"] == EXPECTED_READ_ONLY
    assert directions["readwrite"] == 0
    assert directions["total"] == EXPECTED_BINDINGS

    magnets = _scalar(session, MAGNET_ROLLUP_CYPHER, magnet_uri=MAGNET_CLASS_URI)
    assert magnets == EXPECTED_MAGNETS, (
        "the transitive Magnet rollup is what proves the class hierarchy "
        "imported and is traversable, not just that the device nodes exist"
    )


@contextmanager
def graph_seeder_session(store: WatchedStore) -> Iterator[WatchedSession]:
    """A watched driver session on ``store``."""
    from osprey.services.graph_store import seeder as graph_seeder

    with graph_seeder.open_session(
        store.uri,
        GRAPHDB_TEST_USERNAME,
        GRAPHDB_TEST_PASSWORD,
        database=GRAPHDB_TEST_DATABASE,
    ) as session:
        yield WatchedSession(session, store)


def _edges(session: WatchedSession, rel: str) -> list[tuple[str, str]]:
    """Every ``rel`` edge in the store as ``(source uri, target uri)``."""
    return [(row["src"], row["dst"]) for row in session.records(EDGES_CYPHER.format(rel=rel))]


def _assert_view_relations_and_names(
    session: WatchedSession, records: FacilityRecords, ttl: str
) -> None:
    """Assert the place edges and both match strings against the records.

    The expected sets come from the records by the rule the view states
    (``tests/facility/_graph_relations.py``), never from the store or the
    writer, so a writer or an import that drifts from the rule fails here.
    """
    from osprey.mcp_server.channel_finder_graph.tools.capabilities import _SYNONYMS
    from osprey.services.graph_store.prompt_snapshot import RECORD_NAME_MATCH_RULE

    # --- partOf / locatedIn: counts and targets ------------------------------
    for rel, expected in (
        ("PARTOF", expected_part_of(records)),
        ("LOCATEDIN", expected_located_in(records)),
    ):
        edges = _edges(session, rel)
        assert expected, f"the demo records imply no {rel} edge: the check would be vacuous"
        assert len(edges) == len(expected), f"{rel}: {len(edges)} edges, {len(expected)} expected"
        assert set(edges) == expected, f"{rel} targets differ from the records"

    # --- Record-name rule: whole entry, never a substring --------------------
    holders = expected_name_holders(records, SUBSTRING_NAME)
    assert holders, f"no demo record is named {SUBSTRING_NAME!r}"
    wanted = SUBSTRING_NAME.lower()
    longer = {
        str(name)
        for kind in NAMED_KINDS
        for record in records.of_kind(kind)
        for name in record.get("names") or ()
        if wanted in str(name).lower() and str(name).lower() != wanted
    }
    assert longer, f"{SUBSTRING_NAME!r} is no substring of another name: the check is vacuous"
    rows = session.records(
        f"{RECORD_NAME_MATCH_RULE} RETURN n.uri AS uri", {"name": SUBSTRING_NAME}
    )
    found = [row["uri"] for row in rows]
    assert len(found) == len(set(found))
    assert set(found) == holders, f"the record-name rule returned {sorted(found)}"

    # --- The manifest's class synonym match ----------------------------------
    rows = session.records(
        f"MATCH (cls:Class) WHERE {_SYNONYMS['match']} RETURN cls.uri AS uri", {"synonym": SYNONYM}
    )
    found = [row["uri"] for row in rows]
    expected_classes = _classes_with_alias(records, ttl, SYNONYM)
    assert expected_classes, f"no seeded class carries the alias {SYNONYM!r}"
    assert len(found) == len(set(found))
    assert set(found) == expected_classes, f"the synonym match returned {sorted(found)}"


def _classes_with_alias(records: FacilityRecords, ttl: str, alias: str) -> set[str]:
    """The IRIs of the view's classes whose vocabulary aliases hold ``alias``.

    Aliases come from the vocabulary and the facility file's own classes, the
    two sources the view's writer reads; the view decides which classes are
    declared (those the devices carry and every class above them), so the set
    is restricted to the classes its Turtle declares.
    """
    from rdflib import Graph, URIRef

    from osprey.facility.validate import vocabulary

    sem = "https://narad.example.org/schema/shared_semantics/"
    wanted = alias.lower()
    aliases: dict[str, list[str]] = {
        row.get("iri") or f"{sem}{row['name']}": list(row.get("aliases") or [])
        for row in vocabulary()["classes"]
    }
    for row in records.document.get("classes") or []:
        aliases[f"{sem}{row['class']}"] = list(row.get("aliases") or [])
    declared = {
        str(subject)
        for subject in Graph()
        .parse(data=ttl, format="turtle")
        .subjects(URIRef(_RDF_TYPE), URIRef(_OWL_CLASS))
    }
    return {
        cls
        for cls, names in aliases.items()
        if cls in declared and any(str(name).lower() == wanted for name in names)
    }


def test_a_channels_names_are_a_list_on_its_binding(
    named_store: WatchedStore, named_document: dict
) -> None:
    """Each named record's names arrive as a list ``altLabel``, a channel's on its binding.

    n10s is seeded with ``handleMultival: ARRAY`` for ``skos:altLabel``: a
    single-name record still carries a list, which is what the record-name rule
    iterates. A channel's names sit on its ``:ChannelBinding`` node.
    """
    code = str(named_document["identity"]["code"])
    with graph_seeder_session(named_store) as session:
        channels = [c for c in named_document["channels"] if c.get("names")]
        assert channels, "the named document has no named channel"
        for channel in channels:
            row = session.single(
                "MATCH (b:ChannelBinding {uri: $uri}) RETURN b.altLabel AS names",
                {"uri": iri(code, "channel", str(channel["id"]))},
            )
            assert row is not None, f"channel {channel['id']} is no :ChannelBinding"
            assert isinstance(row["names"], list), row["names"]
            assert sorted(row["names"]) == sorted(set(channel["names"]))

        named = [
            (kind, record)
            for kind, key in (("place", "places"), ("device", "devices"))
            for record in named_document[key]
            if record.get("names")
        ]
        for kind, record in named:
            row = session.single(
                "MATCH (n:Resource {uri: $uri}) RETURN n.altLabel AS names",
                {"uri": iri(code, kind, str(record["id"]))},
            )
            assert row is not None
            assert isinstance(row["names"], list), row["names"]
            assert sorted(row["names"]) == sorted(set(record["names"]))
