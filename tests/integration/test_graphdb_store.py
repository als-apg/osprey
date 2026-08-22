"""End-to-end coverage of the graph store against a real Neo4j + neosemantics.

This is the only place the seeding primitives in
:mod:`osprey.services.facility_knowledge.seeder.graph_seeder` meet an actual
n10s plugin.  Every other graphdb test mocks the driver, which can prove the
call shapes but not the two things that decide whether the feature works: that
``n10s.graphconfig.init`` with :data:`~...graph_seeder.N10S_GRAPH_CONFIG`
produces a graph the operator queries can traverse, and that the shipped
``als_gtb.ttl`` imports into exactly the corpus those queries were verified
against.  So the assertions here are the four counts verified on the
als-ontology prototype snapshot (76 devices / 964 bindings / 260 write-only +
704 read-only / 47 magnets rolled up through the class hierarchy), plus n10s's
own ``terminationStatus`` and its 8114 triples.

Counts only, never wall clock: seeding time depends on the host's disk and on
whether the image was cold, and a timing assertion here would fail on a loaded
CI runner while proving nothing about the graph.

The container recipe — the pinned image, the n10s jar (from the pinned release
or from ``OSPREY_TEST_N10S_JAR``), the APOC jar copied out of the image, and the
procedure allowlist that replaces ``NEO4J_PLUGINS`` — lives in
:mod:`tests._graphdb_container`, which every graph lane shares.  Without Docker,
without the image, or without the jar this module skips and says which.
"""

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path

import pytest

from tests._graphdb_container import (
    GRAPHDB_TEST_PASSWORD,
    GRAPHDB_TEST_USERNAME,
    graphdb_store,
)

logger = logging.getLogger(__name__)

# xdist_group("docker"): this module starts a real container, and the docker
# group is what keeps every such file on one xdist worker — two testcontainers
# sessions starting at once race their own reapers over the port publisher.
pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("docker")]


# ---------------------------------------------------------------------------
# Verified counts for the shipped als_gtb.ttl
# ---------------------------------------------------------------------------
# Source: ~/code/als-ontology/neo4j/queries/als-operator-queries.cypher, whose
# header records these as verified against this snapshot on 2026-05-28.

#: Triples n10s reports loading from the shipped TTL.
EXPECTED_TRIPLES_LOADED = 8114
#: Distinct devices, i.e. resources carrying at least one channel binding.
EXPECTED_DEVICES = 76
#: ``(:ChannelBinding)`` nodes.
EXPECTED_BINDINGS = 964
EXPECTED_WRITE_ONLY = 260
EXPECTED_READ_ONLY = 704
#: Devices whose type rolls up to ``Magnet`` through ``rdfs:subClassOf`` —
#: Quadrupole + Dipole + HCorrector + VCorrector + Solenoid + BuckingCoil.
#: This is the count that proves the *hierarchy* imported, not just the nodes.
EXPECTED_MAGNETS = 47

#: Ontology root the magnet rollup walks up to.  A driver parameter here, where
#: the prototype's Browser-oriented file inlines it as a literal.
MAGNET_CLASS_URI = "https://narad.example.org/schema/shared_semantics/Magnet"


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def graphdb_uri(graphdb_plugin_dir: Path):
    """Start the graph store and yield its bolt URI.

    Session-scoped: this module reads the store rather than wiping it, so one
    seeded container serves every test here.
    """
    with graphdb_store(graphdb_plugin_dir) as uri:
        yield uri


@pytest.fixture(scope="session")
def als_gtb_ttl() -> str:
    """The shipped example TTL, read the way installed code reads it.

    ``importlib.resources``, not a path into ``src/``: the TTL ships inside the
    package, and a filesystem path would pass here while proving nothing about
    the installed layout the seeding verb actually goes through.
    """
    resource = (
        files("osprey.templates").joinpath("services").joinpath("graphdb").joinpath("als_gtb.ttl")
    )
    return resource.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _scalar(session, cypher: str, **params) -> int:
    """Run a single-column count query and return the number."""
    record = session.run(cypher, **params).single()
    assert record is not None, f"query returned no row: {cypher}"
    return int(record["n"])


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_bootstrap_seed_and_force_reseed(graphdb_uri: str, als_gtb_ttl: str) -> None:
    """Drive the real seeder through a store's whole life against real n10s.

    One test rather than five, deliberately: every step's precondition is the
    previous step's outcome (a re-bootstrap can only be proven a no-op against
    a store this test bootstrapped, and ``--force`` can only be proven to
    recover a *seeded* store), and splitting them would either re-seed the
    corpus per test or leave the order dependency implicit between functions.
    """
    from osprey.services.facility_knowledge.seeder import graph_seeder

    expected_sha = graph_seeder.ttl_sha256(als_gtb_ttl)

    with graph_seeder.open_session(
        graphdb_uri, GRAPHDB_TEST_USERNAME, GRAPHDB_TEST_PASSWORD
    ) as session:
        # --- 1. Bootstrap a fresh store -------------------------------------
        first = graph_seeder.bootstrap(session)
        assert first.status is graph_seeder.BootstrapStatus.INITIALIZED, first.message
        assert first.ok
        assert graph_seeder.resource_count(session) == 0, (
            "a bootstrapped-but-unseeded store must read as empty: n10s' own "
            "_GraphConfig/_NsPrefDef nodes carry no :Resource label"
        )
        assert graph_seeder.read_marker(session) is None

        # --- 2. Seed the shipped TTL ----------------------------------------
        result = graph_seeder.import_ttl(session, als_gtb_ttl)
        assert result.termination_status == graph_seeder.TERMINATION_OK, (
            f"n10s refused the shipped TTL: {result.extra_info}"
        )
        assert result.ok
        assert result.triples_loaded == EXPECTED_TRIPLES_LOADED
        graph_seeder.write_marker(session, expected_sha)
        assert graph_seeder.read_marker(session) == expected_sha

        # --- 3. The verified counts -----------------------------------------
        _assert_verified_counts(session)
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

        reimport = graph_seeder.import_ttl(session, als_gtb_ttl)
        assert reimport.termination_status == graph_seeder.TERMINATION_OK, reimport.extra_info
        assert reimport.triples_loaded == EXPECTED_TRIPLES_LOADED
        graph_seeder.write_marker(session, expected_sha)

        assert graph_seeder.read_marker(session) == expected_sha
        _assert_verified_counts(session)


def _assert_verified_counts(session) -> None:
    """Assert the four counts the prototype verified for this corpus.

    Factored out because ``--force`` has to land on the *same* graph the first
    seed did — a re-import that produced a different shape (duplicated nodes, a
    missing hierarchy) is the failure mode --force exists to avoid.
    """
    assert _scalar(session, DEVICE_COUNT_CYPHER) == EXPECTED_DEVICES
    assert _scalar(session, BINDING_COUNT_CYPHER) == EXPECTED_BINDINGS

    directions = session.run(BINDING_DIRECTION_CYPHER).single()
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
