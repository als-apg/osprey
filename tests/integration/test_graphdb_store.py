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

**Why the container starts without ``NEO4J_PLUGINS``.** The shipped compose
template sets it, and the Neo4j entrypoint honours it by fetching the n10s jar
from GitHub *on every container start* — which would put a release download on
the critical path of this test each time it runs, and make it fail on a host
with no egress to github.com even when the image is already local.  This
fixture instead resolves both jars once per session and bind-mounts them at
``/plugins``, setting the procedure allowlist/unrestricted variables directly —
they are the only other thing ``NEO4J_PLUGINS`` would have done.  n10s comes
from a pinned release URL (cacheable, and overridable offline via
``OSPREY_TEST_N10S_JAR``); APOC is *bundled in the image* at
``/var/lib/neo4j/labs`` and is copied out of it, so it costs no network at all.
The resulting container loads the same two plugins a deployed graphdb does.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
from importlib.resources import files
from pathlib import Path

import pytest
import requests

from tests._container_support import (
    is_docker_available,
    is_image_present,
    start_or_skip,
    stop_quietly,
)

logger = logging.getLogger(__name__)

# xdist_group("docker"): this module starts a real container, and the docker
# group is what keeps every such file on one xdist worker — two testcontainers
# sessions starting at once race their own reapers over the port publisher.
pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("docker")]


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------

#: Same server pin as the shipped compose template.  Load-bearing twice over:
#: neosemantics publishes no build for 5.23+, and the n10s release below is the
#: build made for exactly this server line.
NEO4J_IMAGE = "neo4j:5.20-community"

#: neosemantics release matching :data:`NEO4J_IMAGE`.  n10s versions track the
#: server they are built against, so this moves with the image pin or not at
#: all — a mismatched jar loads and then fails every procedure call.
N10S_VERSION = "5.20.0"

N10S_JAR_URL = (
    f"https://github.com/neo4j-labs/neosemantics/releases/download/"
    f"{N10S_VERSION}/neosemantics-{N10S_VERSION}.jar"
)

#: Escape hatch for hosts with no egress to github.com: point it at an already
#: downloaded neosemantics jar and the fixture mounts that instead of fetching.
N10S_JAR_ENV = "OSPREY_TEST_N10S_JAR"

#: Where the image keeps the APOC jar it would otherwise copy into /plugins.
IMAGE_LABS_DIR = "/var/lib/neo4j/labs"

#: Password for the throwaway store.  Satisfies the same rule the deploy-time
#: validator enforces on ``GRAPHDB_PASSWORD`` (>= 8 chars, no ``/``), so the
#: composite ``NEO4J_AUTH`` this becomes is shaped like a real one.
GRAPHDB_TEST_PASSWORD = "ospreytest1234"

GRAPHDB_TEST_USERNAME = "neo4j"

# --- Verified counts for the shipped als_gtb.ttl ---------------------------
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


def _fetch_n10s_jar(dest_dir: Path) -> None:
    """Put the pinned neosemantics jar in *dest_dir*, or skip the session.

    A jar named by :data:`N10S_JAR_ENV` is used as-is; otherwise the pinned
    release is downloaded.  A download failure skips rather than fails: it says
    something about the host's network, not about the graph store.
    """
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
    """Copy the image's own APOC jar into *dest_dir*.

    APOC ships inside ``neo4j:*-community`` and the entrypoint only *moves* it
    into ``/plugins`` when ``NEO4J_PLUGINS`` asks for it.  Since this fixture
    does not set that variable, it does the move itself — from a container that
    is created and never started, so nothing but a filesystem read happens.
    """
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


@pytest.fixture(scope="session")
def graphdb_plugin_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding n10s + APOC, resolved once for the whole session.

    Session-scoped because the expensive half is a release download and the
    container that mounts it is session-scoped too.
    """
    if not is_docker_available():
        pytest.skip("docker daemon is not reachable")

    if not is_image_present(NEO4J_IMAGE):
        # Pull explicitly: the APOC copy below reads the image directly and,
        # unlike ``containers.run``, ``containers.create`` does not pull for us.
        # Testcontainers would have pulled the same image moments later anyway.
        import docker

        logger.info(f"pulling {NEO4J_IMAGE} (not present locally)")
        try:
            docker.from_env().images.pull(NEO4J_IMAGE)
        except Exception as exc:
            pytest.skip(f"could not pull {NEO4J_IMAGE}: {exc}")

    plugin_dir = tmp_path_factory.mktemp("graphdb-plugins")
    _fetch_n10s_jar(plugin_dir)
    _copy_bundled_apoc(plugin_dir)

    # The server runs as a non-root user and only ever reads these.
    for jar in plugin_dir.glob("*.jar"):
        jar.chmod(0o644)
    plugin_dir.chmod(0o755)
    return plugin_dir


@pytest.fixture(scope="session")
def graphdb_uri(graphdb_plugin_dir: Path):
    """Start the graph store and yield its bolt URI.

    The environment mirrors the shipped compose template minus
    ``NEO4J_PLUGINS`` — see the module docstring for why that one is left out
    and what replaces it.
    """
    try:
        from testcontainers.community.neo4j import Neo4jContainer
    except ImportError:  # pragma: no cover - depends on the installed extras
        pytest.skip("testcontainers' neo4j module is not installed")

    def _build() -> Neo4jContainer:
        container = Neo4jContainer(image=NEO4J_IMAGE, password=GRAPHDB_TEST_PASSWORD)
        container.with_volume_mapping(str(graphdb_plugin_dir), "/plugins", "rw")
        # The allowlist is the half of NEO4J_PLUGINS that is not a download:
        # without it every n10s.* call fails with "not on the allowlist", and
        # n10s needs the unrestricted grant because it calls into APOC.
        container.with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*,n10s.*")
        container.with_env("NEO4J_dbms_security_procedures_allowlist", "apoc.*,n10s.*")
        return container

    container = start_or_skip(_build, label="graphdb (neo4j + n10s)")
    try:
        yield container.get_connection_url()
    finally:
        stop_quietly(container)


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
