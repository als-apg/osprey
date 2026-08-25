"""One recipe for a throwaway Neo4j store carrying n10s + APOC.

Three lanes need the same container — the store integration lane
(``tests/integration/test_graphdb_store.py``), the corpus enrichment lane
(``tests/services/facility_knowledge/test_graph_enrichment_live.py``) and the
runnable-examples lane (``tests/mcp_server/channel_finder_graph/test_examples_live.py``).
They differ in what they do to the store, not in how it is built, so the build
lives here once.

**Why the plugins are mounted rather than downloaded by the server.** The
shipped compose template sets ``NEO4J_PLUGINS`` and the Neo4j entrypoint honours
it by fetching the n10s jar at boot, which needs egress from inside the
container. :func:`resolve_plugin_dir` instead resolves both jars on the host and
:func:`graphdb_store` bind-mounts them at ``/plugins``, setting the procedure
allowlist/unrestricted variables directly — they are the only other thing
``NEO4J_PLUGINS`` would have done. n10s comes from the pinned release (or from a
local jar named by ``OSPREY_TEST_N10S_JAR``); APOC is *bundled in the image* at
``/var/lib/neo4j/labs`` and is copied out of it. The result loads the same two
plugins a deployed graphdb does.

**Skip order.** :func:`resolve_plugin_dir` probes the testcontainers neo4j
module first, then the daemon, then the image, and only then downloads the jar —
a host without the extra installed says so instead of fetching a release it will
never mount.

**Scope.** The plugin directory is session-scoped (fixture
``graphdb_plugin_dir`` in ``tests/conftest.py``) because it is the expensive
half and its content does not depend on the lane. The *store* is deliberately
not shared: two of the three lanes wipe it between corpora, so they start their
own module-scoped container from this same recipe.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
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


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------

#: Same server pin as the shipped compose template.  Load-bearing twice over:
#: the 5.26 LTS line is the newest one the neosemantics plugin manifest
#: covers, and the n10s release below is the build made for exactly this
#: server line.
NEO4J_IMAGE = "neo4j:5.26-community"

#: neosemantics release matching :data:`NEO4J_IMAGE`.  n10s versions track the
#: server they are built against, so this moves with the image pin or not at
#: all — a mismatched jar loads and then fails every procedure call.
N10S_VERSION = "5.26.0"

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


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------


def _fetch_n10s_jar(dest_dir: Path) -> None:
    """Put the pinned neosemantics jar in *dest_dir*, or skip.

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
    into ``/plugins`` when ``NEO4J_PLUGINS`` asks for it.  Since this recipe
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


def resolve_plugin_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a directory holding n10s + APOC, or skip with the reason.

    This is every graph lane's skip gate, because the plugins are the first
    thing a store depends on: a missing testcontainers extra, an unreachable
    daemon, a missing image or an unreachable release is reported once, with its
    reason, rather than as a bare skip on each test.

    The probes run cheapest-and-most-decisive first — the testcontainers import
    before the daemon, both before the jar download — so a host that could never
    start the container does not spend a release download finding that out.
    """
    try:
        import testcontainers.community.neo4j  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the installed extras
        pytest.skip("testcontainers' neo4j module is not installed")

    if not is_docker_available():
        pytest.skip(
            "docker daemon is not reachable — this can only be proved against a "
            "real Neo4j + neosemantics store"
        )

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


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


@contextmanager
def graphdb_store(plugin_dir: Path) -> Iterator[str]:
    """Start a throwaway graph store on *plugin_dir* and yield its bolt URI.

    The environment mirrors the shipped compose template minus
    ``NEO4J_PLUGINS`` — see the module docstring for why that one is left out
    and what replaces it.  Testcontainers publishes bolt on an ephemeral port
    under a generated name, so this cannot collide with a ``graphdb`` service
    already deployed on the host.
    """
    from testcontainers.community.neo4j import Neo4jContainer

    def _build() -> Neo4jContainer:
        container = Neo4jContainer(image=NEO4J_IMAGE, password=GRAPHDB_TEST_PASSWORD)
        container.with_volume_mapping(str(plugin_dir), "/plugins", "rw")
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
