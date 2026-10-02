"""One recipe for a throwaway Neo4j store carrying n10s + APOC.

Every graph lane's store comes from here. What they differ in is not how the
store is built but what they want back from it, and what an unstartable
container means for them — and those two differences are the entry points:
:func:`graphdb_store` yields a bolt URI and skips when the container will not
start, :func:`watched_graphdb_store` yields the same store as a
:class:`WatchedStore` and skips alike, and
:func:`graphdb_store_published_port` yields the published host port and fails.

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
not shared: the lanes that wipe it between corpora start their own
module-scoped container from this same recipe.

**A store that stops answering.** A started store can stall mid-module, most
often because the host is saturated, and come back. The driver then raises
``ServiceUnavailable`` from its socket layer. A read through
:meth:`WatchedStore.read` or :class:`WatchedSession` that gets no answer waits
for the store to answer a query again, or for its container to exit, for up to
:data:`STORE_RECOVERY_S`, then runs once more; the answer it then gets is a
real result. A store that does not answer again fails the read as
:class:`GraphStoreUnavailable`, naming the store and saying what state its
container was in, which is what tells a stalled store on a loaded host from one
that exited. A seeding block inside :meth:`WatchedStore.reading` never waits:
it fails at once.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import pytest
import requests
from neo4j.exceptions import ServiceUnavailable

from tests._container_support import (
    container_state,
    is_docker_available,
    is_image_present,
    start_or_fail,
    start_or_skip,
    stop_quietly,
    wait_until_ready,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

#: Database the throwaway store serves. The Community image serves exactly one,
#: and this is its name.
GRAPHDB_TEST_DATABASE = "neo4j"

#: Port the server listens on inside the container. The published host port a
#: lane reaches it on is ephemeral and read back from the started container.
NEO4J_BOLT_PORT = 7687


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


def _neo4j_container(plugin_dir: Path) -> Neo4jContainer:  # noqa: F821
    """Build an unstarted graph store carrying the plugins in *plugin_dir*.

    The one construction site in the tree, and therefore the one statement of
    which module :class:`Neo4jContainer` comes from. The environment mirrors
    the shipped compose template minus ``NEO4J_PLUGINS`` — see the module
    docstring for why that one is left out and what replaces it. The missing
    extra is not guarded here because :func:`resolve_plugin_dir` probes for it
    first on every lane, which is why its probe order is what it is.
    """
    from testcontainers.community.neo4j import Neo4jContainer

    container = Neo4jContainer(image=NEO4J_IMAGE, password=GRAPHDB_TEST_PASSWORD)
    container.with_volume_mapping(str(plugin_dir), "/plugins", "rw")
    # The allowlist is the half of NEO4J_PLUGINS that is not a download:
    # without it every n10s.* call fails with "not on the allowlist", and
    # n10s needs the unrestricted grant because it calls into APOC.
    container.with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*,n10s.*")
    container.with_env("NEO4J_dbms_security_procedures_allowlist", "apoc.*,n10s.*")
    return container


@contextmanager
def graphdb_store(
    plugin_dir: Path,
    *,
    label: str = "graphdb (neo4j + n10s)",
) -> Iterator[str]:
    """Start a throwaway graph store on *plugin_dir* and yield its bolt URI.

    Skips rather than fails when the container will not start: these lanes run
    on contributor machines where an absent engine is a fact about the host.
    Testcontainers publishes bolt on an ephemeral port under a generated name,
    so this cannot collide with a ``graphdb`` service already deployed on the
    host.

    Args:
        plugin_dir: Directory holding n10s + APOC, from :func:`resolve_plugin_dir`.
        label: Human-readable name for the store, used in skip messages.

    Yields:
        The bolt URI to reach the store on.

    Raises:
        Skipped: Via ``pytest.skip`` when the container will not start.
    """
    container = start_or_skip(lambda: _neo4j_container(plugin_dir), label=label)
    try:
        yield container.get_connection_url()
    finally:
        stop_quietly(container)


@contextmanager
def graphdb_store_published_port(
    plugin_dir: Path,
    *,
    label: str = "graphdb (neo4j + n10s)",
) -> Iterator[int]:
    """Start a throwaway graph store and yield its published **host** port.

    The fail-hard counterpart to :func:`graphdb_store`, for a lane that has
    already established the daemon is reachable and would otherwise report
    success having run nothing. It also retries the testcontainers
    port-publish race, which :func:`start_or_skip` turns into a skip on the
    first attempt — reading the published port is part of the retried
    operation there, not a step after it.

    Args:
        plugin_dir: Directory holding n10s + APOC, from :func:`resolve_plugin_dir`.
        label: Human-readable name for the store, used in the failure.

    Yields:
        The host port the store's bolt endpoint is published on.

    Raises:
        AssertionError: When the container would not start.
    """
    container, port = start_or_fail(lambda: _neo4j_container(plugin_dir), label, NEO4J_BOLT_PORT)
    logger.info(f"{label}: bolt published on host port {port}")
    try:
        yield port
    finally:
        stop_quietly(container)


# ---------------------------------------------------------------------------
# Waiting for a lost store
# ---------------------------------------------------------------------------

#: Connect and connection-acquisition deadline of one readiness probe, in
#: seconds. Short because the probe is repeated. The acquisition deadline is
#: the one that matters: a stalled store accepts the TCP connection and never
#: sends the four-byte bolt handshake, and the driver's default acquisition
#: deadline is 60 s, which is what a lost read spends by default.
STORE_PROBE_TIMEOUT_S = 5.0

#: Seconds between readiness probes.
STORE_PROBE_INTERVAL_S = 1.0

#: Seconds a lost store may take to answer again. The wait ends the moment the
#: store answers, so a healthy run never pays it. The store's log is quiet once
#: it has booted, and during a stall the daemon does not answer either, so
#: there is no progress signal to extend on: this is a plain window that ends
#: early only on an answer or an exited container. The window, the lost read
#: before it (up to the driver's 60 s acquisition deadline) and one liveness
#: read that a stalled daemon can hold for its client's default minute must
#: all fit together inside a 300 s per-test cap.
STORE_RECOVERY_S = 120.0


def _store_answers(uri: str) -> None:
    """Ask the store at *uri* to serve ``RETURN 1`` on the test database.

    This is the readiness question: a bolt handshake alone would not show that
    the database is online. A fresh driver is built per call, with short
    connect and acquisition deadlines, and closed whatever happens.

    ``max_transaction_retry_time=0.0`` is load-bearing. Without it the driver's
    ``execute_query`` retries internally for about half a minute against a
    port nothing answers on before it raises; with it, it raises at once. A
    probe that retried internally would turn the wait around it into a fixed
    sleep.

    Args:
        uri: The bolt URI the store is reached on.

    Raises:
        Exception: Whatever the driver raises when the store does not answer.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        uri,
        auth=(GRAPHDB_TEST_USERNAME, GRAPHDB_TEST_PASSWORD),
        connection_timeout=STORE_PROBE_TIMEOUT_S,
        connection_acquisition_timeout=STORE_PROBE_TIMEOUT_S,
        max_transaction_retry_time=0.0,
    )
    try:
        driver.execute_query("RETURN 1", database_=GRAPHDB_TEST_DATABASE)
    finally:
        driver.close()


def _wait_for_store(container: object, uri: str, label: str) -> None:
    """Return once the store at *uri* answers a query again.

    Waits on *container* as well, so a container that has exited ends the wait
    at once, quoting its exit code and the tail of its log.

    Args:
        container: The started container the store runs in.
        uri: The bolt URI the store is reached on.
        label: Human-readable name for the store, used in the failure.

    Raises:
        ContainerExitedError: At once, when the container has exited.
        AssertionError: When :data:`STORE_RECOVERY_S` passes with no answer.
            Both are reported by the caller, never swallowed.
    """
    wait_until_ready(
        partial(_store_answers, uri),
        f"{label} at {uri}",
        timeout=STORE_RECOVERY_S,
        interval=STORE_PROBE_INTERVAL_S,
        container=container,
        ceiling=STORE_RECOVERY_S,
    )


# ---------------------------------------------------------------------------
# A store that stops answering
# ---------------------------------------------------------------------------


class GraphStoreUnavailable(AssertionError):
    """A store read got no answer because the store stopped answering after it started.

    It subclasses :class:`AssertionError` so that a caller that does nothing
    special reports it as the real failure it is, never as a skip. It is its
    own type so that a reader can tell it apart from a parity assertion in the
    short summary.
    """


class WatchedStore:
    """Watches a started store's reads and turns a lost store into
    :class:`GraphStoreUnavailable`.

    It keeps a count of losses for the message and does not stop later reads
    from contacting the store.

    Args:
        uri: The bolt URI the store is reached on.
        label: Human-readable name for the store, used in the failure.
        inspect: Returns one line saying what state the store's container is
            in. Called once per read that gets no answer, and its line added to
            the failure. Without it the failure says nothing about the
            container.
        wait_for_store: Returns once the store answers again, and raises
            ``AssertionError`` when it does not. Called at most once per read,
            after the read lost the store. Without it a lost read fails at once.
    """

    def __init__(
        self,
        uri: str,
        *,
        label: str,
        inspect: Callable[[], str] | None = None,
        wait_for_store: Callable[[], None] | None = None,
    ) -> None:
        self.uri = uri
        self.label = label
        self.losses: list[str] = []
        self._inspect = inspect
        self._wait_for_store = wait_for_store

    @contextmanager
    def reading(self) -> Iterator[None]:
        """Run the block as a read of this store.

        It never waits. A block that writes, such as seeding, cannot be run a
        second time.

        Raises:
            GraphStoreUnavailable: When the block raises ``ServiceUnavailable``,
                or an error raised from one. Every other exception passes
                through untouched.
        """
        try:
            yield
        except Exception as exc:
            if _lost_store(exc) is None:
                raise
            raise self._unavailable(exc) from exc

    def read(self, fetch: Callable[[], T]) -> T:
        """Run *fetch* as a read of this store and return what it returns.

        A read that loses the store waits for the store to answer again, then
        runs *fetch* once more: one wait and one re-read per call, only for a
        loss. The answer the re-read gets is a real result.

        Args:
            fetch: The read, fetching its records in full before it returns.

        Returns:
            What *fetch* returned.

        Raises:
            GraphStoreUnavailable: When the store was lost and either did not
                answer again or was lost a second time by the re-read. Every
                other exception passes through untouched.
        """
        try:
            return fetch()
        except Exception as exc:
            if _lost_store(exc) is None:
                raise
            if self._wait_for_store is None:
                raise self._unavailable(exc) from exc
            lost = exc

        logger.warning(f"{self.label} at {self.uri} stopped answering; waiting for it")
        try:
            self._wait_for_store()
        except AssertionError as waited:
            raise self._unavailable(
                lost,
                after=(
                    f"It was waited on for up to {STORE_RECOVERY_S:g}s and did not "
                    f"answer again: {waited}"
                ),
            ) from lost

        self._record(lost)
        logger.info(f"{self.label} at {self.uri} answered again; reading once more")
        try:
            return fetch()
        except Exception as again:
            if _lost_store(again) is None:
                raise
            raise self._unavailable(
                again,
                after="It answered again, and the same read then got no answer a second time.",
            ) from again

    def _record(self, exc: BaseException) -> None:
        """Add the loss *exc* to :attr:`losses`, naming the test that was reading."""
        where = os.environ.get("PYTEST_CURRENT_TEST", "a read outside any test")
        self.losses.append(f"{where}: {type(exc).__name__}: {exc}")

    def _unavailable(self, exc: BaseException, *, after: str = "") -> GraphStoreUnavailable:
        """Record the loss *exc* and build the failure that reports it."""
        self._record(exc)
        message = (
            f"{self.label} at {self.uri} stopped answering during {self.losses[-1]}\n"
            "No answer came back to compare, so this is not a parity result. The "
            "container started before this read; look at the container and the host "
            "it runs on (docker ps -a, daemon load), not at the code under test."
        )
        if after:
            message += f"\n{after}"
        if self._inspect is not None:
            message += f"\nContainer state when the read failed: {_read_state(self._inspect)}"
        if len(self.losses) > 1:
            message += (
                f"\nThis store has stopped answering {len(self.losses)} times in this "
                f"module; the first was during {self.losses[0]}"
            )
        return GraphStoreUnavailable(message)


def _lost_store(exc: BaseException) -> ServiceUnavailable | None:
    """The ``ServiceUnavailable`` *exc* is or was raised from, if any.

    Walks the ``__cause__``/``__context__`` chain because the graph context
    maps the driver's ``ServiceUnavailable`` to its own unreachable-store error
    and raises that ``from`` the original, so the type alone decides nothing.
    A :class:`GraphStoreUnavailable` is a loss already recorded and reported,
    so it is none.
    """
    if isinstance(exc, GraphStoreUnavailable):
        return None
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ServiceUnavailable):
            return current
        current = current.__cause__ or current.__context__
    return None


def _read_state(inspect: Callable[[], str]) -> str:
    """*inspect*'s line, or what stopped it; never raises."""
    try:
        return inspect()
    except Exception as exc:
        # The lost read is the failure to report, not the inspector's.
        return f"could not be read ({type(exc).__name__}: {exc})"


class WatchedSession:
    """A driver session whose every read, record fetch included, runs inside
    :meth:`WatchedStore.read`.

    The records are fetched inside the guard because the driver pulls them
    lazily: a result handed back unread would fail outside it.

    Args:
        session: The driver session to read through.
        store: The store the session is on.
    """

    def __init__(self, session: Any, store: WatchedStore) -> None:
        self._session = session
        self._store = store

    def single(self, cypher: str, params: Mapping[str, Any] | None = None) -> Any:
        """Run *cypher* and return the driver's ``.single()`` of its result."""
        return self._store.read(lambda: self._session.run(cypher, dict(params or {})).single())

    def records(self, cypher: str, params: Mapping[str, Any] | None = None) -> list[Any]:
        """Run *cypher* and return every record of its result."""
        return self._store.read(lambda: list(self._session.run(cypher, dict(params or {}))))


@contextmanager
def watched_graphdb_store(
    plugin_dir: Path,
    *,
    label: str = "graphdb (neo4j + n10s)",
) -> Iterator[WatchedStore]:
    """Start a throwaway graph store and yield it watched, with its container's state.

    :func:`graphdb_store` for a lane that reads through :class:`WatchedStore`:
    the same start, skip and teardown, but the store it yields can say what
    state its container was in when a read got no answer, because this is the
    one place that holds both the container and the URI. The store it yields
    waits for its own container when a read loses it.

    Args:
        plugin_dir: Directory holding n10s + APOC, from :func:`resolve_plugin_dir`.
        label: Human-readable name for the store, used in skip messages and
            in the failure.

    Yields:
        The started store, watched.

    Raises:
        Skipped: Via ``pytest.skip`` when the container will not start.
    """
    container = start_or_skip(lambda: _neo4j_container(plugin_dir), label=label)
    uri = container.get_connection_url()
    try:
        yield WatchedStore(
            uri,
            label=label,
            inspect=partial(container_state, container),
            wait_for_store=partial(_wait_for_store, container, uri, label),
        )
    finally:
        stop_quietly(container)
