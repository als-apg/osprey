"""Container and config helpers for the qmd/ARIEL integration lane.

The leading underscore keeps this out of pytest collection, matching
``tests/_container_support.py``: ``python_files`` matches ``test_*.py`` only, and
a collected helper module would report its imports as test failures.

Fixtures live in ``conftest.py``; everything importable by a test module lives
here, so nothing has to import a conftest.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest
import requests

from tests._container_support import (
    is_docker_available,
    is_image_present,
    start_or_fail,
    stop_quietly,
)

logger = logging.getLogger(__name__)

#: Fallback sidecar image, used when the deployment's override is unset. This is
#: the tag a local build produces, so an interactive run needs no environment.
DEFAULT_QMD_SIDECAR_IMAGE = "osprey-qmd:local-validate"

#: The deployment-wide override for the sidecar image, established by the image
#: task and consumed by the compose ``image:`` line.
QMD_IMAGE_ENV = "OSPREY_QMD_IMAGE"


def qmd_sidecar_image() -> str:
    """The sidecar image this lane tests.

    Honours ``OSPREY_QMD_IMAGE`` so the CI lane tests **the image it just
    built** rather than whatever ``osprey-qmd:local-validate`` happens to be on
    the runner. Hardcoding the local tag is the quiet failure: on a clean runner
    the lane errors, and on a runner carrying a stale tag it passes green
    against the wrong image, reporting that a build it never touched is good.

    Returns:
        The image reference, from the environment or
        :data:`DEFAULT_QMD_SIDECAR_IMAGE`.
    """
    return os.environ.get(QMD_IMAGE_ENV) or DEFAULT_QMD_SIDECAR_IMAGE


def build_sidecar_container(corpus_root: Path, collection: str):
    """Construct — but do not start — the sidecar container.

    Split out from :func:`start_qmd_sidecar` so the image selection is testable
    without a daemon. That seam is the point: a test that only exercised
    :func:`qmd_sidecar_image` would not notice a hardcoded tag reappearing at
    the construction site, which is exactly how this regressed once.

    Args:
        corpus_root: Host directory bind-mounted read-only as the corpus.
        collection: qmd collection name to index it under.

    Returns:
        A configured, unstarted ``DockerContainer``.
    """
    from testcontainers.core.container import DockerContainer

    container = DockerContainer(qmd_sidecar_image())
    container.with_env("OSPREY_QMD_COLLECTIONS", f"{collection}={SIDECAR_CORPUS_TARGET}")
    # Poll the freshness marker aggressively: the 5 s default is tuned for a
    # production sweep, and every re-index assertion here waits on it.
    container.with_env("OSPREY_QMD_MARKER_POLL_INTERVAL", "1")
    container.with_env("OSPREY_QMD_UPDATE_INTERVAL", "3600")
    container.with_exposed_ports(SIDECAR_CONTAINER_PORT)
    container.with_volume_mapping(str(corpus_root), SIDECAR_CORPUS_TARGET, "ro")
    return container


#: Container-side mount point for the ARIEL markdown mirror.
SIDECAR_CORPUS_TARGET = "/corpus/ariel"

#: The port the sidecar's forwarder owns inside the container. Not qmd's own
#: 8181, which the daemon binds on IPv6 loopback only.
SIDECAR_CONTAINER_PORT = 8180

#: How long to wait for the sidecar's ``/health``. The entrypoint runs the whole
#: startup index pass *before* it opens the port, so this covers indexing too.
SIDECAR_HEALTH_TIMEOUT = 300.0

#: How long to wait for a corpus change to reach the index after a marker touch.
#: The container polls markers every ``OSPREY_QMD_MARKER_POLL_INTERVAL`` seconds
#: and then re-indexes; on macOS the bind-mount stat() also has to propagate into
#: the Docker VM.
SIDECAR_REINDEX_TIMEOUT = 180.0


def qmd_ariel_config_dict(database_url: str, mirror_root: Path | str) -> dict[str, Any]:
    """Build the raw ``ariel`` config section this lane runs against.

    ``mirror_path`` is written under ``settings:`` — the spelling the shipped
    templates use — so the loader's settings merge is exercised rather than
    bypassed.

    Args:
        database_url: DSN for this module's database.
        mirror_root: Absolute path of the markdown mirror.

    Returns:
        A dict shaped like the ``ariel`` section of a project ``config.yml``.
    """
    return {
        "database": {"uri": database_url},
        "search_modules": {
            "keyword": {"enabled": True},
            "semantic": {"enabled": False},
            # rerank is off for the lane, not because it is broken but because it
            # costs ~4x per query (measured: 0.85 s -> 4.19 s on a two-document
            # corpus) and nothing asserted here is about rank quality. Passed
            # explicitly so the assertions document the mode they ran in.
            "hybrid": {"enabled": True, "settings": {"rerank": False, "candidate_limit": 40}},
        },
        "enhancement_modules": {
            "text_embedding": {"enabled": False},
            "semantic_processor": {"enabled": False},
            "qmd_export": {"enabled": True, "settings": {"mirror_path": str(mirror_root)}},
        },
    }


async def open_migrated_repository(config):
    """Open a pool, apply migrations, and return ``(pool, repository)``.

    Args:
        config: An :class:`ARIELConfig`.

    Returns:
        Tuple of the open pool and a repository bound to it. The caller owns the
        pool and must close it.
    """
    from osprey.services.ariel_search.database import (
        ARIELRepository,
        create_connection_pool,
        run_migrations,
    )

    pool = await create_connection_pool(config.database)
    await run_migrations(pool, config)
    return pool, ARIELRepository(pool, config)


def require_sidecar_image() -> None:
    """Refuse to start a sidecar whose image was never built.

    Nothing in the test suite builds the image, so its absence means one of two
    different things and they need opposite handling:

    * ``OSPREY_QMD_IMAGE`` set -- the caller is the lane that builds the image
      and names the tag it built. A missing image there is a build failure, and
      skipping would report success having proved nothing. Fail loudly.
    * ``OSPREY_QMD_IMAGE`` unset -- a general lane or a developer machine that
      simply never built it. These suites are collected there because the unit
      lane runs ``tests/`` without deselecting the container markers, so a skip
      is the correct outcome; failing would red a lane over an optional image.

    Without the split, whichever behaviour is chosen is wrong somewhere: a hard
    failure reds every unit lane on every PR, and an unconditional skip hollows
    out the one lane that exists to prove the sidecar works.

    Raises:
        AssertionError: If the image named by ``OSPREY_QMD_IMAGE`` is absent.

    Raises:
        Skipped: Via ``pytest.skip`` when no image was requested and none is
            built locally.
    """
    image = qmd_sidecar_image()
    if is_image_present(image):
        return
    if os.environ.get(QMD_IMAGE_ENV):
        raise AssertionError(
            f"{QMD_IMAGE_ENV}={image!r} names an image that is not present on this "
            "daemon. The lane that sets this variable builds the image first, so "
            "this is a failed or mis-tagged build, not a missing dependency."
        )
    pytest.skip(
        f"qmd sidecar image {image!r} is not built on this machine; "
        "build it before running the sidecar suites, or set "
        f"{QMD_IMAGE_ENV} to an image that is present"
    )


def start_qmd_sidecar(request: pytest.FixtureRequest, corpus_root: Path, collection: str):
    """Start the prebuilt sidecar over ``corpus_root`` and wait for ``/health``.

    The corpus must already hold at least one document. The entrypoint is
    fail-closed by design — it refuses to serve an empty index and exits — so
    starting the container before the mirror is written produces a container that
    dies rather than one that answers nothing.

    Args:
        request: Fixture request, used to register container teardown.
        corpus_root: Host directory bind-mounted read-only as the corpus.
        collection: qmd collection name to index it under.

    Returns:
        Tuple of the started container and the host port the forwarder answers on.
    """
    if not is_docker_available():
        pytest.skip("docker daemon is not reachable")
    try:
        import testcontainers.core.container  # noqa: F401
    except ImportError:
        pytest.skip("testcontainers not installed")

    require_sidecar_image()

    if not list(corpus_root.rglob("*.md")):
        raise AssertionError(
            f"refusing to start the sidecar over an empty corpus at {corpus_root}: the "
            "entrypoint exits rather than serving an empty index, which would surface "
            "as an unexplained container death instead of this message"
        )

    container, port = start_or_fail(
        lambda: build_sidecar_container(corpus_root, collection),
        f"qmd sidecar ({qmd_sidecar_image()})",
        SIDECAR_CONTAINER_PORT,
    )
    request.addfinalizer(lambda: stop_quietly(container))

    wait_for_health(container, port)
    return container, port


def wait_for_health(container, port: int) -> None:
    """Block until the sidecar answers ``/health``, or fail with its own logs.

    A timeout is a failure, not a skip: the image is present and the daemon was
    asked to start, so silence is a defect in the sidecar or the corpus rather
    than an absent dependency, and skipping it would be exactly the vacuous green
    this lane exists to rule out.
    """
    deadline = time.monotonic() + SIDECAR_HEALTH_TIMEOUT
    last = "no attempt made"
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if response.ok and '"ok"' in response.text:
                return
            last = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1.0)

    raise AssertionError(
        f"qmd sidecar did not answer /health within {SIDECAR_HEALTH_TIMEOUT:.0f}s "
        f"(last: {last})\n--- container logs ---\n{_tail_logs(container)}"
    )


def wait_for_indexed(client, collection: str, query: str, predicate, *, what: str):
    """Poll ``client.query`` until ``predicate`` accepts the hit list.

    The sidecar re-indexes asynchronously — a marker touch is observed by a poll
    loop inside the container — so every assertion about *new* corpus content has
    to wait rather than read once.

    Args:
        client: A ``QMDClient`` pointed at the sidecar.
        collection: Collection to query.
        query: Query text.
        predicate: Callable taking the hit list and returning a bool.
        what: Human description used in the failure message.

    Returns:
        The first hit list the predicate accepted.
    """
    deadline = time.monotonic() + SIDECAR_REINDEX_TIMEOUT
    hits: list[Any] = []
    while time.monotonic() < deadline:
        hits = client.query(collection, query, limit=50, rerank=False, candidate_limit=40)
        if predicate(hits):
            return hits
        time.sleep(2.0)

    raise AssertionError(
        f"the sidecar never reported {what} within {SIDECAR_REINDEX_TIMEOUT:.0f}s; "
        f"last hit files: {[h.file for h in hits]}"
    )


def _tail_logs(container) -> str:
    """Best-effort tail of a container's combined output, for failure messages."""
    try:
        stdout, stderr = container.get_logs()
        text = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")
        return text[-4000:]
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return f"(could not read container logs: {exc})"
