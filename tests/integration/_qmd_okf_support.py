"""Container helpers for the OKF half of the qmd sidecar lane.

The leading underscore keeps this out of pytest collection, matching
``tests/_container_support.py``: ``python_files`` matches ``test_*.py`` only, and
a collected helper module would report its imports as test failures.

Deliberately self-contained rather than importing the ARIEL lane's sibling
helper. The two modules were written concurrently and each is the entry point
for a different half of the feature; coupling them would mean an edit on one
side breaking the other's lane for reasons that have nothing to do with what it
tests. What is shared is the genuinely shared thing —
``tests._container_support`` — and the image tag, which comes from the
environment so the CI lane and a local run name it the same way.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest
import requests

from tests._container_support import is_docker_available, start_or_fail, stop_quietly
from tests.integration._qmd_ariel_support import require_sidecar_image

logger = logging.getLogger(__name__)

#: The sidecar image, honouring ``OSPREY_QMD_IMAGE`` — the same variable the
#: rendered compose fragment overrides ``image:`` with, and the one the CI lane
#: sets to the tag it just built. The default is the locally verified tag, so a
#: developer run needs no environment at all.
#:
#: Whether a missing image is a skip or a loud failure depends on which of those
#: two callers is asking; :func:`require_sidecar_image` owns that rule and states
#: why it has to be conditional.
QMD_SIDECAR_IMAGE = os.environ.get("OSPREY_QMD_IMAGE") or "osprey-qmd:local-validate"

#: Container-side mount point for the OKF bundle, and the collection it is
#: indexed under. The collection name is the contract every OKF query filters
#: on (``okf.bundle.OKF_COLLECTION``); a test asserts the two agree rather than
#: restating the literal.
SIDECAR_BUNDLE_TARGET = "/corpus/okf"

#: How long to wait for ``/health``. The entrypoint runs the whole startup index
#: pass *before* it opens the port, so this covers indexing the fixture bundle.
SIDECAR_HEALTH_TIMEOUT = 300.0

#: How long to wait for a drafted document to reach the index after the touch
#: marker advances. The container polls markers on its own interval and then
#: re-indexes; on macOS the bind-mount stat() also has to propagate into the
#: Docker VM.
SIDECAR_REINDEX_TIMEOUT = 180.0

#: Marker poll interval the lane runs the sidecar with. Far below the shipped
#: default, because every freshness assertion here waits on it — the *mechanism*
#: is what is under test, not the production tuning.
MARKER_POLL_INTERVAL = "1"


def require_docker() -> None:
    """Refuse to run without a daemon, and say so in the skip reason.

    A skip here is legitimate — a contributor without Docker should not see a
    red suite — but it is the one skip this lane tolerates, so it is spelled
    once and worded to be unmistakable in ``-rs`` output.
    """
    if not is_docker_available():
        pytest.skip("docker daemon is not reachable — this lane needs a live container engine")


def start_okf_sidecar(request: pytest.FixtureRequest, bundle_root: Path):
    """Start the prebuilt sidecar over an OKF bundle and wait for ``/health``.

    The bundle must already hold at least one document. The entrypoint is
    fail-closed by design — it refuses to serve an empty index and exits — so
    starting over an empty directory produces a container that dies rather than
    one that answers nothing.

    The bundle is mounted **read-write**, unlike the sidecar's own compose
    mount: the draft-to-findable test writes into it through the same directory
    the container indexes, which is exactly the shared-corpus shape the
    deployment renders.

    Args:
        request: Fixture request, used to register container teardown.
        bundle_root: Host directory bind-mounted as the corpus.

    Returns:
        Tuple of the started container and the host port the forwarder answers
        on.
    """
    require_docker()
    try:
        from testcontainers.core.container import DockerContainer
    except ImportError:
        pytest.skip("testcontainers not installed")

    require_sidecar_image()

    if not list(bundle_root.rglob("*.md")):
        raise AssertionError(
            f"refusing to start the sidecar over an empty bundle at {bundle_root}: the "
            "entrypoint exits rather than serving an empty index, which would surface as "
            "an unexplained container death instead of this message"
        )

    from osprey.services.facility_knowledge.okf.bundle import OKF_COLLECTION

    def build():
        container = DockerContainer(QMD_SIDECAR_IMAGE)
        container.with_env("OSPREY_QMD_COLLECTIONS", f"{OKF_COLLECTION}={SIDECAR_BUNDLE_TARGET}")
        container.with_env("OSPREY_QMD_MARKER_POLL_INTERVAL", MARKER_POLL_INTERVAL)
        # Effectively disable the fallback sweep. The freshness test must prove
        # the MARKER shortened the lag; a sweep that happened to fire would make
        # it pass for the wrong reason.
        container.with_env("OSPREY_QMD_UPDATE_INTERVAL", "3600")
        container.with_exposed_ports(8180)
        container.with_volume_mapping(str(bundle_root), SIDECAR_BUNDLE_TARGET, "rw")
        return container

    container, port = start_or_fail(
        build, label=f"qmd OKF sidecar ({QMD_SIDECAR_IMAGE})", port=8180
    )
    request.addfinalizer(lambda: stop_quietly(container))

    wait_for_health(container, port)
    return container, port


def wait_for_health(container, port: int) -> None:
    """Block until the sidecar answers ``/health``, or fail with its own logs.

    A timeout is a failure, not a skip: the image is present and the daemon was
    asked to start, so silence is a defect in the sidecar or the corpus rather
    than an absent dependency.
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
        f"(last: {last})\n--- container logs ---\n{tail_logs(container)}"
    )


def tail_logs(container) -> str:
    """Best-effort tail of a container's combined output, for failure messages."""
    try:
        stdout, stderr = container.get_logs()
        text = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")
        return text[-4000:]
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return f"(could not read container logs: {exc})"
