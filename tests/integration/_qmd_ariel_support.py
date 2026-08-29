"""Container and config helpers for the qmd/ARIEL integration lane.

The leading underscore keeps this out of pytest collection, matching
``tests/_container_support.py``: ``python_files`` matches ``test_*.py`` only, and
a collected helper module would report its imports as test failures.

Fixtures live in ``conftest.py``; everything importable by a test module lives
here, so nothing has to import a conftest.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import requests

from osprey.port_layout import default_port
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


#: The other image this lane builds: the same Dockerfile with the three model
#: fetches skipped (``--build-arg OSPREY_QMD_MODELS_MOUNTED=1``), which is what
#: a build host with no route to huggingface.co produces. Cheap to build — it
#: downloads none of the 2.1 GB — and it is the only image the mounted-delivery
#: path can be tested against, because a baked image already has the models.
MOUNTED_IMAGE_ENV = "OSPREY_QMD_MOUNTED_IMAGE"

#: Where that image looks for its models, and therefore where the deployment's
#: read-only bind mount lands. The literal is the image's own
#: ``OSPREY_QMD_MODEL_DIR``; ``tests/deployment/test_qmd_compose_fragment.py``
#: pins the compose fragment's copy of it to the Dockerfile.
SIDECAR_MODEL_DIR = "/opt/qmd/.cache/qmd/models"

#: The sidecar's writable state directory — the index, and the stamp file that
#: records which model files were already verified. Deliberately outside the
#: model tree, so nothing the container writes lands on the read-only mount.
SIDECAR_STATE_DIR = "/var/lib/qmd"

#: Per-model environment variable carrying the digest the image expects, keyed
#: by the model's on-disk name. Setting these at ``docker run`` overrides the
#: ENV the build baked in, which is what lets a kilobyte fixture stand in for a
#: 318 MB model: the check is "this file is the file this image was built with",
#: and both halves of that are supplied here. The identity file the entrypoint
#: sources uses different names (``OSPREY_QMD_EMBED_MODEL_SHA256``), so it
#: cannot clobber them.
MODEL_DIGEST_ENV = {
    "hf_ggml-org_embeddinggemma-300M-Q8_0.gguf": "OSPREY_QMD_EMBED_SHA256",
    "hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf": "OSPREY_QMD_RERANK_SHA256",
    "hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf": "OSPREY_QMD_GENERATE_SHA256",
}

#: How long the model check gets to reach its verdict. It runs before anything
#: slow — before the index pass, before the daemon — so this covers container
#: start plus three sha256sums of whatever was staged.
MODEL_VERDICT_TIMEOUT = 120.0


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
    container.with_env("OSPREY_QMD_PORT", str(SIDECAR_CONTAINER_PORT))
    container.with_exposed_ports(SIDECAR_CONTAINER_PORT)
    container.with_volume_mapping(str(corpus_root), SIDECAR_CORPUS_TARGET, "ro")
    return container


#: Container-side mount point for the ARIEL markdown mirror.
SIDECAR_CORPUS_TARGET = "/corpus/ariel"

#: The port the sidecar's forwarder owns inside the container. Not qmd's own
#: 8181, which the daemon binds on IPv6 loopback only. The entrypoint takes it
#: from ``OSPREY_QMD_PORT`` and refuses to guess, so this fixture — which runs
#: the image outside a rendered deployment — sets it to qmd's own layout slot
#: and publishes the same number.
SIDECAR_CONTAINER_PORT = default_port("qmd")

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


def require_mounted_models_image() -> str:
    """The skip-ARG image, or the right kind of non-result.

    Same split as :func:`require_sidecar_image`, for the same reason and with
    the opposite default. The variable is set by one CI step, the one that
    builds this image; anywhere else — a developer machine, or this lane's
    other steps, which run before that build — its absence is honest and a skip
    is the correct outcome. A variable that IS set and names an absent image is
    a failed build, and skipping there would report success having proved
    nothing about the delivery path this whole feature exists for.

    Returns:
        The image reference to run.

    Raises:
        AssertionError: If the named image is not present on this daemon.
        Skipped: Via ``pytest.skip`` when the variable is unset.
    """
    image = os.environ.get(MOUNTED_IMAGE_ENV)
    if not image:
        pytest.skip(
            f"{MOUNTED_IMAGE_ENV} is unset; these cases need the sidecar image built "
            "with the model fetches skipped (--build-arg OSPREY_QMD_MODELS_MOUNTED=1), "
            "which only the qmd sidecar CI lane builds"
        )
    if not is_image_present(image):
        raise AssertionError(
            f"{MOUNTED_IMAGE_ENV}={image!r} names an image that is not present on this "
            "daemon. The step that sets this variable builds the image first, so this "
            "is a failed or mis-tagged build, not a missing dependency."
        )
    return image


def stage_model_fixtures(directory: Path) -> dict[str, str]:
    """Write three stand-in model files and return their digests.

    Kilobyte files, not models: the entrypoint's check is a name, a non-zero
    size and a SHA256 comparison, and none of those care what the bytes mean.
    Staging the real 2.1 GB in CI to assert a digest comparison would buy
    nothing and cost the download this feature exists to avoid. The contents
    differ per file so a check that compared the wrong file against the wrong
    digest would fail rather than pass by coincidence.

    Args:
        directory: Host directory to stage into, mounted read-only afterwards.

    Returns:
        Mapping of ``OSPREY_QMD_*_SHA256`` variable name to the digest of the
        file it describes, ready to pass as container environment.
    """
    digests = {}
    for index, (name, variable) in enumerate(MODEL_DIGEST_ENV.items()):
        payload = bytes([index + 1]) * 1024
        (directory / name).write_bytes(payload)
        digests[variable] = hashlib.sha256(payload).hexdigest()
    return digests


@contextmanager
def sidecar_over_models_dir(
    image: str,
    models_dir: Path,
    env: dict[str, str] | None = None,
    state_volume: str | None = None,
):
    """Run the sidecar over ``models_dir``, mounted where the image looks.

    Detached and unpublished: every case here is decided by what the container
    logs before it opens a port, and both of them end with the container
    exiting on its own — a refusal immediately, a verified start once the
    fail-closed index gate finds nothing to serve. Neither is a container to
    wait on with a health probe.

    Args:
        image: Image reference to run.
        models_dir: Host directory to bind read-only at the image's model
            directory.
        env: Extra environment for the container.
        state_volume: Named volume to mount as the writable state directory, for
            the cases that need one container's state to reach the next.

    Yields:
        The started container.
    """
    import docker

    client = docker.from_env()
    volumes: dict[str, dict[str, str]] = {
        str(models_dir): {"bind": SIDECAR_MODEL_DIR, "mode": "ro"}
    }
    if state_volume:
        volumes[state_volume] = {"bind": SIDECAR_STATE_DIR, "mode": "rw"}
    # The entrypoint takes its port from OSPREY_QMD_PORT before it looks at
    # the models and refuses to guess, so a run outside a rendered deployment
    # has to say the port even though nothing here ever connects to it.
    environment = {"OSPREY_QMD_PORT": str(SIDECAR_CONTAINER_PORT), **(env or {})}
    container = client.containers.run(
        image,
        detach=True,
        environment=environment,
        volumes=volumes,
    )
    try:
        yield container
    finally:
        with suppress(Exception):
            container.remove(force=True)


def wait_for_sidecar_log(container, needle: str, timeout: float = MODEL_VERDICT_TIMEOUT) -> str:
    """Block until ``needle`` appears in the container's output.

    A container that exits without it gets one final read before the failure,
    because the last lines and the exit are written at the same moment.

    Args:
        container: A started container.
        needle: Substring to wait for.
        timeout: Seconds to wait.

    Returns:
        The full log text at the moment the needle was found.

    Raises:
        AssertionError: If the container exits or the timeout passes first. The
            message carries the log, which is the diagnosis.
    """
    deadline = time.monotonic() + timeout
    while True:
        text = _combined_logs(container)
        if needle in text:
            return text
        exited = _exit_code(container) is not None
        if exited or time.monotonic() >= deadline:
            raise AssertionError(
                f"the sidecar never logged {needle!r} "
                + ("before exiting" if exited else f"within {timeout:.0f}s")
                + f"\n--- container logs ---\n{text[-4000:]}"
            )
        time.sleep(0.5)


def wait_for_sidecar_exit(container, timeout: float = MODEL_VERDICT_TIMEOUT) -> int:
    """Return the container's exit status, waiting for it if necessary."""
    result = container.wait(timeout=timeout)
    return int(result.get("StatusCode", -1))


def sidecar_logs(container) -> str:
    """Everything the container has written so far.

    Read a refusal through this only after the container has exited. A refusal
    is several lines long — the cause, the staging instructions, the three
    names — and a read that lands between the first line and the last returns a
    truthfully incomplete message, which reads exactly like a message that was
    never finished.
    """
    return _combined_logs(container)


def _exit_code(container) -> int | None:
    """The container's exit status, or ``None`` while it is still running."""
    with suppress(Exception):
        container.reload()
        if container.status == "exited":
            return int(container.attrs["State"]["ExitCode"])
    return None


def _combined_logs(container) -> str:
    """Everything the container has written so far, best effort."""
    try:
        return container.logs().decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return f"(could not read container logs: {exc})"


def _tail_logs(container) -> str:
    """Best-effort tail of a container's combined output, for failure messages."""
    try:
        stdout, stderr = container.get_logs()
        text = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")
        return text[-4000:]
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return f"(could not read container logs: {exc})"


# ---------------------------------------------------------------------------
# Vocabulary expansion
# ---------------------------------------------------------------------------

#: The facility vocabulary the expansion cases run against. Both concepts are
#: chosen so the shorthand an operator types shares no token with the prose the
#: mirror holds -- ``t/s`` never appears in a logbook entry, ``troubleshoot``
#: does -- which is what makes the expansion's effect measurable rather than
#: incidental. ``kind`` differs between them on purpose: the two direction
#: gates (``canonical_to_acronym`` / ``canonical_to_shorthand``) are keyed on
#: it, and the direction this lane exercises (form to canonical) is the one
#: neither gate can switch off.
VOCABULARY_YML = """\
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - ts
      - t/s
  - canonical: beam position monitor
    kind: acronym
    forms:
      - bpm
"""

#: The query every expansion case sends: two facility shorthands and nothing
#: else, so an unexpanded run has no lexical purchase on the seeded prose at all.
VOCABULARY_QUERY = "t/s bpm"

#: The expansions :data:`VOCABULARY_QUERY` must resolve to, as
#: ``original -> alternatives``. Spelled here rather than inside the assertion
#: so the vocabulary above and the expectation cannot drift apart.
VOCABULARY_EXPECTED_PAIRS = {
    "t/s": ("troubleshoot",),
    "bpm": ("beam position monitor",),
}

#: The entry only an expanded query is supposed to reach, seeded as **its own
#: group**. Deliberately not a member of the test module's ``SAFE_ROWS``: the
#: sidecar fixture waits for ``len(hits) >= len(SAFE_ROWS)`` before it declares
#: the corpus indexed, and three post-filter tests compare ``_safe_ids(...)``
#: against exact subsets of that group, so an eighth member would quietly change
#: what all of those assertions mean. It is not part of the hostile-id group
#: either: nothing here is about encoding.
#:
#: The identifier itself carries **neither** shorthand nor canonical phrase, and
#: that is load-bearing rather than tidy: the mirror writer puts the id in the
#: document's ``# Entry <id>`` heading, so qmd indexes it as prose. An id spelling
#: out ``troubleshoot`` or ``bpm`` hands the unexpanded query a lexical match on
#: the title alone — measured at the time: it took the entry to rank 1 with no
#: expansion at all, which would have made the effect test pass for the wrong
#: reason in one direction and fail in the other.
VOCABULARY_ENTRY_ID = "vocab-needle-0001"

#: Its prose. Written the way an operator writes, and containing **only** the
#: canonical phrases. If ``ts``, ``t/s`` or ``bpm`` ever appears in this string,
#: the unexpanded query could match it lexically and the effect test would pass
#: without the expansion having done anything.
VOCABULARY_PROSE = (
    "Had to troubleshoot the beam position monitor offset in the storage ring "
    "this morning; recalibrated after the orbit drift."
)

#: Near misses, and the reason the effect test measures anything at all.
#: Measured: with no other document in the corpus mentioning a beam position
#: monitor, the sidecar's vector leg retrieves the entry above from a bare
#: ``t/s bpm`` on its own — an embedding model already knows ``bpm``, so the
#: unexpanded query lands on the only candidate there is and the expansion
#: cannot be shown to have done anything. These rows each carry **one** of the
#: two canonical concepts and neither shorthand, so an unexpanded query has real
#: competition while the expanded query, which names both concepts, ranks the
#: entry that has both above all of them.
#:
#: **How many is a measurement, not a preference.** Four such rows overshot:
#: they pushed the *expanded* rank down to 3 as well, which is the last slot the
#: effect test's top-3 assertion allows — no headroom at all. With the three
#: below, measured twice on fresh containers and fresh indexes against
#: ``als-assistant-qmd:local``, byte-identical both times: the needle ranks **2**
#: expanded (behind ``beam_current_setpoint`` only) and **7** unexpanded. So the
#: positive half has one spare slot and the negative half four. Adding a row here
#: costs the positive half; removing one costs the negative half.
VOCABULARY_DISTRACTOR_ROWS: list[tuple[str, str]] = [
    (
        "vocab-distractor-rack",
        "The beam position monitor electronics rack in sector three was swapped "
        "during the shutdown.",
    ),
    (
        "vocab-distractor-orbit",
        "Orbit drift logged by the beam position monitor system overnight.",
    ),
    (
        "vocab-distractor-archive",
        "Beam position monitor readings were archived after the injection study.",
    ),
]

#: An author of its own, and a year no date filter in the module spans, so these
#: rows cannot land inside a post-filter expectation.
VOCABULARY_AUTHOR = "Vocabulary Probe"
VOCABULARY_SOURCE_SYSTEM = "ALS OLOG"
VOCABULARY_TIMESTAMP = datetime(2009, 5, 12, 12, 0, tzinfo=UTC)


def write_vocabulary_file(directory: Path) -> Path:
    """Write :data:`VOCABULARY_YML` and return its absolute path.

    Args:
        directory: Host directory to write into.

    Returns:
        Path of the written ``vocabulary.yml``, ready for
        ``ariel.vocabulary.path``, which takes an absolute path as-is.
    """
    path = directory / "vocabulary.yml"
    path.write_text(VOCABULARY_YML, encoding="utf-8")
    return path


def qmd_ariel_vocabulary_config(
    base: dict[str, Any],
    vocabulary_path: Path | str,
    *,
    expand_modes: list[str] | None = None,
) -> dict[str, Any]:
    """Copy ``base`` with an enabled ``vocabulary`` block bolted on.

    A copy rather than a mutation: the session-scoped sidecar world hands out
    one config dict and every other test in the module reads it.

    Args:
        base: The lane's config section, from :func:`qmd_ariel_config_dict`.
        vocabulary_path: Absolute path of the vocabulary file to load.
        expand_modes: Value for ``ariel.vocabulary.expand_modes``. ``None``
            leaves the key out, which means "every enabled search module".

    Returns:
        A new config dict carrying the vocabulary block.
    """
    vocabulary: dict[str, Any] = {"enabled": True, "path": str(vocabulary_path)}
    if expand_modes is not None:
        vocabulary["expand_modes"] = list(expand_modes)
    return {**base, "vocabulary": vocabulary}
