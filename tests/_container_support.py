"""Shared helpers for testcontainers-backed tests.

Several test packages (``tests/connectors/``, ``tests/services/ariel_search/``,
``tests/e2e/``) spin up real containers and must degrade to a skip rather than
an error when the host has no Docker engine. This module is the single home for
the environment probes and start helpers they share.

Two start helpers live here because container suites split into two kinds, and
which one a suite wants is a real decision rather than a preference:
:func:`start_or_skip` for suites where an absent engine must not fail a
contributor's run, and :func:`start_or_fail` for suites that have already
established a daemon is present and for which a skip would be a vacuous green.
They sit side by side so that contrast is visible at the point of choosing.

The leading underscore keeps the module out of pytest collection: ``python_files``
matches ``test_*.py``/``*_test.py`` only, and a helper module that got collected
would report its imports as test failures.

Note on imports: ``docker`` is a ``dev``-extra dependency, so it is imported
lazily inside the functions that need it. A module-scope import would turn a
missing extra into a collection error for every test package that imports from
here. ``requests`` is a base dependency and is safe at module scope.
"""

import logging
import time
from collections.abc import Callable
from contextlib import suppress
from typing import TypeVar

import pytest
import requests

logger = logging.getLogger(__name__)

ContainerT = TypeVar("ContainerT")


def is_docker_available() -> bool:
    """Return True if the Docker daemon is reachable.

    Used to gate testcontainers-backed fixtures so that contributors
    without a running Docker engine see a skip rather than an error.
    """
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception as e:
        logger.warning(f"Docker not available: {e}")
        return False


def is_image_present(reference: str) -> bool:
    """Return True if a local image matching ``reference`` exists.

    Distinguishes "this machine has no engine" from "this machine has an engine
    but was never asked to build the image", which are different situations that
    a plain start failure reports identically.

    Args:
        reference: Image reference, e.g. ``osprey-qmd:local-validate``.

    Returns:
        True when the local daemon can resolve the reference. False on an absent
        image, an unreachable daemon, or a missing ``docker`` package -- callers
        gate on :func:`is_docker_available` first, so those collapse together.
    """
    try:
        import docker

        docker.from_env().images.get(reference)
        return True
    except Exception as e:
        logger.warning(f"Image {reference} not present: {e}")
        return False


def stop_quietly(container: object) -> None:
    """Stop a container, swallowing teardown errors.

    ``stop()`` calls ``remove(force=True)``, which raises ``NotFound`` when the
    container is already gone — the same case testcontainers' own reaper
    suppresses. Letting that escape turns a clean run into an ERROR at teardown.

    Used both to discard a container built by a failed start attempt and as the
    teardown for fixtures that own one.
    """
    with suppress(Exception):
        container.stop()  # type: ignore[attr-defined]


def start_or_skip(
    factory: Callable[[], ContainerT],
    label: str,
    *,
    backoff: float = 3.0,
) -> ContainerT:
    """Start a container built by ``factory``, skipping the test on failure.

    ``factory`` must *build and return a new container object* on every call.
    A retry always builds a fresh one: ``DockerContainer.start()`` assigns
    ``self._container`` before the readiness wait, so reusing the object after a
    timeout would orphan the container started by the first attempt.

    The ``except`` chain is ordered, not a tuple, and the order is load-bearing
    because ``ImageNotFound ⊂ NotFound ⊂ APIError ⊂ RequestException``:

    * ``docker.errors.NotFound`` — a missing image or resource. Not transient,
      so skip immediately; a second 120s readiness wait cannot help.
    * ``requests.exceptions.RequestException`` — a flaky daemon or registry
      call. Retried once after ``backoff`` seconds.
    * ``Exception`` — anything else, notably ``TimeoutError``. Skip immediately.

    ``backoff`` is keyword-only so tests can pass ``backoff=0`` instead of
    paying real wall clock for the retry sleep.

    Args:
        factory: Zero-argument callable returning a fresh, unstarted container.
        label: Human-readable name for the container, used in skip messages.
        backoff: Seconds to wait before the single retry.

    Returns:
        The started container.

    Raises:
        Skipped: Via ``pytest.skip`` when the container cannot be started.
    """
    import docker.errors

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        container = factory()
        try:
            container.start()  # type: ignore[attr-defined]
            return container
        except docker.errors.NotFound as exc:
            stop_quietly(container)
            pytest.skip(_skip_message(label, exc, attempt))
        except requests.exceptions.RequestException as exc:
            stop_quietly(container)
            if attempt == max_attempts:
                pytest.skip(_skip_message(label, exc, attempt))
            logger.warning(
                f"{label}: start attempt {attempt} failed "
                f"({type(exc).__name__}: {exc}); retrying in {backoff}s"
            )
            time.sleep(backoff)
        except Exception as exc:
            stop_quietly(container)
            pytest.skip(_skip_message(label, exc, attempt))

    raise AssertionError("unreachable: retry loop always skips or returns")


#: Attempts a container start gets before a fail-hard lane gives up. The failure
#: this absorbs is a testcontainers race, not a slow image: it reads the
#: published port mapping the instant the container starts and raises when
#: Docker has not published it yet. Builtin ``ConnectionError`` is an
#: ``OSError``, so a helper that skips on any exception would turn that race
#: into a vacuous green.
CONTAINER_START_ATTEMPTS = 4


def start_or_fail(
    factory: Callable[[], ContainerT],
    label: str,
    port: int,
    *,
    backoff: float = 2.0,
) -> tuple[ContainerT, int]:
    """Start a container built by ``factory`` and read its published port.

    The fail-hard counterpart to :func:`start_or_skip`, for lanes that have
    already established a daemon is present. Two things make ``start_or_skip``
    wrong for such a lane, and dangerously so:

    1. Docker availability is checked before this is called, so past this point
       a container that will not start is a defect in the image, the corpus or
       the host, not an absent optional dependency — and a lane whose whole job
       is to prove a container-backed feature works must not be able to report
       success having executed nothing.
    2. The commonest failure is transient and ``start_or_skip`` does not retry
       it. testcontainers raises a **builtin** ``ConnectionError`` ("Port mapping
       for container ... is not available"). Builtin ``ConnectionError`` is an
       ``OSError``, not a ``requests.exceptions.RequestException``, so it lands
       in that helper's bare ``except Exception`` and skips on the first attempt.
       Observed on a second consecutive run: ``12 skipped ... exit 0``.

    **Reading the published port is part of the retried operation, not a step
    after it.** testcontainers queries the mapping the instant the container
    starts, so the same race bites twice. Retrying only ``start()`` leaves the
    second one live and it surfaces as a container that started fine and a test
    that died reading its port — observed as ``23 errors`` in a run whose
    containers were all healthy. Retrying inside the loop also means a failed
    attempt is discarded and a *fresh* container is built, rather than the port
    being re-read from the same one that never published it.

    Args:
        factory: Zero-argument callable returning a fresh, unstarted container.
            Called again per attempt — ``start()`` assigns the container handle
            before its readiness wait, so reusing the object would orphan the
            container a failed attempt started.
        label: Human-readable name for the container, used in the failure.
        port: Container port whose host mapping to resolve.
        backoff: Seconds between attempts.

    Returns:
        Tuple of the started container and its published host port.

    Raises:
        AssertionError: If every attempt failed.
    """
    failures: list[str] = []
    for attempt in range(1, CONTAINER_START_ATTEMPTS + 1):
        container = factory()
        try:
            container.start()  # type: ignore[attr-defined]
            return container, int(container.get_exposed_port(port))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — every start failure is retried alike
            stop_quietly(container)
            failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            logger.warning("%s: start attempt %d failed (%s)", label, attempt, exc)
            if attempt < CONTAINER_START_ATTEMPTS:
                time.sleep(backoff)

    remedy = ""
    if any("testcontainers-ryuk" in failure for failure in failures):
        # The first failure is the cause and the 409s are cascade: testcontainers
        # left a half-started reaper behind and every retry collides with its
        # name. Surfaced here because the cascade buries the real first line and
        # reads like a defect in the image. See tests/README.md §6 "Containers".
        remedy = (
            "\n\nThe 409 'container name testcontainers-ryuk-... already in use' lines are "
            "CASCADE, not the cause — read the first failure above. testcontainers' own "
            "reaper races Docker Desktop's port forwarder on macOS. Remove the stale "
            "reaper (`docker rm -f <its id>`) and re-run with "
            "TESTCONTAINERS_RYUK_DISABLED=1, which tests/README.md recommends for local "
            "macOS runs only — every fixture here stops its own containers. Do NOT set it "
            "in CI; Linux daemons do not show the race."
        )
    raise AssertionError(
        f"{label}: container would not start in {CONTAINER_START_ATTEMPTS} attempts, and the "
        f"docker daemon is reachable — this is a real failure, not a missing dependency.\n"
        + "\n".join(failures)
        + remedy
    )


def _skip_message(label: str, exc: BaseException, attempt: int) -> str:
    """Build a skip message naming the container, attempt count, and cause.

    The exception type *is* the diagnosis, so it is reported verbatim rather
    than mapped onto a bucket taxonomy.
    """
    return (
        f"{label}: container failed to start after {attempt} attempt(s) — "
        f"{type(exc).__name__}: {exc}"
    )
