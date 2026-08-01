"""Shared helpers for testcontainers-backed tests.

Several test packages (``tests/connectors/``, ``tests/services/ariel_search/``,
``tests/e2e/``) spin up real containers and must degrade to a skip rather than
an error when the host has no Docker engine. This module is the single home for
the environment probes and start helpers they share.

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


def _skip_message(label: str, exc: BaseException, attempt: int) -> str:
    """Build a skip message naming the container, attempt count, and cause.

    The exception type *is* the diagnosis, so it is reported verbatim rather
    than mapped onto a bucket taxonomy.
    """
    return (
        f"{label}: container failed to start after {attempt} attempt(s) — "
        f"{type(exc).__name__}: {exc}"
    )
