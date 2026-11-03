"""The ``backend`` fixture every contract case runs on, once per backend.

Each backend is one entry of :data:`OPENERS`: a context manager that opens a
fresh, empty record for one test and yields a :class:`Harness`. A case never
asks which backend it is on; a difference between backends is a contract bug,
never a branch in a test.

``Harness.settle`` brings the backend to the state a later reader sees once
background work is done (a merge of duplicate parts, say). A backend with no
such work settles as a no-op.

The ClickHouse record runs in one throwaway server per session, the pinned
image started through :func:`tests._container_support.start_or_skip`, staged
by :mod:`osprey.agent_record.backends.clickhouse_stage` through ``docker
exec`` as the store's admin, and read and written as the agent-record account
the stage created. Each test starts from emptied tables; its settle forces the
merges a store would do in the background (``OPTIMIZE ... FINAL``).
"""

from __future__ import annotations

import secrets
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from osprey.agent_record.backends import clickhouse_ddl, clickhouse_stage
from osprey.agent_record.backends.clickhouse import ClickHouseBackend
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.config import DEFAULT_CLICKHOUSE_IMAGE
from osprey.agent_record.credentials import CLICKHOUSE_ADMIN_USER
from tests._container_support import (
    CONTAINER_START_ATTEMPTS,
    ContainerExitedError,
    is_docker_available,
    start_or_skip,
    stop_quietly,
    wait_until_ready,
)

_CLICKHOUSE_HTTP_PORT = 8123


@dataclass(frozen=True)
class Harness:
    """One open backend and the call that settles it."""

    backend: Any
    settle: Callable[[], None]


@contextmanager
def _embedded(_request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Harness]:
    with EmbeddedBackend(tmp_path / "record.duckdb") as backend:
        yield Harness(backend, lambda: None)


@dataclass(frozen=True)
class _ClickHouseStore:
    """One staged store: where to reach it, the account's password, an admin executor."""

    url: str
    password: str
    admin: clickhouse_stage.Executor


def _docker_executor(container_id: str) -> clickhouse_stage.Executor:
    """``docker exec -i <container> clickhouse-client``, as the admin the container's env names."""

    def run(args: Sequence[str], stdin: str | None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "exec", "-i", container_id, "clickhouse-client", *args],
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )

    return run


@pytest.fixture(scope="session")
def clickhouse_store() -> Iterator[_ClickHouseStore]:
    """A started, staged ClickHouse record store for the whole session."""
    try:
        from testcontainers.core.container import DockerContainer
    except ImportError:
        pytest.skip("testcontainers not installed")
    # Building the container object already asks the engine, so a host with
    # none is told apart here, as a skip, before anything is built.
    if not is_docker_available():
        pytest.skip("Docker not available — the ClickHouse record runs in a container")
    admin_password = secrets.token_hex(16)

    def build():
        return (
            DockerContainer(DEFAULT_CLICKHOUSE_IMAGE)
            .with_env("CLICKHOUSE_USER", CLICKHOUSE_ADMIN_USER)
            .with_env("CLICKHOUSE_PASSWORD", admin_password)
            .with_env("CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT", "1")
            .with_exposed_ports(_CLICKHOUSE_HTTP_PORT)
        )

    def answers(url: str) -> None:
        response = httpx.post(
            url,
            content=b"SELECT 1",
            headers={
                "X-ClickHouse-User": CLICKHOUSE_ADMIN_USER,
                "X-ClickHouse-Key": admin_password,
            },
            timeout=2.0,
        )
        response.raise_for_status()

    label = "agent record ClickHouse"
    exits: list[str] = []
    for attempt in range(1, CONTAINER_START_ATTEMPTS + 1):
        container = start_or_skip(build, label=label)
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(_CLICKHOUSE_HTTP_PORT))
        url = f"http://{host}:{port}/"
        try:
            wait_until_ready(lambda url=url: answers(url), label, timeout=90.0, container=container)
        except ContainerExitedError as exc:
            stop_quietly(container)
            exits.append(f"attempt {attempt}: {exc}")
            continue
        except BaseException:
            stop_quietly(container)
            raise
        break
    else:
        raise AssertionError(
            f"{label}: the container exited before answering on every one of "
            f"{CONTAINER_START_ATTEMPTS} fresh starts.\n" + "\n\n".join(exits)
        )
    try:
        admin = _docker_executor(container.get_wrapped_container().id)
        password = secrets.token_hex(16)
        clickhouse_stage.stage(admin, password)
        yield _ClickHouseStore(url=url, password=password, admin=admin)
    finally:
        stop_quietly(container)


@contextmanager
def _clickhouse(request: pytest.FixtureRequest, _tmp_path: Path) -> Iterator[Harness]:
    store: _ClickHouseStore = request.getfixturevalue("clickhouse_store")
    tables = [f"`{clickhouse_ddl.DATABASE}`.`{t.name}`" for t in clickhouse_ddl.TABLES]
    clickhouse_stage.apply_script(store.admin, [f"TRUNCATE TABLE {t}" for t in tables])

    def settle() -> None:
        clickhouse_stage.apply_script(store.admin, [f"OPTIMIZE TABLE {t} FINAL" for t in tables])

    with ClickHouseBackend(store.url, store.password) as backend:
        yield Harness(backend, settle)


#: Backend name to the opener of a fresh record for one test.
OPENERS: dict[str, Callable[[pytest.FixtureRequest, Path], Any]] = {
    "clickhouse": _clickhouse,
    "embedded": _embedded,
}

#: Backends that start a container share the suite's docker xdist group.
_DOCKER_BACKENDS = frozenset({"clickhouse"})


@pytest.fixture(
    params=[
        pytest.param(name, marks=pytest.mark.xdist_group("docker"))
        if name in _DOCKER_BACKENDS
        else name
        for name in sorted(OPENERS)
    ],
    ids=sorted(OPENERS),
)
def harness(request, tmp_path: Path) -> Iterator[Harness]:
    with OPENERS[request.param](request, tmp_path) as opened:
        yield opened


@pytest.fixture
def backend(harness: Harness):
    return harness.backend


@pytest.fixture
def settle(harness: Harness) -> Callable[[], None]:
    return harness.settle
