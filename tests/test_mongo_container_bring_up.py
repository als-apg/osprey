"""The throwaway-MongoDB recipe's bring-up and server flags, against scripted fakes.

A container that exits before it answers is rebuilt fresh, a bounded number of
times; one that stays up and will not answer fails at once. The flags the
recipe passes start mongod's TTL sweeper off. No container engine is needed.
"""

from __future__ import annotations

import sys
import types

import pytest

from tests import _mongo_container
from tests._container_support import (
    CONTAINER_START_ATTEMPTS,
    ContainerExitedError,
)

# ---------------------------------------------------------------------------
# Bring-up: a container that exits before it answers is rebuilt, a live one is not
# ---------------------------------------------------------------------------


class _FakeMongo:
    """A started container as far as the recipe reads one."""

    def __init__(self, number: int) -> None:
        self.number = number

    def get_container_host_ip(self) -> str:
        return "127.0.0.1"

    def get_exposed_port(self, _port: int) -> int:
        return 40000 + self.number


class _BringUp:
    """Scripts the recipe's three collaborators and records what it did with them.

    ``waits`` is one entry per bring-up: an exception to raise from the readiness
    wait, or ``None`` for a store that answered.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, waits: list[BaseException | None]) -> None:
        self.waits = list(waits)
        self.started: list[_FakeMongo] = []
        self.stopped: list[_FakeMongo] = []
        # The recipe imports the container class before it starts anything, and
        # skips when the extra is absent; the class itself is never reached here.
        community = types.ModuleType("testcontainers.community.mongodb")
        community.MongoDbContainer = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "testcontainers.community.mongodb", community)
        monkeypatch.setattr(_mongo_container, "start_or_skip", self._start)
        monkeypatch.setattr(_mongo_container, "wait_until_ready", self._wait)
        monkeypatch.setattr(_mongo_container, "stop_quietly", self.stopped.append)

    # The container helper this stands in for is asked for its label by keyword.
    def _start(self, _factory: object, *, label: str) -> _FakeMongo:  # noqa: ARG002
        container = _FakeMongo(len(self.started) + 1)
        self.started.append(container)
        return container

    def _wait(self, _probe: object, _label: str, *, container: _FakeMongo) -> None:
        assert container is self.started[-1], "the wait must watch the container just started"
        outcome = self.waits.pop(0)
        if outcome is not None:
            raise outcome


def _bring_up() -> tuple[str, int]:
    with _mongo_container.started_mongo("mongodb-test", username="u", password="p") as reached:
        return reached


def test_a_store_that_exits_during_its_boot_is_rebuilt_fresh(monkeypatch) -> None:
    """The image's second boot can lose its port to the first; that container is gone."""
    scripted = _BringUp(monkeypatch, [ContainerExitedError("exit 48"), None])

    host, port = _bring_up()

    assert (host, port) == ("127.0.0.1", 40002), "the address must be the SECOND container's"
    assert [c.number for c in scripted.started] == [1, 2]
    assert [c.number for c in scripted.stopped] == [1, 2], (
        "the exited container is stopped before the rebuild, and the good one at the end"
    )


def test_a_live_store_that_never_answers_is_not_rebuilt(monkeypatch) -> None:
    """Credentials or flags: the same container on a second look, so fail at once."""
    scripted = _BringUp(monkeypatch, [AssertionError("answered but would not serve the probe")])

    with pytest.raises(AssertionError, match="would not serve the probe"):
        _bring_up()

    assert len(scripted.started) == 1
    assert [c.number for c in scripted.stopped] == [1], (
        "a failed bring-up must not leak its container"
    )


def test_a_store_that_exits_on_every_fresh_start_fails_naming_every_attempt(monkeypatch) -> None:
    """Bounded: an image or flag defect exits every time and must surface as one."""
    exits: list[BaseException | None] = [
        ContainerExitedError(f"exit on start {n}") for n in range(1, CONTAINER_START_ATTEMPTS + 1)
    ]
    scripted = _BringUp(monkeypatch, exits)

    with pytest.raises(AssertionError) as caught:
        _bring_up()

    message = str(caught.value)
    assert not isinstance(caught.value, ContainerExitedError)
    assert len(scripted.started) == CONTAINER_START_ATTEMPTS
    assert len(scripted.stopped) == CONTAINER_START_ATTEMPTS
    for n in range(1, CONTAINER_START_ATTEMPTS + 1):
        assert f"attempt {n}: exit on start {n}" in message


# ---------------------------------------------------------------------------
# Server flags: the sweeper starts off, and a test turns it on at runtime
# ---------------------------------------------------------------------------

#: The parameter that switches mongod's TTL sweeper off, as the recipe passes it.
SWEEPER_OFF = "ttlMonitorEnabled=false"

#: The sweep interval the recipe passes, so a pass a test turns on is quick.
SWEEP_INTERVAL = "ttlMonitorSleepSecs=1"


def _set_parameters(flags: list[str]) -> list[str]:
    """The values passed to ``--setParameter``, in order."""
    return [
        value for flag, value in zip(flags, flags[1:], strict=False) if flag == "--setParameter"
    ]


def test_the_sweeper_starts_off() -> None:
    """A seeded store keeps the fixture's clock, so no sweep runs unless a test asks."""
    default = _mongo_container._server_flags(None)
    assert _set_parameters(default) == [SWEEP_INTERVAL, SWEEPER_OFF]

    caller = ["--wiredTigerCollectionBlockCompressor", "zstd"]
    flags = _mongo_container._server_flags(caller)
    assert flags[: len(caller)] == caller, "the caller's flags come first, as given"
    assert _set_parameters(flags[len(caller) :]) == [SWEEP_INTERVAL, SWEEPER_OFF]
