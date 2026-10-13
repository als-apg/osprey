"""The restore estimate of a guarded run is read from the run's one journal.

:func:`osprey.runtime.guarded_run.restore_estimate` counts the writes a restore
of the open run needs over every address the run journaled, whichever thread
wrote it, with the run's longest write latency and the time its own read of the
current values took. The journal is copied under the run state's lock and the
machine is read after it is released. Channels are a dict served through
``osprey.runtime``; every clock is fake.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import osprey.runtime
import osprey.runtime._run_state as _run_state
import osprey.runtime.journal as journal_mod
from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
from osprey.runtime.guarded_run import RestoreEstimate, journaled_run, restore_estimate
from osprey.runtime.journal import guarded_write
from osprey_connectors.control_system.limits_validator import ChannelLimitsConfig
from tests.runtime._guarded_run_fixtures import (
    GENERATION,
    fresh_run_state,  # noqa: F401 - used through pytestmark
    write_config,
)

pytestmark = pytest.mark.usefixtures("fresh_run_state")

#: How long a thread may take to finish; a hang guard, never a statement about speed.
TIMEOUT_S = 60.0


class _Clock:
    """A settable monotonic clock; writes and reads advance it by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Machine:
    """A dict of channel values; a write takes ``write_s`` and a read ``read_s`` on the clock."""

    def __init__(self, values: dict[str, Any], clock: _Clock) -> None:
        self.values = dict(values)
        self.clock = clock
        self.write_s = 0.0
        self.read_s = 0.0
        self.read_error: Exception | None = None
        self.limits: dict[str, ChannelLimitsConfig] = {}

    def read_channels(self, addresses: Sequence[str], **_kwargs: Any) -> list[Any]:
        self.clock.now += self.read_s
        if self.read_error is not None:
            raise self.read_error
        return [self.values[a] for a in addresses]

    def write_channel(self, address: str, value: Any, **_kwargs: Any) -> None:
        self.clock.now += self.write_s
        self.values[address] = value

    def channel_limits(self, address: str) -> ChannelLimitsConfig | None:
        return self.limits.get(address)

    def guarded(self, address: str, value: Any) -> None:
        guarded_write([address], lambda: self.write_channel(address, value), lambda exc: exc)


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo on a live baseline that asks nobody, entered and stamped ``live``."""
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root, {"enabled": False})
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.setenv(ENV_CONTROL_TARGET_GENERATION, str(GENERATION))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    return root


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> _Machine:
    """``A`` at 1.0, ``B`` at 2.0 and ``C`` at 3.0, no limits record, on a fake clock."""
    clock = _Clock()
    monkeypatch.setattr(journal_mod, "_clock", clock)
    fake = _Machine({"A": 1.0, "B": 2.0, "C": 3.0}, clock)
    for name in ("read_channels", "write_channel", "channel_limits"):
        monkeypatch.setattr(osprey.runtime, name, getattr(fake, name))
    return fake


def _in_thread(target: Any) -> None:
    errors: list[BaseException] = []

    def run() -> None:
        try:
            target()
        except BaseException as exc:  # kept for the test to assert on
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(TIMEOUT_S)
    assert not thread.is_alive(), "the thread did not finish"
    assert errors == []


def test_with_no_run_open_there_is_no_estimate() -> None:
    with pytest.raises(RuntimeError, match="no journaled guarded run is open"):
        restore_estimate()


@pytest.mark.usefixtures("machine")
def test_an_empty_run_estimates_no_write_and_no_read() -> None:
    with journaled_run(None):
        assert restore_estimate() == RestoreEstimate(writes=0, write_latency=None, read_latency=0.0)


def test_writes_from_every_thread_of_the_run_are_counted(machine: _Machine) -> None:
    with journaled_run(None):
        machine.write_s = 0.5
        machine.guarded("A", 10.0)

        def helper() -> None:
            machine.write_s = 2.0
            machine.guarded("B", 20.0)
            machine.write_s = 1.0
            machine.guarded("C", 30.0)

        _in_thread(helper)
        machine.read_s = 0.25
        estimate = restore_estimate()

    assert estimate == RestoreEstimate(writes=3, write_latency=2.0, read_latency=0.25)


def test_a_walk_back_under_a_max_step_counts_its_steps(machine: _Machine) -> None:
    machine.limits["A"] = ChannelLimitsConfig(
        channel_address="A", min_value=-100.0, max_value=100.0, max_step=1.0, writable=True
    )
    max_steps: dict[str, float | None] = {}
    with journaled_run(None):
        machine.guarded("A", 4.0)
        machine.guarded("B", 20.0)
        estimate = restore_estimate(max_steps)

    assert estimate.writes == 3 + 1
    assert max_steps == {"A": 1.0, "B": None}


def test_a_read_that_raises_counts_one_write_per_address(machine: _Machine) -> None:
    machine.limits["A"] = ChannelLimitsConfig(
        channel_address="A", min_value=-100.0, max_value=100.0, max_step=1.0, writable=True
    )
    with journaled_run(None):
        machine.guarded("A", 4.0)
        machine.guarded("B", 20.0)
        machine.read_error = RuntimeError("the gateway is down")
        machine.read_s = 3.0
        estimate = restore_estimate()

    assert estimate.writes == 2
    assert estimate.read_latency == 3.0


def test_the_machine_is_read_with_the_run_state_lock_released(
    monkeypatch: pytest.MonkeyPatch, machine: _Machine
) -> None:
    """A read under the lock would leave the other thread's acquire to time out."""
    taken: list[bool] = []
    read = machine.read_channels

    def read_while_another_thread_takes_the_lock(
        addresses: Sequence[str], **kwargs: Any
    ) -> list[Any]:
        def take() -> None:
            got = _run_state.LOCK.acquire(timeout=TIMEOUT_S)
            taken.append(got)
            if got:
                _run_state.LOCK.release()

        _in_thread(take)
        return read(addresses, **kwargs)

    with journaled_run(None):
        machine.guarded("A", 10.0)
        monkeypatch.setattr(
            osprey.runtime, "read_channels", read_while_another_thread_takes_the_lock
        )
        estimate = restore_estimate()

    assert taken == [True]
    assert estimate.writes == 1
