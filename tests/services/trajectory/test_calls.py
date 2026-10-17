"""A guarded call: a connector read or write whose outcome outlives a cancelled caller.

The call runs as its own task. A caller cancelled while the call is in flight
stays until the call returns, records its outcome, and only then raises the
cancellation, so the run lock the caller holds is released after the write has
returned and is listed. The writes here block on an event the test sets; the
lock and ``command_write`` are the product's own.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tests._guarded_run import reset_run_state, write_config

from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import (
    ENV_CONTROL_IDENTITY,
    ActionContext,
    OspreyRunBusy,
    command_write,
    lock,
)
from osprey.services.trajectory.calls import guarded_call
from osprey_connectors import posture_store
from osprey_connectors.control_system.base import ChannelWriteResult, WriteOutcome
from osprey_connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)

ARM = "SR:BPM:ARM"
REQUEST = "SR:INJ:REQ"
COMMANDS = frozenset({ARM, REQUEST})
ALICE_LIVE = ActionContext(target="live", owner="alice", approved_target="live")


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host repo on a live baseline, stamped ``live``, whose write path reads owner alice."""
    reset_run_state(monkeypatch)
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root, {"enabled": False})
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(ENV_CONTROL_IDENTITY, raising=False)
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    return root


class _Gate:
    """A write that blocks until released, then returns its value or raises."""

    def __init__(self, value: Any = 1.5, error: BaseException | None = None) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.returned = False
        self.value = value
        self.error = error

    async def write(self) -> Any:
        self.started.set()
        await self.release.wait()
        self.returned = True
        if self.error is not None:
            raise self.error
        return self.value


class _Connector:
    """A connector whose batch write blocks on *gate*; it records every batch it was asked."""

    def __init__(self, gate: _Gate) -> None:
        self._control_target = "live"
        self._limits_validator = LimitsValidator(
            {
                ARM: ChannelLimitsConfig(ARM, 0, 1, writable=True),
                REQUEST: ChannelLimitsConfig(REQUEST, 0, 1000, writable=True),
            },
            {"mode": "exclusive", "on_violation": "error"},
            {
                ARM: {"min_value": 0, "max_value": 1, "confirm": False},
                REQUEST: {"min_value": 0, "max_value": 1000, "confirm": False},
            },
        )
        self.gate = gate
        self.calls: list[list[tuple[str, Any]]] = []

    def _resolve_confirm(self, address: str) -> bool:
        return bool(self._limits_validator.resolve_confirm(address))

    def _current_value_reader(self) -> None:
        return None

    async def write_multiple_channels(
        self, operations: list[tuple[str, Any]], **kwargs: Any
    ) -> list[ChannelWriteResult]:
        self.calls.append(list(operations))
        await self.gate.write()
        return [
            ChannelWriteResult(
                channel_address=address,
                value_written=value,
                outcome=WriteOutcome.UNREQUESTED,
            )
            for address, value in operations
        ]


def _lock_is_held() -> bool:
    """Whether this process still holds the run lock: a second ask is busy while it does."""
    try:
        with lock(None, context=ALICE_LIVE):
            return False
    except OspreyRunBusy:
        return True


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_a_call_returns_its_value_and_records_it() -> None:
    gate = _Gate(value=2.0)
    gate.release.set()
    moved: list[Any] = []

    assert await guarded_call(gate.write, record=moved.append) == 2.0
    assert moved == [2.0]


async def test_a_raising_write_records_the_exception_and_raises_it() -> None:
    error = RuntimeError("the IOC refused the put")
    gate = _Gate(error=error)
    gate.release.set()
    moved: list[Any] = []

    with pytest.raises(RuntimeError, match="refused the put"):
        await guarded_call(gate.write, record=moved.append)
    assert moved == [error]


async def test_cancelling_during_a_blocked_write_holds_the_lock_until_it_returns() -> None:
    gate = _Gate(value=0.25)
    moved: list[Any] = []
    released_after: list[bool] = []

    async def action() -> None:
        with lock(None, context=ALICE_LIVE):
            try:
                await guarded_call(gate.write, record=moved.append)
            finally:
                released_after.append(gate.returned)

    caller = asyncio.create_task(action())
    await gate.started.wait()
    caller.cancel()
    await _settle()

    # The caller is cancelled, but the write is still in flight: nothing is
    # recorded, the caller has not left, and the lock is still held.
    assert not caller.done()
    assert moved == []
    assert _lock_is_held()

    gate.release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert gate.returned
    assert moved == [0.25]
    assert released_after == [True]
    assert not _lock_is_held()


async def test_a_raising_write_under_a_cancelled_caller_records_the_exception() -> None:
    error = ConnectionError("channel disconnected mid-put")
    gate = _Gate(error=error)
    moved: list[Any] = []

    caller = asyncio.create_task(guarded_call(gate.write, record=moved.append))
    await gate.started.wait()
    caller.cancel()
    await _settle()
    assert not caller.done()

    gate.release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert moved == [error]


async def test_a_caller_cancelled_again_still_waits_and_records_once() -> None:
    gate = _Gate(value=7)
    moved: list[Any] = []

    caller = asyncio.create_task(guarded_call(gate.write, record=moved.append))
    await gate.started.wait()
    for _ in range(3):
        caller.cancel()
        await _settle()
        assert not caller.done()

    gate.release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert moved == [7]


async def test_a_deadline_on_the_caller_waits_for_the_write_and_times_out_after() -> None:
    gate = _Gate(value=3)
    moved: list[Any] = []

    async def release_later() -> None:
        await asyncio.sleep(0.05)
        assert moved == []
        gate.release.set()

    releaser = asyncio.create_task(release_later())
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await guarded_call(gate.write, record=moved.append)
    await releaser
    assert moved == [3]


async def test_a_call_cancelled_on_its_own_records_the_cancellation() -> None:
    moved: list[Any] = []
    started = asyncio.Event()

    async def call() -> None:
        started.set()
        await asyncio.sleep(10)

    inner: list[asyncio.Task[Any]] = []

    def make() -> Any:
        task = asyncio.ensure_future(call())
        inner.append(task)
        return task

    caller = asyncio.create_task(guarded_call(make, record=moved.append))
    await started.wait()
    inner[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert len(moved) == 1
    assert isinstance(moved[0], asyncio.CancelledError)


async def test_a_failing_record_never_swallows_the_callers_cancellation() -> None:
    gate = _Gate()

    def record(outcome: Any) -> None:  # noqa: ARG001
        raise ValueError("record broke")

    caller = asyncio.create_task(guarded_call(gate.write, record=record))
    await gate.started.wait()
    caller.cancel()
    await _settle()
    gate.release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller


async def test_a_command_write_batch_cancelled_in_flight_keeps_the_lock_and_is_recorded() -> None:
    gate = _Gate()
    connector = _Connector(gate)
    moved: list[Any] = []

    async def action() -> None:
        with lock(None, context=ALICE_LIVE):
            await guarded_call(
                lambda: command_write(connector, [ARM, REQUEST], 1, commands=COMMANDS),
                record=moved.append,
            )

    caller = asyncio.create_task(action())
    await gate.started.wait()
    caller.cancel()
    await _settle()

    assert not caller.done()
    assert moved == []
    assert _lock_is_held()

    gate.release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert connector.calls == [[(ARM, 1), (REQUEST, 1)]]
    assert moved == [[ARM, REQUEST]]
    assert not _lock_is_held()


async def test_a_command_write_batch_returns_what_it_fired_and_records_it() -> None:
    gate = _Gate()
    gate.release.set()
    connector = _Connector(gate)
    moved: list[Any] = []

    with lock(None, context=ALICE_LIVE):
        fired = await guarded_call(
            lambda: command_write(connector, [ARM], 1, commands=COMMANDS),
            record=moved.append,
        )

    assert fired == [ARM]
    assert moved == [[ARM]]
