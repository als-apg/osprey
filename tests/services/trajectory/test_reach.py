"""Reachability is one live read of ``injection_busy``, asked anew on every call.

The connector here is a stand-in that records each read and answers, raises or
hangs as the test sets it; everything else is the real started-service shape.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from osprey.services.trajectory.reach import (
    PROBE_TIMEOUT_S,
    TARGET_UNREACHABLE,
    TRAJECTORY_UNAVAILABLE,
    probe,
)
from osprey.services.trajectory.startup import (
    TrajectoryCommands,
    TrajectoryDevice,
    TrajectoryStartup,
)
from osprey_connectors.errors import ChannelReadFailedError

BUSY = "SR:INJ:BUSY"
COMMANDS = TrajectoryCommands(
    model="SR",
    devices=tuple(
        TrajectoryDevice(
            name=f"SR/BPM{n}",
            x_row=f"SR:BPM{n}:X",
            y_row=f"SR:BPM{n}:Y",
            sum_row=f"SR:BPM{n}:SUM",
            arm="SR:BPM:ARM",
            armed="SR:BPM:ARMED",
        )
        for n in (1, 2, 3)
    ),
    injection_request="SR:INJ:REQ",
    injection_busy=BUSY,
    trigger_sample=10,
)


class _Connector:
    """Answers each read with the next outcome: a value, an exception, or ``"hang"``."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.reads: list[tuple[str, float | None]] = []

    async def read_channel(self, address: str, timeout: float | None = None) -> Any:
        self.reads.append((address, timeout))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if outcome == "hang":
            await asyncio.sleep(3600)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def write_channel(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("reachability never writes")


def _started(connector: Any, commands: TrajectoryCommands | None = COMMANDS) -> TrajectoryStartup:
    return TrajectoryStartup(
        target="live", connector=connector, commands=commands, guarded_run_dir=Path("/unused")
    )


async def test_a_target_that_answers_is_reachable() -> None:
    connector = _Connector(0)
    assert await probe(_started(connector)) is None
    assert connector.reads == [(BUSY, PROBE_TIMEOUT_S)]


async def test_the_read_is_of_the_busy_flag_with_a_short_timeout() -> None:
    connector = _Connector(1)
    assert await probe(_started(connector), timeout=0.25) is None
    assert connector.reads == [(BUSY, 0.25)]
    assert PROBE_TIMEOUT_S <= 5


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("no answer"),
        ConnectionError("no server"),
        ChannelReadFailedError([BUSY], "disconnected"),
        RuntimeError("anything else"),
    ],
    ids=["timeout", "connection", "read-failed", "other"],
)
async def test_a_failed_read_is_target_unreachable(failure: BaseException) -> None:
    assert await probe(_started(_Connector(failure))) == TARGET_UNREACHABLE


async def test_a_read_that_never_comes_back_is_target_unreachable() -> None:
    connector = _Connector("hang")
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    assert await probe(_started(connector), timeout=0.05) == TARGET_UNREACHABLE
    assert loop.time() - started_at < 2.0


async def test_no_model_is_trajectory_unavailable_and_reads_nothing() -> None:
    connector = _Connector(0)
    assert await probe(_started(connector, commands=None)) == TRAJECTORY_UNAVAILABLE
    assert connector.reads == []


async def test_nothing_is_latched_between_calls() -> None:
    connector = _Connector(ConnectionError("down"), 0, ConnectionError("down again"))
    started = _started(connector)
    assert await probe(started) == TARGET_UNREACHABLE
    assert await probe(started) is None
    assert await probe(started) == TARGET_UNREACHABLE
    assert len(connector.reads) == 3


async def test_cancelling_the_caller_is_not_swallowed() -> None:
    connector = _Connector("hang")
    task = asyncio.ensure_future(probe(_started(connector), timeout=60))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
