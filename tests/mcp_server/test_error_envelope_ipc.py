"""The error envelope must not be able to tell that a process boundary was crossed.

Once the control-system connector lives in a child process, every refusal an
operator reads has travelled down a pipe as a frame and been rebuilt on this
side. The claim these tests make is that the rebuilt exception is
indistinguishable from the one an in-process connector would have raised — not
"an error was reported", but the *same* class, carrying the *same* fields,
rendering the *same* agent-facing envelope, down to the allowed-range text the
operator is told to report.

Nothing is stubbed on the far side: every test here spawns the real
``python -m osprey_connectors.ipc.host``, drives it through the real
:class:`~osprey_connectors.ipc.proxy.ConnectorHostProxy`, and runs whatever
comes back through the real ``connector_error_handler``. Two connectors are
used, for two different claims:

- :mod:`tests.mcp_server._raising_connector`, reached by dotted path, raises
  each refusal class with *every* field populated — including ``max_step`` and
  ``current_value``, which no naturally occurring violation sets without an
  EPICS readback — so field fidelity is tested exhaustively rather than
  wherever the defaults happen to land.
- The real mock connector behind a real limits database, so the same claim
  holds for a violation nobody wrote by hand: config in, ``LimitsValidator``
  refusal out, envelope on this side.

The last tests cover the other direction — the child dying rather than
refusing — and pin that it routes through
``error_handling.invalidate_active_connector``, the one seam the connector-host
manager re-points so that a lost child means "respawn for the same target".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from osprey.mcp_server.control_system import error_handling
from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
from osprey_connectors.ipc import frames
from osprey_connectors.ipc.proxy import ConnectorHostProxy
from tests.mcp_server._raising_connector import (
    BLOCKED_CHANNEL,
    BLOCKED_MESSAGE,
    BLOCKED_REASON,
    LIMITS_CHANNEL,
    LIMITS_FIELDS,
    READ_VALUE,
    make_limits_violation,
    make_write_blocked,
)
from tests.mcp_server.conftest import assert_raises_error

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The repo root is on the path too, so the child can import the test-support
#: connector by its dotted ``tests.`` path.
PYTHONPATH = os.pathsep.join(
    [
        str(REPO_ROOT),
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "packages" / "osprey-connectors" / "src"),
    ]
)

#: The raising connector, by dotted path, so ``live`` resolves to it verbatim.
RAISING_TYPE = "tests.mcp_server._raising_connector.RaisingConnector"

#: The mock connector, by dotted path, for the same reason.
MOCK_TYPE = "osprey_connectors.control_system.mock_connector.MockConnector"

#: Generous enough that a loaded machine is not a failure, tight enough that a
#: wedged child fails this test rather than the run.
TIMEOUT_S = 20.0

#: The channel the limits database below bounds to ``[0.0, 100.0]``.
BOUNDED_CHANNEL = "TEST:PV:SETPOINT"


# --------------------------------------------------------------- the harness


async def _next_frame(stream):
    """The next whole frame on ``stream``, reassembled from the raw pipe.

    Used only for the init handshake, which happens before the proxy takes
    ownership of the pipe and starts its own reader task.
    """
    parser = frames.FrameReader()
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            raise AssertionError("the connector host closed its frame channel during init")
        for frame in parser.feed(chunk):
            return frame


async def _spawn(control_system: dict, config_file: Path, cwd: Path):
    """Start a real child, complete its init handshake, and wrap it in a proxy.

    Returns ``(process, proxy)``. The post-connect report is asserted to be a
    result frame rather than an error, so a child that failed to build its
    connector fails here with the child's own message instead of surfacing
    later as an unrelated timeout.
    """
    env = {key: value for key, value in os.environ.items() if key != "CONFIG_FILE"}
    env["PYTHONPATH"] = PYTHONPATH
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "osprey_connectors.ipc.host",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        env=env,
    )

    request_id = frames.new_request_id()
    process.stdin.write(
        frames.encode_request(
            request_id,
            "init",
            {
                "control_system": control_system,
                "target": "live",
                "config_file": str(config_file),
            },
        )
    )
    await process.stdin.drain()

    frame = await asyncio.wait_for(_next_frame(process.stdout), TIMEOUT_S)
    assert isinstance(frame, frames.ResultFrame), f"the child refused to start: {frame!r}"
    assert frame.request_id == request_id
    return process, ConnectorHostProxy.from_process(process)


async def _stop(process, proxy) -> None:
    """Tear the pair down without letting teardown mask a test failure."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proxy.disconnect(), TIMEOUT_S)
    if process.returncode is None:
        process.kill()
    await asyncio.wait_for(process.wait(), TIMEOUT_S)


def _write_config(tmp_path: Path, control_system: dict) -> Path:
    """A project config the child reads its write posture and limits from."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"control_system": control_system}))
    return config_file


def _raising_control_system() -> dict:
    return {
        "type": RAISING_TYPE,
        # The base class's guard returns a blocked *result* when writes are
        # off, and the connector's own raise would never be reached.
        "writes_enabled": True,
        "connector": {RAISING_TYPE: {}},
    }


@pytest.fixture
async def raising_pair(tmp_path):
    """A child serving the connector that raises each refusal on cue."""
    control_system = _raising_control_system()
    config_file = _write_config(tmp_path, control_system)
    process, proxy = await _spawn(control_system, config_file, tmp_path)
    try:
        yield proxy
    finally:
        await _stop(process, proxy)


@pytest.fixture
async def limits_pair(tmp_path):
    """A child running the real mock connector behind a real limits database."""
    limits_db = tmp_path / "channel_limits.json"
    limits_db.write_text(
        json.dumps({BOUNDED_CHANNEL: {"min_value": 0.0, "max_value": 100.0, "writable": True}})
    )
    control_system = {
        "type": MOCK_TYPE,
        "writes_enabled": True,
        "limits_checking": {
            "enabled": True,
            "database_path": str(limits_db),
            "allow_unlisted_channels": False,
            "on_violation": "error",
        },
        "connector": {MOCK_TYPE: {"response_delay_ms": 0, "noise_level": 0.0}},
    }
    config_file = _write_config(tmp_path, control_system)
    process, proxy = await _spawn(control_system, config_file, tmp_path)
    try:
        yield proxy
    finally:
        await _stop(process, proxy)


async def _envelope_for(exc: BaseException, tool_name: str = "channel_write") -> dict:
    """The agent-facing envelope ``connector_error_handler`` renders for ``exc``."""
    with assert_raises_error() as captured:
        async with connector_error_handler(tool_name):
            raise exc
    return captured["envelope"]


# ------------------------------------------------- field fidelity (CC-1 (a))


async def test_a_limits_violation_arrives_as_the_real_class_with_every_field(raising_pair):
    """The proxied exception is the class itself, not a stand-in carrying text."""
    with pytest.raises(ChannelLimitsViolationError) as exc_info:
        await raising_pair.write_channel(LIMITS_CHANNEL, LIMITS_FIELDS["attempted_value"])

    exc = exc_info.value
    # Identity, not merely isinstance: a subclass or a look-alike would be a
    # different thing arriving at the envelope's isinstance branches.
    assert type(exc) is ChannelLimitsViolationError
    assert {name: getattr(exc, name) for name in LIMITS_FIELDS} == LIMITS_FIELDS


async def test_the_envelopes_isinstance_branch_fires_on_the_reconstructed_class(raising_pair):
    """``osprey.errors`` is an alias of the connectors module, and it has to stay one.

    The handler's ``except`` clauses name the classes through ``osprey.errors``;
    the child rebuilt this instance from ``osprey_connectors.errors``. If those
    ever stopped being the same module object, every refusal would fall through
    to ``internal_error`` — silently, and only across the process boundary.
    """
    from osprey.errors import ChannelLimitsViolationError as ShimLimitsError

    with pytest.raises(ShimLimitsError):
        await raising_pair.write_channel(LIMITS_CHANNEL, LIMITS_FIELDS["attempted_value"])


async def test_a_write_refusal_arrives_with_its_reason_and_its_own_message(raising_pair):
    """A custom refusal message survives verbatim, rather than being re-rendered."""
    with pytest.raises(ChannelWriteBlockedError) as exc_info:
        await raising_pair.write_channel(BLOCKED_CHANNEL, 1.0)

    exc = exc_info.value
    assert exc.channel_address == BLOCKED_CHANNEL
    assert exc.reason == BLOCKED_REASON
    assert str(exc) == BLOCKED_MESSAGE


# -------------------------------------------- envelope equivalence (CC-1 (b))


async def test_the_limits_envelope_is_identical_to_the_in_process_one(raising_pair):
    """Same envelope, key for key, whichever side of the pipe raised it."""
    with pytest.raises(ChannelLimitsViolationError) as exc_info:
        await raising_pair.write_channel(LIMITS_CHANNEL, LIMITS_FIELDS["attempted_value"])

    proxied = await _envelope_for(exc_info.value)
    in_process = await _envelope_for(make_limits_violation())

    assert proxied == in_process
    assert proxied["error_type"] == "limits_violation"


async def test_the_limits_envelope_tells_the_operator_the_allowed_range(raising_pair):
    """The range is what the operator is told to report, so it is asserted verbatim."""
    with pytest.raises(ChannelLimitsViolationError) as exc_info:
        await raising_pair.write_channel(LIMITS_CHANNEL, LIMITS_FIELDS["attempted_value"])

    envelope = await _envelope_for(exc_info.value)

    assert envelope["error_message"] == (
        f"Channel limits violated during channel_write: {LIMITS_FIELDS['violation_reason']} "
        "(allowed range: [-25.0, 100.0])"
    )
    # The structured half carries every field the class does — including the
    # two that only an explicit step check ever populates.
    assert envelope["details"] == {
        "channel": LIMITS_CHANNEL,
        "attempted_value": LIMITS_FIELDS["attempted_value"],
        "violation_type": LIMITS_FIELDS["violation_type"],
        "reason": LIMITS_FIELDS["violation_reason"],
        "min_value": LIMITS_FIELDS["min_value"],
        "max_value": LIMITS_FIELDS["max_value"],
        "max_step": LIMITS_FIELDS["max_step"],
        "current_value": LIMITS_FIELDS["current_value"],
    }
    assert "Report the violation to the operator with the allowed range." in envelope["suggestions"]


async def test_the_refusal_envelope_is_identical_to_the_in_process_one(raising_pair):
    with pytest.raises(ChannelWriteBlockedError) as exc_info:
        await raising_pair.write_channel(BLOCKED_CHANNEL, 1.0)

    proxied = await _envelope_for(exc_info.value)
    in_process = await _envelope_for(make_write_blocked())

    assert proxied == in_process
    assert proxied["error_type"] == "write_refused"
    assert proxied["details"] == {"channel": BLOCKED_CHANNEL, "reason": BLOCKED_REASON}
    assert BLOCKED_MESSAGE in proxied["error_message"]


# ------------------------------------ the same claim, via a real refusal path


async def test_a_real_limits_database_refusal_renders_its_range_across_the_boundary(limits_pair):
    """Nobody constructed this exception: config went in, a refusal came back.

    The mock connector's own ``LimitsValidator`` raises it inside the child,
    which is the path a deployment actually takes, and the range the operator
    reads on this side is the one the database configured.
    """
    with pytest.raises(ChannelLimitsViolationError) as exc_info:
        await limits_pair.write_channel(BOUNDED_CHANNEL, 150.0)

    exc = exc_info.value
    assert exc.channel_address == BOUNDED_CHANNEL
    assert exc.attempted_value == 150.0
    assert exc.violation_type == "MAX_EXCEEDED"
    assert (exc.min_value, exc.max_value) == (0.0, 100.0)

    envelope = await _envelope_for(exc)

    assert envelope["error_type"] == "limits_violation"
    assert "(allowed range: [0.0, 100.0])" in envelope["error_message"]
    assert envelope["details"]["violation_type"] == "MAX_EXCEEDED"
    # Fields the database does not configure stay absent rather than arriving
    # as nulls an agent would have to interpret.
    assert "max_step" not in envelope["details"]
    assert "current_value" not in envelope["details"]


async def test_a_refusal_does_not_end_the_child(raising_pair):
    """A refused write is an answer, not a casualty: the next call still works."""
    with pytest.raises(ChannelLimitsViolationError):
        await raising_pair.write_channel(LIMITS_CHANNEL, LIMITS_FIELDS["attempted_value"])

    value = await raising_pair.read_channel("TEST:ANY:CHANNEL")

    assert value.value == pytest.approx(READ_VALUE)


# -------------------------------------------- the invalidate seam (CC-1 (c))


async def test_a_dead_child_routes_through_the_invalidate_seam(tmp_path, monkeypatch):
    """Child death reaches the envelope as a connection error, via the one seam.

    The seam is what the connector-host manager re-points so that a lost child
    means "respawn for the same target, no generation bump". Monkeypatching it
    here is how this test proves the branch goes through exactly one call site
    rather than reaching into the server context inline.
    """
    invalidated: list[str] = []

    async def _record(connector_name: str) -> None:
        invalidated.append(connector_name)

    monkeypatch.setattr(error_handling, "invalidate_active_connector", _record)

    control_system = _raising_control_system()
    config_file = _write_config(tmp_path, control_system)
    process, proxy = await _spawn(control_system, config_file, tmp_path)
    try:
        # A call already on the wire when the child is killed outright: no
        # refusal frame, no goodbye — the failure mode a respawn has to answer
        # for, rather than a clean disconnect.
        call = asyncio.ensure_future(proxy.read_channel("TEST:ANY:CHANNEL"))
        await asyncio.sleep(0)
        process.kill()
        await asyncio.wait_for(process.wait(), TIMEOUT_S)

        with pytest.raises(ConnectionError) as exc_info:
            await asyncio.wait_for(call, TIMEOUT_S)
    finally:
        await _stop(process, proxy)

    with assert_raises_error(error_type="connection_error") as captured:
        async with connector_error_handler("channel_read", "control_system"):
            raise exc_info.value

    assert invalidated == ["control_system"]
    assert "connector-host child" in captured["envelope"]["error_message"]


async def test_the_seams_default_behaviour_is_still_a_context_invalidation(monkeypatch):
    """The seam must actually do today what the branch did inline before it."""
    from osprey.mcp_server.control_system import server_context as server_context_module

    invalidated: list[str] = []

    class _Context:
        async def invalidate_connector(self, name):
            invalidated.append(name)

    monkeypatch.setattr(server_context_module, "get_server_context", _Context)

    await error_handling.invalidate_active_connector("control_system")

    assert invalidated == ["control_system"]
