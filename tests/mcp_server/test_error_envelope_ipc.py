"""The error envelope must not be able to tell that a process boundary was crossed.

Once the control-system connector lives in a child process, every refusal an
operator reads has travelled down a pipe as a frame and been rebuilt on this
side. The claim these tests make is that the rebuilt exception is
indistinguishable from the one an in-process connector would have raised — not
"an error was reported", but the *same* class, carrying the *same* fields,
rendering the *same* agent-facing envelope, down to the allowed-range text the
operator is told to report.

Nothing is stubbed on the far side: every test here spawns the real
``python -m osprey_connectors.ipc._host``, drives it through the real
:class:`~osprey_connectors.ipc._proxy.ConnectorHostProxy`, and runs whatever
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

The last tests cover the other direction — the child dying, wedging, or
losing its connector rather than refusing — and pin that the supervisor in
:mod:`osprey.mcp_server.control_system.server_context` recovers once per child,
on the same target, while the envelope only reports what it did.
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

from osprey.mcp_server.control_system.connector_host_manager import SwitchError
from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey.mcp_server.errors import NothingAnsweredError
from osprey_connectors.errors import (
    ChannelLimitsViolationError,
    ChannelUnreachableError,
    ChannelWriteBlockedError,
    ConnectorUnavailableError,
)
from osprey_connectors.ipc import _frames as frames
from osprey_connectors.ipc._proxy import ConnectorHostProxy
from osprey_connectors_tests._harness import isolate_verdict_environment, write_control_context
from tests.facility.served_tree import in_process_config, served_tree
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

#: The repo root and the connectors suite's root are on the path too, so the
#: child can import a test-support connector by its dotted path.
PYTHONPATH = os.pathsep.join(
    [
        str(REPO_ROOT),
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "packages" / "osprey-connectors" / "src"),
        # The connectors suite's test-support connectors (osprey_connectors_tests).
        str(REPO_ROOT / "packages" / "osprey-connectors" / "tests"),
    ]
)

#: The raising connector, by dotted path, so ``live`` resolves to it verbatim.
RAISING_TYPE = "tests.mcp_server._raising_connector.RaisingConnector"

#: The mock connector, by dotted path, for the same reason.
IN_PROCESS_TYPE = "osprey_connectors.control_system._va_in_process_connector.VAInProcessConnector"

#: Generous enough that a loaded machine is not a failure, tight enough that a
#: wedged child fails this test rather than the run.
TIMEOUT_S = 20.0

#: The control target every child here serves, and the one whose switch is turned on.
CHILD_TARGET = "live"

#: The write tier that lets a write reach the connector. With anything else the
#: base class's guard returns a refused *result*, and neither the connector's own
#: raise nor the limits database is ever reached.
WRITES_ALLOWED = {"writes": "allowed"}

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


def _arm_child(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Arm the child's writes before it is spawned.

    The child's reference monitor asks the write verdict, which reads the login's
    record from the agent-data root in the environment the child inherits. So the
    verdict environment is isolated here (the record under *root*, the identity
    pinned, nothing stamped or bound), and an ownerless record with the switch on
    for :data:`CHILD_TARGET` is written before the spawn.
    """
    isolate_verdict_environment(monkeypatch, root)
    write_control_context(root, target=CHILD_TARGET, owned_by=None, switch={CHILD_TARGET: "on"})


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
        "osprey_connectors.ipc._host",
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
                "target": CHILD_TARGET,
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
        **WRITES_ALLOWED,
        "connector": {RAISING_TYPE: {}},
    }


@pytest.fixture
async def raising_pair(tmp_path, monkeypatch):
    """A child serving the connector that raises each refusal on cue, writes armed."""
    _arm_child(monkeypatch, tmp_path / "agent-data")
    control_system = _raising_control_system()
    config_file = _write_config(tmp_path, control_system)
    process, proxy = await _spawn(control_system, config_file, tmp_path)
    try:
        yield proxy
    finally:
        await _stop(process, proxy)


@pytest.fixture
async def limits_pair(tmp_path, monkeypatch):
    """A child running the real mock connector behind a real limits database, writes armed."""
    _arm_child(monkeypatch, tmp_path / "agent-data")
    limits_db = tmp_path / "channel_limits.json"
    limits_db.write_text(
        json.dumps({BOUNDED_CHANNEL: {"min_value": 0.0, "max_value": 100.0, "writable": True}})
    )
    control_system = {
        "type": IN_PROCESS_TYPE,
        **WRITES_ALLOWED,
        "limits_checking": {
            "enabled": True,
            "database_path": str(limits_db),
            "mode": "exclusive",
            "on_violation": "error",
        },
        "connector": {
            IN_PROCESS_TYPE: in_process_config(
                served_tree(tmp_path / "served", [BOUNDED_CHANNEL]), response_delay_ms=0
            )
        },
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
    """The child's refusal arrives as the very class the handler's ``except`` names.

    The handler's ``except`` clauses name the classes from ``osprey_connectors.errors``,
    and the child's refusal frame is rebuilt into an instance of that same class. If
    the rebuilt class were ever a different object, every refusal would fall through
    to ``internal_error`` — silently, and only across the process boundary.
    """
    with pytest.raises(ChannelLimitsViolationError):
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


# ------------------------------------------- a child that misses its deadline


#: The pool tests' misbehaving mock, by dotted path: ``SLOW:`` reads hang in a
#: child that still answers, ``WEDGE:`` reads block the child outright, ``BUG:``
#: reads raise an exception outside the IPC contract, ``GONE:`` reads find their
#: channel unreachable and ``LOST:`` reads find the connector unavailable.
SLOW_TYPE = "osprey_connectors_tests.ipc._pool_connectors.SlowInProcessConnector"

#: The deadline the manager's children are given here, lowered from the
#: reference value so a call that names no timeout misses it within the test.
TEST_DEADLINE_S = 1.0


@pytest.fixture
async def served_from_a_child(tmp_path, monkeypatch, make_manager):
    """A server context serving ``live`` from a real connector-host child.

    The deadline is lowered through the name the manager module imported, so
    the proxy the manager builds is the one under test. The context is the
    registry the tools resolve, so a call through its handle is supervised
    exactly as a tool's is.
    """
    from osprey.mcp_server.control_system import connector_host_manager
    from osprey.mcp_server.control_system import server_context as server_context_module

    monkeypatch.setenv("PYTHONPATH", PYTHONPATH)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.setattr(connector_host_manager, "DEFAULT_CALL_DEADLINE_S", TEST_DEADLINE_S)
    view = served_tree(
        tmp_path / "served", (), ("SLOW:X", "WEDGE:X", "BUG:X", "GONE:X", "LOST:X", "SR:A")
    )
    raw = {
        "control_system": {
            "type": SLOW_TYPE,
            "writes_enabled": False,
            "connector": {
                SLOW_TYPE: in_process_config(view, response_delay_ms=0),
                "virtual_accelerator": in_process_config(view, response_delay_ms=0),
            },
        }
    }
    manager = make_manager(raw=raw)
    await manager.ensure_started()
    context = server_context_module.ControlSystemContext()
    context._config = manager._config
    context._connector_hosts = manager
    context._switch_capable = True
    context._connectors["control_system"] = server_context_module.ConnectorEntry(
        config=raw["control_system"], connector_type="control_system"
    )
    monkeypatch.setattr(server_context_module, "_registry", context)
    return manager


async def _read_through_the_handle(channel: str, *, timeout: float | None = None) -> None:
    """One read as a tool makes it: through the supervised handle, inside the handler."""
    from osprey.mcp_server.control_system.server_context import get_server_context

    async with connector_error_handler("channel_read", "control_system"):
        connector = await get_server_context().control_system()
        await connector.read_channel(channel, timeout=timeout)


async def _timed_out_read(channel: str) -> dict:
    """The envelope a read naming no timeout renders once it misses the deadline."""
    with assert_raises_error(error_type="timeout_error") as captured:
        await _read_through_the_handle(channel)
    return captured["envelope"]


async def test_a_slow_child_that_still_answers_a_ping_is_kept(served_from_a_child):
    manager = served_from_a_child
    pid = manager.status()["child_pid"]

    envelope = await _timed_out_read("SLOW:X")

    assert "still answers a ping, so it was kept" in envelope["error_message"]
    assert envelope["details"]["recovery"][0]["action"] == "child_kept"
    assert manager.status()["child_pid"] == pid
    value = await manager.active_proxy().read_channel("SR:A", timeout=TIMEOUT_S)
    assert value is not None


async def test_a_wedged_child_that_answers_no_ping_is_replaced_on_the_same_target(
    served_from_a_child,
):
    manager = served_from_a_child
    pid = manager.status()["child_pid"]
    generation = manager.active_generation()

    envelope = await _timed_out_read("WEDGE:X")

    assert "did not answer a ping and has been replaced" in envelope["error_message"]
    assert envelope["details"]["recovery"][0]["action"] == "child_replaced"
    assert manager.status()["child_pid"] != pid
    assert manager.active_generation() == generation
    assert manager.active_target() == "live"
    value = await manager.active_proxy().read_channel("SR:A", timeout=TIMEOUT_S)
    assert value is not None


async def test_an_unknown_read_in_a_child_is_not_found_and_keeps_the_child(served_from_a_child):
    manager = served_from_a_child
    proxy = manager.active_proxy()
    pid = manager.status()["child_pid"]

    with assert_raises_error(error_type="not_found") as captured:
        async with connector_error_handler("channel_read", "control_system"):
            await proxy.read_channel("NOT:SERVED", timeout=TIMEOUT_S)

    assert captured["envelope"]["details"]["channel"] == "NOT:SERVED"
    assert manager.active_proxy() is proxy
    assert manager.status()["child_pid"] == pid
    value = await manager.active_proxy().read_channel("SR:A", timeout=TIMEOUT_S)
    assert value is not None


async def test_a_connector_bug_in_a_child_is_an_internal_error_and_keeps_the_child(
    served_from_a_child,
):
    manager = served_from_a_child
    proxy = manager.active_proxy()
    pid = manager.status()["child_pid"]

    with assert_raises_error(error_type="internal_error") as captured:
        async with connector_error_handler("channel_read", "control_system"):
            await proxy.read_channel("BUG:X", timeout=TIMEOUT_S)

    envelope = captured["envelope"]
    assert "KeyError" in envelope["error_message"]
    assert "recovery" not in (envelope.get("details") or {})
    assert manager.active_proxy() is proxy
    assert manager.status()["child_pid"] == pid


async def test_one_unreachable_channel_keeps_the_child(served_from_a_child):
    """A dead IOC is that channel's failure: reported against it, nothing restarted."""
    manager = served_from_a_child
    pid = manager.status()["child_pid"]
    spawned = len(manager.spawned)

    with assert_raises_error(error_type="channel_unreachable") as captured:
        await _read_through_the_handle("GONE:X", timeout=TIMEOUT_S)

    envelope = captured["envelope"]
    assert envelope["details"]["channel"] == "GONE:X"
    assert "recovery" not in envelope["details"]
    assert manager.status()["child_pid"] == pid
    assert len(manager.spawned) == spawned
    value = await manager.active_proxy().read_channel("SR:A", timeout=TIMEOUT_S)
    assert value is not None


async def test_an_unavailable_connector_is_replaced_once_for_a_whole_batch(served_from_a_child):
    """Twenty reads that find the connector gone replace its child once, and each says so."""
    from osprey.mcp_server.control_system.server_context import get_server_context

    manager = served_from_a_child
    pid = manager.status()["child_pid"]
    spawned = len(manager.spawned)
    connector = await get_server_context().control_system()

    failures = await asyncio.gather(
        *(connector.read_channel("LOST:X", timeout=TIMEOUT_S) for _ in range(20)),
        return_exceptions=True,
    )

    assert all(isinstance(failure, ConnectorUnavailableError) for failure in failures)
    assert {failure.recovery[0]["action"] for failure in failures} == {"child_replaced"}
    assert len(manager.spawned) == spawned + 1
    assert manager.status()["child_pid"] != pid
    assert manager.active_target() == "live"

    with assert_raises_error(error_type="nothing_answered") as captured:
        async with connector_error_handler("channel_read", "control_system"):
            raise NothingAnsweredError(
                None,
                missing={f"LOST:{n}": str(failure) for n, failure in enumerate(failures[:2])},
                errors=list(failures[:2]),
            )
    recovery = captured["envelope"]["details"]["recovery"]
    assert [record["action"] for record in recovery] == ["child_replaced"]


async def test_a_wedged_child_whose_respawn_is_refused_reports_respawn_refused(
    served_from_a_child, monkeypatch
):
    """A ping that fails and a replacement the manager refuses: said so, tried once."""
    manager = served_from_a_child
    attempts: list[object] = []

    async def refuse(proxy):
        attempts.append(proxy)
        raise SwitchError("live", "spawn", "spawn_failed", "the fresh child would not start")

    monkeypatch.setattr(manager, "replace_child", refuse)

    envelope = await _timed_out_read("WEDGE:X")

    assert envelope["details"]["recovery"][0]["action"] == "respawn_refused"
    assert "was refused" in envelope["error_message"]
    assert len(attempts) == 1


async def test_a_wedged_child_another_actor_already_replaced_is_reported_without_a_target(
    served_from_a_child, monkeypatch
):
    """A child replaced elsewhere first: the record says it is replaced, not where or by whom."""
    manager = served_from_a_child
    attempts: list[object] = []

    async def already_replaced(proxy):
        attempts.append(proxy)
        return False

    monkeypatch.setattr(manager, "replace_child", already_replaced)

    envelope = await _timed_out_read("WEDGE:X")

    record = envelope["details"]["recovery"][0]
    assert record["action"] == "child_replaced"
    assert "has been replaced" in record["text"]
    assert "same target" not in record["text"]
    assert len(attempts) == 1


async def test_a_dead_child_is_replaced_by_its_supervisor_and_the_envelope_reports_it(
    served_from_a_child,
):
    """A call on the wire when the child dies fails once; the child is replaced once."""
    from osprey.mcp_server.control_system.server_context import get_server_context

    manager = served_from_a_child
    pid = manager.status()["child_pid"]
    connector = await get_server_context().control_system()

    call = asyncio.ensure_future(connector.read_channel("SLOW:X", timeout=TIMEOUT_S))
    await asyncio.sleep(0.3)
    os.kill(pid, 9)

    with pytest.raises(ConnectorUnavailableError) as raised:
        await asyncio.wait_for(call, TIMEOUT_S)
    assert [record["action"] for record in raised.value.recovery] == ["child_replaced"]

    with assert_raises_error(error_type="connection_error") as captured:
        async with connector_error_handler("channel_read", "control_system"):
            raise raised.value

    envelope = captured["envelope"]
    assert "connector-host child" in envelope["error_message"]
    assert [record["action"] for record in envelope["details"]["recovery"]] == ["child_replaced"]
    assert manager.status()["child_pid"] != pid
    assert manager.active_target() == "live"
    assert not isinstance(raised.value, ChannelUnreachableError)
