"""``channel_write`` binds itself to the target it was approved on.

The approval prompt names a target and a generation, read out of the
deployment's control-context record. Between that prompt and the moment the
write reaches the control system, a target switch can land — and a value
approved for the simulator must never be applied to the real machine because it
arrived a second late.

The tool therefore observes ``(target, generation)`` three times — as the
approval prompt rendered it (from the stamp the hook leaves behind), at entry,
and immediately before the connector call — and refuses on any difference.
These tests drive all three windows:

* the approval window with a stamp file written before the call, which is
  exactly what the hook leaves on disk;
* the execution window by rewriting the record inside the connector resolution,
  a real seam of the tool's own execution path that runs after the entry
  capture and before the pre-write re-read — precisely where a switch
  completing in another task would land;
* the published-versus-serving disagreement by giving the connector-host
  manager a started child on a binding the record does not name.

A fourth refusal is not about this call's own window at all: the record says
where the deployment is, and this server's report says where its connector host
actually got to. While a live server is mid-switch, or while this session's own
server failed to follow one, the write is refused naming the PIDs — the
:func:`~osprey_connectors.control_context.converged` verdict, asked here
exactly as the executor and the notebook kernel ask it. A failed swap strands
the one session whose server could not follow and leaves every other session,
kernel and bare ``claude`` writing (FR-8), so those tests come in pairs: the
refusal, and the neighbour that must not be refused with it.

Every assertion is made on the shipped envelope (the parsed tool result, or the
structured error a refusal raises), never on an internal helper: the envelope is
the only thing an agent or an operator ever sees.

Every call here runs under a record with both switches on and the ask stamp the
approval hook files, so the tool's stage 1 lets it through to the binding
checks. Approval asks here, so a call also needs the prompt's target stamp:
``_run_single`` files it on the binding the call enters on, as the hook would,
and the approval-window tests file their own or none. A call that enters with no usable record never gets that far: the write
verdict refuses it at entry, and the first section pins that.
"""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from osprey.audit.posture import POSTURE_SESSION_ENV_VAR
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import ConnectorHostManager
from osprey.mcp_server.control_system.server_context import initialize_server_context
from osprey.mcp_server.control_system.tools import channel_write as channel_write_module
from osprey_connectors import control_context, posture_store
from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    state_dir_under,
    write_control_context,
    write_server_report,
)
from tests._approval_stamp_fixture import ApprovalStamps
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)

#: The shipped envelope's shape, pinned so that binding a write to a target
#: cannot smuggle a field into the payload an agent reads. A refusal is an error
#: envelope; a write that proceeds must look exactly as it did before.
EXPECTED_TOP_LEVEL_KEYS = {"status", "description", "summary", "access_details"}
EXPECTED_SUMMARY_KEYS = {"total_writes", "outcomes", "results"}
EXPECTED_RESULT_KEYS = {
    "channel",
    "value",
    "outcome",
    "refusal_reason",
    "error",
    "observed_value",
    "alarm_status",
    "alarm_severity",
    "notes",
}

#: The session stamp these tests run this server under, wherever the identity
#: of the reader matters. A report carrying it is this server's own; a report
#: carrying :data:`OTHER_SESSION` belongs to a second window on the same
#: deployment, and the two are what FR-8's "strands one session" is measured
#: with.
THIS_SESSION = "web:this-session"
OTHER_SESSION = "web:other-session"

#: Display metadata as the server's own report records it. Irrelevant to the
#: binding — which is the target and the generation only — but written anyway so
#: the fixture reports are the shape a reader really meets.
_TARGETS_META = {
    "live": {"label": "LIVE MACHINE", "endpoint": "gateway.example.com:5064", "real_machine": True},
    "va": {
        "label": "virtual accelerator (simulation)",
        "endpoint": "localhost:5074",
        "real_machine": False,
    },
}


def _get_channel_write():
    from osprey.mcp_server.control_system.tools.channel_write import channel_write

    return get_tool_fn(channel_write)


#: The project these tests write from: the simulator in process beside a named
#: real machine, so both ``va`` and ``live`` are targets here, under a write
#: tier that allows.
_CONFIG = (
    "control_system:\n  type: virtual_accelerator\n  writes: allowed\n  connector:\n"
    "    virtual_accelerator:\n      serving: in_process\n"
    "    epics:\n      prefix: 'RING:'\n"
)

#: Both machines' switches armed: the write verdict lets every call here reach the
#: binding checks, whichever target its record names.
_SWITCH_ARMED = {"va": "asked", "live": "asked"}

#: Both machines at the no-prompt position, for the paths that never ask.
_SWITCH_NO_PROMPT = {"va": "allowed", "live": "allowed"}


#: An approval section whose policy lets ``channel_write`` run without asking.
_APPROVAL_SKIPS = "approval:\n  enabled: true\n  tools:\n    channel_write: skip\n"


def _prepare(tmp_path, monkeypatch, *, session=None, config=_CONFIG):
    """Project, server context, and a state root nothing else writes into.

    The agent-data root is stamped on the environment rather than patched into
    one module, because two readers resolve it here — the record library and
    ``target_state``, which still owns the approval stamps — and a test that
    pointed only one of them at ``tmp_path`` would silently read the developer's
    own deployment for the other. ``resolve_shared_data_root`` is patched as
    well so the unstamped derivation lands in the same place.

    *session* is this process's ``OSPREY_POSTURE_SESSION``: the identity
    :func:`~osprey_connectors.control_context.blocking_pids` matches a report
    against. ``None`` — the default — is a bare ``claude``, which owns no
    report and is therefore held up only by a swap in flight.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(config)
    root = tmp_path / "var" / "agent_data"
    isolate_verdict_environment(monkeypatch, root)
    state_dir_under(root).mkdir(parents=True, exist_ok=True)
    if session is None:
        monkeypatch.delenv(POSTURE_SESSION_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(POSTURE_SESSION_ENV_VAR, session)
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
    # The real machine is named only so the write verdict can answer for a
    # record on ``live``. Serving stays in process, as for a simulator-only
    # deployment: the binding checks are the subject here, and a connector-host
    # child would need a built simulator view to start at all.
    from osprey.mcp_server.control_system import server_context

    monkeypatch.setattr(server_context, "switch_capable", lambda _raw: False)
    control_context.invalidate_cache()
    initialize_server_context()
    return root


def _root():
    """The agent-data root :func:`_prepare` stamped.

    The shared writers take the root explicitly — a suite has to be able to lay
    a deployment down before it points the environment at one — while every
    test here has already stamped it. Reading the stamp back through the store
    keeps the root from being spelled twice in one file, and answers the ROOT
    rather than the identity's directory under it: every shared writer appends
    the ``control_target/<identity>`` hops itself, so a directory that already
    carries them names a file no reader ever resolves.
    """
    return posture_store.agent_data_root()


def _write_record(target, generation, **fields):
    """Write the deployment's record, as the owner writes it after a switch.

    Both switches are on unless *fields* says otherwise.
    """
    fields.setdefault("switch", _SWITCH_ARMED)
    return write_control_context(_root(), target, generation, **fields)


def _write_report(server_pid=None, **fields):
    """File one controls server's report, as that server publishes it.

    Defaults to *this* process, which is the server these tests are running
    inside: a report filed under any other PID is another window's, and the
    session rules turn on exactly that difference.
    """
    pid = os.getpid() if server_pid is None else server_pid
    return write_server_report(_root(), pid, targets=_TARGETS_META, **fields)


def _iso(offset_s=0.0):
    """An ISO-8601 instant *offset_s* seconds from now, as a server stamps one."""
    return (datetime.now(UTC) + timedelta(seconds=offset_s)).isoformat()


def _applying(generation, *, expires_in_s=60.0):
    """An ``applying`` switch block for *generation*, bounded from now.

    A block alone does not say the server holds a connector; the binding
    beside it does.

    A negative *expires_in_s* is the stuck server: the block is past the bound
    its own publisher wrote, which stops holding up the deployment and holds up
    only the session whose server it is.
    """
    return {
        "status": control_context.REPORT_APPLYING,
        "generation": generation,
        "at": _iso(),
        "expires_at": _iso(expires_in_s),
    }


def _failed(generation):
    """A ``failed`` switch block for *generation* — a server that could not follow."""
    return {
        "status": control_context.REPORT_FAILED,
        "generation": generation,
        "at": _iso(),
    }


def _live_other_pid():
    """A PID that is running and is not this process.

    The liveness filter in :func:`~osprey_connectors.control_context.live_reports`
    binds its predicate as a default argument, so a report belonging to
    "another server" has to name a process that really exists rather than one
    a patch pretends into being.
    """
    parent = os.getppid()
    if parent > 0 and parent != os.getpid() and control_context.is_process_alive(parent):
        return parent
    pytest.skip("no second live PID available to stand in for another controls server")
    return None  # pragma: no cover - skip does not return


def _dead_pid():
    """A PID that has certainly exited: a child run to completion and reaped."""
    proc = subprocess.Popen([sys.executable, "-c", ""])  # fixed argv
    proc.wait(timeout=60)
    return proc.pid


def _write_result(channel="TEST:PV", value=42.0):
    """A confirmed write, as a connector really returns one.

    A real ``ChannelWriteResult`` rather than a ``MagicMock``: a mock answers
    every attribute truthily, so a projection reading a field no connector
    populates would pass here and misreport in the field — and this file's
    whole point is that the envelope an operator sees is exactly what it was.
    """
    return ChannelWriteResult(
        channel_address=channel,
        value_written=value,
        outcome=WriteOutcome.CONFIRMED,
        observed_value=value,
    )


#: ``_stamp_approval``'s default session: whichever one this process is running
#: under, which is what the approval hook would have stamped here.
_OUR_SESSION = object()


def _stamp_approval(operations, *, target, generation, confirm=None, session=_OUR_SESSION):
    """Leave the stamp a rendered ``channel_write`` approval leaves behind.

    Written the way the hook writes it — same directory, same fields — but
    through the tool's own name and key derivations, because what this file
    tests is the comparison. That the hook derives the SAME name from the same
    payload and session is pinned on the hook's side, in ``tests/hooks``.

    *session* is the audit session the prompt was rendered under, and it goes
    into the payload only: a stamp naming another session is filed here under
    THIS one's name, which is the misfiling the payload check exists to catch.
    The name keeps two ordinary sessions apart on its own, and that separation
    is the hook suite's to pin.
    """
    key = channel_write_module._approval_stamp_key(operations, confirm)
    directory = target_state.state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / channel_write_module._approval_stamp_name(key)
    path.write_text(
        json.dumps(
            {
                "tool": "channel_write",
                "key": key,
                "target": target,
                "generation": generation,
                "session": (
                    os.environ.get(POSTURE_SESSION_ENV_VAR) or None
                    if session is _OUR_SESSION
                    else session
                ),
                "rendered_at": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return path


@contextmanager
def _patched(connector, *, when_resolved=None):
    """Serve *connector* to the tool, optionally moving the target on the way.

    ``when_resolved`` runs while the tool is resolving its connector — after the
    entry capture, before the pre-write re-read. That is the window a switch
    landing in another task occupies, and driving it here needs no threads: the
    tool's own control flow passes through this seam.
    """

    async def _create(_config, *, control_target=None):  # noqa: ARG001 - ConnectorFactory.create_control_system_connector fixes this keyword-only parameter
        if when_resolved is not None:
            when_resolved()
        return connector

    with (
        patch(
            "osprey_connectors.factory.ConnectorFactory.create_control_system_connector",
            new=_create,
        ),
        patch(
            "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
        ApprovalStamps(target_state.state_dir()).approved("channel_write"),
    ):
        yield


def _stamp_current(operations):
    """The prompt's target stamp for *operations*, on the binding the deployment is on now."""
    binding = channel_write_module._read_target_binding()
    target, generation = binding if binding is not None else (None, None)
    return _stamp_approval(operations, target=target, generation=generation)


async def _run_single(connector, *, when_resolved=None, channel="TEST:PV", value=42.0, stamp=True):
    """One write; *stamp* files its target stamp first, rendered on the current binding."""
    operations = [{"channel": channel, "value": value}]
    if stamp:
        _stamp_current(operations)
    with _patched(connector, when_resolved=when_resolved):
        fn = _get_channel_write()
        return await fn(operations=operations)


# ---------------------------------------------------------------------------
# The binding does not exist: a deployment with no published target
# ---------------------------------------------------------------------------


async def test_without_target_state_the_verdict_refuses_before_any_binding(tmp_path, monkeypatch):
    """No record at all: every switch is off, so the write verdict refuses at entry.

    There is no binding to compare, and none is needed: the deployment's
    baseline is the target the verdict is asked about, and a deployment with no
    record has turned nothing on. The connector is never resolved.
    """
    _prepare(tmp_path, monkeypatch)
    connector = AsyncMock()
    resolved = []

    with assert_raises_error(error_type="writes_disabled") as ctx:
        await _run_single(connector, when_resolved=lambda: resolved.append(True))

    assert ctx["envelope"]["details"] == {"reason": "switch_off", "target": "va"}
    assert resolved == []
    connector.write_channel.assert_not_awaited()


# ---------------------------------------------------------------------------
# The binding holds
# ---------------------------------------------------------------------------


async def test_write_proceeds_when_the_target_is_unchanged(tmp_path, monkeypatch):
    """A record that does not move across the call changes nothing."""
    _prepare(tmp_path, monkeypatch)
    _write_record("va", 3)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    assert set(data) == EXPECTED_TOP_LEVEL_KEYS
    connector.write_channel.assert_awaited_once()


async def test_same_target_respawn_does_not_trip_the_binding(tmp_path, monkeypatch):
    """A respawn replaces the child, not the target — the write goes through.

    The generation counts target changes, and it lives in the record, which a
    respawn does not touch. A child that died and came back on the same target
    changes only this server's own report, and refusing there would make every
    connector recovery look like a switch to the operator.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("va", 2)
    _write_report(session=THIS_SESSION, applied_target="va", applied_generation=2, children=[4321])

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(
        await _run_single(connector, when_resolved=lambda: target_state.record_child_pids([9876]))
    )

    assert data["status"] == "success"
    assert data["summary"]["outcomes"] == {"confirmed": 1}
    connector.write_channel.assert_awaited_once()
    # The respawn really did land in the middle of the call.
    assert target_state.read()["children"] == [9876]


# ---------------------------------------------------------------------------
# The binding breaks
# ---------------------------------------------------------------------------


async def test_target_change_between_entry_and_write_is_refused(tmp_path, monkeypatch):
    """A switch landing mid-call refuses the write and names both bindings."""
    _prepare(tmp_path, monkeypatch)
    _write_record("va", 3)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="target_changed") as ctx:
        await _run_single(connector, when_resolved=lambda: _write_record("live", 4))

    message = ctx["envelope"]["error_message"]
    assert "approved on target 'va' (generation 3)" in message
    assert "now target 'live' (generation 4)" in message
    assert "re-run the write" in message
    details = ctx["envelope"]["details"]
    assert details["approved_target"] == "va"
    assert details["approved_generation"] == 3
    assert details["current_target"] == "live"
    assert details["current_generation"] == 4
    # Refused before the control system was touched.
    connector.write_channel.assert_not_awaited()


async def test_generation_change_alone_is_refused(tmp_path, monkeypatch):
    """The generation is half the binding: moving it alone still refuses.

    A record whose generation moved without the target's name changing has
    switched away and back. Either way the value the operator approved was
    approved against a deployment state that no longer exists.
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("va", 3)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="target_changed") as ctx:
        await _run_single(connector, when_resolved=lambda: _write_record("va", 4))

    message = ctx["envelope"]["error_message"]
    assert "approved on target 'va' (generation 3)" in message
    assert "now target 'va' (generation 4)" in message
    connector.write_channel.assert_not_awaited()


async def test_a_batch_write_is_refused_before_the_connector_call(tmp_path, monkeypatch):
    """The batch path is bound exactly as the single-write path is."""
    _prepare(tmp_path, monkeypatch)
    _write_record("live", 0)

    connector = AsyncMock()
    connector.write_multiple_channels.return_value = []
    operations = [
        {"channel": "PV:A", "value": 1.0},
        {"channel": "PV:B", "value": 2.0},
    ]
    _stamp_current(operations)

    with _patched(connector, when_resolved=lambda: _write_record("va", 1)):
        fn = _get_channel_write()
        with assert_raises_error(error_type="target_changed") as ctx:
            await fn(operations=operations)

    message = ctx["envelope"]["error_message"]
    assert "approved on target 'live' (generation 0)" in message
    assert "now target 'va' (generation 1)" in message
    connector.write_multiple_channels.assert_not_awaited()


async def test_a_call_entering_with_no_record_is_refused_before_one_appears(tmp_path, monkeypatch):
    """A record published mid-call cannot rescue a call that entered without one.

    "No record" and "a record" are different claims about the deployment, and
    the first is every switch off: the verdict refuses at entry, before the
    connector is resolved, so a record appearing later is never consulted.
    """
    _prepare(tmp_path, monkeypatch)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="writes_disabled") as ctx:
        await _run_single(connector, when_resolved=lambda: _write_record("live", 7))

    assert ctx["envelope"]["details"]["reason"] == "switch_off"
    assert control_context.read_record() is None
    connector.write_channel.assert_not_awaited()


async def test_a_record_disappearing_mid_call_is_refused(tmp_path, monkeypatch):
    """A record that vanished mid-call is a deployment that stopped, so refuse."""
    _prepare(tmp_path, monkeypatch)
    record_path = _write_record("va", 5)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="target_changed") as ctx:
        await _run_single(connector, when_resolved=record_path.unlink)

    message = ctx["envelope"]["error_message"]
    assert "approved on target 'va' (generation 5)" in message
    assert "now an unpublished target" in message
    assert ctx["envelope"]["details"]["current_target"] is None
    connector.write_channel.assert_not_awaited()


async def test_a_corrupt_record_is_refused_by_the_verdict(tmp_path, monkeypatch):
    """An unreadable record refuses at entry: the verdict fails closed on it.

    The binding reader collapses a corrupt file into "no record", which is
    stable across the call; the write verdict reads the same file as a record it
    could not read, and an unknown switch is not an "on".
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("va", 5).write_text("{not json", encoding="utf-8")

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="writes_disabled") as ctx:
        await _run_single(connector)

    assert ctx["envelope"]["details"]["reason"] == "record_unavailable"
    connector.write_channel.assert_not_awaited()


# ---------------------------------------------------------------------------
# the approval window: what the prompt showed vs what the call entered on
# ---------------------------------------------------------------------------


async def test_a_switch_while_the_operator_was_deciding_is_refused(tmp_path, monkeypatch):
    """The render-to-click window: the stamp disagrees with entry, so refuse.

    Nothing the server can observe by itself covers this window — the prompt is
    rendered before the tool is called at all. The stamp is what carries the
    binding the human was actually shown across that gap, and it is the binding
    the refusal calls "approved on".
    """
    _prepare(tmp_path, monkeypatch)
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    # The prompt was rendered while the deployment was on the simulator; by the
    # time the operator clicked, a switch had landed.
    _stamp_approval(operations, target="va", generation=3)
    _write_record("live", 4)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="target_changed") as ctx:
            await fn(operations=operations)

    envelope = ctx["envelope"]
    assert "approved on target 'va' (generation 3)" in envelope["error_message"]
    assert "now target 'live' (generation 4)" in envelope["error_message"]
    assert "re-run the write" in envelope["error_message"]
    assert envelope["details"]["window"] == channel_write_module.WINDOW_APPROVAL
    connector.write_channel.assert_not_awaited()


async def test_a_stamp_that_agrees_with_entry_lets_the_write_through(tmp_path, monkeypatch):
    """The ordinary approved write: prompt, entry and pre-write all agree."""
    _prepare(tmp_path, monkeypatch)
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _write_record("va", 3)
    _stamp_approval(operations, target="va", generation=3)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        data = extract_response_dict(await fn(operations=operations))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


async def test_a_stamp_for_a_different_write_is_not_this_calls_approval(tmp_path, monkeypatch):
    """The stamp is keyed by the payload: another write's stamp vouches for nothing.

    Without the key, one approval would vouch for a different set of channels
    and values — which is the opposite of what an approval is. This write has
    no stamp of its own, so where approval asks it is refused.
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("live", 4)
    _stamp_approval([{"channel": "OTHER:PV", "value": 1.0}], target="live", generation=4)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="approval_missing"):
            await fn(operations=[{"channel": "TEST:PV", "value": 42.0}])

    connector.write_channel.assert_not_awaited()


async def test_a_stamp_from_another_session_on_this_checkout_is_ignored(tmp_path, monkeypatch):
    """A second session's approval must not authorize this one.

    Two windows share one agent-data root. A stamp whose ``session`` is not this
    process's belongs to the other window's prompt, so it is not consulted, and
    this write is left with no stamp of its own — refused where approval asks.
    The stamp here agrees with entry, so had it been consulted the write would
    have run: the refusal is the proof it was ignored.
    """
    _prepare(tmp_path, monkeypatch, session="this-session")
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _write_record("live", 4)
    _stamp_approval(operations, target="live", generation=4, session="another-session")

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="approval_missing"):
            await fn(operations=operations)

    connector.write_channel.assert_not_awaited()


async def test_a_prompt_rendered_on_an_unpublished_target_still_binds(tmp_path, monkeypatch):
    """A stamp naming no target is an answer, not an absence.

    The prompt told the operator the target could not be resolved. A record that
    exists by the time the call arrives is a different deployment state from the
    one they were shown, so it refuses — and the message says so in both
    directions.
    """
    _prepare(tmp_path, monkeypatch)
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _stamp_approval(operations, target=None, generation=None)
    _write_record("live", 0)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="target_changed") as ctx:
            await fn(operations=operations)

    envelope = ctx["envelope"]
    assert "approved on an unpublished target" in envelope["error_message"]
    assert "now target 'live' (generation 0)" in envelope["error_message"]
    assert envelope["details"]["window"] == channel_write_module.WINDOW_APPROVAL
    connector.write_channel.assert_not_awaited()


async def test_no_stamp_is_refused_where_approval_asks(tmp_path, monkeypatch):
    """Stage 1 passed, but the prompt left no target stamp: refused.

    Without the stamp the target the operator was shown cannot be compared with
    the one the call entered on, so a switch between them would go unseen. A
    project rendered before the hook stamped anything is refused too, until it
    is rebuilt.
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("live", 4)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="approval_missing") as ctx:
        await _run_single(connector, stamp=False)

    assert ctx["envelope"]["error_message"].startswith("not run:")
    connector.write_channel.assert_not_awaited()


async def test_no_stamp_is_refused_on_the_no_prompt_path_whatever_the_policy_says(
    tmp_path, monkeypatch
):
    """``allowed`` skips the question, not the record, and no policy waives it.

    ``channel_write`` is governed: the switch's position decides it, so a
    ``skip`` policy changes nothing, and the hook files the window stamp on the
    no-prompt path too. A write arriving without one came round the hook.
    """
    _prepare(tmp_path, monkeypatch, config=_CONFIG + _APPROVAL_SKIPS)
    _write_record("live", 4, switch=_SWITCH_NO_PROMPT)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="approval_missing"):
        await _run_single(connector, stamp=False)

    connector.write_channel.assert_not_awaited()


async def test_a_stamp_approved_for_a_different_confirmation_is_not_consulted(
    tmp_path, monkeypatch
):
    """``confirm`` is half the payload, so it is half the stamp's identity.

    The prompt the operator saw named a confirmation setting; a call made with
    another one is a different write, with no stamp of its own, and is refused.
    Keying on it is also what keeps the two halves of the hash in step — if one
    side stopped hashing ``confirm`` the keys would agree only for the omitted
    case.
    """
    _prepare(tmp_path, monkeypatch)
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _write_record("live", 4)
    # An approval rendered for a write that would NOT be confirmed. This call
    # asks for confirmation, so that prompt does not vouch for it.
    _stamp_approval(operations, target="live", generation=4, confirm=False)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="approval_missing"):
            await fn(operations=operations, confirm=True)

    connector.write_channel.assert_not_awaited()


async def test_a_stamp_matching_this_calls_confirmation_binds(tmp_path, monkeypatch):
    """The same write with the same ``confirm`` finds its own approval.

    The other half of the parity: the key the tool derives for an explicit
    ``confirm`` has to be the key the stamp was filed under, or no explicitly
    confirmed write would ever be compared at all.
    """
    _prepare(tmp_path, monkeypatch)
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _write_record("live", 4)
    _stamp_approval(operations, target="va", generation=3, confirm=True)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="target_changed") as ctx:
            await fn(operations=operations, confirm=True)

    assert ctx["envelope"]["details"]["window"] == channel_write_module.WINDOW_APPROVAL
    connector.write_channel.assert_not_awaited()


# ---------------------------------------------------------------------------
# the stale-hook warning on a stamp miss
# ---------------------------------------------------------------------------


def _tool_warnings(caplog):
    """Warnings this tool emitted, and nothing else's.

    ``caplog`` captures the whole run, and building a server context warns about
    an absent archiver section — asserting on the raw text would make "quiet"
    mean "no warning from anywhere", which is not what any of these tests claim.
    """
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == channel_write_module.logger.name and record.levelname == "WARNING"
    ]


async def test_a_miss_with_this_servers_stamps_present_warns_about_the_render(
    tmp_path, monkeypatch, caplog
):
    """Stamps from this server plus a miss means the two key spellings disagree.

    The failure this warning exists for is silent by construction: a project
    rendered before the stamp key changed files its stamps under the old
    derivation, every lookup misses, and every asked write is refused with
    nothing naming the cause. Stamps this process's own prompts left behind
    are the evidence that the hook is stamping and only the key is wrong.
    """
    _prepare(tmp_path, monkeypatch, session="this-session")
    _write_record("live", 4)
    # A stamp this server rendered — for some other write, as an un-rebuilt
    # project's stamps all effectively are.
    _stamp_approval([{"channel": "OTHER:PV", "value": 1.0}], target="live", generation=4)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with (
        caplog.at_level("WARNING", logger=channel_write_module.logger.name),
        assert_raises_error(error_type="approval_missing"),
    ):
        await _run_single(connector, stamp=False)

    assert any("osprey build" in message for message in _tool_warnings(caplog))
    connector.write_channel.assert_not_awaited()


async def test_a_miss_with_only_another_sessions_stamps_stays_quiet(tmp_path, monkeypatch, caplog):
    """Two sessions share the directory: the other one's stamps prove nothing.

    Without the session filter this is the ordinary case — a second window on
    the same checkout has stamps on disk, and every unstamped write in this one
    would tell the operator to rebuild a project that is perfectly current.
    """
    _prepare(tmp_path, monkeypatch, session="this-session")
    _write_record("live", 4)
    _stamp_approval(
        [{"channel": "OTHER:PV", "value": 1.0}],
        target="live",
        generation=4,
        session="another-session",
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with (
        caplog.at_level("WARNING", logger=channel_write_module.logger.name),
        assert_raises_error(error_type="approval_missing"),
    ):
        await _run_single(connector, stamp=False)

    assert _tool_warnings(caplog) == []
    connector.write_channel.assert_not_awaited()


async def test_a_miss_with_no_stamps_at_all_stays_quiet(tmp_path, monkeypatch, caplog):
    """No stamps on disk prove nothing about the render, so the refusal gives no advice."""
    _prepare(tmp_path, monkeypatch)
    _write_record("live", 4)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with (
        caplog.at_level("WARNING", logger=channel_write_module.logger.name),
        assert_raises_error(error_type="approval_missing"),
    ):
        await _run_single(connector, stamp=False)

    assert _tool_warnings(caplog) == []
    connector.write_channel.assert_not_awaited()


# ---------------------------------------------------------------------------
# the record vs the host actually serving
# ---------------------------------------------------------------------------


async def test_a_serving_host_that_disagrees_with_the_file_refuses(tmp_path, monkeypatch):
    """A failed publish leaves the record behind; a write must not ride on it.

    The record is what the operator was shown and the manager is what is
    actually serving. A disagreement means the switch failed or has not landed —
    the identity of the deployment is in doubt at the exact moment a value would
    go out, and neither answer may be preferred.
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("va", 3)
    monkeypatch.setattr(ConnectorHostManager, "is_started", lambda self: True)
    monkeypatch.setattr(ConnectorHostManager, "active_binding", lambda self: ("live", 4))
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _stamp_current(operations)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="target_changed") as ctx:
            await fn(operations=operations)

    envelope = ctx["envelope"]
    assert "approved on target 'va' (generation 3)" in envelope["error_message"]
    assert "now target 'live' (generation 4)" in envelope["error_message"]
    assert envelope["details"]["window"] == channel_write_module.WINDOW_SERVING
    connector.write_channel.assert_not_awaited()


async def test_a_serving_host_that_agrees_with_the_file_writes(tmp_path, monkeypatch):
    """The normal switched session: record and manager say the same thing."""
    _prepare(tmp_path, monkeypatch)
    _write_record("va", 3)
    monkeypatch.setattr(ConnectorHostManager, "is_started", lambda self: True)
    monkeypatch.setattr(ConnectorHostManager, "active_binding", lambda self: ("va", 3))

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


async def test_a_manager_that_never_started_is_not_consulted(tmp_path, monkeypatch):
    """An in-process deployment has no second opinion, and needs none.

    The manager exists as an object on the server context but has started no
    child; its ``(baseline, 0)`` is not a claim about anything that is serving,
    and reading it as one would refuse writes on every deployment that never
    switches.
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("va", 7)
    monkeypatch.setattr(
        ConnectorHostManager,
        "active_binding",
        lambda self: pytest.fail("an unstarted manager must not be asked what it is serving"),
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


# ---------------------------------------------------------------------------
# convergence: a swap in flight, and a server that could not follow one
# ---------------------------------------------------------------------------


async def test_a_swap_in_flight_on_this_server_refuses_naming_the_pid(tmp_path, monkeypatch):
    """This server is between two targets, so no value may go out.

    The record already says where the deployment is; the report says this
    server's connector host has not got there. A value written now would reach
    whichever target the host happens to be holding, which is precisely the
    thing the whole binding exists to prevent — and the refusal names the PID,
    because an operator has to know which process must finish.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(
        session=THIS_SESSION,
        applied_target="va",
        applied_generation=3,
        last_switch=_applying(4),
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="switch_in_progress") as ctx:
        await _run_single(connector)

    envelope = ctx["envelope"]
    assert f"switch_in_progress:{os.getpid()}" in envelope["error_message"]
    assert envelope["details"]["pids"] == [os.getpid()]
    assert envelope["details"]["target"] == "live"
    assert envelope["details"]["generation"] == 4
    connector.write_channel.assert_not_awaited()


async def test_a_swap_in_flight_on_another_session_refuses_this_one_too(tmp_path, monkeypatch):
    """An in-flight swap stops every session, not only the one switching.

    The server holding the connector is between two targets for the whole
    deployment. Unlike a failure, which strands one session, this is the one
    condition that is everybody's business until it lands or times out.
    """
    other = _live_other_pid()
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(session=THIS_SESSION, applied_target="live", applied_generation=4)
    _write_report(
        other,
        session=OTHER_SESSION,
        applied_target="live",
        applied_generation=3,
        last_switch=_applying(4),
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="switch_in_progress") as ctx:
        await _run_single(connector)

    assert ctx["envelope"]["details"]["pids"] == [other]
    connector.write_channel.assert_not_awaited()


async def test_a_swap_in_flight_refuses_a_session_that_owns_no_report(tmp_path, monkeypatch):
    """A bare ``claude`` has no report of its own and is still held up.

    It owns nothing that could be behind, so the second rule can never name it —
    but the connector it would write through belongs to a server that is
    mid-swap, and that is enough.
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("live", 4)
    _write_report(
        session=OTHER_SESSION, applied_target="live", applied_generation=3, last_switch=_applying(4)
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="switch_in_progress"):
        await _run_single(connector)

    connector.write_channel.assert_not_awaited()


async def test_a_swap_reported_for_another_generation_does_not_refuse(tmp_path, monkeypatch):
    """A block naming a generation the record has moved past coordinates nothing.

    The generation is the only thing the fleet agrees on. A straggler still
    carrying an older block is judged by what it is bound to, and this one is
    bound exactly where the record says.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 5)
    _write_report(
        session=THIS_SESSION,
        applied_target="live",
        applied_generation=5,
        last_switch=_applying(4),
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


async def test_a_stuck_swap_on_another_session_stops_holding_this_one_up(tmp_path, monkeypatch):
    """Past its own bound, an ``applying`` block holds up only its own session.

    A process killed mid-swap leaves a block that nothing will ever finish.
    Bounding it is what keeps that from refusing writes on a deployment for
    ever, and the bound is the one the publishing server wrote.
    """
    other = _live_other_pid()
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(session=THIS_SESSION, applied_target="live", applied_generation=4)
    _write_report(other, session=OTHER_SESSION, last_switch=_applying(4, expires_in_s=-60))

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


async def test_a_stuck_swap_on_this_server_still_refuses_this_server(tmp_path, monkeypatch):
    """The other half of the bound: a stuck swap of one's own is still stuck.

    Expiry says "this is no longer in flight", not "this is fine". The server
    this session writes through never arrived, so its own writes stay refused
    while every other session carries on.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(
        session=THIS_SESSION,
        applied_target="va",
        applied_generation=3,
        last_switch=_applying(4, expires_in_s=-60),
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="switch_in_progress") as ctx:
        await _run_single(connector)

    assert ctx["envelope"]["details"]["pids"] == [os.getpid()]
    connector.write_channel.assert_not_awaited()


async def test_a_failed_swap_refuses_this_servers_writes(tmp_path, monkeypatch):
    """FR-8: the session whose server could not follow is the one refused.

    A failed swap leaves this server on a target the deployment has moved off.
    Its MCP writes are refused for as long as that is true — the value would go
    to the machine the record says the deployment is no longer pointed at.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(
        session=THIS_SESSION,
        applied_target="va",
        applied_generation=3,
        last_switch=_failed(4),
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="switch_in_progress") as ctx:
        await _run_single(connector)

    envelope = ctx["envelope"]
    assert f"switch_in_progress:{os.getpid()}" in envelope["error_message"]
    assert envelope["details"]["pids"] == [os.getpid()]
    connector.write_channel.assert_not_awaited()


async def test_a_failed_swap_elsewhere_leaves_this_session_writing(tmp_path, monkeypatch):
    """FR-8's other half: a failure strands one session and no more.

    Another window's server failed to follow the switch. This session's server
    is on the record's target, so its writes reach exactly the machine the
    operator was shown, and refusing them would turn one stranded window into a
    deployment-wide outage.
    """
    other = _live_other_pid()
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(session=THIS_SESSION, applied_target="live", applied_generation=4)
    _write_report(
        other,
        session=OTHER_SESSION,
        applied_target="va",
        applied_generation=3,
        last_switch=_failed(4),
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


async def test_a_server_bound_to_an_older_generation_refuses_its_own_session(tmp_path, monkeypatch):
    """The headline case: ``applied_generation != record.generation`` refuses.

    No block at all, no failure reported — just a server whose connector host
    is still on the previous generation. That is the whole condition the write
    tools gate on, and it is refused without needing a status word to say so.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(session=THIS_SESSION, applied_target="live", applied_generation=3)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="switch_in_progress") as ctx:
        await _run_single(connector)

    assert ctx["envelope"]["details"]["pids"] == [os.getpid()]
    connector.write_channel.assert_not_awaited()


async def test_a_fresh_server_that_has_bound_nothing_yet_does_not_refuse(tmp_path, monkeypatch):
    """A null binding is "not there yet", never "on the baseline".

    A server that has just started has answered no init frame, so both halves of
    its binding are null. Reading that as a disagreement would refuse the first
    write of every session before its host has finished coming up.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(session=THIS_SESSION)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


async def test_a_server_bound_where_the_record_says_writes(tmp_path, monkeypatch):
    """The settled deployment: report and record agree, so nothing is in the way."""
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(session=THIS_SESSION, applied_target="live", applied_generation=4)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


async def test_a_dead_servers_swap_does_not_refuse(tmp_path, monkeypatch):
    """Residue is not a fleet: a report whose server exited refuses nobody.

    The file outlives the process that wrote it until the next server sweeps
    it. Reading it as a live claim would let a crashed window block writes
    across the deployment with nothing an operator could stop.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(
        _dead_pid(),
        session=OTHER_SESSION,
        applied_target="live",
        applied_generation=3,
        last_switch=_applying(4),
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


async def test_without_a_record_the_verdict_refuses_before_convergence(tmp_path, monkeypatch):
    """No record means no generation to be behind, and no switch turned on.

    A report claiming a swap in flight is never judged: the call is refused at
    entry by the write verdict, in its own words, not as a switch in progress.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_report(
        session=THIS_SESSION, applied_target="live", applied_generation=3, last_switch=_applying(4)
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="writes_disabled") as ctx:
        await _run_single(connector)

    assert ctx["envelope"]["details"]["reason"] == "switch_off"
    connector.write_channel.assert_not_awaited()


async def test_a_stale_approval_is_reported_before_an_unsettled_fleet(tmp_path, monkeypatch):
    """Both refusals apply; the operator is told the one about their own click.

    A stamp that disagrees with entry says the value in front of the operator
    was approved for a different target — the more specific account, and the
    one that tells them a fresh approval is needed rather than a wait. The
    convergence gate is checked immediately after, so nothing is written either
    way.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _stamp_approval(operations, target="va", generation=3)
    _write_record("live", 4)
    _write_report(
        session=THIS_SESSION, applied_target="live", applied_generation=3, last_switch=_applying(4)
    )

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="target_changed") as ctx:
            await fn(operations=operations)

    assert ctx["envelope"]["details"]["window"] == channel_write_module.WINDOW_APPROVAL
    connector.write_channel.assert_not_awaited()
