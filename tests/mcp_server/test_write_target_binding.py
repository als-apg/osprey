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
from osprey.connectors.control_system.base import ChannelWriteResult, WriteOutcome
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import ConnectorHostManager
from osprey.mcp_server.control_system.server_context import initialize_server_context
from osprey.mcp_server.control_system.tools import channel_write as channel_write_module
from osprey_connectors import control_context, session_store
from tests._control_context_fixtures import write_control_context, write_server_report
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


def _prepare(tmp_path, monkeypatch, *, session=None):
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
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    root = tmp_path / "var" / "agent_data"
    (root / control_context.STATE_DIR_NAME).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    if session is None:
        monkeypatch.delenv(POSTURE_SESSION_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(POSTURE_SESSION_ENV_VAR, session)
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
    control_context.invalidate_cache()
    initialize_server_context()
    return root


def _root():
    """The agent-data root :func:`_prepare` stamped.

    The shared writers take the root explicitly — a suite has to be able to lay
    a deployment down before it points the environment at one — while every
    test here has already stamped it. Deriving it back from the resolved state
    directory keeps the two from being spelled twice in one file.
    """
    return control_context.state_dir().parent


def _write_record(target, generation, **fields):
    """Write the deployment's record, as the owner writes it after a switch."""
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
    proc = subprocess.Popen([sys.executable, "-c", ""])  # noqa: S603 - fixed argv
    proc.wait()
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

    async def _create(_config, *, control_target=None):
        if when_resolved is not None:
            when_resolved()
        return connector

    with (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new=_create,
        ),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
    ):
        yield


async def _run_single(connector, *, when_resolved=None, channel="TEST:PV", value=42.0):
    with _patched(connector, when_resolved=when_resolved):
        fn = _get_channel_write()
        return await fn(operations=[{"channel": channel, "value": value}])


# ---------------------------------------------------------------------------
# The binding does not exist: a deployment with no published target
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_write_proceeds_and_looks_unchanged_without_target_state(tmp_path, monkeypatch):
    """No record at all: the write runs, and the envelope is what it was.

    This is the baseline in-process deployment — nothing publishes a target, so
    there is nothing to bind to and nothing to refuse. The envelope's shape is
    asserted key-by-key: a deployment that never switches targets must not be
    able to tell that this feature exists.
    """
    _prepare(tmp_path, monkeypatch)
    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert set(data) == EXPECTED_TOP_LEVEL_KEYS
    assert set(data["summary"]) == EXPECTED_SUMMARY_KEYS
    assert set(data["summary"]["results"][0]) == EXPECTED_RESULT_KEYS
    assert data["status"] == "success"
    assert data["summary"]["outcomes"] == {"confirmed": 1}
    connector.write_channel.assert_awaited_once()


# ---------------------------------------------------------------------------
# The binding holds
# ---------------------------------------------------------------------------


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
async def test_a_batch_write_is_refused_before_the_connector_call(tmp_path, monkeypatch):
    """The batch path is bound exactly as the single-write path is."""
    _prepare(tmp_path, monkeypatch)
    _write_record("live", 0)

    connector = AsyncMock()
    connector.write_multiple_channels.return_value = []

    with _patched(connector, when_resolved=lambda: _write_record("va", 1)):
        fn = _get_channel_write()
        with assert_raises_error(error_type="target_changed") as ctx:
            await fn(
                operations=[
                    {"channel": "PV:A", "value": 1.0},
                    {"channel": "PV:B", "value": 2.0},
                ]
            )

    message = ctx["envelope"]["error_message"]
    assert "approved on target 'live' (generation 0)" in message
    assert "now target 'va' (generation 1)" in message
    connector.write_multiple_channels.assert_not_awaited()


@pytest.mark.unit
async def test_a_record_appearing_mid_call_is_refused(tmp_path, monkeypatch):
    """A deployment that started publishing mid-call is a change, not a nothing.

    "No record" and "a record" are different claims about the deployment. The
    write was approved while nothing published a target; by the time it would
    execute something does, and what that something is was never shown to the
    operator.
    """
    _prepare(tmp_path, monkeypatch)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="target_changed") as ctx:
        await _run_single(connector, when_resolved=lambda: _write_record("live", 7))

    message = ctx["envelope"]["error_message"]
    assert "approved on an unpublished target" in message
    assert "now target 'live' (generation 7)" in message
    details = ctx["envelope"]["details"]
    assert details["approved_target"] is None
    assert details["approved_generation"] is None
    assert details["current_target"] == "live"
    connector.write_channel.assert_not_awaited()


@pytest.mark.unit
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


@pytest.mark.unit
async def test_a_corrupt_record_reads_as_no_record_and_does_not_refuse(tmp_path, monkeypatch):
    """An unreadable record is "no answer" at both ends, so the write runs.

    The record's readers are fail-closed by contract: every failure mode arrives
    as the same value. A file that is corrupt for the whole call is therefore
    absent for the whole call, which is stable — and turning a corrupt file into
    a refusal would break writes on a deployment that never switches at all.
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("va", 5).write_text("{not json", encoding="utf-8")

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


# ---------------------------------------------------------------------------
# the approval window: what the prompt showed vs what the call entered on
# ---------------------------------------------------------------------------


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
async def test_a_stamp_for_a_different_write_is_not_this_calls_approval(tmp_path, monkeypatch):
    """The stamp is keyed by the payload: another write's stamp is not consulted.

    Without the key, one approval would vouch for a different set of channels
    and values — which is the opposite of what an approval is.
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("live", 4)
    _stamp_approval([{"channel": "OTHER:PV", "value": 1.0}], target="va", generation=3)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        data = extract_response_dict(await fn(operations=[{"channel": "TEST:PV", "value": 42.0}]))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


@pytest.mark.unit
async def test_a_stamp_from_another_session_on_this_checkout_is_ignored(tmp_path, monkeypatch):
    """A second session's approval must not refuse — or authorize — this one.

    Two windows share one agent-data root. A stamp whose ``session`` is not this
    process's belongs to the other window's prompt, so it is not consulted; the
    call keeps the comparison it can make honestly. The binding here would
    refuse if it were consulted, which is what makes the success meaningful.
    """
    _prepare(tmp_path, monkeypatch, session="this-session")
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _write_record("live", 4)
    _stamp_approval(operations, target="va", generation=3, session="another-session")

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        data = extract_response_dict(await fn(operations=operations))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


@pytest.mark.unit
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


@pytest.mark.unit
async def test_no_stamp_means_no_approval_comparison(tmp_path, monkeypatch):
    """An unstamped call is not refused: older renders must keep working.

    A project rendered before the hook stamped anything, and a deployment whose
    policy allows the write without asking, both arrive here with no stamp. The
    entry-to-write comparison still applies; the approval window simply cannot
    be checked, and inventing a refusal from its absence would break every one
    of those deployments.
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("live", 4)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


@pytest.mark.unit
async def test_a_stamp_approved_for_a_different_confirmation_is_not_consulted(
    tmp_path, monkeypatch
):
    """``confirm`` is half the payload, so it is half the stamp's identity.

    The prompt the operator saw named a confirmation setting; a call made with
    another one is a different write. Keying on it is also what keeps the two
    halves of the hash in step — if one side stopped hashing ``confirm`` the
    keys would agree only for the omitted case, and the window check would go
    quiet for every explicit one without failing anything.
    """
    _prepare(tmp_path, monkeypatch)
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _write_record("live", 4)
    # An approval rendered for a write that would NOT be confirmed. This call
    # asks for confirmation, so that prompt does not vouch for it.
    _stamp_approval(operations, target="va", generation=3, confirm=False)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        data = extract_response_dict(await fn(operations=operations, confirm=True))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


@pytest.mark.unit
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


@pytest.mark.unit
async def test_a_miss_with_this_servers_stamps_present_warns_about_the_render(
    tmp_path, monkeypatch, caplog
):
    """Stamps from this server plus a miss means the two key spellings disagree.

    The failure this warning exists for is silent by construction: a project
    rendered before the stamp key changed files its stamps under the old
    derivation, every lookup misses, and the approval-window check stops running
    without anything going red. Stamps this process's own prompts left behind
    are the evidence that the hook is stamping and only the key is wrong.
    """
    _prepare(tmp_path, monkeypatch, session="this-session")
    _write_record("live", 4)
    # A stamp this server rendered — for some other write, as an un-rebuilt
    # project's stamps all effectively are.
    _stamp_approval([{"channel": "OTHER:PV", "value": 1.0}], target="live", generation=4)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with caplog.at_level("WARNING", logger=channel_write_module.logger.name):
        data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success", "the warning is advice, never a refusal"
    assert any("osprey build" in message for message in _tool_warnings(caplog))
    connector.write_channel.assert_awaited_once()


@pytest.mark.unit
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

    with caplog.at_level("WARNING", logger=channel_write_module.logger.name):
        data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    assert _tool_warnings(caplog) == []
    connector.write_channel.assert_awaited_once()


@pytest.mark.unit
async def test_a_miss_with_no_stamps_at_all_stays_quiet(tmp_path, monkeypatch, caplog):
    """A deployment whose policy never asks has no stamps and needs no advice."""
    _prepare(tmp_path, monkeypatch)
    _write_record("live", 4)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with caplog.at_level("WARNING", logger=channel_write_module.logger.name):
        data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    assert _tool_warnings(caplog) == []
    connector.write_channel.assert_awaited_once()


# ---------------------------------------------------------------------------
# the record vs the host actually serving
# ---------------------------------------------------------------------------


@pytest.mark.unit
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

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="target_changed") as ctx:
            await fn(operations=[{"channel": "TEST:PV", "value": 42.0}])

    envelope = ctx["envelope"]
    assert "approved on target 'va' (generation 3)" in envelope["error_message"]
    assert "now target 'live' (generation 4)" in envelope["error_message"]
    assert envelope["details"]["window"] == channel_write_module.WINDOW_SERVING
    connector.write_channel.assert_not_awaited()


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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
    _write_report(other, session=OTHER_SESSION, last_switch=_applying(4))

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="switch_in_progress") as ctx:
        await _run_single(connector)

    assert ctx["envelope"]["details"]["pids"] == [other]
    connector.write_channel.assert_not_awaited()


@pytest.mark.unit
async def test_a_swap_in_flight_refuses_a_session_that_owns_no_report(tmp_path, monkeypatch):
    """A bare ``claude`` has no report of its own and is still held up.

    It owns nothing that could be behind, so the second rule can never name it —
    but the connector it would write through belongs to a server that is
    mid-swap, and that is enough.
    """
    _prepare(tmp_path, monkeypatch)
    _write_record("live", 4)
    _write_report(session=OTHER_SESSION, last_switch=_applying(4))

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="switch_in_progress"):
        await _run_single(connector)

    connector.write_channel.assert_not_awaited()


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
async def test_a_dead_servers_swap_does_not_refuse(tmp_path, monkeypatch):
    """Residue is not a fleet: a report whose server exited refuses nobody.

    The file outlives the process that wrote it until the next server sweeps
    it. Reading it as a live claim would let a crashed window block writes
    across the deployment with nothing an operator could stop.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_record("live", 4)
    _write_report(_dead_pid(), session=OTHER_SESSION, last_switch=_applying(4))

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


@pytest.mark.unit
async def test_without_a_record_there_is_no_convergence_to_judge(tmp_path, monkeypatch):
    """No record means no generation to be behind, so reports decide nothing.

    The whole verdict is "where the record says" against "where this server
    got to". With no record there is no first half, and the deployment is the
    unswitched one every other test in this file's first section describes.
    """
    _prepare(tmp_path, monkeypatch, session=THIS_SESSION)
    _write_report(session=THIS_SESSION, last_switch=_applying(4))

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


@pytest.mark.unit
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
    _write_report(session=THIS_SESSION, last_switch=_applying(4))

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        with assert_raises_error(error_type="target_changed") as ctx:
            await fn(operations=operations)

    assert ctx["envelope"]["details"]["window"] == channel_write_module.WINDOW_APPROVAL
    connector.write_channel.assert_not_awaited()
