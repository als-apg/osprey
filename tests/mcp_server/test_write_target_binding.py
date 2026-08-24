"""``channel_write`` binds itself to the target it was approved on.

The approval prompt names a target and a generation, read out of the state file
the controls server publishes. Between that prompt and the moment the write
reaches the control system, a target switch can land — and a value approved for
the simulator must never be applied to the real machine because it arrived a
second late.

The tool therefore observes ``(target, generation)`` three times — as the
approval prompt rendered it (from the stamp the hook leaves behind), at entry,
and immediately before the connector call — and refuses on any difference. These
tests drive all three windows:

* the approval window with a stamp file written before the call, which is
  exactly what the hook leaves on disk;
* the execution window by flipping the state inside the connector resolution, a
  real seam of the tool's own execution path that runs after the entry capture
  and before the pre-write re-read — precisely where a switch completing in
  another task would land;
* the published-versus-serving disagreement by giving the connector-host manager
  a started child on a binding the state file does not name.

Every assertion is made on the shipped envelope (the parsed tool result, or the
structured error a refusal raises), never on an internal helper: the envelope is
the only thing an agent or an operator ever sees.
"""

import json
import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.connectors.control_system.base import WriteVerification
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import ConnectorHostManager
from osprey.mcp_server.control_system.server_context import initialize_server_context
from osprey.mcp_server.control_system.tools import channel_write as channel_write_module
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)

#: The shipped envelope's shape, pinned so that binding a write to a target
#: cannot smuggle a field into the payload an agent reads. A refusal is an error
#: envelope; a write that proceeds must look exactly as it did before.
EXPECTED_TOP_LEVEL_KEYS = {"status", "description", "summary", "access_details"}
EXPECTED_SUMMARY_KEYS = {
    "total_writes",
    "successful",
    "failed",
    "refused",
    "verification_failed",
    "results",
}
EXPECTED_RESULT_KEYS = {
    "channel",
    "value",
    "success",
    "write_state",
    "error",
    "blocked",
    "refusal_reason",
    "verification",
}

#: Display metadata as the server's single writer records it. Irrelevant to the
#: binding — which is the target and the generation only — but written anyway so
#: the fixture records are the shape a reader really meets.
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


def _prepare(tmp_path, monkeypatch):
    """Project, server context, and a state root nothing else writes into.

    ``state_dir()`` resolves through the name ``resolve_shared_data_root`` bound
    in :mod:`~osprey.mcp_server.control_system.target_state`; rebinding that one
    name keeps every other path rule real while pointing the state files at a
    directory this test owns.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    root = tmp_path / "var" / "agent_data"
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
    initialize_server_context()


def _publish_start(target, generation, children=None):
    """Write this process's state file as a server on *target* would."""
    target_state.write_on_start(
        target, _TARGETS_META, generation=generation, children=children or []
    )


def _stamp_approval(operations, *, target, generation, verification_level=None, server_pid=None):
    """Leave the stamp a rendered ``channel_write`` approval leaves behind.

    Written the way the hook writes it — same directory, same file name, same
    fields — but through the tool's own key derivation, because what this file
    tests is the comparison. That the hook derives the SAME key from the same
    payload is pinned on the hook's side, in ``tests/hooks``.
    """
    key = channel_write_module._approval_stamp_key(operations, verification_level)
    directory = target_state.state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"{channel_write_module.APPROVAL_STAMP_PREFIX}{key}"
        f"{channel_write_module.APPROVAL_STAMP_SUFFIX}"
    )
    path.write_text(
        json.dumps(
            {
                "tool": "channel_write",
                "key": key,
                "target": target,
                "generation": generation,
                "server_pid": os.getpid() if server_pid is None else server_pid,
                "rendered_at": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_result(channel="TEST:PV", value=42.0):
    """A plain successful write result with a real verification object."""
    result = MagicMock()
    result.channel_address = channel
    result.value_written = value
    result.success = True
    result.error_message = None
    result.blocked = False
    result.refusal_reason = None
    result.verification = WriteVerification(
        level="callback",
        verified=True,
        readback_value=value,
        tolerance_used=0.1,
        notes="",
    )
    return result


@contextmanager
def _patched(connector, *, when_resolved=None):
    """Serve *connector* to the tool, optionally moving the target on the way.

    ``when_resolved`` runs while the tool is resolving its connector — after the
    entry capture, before the pre-write re-read. That is the window a switch
    landing in another task occupies, and driving it here needs no threads: the
    tool's own control flow passes through this seam.
    """

    async def _create(_config):
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
    """No state file at all: the write runs, and the envelope is what it was.

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
    assert data["summary"]["successful"] == 1
    connector.write_channel.assert_awaited_once()


# ---------------------------------------------------------------------------
# The binding holds
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_write_proceeds_when_the_target_is_unchanged(tmp_path, monkeypatch):
    """A state file that does not move across the call changes nothing."""
    _prepare(tmp_path, monkeypatch)
    _publish_start("va", 3)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    assert set(data) == EXPECTED_TOP_LEVEL_KEYS
    connector.write_channel.assert_awaited_once()


@pytest.mark.unit
async def test_same_target_respawn_does_not_trip_the_binding(tmp_path, monkeypatch):
    """A respawn replaces the child, not the target — the write goes through.

    The generation counts target changes, so a child that died and came back on
    the same target leaves the binding intact. Nothing a write was promised has
    changed, and refusing here would make every connector recovery look like a
    switch to the operator.
    """
    _prepare(tmp_path, monkeypatch)
    _publish_start("va", 2, children=[4321])

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(
        await _run_single(connector, when_resolved=lambda: target_state.record_child_pids([9876]))
    )

    assert data["status"] == "success"
    assert data["summary"]["successful"] == 1
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
    _publish_start("va", 3)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="target_changed") as ctx:
        await _run_single(connector, when_resolved=lambda: target_state.publish_switch("live", 4))

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

    A server whose generation moved without the target's name changing has
    switched away and back, or has been overtaken by a record this process did
    not write. Either way the value the operator approved was approved against a
    session that no longer exists.
    """
    _prepare(tmp_path, monkeypatch)
    _publish_start("va", 3)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="target_changed") as ctx:
        await _run_single(connector, when_resolved=lambda: target_state.publish_switch("va", 4))

    message = ctx["envelope"]["error_message"]
    assert "approved on target 'va' (generation 3)" in message
    assert "now target 'va' (generation 4)" in message
    connector.write_channel.assert_not_awaited()


@pytest.mark.unit
async def test_a_batch_write_is_refused_before_the_connector_call(tmp_path, monkeypatch):
    """The batch path is bound exactly as the single-write path is."""
    _prepare(tmp_path, monkeypatch)
    _publish_start("live", 0)

    connector = AsyncMock()
    connector.write_multiple_channels.return_value = []

    with _patched(connector, when_resolved=lambda: target_state.publish_switch("va", 1)):
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
async def test_target_state_appearing_mid_call_is_refused(tmp_path, monkeypatch):
    """A server that started publishing mid-call is a change, not a nothing.

    "No record" and "a record" are different claims about the session. The write
    was approved while nothing published a target; by the time it would execute
    something does, and what that something is was never shown to the operator.
    """
    _prepare(tmp_path, monkeypatch)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="target_changed") as ctx:
        await _run_single(connector, when_resolved=lambda: _publish_start("live", 7))

    message = ctx["envelope"]["error_message"]
    assert "approved on an unpublished target" in message
    assert "now target 'live' (generation 7)" in message
    details = ctx["envelope"]["details"]
    assert details["approved_target"] is None
    assert details["approved_generation"] is None
    assert details["current_target"] == "live"
    connector.write_channel.assert_not_awaited()


@pytest.mark.unit
async def test_target_state_disappearing_mid_call_is_refused(tmp_path, monkeypatch):
    """A record that vanished mid-call is a server that stopped, so refuse."""
    _prepare(tmp_path, monkeypatch)
    _publish_start("va", 5)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with assert_raises_error(error_type="target_changed") as ctx:
        await _run_single(connector, when_resolved=target_state.delete_on_shutdown)

    message = ctx["envelope"]["error_message"]
    assert "approved on target 'va' (generation 5)" in message
    assert "now an unpublished target" in message
    assert ctx["envelope"]["details"]["current_target"] is None
    connector.write_channel.assert_not_awaited()


@pytest.mark.unit
async def test_a_corrupt_record_reads_as_no_record_and_does_not_refuse(tmp_path, monkeypatch):
    """An unreadable state file is "no answer" at both ends, so the write runs.

    Readers of this file are fail-closed by contract: every failure mode arrives
    as the same value. A file that is corrupt for the whole call is therefore
    absent for the whole call, which is stable — and turning a corrupt file into
    a refusal would break writes on a deployment that never switches at all.
    """
    _prepare(tmp_path, monkeypatch)
    _publish_start("va", 5)
    target_state.state_file_path().write_text("{not json")

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
    # The prompt was rendered while the session was on the simulator; by the time
    # the operator clicked, a switch had landed.
    _stamp_approval(operations, target="va", generation=3)
    _publish_start("live", 4)

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
    _publish_start("va", 3)
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
    _publish_start("live", 4)
    _stamp_approval([{"channel": "OTHER:PV", "value": 1.0}], target="va", generation=3)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    with _patched(connector):
        fn = _get_channel_write()
        data = extract_response_dict(await fn(operations=[{"channel": "TEST:PV", "value": 42.0}]))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


@pytest.mark.unit
async def test_a_stamp_from_another_server_on_this_checkout_is_ignored(tmp_path, monkeypatch):
    """A second session's approval must not refuse — or authorize — this one.

    Two sessions share one state directory. A stamp whose ``server_pid`` is not
    this process belongs to the other session's prompt, so it is not consulted;
    the call keeps the comparison it can make honestly.
    """
    _prepare(tmp_path, monkeypatch)
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _publish_start("live", 4)
    _stamp_approval(operations, target="va", generation=3, server_pid=os.getpid() + 1)

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
    exists by the time the call arrives is a different session from the one they
    were shown, so it refuses — and the message says so in both directions.
    """
    _prepare(tmp_path, monkeypatch)
    operations = [{"channel": "TEST:PV", "value": 42.0}]
    _stamp_approval(operations, target=None, generation=None)
    _publish_start("live", 0)

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
    _publish_start("live", 4)

    connector = AsyncMock()
    connector.write_channel.return_value = _write_result()

    data = extract_response_dict(await _run_single(connector))

    assert data["status"] == "success"
    connector.write_channel.assert_awaited_once()


# ---------------------------------------------------------------------------
# the published record vs the host actually serving
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_a_serving_host_that_disagrees_with_the_file_refuses(tmp_path, monkeypatch):
    """A failed publish leaves the file behind; a write must not ride on it.

    The state file is what the operator was shown and the manager is what is
    actually serving. One writer owns both, so a disagreement means the publish
    failed or has not landed — the identity of the session is in doubt at the
    exact moment a value would go out, and neither answer may be preferred.
    """
    _prepare(tmp_path, monkeypatch)
    _publish_start("va", 3)
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
    """The normal switched session: file and manager say the same thing."""
    _prepare(tmp_path, monkeypatch)
    _publish_start("va", 3)
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
    _publish_start("va", 7)
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
