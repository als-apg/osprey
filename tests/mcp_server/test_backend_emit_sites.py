"""Agent-activity emit sites for backend-direct tools.

Verifies that channel_write, launch_run, and the artifact focus tools report
agent activity via ``notify_agent_activity`` — and, just as important, that
refusal paths emit NOTHING and that tool results are unchanged when the web
terminal is down.

Patch seam: each tool module imports ``notify_agent_activity`` directly, so
the mock must target the caller's namespace (e.g.
``osprey.mcp_server.control_system.tools.channel_write.notify_agent_activity``),
not ``osprey.mcp_server.http``.
"""

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)

pytestmark = pytest.mark.unit

_CW_MOD = "osprey.mcp_server.control_system.tools.channel_write"
_LAUNCH_MOD = "osprey.mcp_server.bluesky.tools.launch"
_FOCUS_MOD = "osprey.mcp_server.workspace.tools.focus_tools"


def _free_port() -> int:
    """Reserve a localhost port and release it (nothing will be listening)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ── channel_write ───────────────────────────────────────────────────────────


def _make_write_result(
    channel="TEST:PV",
    value=1.0,
    success=True,
    error_message=None,
    blocked=False,
    refusal_reason=None,
):
    result = MagicMock()
    result.channel_address = channel
    result.value_written = value
    result.success = success
    result.error_message = error_message
    result.blocked = blocked
    result.refusal_reason = refusal_reason
    result.verification = None
    return result


def _get_channel_write():
    from osprey.mcp_server.control_system.tools.channel_write import channel_write

    return get_tool_fn(channel_write)


def _channel_write_patches(mock_connector):
    return (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
    )


async def test_channel_write_limits_violation_no_emit(tmp_path, monkeypatch):
    """A validation refusal (limits_violation) must emit nothing."""
    from osprey.errors import ChannelLimitsViolationError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")

    mock_validator = MagicMock()
    mock_validator.validate.side_effect = ChannelLimitsViolationError(
        channel_address="TEST:PV",
        value=9999.0,
        violation_type="MAX_EXCEEDED",
        violation_reason="Value 9999.0 above maximum 100.0",
        min_value=0.0,
        max_value=100.0,
    )

    with (
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=mock_validator,
        ),
        patch(f"{_CW_MOD}.notify_agent_activity") as notify,
    ):
        fn = _get_channel_write()
        with assert_raises_error(error_type="limits_violation"):
            await fn(operations=[{"channel": "TEST:PV", "value": 9999.0}])

    notify.assert_not_called()


async def test_channel_write_partial_success_emits_executed_only(tmp_path, monkeypatch):
    """Some success + some blocked: exactly ONE emit naming only executed channels."""
    from osprey.mcp_server.control_system.server_context import initialize_server_context

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    results = [
        _make_write_result(channel="SR01:HCM1:SP", value=1.0, success=True),
        _make_write_result(
            channel="SR02:VCM3:SP",
            value=2.0,
            success=False,
            error_message="Write to 'SR02:VCM3:SP' blocked: writes are disabled.",
            blocked=True,
            refusal_reason="WRITES_DISABLED",
        ),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

    conn_patch, validator_patch = _channel_write_patches(mock_connector)
    with conn_patch, validator_patch, patch(f"{_CW_MOD}.notify_agent_activity") as notify:
        fn = _get_channel_write()
        result = await fn(
            operations=[
                {"channel": "SR01:HCM1:SP", "value": 1.0},
                {"channel": "SR02:VCM3:SP", "value": 2.0},
            ]
        )

    # Tool result itself unchanged: partial refusal is still a success envelope.
    data = extract_response_dict(result)
    assert data["status"] == "success"

    assert notify.call_count == 1
    args = notify.call_args.args
    kwargs = notify.call_args.kwargs
    assert args[0] == "channel_write"
    assert args[1] == "channel"
    detail = kwargs["detail"]
    assert "SR01:HCM1:SP" in detail
    assert "SR02:VCM3:SP" not in detail


async def test_channel_write_all_blocked_no_emit(tmp_path, monkeypatch):
    """Every op refused by the reference monitor: typed refusal, ZERO emit."""
    from osprey.mcp_server.control_system.server_context import initialize_server_context

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    results = [
        _make_write_result(
            channel="PV:A",
            value=1.0,
            success=False,
            error_message="Write to 'PV:A' blocked: writes are disabled.",
            blocked=True,
            refusal_reason="WRITES_DISABLED",
        ),
        _make_write_result(
            channel="PV:B",
            value=2.0,
            success=False,
            error_message="Write to 'PV:B' blocked: writes are disabled.",
            blocked=True,
            refusal_reason="WRITES_DISABLED",
        ),
    ]
    mock_connector = AsyncMock()
    mock_connector.write_multiple_channels.return_value = results

    conn_patch, validator_patch = _channel_write_patches(mock_connector)
    with conn_patch, validator_patch, patch(f"{_CW_MOD}.notify_agent_activity") as notify:
        fn = _get_channel_write()
        with assert_raises_error(error_type="write_refused"):
            await fn(
                operations=[
                    {"channel": "PV:A", "value": 1.0},
                    {"channel": "PV:B", "value": 2.0},
                ]
            )

    notify.assert_not_called()


async def test_channel_write_full_success_single_emit(tmp_path, monkeypatch):
    """A fully successful batch emits exactly once, naming every channel."""
    from osprey.mcp_server.control_system.server_context import initialize_server_context

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    write_result = _make_write_result(channel="SR01:HCM1:SP", value=42.0)
    mock_connector = AsyncMock()
    mock_connector.write_channel.return_value = write_result

    conn_patch, validator_patch = _channel_write_patches(mock_connector)
    with conn_patch, validator_patch, patch(f"{_CW_MOD}.notify_agent_activity") as notify:
        fn = _get_channel_write()
        result = await fn(operations=[{"channel": "SR01:HCM1:SP", "value": 42.0}])

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert notify.call_count == 1
    assert notify.call_args.args[:2] == ("channel_write", "channel")
    assert notify.call_args.kwargs["detail"] == "SR01:HCM1:SP"


# ── launch_run ──────────────────────────────────────────────────────────────


@pytest.fixture
def _bluesky_context(tmp_path, monkeypatch):
    from osprey.mcp_server.bluesky.server_context import (
        initialize_server_context,
        reset_server_context,
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({"control_system": {"writes_enabled": True}}))
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "valid-token")
    initialize_server_context()
    yield
    reset_server_context()


def _get_launch_run():
    from osprey.mcp_server.bluesky.tools import launch

    return get_tool_fn(launch.launch_run)


async def test_launch_run_success_emits_run_id(_bluesky_context):
    body = {"id": "abc123", "status": "running", "launched_by": "draft"}
    with (
        patch(f"{_LAUNCH_MOD}._http_post_json", return_value=(200, body)),
        patch(f"{_LAUNCH_MOD}.notify_agent_activity") as notify,
    ):
        result = await _get_launch_run()(draft_revision=7)

    data = extract_response_dict(result)
    assert data["status"] == "running"
    assert notify.call_count == 1
    assert notify.call_args.args[:2] == ("launch_run", "run")
    assert notify.call_args.kwargs["detail"] == "abc123"


async def test_launch_run_failure_no_emit(_bluesky_context):
    with (
        patch(
            f"{_LAUNCH_MOD}._http_post_json",
            return_value=(500, {"detail": "launch failed: boom"}),
        ),
        patch(f"{_LAUNCH_MOD}.notify_agent_activity") as notify,
    ):
        with assert_raises_error(error_type="bluesky_bridge_error"):
            await _get_launch_run()(draft_revision=7)

    notify.assert_not_called()


async def test_launch_run_refusal_no_emit(_bluesky_context, monkeypatch):
    """Client-side refusal (no token) never reaches the emit site."""
    from osprey.mcp_server.bluesky.server_context import (
        initialize_server_context,
        reset_server_context,
    )

    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    reset_server_context()
    initialize_server_context()

    with (
        patch(f"{_LAUNCH_MOD}._http_post_json") as post,
        patch(f"{_LAUNCH_MOD}.notify_agent_activity") as notify,
    ):
        with assert_raises_error(error_type="run_launch_unarmed"):
            await _get_launch_run()(draft_revision=7)

    post.assert_not_called()
    notify.assert_not_called()


# ── artifact focus tools ────────────────────────────────────────────────────


async def _save_artifact(title="Test Artifact"):
    from osprey.mcp_server.workspace.tools.artifact_save import artifact_save

    save_fn = get_tool_fn(artifact_save)
    save_result = await save_fn(title=title, content="# Hello", content_type="markdown")
    return extract_response_dict(save_result)["artifact_id"]


async def test_artifact_focus_emits_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact_id = await _save_artifact(title="Orbit Plot")

    from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

    with patch(f"{_FOCUS_MOD}.notify_agent_activity") as notify:
        result = await get_tool_fn(artifact_focus)(artifact_id=artifact_id)

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert notify.call_count == 1
    assert notify.call_args.args[:2] == ("artifact_focus", "artifact")
    assert notify.call_args.kwargs["detail"] == "Orbit Plot"


async def test_artifact_focus_not_found_no_emit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

    with patch(f"{_FOCUS_MOD}.notify_agent_activity") as notify:
        with assert_raises_error(error_type="not_found"):
            await get_tool_fn(artifact_focus)(artifact_id="nonexistent-id")

    notify.assert_not_called()


# ── terminal down: real helper against a dead port ──────────────────────────


async def test_focus_result_unchanged_when_terminal_down(tmp_path, monkeypatch):
    """Real notify_agent_activity against a dead port: tool result unchanged."""
    monkeypatch.chdir(tmp_path)
    artifact_id = await _save_artifact(title="Down Terminal")

    from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

    port = _free_port()
    with patch(
        "osprey.mcp_server.http.web_terminal_url",
        return_value=f"http://127.0.0.1:{port}",
    ):
        result = await get_tool_fn(artifact_focus)(artifact_id=artifact_id)

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["artifact_id"] == artifact_id
    assert data["title"] == "Down Terminal"


async def test_channel_write_result_unchanged_when_terminal_down(tmp_path, monkeypatch):
    """Real notify_agent_activity against a dead port: write result unchanged."""
    from osprey.mcp_server.control_system.server_context import initialize_server_context

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    write_result = _make_write_result(channel="TEST:PV", value=42.0)
    mock_connector = AsyncMock()
    mock_connector.write_channel.return_value = write_result

    port = _free_port()
    conn_patch, validator_patch = _channel_write_patches(mock_connector)
    with (
        conn_patch,
        validator_patch,
        patch(
            "osprey.mcp_server.http.web_terminal_url",
            return_value=f"http://127.0.0.1:{port}",
        ),
    ):
        fn = _get_channel_write()
        result = await fn(operations=[{"channel": "TEST:PV", "value": 42.0}])

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["successful"] == 1
