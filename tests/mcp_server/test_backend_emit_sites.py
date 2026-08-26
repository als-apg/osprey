"""Agent-activity emit sites for backend-direct tools.

Verifies that the mutating backend tools report agent activity via
``notify_agent_activity_async`` — and, just as important, that a path which did
not act emits NOTHING and that tool results are unchanged when the web terminal
is down.

"Did not act" is the test, not "was refused": a control-system write refused in
a readonly run *is* reported, because the operator needs to see that one was
attempted. That emit comes from the shared gate module rather than from either
executor tool, so it has its own patch seam (``_GATES_MOD``) and the two are
asserted separately.

Layout: one ``# ── <tool family> ──`` section per tool family, each owning its
own helpers and fixtures, followed by a trailing cross-cutting "terminal down"
section. New emit sites append a new section immediately BEFORE that trailing
section; shared helpers stay generic and live at module top.

Patch seam: each tool module imports ``notify_agent_activity_async`` directly,
so the mock must target the caller's namespace (e.g.
``osprey.mcp_server.control_system.tools.channel_write.notify_agent_activity_async``),
not ``osprey.mcp_server.http``. The helper owns the thread-hop off the event
loop; that property is pinned once in ``test_notify_agent_activity.py``, not
per tool.
"""

import contextlib
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
_QUEUE_MOD = "osprey.mcp_server.bluesky.tools.queue"
_FOCUS_MOD = "osprey.mcp_server.workspace.tools.focus_tools"
_ARIEL_ENTRY_MOD = "osprey.mcp_server.ariel.tools.entry"
_ARIEL_PUBLISH_MOD = "osprey.mcp_server.ariel.tools.publish"
_PHOEBUS_MOD = "osprey.mcp_server.phoebus.tools.bridge_tools"
_EXECUTE_MOD = "osprey.mcp_server.python_executor.tools.python_execute"
_EXECUTE_FILE_MOD = "osprey.mcp_server.python_executor.tools.python_execute_file"
_LATTICE_MOD = "osprey.mcp_server.workspace.tools.lattice_tools"
_SETUP_MOD = "osprey.mcp_server.workspace.tools.setup"
_SCREEN_MOD = "osprey.mcp_server.workspace.tools.screen_capture"


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
        patch(f"{_CW_MOD}.notify_agent_activity_async") as notify,
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
    with conn_patch, validator_patch, patch(f"{_CW_MOD}.notify_agent_activity_async") as notify:
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
    with conn_patch, validator_patch, patch(f"{_CW_MOD}.notify_agent_activity_async") as notify:
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
    with conn_patch, validator_patch, patch(f"{_CW_MOD}.notify_agent_activity_async") as notify:
        fn = _get_channel_write()
        result = await fn(operations=[{"channel": "SR01:HCM1:SP", "value": 42.0}])

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert notify.call_count == 1
    assert notify.call_args.args[:2] == ("channel_write", "channel")
    assert notify.call_args.kwargs["detail"] == "SR01:HCM1:SP"


# ── bluesky queue tools ─────────────────────────────────────────────────────


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


def _get_queue_tool(name: str):
    from osprey.mcp_server.bluesky.tools import queue

    return get_tool_fn(getattr(queue, name))


async def test_queue_add_success_emits_run_id(_bluesky_context):
    body = {"run_id": "abc123", "revision": 7, "item": {"item_uid": "u1"}}
    with (
        patch(f"{_QUEUE_MOD}._http_post_json", return_value=(200, body)),
        patch(f"{_QUEUE_MOD}.notify_agent_activity_async") as notify,
    ):
        result = await _get_queue_tool("queue_add")(draft_revision=7)

    data = extract_response_dict(result)
    assert data["run_id"] == "abc123"
    assert notify.call_count == 1
    assert notify.call_args.args[:2] == ("queue_add", "run")
    assert notify.call_args.kwargs["detail"] == "abc123"


async def test_queue_add_failure_no_emit(_bluesky_context):
    with (
        patch(
            f"{_QUEUE_MOD}._http_post_json",
            return_value=(500, {"detail": "enqueue failed: boom"}),
        ),
        patch(f"{_QUEUE_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="bluesky_bridge_error"):
            await _get_queue_tool("queue_add")(draft_revision=7)

    notify.assert_not_called()


async def test_queue_start_success_emits(_bluesky_context):
    with (
        patch(f"{_QUEUE_MOD}._http_post_json", return_value=(200, {"started": True, "msg": ""})),
        patch(f"{_QUEUE_MOD}.notify_agent_activity_async") as notify,
    ):
        await _get_queue_tool("queue_start")()

    assert notify.call_count == 1
    assert notify.call_args.args[:2] == ("queue_start", "run")


async def test_queue_start_client_side_refusal_no_emit(_bluesky_context, monkeypatch):
    """A client-side refusal (kill switch off) never reaches the emit site.

    ``queue_start`` refuses before any HTTP call when the bound lane's target
    is not armed for writes, so neither the bridge nor the activity emitter is
    touched.
    """
    # Takes the bound lane's control target: write posture is resolved per
    # target, so the stub has to accept the one `queue_start` passes it.
    monkeypatch.setattr(f"{_QUEUE_MOD}._writes_enabled", lambda _lane_target: False)

    with (
        patch(f"{_QUEUE_MOD}._http_post_json") as post,
        patch(f"{_QUEUE_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="writes_disabled"):
            await _get_queue_tool("queue_start")()

    post.assert_not_called()
    notify.assert_not_called()


# ── artifact focus tools ────────────────────────────────────────────────────


async def _save_artifact(title="Test Artifact"):
    from osprey.mcp_server.workspace.tools.artifact_register import artifact_register

    save_fn = get_tool_fn(artifact_register)
    save_result = await save_fn(title=title, content="# Hello", content_type="markdown")
    return extract_response_dict(save_result)["artifact_id"]


async def test_artifact_focus_emits_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact_id = await _save_artifact(title="Orbit Plot")

    from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

    # Gallery acknowledges the POST — focus reports gallery failures as tool
    # errors, and this test is about the notify seam, not the gallery.
    with (
        patch(f"{_FOCUS_MOD}._post_json_with_response", return_value=(200, {"status": "ok"})),
        patch(f"{_FOCUS_MOD}.notify_agent_activity_async") as notify,
    ):
        result = await get_tool_fn(artifact_focus)(artifact_id=artifact_id)

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert notify.call_count == 1
    assert notify.call_args.args[:2] == ("artifact_focus", "artifact")
    assert notify.call_args.kwargs["detail"] == "Orbit Plot"


async def test_artifact_focus_not_found_no_emit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

    with patch(f"{_FOCUS_MOD}.notify_agent_activity_async") as notify:
        with assert_raises_error(error_type="not_found"):
            await get_tool_fn(artifact_focus)(artifact_id="nonexistent-id")

    notify.assert_not_called()


# ── ariel entry_create / entry_publish ──────────────────────────────────────


@pytest.fixture
def _ariel_context(tmp_path, monkeypatch):
    """Initialize the ARIEL MCP singleton against a throwaway config.yml.

    The singleton is reset afterwards so ARIEL state never leaks into the other
    sections of this module.
    """
    from osprey.mcp_server.ariel.server_context import (
        initialize_ariel_context,
        reset_ariel_context,
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        yaml.dump({"ariel": {"database": {"uri": "postgresql://localhost/test"}}})
    )
    initialize_ariel_context()
    yield
    reset_ariel_context()


def _patch_ariel_service(mock_service):
    """Patch ARIELContext.service to hand every tool the same mock service."""
    return patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    )


def _get_ariel_tool(module_name: str, tool_name: str):
    import importlib

    module = importlib.import_module(f"osprey.mcp_server.ariel.tools.{module_name}")
    return get_tool_fn(getattr(module, tool_name))


async def test_ariel_entry_create_direct_emits_panel(_ariel_context):
    """A direct (non-draft) write emits one passive panel activity for ARIEL."""
    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with (
        _patch_ariel_service(mock_service),
        patch(f"{_ARIEL_ENTRY_MOD}.notify_agent_activity_async") as notify,
    ):
        result = await _get_ariel_tool("entry", "entry_create")(
            subject="Beam lost", details="Injector trip at 03:12", draft=False
        )

    data = extract_response_dict(result)
    entry_id = data["entry_id"]
    assert notify.call_count == 1
    assert notify.call_args.args[:2] == ("entry_create", "panel")
    assert notify.call_args.kwargs["panel"] == "ariel"
    assert notify.call_args.kwargs["detail"] == entry_id


async def test_ariel_entry_create_direct_does_not_steal_focus(_ariel_context):
    """The direct-write branch reports activity passively — no panel focus.

    The draft branch deliberately focuses the ARIEL panel so a human can finish
    the entry; a direct write has nothing for the human to do, so stealing the
    active panel would be a regression.
    """
    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with (
        _patch_ariel_service(mock_service),
        patch(f"{_ARIEL_ENTRY_MOD}.notify_agent_activity_async"),
        patch("osprey.mcp_server.http.notify_panel_focus") as focus,
    ):
        await _get_ariel_tool("entry", "entry_create")(
            subject="Beam lost", details="Injector trip at 03:12", draft=False
        )

    focus.assert_not_called()


async def test_ariel_entry_create_emits_before_attachment_failure(_ariel_context, tmp_path):
    """The entry is persisted, so a later attachment failure must not lose the emit.

    This is why the emit sits between the upsert and attachment processing: the
    tool call ends in an error envelope here, but the entry really does exist.
    """
    attachment = tmp_path / "trip.log"
    attachment.write_text("trip")

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with (
        _patch_ariel_service(mock_service),
        patch(
            "osprey.services.ariel_search.attachments.process_attachments_for_entry",
            new=AsyncMock(side_effect=RuntimeError("attachment store offline")),
        ),
        patch(f"{_ARIEL_ENTRY_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="internal_error"):
            await _get_ariel_tool("entry", "entry_create")(
                subject="Beam lost",
                details="Injector trip at 03:12",
                file_paths=[str(attachment)],
                draft=False,
            )

    assert notify.call_count == 1
    assert notify.call_args.kwargs["panel"] == "ariel"


async def test_ariel_entry_create_validation_refusal_no_emit(_ariel_context):
    """Argument validation refuses before any write, so nothing is emitted."""
    with patch(f"{_ARIEL_ENTRY_MOD}.notify_agent_activity_async") as notify:
        with assert_raises_error(error_type="validation_error"):
            await _get_ariel_tool("entry", "entry_create")(
                subject="   ", details="Injector trip at 03:12", draft=False
            )

    notify.assert_not_called()


async def test_ariel_entry_create_upsert_failure_no_emit(_ariel_context):
    """A failed upsert means no entry exists — emit nothing."""
    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.side_effect = RuntimeError("database unreachable")

    with (
        _patch_ariel_service(mock_service),
        patch(f"{_ARIEL_ENTRY_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="internal_error"):
            await _get_ariel_tool("entry", "entry_create")(
                subject="Beam lost", details="Injector trip at 03:12", draft=False
            )

    notify.assert_not_called()


async def test_ariel_entry_publish_success_emits_facility_id(_ariel_context):
    """A successful upstream publish emits the facility-assigned entry id."""
    from osprey.services.ariel_search.models import FacilityEntryCreateResult, SyncStatus

    mock_service = AsyncMock()
    mock_service.publish_entry.return_value = FacilityEntryCreateResult(
        entry_id="published-001",
        source_system="ALS eLog",
        sync_status=SyncStatus.SYNCED,
        message="Published successfully",
    )

    with (
        _patch_ariel_service(mock_service),
        patch(f"{_ARIEL_PUBLISH_MOD}.notify_agent_activity_async") as notify,
    ):
        result = await _get_ariel_tool("publish", "entry_publish")(
            entry_id="e1", logbook="Operations"
        )

    data = extract_response_dict(result)
    assert data["entry_id"] == "published-001"
    assert notify.call_count == 1
    assert notify.call_args.args[:2] == ("entry_publish", "panel")
    assert notify.call_args.kwargs["panel"] == "ariel"
    assert notify.call_args.kwargs["detail"] == "published-001"


async def test_ariel_entry_publish_validation_refusal_no_emit(_ariel_context):
    """An empty entry_id is refused before the service is touched."""
    with patch(f"{_ARIEL_PUBLISH_MOD}.notify_agent_activity_async") as notify:
        with assert_raises_error(error_type="validation_error"):
            await _get_ariel_tool("publish", "entry_publish")(entry_id="")

    notify.assert_not_called()


async def test_ariel_entry_publish_not_found_no_emit(_ariel_context):
    """Nothing was published, so nothing is emitted."""
    mock_service = AsyncMock()
    mock_service.publish_entry.side_effect = KeyError("Entry e99 not found")

    with (
        _patch_ariel_service(mock_service),
        patch(f"{_ARIEL_PUBLISH_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="not_found"):
            await _get_ariel_tool("publish", "entry_publish")(entry_id="e99")

    notify.assert_not_called()


async def test_ariel_entry_publish_auth_required_no_emit(_ariel_context):
    """A credential refusal blocks the upstream write — emit nothing."""
    from osprey.services.ariel_search.exceptions import AuthenticationRequiredError

    mock_service = AsyncMock()
    mock_service.publish_entry.side_effect = AuthenticationRequiredError(
        "OLOG publishing requires credentials.", source_system="ALS eLog"
    )

    with (
        _patch_ariel_service(mock_service),
        patch(f"{_ARIEL_PUBLISH_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="auth_required"):
            await _get_ariel_tool("publish", "entry_publish")(entry_id="e1")

    notify.assert_not_called()


async def test_ariel_entry_publish_not_supported_no_emit(_ariel_context):
    """An adapter without write support never publishes — emit nothing."""
    mock_service = AsyncMock()
    mock_service.publish_entry.side_effect = NotImplementedError(
        "Adapter does not support writing entries"
    )

    with (
        _patch_ariel_service(mock_service),
        patch(f"{_ARIEL_PUBLISH_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="not_supported"):
            await _get_ariel_tool("publish", "entry_publish")(entry_id="e1")

    notify.assert_not_called()


# ── phoebus_drive ───────────────────────────────────────────────────────────
#
# Bridge contract this section pins: on HTTP 200 the bridge reports ``fired``,
# which is true only when a real interactive control was driven. A synthetic
# drive that resolves no control still answers 200 with ``fired: false`` and
# nothing was written — that must stay silent. A semantic ``type`` writes the
# widget's PV through the runtime without firing a GUI control, so it reports
# ``fired: false`` even though the value landed — that must emit.


@pytest.fixture
def _phoebus_active_display_allowed(monkeypatch):
    """Pin require-handle mode OFF so ``display="active"`` reaches the bridge.

    Without this the setting falls through to whatever config.yml the test
    process resolves, which would make the emit paths depend on ambient state.
    """
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "0")


def _get_phoebus_drive():
    from osprey.mcp_server.phoebus.tools.bridge_tools import phoebus_drive

    return get_tool_fn(phoebus_drive)


def _phoebus_bridge(status: int, body: dict):
    """Patch the drive HTTP boundary with one canned bridge response."""
    return patch(f"{_PHOEBUS_MOD}._http_post_drive", return_value=(status, body))


async def test_phoebus_drive_click_fired_emits(_phoebus_active_display_allowed):
    """A click that fired a control wrote to the machine — emit once, kind channel."""
    with (
        _phoebus_bridge(200, {"fired": True, "detail": "fired via ButtonBase.fire()"}),
        patch(f"{_PHOEBUS_MOD}.notify_agent_activity_async") as notify,
    ):
        result = await _get_phoebus_drive()(widget="SetButton", verb="click")

    assert extract_response_dict(result)["fired"] is True
    notify.assert_called_once_with("phoebus_drive", "channel", detail="click SetButton on active")


async def test_phoebus_drive_synthetic_type_fired_emits():
    """A synthetic type that fired echoes the normalized verb and the display ref."""
    with (
        _phoebus_bridge(200, {"fired": True, "detail": "committed 42"}),
        patch(f"{_PHOEBUS_MOD}.notify_agent_activity_async") as notify,
    ):
        await _get_phoebus_drive()(widget="Setpoint", verb="TYPE", value="42", display="handle:d-3")

    notify.assert_called_once_with("phoebus_drive", "channel", detail="type Setpoint on handle:d-3")


async def test_phoebus_drive_synthetic_200_not_fired_no_emit(_phoebus_active_display_allowed):
    """Bridge contract: a synthetic 200 with fired=false resolved no control — no write."""
    with (
        _phoebus_bridge(200, {"fired": False, "detail": "no interactive control resolved"}),
        patch(f"{_PHOEBUS_MOD}.notify_agent_activity_async") as notify,
    ):
        result = await _get_phoebus_drive()(widget="Readback", verb="type", value="42")

    assert extract_response_dict(result)["fired"] is False
    notify.assert_not_called()


@pytest.mark.parametrize("verb,mode", [("type", "semantic"), ("TYPE", "SEMANTIC")])
async def test_phoebus_drive_semantic_type_not_fired_emits(
    verb, mode, _phoebus_active_display_allowed
):
    """Bridge contract: semantic type writes the PV directly and reports fired=false — emit.

    Parametrized over verb/mode case because the emit condition compares the
    NORMALIZED ``verb_l``/``mode_l``: comparing the raw arguments would silence
    this write for the equally supported upper-case spelling.
    """
    with (
        _phoebus_bridge(200, {"fired": False, "detail": "wrote PV SR:CORR:SP"}),
        patch(f"{_PHOEBUS_MOD}.notify_agent_activity_async") as notify,
    ):
        await _get_phoebus_drive()(widget="Setpoint", verb=verb, value="1.5", mode=mode)

    notify.assert_called_once_with("phoebus_drive", "channel", detail="type Setpoint on active")


async def test_phoebus_drive_semantic_click_bypass_no_emit(_phoebus_active_display_allowed):
    """Semantic mode bypasses click entirely — nothing was driven, so emit nothing."""
    with (
        _phoebus_bridge(200, {"fired": False, "detail": "semantic mode bypasses click"}),
        patch(f"{_PHOEBUS_MOD}.notify_agent_activity_async") as notify,
    ):
        await _get_phoebus_drive()(widget="SetButton", verb="click", mode="semantic")

    notify.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"widget": "0", "verb": "frobnicate"},
        {"widget": "0", "verb": "click", "mode": "warp"},
        {"widget": "0", "verb": "type"},
    ],
)
async def test_phoebus_drive_validation_refusal_no_emit(kwargs, _phoebus_active_display_allowed):
    """Argument validation refuses before the bridge is contacted — emit nothing."""
    with (
        patch(f"{_PHOEBUS_MOD}._http_post_drive") as post,
        patch(f"{_PHOEBUS_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="validation_error"):
            await _get_phoebus_drive()(**kwargs)

    post.assert_not_called()
    notify.assert_not_called()


async def test_phoebus_drive_handle_required_refusal_no_emit(monkeypatch):
    """Require-handle mode refuses implicit 'active' before any drive — emit nothing."""
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "1")

    with (
        patch(f"{_PHOEBUS_MOD}._http_post_drive") as post,
        patch(f"{_PHOEBUS_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="phoebus_handle_required"):
            await _get_phoebus_drive()(widget="SetButton", verb="click")

    post.assert_not_called()
    notify.assert_not_called()


async def test_phoebus_drive_bridge_rejected_no_emit(_phoebus_active_display_allowed):
    """A non-200 from the bridge means no widget was driven — emit nothing."""
    with (
        _phoebus_bridge(400, {"error": "Unknown widget '0'", "status": 400}),
        patch(f"{_PHOEBUS_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="phoebus_rejected"):
            await _get_phoebus_drive()(widget="0", verb="click")

    notify.assert_not_called()


async def test_phoebus_drive_unreachable_no_emit(_phoebus_active_display_allowed):
    """An unreachable bridge never drove anything — emit nothing."""
    import urllib.error

    with (
        patch(f"{_PHOEBUS_MOD}._http_post_drive", side_effect=urllib.error.URLError("refused")),
        patch(f"{_PHOEBUS_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="phoebus_unreachable"):
            await _get_phoebus_drive()(widget="SetButton", verb="click")

    notify.assert_not_called()


# ── python executor: execute / execute_file ─────────────────────────────────
#
# Launch discriminator this section pins: ``ExecutionResult.execution_time_seconds``
# is None only for a result built by ``execute_code``'s setup handler, i.e. the
# script was never handed to a subprocess. Every result that came back from the
# subprocess path carries an elapsed time — including the timed-out and the
# script-error returns — and those scripts may already have written to the
# machine, so a failed run still has to report.
#
# Mode: the emit condition is the complement of the readonly gate
# (``execution_mode != "readonly"``), not equality with "readwrite". The mode
# argument is an unvalidated free string, so any other spelling reaches the
# subprocess with its writes intact and must stay visible.

_EXECUTE_WRITE_CODE = "epics.caput('SR:CORR:SP', 1.0)\n"
_EXECUTE_DETAIL = "ran a script with control-system writes"
_EXECUTE_TOOLS = ["execute", "execute_file"]

#: Second patch seam for this family. A refused write is reported by the shared
#: gate module rather than by either tool, so ``{tool_mod}.notify_agent_activity_async``
#: does not see it — which is what lets a test assert "the run emitted nothing"
#: and "the refusal was reported" independently of each other.
_GATES_MOD = "osprey.mcp_server.python_executor.tools._execution_gates"


@contextlib.contextmanager
def _audit_to(root):
    """Point the refusal audit log at *root* instead of the real state zone."""
    from osprey.services.python_executor import refusal_audit

    with patch.object(refusal_audit, "audit_dir", lambda: root / "var" / "audit"):
        yield


def _audit_records(root):
    """Every refusal recorded under *root*, or ``[]`` when none were."""
    import json

    from osprey.services.python_executor.refusal_audit import REFUSAL_LOG_FILENAME

    path = root / "var" / "audit" / REFUSAL_LOG_FILENAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _execute_patterns(has_writes: bool) -> dict:
    """Canned ``detect_control_system_operations()`` return value."""
    return {
        "has_writes": has_writes,
        "has_reads": False,
        "detected_patterns": {"caput": ["caput('SR:CORR:SP', 1.0)"]} if has_writes else {},
    }


def _execute_result(*, success=True, launched=True, stderr="", error_message=None):
    """Build one adapter outcome.

    ``launched=False`` reproduces the shape ``execute_code`` returns from its
    setup handler (no execution folder, no subprocess, no elapsed time).
    """
    from osprey.mcp_server.python_executor.executor import ExecutionResult

    return ExecutionResult(
        success=success,
        stdout="",
        stderr=stderr,
        execution_time_seconds=0.25 if launched else None,
        error_message=error_message,
    )


def _execute_tool_call(tool_name: str, tmp_path):
    """Bind one executor tool to a uniform ``(patch_module, call)`` pair.

    Both tools run the same write-bearing source; ``execute_file`` reads it from
    a script inside the (patched) project root so the containment check passes.
    """
    if tool_name == "execute":
        from osprey.mcp_server.python_executor.tools.python_execute import execute

        fn = get_tool_fn(execute)

        async def call(**kwargs):
            return await fn(
                code=_EXECUTE_WRITE_CODE,
                description="write script",
                save_output=False,
                **kwargs,
            )

        return _EXECUTE_MOD, call

    from osprey.mcp_server.python_executor.tools.python_execute_file import execute_file

    fn = get_tool_fn(execute_file)
    script = tmp_path / "write_script.py"
    script.write_text(_EXECUTE_WRITE_CODE, encoding="utf-8")

    async def call(**kwargs):
        return await fn(
            file_path=str(script),
            description="write script",
            save_output=False,
            **kwargs,
        )

    return _EXECUTE_FILE_MOD, call


@contextlib.contextmanager
def _execute_env(mod, tmp_path, *, exec_result=None, has_writes=True, writes_enabled=True):
    """Patch every boundary the executor tools cross; yield ``(notify, execute_code)``."""
    from osprey.services.python_executor.execution.control import ExecutionControlConfig

    exec_code = AsyncMock(return_value=exec_result if exec_result else _execute_result())
    with (
        patch(
            "osprey.mcp_server.python_executor.executor._resolve_project_root",
            return_value=tmp_path,
        ),
        patch(
            "osprey.services.python_executor.analysis.pattern_detection"
            ".detect_control_system_operations",
            return_value=_execute_patterns(has_writes),
        ),
        patch(
            "osprey.services.python_executor.execution.control.get_execution_control_config",
            return_value=ExecutionControlConfig(control_system_writes_enabled=writes_enabled),
        ),
        patch("osprey.mcp_server.python_executor.executor.execute_code", exec_code),
        patch(f"{mod}.notify_agent_activity_async") as notify,
    ):
        yield notify, exec_code


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_write_run_emits(tool_name, tmp_path, monkeypatch):
    """A readwrite run of a write-bearing script reports once, kind channel."""
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)

    with _execute_env(mod, tmp_path) as (notify, _):
        result = await call(execution_mode="readwrite")

    assert extract_response_dict(result)["has_errors"] is False
    notify.assert_called_once_with(tool_name, "channel", detail=_EXECUTE_DETAIL)


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_script_error_after_launch_still_emits(tool_name, tmp_path, monkeypatch):
    """The script ran and then raised — writes before the error already landed."""
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)
    failed = _execute_result(
        success=False,
        launched=True,
        stderr="Traceback...\nNameError: name 'undefined' is not defined",
        error_message="NameError: name 'undefined' is not defined",
    )

    with _execute_env(mod, tmp_path, exec_result=failed) as (notify, _):
        with assert_raises_error(error_type="execution_error"):
            await call(execution_mode="readwrite")

    notify.assert_called_once_with(tool_name, "channel", detail=_EXECUTE_DETAIL)


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_launch_failure_no_emit(tool_name, tmp_path, monkeypatch):
    """Sandbox setup failed before any subprocess started — nothing ran, nothing wrote."""
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)
    never_launched = _execute_result(
        success=False,
        launched=False,
        stderr="Traceback...\nOSError: no execution folder",
        error_message="Execution setup failed: no execution folder",
    )

    with _execute_env(mod, tmp_path, exec_result=never_launched) as (notify, _):
        with assert_raises_error(error_type="execution_error"):
            await call(execution_mode="readwrite")

    notify.assert_not_called()


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_readonly_gate_refusal_alerts_the_operator(tool_name, tmp_path, monkeypatch):
    """A refused write is an operator-visible event, not just an error to the agent.

    The refusal emit comes from the shared gate module, not the tool module, so
    the two seams are asserted separately: the tool's own "ran a script with
    control-system writes" emit must stay silent (nothing ran), while the gate
    reports that a write was attempted and blocked.
    """
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)

    with _execute_env(mod, tmp_path) as (notify, exec_code):
        with patch(f"{_GATES_MOD}.notify_agent_activity_async") as blocked, _audit_to(tmp_path):
            with assert_raises_error(error_type="safety_error"):
                await call(execution_mode="readonly")

    exec_code.assert_not_called()
    notify.assert_not_called()
    blocked.assert_called_once()
    emitted_tool, kind = blocked.call_args.args
    assert (emitted_tool, kind) == (tool_name, "channel")
    assert "BLOCKED" in blocked.call_args.kwargs["detail"]


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_readonly_gate_refusal_is_audited(tool_name, tmp_path, monkeypatch):
    """...and leaves a durable record naming the source that was refused."""
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)

    with _execute_env(mod, tmp_path) as (_, exec_code):
        with patch(f"{_GATES_MOD}.notify_agent_activity_async"), _audit_to(tmp_path):
            with assert_raises_error(error_type="safety_error"):
                await call(execution_mode="readonly")

    (record,) = _audit_records(tmp_path)
    assert record["layer"] == "pattern_detection"
    assert record["mode"] == "readonly"
    assert record["tool"] == tool_name
    assert _EXECUTE_WRITE_CODE.strip() in record["source"]


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_runtime_refusal_alerts_and_audits(tool_name, tmp_path, monkeypatch):
    """The guard refused mid-run: the tool recognises it in stderr and reports.

    This is the layer that catches the spellings no static check sees, so it is
    the one that most needs to reach the operator. Nothing but the subprocess's
    stderr carries the news back.
    """
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL

    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)
    refused = _execute_result(
        success=False,
        launched=True,
        stderr=f"Traceback...\nRuntimeError: {READONLY_REFUSAL}",
        error_message=f"RuntimeError: {READONLY_REFUSAL}",
    )

    with _execute_env(mod, tmp_path, exec_result=refused, has_writes=False) as (notify, _):
        with patch(f"{_GATES_MOD}.notify_agent_activity_async") as blocked, _audit_to(tmp_path):
            with assert_raises_error(error_type="execution_error"):
                await call(execution_mode="readonly")

    notify.assert_not_called()
    blocked.assert_called_once()
    (record,) = _audit_records(tmp_path)
    assert record["layer"] == "runtime_guard"
    assert record["tool"] == tool_name


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_connector_refusal_is_reported_too(tool_name, tmp_path, monkeypatch):
    """A write refused by the connector — the *approved* path in the wrong mode.

    Its message is not the guard's, so this pins that the tool matches on what
    the two share, and that the audit record keeps the channel name the
    connector's message carries rather than a generic marker.
    """
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)
    connector_message = (
        "Write to 'SR:CORR:SP' blocked: this script is running in readonly "
        "execution mode. Resubmit it with execution_mode='readwrite' "
        "(human approval required) if the write is intended."
    )
    refused = _execute_result(
        success=False,
        launched=True,
        stderr=f"Traceback...\nChannelWriteBlockedError: {connector_message}",
        error_message=f"ChannelWriteBlockedError: {connector_message}",
    )

    with _execute_env(mod, tmp_path, exec_result=refused, has_writes=False) as (_, _exec):
        with patch(f"{_GATES_MOD}.notify_agent_activity_async") as blocked, _audit_to(tmp_path):
            with assert_raises_error(error_type="execution_error"):
                await call(execution_mode="readonly")

    blocked.assert_called_once()
    (record,) = _audit_records(tmp_path)
    assert record["layer"] == "runtime_guard"
    assert any("SR:CORR:SP" in line for line in record["trigger"])


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_ordinary_failure_is_not_reported_as_a_refusal(
    tool_name, tmp_path, monkeypatch
):
    """A script that merely crashed must not land in the refusal audit log."""
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)
    crashed = _execute_result(
        success=False,
        launched=True,
        stderr="Traceback...\nZeroDivisionError: division by zero",
        error_message="ZeroDivisionError: division by zero",
    )

    with _execute_env(mod, tmp_path, exec_result=crashed, has_writes=False) as (notify, _):
        with patch(f"{_GATES_MOD}.notify_agent_activity_async") as blocked, _audit_to(tmp_path):
            with assert_raises_error(error_type="execution_error"):
                await call(execution_mode="readonly")

    notify.assert_not_called()
    blocked.assert_not_called()
    assert _audit_records(tmp_path) == []


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_safety_check_refusal_no_emit(tool_name, tmp_path, monkeypatch):
    """The pre-execution safety check refuses before launch — emit nothing."""
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)

    with _execute_env(mod, tmp_path) as (notify, exec_code):
        with patch(
            "osprey.services.python_executor.analysis.safety_checks.quick_safety_check",
            return_value=(False, ["subprocess use is blocked"]),
        ):
            with assert_raises_error(error_type="safety_error"):
                await call(execution_mode="readwrite")

    exec_code.assert_not_called()
    notify.assert_not_called()


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_readwrite_without_write_patterns_no_emit(tool_name, tmp_path, monkeypatch):
    """readwrite alone is not a write: with no detected write patterns, stay silent."""
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)

    with _execute_env(mod, tmp_path, has_writes=False) as (notify, exec_code):
        await call(execution_mode="readwrite")

    exec_code.assert_called_once()
    notify.assert_not_called()


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_uncanonical_write_mode_rejected_no_emit(tool_name, tmp_path, monkeypatch):
    """A non-canonical mode spelling is refused at the boundary — emit nothing.

    "READWRITE" used to clear both write gates as an unvalidated free string
    and run the script; it is rejected before any gate, so nothing runs
    and nothing reports.
    """
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)

    with _execute_env(mod, tmp_path) as (notify, exec_code):
        with assert_raises_error(error_type="validation_error"):
            await call(execution_mode="READWRITE")

    exec_code.assert_not_called()
    notify.assert_not_called()


@pytest.mark.parametrize("tool_name", _EXECUTE_TOOLS)
async def test_execute_deployment_writes_disabled_no_emit(tool_name, tmp_path, monkeypatch):
    """The deployment kill switch refuses before launch — emit nothing."""
    monkeypatch.chdir(tmp_path)
    mod, call = _execute_tool_call(tool_name, tmp_path)

    with _execute_env(mod, tmp_path, writes_enabled=False) as (notify, exec_code):
        with assert_raises_error(error_type="safety_error"):
            await call(execution_mode="readwrite")

    exec_code.assert_not_called()
    notify.assert_not_called()


# ── lattice dashboard mutators ──────────────────────────────────────────────
#
# The six mutators share one refusal shape: every failure arrives as an
# exception out of ``_dashboard_request`` and every handler funnels into
# ``make_error``, which raises before the emit can run. Both failure classes
# the module handles by name — an unreachable dashboard and a non-2xx response
# — are therefore exercised against all six rather than per-mutator variants.
# The read-only tools share the same body minus the emit, so they are pinned
# as a group too.

_LATTICE_MUTATORS = [
    ("lattice_init", {"lattice_path": "machine_data/als.m"}),
    ("lattice_set_param", {"family": "QF", "value": 1.25}),
    ("lattice_refresh", {}),
    ("lattice_set_baseline", {}),
    ("lattice_update_settings", {"settings": {"da": {"n_angles": 25}}}),
    ("lattice_clear_baseline", {}),
]

_LATTICE_READERS = [
    ("lattice_state", {}),
    ("lattice_get_figure", {"name": "optics"}),
    ("lattice_get_data", {"name": "optics"}),
    ("lattice_get_settings", {}),
]


def _get_lattice_tool(name: str):
    from osprey.mcp_server.workspace.tools import lattice_tools

    return get_tool_fn(getattr(lattice_tools, name))


def _lattice_request(*, return_value=None, side_effect=None):
    """Patch the dashboard HTTP boundary with a canned answer or failure."""
    body = {"summary": {"energy": 2.0}, "families": {"QF": {}}}
    return patch(
        f"{_LATTICE_MOD}._dashboard_request",
        new=AsyncMock(
            return_value=body if return_value is None else return_value, side_effect=side_effect
        ),
    )


def _lattice_unreachable():
    import httpx

    return httpx.ConnectError("connection refused")


def _lattice_http_error(status: int = 400, text: str = "unknown family"):
    import httpx

    request = httpx.Request("POST", "http://dash/api")
    return httpx.HTTPStatusError(
        "bad status", request=request, response=httpx.Response(status, text=text, request=request)
    )


@pytest.mark.parametrize(
    "tool_name,kwargs,detail",
    [
        ("lattice_init", {"lattice_path": "machine_data/als.m"}, "machine_data/als.m"),
        ("lattice_set_param", {"family": "QF", "value": 1.25}, "QF = 1.25"),
        ("lattice_refresh", {}, "recomputing fast figures"),
        ("lattice_refresh", {"figure": "da"}, "recomputing da"),
        ("lattice_set_baseline", {}, "baseline set"),
        (
            "lattice_update_settings",
            {"settings": {"lma": {"n_steps": 8}, "da": {"n_angles": 25}}},
            "settings: da, lma",
        ),
        ("lattice_clear_baseline", {}, "baseline cleared"),
    ],
)
async def test_lattice_mutator_emits(tool_name, kwargs, detail):
    """Each acknowledged mutation reports once against the 'lattice' panel."""
    with (
        _lattice_request(),
        patch(f"{_LATTICE_MOD}.notify_agent_activity_async") as notify,
    ):
        result = await _get_lattice_tool(tool_name)(**kwargs)

    assert extract_response_dict(result)
    notify.assert_called_once_with(tool_name, "panel", panel="lattice", detail=detail)


@pytest.mark.parametrize("tool_name,kwargs", _LATTICE_MUTATORS)
async def test_lattice_mutator_unreachable_dashboard_no_emit(tool_name, kwargs):
    """A dashboard that never answered changed nothing — emit nothing."""
    with (
        _lattice_request(side_effect=_lattice_unreachable()),
        patch(f"{_LATTICE_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="service_unavailable"):
            await _get_lattice_tool(tool_name)(**kwargs)

    notify.assert_not_called()


@pytest.mark.parametrize("tool_name,kwargs", _LATTICE_MUTATORS)
async def test_lattice_mutator_rejected_request_no_emit(tool_name, kwargs):
    """A non-2xx answer means the dashboard refused the change — emit nothing."""
    with (
        _lattice_request(side_effect=_lattice_http_error()),
        patch(f"{_LATTICE_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="lattice_error"):
            await _get_lattice_tool(tool_name)(**kwargs)

    notify.assert_not_called()


@pytest.mark.parametrize("tool_name,kwargs", _LATTICE_READERS)
async def test_lattice_read_only_tool_never_emits(tool_name, kwargs):
    """Reading state or a figure mutates nothing, so it stays out of the feed."""
    with (
        _lattice_request(return_value={"status": "ok"}),
        patch(f"{_LATTICE_MOD}.notify_agent_activity_async") as notify,
    ):
        await _get_lattice_tool(tool_name)(**kwargs)

    notify.assert_not_called()


# ── setup_patch / manage_window ─────────────────────────────────────────────
#
# setup_patch edits the two files that hold the agent's credentials, so its
# detail names the file and the key path and NOTHING else: the ring is
# persistent and served back over HTTP, and a leaked `.mcp.json` value is an
# API key. The leak tests use sentinel values distinctive enough that no
# coincidental substring can let a leak through, and check every notify
# argument rather than just the detail.
#
# The keys patched below are all outside the protected set — `command`, `args`,
# `env.*` and every `control_system.*` key are refused before the file is read
# (tests/mcp_server/test_setup_patch_protected.py), so a landed-patch emit can
# only be observed on a key an agent-side writer may actually set. The sentinels
# therefore ride on an unprotected `.mcp.json` key; the file still carries a
# real-looking credential beside it, which is what makes the value exclusion
# worth asserting at all.
#
# Both tools refuse before touching anything, so each refusal shape is pinned
# as a no-emit: an out-of-whitelist file, a malformed key path, a missing
# file and an unparseable file for setup_patch; an unknown action, both
# incomplete-parameter shapes and a failing backend for manage_window.

_OLD_SENTINEL = "zzqqOLDVALUEsentinel42"
_NEW_SENTINEL = "wwvvNEWVALUEsentinel99"

_MANAGE_WINDOW_ACTIONS = [
    ("bring_to_front", {}),
    ("move", {"x": 10, "y": 20}),
    ("resize", {"width": 800, "height": 600}),
]


@pytest.fixture
def setup_project(tmp_path):
    """Project root holding both patchable files, with config resolution pinned."""
    (tmp_path / "config.yml").write_text(
        yaml.dump({"control_system": {"type": "mock", "writes_enabled": False}})
    )
    (tmp_path / ".mcp.json").write_text(
        '{\n  "mcpServers": {\n    "demo": {\n'
        f'      "description": "{_OLD_SENTINEL}",\n'
        '      "env": {\n        "API_KEY": "sk-live-credential"\n'
        "      }\n    }\n  }\n}\n"
    )
    with patch(f"{_SETUP_MOD}.resolve_config_path", return_value=tmp_path / "config.yml"):
        yield tmp_path


def _get_setup_patch():
    from osprey.mcp_server.workspace.tools.setup import setup_patch

    return get_tool_fn(setup_patch)


def _get_manage_window():
    from osprey.mcp_server.workspace.tools.screen_capture import manage_window

    return get_tool_fn(manage_window)


def _screen_backend(*, side_effect=None):
    """Patch the window backend with an async stub, optionally a failing one."""
    backend = AsyncMock()
    if side_effect is not None:
        backend.bring_to_front.side_effect = side_effect
        backend.move_window.side_effect = side_effect
        backend.resize_window.side_effect = side_effect
    return patch(f"{_SCREEN_MOD}.get_backend", return_value=backend)


def _backend_unavailable():
    from osprey.mcp_server.workspace.tools.screen_capture_backends import (
        BackendUnavailableError,
    )

    return BackendUnavailableError("no window manager", ["Install wmctrl."])


async def test_setup_patch_emits_config_activity(setup_project):
    """A landed patch reports the file and key path under the 'config' kind."""
    with patch(f"{_SETUP_MOD}.notify_agent_activity_async") as notify:
        result = await _get_setup_patch()(file="config.yml", key_path="ui.theme", value="dark")

    assert extract_response_dict(result)["key_path"] == "ui.theme"
    notify.assert_called_once_with("setup_patch", "config", detail="config.yml: ui.theme")


async def test_setup_patch_emits_for_json_target(setup_project):
    """The `.mcp.json` branch reports too — it is the same mutation."""
    with patch(f"{_SETUP_MOD}.notify_agent_activity_async") as notify:
        await _get_setup_patch()(
            file=".mcp.json", key_path="mcpServers.demo.description", value=_NEW_SENTINEL
        )

    notify.assert_called_once_with(
        "setup_patch", "config", detail=".mcp.json: mcpServers.demo.description"
    )


async def test_setup_patch_detail_never_carries_the_patched_values(setup_project):
    """Hard security bound: neither the old nor the new value may reach the feed."""
    import json

    with patch(f"{_SETUP_MOD}.notify_agent_activity_async") as notify:
        await _get_setup_patch()(
            file=".mcp.json", key_path="mcpServers.demo.description", value=_NEW_SENTINEL
        )

    # The patch really did swap the sentinels, so their absence below is a
    # property of the emit and not of an inert test.
    on_disk = json.loads((setup_project / ".mcp.json").read_text())
    assert on_disk["mcpServers"]["demo"]["description"] == _NEW_SENTINEL

    call = notify.call_args
    reported = [str(arg) for arg in call.args] + [str(v) for v in call.kwargs.values()]
    for sentinel in (_OLD_SENTINEL, _NEW_SENTINEL):
        for piece in reported:
            assert sentinel not in piece, f"value leaked into the activity feed: {piece!r}"


async def test_setup_patch_marks_control_system_keys_as_safety_config():
    """A safety-relevant key path is distinguishable at a glance in the feed.

    Asserted at the detail builder rather than through the tool: every
    `control_system.*` key is protected now, so the tool refuses such a patch
    before it can emit a landed-patch detail (the refusal emits its own, pinned
    in tests/mcp_server/test_setup_patch_protected.py). The marker still has to
    be right — `_activity_detail` is what any future emit site would call — so
    it is pinned where it can still be observed.
    """
    from osprey.mcp_server.workspace.tools.setup import _activity_detail

    assert (
        _activity_detail("config.yml", "control_system.writes_enabled")
        == "safety config — config.yml: control_system.writes_enabled"
    )


async def test_setup_patch_safety_prefix_is_exact_case(setup_project):
    """The prefix match is case-sensitive, like the hot/cold key lookups.

    Nothing normalises `key_path`, so both spellings are pinned: the canonical
    lowercase one carries the marker and a differently-cased path — which names
    a different key, and gets no hot/cold note either — does not.
    """
    with patch(f"{_SETUP_MOD}.notify_agent_activity_async") as notify:
        await _get_setup_patch()(
            file="config.yml", key_path="Control_System.writes_enabled", value="true"
        )

    notify.assert_called_once_with(
        "setup_patch", "config", detail="config.yml: Control_System.writes_enabled"
    )


async def test_setup_patch_unpatchable_file_no_emit(setup_project):
    """A file outside the whitelist was never opened — emit nothing."""
    with patch(f"{_SETUP_MOD}.notify_agent_activity_async") as notify:
        with assert_raises_error(error_type="validation_error"):
            await _get_setup_patch()(file="settings.json", key_path="permissions.deny", value="[]")

    notify.assert_not_called()


@pytest.mark.parametrize("key_path", ["", "../../etc/passwd", "/abs/path", "has space"])
async def test_setup_patch_invalid_key_path_no_emit(setup_project, key_path):
    """A rejected key path changed nothing — emit nothing."""
    with patch(f"{_SETUP_MOD}.notify_agent_activity_async") as notify:
        with assert_raises_error(error_type="validation_error"):
            await _get_setup_patch()(file="config.yml", key_path=key_path, value="1")

    notify.assert_not_called()


async def test_setup_patch_missing_file_no_emit(setup_project):
    """Nothing to patch means nothing to report."""
    (setup_project / ".mcp.json").unlink()

    with patch(f"{_SETUP_MOD}.notify_agent_activity_async") as notify:
        with assert_raises_error(error_type="not_found"):
            await _get_setup_patch()(
                file=".mcp.json", key_path="mcpServers.demo.description", value="1"
            )

    notify.assert_not_called()


async def test_setup_patch_unparseable_file_no_emit(setup_project):
    """A file that could not be read back was never rewritten — emit nothing."""
    (setup_project / ".mcp.json").write_text("{ this is not json")

    with patch(f"{_SETUP_MOD}.notify_agent_activity_async") as notify:
        with assert_raises_error(error_type="internal_error"):
            await _get_setup_patch()(
                file=".mcp.json", key_path="mcpServers.demo.description", value="1"
            )

    notify.assert_not_called()


@pytest.mark.parametrize("action,kwargs", _MANAGE_WINDOW_ACTIONS)
async def test_manage_window_emits_ui_activity(action, kwargs):
    """Each completed window action reports once under the 'ui' kind."""
    with (
        _screen_backend(),
        patch(f"{_SCREEN_MOD}.notify_agent_activity_async") as notify,
    ):
        result = await _get_manage_window()(app="Phoebus", action=action, **kwargs)

    assert extract_response_dict(result)["status"] == "success"
    notify.assert_called_once_with("manage_window", "ui", detail=f"{action} Phoebus")


async def test_manage_window_detail_preserves_app_name_case():
    """Nothing normalises `app`, so the feed shows the name the operator sees."""
    with (
        _screen_backend(),
        patch(f"{_SCREEN_MOD}.notify_agent_activity_async") as notify,
    ):
        await _get_manage_window()(app="google Chrome", action="bring_to_front")

    notify.assert_called_once_with("manage_window", "ui", detail="bring_to_front google Chrome")


@pytest.mark.parametrize("action", ["fullscreen", "Bring_To_Front", "MOVE"])
async def test_manage_window_unknown_action_no_emit(action):
    """Action matching is exact-case; every unmatched spelling is refused, not reported."""
    with (
        _screen_backend(),
        patch(f"{_SCREEN_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="validation_error"):
            await _get_manage_window()(app="Phoebus", action=action, x=1, y=2)

    notify.assert_not_called()


@pytest.mark.parametrize(
    "action,kwargs",
    [
        ("move", {}),
        ("move", {"x": 10}),
        ("resize", {}),
        ("resize", {"width": 800}),
    ],
)
async def test_manage_window_incomplete_parameters_no_emit(action, kwargs):
    """A refusal before the backend call moved no window — emit nothing."""
    with (
        _screen_backend(),
        patch(f"{_SCREEN_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="validation_error"):
            await _get_manage_window()(app="Phoebus", action=action, **kwargs)

    notify.assert_not_called()


@pytest.mark.parametrize("action,kwargs", _MANAGE_WINDOW_ACTIONS)
async def test_manage_window_backend_failure_no_emit(action, kwargs):
    """A backend that could not act leaves the window alone — emit nothing."""
    with (
        _screen_backend(side_effect=_backend_unavailable()),
        patch(f"{_SCREEN_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="platform_error"):
            await _get_manage_window()(app="Phoebus", action=action, **kwargs)

    notify.assert_not_called()


async def test_manage_window_unknown_app_no_emit():
    """A window the backend cannot find was never touched — emit nothing."""
    with (
        _screen_backend(side_effect=ValueError("no window for app 'Nope'")),
        patch(f"{_SCREEN_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="validation_error"):
            await _get_manage_window()(app="Nope", action="bring_to_front")

    notify.assert_not_called()


# ── terminal down: real helper against a dead port ──────────────────────────


async def test_focus_result_unchanged_when_terminal_down(tmp_path, monkeypatch):
    """Real notify_agent_activity against a dead port: tool result unchanged."""
    monkeypatch.chdir(tmp_path)
    artifact_id = await _save_artifact(title="Down Terminal")

    from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

    port = _free_port()
    # Gallery acknowledges the POST (a gallery failure is a tool error by
    # contract); only the TERMINAL is down here — the notify half under test.
    with (
        patch(
            "osprey.mcp_server.workspace.tools.focus_tools._post_json_with_response",
            return_value=(200, {"status": "ok"}),
        ),
        patch(
            "osprey.mcp_server.http.web_terminal_url",
            return_value=f"http://127.0.0.1:{port}",
        ),
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
