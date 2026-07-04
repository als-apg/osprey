"""Unit tests for the Phoebus MCP bridge tools.

The HTTP boundary (``_http_get_json`` / ``_http_get_bytes`` /
``_http_post_drive`` / ``_http_post_open``) is patched so these run with no
Phoebus product and no network.
"""

import json
import urllib.error
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from osprey.mcp_server.phoebus.tools import bridge_tools
from osprey.mcp_server.phoebus.tools.bridge_tools import _OPEN_READY_TIMEOUT
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

_MOD = "osprey.mcp_server.phoebus.tools.bridge_tools"


def _fn(name):
    return get_tool_fn(getattr(bridge_tools, name))


def _register_demo_panel(tmp_path, monkeypatch, name="osprey_demo"):
    """Register a phoebus panel in a temp config.yml and point OSPREY_CONFIG at it.

    Panels resolve from ``phoebus.panels.<name>`` in config.yml — the real
    deployment path for every facility. There is no built-in default panel, so
    open_panel tests must register the name they open.
    """
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"phoebus": {"panels": {name: f"/path/{name}.bob"}}}))
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))


# ── list_displays ──────────────────────────────────────────────────────────
async def test_list_displays_success():
    displays = [{"name": "demo", "ready": True, "active": True}]
    with patch(f"{_MOD}._http_get_json", return_value=(200, displays)):
        result = await _fn("phoebus_list_displays")()
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["displays"][0]["name"] == "demo"


async def test_list_displays_unreachable():
    with patch(f"{_MOD}._http_get_json", side_effect=urllib.error.URLError("refused")):
        with assert_raises_error(error_type="phoebus_unreachable"):
            await _fn("phoebus_list_displays")()


# ── perceive ───────────────────────────────────────────────────────────────
async def test_perceive_success():
    body = {"display": {"name": "demo"}, "widgets": [{"name": "Setpoint", "type": "textentry"}]}
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _fn("phoebus_perceive")(display="active")
    # display ref is URL-encoded into the path
    assert "/perceive?display=active" in m.call_args.args[0]
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["perception"]["widgets"][0]["name"] == "Setpoint"


async def test_perceive_bridge_error():
    with patch(
        f"{_MOD}._http_get_json", return_value=(400, {"error": "No active display", "status": 400})
    ):
        with assert_raises_error(error_type="phoebus_error") as ctx:
            await _fn("phoebus_perceive")()
    assert "No active display" in ctx["envelope"]["error_message"]


# ── perceive_region ────────────────────────────────────────────────────────
async def test_perceive_region_validation():
    with assert_raises_error(error_type="validation_error"):
        await _fn("phoebus_perceive_region")(x=0, y=0, w=0, h=10)


async def test_perceive_region_success():
    body = {"display": {"name": "demo"}, "widgets": []}
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _fn("phoebus_perceive_region")(x=5, y=6, w=100, h=80, display="active")
    url = m.call_args.args[0]
    assert "/perceive/region?" in url and "w=100" in url and "h=80" in url
    assert extract_response_dict(result)["status"] == "success"


# ── snapshot ───────────────────────────────────────────────────────────────
async def test_snapshot_dpi_validation():
    with assert_raises_error(error_type="validation_error"):
        await _fn("phoebus_snapshot")(widget="Setpoint", dpi=0)
    with assert_raises_error(error_type="validation_error"):
        await _fn("phoebus_snapshot")(widget="Setpoint", dpi=99)


async def test_snapshot_success_writes_png(tmp_path):
    headers = {"X-Bridge-Origin-X": "10.0", "X-Bridge-Origin-Y": "20.0", "X-Bridge-Scale": "1.0"}
    png = b"\x89PNG\r\n\x1a\nfake"
    fake_entry = MagicMock()
    fake_entry.to_tool_response.return_value = {"status": "success", "artifact_id": "abc"}
    fake_store = MagicMock()
    fake_store.save_data.return_value = fake_entry

    with (
        patch(f"{_MOD}._snapshot_dir", return_value=tmp_path),
        patch(f"{_MOD}._http_get_bytes", return_value=(200, headers, png)),
        patch("osprey.stores.artifact_store.get_artifact_store", return_value=fake_store),
    ):
        result = await _fn("phoebus_snapshot")(widget="Setpoint", display="active", dpi=2.0)

    data = extract_response_dict(result)
    assert data["status"] == "success"
    written = list(tmp_path.glob("phoebus_Setpoint_*.png"))
    assert len(written) == 1 and written[0].read_bytes() == png
    # registration headers were threaded into the saved artifact data
    saved = fake_store.save_data.call_args.kwargs["data"]
    assert saved["scale"] == "1.0" and saved["origin_x"] == "10.0"


async def test_snapshot_bridge_error():
    err = json.dumps({"error": "widget not rendered", "status": 500}).encode()
    with patch(f"{_MOD}._http_get_bytes", return_value=(500, {}, err)):
        with assert_raises_error(error_type="phoebus_error") as ctx:
            await _fn("phoebus_snapshot")(widget="Setpoint")
    assert "not rendered" in ctx["envelope"]["error_message"]


# ── drive ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "kwargs",
    [
        {"widget": "0", "verb": "frobnicate"},  # bad verb
        {"widget": "0", "verb": "click", "mode": "x"},  # bad mode
        {"widget": "0", "verb": "type"},  # type without value
    ],
)
async def test_drive_validation(kwargs):
    with assert_raises_error(error_type="validation_error"):
        await _fn("phoebus_drive")(**kwargs)


async def test_drive_success():
    with patch(
        f"{_MOD}._http_post_drive",
        return_value=(200, {"fired": True, "detail": "fired via ButtonBase.fire()"}),
    ) as m:
        result = await _fn("phoebus_drive")(widget="SetButton", verb="click")
    payload = m.call_args.args[0]
    assert payload == {
        "display": "active",
        "widget": "SetButton",
        "verb": "click",
        "value": None,
        "mode": "synthetic",
    }
    data = extract_response_dict(result)
    assert data["status"] == "success" and data["fired"] is True


async def test_drive_type_lowercases_and_forwards_value():
    with patch(
        f"{_MOD}._http_post_drive", return_value=(200, {"fired": True, "detail": "ok"})
    ) as m:
        await _fn("phoebus_drive")(widget="Setpoint", verb="TYPE", value="42", mode="SEMANTIC")
    payload = m.call_args.args[0]
    assert payload["verb"] == "type" and payload["mode"] == "semantic" and payload["value"] == "42"


async def test_drive_rejected():
    with patch(
        f"{_MOD}._http_post_drive", return_value=(400, {"error": "Unknown verb 'x'", "status": 400})
    ):
        with assert_raises_error(error_type="phoebus_rejected") as ctx:
            await _fn("phoebus_drive")(widget="0", verb="click")
    assert "Unknown verb" in ctx["envelope"]["error_message"]


async def test_drive_unreachable():
    with patch(f"{_MOD}._http_post_drive", side_effect=urllib.error.URLError("refused")):
        with assert_raises_error(error_type="phoebus_unreachable"):
            await _fn("phoebus_drive")(widget="0", verb="click")


# ── open_panel ─────────────────────────────────────────────────────────────
async def test_open_panel_success_ready_immediately(tmp_path, monkeypatch):
    """Bridge returns ready=True on the POST /open response — no polling needed."""
    _register_demo_panel(tmp_path, monkeypatch)

    body = {"id": "d-1", "resource": "/path/osprey_demo.bob", "ready": True}
    with patch(f"{_MOD}._http_post_open", return_value=(200, body)) as mock_open:
        result = await _fn("phoebus_open_panel")(name="osprey_demo")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["handle"] == "handle:d-1"
    assert data["id"] == "d-1"
    assert data["ready"] is True
    posted = mock_open.call_args.args[0]
    assert "resource" in posted and posted["resource"].endswith("osprey_demo.bob")


async def test_open_panel_polls_until_ready(tmp_path, monkeypatch):
    """Bridge returns ready=False initially; open_panel polls displays until ready=True."""
    _register_demo_panel(tmp_path, monkeypatch)

    open_body = {"id": "d-2", "resource": "/path/demo.bob", "ready": False}
    displays_body = [{"id": "d-2", "ready": True, "active": True, "name": "demo"}]

    with (
        patch(f"{_MOD}._http_post_open", return_value=(200, open_body)),
        patch(f"{_MOD}._http_get_json", return_value=(200, displays_body)),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await _fn("phoebus_open_panel")(name="osprey_demo")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["ready"] is True
    assert data["handle"] == "handle:d-2"


async def test_open_panel_timeout_returns_not_ready(tmp_path, monkeypatch):
    """When bridge never becomes ready within timeout, ready=False is returned.

    ``time.monotonic`` is patched so the deadline expires before the first loop
    iteration — no real sleep or polling occurs.
    """
    _register_demo_panel(tmp_path, monkeypatch)

    open_body = {"id": "d-3", "resource": "/path/demo.bob", "ready": False}
    still_loading = [{"id": "d-3", "ready": False, "active": True, "name": "demo"}]

    # Patch the module-level _monotonic alias (not the global time.monotonic that
    # anyio uses internally).  First call sets the deadline; second call in the
    # while-condition already exceeds it → loop never executes → no real sleep.
    monotonic_seq = iter([0.0, _OPEN_READY_TIMEOUT + 1.0])

    with (
        patch(f"{_MOD}._http_post_open", return_value=(200, open_body)),
        patch(f"{_MOD}._http_get_json", return_value=(200, still_loading)),
        patch(f"{_MOD}._monotonic", side_effect=monotonic_seq),
    ):
        result = await _fn("phoebus_open_panel")(name="osprey_demo")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["ready"] is False


# ── open_panel auto-focus ──────────────────────────────────────────────────
async def test_open_panel_focuses_own_web_panel_by_default(tmp_path, monkeypatch):
    """A successful open notifies the web terminal to focus THIS instance's
    panel tab, identified by OSPREY_SERVER_NAME (set per-clone by the registry)."""
    _register_demo_panel(tmp_path, monkeypatch)
    monkeypatch.setenv("OSPREY_SERVER_NAME", "phoebus2")

    body = {"id": "d-1", "resource": "/path/osprey_demo.bob", "ready": True}
    with (
        patch(f"{_MOD}._http_post_open", return_value=(200, body)),
        patch(f"{_MOD}.notify_panel_focus") as mock_focus,
    ):
        result = await _fn("phoebus_open_panel")(name="osprey_demo")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["focused"] is True
    mock_focus.assert_called_once_with("phoebus2")


async def test_open_panel_focus_defaults_to_phoebus_panel(tmp_path, monkeypatch):
    """Without OSPREY_SERVER_NAME (template instance / older .mcp.json), the
    focus target falls back to the canonical 'phoebus' panel id."""
    _register_demo_panel(tmp_path, monkeypatch)
    monkeypatch.delenv("OSPREY_SERVER_NAME", raising=False)

    body = {"id": "d-1", "resource": "/path/osprey_demo.bob", "ready": True}
    with (
        patch(f"{_MOD}._http_post_open", return_value=(200, body)),
        patch(f"{_MOD}.notify_panel_focus") as mock_focus,
    ):
        await _fn("phoebus_open_panel")(name="osprey_demo")

    mock_focus.assert_called_once_with("phoebus")


async def test_open_panel_focus_opt_out(tmp_path, monkeypatch):
    """focus=False (batch/background opens) suppresses the panel switch."""
    _register_demo_panel(tmp_path, monkeypatch)

    body = {"id": "d-1", "resource": "/path/osprey_demo.bob", "ready": True}
    with (
        patch(f"{_MOD}._http_post_open", return_value=(200, body)),
        patch(f"{_MOD}.notify_panel_focus") as mock_focus,
    ):
        result = await _fn("phoebus_open_panel")(name="osprey_demo", focus=False)

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["focused"] is False
    mock_focus.assert_not_called()


async def test_open_panel_focus_failure_is_nonfatal(tmp_path, monkeypatch):
    """A failing focus notification (web terminal gone, config unreadable)
    must never fail the open itself — the display IS open at that point."""
    _register_demo_panel(tmp_path, monkeypatch)

    body = {"id": "d-1", "resource": "/path/osprey_demo.bob", "ready": True}
    with (
        patch(f"{_MOD}._http_post_open", return_value=(200, body)),
        patch(f"{_MOD}.notify_panel_focus", side_effect=OSError("terminal gone")),
    ):
        result = await _fn("phoebus_open_panel")(name="osprey_demo")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["focused"] is False


async def test_open_panel_empty_id_raises(tmp_path, monkeypatch):
    """A 200 response with a missing/blank id raises phoebus_open_failed immediately."""
    _register_demo_panel(tmp_path, monkeypatch)

    body_no_id = {"resource": "/path/demo.bob", "ready": False}  # no "id" key
    with patch(f"{_MOD}._http_post_open", return_value=(200, body_no_id)):
        with assert_raises_error(error_type="phoebus_open_failed") as ctx:
            await _fn("phoebus_open_panel")(name="osprey_demo")
    assert "no display id" in ctx["envelope"]["error_message"]


async def test_open_panel_unknown_name():
    """Requesting an unregistered panel name raises ToolError(unknown_panel)."""
    with assert_raises_error(error_type="unknown_panel") as ctx:
        await _fn("phoebus_open_panel")(name="nonexistent_panel")
    assert "nonexistent_panel" in ctx["envelope"]["error_message"]


async def test_open_panel_config_registered(tmp_path, monkeypatch):
    """A panel registered under phoebus.panels in config.yml is opened correctly."""
    panel_file = tmp_path / "custom.bob"
    panel_file.write_text("fake bob content")
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"phoebus": {"panels": {"custom": str(panel_file)}}}))
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))

    body = {"id": "d-4", "resource": str(panel_file), "ready": True}
    with patch(f"{_MOD}._http_post_open", return_value=(200, body)) as mock_open:
        result = await _fn("phoebus_open_panel")(name="custom")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["handle"] == "handle:d-4"
    posted = mock_open.call_args.args[0]
    assert posted["resource"] == str(panel_file)


async def test_open_panel_bridge_400(tmp_path, monkeypatch):
    """A 400 from the bridge surfaces as phoebus_open_failed with the bridge error text."""
    _register_demo_panel(tmp_path, monkeypatch)

    with patch(
        f"{_MOD}._http_post_open",
        return_value=(400, {"error": "file not found", "status": 400}),
    ):
        with assert_raises_error(error_type="phoebus_open_failed") as ctx:
            await _fn("phoebus_open_panel")(name="osprey_demo")
    assert "file not found" in ctx["envelope"]["error_message"]


async def test_open_panel_unreachable(tmp_path, monkeypatch):
    """Bridge unreachable during open raises phoebus_unreachable."""
    _register_demo_panel(tmp_path, monkeypatch)

    with patch(f"{_MOD}._http_post_open", side_effect=urllib.error.URLError("refused")):
        with assert_raises_error(error_type="phoebus_unreachable"):
            await _fn("phoebus_open_panel")(name="osprey_demo")


# ── handle:<id> threading ───────────────────────────────────────────────────
async def test_perceive_with_handle():
    """handle:<id> is URL-encoded and forwarded to the bridge as the display query param."""
    body = {"display": {"name": "d-1"}, "widgets": []}
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _fn("phoebus_perceive")(display="handle:d-1")
    url = m.call_args.args[0]
    import urllib.parse

    assert "handle:d-1" in urllib.parse.unquote(url)
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["display"] == "handle:d-1"


async def test_drive_with_handle_passes_through():
    """handle:<id> is forwarded unchanged in the drive payload."""
    with patch(
        f"{_MOD}._http_post_drive",
        return_value=(200, {"fired": True, "detail": "fired via ButtonBase.fire()"}),
    ) as m:
        result = await _fn("phoebus_drive")(widget="SetButton", verb="click", display="handle:d-1")
    payload = m.call_args.args[0]
    assert payload["display"] == "handle:d-1"
    data = extract_response_dict(result)
    assert data["status"] == "success" and data["fired"] is True


async def test_drive_validation_still_enforced_with_handle():
    """drive verb/mode validation is enforced even when display is a handle string."""
    with assert_raises_error(error_type="validation_error"):
        await _fn("phoebus_drive")(widget="0", verb="frobnicate", display="handle:d-1")


# ── require-handle addressing (PHOEBUS_REQUIRE_HANDLE) ─────────────────────
_REQUIRE_HANDLE_CASES = [
    ("phoebus_perceive", {}),
    ("phoebus_perceive_region", {"x": 0, "y": 0, "w": 10, "h": 10}),
    ("phoebus_snapshot", {"widget": "Setpoint"}),
    ("phoebus_drive", {"widget": "0", "verb": "click"}),
]


@pytest.mark.parametrize("tool_name,kwargs", _REQUIRE_HANDLE_CASES)
async def test_require_handle_env_rejects_implicit_active(tool_name, kwargs, monkeypatch):
    """PHOEBUS_REQUIRE_HANDLE=1 rejects the implicit 'active' fallback on all four tools.

    No bridge patch needed: the rejection happens before any network call.
    """
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "1")
    with assert_raises_error(error_type="phoebus_handle_required") as ctx:
        await _fn(tool_name)(**kwargs)
    assert "phoebus_open_panel" in ctx["envelope"]["error_message"]


async def test_require_handle_config_key_rejects_implicit_active(tmp_path, monkeypatch):
    """phoebus.require_handle: true in config.yml has the same effect as the env var."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"phoebus": {"require_handle": True}}))
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    with assert_raises_error(error_type="phoebus_handle_required"):
        await _fn("phoebus_perceive")()


async def test_require_handle_perceive_with_handle_succeeds(monkeypatch):
    """Flag on + explicit handle:<id> resolves normally (no rejection)."""
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "1")
    body = {"display": {"name": "d-1"}, "widgets": []}
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("phoebus_perceive")(display="handle:d-1")
    assert extract_response_dict(result)["status"] == "success"


async def test_require_handle_perceive_region_with_handle_succeeds(monkeypatch):
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "1")
    body = {"display": {"name": "d-1"}, "widgets": []}
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("phoebus_perceive_region")(x=0, y=0, w=10, h=10, display="handle:d-1")
    assert extract_response_dict(result)["status"] == "success"


async def test_require_handle_snapshot_with_handle_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "1")
    headers = {"X-Bridge-Origin-X": "10.0", "X-Bridge-Origin-Y": "20.0", "X-Bridge-Scale": "1.0"}
    png = b"\x89PNG\r\n\x1a\nfake"
    fake_entry = MagicMock()
    fake_entry.to_tool_response.return_value = {"status": "success", "artifact_id": "abc"}
    fake_store = MagicMock()
    fake_store.save_data.return_value = fake_entry

    with (
        patch(f"{_MOD}._snapshot_dir", return_value=tmp_path),
        patch(f"{_MOD}._http_get_bytes", return_value=(200, headers, png)),
        patch("osprey.stores.artifact_store.get_artifact_store", return_value=fake_store),
    ):
        result = await _fn("phoebus_snapshot")(widget="Setpoint", display="handle:d-1", dpi=1.0)
    assert extract_response_dict(result)["status"] == "success"


async def test_require_handle_drive_with_handle_succeeds(monkeypatch):
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "1")
    with patch(
        f"{_MOD}._http_post_drive",
        return_value=(200, {"fired": True, "detail": "fired via ButtonBase.fire()"}),
    ):
        result = await _fn("phoebus_drive")(widget="SetButton", verb="click", display="handle:d-1")
    assert extract_response_dict(result)["status"] == "success"


async def test_require_handle_off_by_default_active_still_works():
    """With the flag untouched (default off), the implicit 'active' fallback is unchanged."""
    body = {"display": {"name": "demo"}, "widgets": []}
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("phoebus_perceive")()  # display omitted -> defaults to "active"
    assert extract_response_dict(result)["status"] == "success"
