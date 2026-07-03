"""Unit tests for the ``phoebus_open_databrowser`` MCP tool.

The bridge HTTP boundary (``_http_post_open``, reused from ``bridge_tools``)
is patched so these run with no Phoebus product, no bridge, and no network —
per Task 0.5, live behavior depends on the app-aware ``/open`` routing
(Task 0.6, built in parallel) and must not be required here.
"""

from pathlib import Path
from unittest.mock import patch

import yaml

from osprey.mcp_server.phoebus.tools import databrowser_tools
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

_MOD = "osprey.mcp_server.phoebus.tools.databrowser_tools"


def _fn(name):
    return get_tool_fn(getattr(databrowser_tools, name))


def _open_databrowser():
    return _fn("phoebus_open_databrowser")


# ── input validation ────────────────────────────────────────────────────────
async def test_empty_channels_raises_validation_error():
    with assert_raises_error(error_type="validation_error") as ctx:
        await _open_databrowser()(channels=[])
    assert "No channels" in ctx["envelope"]["error_message"]


# ── success path / output shape ─────────────────────────────────────────────
async def test_success_returns_handle_and_plt_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = {"id": "d-7", "resource": "ignored", "ready": True}
    with patch(f"{_MOD}._http_post_open", return_value=(200, body)) as mock_open:
        result = await _open_databrowser()(channels=["SR:DCCT"])

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["handle"] == "handle:d-7"
    assert data["id"] == "d-7"
    assert data["ready"] is True
    assert data["channel_count"] == 1
    plt_file = Path(data["plt_file"])
    assert plt_file.exists()
    assert plt_file.suffix == ".plt"

    posted = mock_open.call_args.args[0]
    assert posted["resource"] == data["plt_file"]


async def test_default_title_lists_channels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = {"id": "d-1", "ready": True}
    with patch(f"{_MOD}._http_post_open", return_value=(200, body)):
        result = await _open_databrowser()(channels=["SR:A", "SR:B"])
    data = extract_response_dict(result)
    content = Path(data["plt_file"]).read_text()
    assert "SR:A, SR:B" in content


async def test_explicit_title_used_verbatim(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = {"id": "d-1", "ready": True}
    with patch(f"{_MOD}._http_post_open", return_value=(200, body)):
        result = await _open_databrowser()(channels=["SR:A"], title="Injector Health")
    data = extract_response_dict(result)
    content = Path(data["plt_file"]).read_text()
    assert "<title>Injector Health</title>" in content


# ── facility-neutral archiver default ───────────────────────────────────────
async def test_no_archiver_configured_omits_archive_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PHOEBUS_ARCHIVER_URL", raising=False)
    body = {"id": "d-1", "ready": True}
    with patch(f"{_MOD}._http_post_open", return_value=(200, body)):
        result = await _open_databrowser()(channels=["SR:DCCT"])
    data = extract_response_dict(result)
    content = Path(data["plt_file"]).read_text()
    assert "<archive>" not in content


async def test_configured_archiver_url_binds_into_plt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        yaml.dump({"phoebus": {"archiver_url": "pbraw://example-archiver.test/archappl_retrieve"}})
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    monkeypatch.delenv("PHOEBUS_ARCHIVER_URL", raising=False)

    body = {"id": "d-1", "ready": True}
    with patch(f"{_MOD}._http_post_open", return_value=(200, body)):
        result = await _open_databrowser()(channels=["SR:DCCT"])
    data = extract_response_dict(result)
    content = Path(data["plt_file"]).read_text()
    assert "pbraw://example-archiver.test/archappl_retrieve" in content


async def test_env_archiver_url_wins_over_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"phoebus": {"archiver_url": "pbraw://from-config.test/x"}}))
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    monkeypatch.setenv("PHOEBUS_ARCHIVER_URL", "pbraw://from-env.test/x")

    body = {"id": "d-1", "ready": True}
    with patch(f"{_MOD}._http_post_open", return_value=(200, body)):
        result = await _open_databrowser()(channels=["SR:DCCT"])
    data = extract_response_dict(result)
    content = Path(data["plt_file"]).read_text()
    assert "pbraw://from-env.test/x" in content
    assert "from-config.test" not in content


# ── structured styling ──────────────────────────────────────────────────────
async def test_per_pv_styling_applied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = {"id": "d-1", "ready": True}
    styling = {
        "pvs": {
            "SR:BPM:X": {
                "color": [200, 0, 0],
                "trace_type": "AREA",
                "display_name": "Horizontal Position",
                "axis": 1,
            }
        },
        "show_grid": False,
    }
    with patch(f"{_MOD}._http_post_open", return_value=(200, body)):
        result = await _open_databrowser()(channels=["SR:BPM:X"], styling=styling)
    data = extract_response_dict(result)
    content = Path(data["plt_file"]).read_text()
    assert "Horizontal Position" in content
    assert "<trace_type>AREA</trace_type>" in content
    assert "<grid>false</grid>" in content


async def test_default_color_rotation_when_no_styling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = {"id": "d-1", "ready": True}
    with patch(f"{_MOD}._http_post_open", return_value=(200, body)):
        result = await _open_databrowser()(channels=["SR:A", "SR:B"])
    data = extract_response_dict(result)
    content = Path(data["plt_file"]).read_text()
    # DEFAULT_COLORS[0] = (0, 100, 200), DEFAULT_COLORS[1] = (200, 0, 0)
    assert "<red>0</red>\n        <green>100</green>\n        <blue>200</blue>" in content
    assert "<red>200</red>\n        <green>0</green>\n        <blue>0</blue>" in content


# ── bridge error paths ───────────────────────────────────────────────────────
async def test_bridge_unreachable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch(f"{_MOD}._http_post_open", side_effect=OSError("refused")):
        with assert_raises_error(error_type="phoebus_unreachable"):
            await _open_databrowser()(channels=["SR:DCCT"])


async def test_bridge_error_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch(
        f"{_MOD}._http_post_open",
        return_value=(400, {"error": "unsupported resource type", "status": 400}),
    ):
        with assert_raises_error(error_type="phoebus_open_failed") as ctx:
            await _open_databrowser()(channels=["SR:DCCT"])
    assert "unsupported resource type" in ctx["envelope"]["error_message"]


async def test_bridge_success_with_no_id_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch(f"{_MOD}._http_post_open", return_value=(200, {"ready": True})):
        with assert_raises_error(error_type="phoebus_open_failed") as ctx:
            await _open_databrowser()(channels=["SR:DCCT"])
    assert "no display id" in ctx["envelope"]["error_message"]
