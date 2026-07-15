"""Unit tests for the launch_run MCP tool — the sole scan write path.

Covers both in-tool safety layers (writes_enabled re-read, client-side token
presence) and the bridge response mapping. The HTTP boundary
(``_http_post_json``) is patched here so these run with no Bluesky bridge
process and no network.
"""

from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import launch
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.launch"


def _fn():
    return get_tool_fn(launch.launch_run)


def _write_config(tmp_path, writes_enabled: bool, promote_token: str | None = None) -> None:
    config: dict = {"control_system": {"writes_enabled": writes_enabled}}
    if promote_token is not None:
        config["scan"] = {"promote_token": promote_token}
    (tmp_path / "config.yml").write_text(yaml.dump(config))


@pytest.fixture(autouse=True)
def _reset_scan_context():
    yield
    reset_server_context()


# ── Layer 1: in-tool writes_enabled re-check (authoritative) ────────────────


async def test_refuses_when_writes_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=False)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "valid-token")
    initialize_server_context()

    with patch(f"{_MOD}._http_post_json") as m:
        with assert_raises_error(error_type="writes_disabled"):
            await _fn()(run_id="abc123")
    m.assert_not_called()


async def test_writes_disabled_refusal_does_not_leak_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=False)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "s3cr3t-token")
    initialize_server_context()

    with patch(f"{_MOD}._http_post_json"):
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _fn()(run_id="abc123")
    assert "s3cr3t-token" not in ctx["envelope"]["error_message"]


async def test_writes_disabled_wins_even_with_no_token(tmp_path, monkeypatch):
    """writes_enabled is checked FIRST — order matters, per the module contract."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=False)
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)
    initialize_server_context()

    with assert_raises_error(error_type="writes_disabled"):
        await _fn()(run_id="abc123")


async def test_writes_enabled_missing_config_fails_closed(tmp_path, monkeypatch):
    """No config.yml at all -> writes_enabled defaults False (fail-closed)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "valid-token")
    initialize_server_context()

    with assert_raises_error(error_type="writes_disabled"):
        await _fn()(run_id="abc123")


# ── Layer 2: client-side token presence (no network call) ──────────────────


async def test_refuses_client_side_when_no_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)
    initialize_server_context()

    with patch(f"{_MOD}._http_post_json") as m:
        with assert_raises_error(error_type="run_promote_unarmed"):
            await _fn()(run_id="abc123")
    m.assert_not_called()


# ── Success path + bridge response mapping ──────────────────────────────────


async def test_launch_run_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "valid-token")
    initialize_server_context()

    body = {"id": "abc123", "status": "running"}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        result = await _fn()(run_id="abc123")

    assert m.call_args.args[0] == "/runs/abc123/promote"
    assert m.call_args.args[1] == {}
    assert m.call_args.kwargs["headers"] == {"X-Promote-Token": "valid-token"}
    data = extract_response_dict(result)
    assert data["status"] == "running"


async def test_launch_run_unknown_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "valid-token")
    initialize_server_context()

    with patch(f"{_MOD}._http_post_json", return_value=(404, {"detail": "unknown run 'abc123'"})):
        with assert_raises_error(error_type="unknown_run") as ctx:
            await _fn()(run_id="abc123")
    assert "unknown run" in ctx["envelope"]["error_message"]


async def test_launch_run_forbidden_token_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "valid-token")
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json",
        return_value=(403, {"detail": "invalid or missing promote token"}),
    ):
        with assert_raises_error(error_type="run_promote_forbidden"):
            await _fn()(run_id="abc123")


async def test_launch_run_conflict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "valid-token")
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json", return_value=(409, {"detail": "run 'abc123' already promoted"})
    ):
        with assert_raises_error(error_type="run_promote_conflict") as ctx:
            await _fn()(run_id="abc123")
    assert "already promoted" in ctx["envelope"]["error_message"]


async def test_launch_run_bridge_unarmed(tmp_path, monkeypatch):
    """The bridge process itself has no BLUESKY_PROMOTE_TOKEN configured (503)."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "valid-token")
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json",
        return_value=(
            503,
            {"detail": "promotion disabled: BLUESKY_PROMOTE_TOKEN is not configured"},
        ),
    ):
        with assert_raises_error(error_type="run_promote_unarmed") as ctx:
            await _fn()(run_id="abc123")
    assert "not configured" in ctx["envelope"]["error_message"]


async def test_launch_bluesky_bridge_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "valid-token")
    initialize_server_context()

    with patch(f"{_MOD}._http_post_json", return_value=(500, {"detail": "promotion failed: boom"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _fn()(run_id="abc123")
    assert "boom" in ctx["envelope"]["error_message"]
