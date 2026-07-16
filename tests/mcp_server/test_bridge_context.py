"""Tests for the scan MCP server context and HTTP boundary.

Covers: BridgeContext env resolution, config.yml fallback, singleton
init/reset, and the ``_http_get_json``/``_http_post_json`` unreachable-bridge
error envelope (patched ``httpx``, no network).
"""

from unittest.mock import patch

import httpx
import pytest
import yaml
from fastmcp.exceptions import ToolError

from osprey.mcp_server.bluesky.server_context import (
    BridgeContext,
    _http_get_json,
    _http_post_json,
    get_server_context,
    initialize_server_context,
    reset_server_context,
)

pytestmark = pytest.mark.unit


def _write_config(tmp_path, config_dict):
    """Write a config.yml to tmp_path and return the path."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(config_dict))
    return config_file


# ---------------------------------------------------------------------------
# Env resolution
# ---------------------------------------------------------------------------


def test_bridge_url_from_env(tmp_path, monkeypatch):
    """BLUESKY_BRIDGE_URL env var wins outright over config.yml."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"bluesky": {"bridge_url": "http://10.0.0.5:9000"}})
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", "http://127.0.0.1:8123/")

    context = BridgeContext()
    context.initialize()

    assert context.bridge_url == "http://127.0.0.1:8123"


def test_promote_token_from_env(tmp_path, monkeypatch):
    """BLUESKY_PROMOTE_TOKEN env var wins outright over config.yml."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"bluesky": {"promote_token": "config-token"}})
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "env-token")

    context = BridgeContext()
    context.initialize()

    assert context.promote_token == "env-token"


def test_env_absent_no_config_uses_defaults(tmp_path, monkeypatch):
    """With no env vars and no config.yml, sane fail-closed defaults apply."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    context = BridgeContext()
    context.initialize()

    assert context.bridge_url == "http://127.0.0.1:8090"
    assert context.promote_token is None


# ---------------------------------------------------------------------------
# Config fallback
# ---------------------------------------------------------------------------


def test_bridge_url_config_fallback(tmp_path, monkeypatch):
    """bluesky.bridge_url in config.yml is used when the env var is unset."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)
    _write_config(tmp_path, {"bluesky": {"bridge_url": "http://192.168.1.10:8090/"}})

    context = BridgeContext()
    context.initialize()

    assert context.bridge_url == "http://192.168.1.10:8090"


def test_promote_token_config_fallback(tmp_path, monkeypatch):
    """bluesky.promote_token in config.yml is used when the env var is unset."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)
    _write_config(tmp_path, {"bluesky": {"promote_token": "dev-token"}})

    context = BridgeContext()
    context.initialize()

    assert context.promote_token == "dev-token"


# ---------------------------------------------------------------------------
# Singleton init/reset
# ---------------------------------------------------------------------------


def test_initialize_idempotent(tmp_path, monkeypatch):
    """Calling initialize() multiple times is a no-op after the first."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"bluesky": {"bridge_url": "http://127.0.0.1:8090"}})

    context = BridgeContext()
    context.initialize()
    context.initialize()  # second call should be a no-op

    assert context.bridge_url == "http://127.0.0.1:8090"


def test_singleton_access(tmp_path, monkeypatch):
    """get_server_context() returns the same instance after initialize."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"bluesky": {"bridge_url": "http://127.0.0.1:8090"}})

    initialize_server_context()
    ctx1 = get_server_context()
    ctx2 = get_server_context()

    assert ctx1 is ctx2

    reset_server_context()


def test_get_before_initialize_raises():
    """get_server_context() raises before initialize_server_context()."""
    reset_server_context()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_server_context()


def test_reset_clears_singleton(tmp_path, monkeypatch):
    """reset_server_context() clears the singleton so get raises again."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"bluesky": {"bridge_url": "http://127.0.0.1:8090"}})

    initialize_server_context()
    reset_server_context()

    with pytest.raises(RuntimeError, match="not initialized"):
        get_server_context()


# ---------------------------------------------------------------------------
# HTTP boundary — unreachable bridge
# ---------------------------------------------------------------------------


def test_http_get_json_unreachable_raises_error_envelope(tmp_path, monkeypatch):
    """_http_get_json raises a bluesky_bridge_unreachable ToolError on connection failure."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"bluesky": {"bridge_url": "http://127.0.0.1:8090"}})
    initialize_server_context()

    with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
        with pytest.raises(ToolError) as exc_info:
            _http_get_json("/runs")

    assert "bluesky_bridge_unreachable" in str(exc_info.value)

    reset_server_context()


def test_http_post_json_unreachable_raises_error_envelope(tmp_path, monkeypatch):
    """_http_post_json raises a bluesky_bridge_unreachable ToolError on connection failure."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"bluesky": {"bridge_url": "http://127.0.0.1:8090"}})
    initialize_server_context()

    with patch("httpx.post", side_effect=httpx.ConnectError("connection refused")):
        with pytest.raises(ToolError) as exc_info:
            _http_post_json("/runs/abc/promote", {}, headers={"X-Promote-Token": "t"})

    assert "bluesky_bridge_unreachable" in str(exc_info.value)

    reset_server_context()


def test_http_get_json_success_returns_status_and_body(tmp_path, monkeypatch):
    """_http_get_json returns (status_code, parsed_json) on a normal response."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"bluesky": {"bridge_url": "http://127.0.0.1:8090"}})
    initialize_server_context()

    fake_response = httpx.Response(200, json={"runs": []}, request=httpx.Request("GET", "http://x"))
    with patch("httpx.get", return_value=fake_response):
        status, body = _http_get_json("/runs")

    assert status == 200
    assert body == {"runs": []}

    reset_server_context()
