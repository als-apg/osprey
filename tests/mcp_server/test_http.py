"""Tests for MCP server HTTP/IPC helpers (``osprey.mcp_server.http``).

Covers URL construction (config + env precedence), the fire-and-forget
``post_json`` swallowing unreachable targets, the response-returning
``_post_json_with_response`` distinguishing rejection from unreachability, and
``notify_panel_register`` mapping HTTP outcomes to its structured dict.
"""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from osprey.mcp_server import http


@pytest.fixture
def patch_config():
    """Patch load_osprey_config as referenced inside osprey.mcp_server.http."""

    def _apply(config: dict):
        return patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value=config,
        )

    return _apply


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gallery_url_defaults(patch_config):
    with patch_config({}):
        assert http.gallery_url() == "http://127.0.0.1:8086"


@pytest.mark.unit
def test_gallery_url_from_config(patch_config):
    with patch_config({"artifact_server": {"host": "0.0.0.0", "port": 9000}}):
        assert http.gallery_url() == "http://0.0.0.0:9000"


@pytest.mark.unit
def test_web_terminal_url_defaults(patch_config, monkeypatch):
    monkeypatch.delenv("OSPREY_WEB_PORT", raising=False)
    with patch_config({}):
        assert http.web_terminal_url() == "http://127.0.0.1:8087"


@pytest.mark.unit
def test_web_terminal_url_env_overrides_config(patch_config, monkeypatch):
    """OSPREY_WEB_PORT wins over the config port (containerized deployments)."""
    monkeypatch.setenv("OSPREY_WEB_PORT", "12345")
    with patch_config({"web_terminal": {"host": "host.internal", "port": 8087}}):
        assert http.web_terminal_url() == "http://host.internal:12345"


@pytest.mark.unit
def test_phoebus_bridge_url_full_env_wins(patch_config, monkeypatch):
    """A full PHOEBUS_BRIDGE_URL short-circuits config entirely (trailing / stripped)."""
    monkeypatch.setenv("PHOEBUS_BRIDGE_URL", "http://phoebus.box:8080/")
    with patch_config({"phoebus": {"host": "ignored", "port": 1}}):
        assert http.phoebus_bridge_url() == "http://phoebus.box:8080"


@pytest.mark.unit
def test_phoebus_bridge_url_port_env_overrides(patch_config, monkeypatch):
    monkeypatch.delenv("PHOEBUS_BRIDGE_URL", raising=False)
    monkeypatch.setenv("PHOEBUS_BRIDGE_PORT", "7000")
    with patch_config({"phoebus": {"host": "1.2.3.4", "port": 7979}}):
        assert http.phoebus_bridge_url() == "http://1.2.3.4:7000"


@pytest.mark.unit
def test_phoebus_bridge_url_default(patch_config, monkeypatch):
    monkeypatch.delenv("PHOEBUS_BRIDGE_URL", raising=False)
    monkeypatch.delenv("PHOEBUS_BRIDGE_PORT", raising=False)
    with patch_config({}):
        assert http.phoebus_bridge_url() == "http://127.0.0.1:7979"


# ---------------------------------------------------------------------------
# post_json (fire-and-forget)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_post_json_sends_encoded_payload():
    with patch("urllib.request.urlopen") as urlopen:
        http.post_json("http://localhost:1/api", {"a": 1})
    req = urlopen.call_args.args[0]
    assert req.method == "POST"
    assert req.data == b'{"a": 1}'
    assert req.headers["Content-type"] == "application/json"


@pytest.mark.unit
def test_post_json_swallows_unreachable():
    """An unreachable target is non-fatal: it logs a warning and returns None."""
    # Assert on the module logger, not caplog: full-suite logging reconfiguration
    # can cut propagation to the root logger, making caplog order-dependent.
    with (
        patch("urllib.request.urlopen", side_effect=OSError("connection refused")),
        patch.object(http, "logger") as mock_logger,
    ):
        assert http.post_json("http://localhost:1/api", {"a": 1}) is None
    logged = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("non-fatal" in msg for msg in logged)


# ---------------------------------------------------------------------------
# _post_json_with_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_post_json_with_response_success():
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = b'{"ok": true}'
    cm = MagicMock()
    cm.__enter__.return_value = resp
    with patch("urllib.request.urlopen", return_value=cm):
        status, body = http._post_json_with_response("http://x/y", {"k": "v"})
    assert status == 200
    assert body == {"ok": True}


@pytest.mark.unit
def test_post_json_with_response_http_error_parses_body():
    """On an HTTPError the status code and parsed error body are returned."""
    err = urllib.error.HTTPError(url="http://x/y", code=403, msg="Forbidden", hdrs=None, fp=None)
    err.read = lambda: b'{"detail": "not allowed"}'
    with patch("urllib.request.urlopen", side_effect=err):
        status, body = http._post_json_with_response("http://x/y", {})
    assert status == 403
    assert body == {"detail": "not allowed"}


@pytest.mark.unit
def test_post_json_with_response_http_error_unparseable_body():
    """A non-JSON error body degrades to an empty dict, keeping the status code."""
    err = urllib.error.HTTPError(url="http://x/y", code=500, msg="ISE", hdrs=None, fp=None)
    err.read = lambda: b"<html>not json</html>"
    with patch("urllib.request.urlopen", side_effect=err):
        status, body = http._post_json_with_response("http://x/y", {})
    assert status == 500
    assert body == {}


@pytest.mark.unit
def test_post_json_with_response_unreachable_raises():
    """Connection-level failures propagate (caller distinguishes them)."""
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(urllib.error.URLError):
            http._post_json_with_response("http://x/y", {})


# ---------------------------------------------------------------------------
# notify_panel_register
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_notify_panel_register_success():
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", return_value=(200, {"panel": "ok"})),
    ):
        out = http.notify_panel_register("p1", "Panel 1", "http://up")
    assert out == {"ok": True, "status": 200, "data": {"panel": "ok"}}


@pytest.mark.unit
def test_notify_panel_register_rejected_surfaces_detail():
    """A non-200 (server rejection) returns ok=False with the server detail."""
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(
            http,
            "_post_json_with_response",
            return_value=(403, {"detail": "not on allowlist"}),
        ),
    ):
        out = http.notify_panel_register("p1", "Panel 1", "http://up")
    assert out == {"ok": False, "status": 403, "detail": "not on allowlist"}


@pytest.mark.unit
def test_notify_panel_register_unreachable():
    """When the web terminal is down, register reports a friendly unreachable dict."""
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=OSError("no route")),
    ):
        out = http.notify_panel_register("p1", "Panel 1", "http://up")
    assert out == {"ok": False, "status": None, "detail": "Web Terminal is not running."}


@pytest.mark.unit
def test_notify_panel_register_passes_health_endpoint():
    """The optional health_endpoint is forwarded in the payload."""
    captured: dict = {}

    def _fake(url, payload, *, timeout):
        captured.update(payload)
        return 200, {}

    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=_fake),
    ):
        http.notify_panel_register("p1", "L", "http://up", path="/sub", health_endpoint="http://h")
    assert captured["health_endpoint"] == "http://h"
    assert captured["path"] == "/sub"
    assert captured["id"] == "p1"


# ---------------------------------------------------------------------------
# thin notify_* wrappers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_notify_panel_visibility_posts_payload():
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "post_json") as post,
    ):
        http.notify_panel_visibility("errors", True)
    url, payload = post.call_args.args
    assert url == "http://wt/api/panel-visibility"
    assert payload == {"panel": "errors", "visible": True}


@pytest.mark.unit
def test_notify_panel_focus_includes_url_when_given():
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "post_json") as post,
    ):
        http.notify_panel_focus("p1", url="http://up")
    _url, payload = post.call_args.args
    assert payload == {"panel": "p1", "url": "http://up"}


@pytest.mark.unit
def test_notify_panel_focus_omits_url_when_none():
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "post_json") as post,
    ):
        http.notify_panel_focus("p1")
    _url, payload = post.call_args.args
    assert payload == {"panel": "p1"}
    assert "url" not in payload
