"""Tests for MCP server HTTP/IPC helpers (``osprey.mcp_server.http``).

Covers URL construction (config + env precedence), the fire-and-forget
``post_json`` swallowing unreachable targets, the response-returning
``_post_json_with_response`` distinguishing rejection from unreachability,
``fetch_panels`` reading live panel state (including the layout-report
freshness fields), and ``notify_panel_register`` / ``notify_panel_arrange``
mapping HTTP outcomes to their structured dicts.

Also covers the panel-token bearer header the three client paths attach: sent
only when ``OSPREY_PANEL_TOKEN`` holds a non-blank value, re-read on every
request, and a rejected credential (401) degrading quietly rather than raising.
"""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from osprey.mcp_server import http
from osprey.port_layout import default_port

# These tests drive the posters themselves, so they opt out of the conftest
# stub that blocks notify_* POSTs from leaving the process. Nothing here can
# reach a live web terminal: every socket-level call targets a dead port or a
# patched ``urlopen``.
pytestmark = pytest.mark.real_http_posters


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
        assert http.gallery_url() == f"http://127.0.0.1:{default_port('artifact')}"


@pytest.mark.unit
def test_gallery_url_from_config(patch_config):
    with patch_config({"artifact_server": {"host": "0.0.0.0", "port": 9000}}):
        assert http.gallery_url() == "http://0.0.0.0:9000"


@pytest.mark.unit
def test_web_terminal_url_defaults(patch_config, monkeypatch):
    monkeypatch.delenv("OSPREY_WEB_PORT", raising=False)
    with patch_config({}):
        assert http.web_terminal_url() == f"http://127.0.0.1:{default_port('web')}"


@pytest.mark.unit
def test_web_terminal_url_env_overrides_config(patch_config, monkeypatch):
    """OSPREY_WEB_PORT wins over the config port (containerized deployments)."""
    monkeypatch.setenv("OSPREY_WEB_PORT", "12345")
    with patch_config({"web_terminal": {"host": "host.internal", "port": default_port("web")}}):
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
# fetch_panels
# ---------------------------------------------------------------------------


def _panels_response(payload: bytes) -> MagicMock:
    """Build a urlopen context-manager mock returning *payload*."""
    resp = MagicMock()
    resp.read.return_value = payload
    cm = MagicMock()
    cm.__enter__.return_value = resp
    return cm


@pytest.mark.unit
def test_fetch_panels_returns_payload_with_freshness_fields():
    """The layout-report fields reach the caller unchanged."""
    body = (
        b'{"enabled": ["artifacts"], "visible": ["artifacts"], "active": null,'
        b' "open_tiles": ["artifacts", "lattice"], "open_tiles_age_s": 4.5,'
        b' "open_tiles_dock": true}'
    )
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", return_value=_panels_response(body)) as urlopen,
    ):
        data = http.fetch_panels()
    assert urlopen.call_args.args[0].full_url == "http://wt/api/panels"
    assert data is not None
    assert data["open_tiles"] == ["artifacts", "lattice"]
    assert data["open_tiles_age_s"] == 4.5
    assert data["open_tiles_dock"] is True


@pytest.mark.unit
def test_fetch_panels_tolerates_server_without_freshness_fields():
    """An older server omitting the fields is not an error — keys are absent."""
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", return_value=_panels_response(b'{"enabled": []}')),
    ):
        data = http.fetch_panels()
    assert data == {"enabled": []}
    assert data.get("open_tiles_age_s") is None


@pytest.mark.unit
def test_fetch_panels_unreachable_returns_none():
    """An unreachable web terminal yields None rather than raising."""
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", side_effect=OSError("connection refused")),
        patch.object(http, "logger") as mock_logger,
    ):
        assert http.fetch_panels() is None
    assert mock_logger.warning.called


@pytest.mark.unit
def test_fetch_panels_non_object_payload_returns_none():
    """A JSON body that is not an object is treated as unusable."""
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", return_value=_panels_response(b"[1, 2]")),
    ):
        assert http.fetch_panels() is None


# ---------------------------------------------------------------------------
# notify_panel_arrange
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_notify_panel_arrange_posts_tiles_and_focus():
    """Tiles, focus and the agent attribution are sent to the arrange route."""
    captured: dict = {}

    def _fake(url, payload, *, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return 200, {"status": "ok", "tiles": payload["tiles"]}

    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=_fake),
    ):
        http.notify_panel_arrange(tiles=["artifacts", "lattice"], focus="lattice")

    assert captured["url"] == "http://wt/api/panel-arrange"
    assert captured["payload"] == {
        "source": "agent",
        "tiles": ["artifacts", "lattice"],
        "focus": "lattice",
    }


@pytest.mark.unit
def test_notify_panel_arrange_preset_omits_tiles_and_focus():
    """A preset call sends only the preset name (plus attribution)."""
    captured: dict = {}

    def _fake(url, payload, *, timeout):
        captured.update(payload)
        return 200, {}

    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=_fake),
    ):
        http.notify_panel_arrange(preset="injection")

    assert captured == {"source": "agent", "preset": "injection"}


@pytest.mark.unit
def test_notify_panel_arrange_success_returns_data():
    applied = {"status": "ok", "tiles": ["artifacts"], "focus": None, "prune_rail": False}
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", return_value=(200, applied)),
    ):
        out = http.notify_panel_arrange(tiles=["artifacts"])
    assert out == {"ok": True, "status": 200, "data": applied}


@pytest.mark.unit
def test_notify_panel_arrange_rejected_surfaces_detail():
    """A 422 from the route reaches the caller verbatim — never swallowed."""
    detail = "Unknown panel ids: ['bogus']. Valid panel ids: ['artifacts', 'lattice']"
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", return_value=(422, {"detail": detail})),
    ):
        out = http.notify_panel_arrange(tiles=["bogus"])
    assert out == {"ok": False, "status": 422, "detail": detail}


@pytest.mark.unit
def test_notify_panel_arrange_stringifies_structured_detail():
    """FastAPI body-validation 422s carry a list; the contract stays textual."""
    structured = [{"loc": ["body", "tiles"], "msg": "value is not a valid list"}]
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", return_value=(422, {"detail": structured})),
    ):
        out = http.notify_panel_arrange(tiles=["artifacts"])
    assert out["ok"] is False
    assert isinstance(out["detail"], str)
    assert "value is not a valid list" in out["detail"]


@pytest.mark.unit
def test_notify_panel_arrange_unreachable():
    """A down web terminal is reported as such, not raised."""
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=OSError("no route")),
    ):
        out = http.notify_panel_arrange(tiles=["artifacts"])
    assert out == {"ok": False, "status": None, "detail": "Web Terminal is not running."}


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
    assert payload == {"panel": "errors", "visible": True, "source": "agent"}


@pytest.mark.unit
def test_notify_panel_focus_includes_url_when_given():
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "post_json") as post,
    ):
        http.notify_panel_focus("p1", url="http://up")
    _url, payload = post.call_args.args
    assert payload == {"panel": "p1", "url": "http://up", "source": "agent"}


@pytest.mark.unit
def test_notify_panel_focus_omits_url_when_none():
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "post_json") as post,
    ):
        http.notify_panel_focus("p1")
    _url, payload = post.call_args.args
    assert payload == {"panel": "p1", "source": "agent"}
    assert "url" not in payload


# ---------------------------------------------------------------------------
# panel-token bearer header
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_token(monkeypatch):
    """Set (or clear) ``OSPREY_PANEL_TOKEN`` for the duration of one test.

    The module's latch is reset around every test by the root conftest, so a
    token one test saw cannot leak into a sibling asserting "no carrier means
    no header".
    """

    def _apply(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv("OSPREY_PANEL_TOKEN", raising=False)
        else:
            monkeypatch.setenv("OSPREY_PANEL_TOKEN", value)

    return _apply


@pytest.mark.unit
def test_post_json_sends_bearer_when_token_set(panel_token):
    panel_token("panel-secret")
    with patch("urllib.request.urlopen") as urlopen:
        http.post_json("http://localhost:1/api", {"a": 1})
    req = urlopen.call_args.args[0]
    assert req.get_header("Authorization") == "Bearer panel-secret"
    # The header is additive: the JSON content type is still declared.
    assert req.headers["Content-type"] == "application/json"


@pytest.mark.unit
def test_post_json_omits_authorization_when_token_unset(panel_token):
    """No carrier means no header at all — never a bare ``Bearer``."""
    panel_token(None)
    with patch("urllib.request.urlopen") as urlopen:
        http.post_json("http://localhost:1/api", {"a": 1})
    req = urlopen.call_args.args[0]
    assert req.get_header("Authorization") is None


@pytest.mark.unit
@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_post_json_omits_authorization_when_token_blank(panel_token, blank):
    """A blank carrier counts as absent — an uninterpolated compose var is ``""``."""
    panel_token(blank)
    with patch("urllib.request.urlopen") as urlopen:
        http.post_json("http://localhost:1/api", {"a": 1})
    req = urlopen.call_args.args[0]
    assert req.get_header("Authorization") is None


@pytest.mark.unit
def test_bearer_survives_the_process_closing_its_own_carrier(panel_token):
    """An in-process app construction scrubs ``OSPREY_PANEL_TOKEN``; the
    client keeps the value it already saw.

    Saving an artifact can auto-launch the artifact companion inside the MCP
    server process, and building that app closes both credential carriers in
    ``os.environ``.  The bearer must not vanish with them.
    """
    from osprey.interfaces.web_auth import close_env_carriers

    panel_token("panel-secret")
    with patch("urllib.request.urlopen") as urlopen:
        http.post_json("http://localhost:1/api", {})
        close_env_carriers()
        assert "OSPREY_PANEL_TOKEN" not in __import__("os").environ
        http.post_json("http://localhost:1/api", {})
        after = urlopen.call_args.args[0].get_header("Authorization")
    assert after == "Bearer panel-secret"


@pytest.mark.unit
def test_latch_never_invents_a_token(panel_token):
    """A process that never held a carrier sends no bearer, latch or not."""
    panel_token(None)
    assert http._panel_auth_headers() == {}
    panel_token("")
    assert http._panel_auth_headers() == {}


@pytest.mark.unit
def test_post_json_reads_token_at_request_time(panel_token):
    """The token is read per request, never captured once at import."""
    panel_token("first")
    with patch("urllib.request.urlopen") as urlopen:
        http.post_json("http://localhost:1/api", {})
        first = urlopen.call_args.args[0].get_header("Authorization")
        panel_token("second")
        http.post_json("http://localhost:1/api", {})
        second = urlopen.call_args.args[0].get_header("Authorization")
    assert (first, second) == ("Bearer first", "Bearer second")


@pytest.mark.unit
def test_post_json_401_stays_silent_and_non_fatal(panel_token):
    """A rejected credential degrades quietly: warn, return None, never raise."""
    panel_token("stale")
    err = urllib.error.HTTPError(
        url="http://localhost:1/api", code=401, msg="Unauthorized", hdrs=None, fp=None
    )
    with (
        patch("urllib.request.urlopen", side_effect=err),
        patch.object(http, "logger") as mock_logger,
    ):
        assert http.post_json("http://localhost:1/api", {"a": 1}) is None
    logged = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("non-fatal" in msg for msg in logged)


@pytest.mark.unit
def test_post_json_with_response_sends_bearer(panel_token):
    panel_token("panel-secret")
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = b"{}"
    cm = MagicMock()
    cm.__enter__.return_value = resp
    with patch("urllib.request.urlopen", return_value=cm) as urlopen:
        http._post_json_with_response("http://x/y", {"k": "v"})
    req = urlopen.call_args.args[0]
    assert req.get_header("Authorization") == "Bearer panel-secret"
    assert req.headers["Content-type"] == "application/json"


@pytest.mark.unit
def test_post_json_with_response_omits_authorization_when_token_unset(panel_token):
    panel_token(None)
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = b"{}"
    cm = MagicMock()
    cm.__enter__.return_value = resp
    with patch("urllib.request.urlopen", return_value=cm) as urlopen:
        http._post_json_with_response("http://x/y", {})
    assert urlopen.call_args.args[0].get_header("Authorization") is None


@pytest.mark.unit
def test_post_json_with_response_401_returns_status(panel_token):
    """A 401 is a status to report, not an exception — no retry, no raise."""
    panel_token("stale")
    err = urllib.error.HTTPError(url="http://x/y", code=401, msg="Unauthorized", hdrs=None, fp=None)
    err.read = lambda: b'{"detail": "missing or invalid panel token"}'
    with patch("urllib.request.urlopen", side_effect=err) as urlopen:
        status, body = http._post_json_with_response("http://x/y", {})
    assert status == 401
    assert body == {"detail": "missing or invalid panel token"}
    assert urlopen.call_count == 1


@pytest.mark.unit
def test_notify_panel_register_surfaces_401_detail(panel_token):
    """The 401 reaches the tool as a normal rejection dict, not a crash."""
    panel_token("stale")
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(
            http,
            "_post_json_with_response",
            return_value=(401, {"detail": "missing or invalid panel token"}),
        ),
    ):
        out = http.notify_panel_register("p1", "Panel 1", "http://up")
    assert out == {
        "ok": False,
        "status": 401,
        "detail": "missing or invalid panel token",
    }


@pytest.mark.unit
def test_fetch_panels_sends_bearer(panel_token):
    panel_token("panel-secret")
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", return_value=_panels_response(b"{}")) as urlopen,
    ):
        http.fetch_panels()
    req = urlopen.call_args.args[0]
    assert req.full_url == "http://wt/api/panels"
    assert req.get_method() == "GET"
    assert req.get_header("Authorization") == "Bearer panel-secret"


@pytest.mark.unit
def test_fetch_panels_omits_authorization_when_token_unset(panel_token):
    panel_token(None)
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", return_value=_panels_response(b"{}")) as urlopen,
    ):
        http.fetch_panels()
    assert urlopen.call_args.args[0].get_header("Authorization") is None


@pytest.mark.unit
def test_fetch_panels_401_returns_none_silently(panel_token):
    """A rejected read is indistinguishable from an absent terminal: None, no raise."""
    panel_token("stale")
    err = urllib.error.HTTPError(
        url="http://wt/api/panels", code=401, msg="Unauthorized", hdrs=None, fp=None
    )
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", side_effect=err) as urlopen,
        patch.object(http, "logger") as mock_logger,
    ):
        assert http.fetch_panels() is None
    assert mock_logger.warning.called
    assert urlopen.call_count == 1


@pytest.mark.unit
def test_fetch_panels_header_failure_degrades_to_none():
    """Building the credential header is inside the guard, not ahead of it.

    ``fetch_panels`` promises ``None`` whenever panel state cannot be had, so a
    failure assembling the bearer header must take the same quiet path as an
    unreachable terminal rather than raising into ``panel_tools`` callers.
    """
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_panel_auth_headers", side_effect=RuntimeError("no env")),
        patch.object(http, "logger") as mock_logger,
    ):
        assert http.fetch_panels() is None
    assert mock_logger.warning.called
