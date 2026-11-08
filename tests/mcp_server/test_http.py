"""Tests for MCP server HTTP/IPC helpers (``osprey.mcp_server.http``).

Covers URL construction (config + env precedence), the fire-and-forget
``post_json`` swallowing unreachable targets, the response-returning
``_post_json_with_response`` distinguishing rejection from unreachability,
``fetch_panels`` reading live panel state (including the layout-report
freshness fields), and ``notify_panel_register`` and the dock-op helper ``panel_op``
mapping HTTP outcomes to their structured dicts.

Also covers the panel-token bearer header the three client paths attach: sent
only when ``OSPREY_PANEL_TOKEN`` holds a non-blank value, re-read on every
request, and a rejected credential (401) degrading quietly rather than raising.
"""

from __future__ import annotations

import os
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
            "osprey_connectors.workspace.load_osprey_config",
            return_value=config,
        )

    return _apply


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


def test_gallery_url_defaults(patch_config):
    with patch_config({}):
        assert http.gallery_url() == f"http://127.0.0.1:{default_port('artifact')}"


def test_gallery_url_from_config(patch_config):
    with patch_config({"artifact_server": {"host": "0.0.0.0", "port": 9000}}):
        assert http.gallery_url() == "http://0.0.0.0:9000"


def test_web_terminal_url_defaults(patch_config, monkeypatch):
    monkeypatch.delenv("OSPREY_WEB_PORT", raising=False)
    with patch_config({}):
        assert http.web_terminal_url() == f"http://127.0.0.1:{default_port('web')}"


def test_web_terminal_url_env_overrides_config(patch_config, monkeypatch):
    """OSPREY_WEB_PORT wins over the config port (containerized deployments)."""
    monkeypatch.setenv("OSPREY_WEB_PORT", "12345")
    with patch_config({"web_terminal": {"host": "host.internal", "port": default_port("web")}}):
        assert http.web_terminal_url() == "http://host.internal:12345"


def test_phoebus_bridge_url_full_env_wins(patch_config, monkeypatch):
    """A full PHOEBUS_BRIDGE_URL short-circuits config entirely (trailing / stripped)."""
    monkeypatch.setenv("PHOEBUS_BRIDGE_URL", "http://phoebus.box:8080/")
    with patch_config({"phoebus": {"host": "ignored", "port": 1}}):
        assert http.phoebus_bridge_url() == "http://phoebus.box:8080"


def test_phoebus_bridge_url_port_env_does_not_override(patch_config, monkeypatch):
    """Only PHOEBUS_BRIDGE_URL outranks the config; a bare port variable does not."""
    monkeypatch.delenv("PHOEBUS_BRIDGE_URL", raising=False)
    monkeypatch.setenv("PHOEBUS_BRIDGE_PORT", "7000")
    with patch_config({"phoebus": {"host": "1.2.3.4", "port": 7979}}):
        assert http.phoebus_bridge_url() == "http://1.2.3.4:7979"


def test_phoebus_bridge_url_default(patch_config, monkeypatch):
    monkeypatch.delenv("PHOEBUS_BRIDGE_URL", raising=False)
    with patch_config({}):
        assert http.phoebus_bridge_url() == "http://127.0.0.1:7979"


def test_phoebus_bridge_url_config_values(patch_config, monkeypatch):
    """With no env overrides, phoebus.host/phoebus.port answer (#829)."""
    monkeypatch.delenv("PHOEBUS_BRIDGE_URL", raising=False)
    with patch_config({"phoebus": {"host": "127.0.0.1", "port": 19921}}):
        assert http.phoebus_bridge_url() == "http://127.0.0.1:19921"


def test_phoebus_bridge_default_pure():
    """The shared config-half helper: configured, defaulted, and a bare `phoebus:` key."""
    assert http.phoebus_bridge_default({}) == "http://127.0.0.1:7979"
    assert http.phoebus_bridge_default({"phoebus": None}) == "http://127.0.0.1:7979"
    assert (
        http.phoebus_bridge_default({"phoebus": {"host": "10.0.0.5", "port": 19921}})
        == "http://10.0.0.5:19921"
    )


# ---------------------------------------------------------------------------
# post_json (fire-and-forget)
# ---------------------------------------------------------------------------


def test_post_json_sends_encoded_payload():
    with patch("urllib.request.urlopen") as urlopen:
        http.post_json("http://localhost:1/api", {"a": 1})
    req = urlopen.call_args.args[0]
    assert req.method == "POST"
    assert req.data == b'{"a": 1}'
    assert req.headers["Content-type"] == "application/json"


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


def test_post_json_with_response_http_error_parses_body():
    """On an HTTPError the status code and parsed error body are returned."""
    err = urllib.error.HTTPError(url="http://x/y", code=403, msg="Forbidden", hdrs=None, fp=None)
    err.read = lambda: b'{"detail": "not allowed"}'
    with patch("urllib.request.urlopen", side_effect=err):
        status, body = http._post_json_with_response("http://x/y", {})
    assert status == 403
    assert body == {"detail": "not allowed"}


def test_post_json_with_response_http_error_unparseable_body():
    """A non-JSON error body degrades to an empty dict, keeping the status code."""
    err = urllib.error.HTTPError(url="http://x/y", code=500, msg="ISE", hdrs=None, fp=None)
    err.read = lambda: b"<html>not json</html>"
    with patch("urllib.request.urlopen", side_effect=err):
        status, body = http._post_json_with_response("http://x/y", {})
    assert status == 500
    assert body == {}


def test_post_json_with_response_unreachable_raises():
    """Connection-level failures propagate (caller distinguishes them)."""
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(urllib.error.URLError):
            http._post_json_with_response("http://x/y", {})


# ---------------------------------------------------------------------------
# notify_panel_register
# ---------------------------------------------------------------------------


def test_notify_panel_register_success():
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", return_value=(200, {"panel": "ok"})),
    ):
        out = http.notify_panel_register("p1", "Panel 1", "http://up")
    assert out == {"ok": True, "status": 200, "data": {"panel": "ok"}}


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


def test_notify_panel_register_unreachable():
    """When the web terminal is down, register reports a friendly unreachable dict."""
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=OSError("no route")),
    ):
        out = http.notify_panel_register("p1", "Panel 1", "http://up")
    assert out == {"ok": False, "status": None, "detail": "Web Terminal is not running."}


def test_notify_panel_register_passes_health_endpoint():
    """The optional health_endpoint is forwarded in the payload."""
    captured: dict = {}
    urls: list[str] = []

    def _fake(url, payload, *, timeout):  # noqa: ARG001 - _post_json_with_response fixes this keyword-only parameter
        urls.append(url)
        captured.update(payload)
        return 200, {}

    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=_fake),
    ):
        http.notify_panel_register("p1", "L", "http://up", path="/sub", health_endpoint="http://h")
    assert urls == ["http://wt/api/panels/register"]
    assert captured["health_endpoint"] == "http://h"
    assert captured["path"] == "/sub"
    assert captured["id"] == "p1"
    assert captured["source"] == "agent"


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


def test_fetch_panels_returns_payload_with_freshness_fields():
    """The layout-report fields reach the caller unchanged."""
    body = (
        b'{"enabled": ["artifacts"], "visible": ["artifacts"], "active": null,'
        b' "layout": {"tile": "t1", "tabs": ["lattice"], "showing": "lattice"},'
        b' "rev": 3, "layout_age_s": 4.5, "layout_dock": true}'
    )
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", return_value=_panels_response(body)) as urlopen,
    ):
        data = http.fetch_panels()
    assert urlopen.call_args.args[0].full_url == "http://wt/api/panels"
    assert data is not None
    assert data["layout"] == {"tile": "t1", "tabs": ["lattice"], "showing": "lattice"}
    assert data["rev"] == 3
    assert data["layout_age_s"] == 4.5
    assert data["layout_dock"] is True


def test_fetch_panels_tolerates_server_without_freshness_fields():
    """An older server omitting the fields is not an error — keys are absent."""
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", return_value=_panels_response(b'{"enabled": []}')),
    ):
        data = http.fetch_panels()
    assert data == {"enabled": []}
    assert data.get("layout_age_s") is None


def test_fetch_panels_unreachable_returns_none():
    """An unreachable web terminal yields None rather than raising."""
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", side_effect=OSError("connection refused")),
        patch.object(http, "logger") as mock_logger,
    ):
        assert http.fetch_panels() is None
    assert mock_logger.warning.called


def test_fetch_panels_non_object_payload_returns_none():
    """A JSON body that is not an object is treated as unusable."""
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", return_value=_panels_response(b"[1, 2]")),
    ):
        assert http.fetch_panels() is None


# ---------------------------------------------------------------------------
# panel_op — the one agent dock-write path
# ---------------------------------------------------------------------------


def _capture_op(status: int = 200, body: dict | None = None):
    """A fake ``_post_json_with_response`` recording each call."""
    calls: list[dict] = []

    def _fake(url, payload, *, timeout):
        calls.append({"url": url, "payload": payload, "timeout": timeout})
        return status, body if body is not None else {"status": "ok", "layout": None, "rev": 4}

    return calls, _fake


@pytest.mark.parametrize(
    ("op", "route"),
    [
        ("drop", "/api/panel-op/drop"),
        ("close", "/api/panel-op/close"),
        ("open", "/api/panel-op/open"),
        ("open_set", "/api/panel-op/open-set"),
        ("open_arrangement", "/api/panel-op/open-arrangement"),
    ],
)
def test_panel_op_posts_each_op_to_its_route(op, route):
    calls, fake = _capture_op()
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=fake),
    ):
        http.panel_op(op, {})
    assert calls[0]["url"] == f"http://wt{route}"


def test_panel_op_mints_a_fresh_request_id_and_waits_eight_seconds():
    calls, fake = _capture_op()
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=fake),
    ):
        http.panel_op("open", {"member": "ariel", "url": "/panel/ariel/#x"})
        http.panel_op("open", {"member": "ariel"})
    first, second = (call["payload"] for call in calls)
    assert {k: v for k, v in first.items() if k != "request_id"} == {
        "member": "ariel",
        "url": "/panel/ariel/#x",
    }
    assert first["request_id"] and first["request_id"] != second["request_id"]
    assert "base_rev" not in first
    assert http.PANEL_OP_TIMEOUT_S == 8
    assert {call["timeout"] for call in calls} == {8}


def test_panel_op_sends_base_rev_when_given():
    calls, fake = _capture_op()
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=fake),
    ):
        http.panel_op("drop", {"source": {"tile": "t1"}, "target": {"outer": "left"}}, base_rev=7)
    assert calls[0]["payload"]["base_rev"] == 7
    assert calls[0]["payload"]["source"] == {"tile": "t1"}


def test_panel_op_success_returns_layout_and_rev():
    layout = {"tile": "t0", "tabs": ["terminal"], "showing": "terminal", "size": 1}
    _, fake = _capture_op(200, {"status": "ok", "layout": layout, "rev": 9})
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=fake),
    ):
        out = http.panel_op("open", {"member": "ariel"})
    assert out == {"ok": True, "status": 200, "layout": layout, "popouts": [], "rev": 9}


def test_panel_op_passes_an_opens_placement_report_through():
    layout = {"tile": "t0", "tabs": ["terminal"], "showing": "terminal", "size": 1}
    report = {"placed": "replaced", "replaced": "okf", "note": "no room for a new tile"}
    _, fake = _capture_op(200, {"status": "ok", "layout": layout, "rev": 9, **report})
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=fake),
    ):
        out = http.panel_op("open", {"member": "ariel"})
    assert out == {"ok": True, "status": 200, "layout": layout, "popouts": [], "rev": 9, **report}


def test_panel_op_relays_a_named_refusal():
    refused = {"status": "refused", "reason": "stale_base", "detail": "read again", "rev": 5}
    _, fake = _capture_op(409, refused)
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=fake),
    ):
        out = http.panel_op("drop", {"source": {"tab": "ariel"}, "target": {"outer": "left"}})
    assert out == {
        "ok": False,
        "status": 409,
        "reason": "stale_base",
        "detail": "read again",
        "rev": 5,
    }


def test_panel_op_stringifies_a_body_validation_error():
    structured = [{"loc": ["body", "base_rev"], "msg": "Field required"}]
    _, fake = _capture_op(422, {"detail": structured})
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=fake),
    ):
        out = http.panel_op("drop", {})
    assert out["ok"] is False
    assert out["reason"] == "invalid_request"
    assert "Field required" in out["detail"]


def test_panel_op_unreachable_is_reported_not_raised():
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "_post_json_with_response", side_effect=OSError("no route")),
    ):
        out = http.panel_op("open", {"member": "ariel"})
    assert out == {
        "ok": False,
        "status": None,
        "reason": "no_web_terminal",
        "detail": "Web Terminal is not running.",
    }


@pytest.mark.parametrize("op", ["place", "close_tab", "close_tile"])
def test_panel_op_refuses_an_unknown_op(op):
    """``close`` is the one close op: a tile shows one panel, so it closes with it."""
    with pytest.raises(ValueError, match="Unknown panel op"):
        http.panel_op(op, {})


# ---------------------------------------------------------------------------
# thin notify_* wrappers
# ---------------------------------------------------------------------------


def test_notify_panel_visibility_posts_payload():
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch.object(http, "post_json") as post,
    ):
        http.notify_panel_visibility("errors", True)
    url, payload = post.call_args.args
    assert url == "http://wt/api/panel-visibility"
    assert payload == {"panel": "errors", "visible": True, "source": "agent"}


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


def test_post_json_sends_bearer_when_token_set(panel_token):
    panel_token("panel-secret")
    with patch("urllib.request.urlopen") as urlopen:
        http.post_json("http://localhost:1/api", {"a": 1})
    req = urlopen.call_args.args[0]
    assert req.get_header("Authorization") == "Bearer panel-secret"
    # The header is additive: the JSON content type is still declared.
    assert req.headers["Content-type"] == "application/json"


@pytest.mark.parametrize(
    "blank",
    [
        # No carrier means no header at all — never a bare ``Bearer``.
        pytest.param(None, id="unset"),
        # A blank carrier counts as absent — an uninterpolated compose var is ``""``.
        pytest.param("", id="empty"),
        pytest.param("   ", id="spaces"),
        pytest.param("\t\n", id="tab-and-newline"),
    ],
)
def test_post_json_omits_authorization_when_token_unset_or_blank(panel_token, blank):
    panel_token(blank)
    with patch("urllib.request.urlopen") as urlopen:
        http.post_json("http://localhost:1/api", {"a": 1})
    req = urlopen.call_args.args[0]
    assert req.get_header("Authorization") is None


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


def test_latch_never_invents_a_token(panel_token):
    """A process that never held a carrier sends no bearer, latch or not."""
    panel_token(None)
    assert http._panel_auth_headers() == {}
    panel_token("")
    assert http._panel_auth_headers() == {}


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


def test_fetch_panels_omits_authorization_when_token_unset(panel_token):
    panel_token(None)
    with (
        patch.object(http, "web_terminal_url", return_value="http://wt"),
        patch("urllib.request.urlopen", return_value=_panels_response(b"{}")) as urlopen,
    ):
        http.fetch_panels()
    assert urlopen.call_args.args[0].get_header("Authorization") is None


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


# ---------------------------------------------------------------------------
# Panel bearer: carrier -> latch -> this process's own credential holder
# ---------------------------------------------------------------------------


def test_panel_auth_headers_fall_back_to_in_process_credentials(monkeypatch):
    """With no carrier in the environment, the bearer is the panel token this
    process already holds — the one a gallery auto-launched in-thread verifies,
    so the process is not refused by its own gallery."""
    from osprey.interfaces.web_auth import (
        BIND_HOST_ENV,
        OPERATOR_SECRET_ENV,
        PANEL_TOKEN_ENV,
        get_web_credentials,
    )

    for name in (PANEL_TOKEN_ENV, OPERATOR_SECRET_ENV, BIND_HOST_ENV):
        monkeypatch.delenv(name, raising=False)
    held = get_web_credentials()  # what building an interface app in-thread does
    assert PANEL_TOKEN_ENV not in os.environ

    assert http._panel_auth_headers() == {"Authorization": f"Bearer {held.panel_token}"}


def test_panel_auth_headers_send_nothing_when_nothing_is_held(monkeypatch):
    """No carrier, no latch, no holder: no bearer at all, never a minted one."""
    from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, peek_web_credentials

    monkeypatch.delenv(PANEL_TOKEN_ENV, raising=False)
    assert peek_web_credentials() is None

    assert http._panel_auth_headers() == {}
    assert peek_web_credentials() is None  # the lookup did not populate anything


def test_panel_auth_headers_prefer_the_carrier_over_the_holder(monkeypatch):
    """A carrier handed in by the launching process wins over the local holder."""
    from osprey.interfaces.web_auth import (
        BIND_HOST_ENV,
        OPERATOR_SECRET_ENV,
        PANEL_TOKEN_ENV,
        get_web_credentials,
    )

    for name in (PANEL_TOKEN_ENV, OPERATOR_SECRET_ENV, BIND_HOST_ENV):
        monkeypatch.delenv(name, raising=False)
    held = get_web_credentials()
    monkeypatch.setenv(PANEL_TOKEN_ENV, "handed-in")

    assert http._panel_auth_headers() == {"Authorization": "Bearer handed-in"}
    assert held.panel_token != "handed-in"


def test_panel_auth_headers_survive_a_refused_credential_population(monkeypatch):
    """Behind the proxy, a refused population leaves the process the token it was handed.

    No panel call precedes the refusal, so the latch holds nothing; the carrier
    is gone and the holder is empty, and the bearer still goes out.
    """
    from osprey.interfaces.web_auth import (
        BIND_HOST_ENV,
        OPERATOR_SECRET_ENV,
        PANEL_TOKEN_ENV,
        get_web_credentials,
        peek_web_credentials,
    )

    monkeypatch.setenv(BIND_HOST_ENV, "127.0.0.1")
    monkeypatch.delenv(OPERATOR_SECRET_ENV, raising=False)
    monkeypatch.setenv(PANEL_TOKEN_ENV, "handed-in")

    with pytest.raises(RuntimeError):
        get_web_credentials()

    assert PANEL_TOKEN_ENV not in os.environ
    assert peek_web_credentials() is None
    assert http._panel_auth_headers() == {"Authorization": "Bearer handed-in"}
