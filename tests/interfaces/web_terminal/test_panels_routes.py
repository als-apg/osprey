"""Tests for the panel state routes (`routes/panels.py`).

``GET /api/panels`` carries the whole panel state, including a stable, opaque
``project_key`` — a truncated sha256 of the resolved project directory — that
the client uses as the ``osprey-dock-layout-<project_key>`` localStorage suffix
for per-project dock-layout persistence. The focus, visibility, close and
register routes mutate that state and broadcast the frames every client
applies; agent-origin requests carry ``source: "agent"`` on their frames,
human ones carry no ``source`` key at all.
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.panels import router

from .conftest import bare_route_app


def _make_app(project_cwd) -> FastAPI:
    """A minimal app exposing the panels router with a set ``project_cwd``.

    Every other field read by ``get_panels`` is accessed via ``getattr`` with a
    default, so a bare state carrying only ``project_cwd`` exercises the route.
    """
    return bare_route_app(router, project_cwd=str(project_cwd))


def _panels(app: FastAPI) -> dict:
    with TestClient(app) as client:
        resp = client.get("/api/panels")
    assert resp.status_code == 200
    return resp.json()


def test_project_key_matches_resolved_sha256(tmp_path):
    """The key is the first 16 hex chars of sha256(resolved project dir)."""
    expected = hashlib.sha256(str(tmp_path.resolve()).encode("utf-8")).hexdigest()[:16]
    body = _panels(_make_app(tmp_path))
    assert body["project_key"] == expected


def test_project_key_stable_across_equivalent_paths(tmp_path):
    """Equivalent paths (trailing slash / ``.`` segment) resolve to one key."""
    plain = _panels(_make_app(tmp_path))["project_key"]
    trailing = _panels(_make_app(str(tmp_path) + "/"))["project_key"]
    dotted = _panels(_make_app(tmp_path / "."))["project_key"]
    assert plain == trailing == dotted


def test_project_key_does_not_disturb_existing_fields(tmp_path):
    """Existing payload fields are unchanged; project_key is purely additive."""
    app = _make_app(tmp_path)
    app.state.enabled_panels = {"ariel", "channels"}
    app.state.custom_panels = [{"id": "grafana", "label": "GRAFANA", "url": "http://x:3000"}]
    app.state.default_panel = "ariel"
    app.state.visible_panels = ["ariel"]
    app.state.active_panel = "ariel"
    app.state.allow_runtime_panels = True
    app.state.panel_presets = [{"name": "L1", "panels": ["ariel"]}]
    app.state.web_ui_mode = "simple"

    body = _panels(app)

    assert set(body["enabled"]) == {"ariel", "channels"}
    assert body["custom"] == [{"id": "grafana", "label": "GRAFANA", "url": "/panel/grafana"}]
    assert body["default"] == "ariel"
    assert body["visible"] == ["ariel"]
    assert body["active"] == "ariel"
    assert body["allow_runtime_panels"] is True
    assert body["presets"] == [{"name": "L1", "panels": ["ariel"]}]
    assert body["ui_mode"] == "simple"
    # The additive key sits alongside the established shape.
    assert set(body) == {
        "enabled",
        "custom",
        "default",
        "visible",
        "active",
        "labels",
        "allow_runtime_panels",
        "presets",
        "ui_mode",
        "rail_position",
        "rail_position_configured",
        "family_rail_defaults",
        "project_key",
        "workspace_has_artifacts",
        "open_tiles",
        "open_tiles_age_s",
        "open_tiles_dock",
        "docs_url",
        "feedback_trackers",
        "feedback_email",
        "feedback_deployment",
        "feedback_escalation_url",
        "config_panel_enabled",
        "scaffold_write_enabled",
        "config_unreadable_path",
        "tour",
    }


def test_blank_utility_targets_reach_the_browser_as_blank(tmp_path):
    """A blanked key must survive the route, not be re-defaulted on the way out.

    This is the second half of the blank posture: ``coerce_config_str`` keeps an
    explicitly blank ``web.docs_url`` / ``web.feedback.*`` blank on ``app.state``,
    and the payload has to carry it through. A ``getattr`` default that fired on
    an empty string here would hand the browser the upstream targets and quietly
    re-enable a channel the deployment retired — the rail would show a docs link
    the control room cannot reach, and the dialog would aim feedback at the
    OSPREY maintainers' tracker.
    """
    app = _make_app(tmp_path)
    app.state.docs_url = ""
    app.state.feedback_trackers = []
    app.state.feedback_email = ""

    body = _panels(app)

    assert body["docs_url"] == ""
    assert body["feedback_trackers"] == []
    assert body["feedback_email"] == ""


def test_a_flagless_app_advertises_no_gated_surface(tmp_path):
    """The Config panel and the scaffold write controls are opt-in, never default-on.

    Their routes refuse when ``config_panel_enabled`` / ``scaffold_write_enabled``
    is absent from ``app.state``, so the payload must say ``False`` too: a
    browser told otherwise paints controls whose every request answers 403.
    """
    body = _panels(_make_app(tmp_path))
    assert body["config_panel_enabled"] is False
    assert body["scaffold_write_enabled"] is False


# ---- workspace_has_artifacts (simple-UX chat-only first boot) ----


def _make_workspace_app(project_cwd, workspace_dir):
    app = _make_app(project_cwd)
    app.state.workspace_dir = workspace_dir
    return app


def test_workspace_flag_absent_state_is_false(tmp_path):
    """No workspace_dir on app.state (bare router) → flag is False, not an error."""
    assert _panels(_make_app(tmp_path))["workspace_has_artifacts"] is False


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param([], False, id="empty"),
        # Housekeeping dotfiles (and files under dot-dirs) must not defeat the
        # chat-only first boot — the workspace still reads empty.
        pytest.param([".DS_Store", ".cache/state.json"], False, id="hidden"),
        pytest.param(["orbit_plot.png"], True, id="toplevel"),
        # Session subdirectories count — the watcher covers all sessions.
        pytest.param(["session-abc/report.html"], True, id="nested"),
        # A configured-but-not-yet-created workspace dir reads empty, no error.
        pytest.param(None, False, id="missing-dir"),
    ],
)
def test_workspace_flag(tmp_path, files, expected):
    """Any non-hidden regular file under the workspace makes it non-empty."""
    ws = tmp_path / "_agent_data"
    if files is not None:
        ws.mkdir()
        for name in files:
            (ws / name).parent.mkdir(parents=True, exist_ok=True)
            (ws / name).write_text("x")
    assert _panels(_make_workspace_app(tmp_path, ws))["workspace_has_artifacts"] is expected


def test_workspace_flag_recomputed_per_request(tmp_path):
    """The first artifact flips the flag on the next request (no caching)."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    app = _make_workspace_app(tmp_path, ws)
    assert _panels(app)["workspace_has_artifacts"] is False
    (ws / "result.csv").write_text("a,b\n")
    assert _panels(app)["workspace_has_artifacts"] is True


# ---- Focus adds rail membership ---- #


def _make_focus_app(visible: list[str] | None = None) -> FastAPI:
    """An app with a stub broadcaster and a known panel inventory.

    ``visible`` is the launcher-rail membership; ``None`` leaves the attribute
    unset so the route's "no explicit list" fallback is exercised.
    """
    app = bare_route_app(
        router,
        enabled_panels={"ariel", "lattice"},
        custom_panels=[{"id": "grafana", "label": "GRAFANA", "url": "http://10.0.0.5:3000"}],
    )
    if visible is not None:
        app.state.visible_panels = visible
    return app


def _frames(app: FastAPI) -> list[dict]:
    """Every broadcast frame the request produced, in order."""
    return [call.args[0] for call in app.state.broadcaster.broadcast.call_args_list]


def test_focus_on_non_member_is_consistent_on_read_back():
    """list_panels must agree with the tool's promise that the panel is visible."""
    app = _make_focus_app(visible=["ariel"])
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "grafana", "source": "agent"})
        body = client.get("/api/panels").json()
    assert body["visible"] == ["ariel", "grafana"]
    assert body["active"] == "grafana"


def test_focus_on_non_member_broadcasts_visibility_before_focus():
    """Ordering is load-bearing: clients add the rail entry, then focus it."""
    app = _make_focus_app(visible=["ariel"])
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "grafana", "source": "agent"})
    frames = _frames(app)
    assert [f["type"] for f in frames] == ["panel_visibility", "panel_focus"]
    assert frames[0] == {
        "type": "panel_visibility",
        "panel": "grafana",
        "visible": True,
        "source": "agent",
    }


def test_focus_on_member_emits_only_a_focus_frame():
    """An agent switch to a member panel: focus frame only, no visibility traffic."""
    app = _make_focus_app(visible=["ariel", "grafana"])
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "grafana", "source": "agent"})
    assert _frames(app) == [{"type": "panel_focus", "panel": "grafana", "source": "agent"}]


def test_focus_without_explicit_membership_treats_enabled_as_the_rail():
    """Mirrors get_panels: an unset list means the enabled built-ins are visible."""
    app = _make_focus_app(visible=None)
    with TestClient(app) as client:
        resp = client.post("/api/panel-focus", json={"panel": "ariel"})
    # The fallback is what makes ``ariel`` a member: without it this human
    # focus would be dropped as a straggler and ``active`` would stay unset.
    assert resp.json()["active_panel"] == "ariel"
    # A member focus adds no membership, and a human focus broadcasts nothing.
    assert _frames(app) == []
    assert not hasattr(app.state, "visible_panels")


def test_focus_without_explicit_membership_drops_a_source_less_non_member():
    """A custom panel outside the enabled fallback is a non-member too."""
    app = _make_focus_app(visible=None)
    with TestClient(app) as client:
        resp = client.post("/api/panel-focus", json={"panel": "grafana"})
    # A source-less focus can only be a human gesture, and a human can only
    # gesture at a panel already on the rail — so this is a straggler report
    # and must neither take the active slot nor create membership.
    assert resp.json()["active_panel"] is None
    assert getattr(app.state, "active_panel", None) is None
    assert _frames(app) == []
    assert not hasattr(app.state, "visible_panels")


# ---- A stale human focus must not resurrect a pruned panel ---- #
#
# Every source-less focus POST is a human gesture REPORT (panel-commands.js),
# and a human can only gesture at a panel already on screen. New membership
# arrives through /api/panels/register or an agent ``open_panel``, never
# through a human focus. So a source-less focus naming a non-member is always
# a straggler that a concurrent arrange overtook: the fire-and-forget report
# was issued before the arrange pruned the panel and arrived after. Applying
# it would resurrect the pruned panel on every client's rail and steal the
# active slot, so the whole write is dropped.


def test_stale_human_focus_does_not_resurrect_a_pruned_panel():
    """Source-less focus on a non-member: no membership add, no active-slot
    steal, no broadcast."""
    app = _make_focus_app(visible=["ariel"])  # an arrange just pruned grafana
    app.state.active_panel = "ariel"
    with TestClient(app) as client:
        resp = client.post("/api/panel-focus", json={"panel": "grafana"})
    assert resp.json() == {"status": "ok", "active_panel": "ariel"}
    assert app.state.active_panel == "ariel"
    assert app.state.visible_panels == ["ariel"]
    app.state.broadcaster.broadcast.assert_not_called()


def test_focus_on_unknown_panel_changes_nothing():
    app = _make_focus_app(visible=["ariel"])
    with TestClient(app) as client:
        resp = client.post("/api/panel-focus", json={"panel": "nope"})
    assert resp.status_code == 422
    assert app.state.visible_panels == ["ariel"]
    app.state.broadcaster.broadcast.assert_not_called()


def test_focus_with_url_keeps_the_url_on_the_focus_frame_only():
    """The visibility frame is membership-only; the url rides the focus frame."""
    app = _make_focus_app(visible=["ariel"])
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "grafana", "url": "/x", "source": "agent"})
    visibility, focus = _frames(app)
    assert "url" not in visibility
    assert focus["url"].endswith("/x")


# ---- Human focus is a report, not a command ---- #
#
# panel-commands.js states the contract: a user-initiated tab switch is
# REPORTED "so the server mirrors the active panel (and does not echo a focus
# event back)". Only agent-attributed focus is a command that must reach every
# client. These tests pin the split.


def test_human_focus_mirrors_active_panel_without_broadcast():
    """A source-less (human) focus updates the mirror and broadcasts nothing."""
    app = _make_focus_app(visible=["ariel", "grafana"])
    with TestClient(app) as client:
        resp = client.post("/api/panel-focus", json={"panel": "grafana"})
        body = client.get("/api/panel-focus").json()
    assert resp.status_code == 200
    assert resp.json()["active_panel"] == "grafana"
    assert body["active_panel"] == "grafana"
    assert app.state.visible_panels == ["ariel", "grafana"]
    app.state.broadcaster.broadcast.assert_not_called()


# ---- Visibility, close and register frames ---- #
#
# Each route stamps ``source`` onto its frame only when the request carried
# it; a human request's frame has no ``source`` key at all (never null).

#: Resolve the register route's SSRF check to a routable LAN address without
#: real DNS.
_LAN_ADDR = [(2, 1, 6, "", ("10.0.0.5", 0))]
_GETADDRINFO_TARGET = "osprey.interfaces.web_terminal.routes.panels.socket.getaddrinfo"
_REGISTER_BODY = {"id": "grafana", "label": "GRAFANA", "url": "http://grafana.lan:3000"}
_REGISTER_FRAME = {
    "type": "panel_register",
    "id": "grafana",
    "label": "GRAFANA",
    "url": "/panel/grafana",
    "healthEndpoint": None,
    "path": "/",
}


def _make_command_app() -> FastAPI:
    """One enabled built-in, runtime registration allowed."""
    return bare_route_app(
        router, enabled_panels={"ariel"}, custom_panels=[], allow_runtime_panels=True
    )


@pytest.mark.parametrize(
    ("path", "body", "expected_frame"),
    [
        pytest.param(
            "/api/panel-visibility",
            {"panel": "ariel", "visible": False},
            {"type": "panel_visibility", "panel": "ariel", "visible": False},
            id="visibility-human",
        ),
        pytest.param(
            "/api/panel-visibility",
            {"panel": "ariel", "visible": True, "source": "agent"},
            {"type": "panel_visibility", "panel": "ariel", "visible": True, "source": "agent"},
            id="visibility-agent",
        ),
        pytest.param(
            "/api/panel-close",
            {"panel": "ariel"},
            {"type": "panel_close", "panel": "ariel"},
            id="close-human",
        ),
        pytest.param(
            "/api/panel-close",
            {"panel": "ariel", "source": "agent"},
            {"type": "panel_close", "panel": "ariel", "source": "agent"},
            id="close-agent",
        ),
        pytest.param("/api/panels/register", _REGISTER_BODY, _REGISTER_FRAME, id="register-human"),
        pytest.param(
            "/api/panels/register",
            {**_REGISTER_BODY, "source": "agent"},
            {**_REGISTER_FRAME, "source": "agent"},
            id="register-agent",
        ),
    ],
)
def test_panel_command_frames(path, body, expected_frame):
    """Exactly one frame per command, carrying the browser-facing URL and the
    request's own attribution — closing never emits a visibility frame."""
    app = _make_command_app()
    with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR), TestClient(app) as client:
        resp = client.post(path, json=body)
    assert resp.status_code == 200
    app.state.broadcaster.broadcast.assert_called_once_with(expected_frame)


def test_panel_close_unknown_panel_is_rejected():
    app = _make_command_app()
    with TestClient(app) as client:
        resp = client.post("/api/panel-close", json={"panel": "nope", "source": "agent"})
    assert resp.status_code == 422
    app.state.broadcaster.broadcast.assert_not_called()
