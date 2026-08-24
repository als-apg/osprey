"""Tests for the panels payload route (`routes/panels.py`).

Task 3.3 (project-key-endpoint): ``GET /api/panels`` carries a stable, opaque
``project_key`` — a truncated sha256 of the resolved project directory — that
the client uses as the ``osprey-dock-layout-<project_key>`` localStorage suffix
for per-project dock-layout persistence.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.panels import router

_HEX16 = re.compile(r"\A[0-9a-f]{16}\Z")


def _make_app(project_cwd) -> FastAPI:
    """A minimal app exposing the panels router with a set ``project_cwd``.

    Every other field read by ``get_panels`` is accessed via ``getattr`` with a
    default, so a bare state carrying only ``project_cwd`` exercises the route.
    """
    app = FastAPI()
    app.include_router(router)
    app.state.project_cwd = str(project_cwd)
    return app


def _panels(app: FastAPI) -> dict:
    with TestClient(app) as client:
        resp = client.get("/api/panels")
    assert resp.status_code == 200
    return resp.json()


def test_project_key_present_and_shape(tmp_path):
    """The payload carries a 16-char lowercase-hex ``project_key``."""
    body = _panels(_make_app(tmp_path))
    assert "project_key" in body
    assert isinstance(body["project_key"], str)
    assert _HEX16.match(body["project_key"])


def test_project_key_matches_resolved_sha256(tmp_path):
    """The key is the first 16 hex chars of sha256(resolved project dir)."""
    expected = hashlib.sha256(str(tmp_path.resolve()).encode("utf-8")).hexdigest()[:16]
    body = _panels(_make_app(tmp_path))
    assert body["project_key"] == expected


def test_project_key_stable_across_requests(tmp_path):
    """Two requests to one app return the same key (restart-stable)."""
    app = _make_app(tmp_path)
    assert _panels(app)["project_key"] == _panels(app)["project_key"]


def test_project_key_stable_across_equivalent_paths(tmp_path):
    """Equivalent paths (trailing slash / ``.`` segment) resolve to one key."""
    plain = _panels(_make_app(tmp_path))["project_key"]
    trailing = _panels(_make_app(str(tmp_path) + "/"))["project_key"]
    dotted = _panels(_make_app(tmp_path / "."))["project_key"]
    assert plain == trailing == dotted


def test_project_key_distinct_across_directories(tmp_path):
    """Two apps with different project dirs return different keys."""
    dir_a = tmp_path / "project-a"
    dir_b = tmp_path / "project-b"
    dir_a.mkdir()
    dir_b.mkdir()
    key_a = _panels(_make_app(dir_a))["project_key"]
    key_b = _panels(_make_app(dir_b))["project_key"]
    assert key_a != key_b


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
        "feedback_github_repo",
        "feedback_email",
        "config_panel_enabled",
        "scaffold_write_enabled",
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
    app.state.feedback_github_repo = ""
    app.state.feedback_email = ""

    body = _panels(app)

    assert body["docs_url"] == ""
    assert body["feedback_github_repo"] == ""
    assert body["feedback_email"] == ""


def test_project_key_matches_documented_construction(tmp_path):
    """Key equals sha256(resolved path)[:16] — guards against digest drift."""
    body = _panels(_make_app(tmp_path))
    full = hashlib.sha256(str(Path(tmp_path).resolve()).encode("utf-8")).hexdigest()
    assert body["project_key"] == full[:16]
    assert len(body["project_key"]) == 16


# ---- workspace_has_artifacts (simple-UX chat-only first boot) ----


def _make_workspace_app(project_cwd, workspace_dir):
    app = _make_app(project_cwd)
    app.state.workspace_dir = workspace_dir
    return app


def test_workspace_flag_absent_state_is_false(tmp_path):
    """No workspace_dir on app.state (bare router) → flag is False, not an error."""
    assert _panels(_make_app(tmp_path))["workspace_has_artifacts"] is False


def test_workspace_flag_empty_dir_is_false(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    assert _panels(_make_workspace_app(tmp_path, ws))["workspace_has_artifacts"] is False


def test_workspace_flag_hidden_files_do_not_count(tmp_path):
    """Housekeeping dotfiles (and files under dot-dirs) must not defeat the
    chat-only first boot — the workspace still reads empty."""
    ws = tmp_path / "_agent_data"
    (ws / ".cache").mkdir(parents=True)
    (ws / ".DS_Store").write_text("x")
    (ws / ".cache" / "state.json").write_text("{}")
    assert _panels(_make_workspace_app(tmp_path, ws))["workspace_has_artifacts"] is False


def test_workspace_flag_true_for_toplevel_file(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "orbit_plot.png").write_bytes(b"\x89PNG")
    assert _panels(_make_workspace_app(tmp_path, ws))["workspace_has_artifacts"] is True


def test_workspace_flag_true_for_nested_file(tmp_path):
    """Session subdirectories count — the watcher covers all sessions."""
    ws = tmp_path / "_agent_data"
    (ws / "session-abc").mkdir(parents=True)
    (ws / "session-abc" / "report.html").write_text("<html></html>")
    assert _panels(_make_workspace_app(tmp_path, ws))["workspace_has_artifacts"] is True


def test_workspace_flag_missing_dir_is_false(tmp_path):
    """A configured-but-not-yet-created workspace dir reads empty, no error."""
    ws = tmp_path / "does-not-exist"
    assert _panels(_make_workspace_app(tmp_path, ws))["workspace_has_artifacts"] is False


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
    app = FastAPI()
    app.include_router(router)
    app.state.broadcaster = MagicMock()
    app.state.enabled_panels = {"ariel", "lattice"}
    app.state.custom_panels = [{"id": "grafana", "label": "GRAFANA", "url": "http://10.0.0.5:3000"}]
    if visible is not None:
        app.state.visible_panels = visible
    return app


def _frames(app: FastAPI) -> list[dict]:
    """Every broadcast frame the request produced, in order."""
    return [call.args[0] for call in app.state.broadcaster.broadcast.call_args_list]


def test_focus_on_non_member_adds_rail_membership():
    """An agent open_panel on a hidden panel must not leave a client-only entry."""
    app = _make_focus_app(visible=["ariel"])
    with TestClient(app) as client:
        resp = client.post("/api/panel-focus", json={"panel": "grafana", "source": "agent"})
    assert resp.status_code == 200
    assert app.state.visible_panels == ["ariel", "grafana"]


def test_focus_on_non_member_is_consistent_on_read_back():
    """list_panels must agree with the tool's promise that the panel is visible."""
    app = _make_focus_app(visible=["ariel"])
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "grafana", "source": "agent"})
        body = client.get("/api/panels").json()
    assert "grafana" in body["visible"]
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


def test_focus_visibility_frame_carries_the_source_tag():
    """The agent glow must fire on the rail entry the agent just added."""
    app = _make_focus_app(visible=["ariel"])
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "grafana", "source": "agent"})
    assert _frames(app)[0]["source"] == "agent"


def test_focus_on_member_emits_only_a_focus_frame():
    """An agent switch to a member panel: focus frame only, no visibility traffic."""
    app = _make_focus_app(visible=["ariel", "grafana"])
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "grafana", "source": "agent"})
    assert _frames(app) == [{"type": "panel_focus", "panel": "grafana", "source": "agent"}]


def test_focus_on_member_leaves_rail_order_untouched():
    app = _make_focus_app(visible=["ariel", "grafana"])
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "ariel"})
    assert app.state.visible_panels == ["ariel", "grafana"]


def test_focus_without_explicit_membership_treats_enabled_as_the_rail():
    """Mirrors get_panels: an unset list means the enabled built-ins are visible."""
    app = _make_focus_app(visible=None)
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "ariel"})
    # A member focus adds no membership, and a human focus broadcasts nothing.
    assert _frames(app) == []
    assert not hasattr(app.state, "visible_panels")


def test_focus_without_explicit_membership_drops_a_source_less_non_member():
    """A custom panel outside the enabled fallback is a non-member too."""
    app = _make_focus_app(visible=None)
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "grafana"})
    # A source-less focus can only be a human gesture, and a human can only
    # gesture at a panel already on the rail — so this is a straggler report
    # and must not create membership.
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
    """Source-less focus on a non-member: no membership add, no broadcast."""
    app = _make_focus_app(visible=["ariel"])  # an arrange just pruned grafana
    with TestClient(app) as client:
        resp = client.post("/api/panel-focus", json={"panel": "grafana"})
    assert resp.status_code == 200
    assert app.state.visible_panels == ["ariel"]
    app.state.broadcaster.broadcast.assert_not_called()


def test_stale_human_focus_does_not_steal_the_active_slot():
    """The dropped straggler must leave ``active_panel`` untouched too."""
    app = _make_focus_app(visible=["ariel"])
    app.state.active_panel = "ariel"
    with TestClient(app) as client:
        resp = client.post("/api/panel-focus", json={"panel": "grafana"})
    assert resp.json() == {"status": "ok", "active_panel": "ariel"}
    assert app.state.active_panel == "ariel"


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
    assert body["active_panel"] == "grafana"
    app.state.broadcaster.broadcast.assert_not_called()


def test_agent_focus_broadcasts_a_focus_frame():
    """An agent switch is a command: every client applies the focus frame."""
    app = _make_focus_app(visible=["ariel", "grafana"])
    with TestClient(app) as client:
        client.post("/api/panel-focus", json={"panel": "grafana", "source": "agent"})
    assert _frames(app) == [{"type": "panel_focus", "panel": "grafana", "source": "agent"}]
