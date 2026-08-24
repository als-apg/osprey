"""Tests for the ``web.config_panel.enabled`` tier gate.

``config.yml`` and ``.claude/`` are what the agent's permission surface is
rendered from, so "may this tier edit them" is a privilege boundary rather than
a UI preference. ``web.config_panel.enabled: false`` therefore withdraws the
Config panel's whole SERVER surface:

* every verb on ``/api/config`` and ``/api/claude-setup`` refuses with 403, and
* ``GET /api/panels`` stops advertising the panel.

The first half is what makes the second half safe to rely on. A client-side
gate alone is undone by typing the route's URL, which is the exact gap this key
exists to close — so the route pins here are the load-bearing ones and the
payload pin is the client's half of the same fact.

Three properties are asserted directly, because each can regress on its own:

* **The gate is FIRST.** It runs ahead of the protected-set logic and ahead of
  anything that touches disk. A disabled panel must refuse a protected-key
  write with the *panel* refusal, never with a protected-key refusal — and
  must leave no backup behind, since a backup is itself a write derived from a
  file this request was never allowed to open.
* **Absent means enabled.** An app with no ``config_panel_enabled`` on
  ``app.state`` (every route unit suite, and any deployment that never mentions
  the key) behaves exactly as it did before this key existed. The default-true
  path is the shipped single-user posture and must stay byte-for-byte the
  behaviour it was.
* **The lifespan resolves it once.** ``create_app`` reads the key into
  ``app.state.config_panel_enabled``; a quoted ``"false"`` is honoured as the
  boolean a human meant, and a value nobody can interpret falls back to the
  enabled default with a warning rather than silently taking an operator's own
  config editor away.
"""

from __future__ import annotations

import logging
from collections import deque
from unittest.mock import patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import coerce_config_flag, create_app
from osprey.interfaces.web_terminal.routes import router
from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_RING_MAX

#: The dotted key under test. Spelled once so a rename shows up as one edit.
CONFIG_PANEL_KEY = "web.config_panel.enabled"

#: A protected key: the write gate itself. Used to prove the panel gate runs
#: BEFORE the protected-set gate — both refuse with 403, so only the detail
#: tells the two apart.
PROTECTED_KEY = "control_system.writes_enabled"

#: An unprotected, agent-relevant key the panel may write when it is enabled.
COSMETIC_KEY = "claude_code.default_model"

#: Minimal config document: enough for the routes to parse, patch and diff.
BASE_CONFIG = {
    "project_name": "config-panel-gate",
    "control_system": {"writes_enabled": False},
    "claude_code": {"default_model": "sonnet"},
}


@pytest.fixture
def project_dir(tmp_path):
    """A project directory carrying a parseable ``config.yml``."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yml").write_text(yaml.safe_dump(BASE_CONFIG, sort_keys=False))
    return project


def _client(project_dir, *, config_panel_enabled):
    """A routes-only app pointed at *project_dir*.

    ``config_panel_enabled`` of ``None`` leaves the attribute OFF ``app.state``
    entirely — the state of every app built without the web terminal's
    lifespan, and the case that proves the routes default to enabled rather
    than to whatever a fixture happened to set.
    """
    app = FastAPI()
    app.include_router(router)
    app.state.config_path = project_dir / "config.yml"
    app.state.project_cwd = str(project_dir)
    app.state.agent_activity_ring = deque(maxlen=ACTIVITY_RING_MAX)
    if config_panel_enabled is not None:
        app.state.config_panel_enabled = config_panel_enabled
    return TestClient(app)


@pytest.fixture
def disabled_client(project_dir):
    with _client(project_dir, config_panel_enabled=False) as client:
        yield client


@pytest.fixture
def enabled_client(project_dir):
    with _client(project_dir, config_panel_enabled=True) as client:
        yield client


@pytest.fixture
def default_client(project_dir):
    """No ``config_panel_enabled`` on state at all — the absent-key posture."""
    with _client(project_dir, config_panel_enabled=None) as client:
        yield client


def _requests(client):
    """Every verb the Config panel reaches, as ``(label, callable)`` pairs.

    Both surfaces in one table: the panel is one privilege, and a verb that
    quietly stayed open would be the whole gap. ``POST /api/claude-setup`` is
    here alongside the PUT because dropping a NEW file into ``.claude/`` is the
    same move as rewriting one already there.
    """
    return [
        ("GET /api/config", lambda: client.get("/api/config")),
        (
            "PUT /api/config",
            lambda: client.put("/api/config", json={"raw": yaml.safe_dump(BASE_CONFIG)}),
        ),
        (
            "PATCH /api/config",
            lambda: client.patch("/api/config", json={"updates": {COSMETIC_KEY: "opus"}}),
        ),
        ("GET /api/claude-setup", lambda: client.get("/api/claude-setup")),
        (
            "PUT /api/claude-setup",
            lambda: client.put("/api/claude-setup", json={"path": "CLAUDE.md", "content": "hi"}),
        ),
        (
            "POST /api/claude-setup",
            lambda: client.post(
                "/api/claude-setup", json={"path": ".claude/skills/x/SKILL.md", "content": "hi"}
            ),
        ),
    ]


class TestDisabledRefusesEveryVerb:
    """``enabled: false`` closes both surfaces completely."""

    def test_every_verb_refuses_with_403(self, disabled_client):
        for label, send in _requests(disabled_client):
            response = send()
            assert response.status_code == 403, f"{label} returned {response.status_code}"

    def test_refusal_names_the_key_that_produced_it(self, disabled_client):
        """An operator who meets the refusal can tell it from a broken deploy."""
        for label, send in _requests(disabled_client):
            detail = send().json()["detail"]
            assert "Config panel is disabled" in detail, f"{label}: {detail!r}"
            assert CONFIG_PANEL_KEY in detail, f"{label}: {detail!r}"


class TestGateRunsFirst:
    """Ahead of the protected set, and ahead of anything that touches disk."""

    def test_protected_patch_reports_the_panel_not_the_protected_key(self, disabled_client):
        response = disabled_client.patch("/api/config", json={"updates": {PROTECTED_KEY: True}})

        assert response.status_code == 403
        detail = response.json()["detail"]
        assert "Config panel is disabled" in detail
        assert PROTECTED_KEY not in detail

    def test_protected_put_reports_the_panel_not_the_protected_key(self, disabled_client):
        document = {**BASE_CONFIG, "control_system": {"writes_enabled": True}}

        response = disabled_client.put("/api/config", json={"raw": yaml.safe_dump(document)})

        assert response.status_code == 403
        assert "Config panel is disabled" in response.json()["detail"]

    def test_nothing_moves_on_disk(self, disabled_client, project_dir):
        """The file is byte-identical and no backup was taken.

        The backup counts: it is a copy of a file this request turned out not
        to be allowed to open, so a gate that let one be written would already
        have leaked the document it refused.
        """
        config_path = project_dir / "config.yml"
        before = config_path.read_bytes()

        for _label, send in _requests(disabled_client):
            send()

        assert config_path.read_bytes() == before
        backups = list(project_dir.rglob("config.yml.bak"))
        assert backups == []

    def test_no_refusal_frame_is_published(self, disabled_client):
        """A disabled panel is not a refused write — nothing reached the file.

        The protected-set refusals publish an activity frame because an attempt
        to rewrite the agent's constraints is something a watching operator
        should see. A request that never got past the front door is a different
        event and must not be reported as that one.
        """
        for _label, send in _requests(disabled_client):
            send()

        ring = disabled_client.app.state.agent_activity_ring
        assert [frame for frame in ring if "refused" in frame["tool"]] == []


class TestEnabledIsUnchanged:
    """Default-true behaviour is exactly what it was before the key existed."""

    def test_explicit_true_serves_config(self, enabled_client):
        response = enabled_client.get("/api/config")

        assert response.status_code == 200
        assert "sections" in response.json()

    def test_absent_state_attribute_serves_config(self, default_client):
        """No ``config_panel_enabled`` on state -> the panel is live."""
        response = default_client.get("/api/config")

        assert response.status_code == 200

    def test_absent_state_attribute_serves_claude_setup(self, default_client):
        response = default_client.get("/api/claude-setup")

        assert response.status_code == 200
        assert "files" in response.json()

    def test_unprotected_patch_still_lands(self, default_client, project_dir):
        response = default_client.patch("/api/config", json={"updates": {COSMETIC_KEY: "opus"}})

        assert response.status_code == 200
        document = yaml.safe_load((project_dir / "config.yml").read_text())
        assert document["claude_code"]["default_model"] == "opus"

    def test_protected_patch_still_reports_the_protected_key(self, default_client):
        """The protected-set gate is untouched by the new one in front of it."""
        response = default_client.patch("/api/config", json={"updates": {PROTECTED_KEY: True}})

        assert response.status_code == 403
        assert PROTECTED_KEY in response.json()["detail"]


class TestPanelsPayload:
    """``GET /api/panels`` never advertises a panel whose routes refuse."""

    def _payload(self, client):
        response = client.get("/api/panels")
        assert response.status_code == 200
        return response.json()

    def test_disabled_payload_says_so(self, disabled_client):
        assert self._payload(disabled_client)["config_panel_enabled"] is False

    def test_enabled_payload_says_so(self, enabled_client):
        assert self._payload(enabled_client)["config_panel_enabled"] is True

    def test_absent_state_attribute_reads_as_enabled(self, default_client):
        assert self._payload(default_client)["config_panel_enabled"] is True

    def test_disabled_payload_never_lists_config(self, project_dir):
        """Not as a built-in, not as a custom panel, not as a focus target.

        The ids are planted deliberately: ``config`` is a drawer tab rather than
        a dock panel today, so nothing puts it in these lists — but a
        config-defined custom panel may claim any id, and the payload of a
        disabled deployment must not be the thing that hands it back.
        """
        with _client(project_dir, config_panel_enabled=False) as client:
            client.app.state.enabled_panels = ["artifacts", "config"]
            client.app.state.visible_panels = ["artifacts", "config"]
            client.app.state.custom_panels = [
                {"id": "config", "label": "CONFIG", "url": "http://localhost:9/"}
            ]
            client.app.state.default_panel = "config"
            client.app.state.active_panel = "config"

            payload = self._payload(client)

        assert "config" not in payload["enabled"]
        assert "config" not in payload["visible"]
        assert [panel["id"] for panel in payload["custom"]] == []
        assert "config" not in payload["labels"]
        assert payload["default"] is None
        assert payload["active"] is None

    def test_enabled_payload_leaves_the_panel_lists_alone(self, project_dir):
        """The filter is the disabled path's only; enabled changes nothing."""
        with _client(project_dir, config_panel_enabled=True) as client:
            client.app.state.enabled_panels = ["artifacts", "config"]
            client.app.state.visible_panels = ["artifacts", "config"]
            client.app.state.default_panel = "config"

            payload = self._payload(client)

        assert "config" in payload["enabled"]
        assert "config" in payload["visible"]
        assert payload["default"] == "config"


class TestCoerceConfigFlag:
    """The pure coercion: what a config file is allowed to say."""

    def test_real_booleans_pass_through(self):
        assert coerce_config_flag(CONFIG_PANEL_KEY, False, True) is False
        assert coerce_config_flag(CONFIG_PANEL_KEY, True, False) is True

    def test_absent_key_takes_the_default(self):
        assert coerce_config_flag(CONFIG_PANEL_KEY, None, True) is True
        assert coerce_config_flag(CONFIG_PANEL_KEY, None, False) is False

    def test_quoted_false_is_honoured(self):
        """The case a plain ``bool()`` gets wrong on a default-ON switch.

        ``bool("false")`` is ``True``: a deployment that wrote the switch OFF
        in quotes would read as having asked for it ON.
        """
        for spelling in ("false", "False", " off ", "no", "0"):
            assert coerce_config_flag(CONFIG_PANEL_KEY, spelling, True) is False

    def test_quoted_true_is_honoured(self):
        for spelling in ("true", "TRUE", "on", "yes", "1"):
            assert coerce_config_flag(CONFIG_PANEL_KEY, spelling, False) is True

    def test_uninterpretable_value_warns_and_takes_the_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = coerce_config_flag(CONFIG_PANEL_KEY, {"enabled": True}, True)

        assert result is True
        assert any(
            CONFIG_PANEL_KEY in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), "expected a WARNING naming the key"


# ---- Startup: the lifespan resolves the key once, onto app.state ----


@pytest.fixture
def workspace_dir(tmp_path):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    return workspace


def _started_app(workspace_dir, configured):
    """Run ``create_app``'s lifespan with *configured* as the key's value.

    ``configured`` of ``None`` omits the key, exercising the absent-key path.
    ``get_config_value`` is patched at its definition site because the lifespan
    imports it inside the function; every other key it reads falls through to
    the default the caller passed, which is what an absent config.yml gives
    them anyway.
    """

    def fake_get_config_value(key, default=None, *args, **kwargs):
        if key == CONFIG_PANEL_KEY and configured is not None:
            return configured
        return default

    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch("osprey.utils.config.get_config_value", fake_get_config_value),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as client:
            yield client


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, True),
        (True, True),
        (False, False),
        ("false", False),
        ("true", True),
        (["not", "a", "flag"], True),
    ],
)
def test_lifespan_resolves_the_flag(workspace_dir, configured, expected):
    generator = _started_app(workspace_dir, configured)
    client = next(generator)
    try:
        assert client.app.state.config_panel_enabled is expected
        assert client.get("/api/panels").json()["config_panel_enabled"] is expected
    finally:
        next(generator, None)


def test_lifespan_disabled_refuses_the_routes(workspace_dir):
    """End to end: the configured key really does close the surface."""
    generator = _started_app(workspace_dir, False)
    client = next(generator)
    try:
        assert client.get("/api/config").status_code == 403
        assert client.get("/api/claude-setup").status_code == 403
    finally:
        next(generator, None)
