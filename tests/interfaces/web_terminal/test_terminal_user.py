"""Tests for per-user identity surfacing in the web terminal.

``OSPREY_TERMINAL_USER`` and ``OSPREY_TERMINAL_LANDING_URL`` are read
env-over-config (mirroring ``app.state.app_name``) and passed into the
``index.html`` template context by the ``root()`` route.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import app as app_module
from osprey.interfaces.web_terminal.app import create_app


@pytest.fixture
def workspace_dir(tmp_path):
    """Create a temporary workspace directory for the app to watch."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "README.md").write_text("# Test workspace\n")
    return ws


@pytest.fixture
def client(workspace_dir):
    """Create a test client with mocked config active through lifespan."""
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(workspace_dir)},
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


class TestTerminalUser:
    """``OSPREY_TERMINAL_USER`` -> ``app.state.terminal_user`` (env-only, no config key)."""

    def test_set_from_env(self, workspace_dir):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value=cfg,
            ),
            patch.dict("os.environ", {"OSPREY_TERMINAL_USER": "alice"}),
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                assert app.state.terminal_user == "alice"
                assert c.get("/").status_code == 200

    def test_empty_when_unset(self, client):
        # The shared `client` fixture supplies no OSPREY_TERMINAL_USER.
        assert client.app.state.terminal_user == ""


class TestLandingURL:
    """``OSPREY_TERMINAL_LANDING_URL`` -> ``app.state.landing_url`` (env-only, no config key)."""

    def test_set_from_env(self, workspace_dir):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value=cfg,
            ),
            patch.dict(
                "os.environ",
                {"OSPREY_TERMINAL_LANDING_URL": "https://facility.example/portal"},
            ),
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                assert app.state.landing_url == "https://facility.example/portal"
                assert c.get("/").status_code == 200

    def test_empty_when_unset(self, client):
        # The shared `client` fixture supplies no OSPREY_TERMINAL_LANDING_URL.
        assert client.app.state.landing_url == ""


class TestRootContext:
    """root() must forward terminal_user/landing_url into the index.html context."""

    def test_context_includes_terminal_user_and_landing_url(self, workspace_dir):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value=cfg,
            ),
            patch.dict(
                "os.environ",
                {
                    "OSPREY_TERMINAL_USER": "bob",
                    "OSPREY_TERMINAL_LANDING_URL": "https://facility.example/portal",
                },
            ),
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                captured = {}
                original = app_module.templates.TemplateResponse

                def _capture(request, name, context=None, *args, **kwargs):
                    captured.update(context or {})
                    return original(request, name, context, *args, **kwargs)

                with patch.object(app_module.templates, "TemplateResponse", side_effect=_capture):
                    resp = c.get("/")

                assert resp.status_code == 200
                assert captured["terminal_user"] == "bob"
                assert captured["landing_url"] == "https://facility.example/portal"

    def test_context_empty_when_unset(self, client):
        captured = {}
        original = app_module.templates.TemplateResponse

        def _capture(request, name, context=None, *args, **kwargs):
            captured.update(context or {})
            return original(request, name, context, *args, **kwargs)

        with patch.object(app_module.templates, "TemplateResponse", side_effect=_capture):
            resp = client.get("/")

        assert resp.status_code == 200
        assert captured["terminal_user"] == ""
        assert captured["landing_url"] == ""


class TestSessionFooter:
    """The display menu's session footer: who you are, and the way out."""

    def _body(self, workspace_dir, env):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch("osprey.interfaces.web_terminal.app._load_web_config", return_value=cfg),
            patch.dict("os.environ", env),
        ):
            with TestClient(create_app(shell_command="echo")) as c:
                return c.get("/").text

    def test_header_identity_menu_names_the_user_and_holds_the_logout_control(self, workspace_dir):
        body = self._body(
            workspace_dir,
            {
                "OSPREY_TERMINAL_USER": "alice",
                "OSPREY_TERMINAL_LANDING_URL": "https://facility.example/portal",
            },
        )

        assert 'class="header-identity"' in body
        assert 'class="header-identity-who-name">alice<' in body
        # The avatar shows the initial, upper-cased.
        assert 'class="header-identity-avatar" aria-hidden="true">A<' in body
        # The chip is the trigger, and says so to assistive tech.
        assert 'id="header-identity-trigger"' in body
        assert 'aria-controls="header-identity-menu"' in body
        # The logout control keeps its id + data-landing-url contract; app.js's
        # initLogoutButton() and the command palette both find it by id.
        assert 'id="logout-btn"' in body
        assert 'data-landing-url="https://facility.example/portal"' in body
        # Identity lives in the header menu now, not in the display card —
        # but Log out renders in BOTH places: the chip's menu and the display
        # card's action row (its own id; initLogoutButton() binds both).
        assert "display-menu-identity" not in body
        assert 'id="display-menu-logout-btn"' in body

    def test_header_is_identity_free_for_a_single_user_deployment(self, client):
        """No OSPREY_TERMINAL_USER: no identity chip, and no logout control —
        the display card's footer is the Settings button alone."""
        body = client.get("/").text

        assert 'class="header-identity"' not in body
        assert 'id="header-identity-trigger"' not in body
        assert 'id="logout-btn"' not in body
        assert 'id="display-menu-logout-btn"' not in body
        assert 'id="display-menu-settings"' in body

    def test_user_without_a_landing_url_is_named_but_offered_no_logout(self, workspace_dir):
        """A user with nowhere to log out TO still gets identified — the line
        states a fact, and only the action depends on landing_url."""
        body = self._body(workspace_dir, {"OSPREY_TERMINAL_USER": "alice"})

        assert 'class="header-identity-who-name">alice<' in body
        assert 'id="logout-btn"' not in body
        assert 'id="display-menu-logout-btn"' not in body

    def test_deployment_name_moved_out_of_the_action_cluster(self, workspace_dir):
        """app_name renders once, on the left beside the product name, and once
        more as the footer's context line — never as a chip in the right-hand
        action cluster where it read as a second user badge."""
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch("osprey.interfaces.web_terminal.app._load_web_config", return_value=cfg),
            patch(
                "osprey.interfaces.web_terminal.app._load_web_ui_config",
                return_value={"app_name": "Control Assistant"},
            ),
            patch.dict("os.environ", {"OSPREY_TERMINAL_USER": "alice"}),
        ):
            with TestClient(create_app(shell_command="echo")) as c:
                body = c.get("/").text

        assert "header-deployment" in body
        assert 'class="header-identity-who-sub">Control Assistant<' in body
        assert "header-app-name" not in body


class TestAuthRole:
    """``X-Osprey-Auth-Role`` -> the identity menu's who-line.

    nginx forwards the role the auth sidecar resolved on every gated request,
    the page GET included, so ``root()`` reads it off the request rather than
    off ``app.state``: it is a per-login fact, not a render-time one. The
    header is decoded by the same bound the audit ledger applies, and a value
    that fails it renders NOTHING — in a topology with no nginx in front, any
    client can send the header, and the chip is a display, not an
    authorization surface.
    """

    ENV = {"OSPREY_TERMINAL_USER": "alice"}

    def _body(self, workspace_dir, headers, env=None):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch("osprey.interfaces.web_terminal.app._load_web_config", return_value=cfg),
            patch.dict("os.environ", env if env is not None else self.ENV),
        ):
            with TestClient(create_app(shell_command="echo")) as c:
                return c.get("/", headers=headers).text

    def test_the_role_is_shown_in_the_who_line(self, workspace_dir):
        body = self._body(workspace_dir, {"X-Osprey-Auth-Role": "operator"})

        assert 'class="header-identity-who-name">alice<' in body
        assert 'class="header-identity-who-role" title="Role for this session">operator<' in body

    @pytest.mark.parametrize(
        "headers",
        [
            pytest.param({}, id="absent"),
            pytest.param({"X-Osprey-Auth-Role": ""}, id="empty"),
        ],
    )
    def test_no_role_header_renders_no_role(self, workspace_dir, headers):
        """The sidecar omits the header when the login carried no role, a
        deployment with no sidecar never sends one, and an ungated nginx
        location clears it to ``""``: absence is a state, not a blank to fill
        with a default."""
        body = self._body(workspace_dir, headers)

        assert 'class="header-identity-who-name">alice<' in body
        assert "header-identity-who-role" not in body

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("op erator", id="interior-space"),
            pytest.param("a" * 129, id="over-long"),
            pytest.param("oper\x7fator", id="control-char"),
        ],
    )
    def test_a_value_the_sidecar_could_not_have_sent_renders_nothing(self, workspace_dir, value):
        """Same bound as the audit ledger's ``_recordable``: printable ASCII,
        no interior space, at most 128 characters. The ledger records such a value as
        its ``<unsafe>`` sentinel because a record must say something was
        there; a chip has no such duty and shows nothing rather than a marker
        an operator would have to interpret."""
        body = self._body(workspace_dir, {"X-Osprey-Auth-Role": value})

        assert "header-identity-who-role" not in body
        # Autoescape would spell the sentinel as an entity; pin the escaped form
        # so this line still fails if the suppression in root() is removed.
        assert "&lt;unsafe&gt;" not in body

    def test_a_markup_shaped_role_is_escaped(self, workspace_dir):
        """``<script>…</script>`` carries no space and is printable ASCII, so
        it clears the bound and reaches the template. Autoescape is what makes
        the chip safe; nothing else here would notice if it were turned off."""
        body = self._body(workspace_dir, {"X-Osprey-Auth-Role": "<script>alert(1)</script>"})

        assert "<script>alert(1)</script>" not in body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body

    def test_a_role_without_a_user_renders_no_chip(self, workspace_dir):
        """Single-user terminals render no identity chip at all; a stray role
        header must not conjure one."""
        body = self._body(workspace_dir, {"X-Osprey-Auth-Role": "operator"}, env={})

        assert 'class="header-identity"' not in body
        assert "header-identity-who-role" not in body

    def test_the_context_carries_the_role(self, workspace_dir):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch("osprey.interfaces.web_terminal.app._load_web_config", return_value=cfg),
            patch.dict("os.environ", self.ENV),
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                captured = {}
                original = app_module.templates.TemplateResponse

                def _capture(request, name, context=None, *args, **kwargs):
                    captured.update(context or {})
                    return original(request, name, context, *args, **kwargs)

                with patch.object(app_module.templates, "TemplateResponse", side_effect=_capture):
                    assert c.get("/", headers={"X-Osprey-Auth-Role": "operator"}).status_code == 200
                    assert captured["auth_role"] == "operator"
                    captured.clear()
                    assert c.get("/").status_code == 200
                    assert captured["auth_role"] == ""
