"""Tests for per-user identity surfacing in the web terminal.

``OSPREY_TERMINAL_USER`` and ``OSPREY_TERMINAL_LANDING_URL`` are read at
startup and rendered by ``root()`` into the title row of the sidebar's ☰ More
menu and the sidebar footer's Sign out row; the forwarded role headers are
read per request.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import app as app_module
from osprey.interfaces.web_terminal.app import create_app
from tests.interfaces.web_terminal._started_app import terminal_client


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
    with terminal_client(workspace_dir) as c:
        yield c


def _span(body: str, cls: str) -> tuple[str, str] | None:
    """The attributes and text of the first ``<span class="cls" …>text<``.

    The class is matched whole (closing quote included), so ``more-menu-role``
    never matches ``more-menu-role-source``.
    """
    match = re.search(rf'<span class="{re.escape(cls)}"([^>]*)>([^<]*)<', body)
    return (match.group(1), match.group(2)) if match else None


class TestMoreMenuIdentity:
    """The More menu's title row says who you are; the footer holds the way out."""

    def _body(self, workspace_dir, env):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch("osprey.interfaces.web_terminal.app._load_web_config", return_value=cfg),
            patch.dict("os.environ", env),
        ):
            with TestClient(create_app(shell_command=["echo"])) as c:
                return c.get("/").text

    def test_more_menu_names_the_user_and_the_footer_holds_the_logout_control(self, workspace_dir):
        body = self._body(
            workspace_dir,
            {
                "OSPREY_TERMINAL_USER": "alice",
                "OSPREY_TERMINAL_LANDING_URL": "https://facility.example/portal",
            },
        )

        assert 'class="more-menu-who"' in body
        user = _span(body, "more-menu-user")
        assert user is not None and user[1] == "alice"
        # The avatar shows the initial, upper-cased, and is hidden from
        # assistive tech (the name beside it already says who this is).
        avatar = _span(body, "more-menu-avatar")
        assert avatar is not None and avatar[1] == "A"
        assert 'aria-hidden="true"' in avatar[0]
        # The logout control keeps its id + data-landing-url contract; app.js's
        # initSignOut() finds it by id. It renders once, as a row of the
        # sidebar footer above ☰ More, and the More menu no longer holds it.
        assert body.count('id="logout-btn"') == 1
        foot = body.index('class="sidebar-foot"')
        assert foot < body.index('id="logout-btn"') < body.index('id="more-btn"')
        assert body.index('id="logout-btn"') < body.index('id="more-menu"')
        assert 'data-landing-url="https://facility.example/portal"' in body
        # With sign-out available, the OSPREY mark opens the same dialog.
        brand = re.search(r'<a class="sidebar-brand"[^>]*>', body)
        assert brand is not None and 'aria-haspopup="dialog"' in brand.group(0)
        # The header identity chip is gone with the header.
        assert "header-identity" not in body

    def test_more_menu_is_identity_free_for_a_single_user_deployment(self, client):
        """No OSPREY_TERMINAL_USER: no title row and no logout control — the
        More menu still offers Settings."""
        body = client.get("/").text

        assert 'class="more-menu-who"' not in body
        assert 'class="more-menu-user"' not in body
        assert 'id="logout-btn"' not in body
        assert 'id="more-settings"' in body
        # The OSPREY mark stays a plain home link.
        brand = re.search(r'<a class="sidebar-brand"[^>]*>', body)
        assert brand is not None and "aria-haspopup" not in brand.group(0)

    def test_deployment_name_is_the_title_row_sub_line(self, workspace_dir):
        """app_name renders as the title row's sub-line under the user name —
        never as a chip of its own where it would read as a second user badge."""
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch("osprey.interfaces.web_terminal.app._load_web_config", return_value=cfg),
            patch(
                "osprey.interfaces.web_terminal.app._load_web_ui_config",
                return_value={"app_name": "Control Assistant"},
            ),
            patch.dict("os.environ", {"OSPREY_TERMINAL_USER": "alice"}),
        ):
            with TestClient(create_app(shell_command=["echo"])) as c:
                body = c.get("/").text

        sub = _span(body, "more-menu-sub")
        assert sub is not None and sub[1] == "Control Assistant"
        assert "header-app-name" not in body


class TestAuthRole:
    """``X-Osprey-Auth-Role`` and ``X-Osprey-Auth-Role-Source`` -> the title row.

    nginx forwards the role the auth sidecar resolved on every gated request,
    and beside it where that role came from, the page GET included — so
    ``root()`` reads both off the request rather than off ``app.state``: they
    are per-login facts, not render-time ones. Both headers are decoded by the
    same bound the audit ledger applies, and a value that fails it renders
    NOTHING — in a topology with no nginx in front, any client can send these
    headers, and the title row is a display, not an authorization surface.

    The source is shown only through ``root()``'s closed vocabulary and only
    beside a role: a qualifier with nothing to qualify would be a separator
    hanging off the end of the line.
    """

    ENV = {"OSPREY_TERMINAL_USER": "alice"}

    #: The source span's full opening tag, asserted whole rather than by its
    #: class alone — ``more-menu-role`` is a substring of
    #: ``more-menu-role-source``, so a bare class check cannot tell the role
    #: from its qualifier.
    SOURCE_SPAN = 'class="more-menu-role-source" title="Where this session\'s role came from">'
    ROLE_PILL = 'class="more-menu-role" title="Role for this session">operator<'

    def _body(self, workspace_dir, headers, env=None):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch("osprey.interfaces.web_terminal.app._load_web_config", return_value=cfg),
            patch.dict("os.environ", env if env is not None else self.ENV),
        ):
            with TestClient(create_app(shell_command=["echo"])) as c:
                return c.get("/", headers=headers).text

    def test_the_role_is_shown_in_the_title_row(self, workspace_dir):
        body = self._body(workspace_dir, {"X-Osprey-Auth-Role": "operator"})

        user = _span(body, "more-menu-user")
        assert user is not None and user[1] == "alice"
        assert self.ROLE_PILL in body
        # A role with no source beside it — the shape every pre-upgrade session
        # decodes to — shows the role alone, with no qualifier span at all.
        assert 'class="more-menu-role-source"' not in body

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

        user = _span(body, "more-menu-user")
        assert user is not None and user[1] == "alice"
        assert "more-menu-role" not in body

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
        there; the title row has no such duty and shows nothing rather than a
        marker an operator would have to interpret."""
        body = self._body(workspace_dir, {"X-Osprey-Auth-Role": value})

        assert "more-menu-role" not in body
        # Autoescape would spell the sentinel as an entity; pin the escaped form
        # so this line still fails if the suppression in root() is removed.
        assert "&lt;unsafe&gt;" not in body

    def test_a_markup_shaped_role_is_escaped(self, workspace_dir):
        """``<script>…</script>`` carries no space and is printable ASCII, so
        it clears the bound and reaches the template. Autoescape is what makes
        the title row safe; nothing else here would notice if it were turned off."""
        body = self._body(workspace_dir, {"X-Osprey-Auth-Role": "<script>alert(1)</script>"})

        assert "<script>alert(1)</script>" not in body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body

    @pytest.mark.parametrize(
        "headers",
        [
            pytest.param({"X-Osprey-Auth-Role": "operator"}, id="role"),
            pytest.param(
                {"X-Osprey-Auth-Role": "operator", "X-Osprey-Auth-Role-Source": "roster"},
                id="role-and-source",
            ),
        ],
    )
    def test_identity_headers_without_a_user_render_no_title_row(self, workspace_dir, headers):
        """Single-user terminals render no identity title row at all; stray
        identity headers must not conjure one."""
        body = self._body(workspace_dir, headers, env={})

        assert 'class="more-menu-who"' not in body
        # Covers the source span too: its class extends the role's.
        assert "more-menu-role" not in body

    @pytest.mark.parametrize(
        ("source", "label"),
        [
            pytest.param("roster", "roster", id="roster"),
            pytest.param("claim", "ID token", id="claim"),
        ],
    )
    def test_the_source_is_shown_after_the_role(self, workspace_dir, source, label):
        """The sidecar's word is translated for the reader: a login bound by
        the roster says so in the roster's own term, while ``claim`` becomes
        "ID token", which is what an operator sees named everywhere else."""
        body = self._body(
            workspace_dir,
            {"X-Osprey-Auth-Role": "operator", "X-Osprey-Auth-Role-Source": source},
        )

        assert self.ROLE_PILL in body
        assert self.SOURCE_SPAN + label + "<" in body

    def test_a_source_without_a_role_renders_nothing(self, workspace_dir):
        """The qualifier is gated on the role, not only on itself. A payload
        shape the sidecar never mints could carry a source alone; the title
        row would then show a separator qualifying nothing."""
        body = self._body(workspace_dir, {"X-Osprey-Auth-Role-Source": "roster"})

        user = _span(body, "more-menu-user")
        assert user is not None and user[1] == "alice"
        assert 'class="more-menu-role-source"' not in body

    def test_a_source_outside_the_vocabulary_renders_nothing(self, workspace_dir):
        """``ldap`` clears the header bound and is still not a source this
        build can name. The role it qualifies is unaffected: the title row
        drops the word it does not know, not the fact it does."""
        body = self._body(
            workspace_dir,
            {"X-Osprey-Auth-Role": "operator", "X-Osprey-Auth-Role-Source": "ldap"},
        )

        assert self.ROLE_PILL in body
        assert 'class="more-menu-role-source"' not in body
        assert "ldap" not in body

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("ro ster", id="interior-space"),
            pytest.param("a" * 129, id="over-long"),
            pytest.param("ros\x7fter", id="control-char"),
        ],
    )
    def test_a_source_the_sidecar_could_not_have_sent_renders_nothing(self, workspace_dir, value):
        """Same bound as the role, and the same answer: the ledger's
        ``<unsafe>`` sentinel is suppressed rather than rendered, here as
        well."""
        body = self._body(
            workspace_dir,
            {"X-Osprey-Auth-Role": "operator", "X-Osprey-Auth-Role-Source": value},
        )

        assert self.ROLE_PILL in body
        assert 'class="more-menu-role-source"' not in body
        assert "&lt;unsafe&gt;" not in body

    def test_the_source_vocabulary_is_the_sidecar_s_own(self):
        """The label map's keys are the sidecar's own constants.

        Importing them at module scope would put a service package in the
        terminal's import closure for two string constants, so the drift check
        is a test — the same trade ``test_http_audit_emitters.py`` makes for
        the header names. Without it, renaming a value in ``recheck.py`` would
        silently blank the qualifier instead of failing anything.
        """
        from osprey.services.auth_sidecar.routes.recheck import (
            ROLE_SOURCE_CLAIM,
            ROLE_SOURCE_ROSTER,
        )

        assert set(app_module._ROLE_SOURCE_LABELS) == {ROLE_SOURCE_ROSTER, ROLE_SOURCE_CLAIM}
