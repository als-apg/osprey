"""Tests for THE prefix contract (Task 2.1).

Multi-user deployments run one Web Terminal container per user behind a
shared nginx front door, each mounted at ``/u/<user>/``. ``compute_url_prefix()``
computes that per-container constant from ``OSPREY_TERMINAL_USER``, and every
served HTML document (``index.html``, ``session.html``) must
carry it as ``window.__OSPREY_PREFIX__`` plus an import map retargeting
root-absolute ``/design-system/`` and ``/static/`` ES-module specifiers under
it -- *before* any module script runs. FastAPI's ``root_path`` is deliberately
left EMPTY: nginx strips the ``/u/<user>`` prefix before proxying, so the app
serves bare paths, and a non-empty ``root_path`` would 404 every StaticFiles
Mount (all CSS/JS/fonts) — see ``create_app()`` and the bare-path regression
test below.

Import maps only retarget module *specifiers* resolved inside already-loaded
module code -- they do NOT touch ``<link href>``, a classic ``<script src>``,
or a module entrypoint's own ``src`` attribute (those are ordinary browser
URL resolutions). So each page's ``<head>`` assets and module-entrypoint
``src`` must also be explicitly prefixed via the ``prefixed()`` Jinja global,
which this file also covers.

Downstream tasks (2.2/2.3/3.x/4.x) all consume this exact contract, so the
shape asserted here is load-bearing.
"""

from __future__ import annotations

import os
import string
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.deployment.web_terminals.personas import USERNAME_CHARSET_RE
from osprey.interfaces.common_middleware import MOUNT_SEGMENT_RE
from osprey.interfaces.web_terminal.app import compute_url_prefix, create_app

# (page id, request path) -- both served HTML documents in scope.
_PAGES = [
    ("index", "/"),
    ("session", "/static/session.html"),
]

# page id -> its module-entrypoint <script type="module" src="..."> path, or
# None if the page has no such entrypoint.
_MODULE_ENTRYPOINTS = {
    "index": "/static/js/app.js",
    "session": "/static/js/session.js",
}


@pytest.fixture
def workspace_dir(tmp_path):
    """Create a temporary workspace directory for the app to watch."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "README.md").write_text("# Test workspace\n")
    return ws


class TestComputeUrlPrefix:
    """Unit-level coverage of the shared prefix helper."""

    def test_set_from_env(self):
        with patch.dict("os.environ", {"OSPREY_TERMINAL_USER": "alice"}):
            assert compute_url_prefix() == "/u/alice"

    def test_empty_when_unset(self):
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("OSPREY_TERMINAL_USER", None)
            assert compute_url_prefix() == ""

    def test_empty_when_blank(self):
        with patch.dict("os.environ", {"OSPREY_TERMINAL_USER": "   "}):
            assert compute_url_prefix() == ""


class TestAMountMustBeSpellable:
    """A name the prefix cannot carry is refused, not spelled.

    The prefix is put into URL paths and into header values — the
    forwarded-prefix every proxied panel request carries, the token exchange's
    ``Location`` — with no escaping on either side, and nginx matches the
    container's mount against the literal name. A name outside that class
    therefore has no front door to arrive through and fails the panel hop on
    the way out, so it is refused where it is read: the container does not
    start, and the message names the variable that has to change.

    A dot-segment is refused for a second reason: ``/u/..`` is spelled from
    characters a path may carry, and still climbs out of the mount it claims to
    name wherever something resolves it.
    """

    #: Names outside the class, one per way of leaving it.
    REFUSED = [
        "renée",
        "bob/../x",
        "alice bob",
        "al%69ce",
        "alice\nbob",
        ".",
        "..",
        "-x",
    ]

    @pytest.mark.parametrize("name", REFUSED)
    def test_a_name_no_front_door_can_route_is_refused(self, name):
        with patch.dict("os.environ", {"OSPREY_TERMINAL_USER": f" {name} "}):
            with pytest.raises(ValueError) as refusal:
                compute_url_prefix()

        message = str(refusal.value)
        assert "OSPREY_TERMINAL_USER" in message
        assert repr(name) in message

    @pytest.mark.parametrize("name", ["alice", "web-1", "a_b", "ALICE", "v1.2"])
    def test_every_name_the_roster_can_carry_is_spelled(self, name):
        """The render gate's charset is narrower, so what it admits clears this.

        Pinned by construction rather than by importing that pattern: the two
        gates answer different questions (which names a deployment can keep
        apart, which names this process can spell), and a row that admits more
        than the roster does is the one that keeps this one from tightening
        into a second roster rule.
        """
        with patch.dict("os.environ", {"OSPREY_TERMINAL_USER": name}):
            assert compute_url_prefix() == f"/u/{name}"

    def test_no_name_the_render_gate_admits_is_refused_here(self):
        """Containment, character by character, so the two cannot cross.

        The render gate is the narrower of the pair and produces every value
        this one reads in a deployed container. Loosening it by one character
        that a mount cannot spell would turn a rendered roster into a container
        that refuses to start, and the roster is written long before anyone
        finds that out.
        """
        alphabet = string.printable + "éü中"

        for char in alphabet:
            for name in (char, f"a{char}", f"a{char}b"):
                if USERNAME_CHARSET_RE.fullmatch(name):
                    assert MOUNT_SEGMENT_RE.fullmatch(name), name

    def test_a_container_named_outside_the_class_never_serves(self, workspace_dir):
        """The refusal lands at construction, before anything is served.

        ``create_app`` computes the mount once, so a misconfigured name costs
        the container its start — an operator reads one message in the logs
        instead of one 500 per panel, none of which names the variable.
        """
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ),
            patch.dict("os.environ", {"OSPREY_TERMINAL_USER": "renée"}),
            pytest.raises(ValueError, match="OSPREY_TERMINAL_USER"),
        ):
            create_app(shell_command="echo")


class TestPrefixInjection:
    """``OSPREY_TERMINAL_USER=alice`` -> baked ``/u/alice`` prefix everywhere."""

    # ``_PAGES`` is shared with the tests that resolve a page's entrypoint by id; this
    # one asserts on the prefix every page carries, so the id column is not read here.
    @pytest.mark.parametrize("page_id,path", _PAGES, ids=[p[0] for p in _PAGES])
    def test_alice_prefix_baked_into_every_page(self, workspace_dir, page_id, path):  # noqa: ARG002
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
                resp = c.get(path)
                assert resp.status_code == 200
                body = resp.text

                prefix_idx = body.index('window.__OSPREY_PREFIX__ = "/u/alice";')
                importmap_idx = body.index('type="importmap"')
                assert '"/design-system/": "/u/alice/design-system/"' in body
                assert '"/static/": "/u/alice/static/"' in body

                # The prefix global must be set before any module script loads.
                first_module_idx = body.find('type="module"')
                if first_module_idx != -1:
                    assert prefix_idx < first_module_idx
                    assert importmap_idx < first_module_idx

                # root_path must stay empty even with a prefix configured: see
                # test_static_mount_served_on_bare_path_when_prefix_set below.
                assert app.root_path == ""

    def test_static_mount_served_on_bare_path_when_prefix_set(self, workspace_dir):
        """Regression: with a prefix configured, real StaticFiles-mounted
        assets must still serve on their BARE path.

        nginx strips ``/u/<user>`` before proxying (nginx.conf.j2), so the app
        only ever receives bare ``/static/…`` / ``/design-system/…`` paths. A
        non-empty ``FastAPI(root_path=…)`` used to make Starlette's Mount
        routing expect the prefix in the path and 404 every asset — loading the
        multi-user UI with no CSS/JS/fonts. This pins the bug fixed at the unit
        level; ``tests/e2e/web_terminals/test_prefix_routing.py`` guards it end
        to end.
        """
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value=cfg,
            ),
            patch.dict("os.environ", {"OSPREY_TERMINAL_USER": "alice"}),
        ):
            app = create_app(shell_command="echo")
            assert app.root_path == ""  # never the prefix — see create_app note
            with TestClient(app) as c:
                # The bare paths nginx actually forwards must serve the assets.
                assert c.get("/static/js/app.js").status_code == 200
                assert c.get("/design-system/js/theme-boot.js").status_code == 200

    def test_no_base_href_introduced(self, workspace_dir):
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
                for _, path in _PAGES:
                    body = c.get(path).text
                    assert "<base" not in body.lower()


class TestHeadAssetAndEntrypointPrefixing:
    """``<link href>``/classic ``<script src>``/module-entrypoint ``src`` must
    also carry the prefix -- the import map alone cannot retarget them.
    """

    @pytest.mark.parametrize("page_id,path", _PAGES, ids=[p[0] for p in _PAGES])
    def test_alice_prefixes_head_assets_and_entrypoint(self, workspace_dir, page_id, path):
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
                body = c.get(path).text

                # A classic <script src> and a <link href> shared by every page.
                assert 'src="/u/alice/design-system/js/theme-boot.js"' in body
                assert 'href="/u/alice/design-system/css/tokens.css"' in body
                # The un-prefixed root-absolute forms must not remain.
                assert 'src="/design-system/js/theme-boot.js"' not in body
                assert 'href="/design-system/css/tokens.css"' not in body

                entrypoint = _MODULE_ENTRYPOINTS[page_id]
                if entrypoint is not None:
                    assert f'src="/u/alice{entrypoint}"' in body
                    assert f'src="{entrypoint}"' not in body

    @pytest.mark.parametrize("page_id,path", _PAGES, ids=[p[0] for p in _PAGES])
    def test_empty_prefix_head_assets_and_entrypoint_unchanged(self, workspace_dir, page_id, path):
        cfg = {"watch_dir": str(workspace_dir)}
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value=cfg,
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                body = c.get(path).text

                assert 'src="/design-system/js/theme-boot.js"' in body
                assert 'href="/design-system/css/tokens.css"' in body

                entrypoint = _MODULE_ENTRYPOINTS[page_id]
                if entrypoint is not None:
                    assert f'src="{entrypoint}"' in body


class TestPrefixEmptyWhenUnset:
    """Unset/empty ``OSPREY_TERMINAL_USER`` -> empty prefix, unchanged behavior."""

    # ``_PAGES`` is shared with the tests that resolve a page's entrypoint by id; this
    # one asserts on the prefix every page carries, so the id column is not read here.
    @pytest.mark.parametrize("page_id,path", _PAGES, ids=[p[0] for p in _PAGES])
    def test_empty_prefix_baked_into_every_page(self, workspace_dir, page_id, path):  # noqa: ARG002
        cfg = {"watch_dir": str(workspace_dir)}
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value=cfg,
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                resp = c.get(path)
                assert resp.status_code == 200
                body = resp.text

                assert 'window.__OSPREY_PREFIX__ = "";' in body
                assert '"/design-system/": "/design-system/"' in body
                assert '"/static/": "/static/"' in body
                assert "<base" not in body.lower()

                assert app.root_path == ""
