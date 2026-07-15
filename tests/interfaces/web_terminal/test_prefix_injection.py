"""Tests for THE prefix contract (Task 2.1).

Multi-user deployments run one Web Terminal container per user behind a
shared nginx front door, each mounted at ``/u/<user>/``. ``compute_url_prefix()``
computes that per-container constant from ``OSPREY_TERMINAL_USER``, and every
served HTML document (``index.html``, ``session.html``, ``safety.html``) must
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
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import compute_url_prefix, create_app

# (page id, request path) -- all three served HTML documents in scope.
_PAGES = [
    ("index", "/"),
    ("session", "/static/session.html"),
    ("safety", "/static/safety.html"),
]

# page id -> its module-entrypoint <script type="module" src="..."> path, or
# None if the page has no such entrypoint. safety.html's only module script
# is an inline `import` with no `src` attribute -- that specifier IS covered
# by the import map, so there is nothing to check here for it.
_MODULE_ENTRYPOINTS = {
    "index": "/static/js/app.js",
    "session": "/static/js/session.js",
    "safety": None,
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


class TestPrefixInjection:
    """``OSPREY_TERMINAL_USER=alice`` -> baked ``/u/alice`` prefix everywhere."""

    @pytest.mark.parametrize("page_id,path", _PAGES, ids=[p[0] for p in _PAGES])
    def test_alice_prefix_baked_into_every_page(self, workspace_dir, page_id, path):
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

    @pytest.mark.parametrize("page_id,path", _PAGES, ids=[p[0] for p in _PAGES])
    def test_empty_prefix_baked_into_every_page(self, workspace_dir, page_id, path):
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
