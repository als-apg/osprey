"""Tests for the panel proxy's outer-prefix rewriting.

Multi-user deployments mount each user's Web Terminal at ``/u/<user>/``. The
panel proxy's ``_rewrite_content`` and its ``x-forwarded-prefix`` header must
account for that outer prefix in addition to the existing ``/panel/<id>``
prefix, so a panel's internal assets/APIs resolve under ``/u/<user>/`` rather
than escaping to the un-prefixed origin. Empty prefix (no
``OSPREY_TERMINAL_USER``) must remain byte-identical to pre-refactor behavior.
The module also covers the prefixes a single panel adds to the fixed list.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app
from osprey.interfaces.web_terminal.routes.proxy import (
    _panel_rewrite_prefixes,
    _path_rewrite_prefix,
    _rewrite_content,
)


def _make_client(workspace_dir, custom_panels):
    """Create a TestClient with custom panels configured."""
    enabled = set(UNIVERSAL_PANELS)
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled, custom_panels, None),
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield app, c


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def app_and_client(workspace_dir):
    """App + client with a custom panel (my-dash → http://localhost:9000)."""
    custom = [
        {"id": "my-dash", "label": "DASH", "url": "http://localhost:9000"},
    ]
    yield from _make_client(workspace_dir, custom)


_PVINFO_PANEL = {
    "id": "pvinfo",
    "label": "PV INFO",
    "url": "http://localhost:9100",
    "path": "/pvinfo/",
    "configDefined": True,
}


@pytest.fixture
def pvinfo_app_and_client(workspace_dir):
    """App + client with one config-defined panel hosted under ``/pvinfo/``."""
    yield from _make_client(workspace_dir, [dict(_PVINFO_PANEL)])


#: An SPA's ``index.html`` as served by a backend hosted under ``/pvinfo/``.
_SPA_INDEX = (
    "<!doctype html><html><head>"
    '<link rel="icon" href="/pvinfo/favicon-32x32.png">'
    '<link rel="manifest" href="/pvinfo/manifest.json">'
    '<script type="module" src="/pvinfo/assets/index-0EzkYvPl.js"></script>'
    '<link rel="stylesheet" href="/pvinfo/assets/index-C0IM9X5z.css">'
    '</head><body><div id="root"></div></body></html>'
)

_SPA_REFERENCES = (
    "favicon-32x32.png",
    "manifest.json",
    "assets/index-0EzkYvPl.js",
    "assets/index-C0IM9X5z.css",
)


def _proxy_body(app, client, url, body, content_type):
    """Proxy ``url`` against a backend that answers ``body`` as ``content_type``."""

    # ``httpx.AsyncClient.request``'s signature: the proxy names every field it sends,
    # and the body asserts on the ones this test is about.
    async def fake_request(*, method, url, headers, content):  # noqa: ARG001
        return httpx.Response(status_code=200, text=body, headers={"content-type": content_type})

    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
    resp = client.get(url)
    assert resp.status_code == 200
    return resp.text


class TestRewriteContentPrefix:
    """Unit-level: ``_rewrite_content`` honors the outer per-user prefix."""

    def test_prefix_applied_with_outer_prefix(self):
        body = 'var x = "/static/js/foo.js";'
        result = _rewrite_content(body, "my-dash", outer_prefix="/u/alice")
        assert '"/u/alice/panel/my-dash/static/js/foo.js"' in result

    def test_prefix_empty_matches_unprefixed_output(self):
        """Empty outer prefix ⇒ byte-identical to the pre-refactor output."""
        body = 'var x = "/static/js/foo.js";'
        result = _rewrite_content(body, "my-dash", outer_prefix="")
        assert result == 'var x = "/panel/my-dash/static/js/foo.js";'

    def test_default_outer_prefix_is_empty(self):
        """Omitting outer_prefix must match explicit empty-string behavior."""
        body = 'var x = "/static/js/foo.js";'
        assert _rewrite_content(body, "my-dash") == _rewrite_content(
            body, "my-dash", outer_prefix=""
        )


class TestProxyPrefixIntegration:
    """End-to-end through the proxy route: outer prefix sourced from OSPREY_TERMINAL_USER."""

    def test_x_forwarded_prefix_with_user(self, app_and_client, monkeypatch):
        app, client = app_and_client
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")

        captured_headers = {}

        # ``httpx.AsyncClient.request``'s signature: the proxy names every field it sends,
        # and the body asserts on the ones this test is about.
        async def fake_request(*, method, url, headers, content):  # noqa: ARG001
            captured_headers.update(headers)
            return httpx.Response(
                status_code=200,
                json={"ok": True},
                headers={"content-type": "application/json"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/api/status")
        assert resp.status_code == 200
        assert captured_headers.get("x-forwarded-prefix") == "/u/alice/panel/my-dash"

    def test_x_forwarded_prefix_empty_user(self, app_and_client, monkeypatch):
        app, client = app_and_client
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)

        captured_headers = {}

        # ``httpx.AsyncClient.request``'s signature: the proxy names every field it sends,
        # and the body asserts on the ones this test is about.
        async def fake_request(*, method, url, headers, content):  # noqa: ARG001
            captured_headers.update(headers)
            return httpx.Response(
                status_code=200,
                json={"ok": True},
                headers={"content-type": "application/json"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/api/status")
        assert resp.status_code == 200
        assert captured_headers.get("x-forwarded-prefix") == "/panel/my-dash"

    def test_rewritten_body_carries_outer_prefix(self, app_and_client, monkeypatch):
        app, client = app_and_client
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")

        js_body = 'var x = "/static/js/foo.js";'

        # ``httpx.AsyncClient.request``'s signature: the proxy names every field it sends,
        # and the body asserts on the ones this test is about.
        async def fake_request(*, method, url, headers, content):  # noqa: ARG001
            return httpx.Response(
                status_code=200,
                text=js_body,
                headers={"content-type": "application/javascript"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/static/js/gallery.js")
        assert resp.status_code == 200
        assert '"/u/alice/panel/my-dash/static/js/foo.js"' in resp.text

    def test_rewritten_body_empty_user_unchanged(self, app_and_client, monkeypatch):
        """Regression: no OSPREY_TERMINAL_USER ⇒ original /panel/<id>/... output."""
        app, client = app_and_client
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)

        js_body = 'var x = "/static/js/foo.js";'

        # ``httpx.AsyncClient.request``'s signature: the proxy names every field it sends,
        # and the body asserts on the ones this test is about.
        async def fake_request(*, method, url, headers, content):  # noqa: ARG001
            return httpx.Response(
                status_code=200,
                text=js_body,
                headers={"content-type": "application/javascript"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/static/js/gallery.js")
        assert resp.status_code == 200
        assert resp.text == 'var x = "/panel/my-dash/static/js/foo.js";'


class TestPanelBundleRewriteCollisions:
    """Shipped prefix-aware panel bundles must not collide with the rewrite list.

    The bluesky-web bundles compute their API prefix at runtime
    (``panelApiPrefix()``) and prepend it to fetch paths. If such a bundle also
    embeds a quote-delimited literal starting with one of the proxy's
    ``_REWRITE_PREFIXES``, the proxy rewrites the literal server-side and the
    runtime prefix is prepended on top — a double-prefixed URL that 404s (or,
    worse, appears to work only while the runtime prefix is broken).

    Asset prefixes are exempt: bare ``/design-system/…``, ``/static/…``, and
    ``/assets/…`` specifiers are exactly what the rewrite exists for (module
    imports and stylesheet hrefs cannot be runtime-prefixed).
    """

    ASSET_PREFIXES = ("/design-system/", "/static/", "/assets/")

    def test_bluesky_panel_bundles_have_no_api_literal_collisions(self):
        import re
        from pathlib import Path

        import osprey.interfaces.bluesky_web as bp
        from osprey.interfaces.web_terminal.routes.proxy import _REWRITE_PREFIXES

        panels_dir = Path(bp.__file__).parent / "panels"
        assert panels_dir.is_dir(), f"panel bundles not found at {panels_dir}"

        api_prefixes = [p for p in _REWRITE_PREFIXES if p not in self.ASSET_PREFIXES]
        offenders: list[str] = []
        for path in sorted(panels_dir.rglob("*")):
            if path.suffix not in {".js", ".html", ".css"}:
                continue
            body = path.read_text()
            for prefix in api_prefixes:
                pattern = r"""(?<=["'`])""" + re.escape(prefix)
                for match in re.finditer(pattern, body):
                    line = body.count("\n", 0, match.start()) + 1
                    offenders.append(f"{path.relative_to(panels_dir)}:{line}: {prefix!r}")
        assert not offenders, (
            "quote-delimited literals in prefix-aware panel bundles collide with "
            "_REWRITE_PREFIXES (the proxy would rewrite them, double-prefixing the "
            "runtime panelApiPrefix()):\n  " + "\n  ".join(offenders)
        )

    def test_an_injected_panel_path_rewrites_nothing_in_a_shipped_body(self):
        """The paths the build injects leave every shipped body they serve unchanged.

        Byte-equality, not a literal scan: the bare ``/dashboard`` prefix does
        match the dispatcher page's ``/dashboard/…`` literals, and the fixed
        ``/dashboard/`` entry consumes them first, so the outcome is the same.
        """
        from pathlib import Path

        import osprey.dispatch
        import osprey.interfaces.bluesky_web as bp

        panels_dir = Path(bp.__file__).parent / "panels"
        bodies = [
            path
            for path in sorted(panels_dir.rglob("*"))
            if path.suffix in {".js", ".html", ".css"}
        ]
        bodies.append(Path(osprey.dispatch.__file__).parent / "dashboard.html")
        assert len(bodies) > 1, f"no shipped panel bodies found under {panels_dir}"

        changed: list[str] = []
        for panel_id, panel_path in (("bluesky", "/bluesky/"), ("events", "/dashboard")):
            for path in bodies:
                body = path.read_text()
                if _rewrite_content(body, panel_id) != _rewrite_content(
                    body, panel_id, "", (panel_path,)
                ):
                    changed.append(f"{panel_id} {panel_path!r}: {path.name}")
        assert not changed, "derived panel prefixes rewrote shipped bodies:\n  " + "\n  ".join(
            changed
        )


class TestPanelPathIsARewritePrefix:
    """A config-defined panel's own ``path`` is a rewrite prefix for its responses."""

    def test_the_path_component_is_the_derived_prefix(self):
        assert _path_rewrite_prefix("/pvinfo/") == ("/pvinfo/",)
        assert _path_rewrite_prefix("/dashboard") == ("/dashboard",)
        assert _path_rewrite_prefix("/bluesky/") == ("/bluesky/",)

    @pytest.mark.parametrize("path", ["/", "", "panel/", "//elsewhere", None])
    def test_a_root_path_derives_nothing(self, path):
        assert _path_rewrite_prefix(path) == ()

    @pytest.mark.parametrize(
        "path",
        [
            "/vnc.html?path=/u/alice/panel/phoebus/websockify&autoconnect=1",
            "/vnc.html?path=panel/phoebus/websockify&autoconnect=1",
        ],
    )
    def test_a_query_string_is_not_part_of_the_prefix(self, path):
        assert _path_rewrite_prefix(path) == ("/vnc.html",)

    def test_a_sub_path_spa_index_resolves_into_the_panel(self):
        result = _rewrite_content(_SPA_INDEX, "pvinfo", "", ("/pvinfo/",))
        for ref in _SPA_REFERENCES:
            assert f'"/panel/pvinfo/pvinfo/{ref}"' in result
        assert '"/pvinfo/' not in result

    def test_the_outer_prefix_composes_with_a_derived_one(self):
        result = _rewrite_content(_SPA_INDEX, "pvinfo", "/u/controls", ("/pvinfo/",))
        assert '"/u/controls/panel/pvinfo/pvinfo/assets/index-0EzkYvPl.js"' in result

    def test_the_prefix_order_does_not_change_the_result(self):
        body = 'const base="/pvinfo";fetch("/pvinfo/api/x");'
        forward = _rewrite_content(body, "pvinfo", "", ("/pvinfo/", "/pvinfo"))
        backward = _rewrite_content(body, "pvinfo", "", ("/pvinfo", "/pvinfo/"))
        assert forward == backward
        assert '"/panel/pvinfo/pvinfo/api/x"' in forward

    def test_the_panel_lookup_answers_from_the_configured_path(self, pvinfo_app_and_client):
        app, client = pvinfo_app_and_client
        text = _proxy_body(app, client, "/panel/pvinfo/pvinfo/", _SPA_INDEX, "text/html")
        assert "/panel/pvinfo/pvinfo/assets/index-0EzkYvPl.js" in text

    def test_a_runtime_panel_contributes_no_prefix(self, workspace_dir):
        panel = {k: v for k, v in _PVINFO_PANEL.items() if k != "configDefined"}
        for app, client in _make_client(workspace_dir, [panel]):
            assert _panel_rewrite_prefixes(SimpleNamespace(app=app), "pvinfo") == ()
            text = _proxy_body(app, client, "/panel/pvinfo/pvinfo/", _SPA_INDEX, "text/html")
            for ref in _SPA_REFERENCES:
                assert f'"/pvinfo/{ref}"' in text

    def _declared(self, workspace_dir, **extra):
        """A client whose ``pvinfo`` panel carries ``extra`` on top of its path."""
        return _make_client(workspace_dir, [{**_PVINFO_PANEL, **extra}])

    _JS_BODY = 'const base="/pvinfo";fetch("/pvinfo/api/x");'

    def _proxied_js(self, workspace_dir, **extra):
        for app, client in self._declared(workspace_dir, **extra):
            return _proxy_body(
                app, client, "/panel/pvinfo/pvinfo/app.js", self._JS_BODY, "text/javascript"
            )
        raise AssertionError("no client")

    def test_a_declared_prefix_joins_the_derived_one(self, workspace_dir, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        text = self._proxied_js(workspace_dir, rewritePrefixes=["/pvinfo"])
        assert text == 'const base="/panel/pvinfo/pvinfo";fetch("/panel/pvinfo/pvinfo/api/x");'

    def test_a_declared_prefix_is_accepted_as_a_bare_string(self, workspace_dir, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        text = self._proxied_js(workspace_dir, rewritePrefixes="/pvinfo")
        assert text == 'const base="/panel/pvinfo/pvinfo";fetch("/panel/pvinfo/pvinfo/api/x");'

    def test_a_repeated_prefix_substitutes_once(self, workspace_dir, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        repeated = self._proxied_js(workspace_dir, rewritePrefixes=["/pvinfo/", "/pvinfo/"])
        single = self._proxied_js(workspace_dir, rewritePrefixes=["/pvinfo/"])
        assert repeated == single
        for app, _client in self._declared(workspace_dir, rewritePrefixes=["/pvinfo/", "/pvinfo/"]):
            assert _panel_rewrite_prefixes(SimpleNamespace(app=app), "pvinfo") == ("/pvinfo/",)

    def test_an_opted_in_json_bootstrap_carries_the_prefixes(self, workspace_dir, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        body = '{"api_url": "/pvinfo/api"}'
        for app, client in self._declared(workspace_dir, rewriteJsonPaths=["/config.json"]):
            opted_in = _proxy_body(
                app, client, "/panel/pvinfo/pvinfo/config.json", body, "application/json"
            )
            other = _proxy_body(
                app, client, "/panel/pvinfo/pvinfo/status.json", body, "application/json"
            )
        assert opted_in == '{"api_url": "/panel/pvinfo/pvinfo/api"}'
        assert other == body

    def test_a_runtime_panel_ignores_a_declared_prefix(self, workspace_dir, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        panel = {k: v for k, v in _PVINFO_PANEL.items() if k != "configDefined"}
        panel["rewritePrefixes"] = ["/pvinfo"]
        for app, client in _make_client(workspace_dir, [panel]):
            text = _proxy_body(
                app, client, "/panel/pvinfo/pvinfo/app.js", self._JS_BODY, "text/javascript"
            )
        assert text == self._JS_BODY

    def test_a_default_path_panel_is_unchanged(self, app_and_client, monkeypatch):
        app, client = app_and_client
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        text = _proxy_body(
            app,
            client,
            "/panel/my-dash/static/js/gallery.js",
            'var x = "/static/js/foo.js";',
            "application/javascript",
        )
        assert text == 'var x = "/panel/my-dash/static/js/foo.js";'
