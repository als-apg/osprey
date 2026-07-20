"""Tests for the panel proxy's outer-prefix rewriting.

Multi-user deployments mount each user's Web Terminal at ``/u/<user>/``. The
panel proxy's ``_rewrite_content`` and its ``x-forwarded-prefix`` header must
account for that outer prefix in addition to the existing ``/panel/<id>``
prefix, so a panel's internal assets/APIs resolve under ``/u/<user>/`` rather
than escaping to the un-prefixed origin. Empty prefix (no
``OSPREY_TERMINAL_USER``) must remain byte-identical to pre-refactor behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app
from osprey.interfaces.web_terminal.routes.proxy import _rewrite_content


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

        async def fake_request(*, method, url, headers, content):
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

        async def fake_request(*, method, url, headers, content):
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

        async def fake_request(*, method, url, headers, content):
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

        async def fake_request(*, method, url, headers, content):
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

    The bluesky-panels bundles compute their API prefix at runtime
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

        import osprey.services.bluesky_panels as bp
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
