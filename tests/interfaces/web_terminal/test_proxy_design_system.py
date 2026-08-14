"""The embedded-panel design-system intercept.

Every panel's HTML loads ``/design-system/css/tokens.css`` root-absolute, which
the proxy's rewrite turns into ``/panel/<id>/design-system/css/tokens.css``.
Those requests must be answered from the HUB's copy of the design system rather
than forwarded to the sidecar — otherwise a sidecar whose build predates a token
change renders a different palette than the terminal embedding it, and the theme
the hub broadcasts resolves to the wrong colors inside that one frame.

These tests pin that behavior, its route-ordering precondition (the intercept is
declared above the ``{path:path}`` catch-all), and its path containment.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app

DESIGN_SYSTEM_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "design_system"
    / "static"
)


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def app_and_client(workspace_dir):
    """App + client with a custom panel pointing at an unused port.

    The sidecar is deliberately unreachable: any design-system request that
    escapes the intercept and reaches the generic proxy fails loudly with a
    502 instead of quietly succeeding.
    """
    enabled = set(UNIVERSAL_PANELS)
    custom = [{"id": "my-dash", "label": "DASH", "url": "http://localhost:9"}]
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled, custom, None),
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield app, c


class TestDesignSystemIntercept:
    def test_serves_hub_tokens_css_not_the_sidecar_copy(self, app_and_client):
        """tokens.css comes byte-for-byte from the hub's own design system."""
        app, client = app_and_client

        # Any call through the generic proxy is a failure of the intercept.
        app.state.proxy_client.request = AsyncMock(
            side_effect=AssertionError("design-system request reached the sidecar")
        )

        resp = client.get("/panel/my-dash/design-system/css/tokens.css")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/css")
        on_disk = (DESIGN_SYSTEM_DIR / "css" / "tokens.css").read_text(encoding="utf-8")
        assert resp.text == on_disk

    def test_intercept_wins_over_the_catch_all_proxy_route(self, app_and_client):
        """Route ordering: the specific design-system path is matched first.

        Declaring the intercept below ``/panel/{panel_id}/{path:path}`` would
        make it unreachable, and this is the only thing that catches that.
        """
        app, client = app_and_client
        forwarded = []

        async def fake_request(*, method, url, headers, content):
            forwarded.append(str(url))
            return httpx.Response(200, text="from-sidecar", headers={"content-type": "text/css"})

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        ds = client.get("/panel/my-dash/design-system/css/base.css")
        assert ds.text != "from-sidecar"
        assert forwarded == []

        # A non-design-system path still goes to the sidecar.
        other = client.get("/panel/my-dash/static/panel.css")
        assert other.text == "from-sidecar"
        assert len(forwarded) == 1

    def test_binary_asset_served_with_guessed_type(self, app_and_client):
        """Non-text assets pass through as bytes rather than being rewritten."""
        _, client = app_and_client
        fonts = list(DESIGN_SYSTEM_DIR.rglob("*.woff2"))
        if not fonts:
            pytest.skip("no binary asset in the design-system tree to exercise")
        rel = fonts[0].relative_to(DESIGN_SYSTEM_DIR).as_posix()

        resp = client.get(f"/panel/my-dash/design-system/{rel}")
        assert resp.status_code == 200
        assert resp.content == fonts[0].read_bytes()

    @pytest.mark.parametrize(
        "attack",
        [
            "../../web_terminal/app.py",
            "../../../../../../etc/passwd",
            "css/../../../__init__.py",
        ],
    )
    def test_path_traversal_is_contained(self, app_and_client, attack):
        """A traversal out of the static root 404s instead of leaking a file."""
        _, client = app_and_client
        resp = client.get(f"/panel/my-dash/design-system/{attack}")
        assert resp.status_code == 404

    def test_missing_asset_404s(self, app_and_client):
        _, client = app_and_client
        resp = client.get("/panel/my-dash/design-system/css/no-such-file.css")
        assert resp.status_code == 404

    def test_self_references_are_rewritten_into_the_panel_namespace(self, app_and_client):
        """Root-absolute /design-system/... inside a JS asset is namespaced.

        osprey-theme-switcher.js dynamically imports '/design-system/js/
        theme-manager.js'; unrewritten that would load the hub's module from
        the browser origin, outside the panel's proxied namespace.
        """
        _, client = app_and_client
        resp = client.get("/panel/my-dash/design-system/js/components/osprey-theme-switcher.js")
        assert resp.status_code == 200
        assert "/panel/my-dash/design-system/js/theme-manager.js" in resp.text
