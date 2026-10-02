"""The hub serves its shared fonts to every embedded panel.

Every panel links ``/static/fonts/fonts.css`` root-absolute, which the proxy's
rewrite turns into ``/panel/<id>/static/fonts/fonts.css``. Those requests are
answered from the hub's own ``shared_fonts`` directory, never forwarded to the
panel's backend, so a URL-backed panel that ships no fonts still renders in the
hub's typeface. The intercept covers ``static/fonts/`` only: every other
``/static/`` path still reaches the backend.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from ._proxy_fakes import panel_app

SHARED_FONTS_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "osprey" / "interfaces" / "shared_fonts"
)


@pytest.fixture
def app_and_client(tmp_path):
    """App + client with a custom panel pointing at an unused port."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    custom = [{"id": "my-dash", "label": "DASH", "url": "http://localhost:9"}]
    yield from panel_app(workspace, custom)


def _backend_must_not_be_called(app):
    app.state.proxy_client.request = AsyncMock(
        side_effect=AssertionError("font request reached the panel backend")
    )


def test_fonts_css_comes_from_the_hub(app_and_client):
    app, client = app_and_client
    _backend_must_not_be_called(app)

    resp = client.get("/panel/my-dash/static/fonts/fonts.css")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/css")
    assert resp.content == (SHARED_FONTS_DIR / "fonts.css").read_bytes()


def test_a_font_file_is_served_as_a_font(app_and_client):
    app, client = app_and_client
    _backend_must_not_be_called(app)
    font = sorted(SHARED_FONTS_DIR.glob("*.ttf"))[0]

    resp = client.get(f"/panel/my-dash/static/fonts/{font.name}")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("font/ttf")
    assert resp.content == font.read_bytes()


def test_a_path_outside_the_fonts_directory_is_refused(app_and_client):
    app, client = app_and_client
    _backend_must_not_be_called(app)

    resp = client.get("/panel/my-dash/static/fonts/%2E%2E/%2E%2E/%2E%2E/%2E%2E/pyproject.toml")

    assert resp.status_code == 404


def test_a_missing_font_is_404_without_reaching_the_backend(app_and_client):
    app, client = app_and_client
    _backend_must_not_be_called(app)

    resp = client.get("/panel/my-dash/static/fonts/no-such-font.ttf")

    assert resp.status_code == 404


def test_other_static_paths_still_reach_the_backend(app_and_client):
    app, client = app_and_client
    requested: list[str] = []

    # ``httpx.AsyncClient.request``'s signature: the proxy names every field it sends.
    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        requested.append(url)
        return httpx.Response(200, text="body{}", headers={"content-type": "text/css"})

    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

    resp = client.get("/panel/my-dash/static/other.css")

    assert resp.status_code == 200
    assert len(requested) == 1
    assert requested[0].endswith("/static/other.css")
