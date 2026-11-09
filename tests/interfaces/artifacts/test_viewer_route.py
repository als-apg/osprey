"""The artifacts app's ``GET /viewers/{key}.js``: an extension type's viewer module.

The route serves the one file :func:`osprey.stores.type_registry.viewer_path`
resolved for the key at load, and nothing else: no request path ever reaches
the filesystem. ``/api/type-registry`` flags the types that have one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from osprey.interfaces.artifacts.app import create_app
from osprey.stores import type_registry as tr

VIEWER_SOURCE = "export function render(container) { container.textContent = 'beam'; }\n"


@pytest.fixture(autouse=True)
def _fresh_registry():
    tr.reset_extension_types()
    yield
    tr.reset_extension_types()


@pytest.fixture
def app(tmp_path, monkeypatch):
    """The artifacts app, started where no config declares any extension type."""
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "no-config.yml"))
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    return create_app(workspace_root=workspace)


@pytest.fixture
def seeded_registry(app, tmp_path, monkeypatch) -> Path:  # noqa: ARG001 - after app startup
    """A loaded registry: ``beam_log`` declares a viewer, ``orbit_fit`` none.

    Seeded after the app's own startup load, which would replace it. A sibling
    ``secret.js`` sits next to the viewer: nothing may serve it.
    """
    viewers = tmp_path / "repo" / "viewers"
    viewers.mkdir(parents=True)
    viewer = viewers / "beam_log.js"
    viewer.write_text(VIEWER_SOURCE)
    (viewers / "secret.js").write_text("export const secret = 1;\n")
    monkeypatch.setattr(
        tr,
        "_EXTENSION_TYPES",
        {
            "beam_log": tr.ExtensionType("beam_log", "Beam Log", "tables", viewer.resolve()),
            "orbit_fit": tr.ExtensionType("orbit_fit", "Orbit Fit", "plots", None),
        },
    )
    return viewer


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
@pytest.mark.usefixtures("seeded_registry")
async def test_the_declared_viewer_is_served(client):
    resp = await client.get("/viewers/beam_log.js")
    assert resp.status_code == 200
    assert resp.text == VIEWER_SOURCE
    assert resp.headers["content-type"].split(";")[0] == "text/javascript"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert "no-cache" in resp.headers["cache-control"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("seeded_registry")
async def test_the_viewer_route_carries_no_content_sandbox(client):
    """Installed facility code, not agent content: ``not_content``, no opaque CSP."""
    resp = await client.get("/viewers/beam_log.js")
    assert resp.status_code == 200
    assert "sandbox" not in resp.headers.get("content-security-policy", "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/viewers/no_such_type.js",  # unknown key
        "/viewers/orbit_fit.js",  # a type without a viewer
        "/viewers/secret.js",  # a file beside a viewer is not a key
        "/viewers/%2E%2E.js",
        "/viewers/%2E%2E%2Fsecret.js",
        "/viewers/..%2Fviewers%2Fsecret.js",
        "/viewers/%2Fetc%2Fpasswd.js",
        "/viewers//etc/passwd.js",
        "/viewers/beam_log.mjs",
        "/viewers/beam_log",
    ],
)
@pytest.mark.usefixtures("seeded_registry")
async def test_anything_but_a_declared_viewer_is_404(client, path):
    resp = await client.get(path)
    assert resp.status_code == 404, path
    assert "secret" not in resp.text


@pytest.mark.asyncio
async def test_an_empty_registry_serves_nothing(client):
    resp = await client.get("/viewers/beam_log.js")
    assert resp.status_code == 404


@pytest.mark.asyncio
@pytest.mark.usefixtures("seeded_registry")
async def test_type_registry_flags_the_types_with_a_viewer(client):
    data = (await client.get("/api/type-registry")).json()
    by_key = {ext["key"]: ext for ext in data["extension_types"]}
    assert by_key["beam_log"]["viewer_url"] == "/viewers/beam_log.js"
    assert by_key["orbit_fit"]["viewer_url"] is None
    # The flag names exactly the route that serves the module.
    assert (await client.get(by_key["beam_log"]["viewer_url"])).text == VIEWER_SOURCE
