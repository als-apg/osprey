"""The hub's ``/panel/artifacts/viewers/{key}.js``: a viewer for the embedded gallery.

The hub's embedded gallery is an opaque-origin document, so its viewer import
is a credential-less CORS request. The hub answers it itself, as a C-ASSETS
path, from the same :func:`osprey.stores.type_registry.viewer_path` the
artifacts app serves — never relaying it to the backend.

The "no cookie is needed" tests carry ``no_auth_seam``: the suite-wide seam
would otherwise authenticate every TestClient request.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route

from osprey.interfaces.artifacts.app import create_app
from osprey.interfaces.common_middleware import (
    FIRST_PARTY_PANEL_VIEWERS,
    is_panel_asset_path,
)
from osprey.interfaces.web_terminal.routes import proxy
from osprey.stores import type_registry as tr

from ._proxy_fakes import panel_app

VIEWER_SOURCE = "export function render(container) { container.textContent = 'beam'; }\n"
NULL_ORIGIN = {"Origin": "null"}


@pytest.fixture(autouse=True)
def _fresh_registry():
    tr.reset_extension_types()
    yield
    tr.reset_extension_types()


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """The web terminal whose registry declares ``beam_log`` with a viewer."""
    viewers = tmp_path / "repo" / "viewers"
    viewers.mkdir(parents=True)
    viewer = viewers / "beam_log.js"
    viewer.write_text(VIEWER_SOURCE)
    (viewers / "secret.js").write_text("export const secret = 1;\n")
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    for app, client in panel_app(workspace, []):
        monkeypatch.setattr(
            tr,
            "_EXTENSION_TYPES",
            {
                "beam_log": tr.ExtensionType("beam_log", "Beam Log", "tables", viewer.resolve()),
                "orbit_fit": tr.ExtensionType("orbit_fit", "Orbit Fit", "plots", None),
            },
        )
        setattr(app.state, proxy._PANEL_STATE_MAP["artifacts"], "http://127.0.0.1:9")
        app.state.proxy_client.request = AsyncMock(
            side_effect=AssertionError("viewer request reached the panel backend")
        )
        yield app, client


@pytest.mark.no_auth_seam
def test_the_hub_serves_the_viewer_with_cors_and_no_cookie(hub):
    _app, client = hub
    resp = client.get("/panel/artifacts/viewers/beam_log.js", headers=NULL_ORIGIN)
    assert resp.status_code == 200
    assert resp.text == VIEWER_SOURCE
    assert resp.headers["content-type"].split(";")[0] == "text/javascript"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert "no-cache" in resp.headers["cache-control"]
    assert resp.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in resp.headers
    assert "set-cookie" not in resp.headers


def test_the_hub_and_the_artifacts_app_serve_the_same_bytes(hub, tmp_path, monkeypatch):
    _app, client = hub
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "no-config.yml"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "_artifacts").mkdir()
    artifacts = create_app(workspace_root=tmp_path / "_artifacts")
    # The startup load replaced the seeded registry; seed it back.
    seeded = {
        "beam_log": tr.ExtensionType(
            "beam_log", "Beam Log", "tables", (tmp_path / "repo/viewers/beam_log.js").resolve()
        )
    }
    monkeypatch.setattr(tr, "_EXTENSION_TYPES", seeded)
    direct = TestClient(artifacts).get("/viewers/beam_log.js")
    via_hub = client.get("/panel/artifacts/viewers/beam_log.js")
    assert direct.status_code == via_hub.status_code == 200
    assert direct.content == via_hub.content


@pytest.mark.no_auth_seam
def test_head_is_answered_too(hub):
    _app, client = hub
    resp = client.head("/panel/artifacts/viewers/beam_log.js", headers=NULL_ORIGIN)
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"


@pytest.mark.no_auth_seam
@pytest.mark.parametrize(
    "path",
    [
        "/panel/artifacts/viewers/no_such_type.js",
        "/panel/artifacts/viewers/orbit_fit.js",
        "/panel/artifacts/viewers/secret.js",
        "/panel/artifacts/viewers/%2E%2E.js",
    ],
)
def test_anything_but_a_declared_viewer_is_404_and_never_relayed(hub, path):
    _app, client = hub
    resp = client.get(path, headers=NULL_ORIGIN)
    assert resp.status_code == 404, path
    assert "secret" not in resp.text


@pytest.mark.no_auth_seam
@pytest.mark.parametrize(
    "path",
    [
        "/panel/artifacts/viewers/%2E%2E%2Fsecret.js",
        "/panel/artifacts/viewers/%2Fetc%2Fpasswd.js",
    ],
)
def test_an_encoded_slash_in_the_key_is_not_an_asset_path(hub, path):
    """Decoded, the key spans two segments: not exempt, so the gate refuses it."""
    _app, client = hub
    resp = client.get(path, headers=NULL_ORIGIN)
    assert resp.status_code == 401, path
    assert "access-control-allow-origin" not in resp.headers


@pytest.mark.no_auth_seam
@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_only_get_and_head_are_exempt(hub, method):
    """Any other method falls through to the gated catch-all."""
    _app, client = hub
    resp = client.request(method, "/panel/artifacts/viewers/beam_log.js")
    assert resp.status_code == 401


def test_the_viewer_route_sits_above_the_catch_all():
    paths = [route.path for route in proxy.router.routes if isinstance(route, Route)]
    route_path = f"{FIRST_PARTY_PANEL_VIEWERS}{{key}}.js"
    assert route_path == "/panel/artifacts/viewers/{key}.js"
    assert paths.index(route_path) < paths.index("/panel/{panel_id}/{path:path}")


@pytest.mark.parametrize(
    ("path", "exempt"),
    [
        ("/panel/artifacts/viewers/beam_log.js", True),
        ("/panel/artifacts/viewers/x.js", True),
        ("/panel/artifacts/viewers/", False),
        ("/panel/artifacts/viewers/.js", False),
        ("/panel/artifacts/viewers/x.mjs", False),
        ("/panel/artifacts/viewers/x.css", False),
        ("/panel/artifacts/viewers/a/b.js", False),
        ("/panel/artifacts/viewers/../static/js/x.js", False),
        ("/panel/artifacts/viewers/..", False),
        ("/panel/artifacts/viewersx/x.js", False),
        ("/panel/my-dash/viewers/x.js", False),
        ("/panel/jupyter/viewers/x.js", False),
        ("/viewers/x.js", False),
    ],
)
def test_is_panel_asset_path_admits_only_the_artifacts_viewer_path(path, exempt):
    assert is_panel_asset_path(path) is exempt


def test_the_proxied_viewport_module_rebases_the_viewer_literal(hub, monkeypatch):
    """``/viewers/`` joins the rewrite prefixes, so the embedded gallery imports via the hub."""
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
    _app, client = hub
    resp = client.get("/panel/artifacts/static/js/artifact-viewport.js")
    assert resp.status_code == 200
    assert "`/u/alice/panel/artifacts/viewers/${encodeURIComponent(" in resp.text
    assert "`/viewers/" not in resp.text


def test_viewer_paths_are_rewrite_prefixes():
    assert "/viewers/" in proxy._REWRITE_PREFIXES
