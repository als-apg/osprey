"""Integration tests for the /api/type-registry endpoint."""

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from osprey.interfaces.artifacts.app import create_app
from osprey.stores import type_registry as tr
from osprey.stores.artifact_forms import BASE_FORMS, form_label
from osprey.stores.type_registry import ARTIFACT_TYPES, TOOL_TYPES

#: The shared fixture both JS suites read; it is this endpoint's answer for
#: the config ``seeded_config`` writes.
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "type-registry.json"

SEEDED_EXTENSION_TYPES = """\
artifact_server:
  extension_types:
    beam_log:
      label: Beam Log
      base_form: tables
      viewer: viewers/beam_log.js
    orbit_fit:
      label: Orbit Fit
      base_form: plots
"""


@pytest.fixture(autouse=True)
def _fresh_registry():
    tr.reset_extension_types()
    yield
    tr.reset_extension_types()


@pytest.fixture
def seeded_config(tmp_path, monkeypatch):
    """A repo whose config declares two extension types, one with a viewer."""
    repo = tmp_path / "repo"
    (repo / "build").mkdir(parents=True)
    (repo / "viewers").mkdir()
    (repo / "viewers" / "beam_log.js").write_text("export function render() {}\n")
    config = repo / "build" / "config.yml"
    config.write_text(f"project_root: {repo}\n" + SEEDED_EXTENSION_TYPES)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.chdir(repo)
    monkeypatch.setattr(tr, "_entry_point_declarations", lambda eps, errors: [])
    return repo


@pytest.fixture
def app(tmp_path):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    return create_app(workspace_root=workspace)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_type_registry_endpoint_returns_200(client):
    resp = await client.get("/api/type-registry")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_type_registry_endpoint_structure(client):
    resp = await client.get("/api/type-registry")
    data = resp.json()
    assert set(data.keys()) == {
        "artifact_types",
        "base_forms",
        "categories",
        "extension_types",
        "tool_types",
    }


@pytest.mark.asyncio
async def test_type_registry_endpoint_artifact_types(client):
    resp = await client.get("/api/type-registry")
    data = resp.json()
    assert set(data["artifact_types"]) == set(ARTIFACT_TYPES)
    for _key, info in data["artifact_types"].items():
        assert "label" in info
        assert "color" in info


@pytest.mark.asyncio
async def test_type_registry_endpoint_tool_types(client):
    resp = await client.get("/api/type-registry")
    data = resp.json()
    assert set(data["tool_types"]) == set(TOOL_TYPES)


@pytest.mark.asyncio
async def test_type_registry_endpoint_base_forms(client):
    data = (await client.get("/api/type-registry")).json()
    assert data["base_forms"] == [{"key": k, "label": form_label(k)} for k in BASE_FORMS]


@pytest.mark.asyncio
async def test_type_registry_endpoint_extension_types(seeded_config, client):
    data = (await client.get("/api/type-registry")).json()
    assert data["extension_types"] == [
        {
            "key": "beam_log",
            "label": "Beam Log",
            "base_form": "tables",
            "viewer_url": "/viewers/beam_log.js",
        },
        {"key": "orbit_fit", "label": "Orbit Fit", "base_form": "plots", "viewer_url": None},
    ]
    # The viewer resolved at load; the API flags it with the route that serves it.
    assert tr.viewer_path("beam_log") == (seeded_config / "viewers" / "beam_log.js").resolve()
    assert tr.viewer_path("orbit_fit") is None


@pytest.mark.asyncio
async def test_shared_fixture_equals_endpoint_answer(seeded_config, client):
    # ``seeded_config`` is requested before ``client`` so the app loads it.
    assert (seeded_config / "build" / "config.yml").is_file()
    data = (await client.get("/api/type-registry")).json()
    assert json.loads(FIXTURE.read_text()) == data
