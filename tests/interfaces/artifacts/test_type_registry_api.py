"""Integration tests for the /api/type-registry endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from osprey.interfaces.artifacts.app import create_app
from osprey.mcp_server.type_registry import ARTIFACT_TYPES, DATA_TYPES, TOOL_TYPES


@pytest.fixture
def app(tmp_path):
    workspace = tmp_path / "osprey-workspace"
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
    assert set(data.keys()) == {"artifact_types", "categories", "data_types", "tool_types"}


@pytest.mark.asyncio
async def test_type_registry_endpoint_artifact_types(client):
    resp = await client.get("/api/type-registry")
    data = resp.json()
    assert set(data["artifact_types"]) == set(ARTIFACT_TYPES)
    for _key, info in data["artifact_types"].items():
        assert "label" in info
        assert "color" in info


@pytest.mark.asyncio
async def test_type_registry_endpoint_data_types(client):
    resp = await client.get("/api/type-registry")
    data = resp.json()
    assert set(data["data_types"]) == set(DATA_TYPES)


@pytest.mark.asyncio
async def test_type_registry_endpoint_tool_types(client):
    resp = await client.get("/api/type-registry")
    data = resp.json()
    assert set(data["tool_types"]) == set(TOOL_TYPES)
