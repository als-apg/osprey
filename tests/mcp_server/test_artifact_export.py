"""Tests for the artifact_export MCP tool."""

import json
from unittest.mock import patch

import pytest

from osprey.mcp_server.export.converter import PlaywrightNotInstalledError
from osprey.stores.artifact_store import ArtifactStore
from tests.mcp_server.conftest import (
    assert_error,
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)


def _get_artifact_export():
    from osprey.mcp_server.workspace.tools.artifact_export import artifact_export

    return get_tool_fn(artifact_export)


@pytest.mark.unit
async def test_export_html_to_png(tmp_path, monkeypatch):
    """HTML artifact is converted to PNG via mocked converter."""
    monkeypatch.chdir(tmp_path)

    store = ArtifactStore(workspace_root=tmp_path)
    entry = store.save_file(
        file_content=b"<html><body>Plot</body></html>",
        filename="plot.html",
        artifact_type="html",
        title="My Plot",
        mime_type="text/html",
        tool_source="execute",
    )

    async def fake_convert(html_path, output_path, fmt="png", width=1200, height=800):
        from pathlib import Path

        Path(output_path).write_bytes(b"\x89PNG fake image")
        return Path(output_path).resolve()

    with (
        patch("osprey.stores.artifact_store.get_artifact_store", return_value=store),
        patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ),
    ):
        fn = _get_artifact_export()
        result = await fn(artifact_id=entry.id)

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["artifact_id"] != entry.id  # new artifact created
    assert data["artifact_type"] == "image"


@pytest.mark.unit
async def test_export_already_target_format(tmp_path, monkeypatch):
    """PNG artifact returns existing entry without conversion."""
    monkeypatch.chdir(tmp_path)

    store = ArtifactStore(workspace_root=tmp_path)
    entry = store.save_file(
        file_content=b"\x89PNG data",
        filename="screenshot.png",
        artifact_type="image",
        title="Screenshot",
        mime_type="image/png",
        tool_source="screen_capture",
    )

    with patch("osprey.stores.artifact_store.get_artifact_store", return_value=store):
        fn = _get_artifact_export()
        result = await fn(artifact_id=entry.id, format="png")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["artifact_id"] == entry.id  # same artifact, no conversion


@pytest.mark.unit
async def test_export_nonexistent_artifact(tmp_path, monkeypatch):
    """Non-existent artifact ID returns not_found error."""
    monkeypatch.chdir(tmp_path)

    store = ArtifactStore(workspace_root=tmp_path)

    with patch("osprey.stores.artifact_store.get_artifact_store", return_value=store):
        fn = _get_artifact_export()
        with assert_raises_error(error_type="not_found") as _exc_ctx:
            await fn(artifact_id="nonexistent-id")

    data = _exc_ctx["envelope"]


@pytest.mark.unit
async def test_export_non_html_artifact(tmp_path, monkeypatch):
    """Non-HTML artifact (e.g. JSON) returns conversion_not_supported error."""
    monkeypatch.chdir(tmp_path)

    store = ArtifactStore(workspace_root=tmp_path)
    entry = store.save_file(
        file_content=b'{"key": "value"}',
        filename="data.json",
        artifact_type="json",
        title="JSON Data",
        mime_type="application/json",
        tool_source="test",
    )

    with patch("osprey.stores.artifact_store.get_artifact_store", return_value=store):
        fn = _get_artifact_export()
        with assert_raises_error(error_type="conversion_not_supported") as _exc_ctx:
            await fn(artifact_id=entry.id)

    data = _exc_ctx["envelope"]


@pytest.mark.unit
async def test_export_playwright_missing(tmp_path, monkeypatch):
    """Missing Playwright raises dependency_missing error."""
    monkeypatch.chdir(tmp_path)

    store = ArtifactStore(workspace_root=tmp_path)
    entry = store.save_file(
        file_content=b"<html></html>",
        filename="plot.html",
        artifact_type="html",
        title="Plot",
        mime_type="text/html",
        tool_source="test",
    )

    async def raise_pw_error(*args, **kwargs):
        raise PlaywrightNotInstalledError("Playwright not installed")

    with (
        patch("osprey.stores.artifact_store.get_artifact_store", return_value=store),
        patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=raise_pw_error,
        ),
    ):
        fn = _get_artifact_export()
        with assert_raises_error(error_type="dependency_missing") as _exc_ctx:
            await fn(artifact_id=entry.id)

    data = _exc_ctx["envelope"]
