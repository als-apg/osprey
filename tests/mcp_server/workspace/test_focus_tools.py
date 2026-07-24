"""Tests for artifact_focus / artifact_pin honest gallery reporting.

These tools previously fired a fire-and-forget POST at the gallery and
returned ``"status": "success"`` unconditionally — a focus that never
happened (gallery down, or gallery serving a different store) was still
reported as done. The contract under test:

  - artifact_focus: the entire effect is gallery-side, so a failed or
    rejected POST is a tool error (gallery_unreachable / gallery_error).
  - artifact_pin: the durable pin lands in the shared index regardless, so
    the response stays success but reports ``gallery_notified`` honestly.
"""

import json
import urllib.error
from unittest.mock import patch

import pytest

from osprey.stores.artifact_store import initialize_artifact_store
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

_MODULE = "osprey.mcp_server.workspace.tools.focus_tools"


@pytest.fixture
def store(tmp_path):
    return initialize_artifact_store(workspace_root=tmp_path / "_agent_data")


@pytest.fixture
def entry(store):
    return store.save_file(
        file_content=b"<html></html>",
        filename="plot.html",
        artifact_type="plot_html",
        title="Test plot",
    )


@pytest.fixture
def _gallery_url():
    with patch(f"{_MODULE}.gallery_url", return_value="http://127.0.0.1:8086"):
        yield


def _focus_fn():
    from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

    return get_tool_fn(artifact_focus)


def _pin_fn():
    from osprey.mcp_server.workspace.tools.focus_tools import artifact_pin

    return get_tool_fn(artifact_pin)


class TestArtifactFocus:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_success_when_gallery_accepts(self, entry, _gallery_url):
        with patch(
            f"{_MODULE}._post_json_with_response",
            return_value=(200, {"status": "ok", "artifact_id": entry.id}),
        ):
            result = json.loads(await _focus_fn()(artifact_id=entry.id))
        assert result["status"] == "success"
        assert result["artifact_id"] == entry.id

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_when_gallery_unreachable(self, entry, _gallery_url):
        with patch(
            f"{_MODULE}._post_json_with_response",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with assert_raises_error(error_type="gallery_unreachable"):
                await _focus_fn()(artifact_id=entry.id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_when_gallery_rejects(self, entry, _gallery_url):
        """A gallery 404 means the user never saw the focus — never claim success."""
        with patch(
            f"{_MODULE}._post_json_with_response",
            return_value=(404, {"detail": f"Artifact {entry.id} not found"}),
        ):
            with assert_raises_error(error_type="gallery_error"):
                await _focus_fn()(artifact_id=entry.id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unknown_artifact_is_not_found(self, store, _gallery_url):
        with assert_raises_error(error_type="not_found"):
            await _focus_fn()(artifact_id="nonexistent")


class TestArtifactPin:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pin_reports_gallery_notified(self, entry, _gallery_url):
        with patch(
            f"{_MODULE}._post_json_with_response",
            return_value=(200, {"status": "ok"}),
        ):
            result = json.loads(await _pin_fn()(artifact_id=entry.id))
        assert result["status"] == "success"
        assert result["pinned"] is True
        assert result["gallery_notified"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pin_persists_but_reports_failed_notify(self, store, entry, _gallery_url):
        """Pin lands in the shared index even when the gallery POST fails."""
        with patch(
            f"{_MODULE}._post_json_with_response",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = json.loads(await _pin_fn()(artifact_id=entry.id))
        assert result["status"] == "success"
        assert result["pinned"] is True
        assert result["gallery_notified"] is False
        # Durable state is set regardless of the gallery being reachable.
        assert store.get_entry(entry.id).pinned is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pin_unknown_artifact_is_not_found(self, store, _gallery_url):
        with assert_raises_error(error_type="not_found"):
            await _pin_fn()(artifact_id="nonexistent")
