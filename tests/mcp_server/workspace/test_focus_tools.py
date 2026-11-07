"""Tests for artifact_focus honest gallery reporting.

A fire-and-forget POST at the gallery that returns ``"status": "success"``
unconditionally reports a focus that never happened (gallery down, or gallery
serving a different store) as done. The entire effect is gallery-side, so a
failed or rejected POST is a tool error (gallery_unreachable / gallery_error).
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
    with patch(f"{_MODULE}.gallery_url", return_value="http://127.0.0.1:10200"):
        yield


def _focus_fn():
    from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

    return get_tool_fn(artifact_focus)


class TestArtifactFocus:
    @pytest.mark.asyncio
    async def test_success_when_gallery_accepts(self, entry, _gallery_url):
        with patch(
            f"{_MODULE}._post_json_with_response",
            return_value=(200, {"status": "ok", "artifact_id": entry.id}),
        ):
            result = json.loads(await _focus_fn()(artifact_id=entry.id))
        assert result["status"] == "success"
        assert result["artifact_id"] == entry.id

    @pytest.mark.asyncio
    async def test_error_when_gallery_unreachable(self, entry, _gallery_url):
        with patch(
            f"{_MODULE}._post_json_with_response",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with assert_raises_error(error_type="gallery_unreachable"):
                await _focus_fn()(artifact_id=entry.id)

    @pytest.mark.asyncio
    async def test_error_when_gallery_rejects(self, entry, _gallery_url):
        """A gallery 404 means the user never saw the focus — never claim success."""
        with patch(
            f"{_MODULE}._post_json_with_response",
            return_value=(404, {"detail": f"Artifact {entry.id} not found"}),
        ):
            with assert_raises_error(error_type="gallery_error"):
                await _focus_fn()(artifact_id=entry.id)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("store")
    async def test_unknown_artifact_is_not_found(self, _gallery_url):
        with assert_raises_error(error_type="not_found"):
            await _focus_fn()(artifact_id="nonexistent")


# ---- the session a focus was set in (ruling 15) ----------------------


@pytest.fixture
def _no_session(monkeypatch):
    """This process in no session at all: every scope variable cleared."""
    from tests.mcp_server.conftest import _SCOPE_ENV

    for name in _SCOPE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def _dispatch_run(_no_session, monkeypatch):
    """A dispatch run: it has a key too, but a dispatch run owns no session."""
    monkeypatch.setenv("OSPREY_SESSION_ID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("OSPREY_DISPATCH_RUN_ID", "run-1")


def _posted(mock):
    """The JSON payload of the one gallery POST *mock* saw."""
    assert mock.call_count == 1
    return mock.call_args.args[1]


class TestFocusCarriesSession:
    @pytest.mark.asyncio
    async def test_web_focus_posts_transcript(self, entry, web_session, _gallery_url):
        with patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})) as post:
            await _focus_fn()(artifact_id=entry.id, fullscreen=True)
        assert _posted(post) == {
            "artifact_id": entry.id,
            "fullscreen": True,
            "transcript_id": web_session.transcript,
        }

    @pytest.mark.asyncio
    async def test_cli_focus_posts_cli_transcript(self, entry, cli_session, _gallery_url):
        with patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})) as post:
            await _focus_fn()(artifact_id=entry.id)
        assert _posted(post) == {"artifact_id": entry.id, "transcript_id": cli_session.transcript}

    @pytest.mark.asyncio
    async def test_dispatch_focus_posts_no_transcript(self, entry, _dispatch_run, _gallery_url):
        with patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})) as post:
            result = json.loads(await _focus_fn()(artifact_id=entry.id))
        assert result["status"] == "success"
        assert _posted(post) == {"artifact_id": entry.id}

    @pytest.mark.asyncio
    async def test_focus_without_session_posts_no_transcript(
        self, entry, _no_session, _gallery_url
    ):
        with patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})) as post:
            await _focus_fn()(artifact_id=entry.id)
        assert "transcript_id" not in _posted(post)

    @pytest.mark.asyncio
    async def test_focus_scope_resolved_at_store_root(self, entry, cli_session, _gallery_url):
        """The pointer is read under the store's workspace root, never a default root."""
        with (
            patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})),
            patch("osprey.stores.artifact_scope.current_scope") as scope,
        ):
            scope.return_value.transcript = None
            await _focus_fn()(artifact_id=entry.id)
        scope.assert_called_once()
        assert scope.call_args.kwargs["root"] == cli_session.root

    def test_docstring_does_not_claim_the_tool_writes_focus_state(self):
        from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

        doc = (getattr(artifact_focus, "fn", artifact_focus).__doc__) or ""
        assert "updates focus_state.txt" not in " ".join(doc.split())
