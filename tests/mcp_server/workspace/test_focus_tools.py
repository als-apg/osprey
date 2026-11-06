"""Tests for artifact_focus / artifact_pin honest gallery reporting.

A fire-and-forget POST at the gallery that returns ``"status": "success"``
unconditionally reports a focus that never happened (gallery down, or gallery
serving a different store) as done. The contract under test:

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
    with patch(f"{_MODULE}.gallery_url", return_value="http://127.0.0.1:10200"):
        yield


def _focus_fn():
    from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

    return get_tool_fn(artifact_focus)


def _pin_fn():
    from osprey.mcp_server.workspace.tools.focus_tools import artifact_pin

    return get_tool_fn(artifact_pin)


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


class TestArtifactPin:
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

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("store")
    async def test_pin_unknown_artifact_is_not_found(self, _gallery_url):
        with assert_raises_error(error_type="not_found"):
            await _pin_fn()(artifact_id="nonexistent")


# ---- the session a pin or focus was set in (ruling 15) ----------------------


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


class TestPinRecordsSession:
    @pytest.mark.asyncio
    async def test_web_pin_records_current_transcript(
        self, store, entry, web_session, _gallery_url
    ):
        """A web session's pin records the transcript the key points at now."""
        web_session.clear("22222222-2222-4222-8222-222222222222")
        with patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})) as post:
            result = json.loads(await _pin_fn()(artifact_id=entry.id))
        assert result["status"] == "success"
        assert store.get_entry(entry.id).metadata["pinned_in"] == web_session.transcript
        assert _posted(post) == {"pinned": True, "transcript_id": web_session.transcript}

    @pytest.mark.asyncio
    async def test_cli_pin_records_cli_transcript(self, store, entry, cli_session, _gallery_url):
        """An ``osprey chat`` pin records the conversation its pointer names, after a /clear."""
        cli_session.clear("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
        with patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})) as post:
            await _pin_fn()(artifact_id=entry.id)
        assert store.get_entry(entry.id).metadata["pinned_in"] == cli_session.transcript
        assert _posted(post)["transcript_id"] == cli_session.transcript

    @pytest.mark.asyncio
    async def test_dispatch_pin_records_nothing(self, store, entry, _dispatch_run, _gallery_url):
        with patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})) as post:
            result = json.loads(await _pin_fn()(artifact_id=entry.id))
        stored = store.get_entry(entry.id)
        assert result["pinned"] is True and stored.pinned is True
        assert "pinned_in" not in stored.metadata
        assert _posted(post) == {"pinned": True}

    @pytest.mark.asyncio
    async def test_pin_without_session_records_nothing(
        self, store, entry, _no_session, _gallery_url
    ):
        with patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})) as post:
            await _pin_fn()(artifact_id=entry.id)
        assert "pinned_in" not in store.get_entry(entry.id).metadata
        assert "transcript_id" not in _posted(post)

    @pytest.mark.asyncio
    async def test_pin_rejected_401_still_lands_in_index(
        self, store, entry, web_session, _gallery_url
    ):
        """The index record is the correctness path; a refused notification is only reported."""
        with patch(
            f"{_MODULE}._post_json_with_response", return_value=(401, {"detail": "unauthorized"})
        ):
            result = json.loads(await _pin_fn()(artifact_id=entry.id))
        assert result["status"] == "success"
        assert result["gallery_notified"] is False
        stored = store.get_entry(entry.id)
        assert stored.pinned is True
        assert stored.metadata["pinned_in"] == web_session.transcript

    @pytest.mark.asyncio
    async def test_unpin_clears_record_and_posts_transcript(
        self, store, entry, web_session, _gallery_url
    ):
        with patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})):
            await _pin_fn()(artifact_id=entry.id)
        with patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})) as post:
            result = json.loads(await _pin_fn()(artifact_id=entry.id, pinned=False))
        assert result["pinned"] is False
        assert "pinned_in" not in store.get_entry(entry.id).metadata
        assert _posted(post) == {"pinned": False, "transcript_id": web_session.transcript}

    @pytest.mark.asyncio
    async def test_pin_scope_resolved_at_store_root(self, entry, cli_session, _gallery_url):
        """The pointer is read under the store's workspace root, never a default root."""
        with (
            patch(f"{_MODULE}._post_json_with_response", return_value=(200, {})),
            patch("osprey.stores.artifact_scope.current_scope") as scope,
        ):
            scope.return_value.transcript = None
            await _pin_fn()(artifact_id=entry.id)
        scope.assert_called_once()
        assert scope.call_args.kwargs["root"] == cli_session.root


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

    def test_docstring_does_not_claim_the_tool_writes_focus_state(self):
        from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

        doc = (getattr(artifact_focus, "fn", artifact_focus).__doc__) or ""
        assert "updates focus_state.txt" not in " ".join(doc.split())
