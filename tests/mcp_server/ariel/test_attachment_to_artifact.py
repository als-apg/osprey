"""The ``attachment_to_artifact`` MCP tool: keep a stored logbook picture in the gallery.

Runs against the keyset harness (one entry with a viewable PNG and a skipped
PDF) and a real artifact store in ``tmp_path``. The gallery is never reached:
its focus endpoint is captured at ``_post_json_with_response``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.stores import artifact_store as artifact_store_module
from osprey.stores.artifact_store import (
    AUDIT_ORIGIN,
    ArtifactStore,
    register_artifact_listener,
    unregister_artifact_listener,
)
from tests.mcp_server.ariel.conftest import (
    KEYSET_ENTRY_ID,
    KEYSET_RENDITION_BYTES,
    get_tool_fn,
    keyset_attachment_rows,
    keyset_entries,
)
from tests.mcp_server.conftest import assert_raises_error

_PNG, _PDF = keyset_entries()[0]["attachments"]
PNG_ID = attachment_id_for(KEYSET_ENTRY_ID, _PNG)
PDF_ID = attachment_id_for(KEYSET_ENTRY_ID, _PDF)


@pytest.fixture
def store(tmp_path, monkeypatch) -> ArtifactStore:
    store = ArtifactStore(workspace_root=tmp_path / "agent_data", auto_launch=False)
    monkeypatch.setattr(artifact_store_module, "_artifact_store", store)
    return store


@pytest.fixture
def gallery():
    """The gallery's focus endpoint, answering 200."""
    with patch("osprey.mcp_server.http._post_json_with_response", return_value=(200, {})) as focus:
        yield focus


async def _keep(attachment_id: str = PNG_ID) -> dict:
    from osprey.mcp_server.ariel.tools.attachment import attachment_to_artifact

    return json.loads(await get_tool_fn(attachment_to_artifact)(attachment_id=attachment_id))


async def test_keeps_the_rendition_as_an_image_artifact(keyset_harness, store, gallery):  # noqa: ARG001
    result = await _keep()

    entry = store.get_entry(result["artifact_id"])
    assert entry is not None
    assert Path(store.get_file_path(entry.id)).read_bytes() == KEYSET_RENDITION_BYTES
    assert entry.artifact_type == "image"
    assert entry.mime_type == "image/png"
    assert entry.category == "visualization"
    assert entry.tool_source == "attachment_to_artifact"
    assert entry.origin == ""
    assert entry.title == f"orbit-plot.png (entry {KEYSET_ENTRY_ID})"
    assert entry.filename.endswith("_orbit-plot.png")
    assert entry.metadata["sha256"] == hashlib.sha256(KEYSET_RENDITION_BYTES).hexdigest()
    assert entry.metadata["logbook_picture"] == {
        "entry_id": KEYSET_ENTRY_ID,
        "attachment_id": PNG_ID,
        "filename": "orbit-plot.png",
    }
    assert KEYSET_ENTRY_ID in entry.description
    assert PNG_ID in entry.description
    assert "orbit-plot.png" in entry.description

    assert result["status"] == "success"
    assert result["entry_id"] == KEYSET_ENTRY_ID
    assert result["attachment_id"] == PNG_ID
    assert result["created"] is True
    assert result["focused"] is True
    assert result["gallery_url"]
    url, payload = gallery.call_args.args
    assert url.endswith("/api/focus")
    assert payload == {"artifact_id": result["artifact_id"]}


async def test_is_a_gallery_artifact_not_an_audit_spill(keyset_harness, store, gallery):  # noqa: ARG001
    seen = []
    register_artifact_listener(seen.append)
    try:
        result = await _keep()
    finally:
        unregister_artifact_listener(seen.append)

    assert [e.id for e in store.list_entries()] == [result["artifact_id"]]
    assert all(e.origin != AUDIT_ORIGIN for e in store.list_entries())
    assert [e.id for e in seen] == [result["artifact_id"]]


async def test_the_same_picture_twice_is_one_artifact(keyset_harness, store, gallery):  # noqa: ARG001
    first = await _keep()
    seen = []
    register_artifact_listener(seen.append)
    try:
        second = await _keep()
    finally:
        unregister_artifact_listener(seen.append)

    assert second["artifact_id"] == first["artifact_id"]
    assert second["created"] is False
    assert second["focused"] is True
    assert len(store.list_entries()) == 1
    assert seen == []
    assert gallery.call_count == 2


async def test_caption_and_its_source_are_kept(keyset_harness, store, gallery, monkeypatch):  # noqa: ARG001
    captioned = keyset_entries()[0]
    captioned["attachments"][0]["caption"] = "Horizontal orbit drift, sector 9."

    async def get_entry(entry_id):
        return captioned if entry_id == KEYSET_ENTRY_ID else None

    monkeypatch.setattr(keyset_harness.repository, "get_entry", get_entry)
    result = await _keep()

    entry = store.get_entry(result["artifact_id"])
    provenance = entry.metadata["logbook_picture"]
    assert provenance["caption"] == "Horizontal orbit drift, sector 9."
    assert provenance["caption_source"]
    assert "Horizontal orbit drift, sector 9." in entry.description
    assert provenance["caption_source"] in entry.description


async def test_an_unreachable_gallery_still_keeps_the_picture(keyset_harness, store):  # noqa: ARG001
    with patch("osprey.mcp_server.http._post_json_with_response", side_effect=OSError("refused")):
        result = await _keep()

    assert result["focused"] is False
    assert store.get_entry(result["artifact_id"]) is not None


async def test_a_picture_that_is_not_viewable_is_refused(keyset_harness, store, gallery):  # noqa: ARG001
    pdf_row = next(r for r in keyset_attachment_rows() if r["attachment_id"] == PDF_ID)

    async def stored_row(_repository, attachment_id):
        return pdf_row if attachment_id == PDF_ID else None

    with (
        patch("osprey.mcp_server.ariel.tools.attachment._read_stored_row", new=stored_row),
        assert_raises_error(error_type="no_results") as ctx,
    ):
        await _keep(PDF_ID)
    assert "copy_status=skipped" in ctx["envelope"]["error_message"]
    assert store.list_entries() == []
    gallery.assert_not_called()


async def test_an_unknown_picture_is_not_found(keyset_harness, store, gallery):  # noqa: ARG001
    with assert_raises_error(error_type="not_found"):
        await _keep("att-" + "a" * 24)
    assert store.list_entries() == []


@pytest.mark.parametrize("bad_id", ["", "nope", "att-xyz", "../etc/passwd"])
async def test_malformed_id_is_validation_error_without_echo(keyset_harness, store, bad_id):  # noqa: ARG001
    with assert_raises_error(error_type="validation_error") as ctx:
        await _keep(bad_id)
    if bad_id:
        assert bad_id not in json.dumps(ctx["envelope"])


@pytest.mark.parametrize("keyset_harness", [{"view_enabled": False}], indirect=True)
async def test_view_off_is_refused_naming_the_key(keyset_harness, store):  # noqa: ARG001
    from osprey.ariel_attachment_view import VIEW_ENABLED_KEY

    with assert_raises_error(error_type="not_supported") as ctx:
        await _keep()
    assert ctx["envelope"]["details"]["key"] == VIEW_ENABLED_KEY
    assert store.list_entries() == []


@pytest.mark.parametrize(
    ("filename", "mime", "expected"),
    [
        ("orbit plot.png", "image/png", "orbit_plot.png"),
        ("scan.tiff", "image/jpeg", "scan.jpg"),
        ("../../etc/passwd", "image/png", "passwd.png"),
        ("C:\\\\shots\\\\beam.bmp", "image/png", "beam.png"),
        (None, "image/webp", "picture.webp"),
    ],
)
def test_artifact_filename_is_the_stem_with_the_rendition_extension(filename, mime, expected):
    from osprey.mcp_server.ariel.tools.attachment import _artifact_filename

    assert _artifact_filename(filename, mime) == expected


def test_registered_gated_and_offered_to_the_main_agent():
    from osprey.cli.templates.claude_code import (
        _ARIEL_TOOLS_THE_MAIN_AGENT_MAY_CALL,
        _ariel_read_tools,
    )
    from osprey.mcp_server.ariel.server import VIEW_GATED_TOOLS
    from osprey.registry.mcp import ARIEL_VIEW_TOOLS, FRAMEWORK_SERVERS

    assert "attachment_to_artifact" in FRAMEWORK_SERVERS["ariel"].permissions_allow
    assert "attachment_to_artifact" not in FRAMEWORK_SERVERS["ariel"].permissions_ask
    assert "attachment_to_artifact" in _ARIEL_TOOLS_THE_MAIN_AGENT_MAY_CALL
    assert "attachment_to_artifact" not in _ariel_read_tools(True)
    assert set(ARIEL_VIEW_TOOLS) == VIEW_GATED_TOOLS
