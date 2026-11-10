"""Tests for the artifact_read MCP tool: the one by-id read.

It always answers metadata and the absolute file path; ``content`` only when the
file is within the 100 KB cap and decodes as text (a JSON file parsed). A small
binary or an over-cap file is a success with ``content_omitted``; a PNG, JPEG, GIF
or WebP within its own cap comes back as image content. An unknown id
is ``not_found``; a file missing on disk is ``file_not_found``.
"""

import base64
import json
from pathlib import Path

import pytest

from osprey.stores.artifact_store import initialize_artifact_store
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn


def _save_entry(store, tool="channel_read", category="channel_values", data=None):
    """Helper to save a data entry with minimal boilerplate."""
    return store.save_data(
        tool=tool,
        data=data or {"value": 42, "units": "mA"},
        title="test entry",
        description="test entry",
        summary={"count": 1},
        access_details={"format": "json"},
        category=category,
    )


async def _read(read_tool, artifact_id):
    """Call artifact_read and decode its JSON answer."""
    return extract_response_dict(await read_tool(artifact_id=artifact_id))


@pytest.fixture
def store(tmp_path):
    """Initialize an ArtifactStore in a temporary workspace."""
    return initialize_artifact_store(workspace_root=tmp_path)


@pytest.fixture
def read_tool():
    """Get the raw async function for artifact_read."""
    from osprey.mcp_server.workspace.tools.artifact_query import artifact_read

    return get_tool_fn(artifact_read)


class TestDataRead:
    """Tests for artifact_read."""

    @pytest.mark.asyncio
    async def test_read_valid_entry(self, store, read_tool):
        entry = _save_entry(store)

        result = await _read(read_tool, entry.id)

        assert result["content"] == {"value": 42, "units": "mA"}
        assert result["artifact_id"] == entry.id
        assert result["title"] == "test entry"
        assert "content_omitted" not in result

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("store")
    async def test_read_missing_entry(self, read_tool):
        with assert_raises_error(error_type="not_found") as _exc_ctx:
            await read_tool(artifact_id="nonexistent_id")
        result = _exc_ctx["envelope"]
        assert "nonexistent_id" in result["error_message"]

    @pytest.mark.asyncio
    async def test_read_missing_data_file(self, store, read_tool):
        entry = _save_entry(store)

        # Delete the data file from disk
        store.get_file_path(entry.id).unlink()

        with assert_raises_error(error_type="file_not_found") as ctx:
            await read_tool(artifact_id=entry.id)
        details = ctx["envelope"]["details"]
        assert details["artifact_id"] == entry.id
        assert details["title"] == "test entry"
        assert details["file_path"].endswith(entry.data_file)

    @pytest.mark.asyncio
    async def test_read_oversized_file(self, store, read_tool):
        """A large text file is a success: metadata, path, preview, no content (req 10)."""
        entry = _save_entry(store)

        # Overwrite data file with content above the 100 KB inline cap
        store.get_file_path(entry.id).write_text("x" * (200 * 1024))

        result = await _read(read_tool, entry.id)
        assert result["content_omitted"] == "too_large"
        assert "content" not in result
        assert result["size_bytes"] == 200 * 1024
        assert result["limit_bytes"] == 100 * 1024
        assert result["artifact_id"] == entry.id
        assert result["file_path"].endswith(entry.data_file)
        assert result["preview"]["shape"] == "text"

    @pytest.mark.asyncio
    async def test_read_just_under_cap_works(self, store, read_tool):
        """Files just under the cap are still returned inline."""
        entry = _save_entry(store)
        # 50 KB of text — well under the 100 KB cap
        payload = "y" * (50 * 1024)
        store.get_file_path(entry.id).write_text(payload)

        result = await _read(read_tool, entry.id)
        assert result["content"] == payload

    @pytest.mark.asyncio
    async def test_oversize_dataframe_split_preview(self, store, read_tool):
        """Archiver-style split-orient dataframes get a structured peek."""
        entry = _save_entry(store)
        # 200 rows × 3 columns of large floats → comfortably over 100 KB
        # once JSON-encoded with timestamps. Pad timestamps to push over the
        # cap deterministically.
        index = [f"2026-05-11T00:{i // 60:02d}:{i % 60:02d}Z" for i in range(200)]
        columns = ["CH_A", "CH_B", "CH_C"]
        rows = [[float(i), float(i) * 2, float(i) * 3] for i in range(200)]
        payload = {
            "query": {"channels": columns, "padding": "z" * (110 * 1024)},
            "dataframe": {"index": index, "columns": columns, "data": rows},
        }
        store.get_file_path(entry.id).write_text(json.dumps(payload))

        preview = (await _read(read_tool, entry.id))["preview"]
        assert preview["shape"] == "dataframe_split"
        assert preview["columns"] == columns
        assert preview["row_count"] == 200
        assert len(preview["first_rows"]) == 5
        assert len(preview["last_rows"]) == 5
        # First row should align with row 0 of the data
        assert preview["first_rows"][0]["values"] == [0.0, 0.0, 0.0]
        # Last row should be row 199
        assert preview["last_rows"][-1]["values"] == [199.0, 398.0, 597.0]
        # First and last rows must not overlap on a 200-row frame
        first_indices = {row["index"] for row in preview["first_rows"]}
        last_indices = {row["index"] for row in preview["last_rows"]}
        assert first_indices.isdisjoint(last_indices)

    @pytest.mark.asyncio
    async def test_oversize_series_preview(self, store, read_tool):
        """The long-format archiver payload gets channel names and counts, not just keys."""
        entry = _save_entry(store)
        stamps = [f"2026-05-11T00:{i // 60:02d}:{i % 60:02d}+00:00" for i in range(200)]
        payload = {
            "query": {"channels": ["CH_A", "CH_B", "CH_DOWN"], "padding": "z" * (110 * 1024)},
            "series": {
                "CH_A": {"timestamps": stamps, "values": [float(i) for i in range(200)]},
                "CH_B": {"timestamps": stamps[:50], "values": [float(i) * 2 for i in range(50)]},
                # archiver_read always emits requested channels, even with no data in range.
                "CH_DOWN": {"timestamps": [], "values": []},
            },
        }
        store.get_file_path(entry.id).write_text(json.dumps(payload))

        preview = (await _read(read_tool, entry.id))["preview"]
        assert preview["shape"] == "timeseries_series"
        assert preview["channels"] == ["CH_A", "CH_B", "CH_DOWN"]
        assert preview["row_count"] == 250
        assert preview["per_channel"]["CH_A"]["points"] == 200
        assert preview["per_channel"]["CH_A"]["first"] == [stamps[0], 0.0]
        assert preview["per_channel"]["CH_A"]["last"] == [stamps[199], 199.0]
        assert preview["per_channel"]["CH_B"]["points"] == 50
        assert preview["per_channel"]["CH_DOWN"]["points"] == 0
        assert preview["per_channel"]["CH_DOWN"]["first"] is None
        assert preview["per_channel"]["CH_DOWN"]["last"] is None

    @pytest.mark.asyncio
    async def test_oversize_series_preview_under_legacy_metadata_envelope(self, store, read_tool):
        """An artifact wrapped in the legacy `_osprey_metadata` envelope still previews.

        Nothing writes that envelope today, but files already on disk carry it.
        """
        entry = _save_entry(store)
        stamps = [f"2026-05-11T00:{i // 60:02d}:{i % 60:02d}+00:00" for i in range(200)]
        payload = {
            "_osprey_metadata": {"tool": "archiver_read"},
            "data": {
                "query": {"channels": ["CH_A"], "padding": "z" * (110 * 1024)},
                "series": {"CH_A": {"timestamps": stamps, "values": list(range(200))}},
            },
        }
        store.get_file_path(entry.id).write_text(json.dumps(payload))

        preview = (await _read(read_tool, entry.id))["preview"]
        assert preview["shape"] == "timeseries_series"
        assert preview["channels"] == ["CH_A"]
        assert preview["row_count"] == 200

    @pytest.mark.parametrize(
        ("payload", "expected_keys"),
        [
            # Plain non-dataframe object: the baseline generic preview.
            pytest.param(
                {"alpha": "a" * (60 * 1024), "beta": "b" * (60 * 1024), "gamma": [1, 2, 3]},
                {"alpha", "beta", "gamma"},
                id="plain_json_object",
            ),
            # A top-level "data" key WITHOUT the `_osprey_metadata` marker is
            # not an envelope and must not be unwrapped.
            pytest.param(
                {"data": {"anything": "z" * (150 * 1024)}, "other": 1},
                {"data", "other"},
                id="plain_data_key_not_unwrapped",
            ),
            # Dict-of-dicts under "series" but no 'timestamps' -- not an
            # archiver payload despite the key name.
            pytest.param(
                {"series": {"a": {"x": 1}, "b": {"x": 2}}, "padding": "z" * (150 * 1024)},
                {"series", "padding"},
                id="series_key_collision",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_oversize_json_object_preview(
        self, store, read_tool, payload: dict, expected_keys: set[str]
    ):
        """Objects with no recognized shape fall through to the generic preview."""
        entry = _save_entry(store)
        store.get_file_path(entry.id).write_text(json.dumps(payload))

        preview = (await _read(read_tool, entry.id))["preview"]
        assert preview["shape"] == "json_object"
        assert set(preview["top_level_keys"]) == expected_keys

    @pytest.mark.asyncio
    async def test_oversize_series_preview_without_query_block(self, store, read_tool):
        """A series payload with no query dict still previews — just without `query`."""
        entry = _save_entry(store)
        stamps = [f"2026-05-11T00:{i // 60:02d}:{i % 60:02d}+00:00" for i in range(200)]
        payload = {
            "series": {"CH_A": {"timestamps": stamps, "values": [float(i) for i in range(200)]}},
            "padding": "z" * (110 * 1024),
        }
        store.get_file_path(entry.id).write_text(json.dumps(payload))

        preview = (await _read(read_tool, entry.id))["preview"]
        assert preview["shape"] == "timeseries_series"
        assert preview["channels"] == ["CH_A"]
        assert "query" not in preview

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            # A top-level JSON array previews its length and first items.
            pytest.param(
                json.dumps(list(range(40000))).encode(),
                {"shape": "json_array", "length": 40000, "first_items": [0, 1, 2, 3, 4]},
                id="json_array",
            ),
            # A bare JSON scalar previews at most 200 characters of its value.
            pytest.param(
                json.dumps("z" * (150 * 1024)).encode(),
                {"shape": "json_scalar", "value_preview": "z" * 200},
                id="json_scalar",
            ),
            # An undecodable (non-UTF-8) file is reported as binary.
            pytest.param(b"\xff\xfe" * (60 * 1024), {"shape": "binary"}, id="binary"),
            # Above the 16 MB parse ceiling, no preview parsing is attempted at all.
            pytest.param(
                b"x" * (17 * 1024 * 1024),
                {"shape": "skipped_too_large_for_preview"},
                id="beyond_parse_limit",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_oversize_preview_shapes(self, store, read_tool, content, expected):
        """Non-object payloads preview their shape (and peek), never a text head/tail."""
        entry = _save_entry(store)
        store.get_file_path(entry.id).write_bytes(content)

        preview = (await _read(read_tool, entry.id))["preview"]
        assert {k: preview.get(k) for k in expected} == expected
        assert "head" not in preview

    @pytest.mark.asyncio
    async def test_read_wraps_unexpected_store_failure_as_internal_error(
        self, store, read_tool, monkeypatch
    ):
        """An unexpected store exception surfaces as internal_error, not a raw traceback."""

        def _boom(_artifact_id):
            raise RuntimeError("index corrupted")

        monkeypatch.setattr(store, "get_entry", _boom)

        with assert_raises_error(error_type="internal_error") as ctx:
            await read_tool(artifact_id="whatever")
        assert "index corrupted" in ctx["envelope"]["error_message"]

    @pytest.mark.asyncio
    async def test_oversize_text_preview(self, store, read_tool):
        """Non-JSON oversized files expose a text head/tail preview."""
        entry = _save_entry(store)
        # Mostly filler with a recognizable head and tail for the assertion
        head = "HEAD_MARKER_LINE\n"
        tail = "TAIL_MARKER_LINE\n"
        filler = "x" * (150 * 1024)
        store.get_file_path(entry.id).write_text(head + filler + tail)

        preview = (await _read(read_tool, entry.id))["preview"]
        assert preview["shape"] == "text"
        assert preview["head"].startswith("HEAD_MARKER_LINE")
        assert preview["tail"].endswith("TAIL_MARKER_LINE\n")

    @pytest.mark.asyncio
    async def test_oversize_error_includes_actionable_suggestions(self, store, read_tool):
        """The over-cap answer points the agent at execute / data_source."""
        entry = _save_entry(store)
        store.get_file_path(entry.id).write_text("x" * (200 * 1024))

        suggestions = (await _read(read_tool, entry.id))["suggestions"]
        joined = "\n".join(suggestions)
        assert "execute" in joined
        assert "data_source" in joined

    @pytest.mark.asyncio
    async def test_json_content_is_a_value_not_a_string(self, store, read_tool):
        """A JSON artifact's content is its parsed value."""
        entry = _save_entry(store, data={"channels": ["SR:C01:CURRENT"]})

        content = (await _read(read_tool, entry.id))["content"]

        assert content == {"channels": ["SR:C01:CURRENT"]}

    @pytest.mark.asyncio
    async def test_read_multiple_entries(self, store, read_tool):
        """Each entry ID returns its own data."""
        e1 = _save_entry(store, data={"sensor": "A"})
        e2 = _save_entry(store, data={"sensor": "B"})

        r1 = (await _read(read_tool, e1.id))["content"]
        r2 = (await _read(read_tool, e2.id))["content"]

        assert r1["sensor"] == "A"
        assert r2["sensor"] == "B"


def _save_file(store, content: bytes, filename: str, metadata=None, **kwargs):
    """Save an arbitrary file artifact."""
    return store.save_file(
        file_content=content,
        filename=filename,
        artifact_type=kwargs.pop("artifact_type", "file"),
        title=kwargs.pop("title", "a file"),
        description="a file",
        mime_type=kwargs.pop("mime_type", "application/octet-stream"),
        tool_source="test",
        metadata=metadata or {},
        **kwargs,
    )


class TestReadAbsorbsGet:
    """artifact_read answers metadata and path, plus small content (req 10)."""

    @pytest.mark.asyncio
    async def test_small_text_inline_with_metadata_and_path(self, store, read_tool):
        entry = _save_file(store, b"hello\nworld\n", "notes.txt", mime_type="text/plain")

        result = await _read(read_tool, entry.id)

        assert result["content"] == "hello\nworld\n"
        for key in (
            "artifact_id",
            "title",
            "description",
            "artifact_type",
            "mime_type",
            "size_bytes",
            "timestamp",
            "form",
            "file_path",
            "gallery_url",
        ):
            assert key in result, key
        assert Path(result["file_path"]).is_absolute()
        assert Path(result["file_path"]).read_bytes() == b"hello\nworld\n"

    @pytest.mark.asyncio
    async def test_small_binary_is_a_success_without_content(self, store, read_tool):
        blob = b"\x00\xff" + bytes(range(256)) * 4
        entry = _save_file(store, blob, "dump.bin")

        result = await _read(read_tool, entry.id)

        assert result["content_omitted"] == "binary"
        assert "content" not in result
        assert result["file_path"].endswith(".bin")


_PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4


class TestReadImage:
    """An image artifact comes back as image content, so the model sees the figure."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("filename", "mime"),
        [
            ("plot.png", "image/png"),
            ("photo.jpg", "image/jpeg"),
            ("anim.gif", "image/gif"),
            ("shot.webp", "image/webp"),
        ],
    )
    async def test_an_image_is_returned_as_image_content_beside_its_metadata(
        self, store, read_tool, filename, mime
    ):
        entry = _save_file(store, _PNG, filename, artifact_type="image", mime_type=mime)

        result = await read_tool(artifact_id=entry.id)

        text, image = result.content
        meta = json.loads(text.text)
        assert meta["artifact_id"] == entry.id
        assert meta["mime_type"] == mime
        assert "content_omitted" not in meta
        assert image.type == "image"
        assert image.mimeType == mime
        assert base64.b64decode(image.data) == _PNG

    @pytest.mark.asyncio
    async def test_an_image_past_the_inline_text_cap_is_still_shown(self, store, read_tool):
        big = _PNG + b"\x00" * (200 * 1024)
        entry = _save_file(store, big, "plot.png", artifact_type="image", mime_type="image/png")

        result = await read_tool(artifact_id=entry.id)

        assert base64.b64decode(result.content[1].data) == big

    @pytest.mark.asyncio
    async def test_an_image_past_the_image_cap_is_omitted_as_too_large(
        self, store, read_tool, monkeypatch
    ):
        from osprey.mcp_server.workspace.tools import artifact_query

        monkeypatch.setattr(artifact_query, "_MAX_IMAGE_SIZE", len(_PNG) - 1)
        entry = _save_file(store, _PNG, "plot.png", artifact_type="image", mime_type="image/png")

        result = await _read(read_tool, entry.id)

        assert result["content_omitted"] == "too_large"

    @pytest.mark.asyncio
    async def test_another_image_type_stays_binary(self, store, read_tool):
        tiff = b"II*\x00" + bytes(range(256))
        entry = _save_file(store, tiff, "scan.tiff", artifact_type="image", mime_type="image/tiff")

        result = await _read(read_tool, entry.id)

        assert result["content_omitted"] == "binary"

    @pytest.mark.asyncio
    async def test_large_text_is_a_success_with_preview_and_suggestions(self, store, read_tool):
        entry = _save_file(store, b"HEAD\n" + b"x" * (150 * 1024), "big.txt")

        result = await _read(read_tool, entry.id)

        assert result["content_omitted"] == "too_large"
        assert "content" not in result
        assert result["preview"]["head"].startswith("HEAD")
        assert any("execute" in s for s in result["suggestions"])
        assert result["file_path"]

    @pytest.mark.asyncio
    async def test_logbook_picture_is_projected(self, store, read_tool):
        picture = {"entry_id": "12345", "attachment": "beam.png"}
        entry = _save_file(
            store, b"\x89PNG\r\n\x1a\n\xff", "beam.png", metadata={"logbook_picture": picture}
        )

        result = await _read(read_tool, entry.id)

        assert result["logbook_picture"] == picture

    @pytest.mark.asyncio
    async def test_other_metadata_never_appears(self, store, read_tool):
        entry = _save_file(
            store,
            b"plain",
            "x.txt",
            metadata={"secret_internal_key": "SENTINEL_VALUE", "logbook_picture": "not-a-dict"},
        )

        raw = await read_tool(artifact_id=entry.id)
        result = json.loads(raw)

        assert "secret_internal_key" not in raw
        assert "SENTINEL_VALUE" not in raw
        assert "metadata" not in result
        assert "logbook_picture" not in result

    @pytest.mark.asyncio
    async def test_missing_file_keeps_projection_out_of_details(self, store, read_tool):
        entry = _save_file(store, b"plain", "x.txt", metadata={"secret_internal_key": "SENTINEL"})
        store.get_file_path(entry.id).unlink()

        with assert_raises_error(error_type="file_not_found") as ctx:
            await read_tool(artifact_id=entry.id)

        assert "SENTINEL" not in json.dumps(ctx["envelope"])
        assert ctx["envelope"]["details"]["artifact_id"] == entry.id
