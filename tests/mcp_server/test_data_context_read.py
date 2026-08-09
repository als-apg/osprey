"""Tests for the data_read MCP tool.

Covers:
  - Reading a valid entry returns full JSON content (raw data, no envelope)
  - Missing entry returns not_found error
  - Missing data file returns file_not_found error
  - Oversized file returns file_too_large error
"""

import json

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


@pytest.fixture
def store(tmp_path):
    """Initialize an ArtifactStore in a temporary workspace."""
    return initialize_artifact_store(workspace_root=tmp_path)


@pytest.fixture
def read_tool():
    """Get the raw async function for data_read."""
    from osprey.mcp_server.workspace.tools.data_context_tools import data_read

    return get_tool_fn(data_read)


class TestDataRead:
    """Tests for data_read."""

    @pytest.mark.asyncio
    async def test_read_valid_entry(self, store, read_tool):
        entry = _save_entry(store)

        result = extract_response_dict(await read_tool(entry_id=entry.id))

        assert result["value"] == 42
        assert result["units"] == "mA"

    @pytest.mark.asyncio
    async def test_read_missing_entry(self, store, read_tool):
        with assert_raises_error(error_type="not_found") as _exc_ctx:
            await read_tool(entry_id="nonexistent_id")
        result = _exc_ctx["envelope"]
        assert "nonexistent_id" in result["error_message"]

    @pytest.mark.asyncio
    async def test_read_missing_data_file(self, store, read_tool):
        entry = _save_entry(store)

        # Delete the data file from disk
        store.get_file_path(entry.id).unlink()

        with assert_raises_error(error_type="file_not_found"):
            await read_tool(entry_id=entry.id)

    @pytest.mark.asyncio
    async def test_read_oversized_file(self, store, read_tool):
        """Files over the 100 KB inline cap raise file_too_large."""
        entry = _save_entry(store)

        # Overwrite data file with content above the 100 KB inline cap
        store.get_file_path(entry.id).write_text("x" * (200 * 1024))

        with assert_raises_error(error_type="file_too_large") as ctx:
            await read_tool(entry_id=entry.id)
        envelope = ctx["envelope"]
        details = envelope["details"]
        assert details["size_bytes"] == 200 * 1024
        assert details["limit_bytes"] == 100 * 1024
        assert details["entry_id"] == entry.id
        assert details["file_path"].endswith(entry.data_file)

    @pytest.mark.asyncio
    async def test_read_just_under_cap_works(self, store, read_tool):
        """Files just under the cap are still returned inline."""
        entry = _save_entry(store)
        # 50 KB of text — well under the 100 KB cap
        payload = "y" * (50 * 1024)
        store.get_file_path(entry.id).write_text(payload)

        content = await read_tool(entry_id=entry.id)
        assert content == payload

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

        with assert_raises_error(error_type="file_too_large") as ctx:
            await read_tool(entry_id=entry.id)
        preview = ctx["envelope"]["details"]["preview"]
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

        with assert_raises_error(error_type="file_too_large") as ctx:
            await read_tool(entry_id=entry.id)
        preview = ctx["envelope"]["details"]["preview"]
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

        with assert_raises_error(error_type="file_too_large") as ctx:
            await read_tool(entry_id=entry.id)
        preview = ctx["envelope"]["details"]["preview"]
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

        with assert_raises_error(error_type="file_too_large") as ctx:
            await read_tool(entry_id=entry.id)
        preview = ctx["envelope"]["details"]["preview"]
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

        with assert_raises_error(error_type="file_too_large") as ctx:
            await read_tool(entry_id=entry.id)
        preview = ctx["envelope"]["details"]["preview"]
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

        with assert_raises_error(error_type="file_too_large") as ctx:
            await read_tool(entry_id=entry.id)
        preview = ctx["envelope"]["details"]["preview"]
        assert {k: preview.get(k) for k in expected} == expected
        assert "head" not in preview

    @pytest.mark.asyncio
    async def test_read_wraps_unexpected_store_failure_as_internal_error(
        self, store, read_tool, monkeypatch
    ):
        """An unexpected store exception surfaces as internal_error, not a raw traceback."""

        def _boom(entry_id):
            raise RuntimeError("index corrupted")

        monkeypatch.setattr(store, "get_entry", _boom)

        with assert_raises_error(error_type="internal_error") as ctx:
            await read_tool(entry_id="whatever")
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

        with assert_raises_error(error_type="file_too_large") as ctx:
            await read_tool(entry_id=entry.id)
        preview = ctx["envelope"]["details"]["preview"]
        assert preview["shape"] == "text"
        assert preview["head"].startswith("HEAD_MARKER_LINE")
        assert preview["tail"].endswith("TAIL_MARKER_LINE\n")

    @pytest.mark.asyncio
    async def test_oversize_error_includes_actionable_suggestions(self, store, read_tool):
        """The over-cap error envelope points the agent at execute / data_source."""
        entry = _save_entry(store)
        store.get_file_path(entry.id).write_text("x" * (200 * 1024))

        with assert_raises_error(error_type="file_too_large") as ctx:
            await read_tool(entry_id=entry.id)
        suggestions = ctx["envelope"]["suggestions"]
        joined = "\n".join(suggestions)
        assert "execute" in joined
        assert "data_source" in joined

    @pytest.mark.asyncio
    async def test_read_returns_raw_json_string(self, store, read_tool):
        """data_read returns the file content as-is (raw JSON, no envelope)."""
        entry = _save_entry(store, data={"channels": ["SR:C01:CURRENT"]})

        raw = await read_tool(entry_id=entry.id)
        parsed = json.loads(raw)

        # The raw content is the data itself — no envelope
        assert "channels" in parsed

    @pytest.mark.asyncio
    async def test_read_multiple_entries(self, store, read_tool):
        """Each entry ID returns its own data."""
        e1 = _save_entry(store, data={"sensor": "A"})
        e2 = _save_entry(store, data={"sensor": "B"})

        r1 = json.loads(await read_tool(entry_id=e1.id))
        r2 = json.loads(await read_tool(entry_id=e2.id))

        assert r1["sensor"] == "A"
        assert r2["sensor"] == "B"
