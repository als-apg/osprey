"""Tests for the archiver_read MCP tool.

Covers: time parsing, raw vs processed data, artifact persistence,
timeout handling, and error format compliance.

Note: archiver_read uses the registry to get the archiver connector.
The connector returns the canonical long-format DataFrame (columns
``timestamp``, ``channel``, ``value``), and archiver_read always saves the
per-channel series to the ArtifactStore.
"""

import json
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from osprey.connectors.archiver._timerange import long_frame
from osprey.mcp_server.control_system.server_context import initialize_server_context
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)

# Five candidates so tests can build channels with up to 5 samples each,
# including different lengths per channel (a channel's samples need not line
# up with any other channel's — see test_archiver_read_handles_ragged_channels).
_CANDIDATE_STAMPS = pd.to_datetime(
    [
        "2024-01-15T10:00:00",
        "2024-01-15T10:01:00",
        "2024-01-15T10:02:00",
        "2024-01-15T10:03:00",
        "2024-01-15T10:04:00",
    ],
    utc=True,
)


def _make_archiver_df(channels_data):
    """Build a mock long-format DataFrame via the real connector-shared builder.

    Delegates to :func:`long_frame`, the same builder every real archiver
    connector uses, so this helper cannot drift from the contract it claims to
    mirror (a hand-rolled equivalent is exactly how the old wide-frame helper
    hid a fully broken tool behind twelve green tests). Each channel's values
    are paired with a prefix of the shared candidate timestamps, so callers
    can pass value lists of different lengths per channel.
    """
    series = {
        ch: pd.Series(values, index=_CANDIDATE_STAMPS[: len(values)])
        for ch, values in channels_data.items()
    }
    return long_frame(series)


def _get_archiver_read():
    from osprey.mcp_server.control_system.tools.archiver_read import archiver_read

    return get_tool_fn(archiver_read)


@pytest.mark.unit
async def test_archiver_read_basic(tmp_path, monkeypatch):
    """Basic archiver read returns summary with data file path."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.1, 500.3]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T11:00:00",
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "artifact_id" in data
    assert "data_file" in data
    assert data["summary"]["channels_queried"] == 1
    assert data["summary"]["total_points"] == 2
    assert "SR:CURRENT:RB" in data["summary"]["per_channel"]
    assert data["summary"]["per_channel"]["SR:CURRENT:RB"]["points"] == 2

    # access_details is returned inline so Claude knows how to read the data file
    ad = data["access_details"]
    assert ad["data_file_structure"]["root_keys"] == ["query", "series"]
    assert "SR:CURRENT:RB" in ad["access_patterns"]
    assert ad["total_points"] == 2
    # The schema text is honest about channels not sharing a timestamp axis,
    # documents the query root key too, and doesn't overclaim ISO-8601/UTC
    # for query.start_time/end_time (those are facility-local strings).
    assert "not" in ad["schema"]["series"].lower()
    assert "aligned" in ad["schema"]["series"].lower()
    assert "facility-local" in ad["schema"]["query"].lower()
    # Values can be null (non-numeric/missing samples), and "points" counts
    # those nulls too — an agent must not assume points == usable numbers.
    assert "null" in ad["schema"]["values"].lower()
    assert "points" in ad["schema"]["values"].lower()


@pytest.mark.unit
async def test_archiver_read_relative_time(tmp_path, monkeypatch):
    """Archiver read with relative time strings (e.g., '1h ago')."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB"],
            start_time="1h ago",
            end_time="now",
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"


@pytest.mark.unit
def test_parse_time_interprets_naive_input_as_facility_local(monkeypatch):
    """A naive operator time string is localized to the facility zone, not UTC.

    Guards the contract the agent timezone rule promises (archiver_read times are
    facility-local). A regression to ``tzinfo=UTC`` would query a shifted window.
    """
    from zoneinfo import ZoneInfo

    from osprey.mcp_server.control_system.tools import archiver_read as mod

    tokyo = ZoneInfo("Asia/Tokyo")  # UTC+9, fixed offset (no DST)
    monkeypatch.setattr(mod, "get_facility_timezone", lambda: tokyo)

    parsed = mod._parse_time("2026-06-01 12:00:00")  # naive, no offset
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 9 * 3600  # facility-local, not UTC

    # An explicit offset in the input is respected, not overridden.
    aware = mod._parse_time("2026-06-01T12:00:00+00:00")
    assert aware.utcoffset().total_seconds() == 0


@pytest.mark.unit
async def test_archiver_read_file_persistence(tmp_path, monkeypatch):
    """Archiver read saves data to _agent_data/artifacts/ via ArtifactStore."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = extract_response_dict(result)
    assert "data_file" in data

    # data_file is a project-CWD-relative path (e.g.
    # ``_agent_data/artifacts/{id}_archiver_read.json``) so the agent can
    # pass it directly to open(). Test resolves it from the project root.
    assert data["data_file"].startswith("_agent_data/artifacts/")
    data_file = tmp_path / data["data_file"]
    assert data_file.exists()

    # Verify the data file is raw JSON (no _osprey_metadata envelope)
    file_content = json.loads(data_file.read_text())
    assert "query" in file_content
    assert "series" in file_content
    assert "SR:CURRENT:RB" in file_content["series"]
    assert "timestamps" in file_content["series"]["SR:CURRENT:RB"]
    assert "values" in file_content["series"]["SR:CURRENT:RB"]
    assert "_osprey_metadata" not in file_content

    # Verify the index file was created (inside the artifacts subdir)
    artifacts_dir = tmp_path / "_agent_data" / "artifacts"
    index_file = artifacts_dir / "artifacts.json"
    assert index_file.exists()


@pytest.mark.unit
async def test_archiver_read_multiple_channels(tmp_path, monkeypatch):
    """Multi-channel archiver read returns summary for all channels."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df(
        {
            "SR:CURRENT:RB": [500.0, 500.1],
            "SR:ENERGY:RB": [1.9, 1.9],
        }
    )
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB", "SR:ENERGY:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["channels_queried"] == 2
    assert "SR:CURRENT:RB" in data["summary"]["per_channel"]
    assert "SR:ENERGY:RB" in data["summary"]["per_channel"]


@pytest.mark.unit
async def test_archiver_read_timeout(tmp_path, monkeypatch):
    """Archiver read timeout returns error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_connector = AsyncMock()
    mock_connector.get_data.side_effect = TimeoutError("archiver query timed out")

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        with assert_raises_error(error_type="timeout_error") as _exc_ctx:
            await fn(
                channels=["SR:CURRENT:RB"],
                start_time="2020-01-01",
                end_time="2024-01-01",
            )

    data = _exc_ctx["envelope"]
    assert "error_message" in data
    assert "suggestions" in data


@pytest.mark.unit
async def test_archiver_read_connection_error(tmp_path, monkeypatch):
    """Archiver connection error returns standard error format."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_connector = AsyncMock()
    mock_connector.get_data.side_effect = ConnectionError("archiver unreachable")

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        with assert_raises_error(error_type="connection_error") as _exc_ctx:
            await fn(
                channels=["SR:CURRENT:RB"],
                start_time="2024-01-15T10:00:00",
            )

    data = _exc_ctx["envelope"]
    assert "error_message" in data
    assert "suggestions" in data


@pytest.mark.unit
async def test_archiver_read_nan_channel_data(tmp_path, monkeypatch):
    """A numeric channel whose samples are all NaN reports no numeric stats.

    ``points`` is the real sample count for the channel (the archiver still
    returned two rows — it just recorded NaN in each), not zero: the two
    concepts (how many samples exist vs. whether any are numeric) are
    independent since long_frame never manufactures rows.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df(
        {
            "SR:CURRENT:RB": [500.0, 500.1],
            "SR:MISSING:RB": [float("nan"), float("nan")],
        }
    )
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB", "SR:MISSING:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"

    # The good channel should have real stats
    good = data["summary"]["per_channel"]["SR:CURRENT:RB"]
    assert good["points"] == 2
    assert good["min"] == 500.0
    assert good["max"] == 500.1
    assert good["mean"] is not None

    # The all-NaN channel reports its point count and omits numeric stats
    # entirely — it must not error, and must not report NaN.
    bad = data["summary"]["per_channel"]["SR:MISSING:RB"]
    assert bad["points"] == 2
    assert "min" not in bad
    assert "max" not in bad
    assert "mean" not in bad

    # The result must be fully JSON-serializable without allow_nan
    json.dumps(data, allow_nan=False)  # Would raise if NaN slipped through


@pytest.mark.unit
async def test_archiver_read_enum_channel_omits_numeric_stats(tmp_path, monkeypatch):
    """An enum/status channel (string values) reports point count, no stats.

    Enum/status PVs (e.g. machine mode, interlock state) are archived as
    strings, not numbers. ``pd.to_numeric(..., errors="coerce")`` turns every
    one of them into NaN, so the channel must take the same "no numeric
    samples" path as an all-NaN numeric channel, without ever raising.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df(
        {
            "SR:CURRENT:RB": [500.0, 500.1],
            "MACHINE:MODE": ["Standby", "Injecting"],
        }
    )
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB", "MACHINE:MODE"],
            start_time="2024-01-15T10:00:00",
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"

    enum_summary = data["summary"]["per_channel"]["MACHINE:MODE"]
    assert enum_summary["points"] == 2
    assert "min" not in enum_summary
    assert "max" not in enum_summary
    assert "mean" not in enum_summary

    # The enum values themselves round-trip untouched in the data file.
    data_file = tmp_path / data["data_file"]
    file_content = json.loads(data_file.read_text())
    assert file_content["series"]["MACHINE:MODE"]["values"] == ["Standby", "Injecting"]

    json.dumps(data, allow_nan=False)


@pytest.mark.unit
async def test_archiver_read_empty_channels(tmp_path, monkeypatch):
    """Empty channel list returns validation error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_archiver_read()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(channels=[], start_time="2024-01-15T10:00:00")

    _exc_ctx["envelope"]


@pytest.mark.unit
async def test_archiver_read_passes_processing_to_connector(tmp_path, monkeypatch):
    """The advertised processing mode must reach the connector, not just the echo."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.1, 500.3]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T11:00:00",
            processing="mean",
            bin_size=60,
        )

    kwargs = mock_connector.get_data.call_args.kwargs
    assert kwargs["processing"] == "mean"
    assert kwargs["precision_ms"] == 60_000


@pytest.mark.unit
async def test_archiver_read_rejects_unknown_processing(tmp_path, monkeypatch):
    """An unsupported mode errors with the valid set, rather than silently downgrading."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    fn = _get_archiver_read()
    with assert_raises_error(error_type="validation_error") as ctx:
        await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
            processing="p99",
        )

    assert "mean" in ctx["envelope"]["error_message"]


@pytest.mark.unit
async def test_archiver_read_echoes_facility_local_time_range(tmp_path, monkeypatch):
    """The tool keeps speaking facility-local; only the connector converts to UTC."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    from zoneinfo import ZoneInfo

    monkeypatch.setattr(
        "osprey.mcp_server.control_system.tools.archiver_read.get_facility_timezone",
        lambda: ZoneInfo("America/Los_Angeles"),
    )

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.1, 500.3]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T11:00:00",
        )

    # January in Los Angeles is PST (-08:00).
    data = extract_response_dict(result)
    assert "-08:00" in data["summary"]["time_range"]["start"]

    # And the connector receives that same facility-local instant, not a
    # pre-converted one — converting is the connector's job.
    start_arg = mock_connector.get_data.call_args.args[1]
    assert start_arg.utcoffset().total_seconds() == -8 * 3600


@pytest.mark.unit
async def test_archiver_read_bin_size_zero_is_full_resolution(tmp_path, monkeypatch):
    """bin_size=0 requests full-resolution data: precision_ms=0 reaches the connector.

    ``resolve_processing`` already treats ``precision_ms <= 0`` as full
    resolution on every backend (EPICS sends the bare PV name), so bin_size=0
    is the operator's only way to ask for undecimated raw samples.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.1, 500.3]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
            processing="raw",
            bin_size=0,
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"

    kwargs = mock_connector.get_data.call_args.kwargs
    assert kwargs["precision_ms"] == 0
    assert kwargs["processing"] == "raw"


@pytest.mark.unit
async def test_archiver_read_bin_size_zero_rejects_non_raw_processing(tmp_path, monkeypatch):
    """bin_size=0 (full resolution) has no bin — only valid with processing='raw'."""
    monkeypatch.chdir(tmp_path)

    fn = _get_archiver_read()
    with assert_raises_error(error_type="validation_error") as ctx:
        await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
            processing="mean",
            bin_size=0,
        )

    assert "bin_size" in ctx["envelope"]["error_message"]


@pytest.mark.unit
async def test_archiver_read_rejects_negative_bin_size(tmp_path, monkeypatch):
    """A negative bin_size is nonsensical and must error, not silently misbehave."""
    monkeypatch.chdir(tmp_path)

    fn = _get_archiver_read()
    with assert_raises_error(error_type="validation_error") as ctx:
        await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
            bin_size=-5,
        )

    assert "bin_size" in ctx["envelope"]["error_message"]


@pytest.mark.unit
async def test_archiver_read_handles_ragged_channels(tmp_path, monkeypatch):
    """Channels with different sample counts are never forced onto a shared axis.

    The schema text promises channels are "NOT aligned" and have independent
    timestamps, but every other test's fixture happens to use equal-length
    value lists. This pins the genuinely ragged case the long format exists
    for: one channel with 3 samples, another with 1, neither padded or
    truncated to match the other.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df(
        {
            "SR:CURRENT:RB": [500.0, 500.1, 500.2],
            "SR:ENERGY:RB": [1.9],
        }
    )
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB", "SR:ENERGY:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = extract_response_dict(result)
    assert data["summary"]["per_channel"]["SR:CURRENT:RB"]["points"] == 3
    assert data["summary"]["per_channel"]["SR:ENERGY:RB"]["points"] == 1
    assert data["summary"]["total_points"] == 4

    data_file = tmp_path / data["data_file"]
    file_content = json.loads(data_file.read_text())
    assert len(file_content["series"]["SR:CURRENT:RB"]["timestamps"]) == 3
    assert len(file_content["series"]["SR:ENERGY:RB"]["timestamps"]) == 1


@pytest.mark.unit
async def test_archiver_read_channel_absent_from_result_reports_zero_points(tmp_path, monkeypatch):
    """A requested channel with zero rows in the result reports points=0, not a crash.

    The pre-fix tool silently dropped such a channel from ``per_channel``
    entirely (``if ch in df.columns`` on a long frame never matches a channel
    name). It must now appear honestly with a zero count and no numeric stats.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    # Only SR:CURRENT:RB has any rows; SR:VOID:RB is requested but the
    # archiver returned nothing for it (e.g. out of archival range).
    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB", "SR:VOID:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"

    void = data["summary"]["per_channel"]["SR:VOID:RB"]
    assert void["points"] == 0
    assert "min" not in void
    assert "max" not in void
    assert "mean" not in void
    assert data["summary"]["total_points"] == 2  # only the real channel's points


@pytest.mark.unit
async def test_archiver_read_emits_empty_series_entry_for_absent_channel(tmp_path, monkeypatch):
    """The data file's series entry for a zero-row channel is exactly {timestamps: [], values: []}.

    This is the other end of the contract consumers like ``build_data_reader``
    (``_viz_common.py``) rely on: every *requested* channel gets an entry in
    ``series``, even one with no data in range, and that entry always has the
    ``{"timestamps": [...], "values": [...]}`` shape rather than being omitted
    or shaped some other way. Pinning this exact literal here, alongside the
    reader-side test that consumes it, keeps the two ends of the contract from
    drifting apart independently.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB", "SR:VOID:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = extract_response_dict(result)
    data_file = tmp_path / data["data_file"]
    file_content = json.loads(data_file.read_text())

    assert file_content["series"]["SR:VOID:RB"] == {"timestamps": [], "values": []}


@pytest.mark.unit
async def test_archiver_read_dedupes_repeated_channel_name(tmp_path, monkeypatch):
    """A caller-repeated channel name is queried and summarized once, not double-counted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB", "SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = extract_response_dict(result)
    assert data["summary"]["channels_queried"] == 1
    assert data["summary"]["total_points"] == 2  # not doubled
    assert list(data["summary"]["per_channel"].keys()) == ["SR:CURRENT:RB"]

    # The dedup happens before the connector call too — the archiver is only
    # ever asked for the channel once.
    queried_channels = mock_connector.get_data.call_args.args[0]
    assert queried_channels == ["SR:CURRENT:RB"]


class TestArchiverReadRealMockConnector:
    """End-to-end coverage through the real ``MockArchiverConnector``.

    Every test above patches ``ConnectorFactory.create_archiver_connector``
    with an ``AsyncMock``, so no ``archiver_read`` test ever exercised a real
    connector's ``get_data`` — including its handling of ``precision_ms``.
    That blind spot is exactly how ``bin_size=0`` reaching the default
    ``mock_archiver`` connector's ``ZeroDivisionError`` (``precision_ms=0`` ->
    ``duration / (precision_ms / 1000.0)``) went unnoticed: every fixture
    above also wrote ``archiver:\\n  type: mock\\n``, which isn't even a
    registered archiver name (``mock_archiver`` is) — it was never reachable
    even if a test here had forgotten to patch the factory.
    """

    @pytest.mark.unit
    async def test_bin_size_zero_full_resolution_succeeds(self, tmp_path, monkeypatch):
        """bin_size=0 (precision_ms=0) must not raise ZeroDivisionError on the real connector."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
        initialize_server_context()

        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:DCCT"],
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:05:00",
            processing="raw",
            bin_size=0,
        )

        data = extract_response_dict(result)
        assert data["status"] == "success"
        # Pin the density, not just "some points". The mock's default
        # sample_rate_hz is 1.0, so full resolution over a 300-second window is
        # 300 samples. Asserting only `> 0` lets the whole feature be silently
        # degraded: restoring the old `max(num_points, 10)` floor for
        # precision_ms <= 0 — explicitly "not a fix" — still returns 10 points
        # and still passes a `> 0` check.
        assert data["summary"]["per_channel"]["SR:DCCT"]["points"] == 300

    @pytest.mark.unit
    async def test_bin_size_zero_returns_more_points_than_binned(self, tmp_path, monkeypatch):
        """Full resolution must strictly out-resolve a binned query on the same window.

        The exact-count assertion above pins the mock's own native rate; this
        pins the *relationship* the feature exists for, and survives a change
        to the mock's configured ``sample_rate_hz``.

        The window is an hour rather than five minutes on purpose: it puts the
        binned count (60) above the old ten-point floor, so a regression that
        pins full resolution at 10 points fails this comparison too. Over a
        five-minute window the binned count is 6, and a ten-point floor would
        still satisfy ``full > binned``.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
        initialize_server_context()

        fn = _get_archiver_read()
        window = {
            "channels": ["SR:DCCT"],
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T11:00:00",
            "processing": "raw",
        }
        full = extract_response_dict(await fn(**window, bin_size=0))
        binned = extract_response_dict(await fn(**window, bin_size=60))

        full_points = full["summary"]["per_channel"]["SR:DCCT"]["points"]
        binned_points = binned["summary"]["per_channel"]["SR:DCCT"]["points"]
        assert binned_points > 0
        assert full_points > binned_points

    @pytest.mark.unit
    async def test_normal_bin_size_succeeds(self, tmp_path, monkeypatch):
        """An ordinary positive bin_size runs the real connector's binning path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
        initialize_server_context()

        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:DCCT"],
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T11:00:00",
            processing="raw",
            bin_size=60,
        )

        data = extract_response_dict(result)
        assert data["status"] == "success"
        # One sample per 60-second bin across a one-hour window. As above, an
        # exact count rather than `> 0`: bin_size is the parameter under test,
        # so a result that ignores it must fail here.
        assert data["summary"]["per_channel"]["SR:DCCT"]["points"] == 60

    @pytest.mark.unit
    async def test_processing_mode_succeeds(self, tmp_path, monkeypatch):
        """A non-raw processing mode runs the real connector's aggregate_series path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
        initialize_server_context()

        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:DCCT"],
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T11:00:00",
            processing="mean",
            bin_size=60,
        )

        data = extract_response_dict(result)
        assert data["status"] == "success"
        assert data["summary"]["per_channel"]["SR:DCCT"]["points"] > 0
