"""Tests for the archiver_read MCP tool.

Covers: time parsing, raw vs processed data, artifact persistence,
timeout handling, and error format compliance.

Note: archiver_read uses the registry to get the archiver connector.
The connector returns the canonical long-format DataFrame (columns
``timestamp``, ``channel``, ``value``), and archiver_read always saves the
per-channel series to the ArtifactStore.
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from osprey.connectors.archiver._timerange import long_frame
from osprey.mcp_server.control_system.server_context import initialize_server_context
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)

# Five candidates so channels can have up to 5 samples each, with different
# lengths per channel.
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
    """Build a mock long-format DataFrame via :func:`long_frame`, the builder
    every real connector uses. Each channel's values pair with a prefix of the
    shared candidate timestamps, so lengths may differ per channel.
    """
    series = {
        ch: pd.Series(values, index=_CANDIDATE_STAMPS[: len(values)])
        for ch, values in channels_data.items()
    }
    return long_frame(series)


def _get_archiver_read():
    from osprey.mcp_server.control_system.tools.archiver_read import archiver_read

    return get_tool_fn(archiver_read)


@pytest.fixture
def archiver_project(tmp_path, monkeypatch):
    """A project CWD wired to the mock archiver, with the server context up."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    initialize_server_context()
    return tmp_path


@pytest.fixture
def archiver_read_tool(archiver_project):
    """``(tool, connector)`` with the archiver connector factory patched out.

    Set ``connector.get_data.return_value`` (or ``.side_effect``), then await
    the tool. Tests that need the real ``MockArchiverConnector`` take
    ``archiver_project`` instead.

    The coverage probes carry benign defaults — every channel available,
    metadata archived but boundless — because a bare ``AsyncMock`` would hand
    the empty-channel path awaitable mocks instead of ``ArchiverMetadata`` and
    blow up on the first datetime comparison. Boundless metadata classifies as
    ``coverage_unknown``, the verdict that asserts nothing about a store this
    mock never had. Tests that exercise a specific verdict override them.
    """
    from osprey.connectors.archiver.base import ArchiverMetadata

    connector = AsyncMock()
    connector.check_availability.side_effect = lambda chans: dict.fromkeys(chans, True)
    connector.get_metadata.side_effect = lambda ch: ArchiverMetadata(channel=ch, is_archived=True)
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=connector,
    ):
        yield _get_archiver_read(), connector


@pytest.mark.unit
async def test_archiver_read_basic(archiver_read_tool):
    """Basic archiver read returns summary with data file path."""
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.1, 500.3]})

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
    assert 'json_data["series"]["<channel>"]' in ad["access_pattern"]
    # The schema text must say channels are not aligned, and that
    # query.start_time/end_time are facility-local strings.
    assert "not" in ad["schema"]["series"].lower()
    assert "aligned" in ad["schema"]["series"].lower()
    assert "facility-local" in ad["schema"]["query"].lower()
    # Values can be null, and "points" counts those nulls too.
    assert "null" in ad["schema"]["values"].lower()
    assert "points" in ad["schema"]["values"].lower()


@pytest.mark.unit
async def test_access_details_states_the_read_rule_once(archiver_read_tool):
    """The inline read guidance is stated once, not once per channel.

    ``access_details`` rides on every response, so it must be identical for a
    1-channel and a 20-channel read, and carry nothing ``summary``/``query``
    already carry.
    """
    fn, connector = archiver_read_tool

    async def read(channels):
        connector.get_data.return_value = _make_archiver_df(dict.fromkeys(channels, [500.0, 500.1]))
        result = await fn(channels=channels, start_time="2024-01-15T10:00:00")
        return extract_response_dict(result)["access_details"]

    one = await read(["SR:CH00:RB"])
    twenty = await read([f"SR:CH{i:02d}:RB" for i in range(20)])

    assert one == twenty

    # Nothing that summary/query already report.
    for duplicated in ("total_points", "processing", "bin_size"):
        assert duplicated not in one

    # The access rule survives, stated once with a placeholder.
    pattern = one["access_pattern"]
    assert 'json_data["series"]' in pattern
    assert "timestamps" in pattern and "values" in pattern
    assert "per_channel" in pattern  # says where to find the channel names
    assert set(one["schema"]) == {"query", "series", "timestamps", "values"}


@pytest.mark.unit
async def test_archiver_read_relative_time(archiver_read_tool):
    """Archiver read with relative time strings (e.g., '1h ago')."""
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})

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
@pytest.mark.parametrize(
    ("expression", "expected_delta"),
    [
        ("45s ago", timedelta(seconds=45)),
        ("30m ago", timedelta(minutes=30)),
        ("1h ago", timedelta(hours=1)),
        ("2d ago", timedelta(days=2)),
        ("1w ago", timedelta(weeks=1)),
        ("0.5h ago", timedelta(minutes=30)),  # fractional amount
        ("1e3s ago", timedelta(seconds=1000)),  # exponent notation
        ("-3h ago", timedelta(hours=-3)),  # negative amount: forward in time
        ("  10m   ago", timedelta(minutes=10)),  # outer and inner whitespace
        ("1H AGO", timedelta(hours=1)),  # case-insensitive
    ],
)
def test_parse_time_relative_expressions(expression, expected_delta, monkeypatch):
    """Every supported unit suffix maps to its own timedelta keyword."""
    from datetime import UTC

    from osprey.mcp_server.control_system.tools import archiver_read as mod

    monkeypatch.setattr(mod, "get_facility_timezone", lambda: UTC)

    before = datetime.now(UTC)
    parsed = mod._parse_time(expression)
    after = datetime.now(UTC)

    assert before - expected_delta <= parsed <= after - expected_delta


@pytest.mark.unit
@pytest.mark.parametrize(
    "expression",
    [
        "2 days ago",  # spelled-out unit — "2 day" is not a float
        "1 hour ago",
        "1.5.2h ago",  # recognized unit, unparseable amount
        "abc ago",  # unrecognized suffix
        "5 ago",  # no unit at all
        " ago",  # empty amount — must not index off the end
        "M ago",
    ],
)
def test_parse_time_unrecognized_relative_falls_through_to_dateutil(expression, monkeypatch):
    """A "... ago" string the unit map cannot serve still reaches the dateutil branch.

    dateutil rejects all of these, so ``ParserError`` proves the fall-through
    happened.
    """
    from datetime import UTC

    from dateutil.parser import ParserError

    from osprey.mcp_server.control_system.tools import archiver_read as mod

    monkeypatch.setattr(mod, "get_facility_timezone", lambda: UTC)

    with pytest.raises(ParserError):
        mod._parse_time(expression)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("start_time", {"start_time": "not-a-timestamp"}),
        # A bad end_time is caught after a good start_time.
        ("end_time", {"start_time": "2024-01-15T10:00:00", "end_time": "not-a-timestamp"}),
    ],
)
async def test_archiver_read_unparseable_time(archiver_project, field, kwargs):
    """A time even dateutil cannot parse errors cleanly, naming the failing field."""
    fn = _get_archiver_read()
    with assert_raises_error(error_type="validation_error") as ctx:
        await fn(channels=["SR:CURRENT:RB"], **kwargs)

    assert field in ctx["envelope"]["error_message"]
    assert "not-a-timestamp" in ctx["envelope"]["error_message"]


@pytest.mark.unit
async def test_archiver_read_file_persistence(tmp_path, archiver_read_tool):
    """Archiver read saves data to the agent-data artifacts dir via ArtifactStore."""
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})

    result = await fn(
        channels=["SR:CURRENT:RB"],
        start_time="2024-01-15T10:00:00",
    )

    data = extract_response_dict(result)
    assert "data_file" in data

    # data_file is a repo-root-relative path (e.g.
    # ``var/agent_data/artifacts/{id}_archiver_read.json``) so the agent can
    # pass it directly to open(). Test resolves it from the repo root.
    assert data["data_file"].startswith(f"{DEFAULT_AGENT_DATA_BASE_DIR}/artifacts/")
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
    artifacts_dir = tmp_path / DEFAULT_AGENT_DATA_BASE_DIR / "artifacts"
    index_file = artifacts_dir / "artifacts.json"
    assert index_file.exists()


@pytest.mark.unit
async def test_archiver_read_multiple_channels(archiver_read_tool):
    """Multi-channel archiver read returns summary for all channels."""
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df(
        {
            "SR:CURRENT:RB": [500.0, 500.1],
            "SR:ENERGY:RB": [1.9, 1.9],
        }
    )

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
async def test_archiver_read_follows_requested_channel_order(tmp_path, archiver_read_tool):
    """Every per-channel structure follows the caller's order, not the frame's.

    The long frame is sorted by channel name, so a caller who asks for Z
    before A must still get Z before A back.
    """
    requested = ["SR:ZED:RB", "SR:VOID:RB", "SR:ALPHA:RB", "SR:MID:RB"]
    # Frame order is alphabetical (long_frame sorts) and omits SR:VOID:RB.
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df(
        {
            "SR:ALPHA:RB": [1.0],
            "SR:MID:RB": [2.0, 2.1, 2.2],
            "SR:ZED:RB": [3.0, 3.1],
        }
    )

    result = await fn(channels=requested, start_time="2024-01-15T10:00:00")

    data = extract_response_dict(result)
    assert list(data["summary"]["per_channel"]) == requested

    file_content = json.loads((tmp_path / data["data_file"]).read_text())
    assert file_content["query"]["channels"] == requested
    assert list(file_content["series"]) == requested

    assert file_content["series"]["SR:ZED:RB"]["values"] == [3.0, 3.1]
    assert file_content["series"]["SR:VOID:RB"]["values"] == []
    assert file_content["series"]["SR:ALPHA:RB"]["values"] == [1.0]
    assert file_content["series"]["SR:MID:RB"]["values"] == [2.0, 2.1, 2.2]


@pytest.mark.unit
async def test_archiver_read_timeout(archiver_read_tool):
    """Archiver read timeout returns error."""
    fn, connector = archiver_read_tool
    connector.get_data.side_effect = TimeoutError("archiver query timed out")

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
async def test_archiver_read_connection_error(archiver_read_tool):
    """Archiver connection error returns standard error format."""
    fn, connector = archiver_read_tool
    connector.get_data.side_effect = ConnectionError("archiver unreachable")

    with assert_raises_error(error_type="connection_error") as _exc_ctx:
        await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = _exc_ctx["envelope"]
    assert "error_message" in data
    assert "suggestions" in data


@pytest.mark.unit
async def test_archiver_read_nan_channel_data(archiver_read_tool):
    """A numeric channel whose samples are all NaN reports no numeric stats.

    ``points`` is the real sample count for the channel (the archiver still
    returned two rows — it just recorded NaN in each), not zero.
    """
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df(
        {
            "SR:CURRENT:RB": [500.0, 500.1],
            "SR:MISSING:RB": [float("nan"), float("nan")],
        }
    )

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

    # The all-NaN channel omits numeric stats entirely (not NaN, which breaks JSON)
    bad = data["summary"]["per_channel"]["SR:MISSING:RB"]
    assert bad["points"] == 2
    assert "min" not in bad
    assert "max" not in bad
    assert "mean" not in bad

    # The result must be fully JSON-serializable without allow_nan
    json.dumps(data, allow_nan=False)  # Would raise if NaN slipped through


@pytest.mark.unit
async def test_archiver_read_enum_channel_omits_numeric_stats(tmp_path, archiver_read_tool):
    """An enum/status channel (string values) reports point count, no stats.

    ``pd.to_numeric(..., errors="coerce")`` turns every string into NaN, so the
    channel takes the same "no numeric samples" path as an all-NaN channel.
    """
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df(
        {
            "SR:CURRENT:RB": [500.0, 500.1],
            "MACHINE:MODE": ["Standby", "Injecting"],
        }
    )

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
async def test_archiver_read_null_samples_become_json_null(tmp_path, archiver_read_tool):
    """A null sample is written as JSON ``null`` — never NaN, never dropped.

    Dropping the sample would misreport the channel's real cadence.
    """
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df(
        {
            "SR:GAPPY:RB": [500.0, float("nan"), 500.2],
            "SR:MISSING:RB": [float("nan"), float("nan")],
        }
    )

    result = await fn(
        channels=["SR:GAPPY:RB", "SR:MISSING:RB"],
        start_time="2024-01-15T10:00:00",
    )

    data = extract_response_dict(result)
    file_content = json.loads((tmp_path / data["data_file"]).read_text())
    series = file_content["series"]

    assert series["SR:GAPPY:RB"]["values"] == [500.0, None, 500.2]
    assert series["SR:MISSING:RB"]["values"] == [None, None]
    # The null keeps its slot: a null sample is still a real sample.
    assert len(series["SR:GAPPY:RB"]["timestamps"]) == 3
    # And the file is valid JSON with no NaN literal in it.
    json.dumps(file_content, allow_nan=False)


@pytest.mark.unit
async def test_archiver_read_integer_channel_keeps_integer_values(tmp_path, archiver_read_tool):
    """An integer-valued channel (bucket count, status code) stays integral,
    never rewritten as ``3.0`` through a float representation.
    """
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df({"SR:BUCKET:COUNT": [3, 4, 5]})

    result = await fn(
        channels=["SR:BUCKET:COUNT"],
        start_time="2024-01-15T10:00:00",
    )

    data = extract_response_dict(result)
    file_content = json.loads((tmp_path / data["data_file"]).read_text())
    values = file_content["series"]["SR:BUCKET:COUNT"]["values"]

    assert values == [3, 4, 5]
    assert [type(v) for v in values] == [int, int, int]


@pytest.mark.unit
async def test_archiver_read_enum_channel_keeps_null_gaps_between_strings(
    tmp_path, archiver_read_tool
):
    """A null inside a string-valued channel stays null — it never becomes a state.

    Rendering the gap as ``"nan"``/``"None"`` would report the machine in a
    mode it was never in.
    """
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df(
        {"MACHINE:MODE": ["Standby", None, "Injecting"]}
    )

    result = await fn(
        channels=["MACHINE:MODE"],
        start_time="2024-01-15T10:00:00",
    )

    data = extract_response_dict(result)
    file_content = json.loads((tmp_path / data["data_file"]).read_text())

    assert file_content["series"]["MACHINE:MODE"]["values"] == ["Standby", None, "Injecting"]


@pytest.mark.unit
async def test_archiver_read_empty_channels(archiver_project):
    """Empty channel list returns validation error."""
    fn = _get_archiver_read()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(channels=[], start_time="2024-01-15T10:00:00")

    _exc_ctx["envelope"]


@pytest.mark.unit
async def test_archiver_read_passes_processing_to_connector(archiver_read_tool):
    """The advertised processing mode must reach the connector, not just the echo."""
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.1, 500.3]})

    await fn(
        channels=["SR:CURRENT:RB"],
        start_time="2024-01-15T10:00:00",
        end_time="2024-01-15T11:00:00",
        processing="mean",
        bin_size=60,
    )

    kwargs = connector.get_data.call_args.kwargs
    assert kwargs["processing"] == "mean"
    assert kwargs["precision_ms"] == 60_000


@pytest.mark.unit
async def test_archiver_read_rejects_unknown_processing(archiver_project):
    """An unsupported mode errors with the valid set, rather than silently downgrading."""
    fn = _get_archiver_read()
    with assert_raises_error(error_type="validation_error") as ctx:
        await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
            processing="p99",
        )

    # The message names the bad mode; the suggestions carry the valid set.
    envelope = ctx["envelope"]
    assert "p99" in envelope["error_message"]
    suggestions = " ".join(envelope["suggestions"])
    for mode in ("raw", "mean", "min", "max", "median", "std", "count"):
        assert mode in suggestions


@pytest.mark.unit
async def test_archiver_read_echoes_facility_local_time_range(archiver_read_tool, monkeypatch):
    """The tool keeps speaking facility-local; only the connector converts to UTC."""
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(
        "osprey.mcp_server.control_system.tools.archiver_read.get_facility_timezone",
        lambda: ZoneInfo("America/Los_Angeles"),
    )

    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.1, 500.3]})

    result = await fn(
        channels=["SR:CURRENT:RB"],
        start_time="2024-01-15T10:00:00",
        end_time="2024-01-15T11:00:00",
    )

    # January in Los Angeles is PST (-08:00).
    data = extract_response_dict(result)
    assert "-08:00" in data["summary"]["time_range"]["start"]

    # The connector receives the facility-local instant; converting is its job.
    start_arg = connector.get_data.call_args.args[1]
    assert start_arg.utcoffset().total_seconds() == -8 * 3600


@pytest.mark.unit
async def test_archiver_read_bin_size_zero_is_full_resolution(archiver_read_tool):
    """bin_size=0 requests full-resolution data: precision_ms=0 reaches the connector."""
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.1, 500.3]})

    result = await fn(
        channels=["SR:CURRENT:RB"],
        start_time="2024-01-15T10:00:00",
        processing="raw",
        bin_size=0,
    )

    data = extract_response_dict(result)
    assert data["status"] == "success"

    kwargs = connector.get_data.call_args.kwargs
    assert kwargs["precision_ms"] == 0
    assert kwargs["processing"] == "raw"


@pytest.mark.unit
async def test_archiver_read_bin_size_zero_rejects_non_raw_processing(archiver_project):
    """bin_size=0 (full resolution) has no bin — only valid with processing='raw'."""
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
async def test_archiver_read_rejects_negative_bin_size(archiver_project):
    """A negative bin_size is nonsensical and must error, not silently misbehave."""
    fn = _get_archiver_read()
    with assert_raises_error(error_type="validation_error") as ctx:
        await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
            bin_size=-5,
        )

    assert "bin_size" in ctx["envelope"]["error_message"]


@pytest.mark.unit
async def test_archiver_read_handles_ragged_channels(tmp_path, archiver_read_tool):
    """Channels with different sample counts are never forced onto a shared axis:
    one channel with 3 samples, another with 1, neither padded nor truncated.
    """
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df(
        {
            "SR:CURRENT:RB": [500.0, 500.1, 500.2],
            "SR:ENERGY:RB": [1.9],
        }
    )

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
async def test_archiver_read_channel_absent_from_result_reports_zero_points(archiver_read_tool):
    """A requested channel with zero rows appears with points=0 and no numeric stats."""
    # Only SR:CURRENT:RB has any rows; SR:VOID:RB is requested but the
    # archiver returned nothing for it (e.g. out of archival range).
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})

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
async def test_archiver_read_emits_empty_series_entry_for_absent_channel(
    tmp_path, archiver_read_tool
):
    """The data file's series entry for a zero-row channel is exactly {timestamps: [], values: []}.

    Consumers like ``build_data_reader`` rely on every *requested* channel
    getting an entry in ``series``, even one with no data in range.
    """
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})

    result = await fn(
        channels=["SR:CURRENT:RB", "SR:VOID:RB"],
        start_time="2024-01-15T10:00:00",
    )

    data = extract_response_dict(result)
    data_file = tmp_path / data["data_file"]
    file_content = json.loads(data_file.read_text())

    assert file_content["series"]["SR:VOID:RB"] == {"timestamps": [], "values": []}


@pytest.mark.unit
async def test_archiver_read_dedupes_repeated_channel_name(archiver_read_tool):
    """A caller-repeated channel name is queried and summarized once, not double-counted."""
    fn, connector = archiver_read_tool
    connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})

    result = await fn(
        channels=["SR:CURRENT:RB", "SR:CURRENT:RB"],
        start_time="2024-01-15T10:00:00",
    )

    data = extract_response_dict(result)
    assert data["summary"]["channels_queried"] == 1
    assert data["summary"]["total_points"] == 2  # not doubled
    assert list(data["summary"]["per_channel"].keys()) == ["SR:CURRENT:RB"]

    # The dedup happens before the connector call too.
    queried_channels = connector.get_data.call_args.args[0]
    assert queried_channels == ["SR:CURRENT:RB"]


class TestArchiverReadRealMockConnector:
    """End-to-end coverage through the real ``MockArchiverConnector``.

    Every test above patches the connector factory, so only these exercise a
    real connector's ``get_data`` — including its handling of ``precision_ms``.
    """

    @pytest.mark.unit
    async def test_bin_size_zero_full_resolution_succeeds(self, archiver_project):
        """bin_size=0 (precision_ms=0) must not raise ZeroDivisionError on the real connector."""
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
        # Full resolution at the mock's 1 Hz over 300 s is exactly 300 samples;
        # `> 0` would still pass a degraded ten-point floor.
        assert data["summary"]["per_channel"]["SR:DCCT"]["points"] == 300

    @pytest.mark.unit
    async def test_bin_size_zero_returns_more_points_than_binned(self, archiver_project):
        """Full resolution must strictly out-resolve a binned query on the same window.

        The hour-long window puts the binned count (60) above the old
        ten-point floor, so a regression pinning full resolution at 10 points
        fails this comparison too.
        """
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
    async def test_normal_bin_size_succeeds(self, archiver_project):
        """An ordinary positive bin_size runs the real connector's binning path."""
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
        # One sample per 60-second bin across a one-hour window.
        assert data["summary"]["per_channel"]["SR:DCCT"]["points"] == 60

    @pytest.mark.unit
    async def test_processing_mode_succeeds(self, tmp_path, archiver_project):
        """A non-raw mode really aggregates: one derived value per bin, not raw in disguise.

        A five-minute window at a 60 s bin puts two real samples in each of
        the first four bins, so ``mean`` is a genuine reduction — and ``raw``
        returns the same six rows for this window, so a row count alone
        cannot tell the two apart.
        """
        fn = _get_archiver_read()
        window = {
            "channels": ["SR:DCCT"],
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T10:05:00",
            "bin_size": 60,
        }

        async def channel_series(mode):
            data = extract_response_dict(await fn(**window, processing=mode))
            assert data["status"] == "success"
            file_content = json.loads((tmp_path / data["data_file"]).read_text())
            return file_content["series"]["SR:DCCT"]

        raw = await channel_series("raw")
        mean = await channel_series("mean")
        counts = await channel_series("count")
        lows = await channel_series("min")
        highs = await channel_series("max")

        # `count` proves the bins really hold several real samples.
        assert counts["values"] == [2, 2, 2, 2, 1, 1]

        assert len(mean["values"]) == len(raw["values"]) == 6
        # With two samples in a bin, their mean is exactly (min + max) / 2.
        assert mean["values"] != raw["values"]
        assert mean["values"] == pytest.approx(
            [(low + high) / 2 for low, high in zip(lows["values"], highs["values"], strict=True)]
        )

        # An aggregate is labelled at its bin; a raw sample keeps its own
        # recorded timestamp, which here falls inside the bin, not on its edge.
        assert mean["timestamps"] == [f"2024-01-15T10:0{minute}:00+00:00" for minute in range(6)]
        assert raw["timestamps"] != mean["timestamps"]


# ---------------------------------------------------------------------------
# Coverage verdicts (pure composer)
# ---------------------------------------------------------------------------


def _probe(available=True, start=None, end=None, is_archived=True, note=None):
    """A _CoverageProbe with an ArchiverMetadata built from bare bounds."""
    from osprey.connectors.archiver.base import ArchiverMetadata
    from osprey.mcp_server.control_system.tools.archiver_read import _CoverageProbe

    md = ArchiverMetadata(
        channel="X", is_archived=is_archived, archival_start=start, archival_end=end
    )
    return _CoverageProbe(available=available, metadata=md, note=note)


_WIN_START = datetime(2026, 7, 5, tzinfo=UTC)
_WIN_END = datetime(2026, 7, 6, tzinfo=UTC)
_ARC_START = datetime(2026, 7, 12, 9, 0, tzinfo=UTC)
_ARC_END = datetime(2026, 8, 11, 10, 40, tzinfo=UTC)


class TestComposeCoverage:
    """The pure composer: verdicts, bounds, degradation — no I/O anywhere."""

    def _compose(self, probes, start=_WIN_START, end=_WIN_END):
        from osprey.mcp_server.control_system.tools.archiver_read import _compose_coverage

        return _compose_coverage(start, end, probes)

    def test_no_empty_channels_composes_nothing(self):
        assert self._compose({}) is None

    def test_window_before_the_oldest_sample_precedes_the_archive(self):
        block = self._compose({"CH": _probe(start=_ARC_START, end=_ARC_END)})
        assert block["verdict"] == "window_precedes_archive"
        entry = block["channels"]["CH"]
        assert entry["verdict"] == "window_precedes_archive"
        assert entry["archive_start"] == _ARC_START.isoformat()
        assert entry["archive_end"] == _ARC_END.isoformat()
        # The message states the real bound and rules out synthesis.
        assert _ARC_START.isoformat() in block["message"]
        assert "extrapolat" in block["message"]

    def test_window_after_the_newest_sample_follows_the_archive(self):
        block = self._compose(
            {"CH": _probe(start=_ARC_START, end=_ARC_END)},
            start=_ARC_END + timedelta(days=1),
            end=_ARC_END + timedelta(days=2),
        )
        assert block["verdict"] == "window_follows_archive"
        assert _ARC_END.isoformat() in block["message"]

    def test_window_overlapping_coverage_is_an_honest_gap(self):
        block = self._compose(
            {"CH": _probe(start=_ARC_START, end=_ARC_END)},
            start=_ARC_START + timedelta(days=1),
            end=_ARC_START + timedelta(days=2),
        )
        assert block["verdict"] == "gap_within_coverage"
        assert "recorded" in block["message"]

    def test_window_touching_the_archive_start_counts_as_overlap(self):
        # end == archival_start is NOT "precedes": the boundary sample's
        # instant is inside coverage, so an empty result there is a gap.
        block = self._compose(
            {"CH": _probe(start=_ARC_START, end=_ARC_END)},
            start=_ARC_START - timedelta(days=1),
            end=_ARC_START,
        )
        assert block["verdict"] == "gap_within_coverage"

    def test_unavailable_channel_was_never_recorded(self):
        block = self._compose({"CH": _probe(available=False, start=_ARC_START, end=_ARC_END)})
        assert block["verdict"] == "never_recorded"
        # No bounds on a channel that has none.
        assert "archive_start" not in block["channels"]["CH"]

    def test_metadata_saying_not_archived_means_never_recorded(self):
        block = self._compose({"CH": _probe(is_archived=False)})
        assert block["verdict"] == "never_recorded"

    def test_boundless_metadata_is_unknown_not_guessed(self):
        # The EPICS/DOOCS connectors report is_archived=True with no bounds.
        block = self._compose({"CH": _probe(start=None, end=None)})
        assert block["verdict"] == "coverage_unknown"
        assert "archive_start" not in block["channels"]["CH"]

    def test_probe_failure_note_is_carried_not_raised(self):
        from osprey.mcp_server.control_system.tools.archiver_read import _CoverageProbe

        probe = _CoverageProbe(available=None, metadata=None, note="metadata probe failed: boom")
        block = self._compose({"CH": probe})
        assert block["verdict"] == "coverage_unknown"
        assert block["channels"]["CH"]["note"] == "metadata probe failed: boom"

    def test_naive_metadata_bounds_are_pinned_utc(self):
        # pymongo hands back naive datetimes read as UTC; comparison must not
        # depend on which layer attached the zone.
        block = self._compose(
            {"CH": _probe(start=_ARC_START.replace(tzinfo=None), end=_ARC_END.replace(tzinfo=None))}
        )
        assert block["verdict"] == "window_precedes_archive"
        assert block["channels"]["CH"]["archive_start"] == _ARC_START.isoformat()

    def test_differing_verdicts_report_mixed_with_per_channel_detail(self):
        block = self._compose(
            {
                "OLD": _probe(start=_ARC_START, end=_ARC_END),
                "GONE": _probe(available=False),
            }
        )
        assert block["verdict"] == "mixed"
        assert block["channels"]["OLD"]["verdict"] == "window_precedes_archive"
        assert block["channels"]["GONE"]["verdict"] == "never_recorded"
        assert "coverage.channels" in block["message"]


class TestCoverageInToolResponse:
    """The block reaches the agent — and costs nothing when nothing is empty."""

    async def test_empty_channel_gets_a_coverage_block(self, archiver_read_tool):
        from osprey.connectors.archiver.base import ArchiverMetadata

        fn, connector = archiver_read_tool
        connector.get_data.return_value = _make_archiver_df({})  # nothing at all
        # The fixture's benign probe defaults are side_effects; clear them or
        # these return_values are silently ignored (side_effect wins in Mock).
        connector.check_availability.side_effect = None
        connector.check_availability.return_value = {"SR:OLD:RB": True}
        connector.get_metadata.side_effect = None
        connector.get_metadata.return_value = ArchiverMetadata(
            channel="SR:OLD:RB",
            is_archived=True,
            archival_start=_ARC_START,
            archival_end=_ARC_END,
        )

        result = await fn(
            channels=["SR:OLD:RB"],
            start_time="2026-07-05T00:00:00+00:00",
            end_time="2026-07-06T00:00:00+00:00",
        )

        data = extract_response_dict(result)
        assert data["status"] == "success"
        cov = data["summary"]["coverage"]
        assert cov["verdict"] == "window_precedes_archive"
        assert cov["channels"]["SR:OLD:RB"]["archive_start"] == _ARC_START.isoformat()

    async def test_fully_answered_query_has_no_block_and_no_probe(self, archiver_read_tool):
        fn, connector = archiver_read_tool
        connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.0]})

        result = await fn(channels=["SR:CURRENT:RB"], start_time="2024-01-15T10:00:00")

        data = extract_response_dict(result)
        assert "coverage" not in data["summary"]
        # The probes are the empty path's cost, and only the empty path's.
        connector.get_metadata.assert_not_awaited()
        connector.check_availability.assert_not_awaited()

    async def test_partial_result_explains_only_the_empty_channel(self, archiver_read_tool):
        from osprey.connectors.archiver.base import ArchiverMetadata

        fn, connector = archiver_read_tool
        connector.get_data.return_value = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})
        connector.check_availability.side_effect = None
        connector.check_availability.return_value = {"SR:VOID:RB": False}
        connector.get_metadata.side_effect = None
        connector.get_metadata.return_value = ArchiverMetadata(
            channel="SR:VOID:RB", is_archived=False
        )

        result = await fn(
            channels=["SR:CURRENT:RB", "SR:VOID:RB"], start_time="2024-01-15T10:00:00"
        )

        data = extract_response_dict(result)
        cov = data["summary"]["coverage"]
        assert cov["verdict"] == "never_recorded"
        assert list(cov["channels"]) == ["SR:VOID:RB"]
        # The answered channel's data is untouched.
        assert data["summary"]["per_channel"]["SR:CURRENT:RB"]["points"] == 2

    async def test_probe_failure_degrades_to_unknown_not_error(self, archiver_read_tool):
        fn, connector = archiver_read_tool
        connector.get_data.return_value = _make_archiver_df({})
        connector.check_availability.side_effect = RuntimeError("store went away")
        connector.get_metadata.side_effect = RuntimeError("store went away")

        result = await fn(channels=["SR:OLD:RB"], start_time="2024-01-15T10:00:00")

        data = extract_response_dict(result)
        assert data["status"] == "success"  # the read itself did not fail
        cov = data["summary"]["coverage"]
        assert cov["verdict"] == "coverage_unknown"
        assert "store went away" in cov["channels"]["SR:OLD:RB"]["note"]
