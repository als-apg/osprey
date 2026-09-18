"""
Unit tests for MYAArchiverConnector.

All tests mock jlab_archiver_client, so no installed JLab environment or
reachable myquery server is required.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from osprey_connectors.archiver.mya_archiver_connector import MYAArchiverConnector

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_END = datetime(2026, 1, 1, 1, 0, 0, tzinfo=UTC)  # 1-hour window


def _connected(client, timezone="UTC"):
    """A connected connector wired to a mock jlab_archiver_client.

    ``connect()`` is bypassed so the tests do not have to mock the client's
    config object; the attributes it would have set are set directly. The
    timezone is pinned to UTC so expected timestamps are readable.
    """
    connector = MYAArchiverConnector()
    connector._connected = True
    connector._client = client
    connector._timeout = 60
    connector._deployment = "ops"
    connector._timezone = __import__("zoneinfo").ZoneInfo(timezone)
    return connector


def _events(values, minutes, name="TEST:PV"):
    """An interval-endpoint series: real events at naive local timestamps."""
    index = pd.to_datetime([datetime(2026, 1, 1, 0, m) for m in minutes])
    return pd.Series(values, index=index, name=name)


def _stats_frame(columns, stats):
    """A mystats-endpoint frame: (timestamp, stat) MultiIndex, channels across.

    Args:
        columns: Mapping of channel name to the values, stat-major.
        stats: The stat names, in the order the values are given.
    """
    index = pd.MultiIndex.from_product(
        [pd.to_datetime([datetime(2026, 1, 1, 0, 0)]), stats],
        names=["timestamp", "stat"],
    )
    return pd.DataFrame(columns, index=index)


# --------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------


def test_signature_matches_the_archiver_contract():
    """Every argument archiver_read passes must be accepted.

    A connector whose get_data lacks `processing` raises TypeError on every
    call, which is how this connector's first version failed in the field.
    """
    import inspect

    from osprey_connectors.archiver.base import ArchiverConnector

    base = inspect.signature(ArchiverConnector.get_data)
    impl = inspect.signature(MYAArchiverConnector.get_data)
    assert list(impl.parameters) == list(base.parameters)


def test_registered_as_a_builtin_archiver():
    """`archiver.type: jlab_archiver` must resolve without a dotted path."""
    from osprey_connectors import types
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

    register_builtin_connectors()
    assert ConnectorFactory._archiver_connectors[types.MYA_ARCHIVER] is MYAArchiverConnector
    assert types.MYA_ARCHIVER in types.CLI_ARCHIVER_TYPES


@pytest.mark.asyncio
async def test_get_data_refuses_when_not_connected():
    connector = MYAArchiverConnector()
    with pytest.raises(RuntimeError, match="not connected"):
        await connector.get_data(["TEST:PV"], _START, _END)


# --------------------------------------------------------------------------------------
# raw -- the interval endpoint
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_uses_the_interval_endpoint_and_returns_a_long_frame():
    client = MagicMock()
    client.interval.Interval.return_value.data = _events([1.0, 2.0, 3.0], [0, 20, 40])
    connector = _connected(client)

    data = await connector.get_data(["TEST:PV"], _START, _END, precision_ms=0)

    assert client.query.IntervalQuery.called
    # mysampler forward-fills onto a shared grid, which the contract forbids.
    assert not client.query.MySamplerQuery.called
    assert list(data.columns) == ["timestamp", "channel", "value"]
    assert str(data["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert data["value"].tolist() == [1.0, 2.0, 3.0]
    assert data["channel"].tolist() == ["TEST:PV"] * 3


@pytest.mark.asyncio
async def test_raw_decimates_at_the_requested_bin_keeping_true_timestamps():
    """A bin keeps its last real sample, at the time it was actually recorded."""
    client = MagicMock()
    client.interval.Interval.return_value.data = _events([1.0, 2.0, 3.0, 4.0], [0, 5, 30, 35])
    connector = _connected(client)

    data = await connector.get_data(["TEST:PV"], _START, _END, precision_ms=30 * 60 * 1000)

    # Two 30-minute bins; the last sample of each survives, at its own timestamp.
    assert data["value"].tolist() == [2.0, 4.0]
    assert [ts.minute for ts in data["timestamp"]] == [5, 35]


@pytest.mark.asyncio
async def test_each_channel_contributes_only_its_own_samples():
    """No shared index: channels with different event times are not padded."""
    client = MagicMock()
    per_channel = {
        "TEST:PV1": _events([1.0, 2.0], [0, 10], name="TEST:PV1"),
        "TEST:PV2": _events([9.0], [45], name="TEST:PV2"),
    }
    client.interval.Interval.side_effect = lambda query: MagicMock(
        data=per_channel[query.channel], run=MagicMock()
    )
    client.query.IntervalQuery.side_effect = lambda **kw: MagicMock(channel=kw["channel"])
    connector = _connected(client)

    data = await connector.get_data(["TEST:PV1", "TEST:PV2"], _START, _END, precision_ms=0)

    assert data["channel"].tolist() == ["TEST:PV1", "TEST:PV1", "TEST:PV2"]
    assert data["value"].tolist() == [1.0, 2.0, 9.0]


@pytest.mark.asyncio
async def test_channel_with_no_data_contributes_no_rows():
    client = MagicMock()
    client.interval.Interval.return_value.data = pd.Series(dtype="float64")
    connector = _connected(client)

    data = await connector.get_data(["TEST:PV"], _START, _END, precision_ms=0)

    assert len(data) == 0
    assert list(data.columns) == ["timestamp", "channel", "value"]
    assert str(data["timestamp"].dtype) == "datetime64[ns, UTC]"


# --------------------------------------------------------------------------------------
# aggregates -- the mystats endpoint
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [("mean", 1.5), ("min", 0.5), ("max", 2.5), ("std", 0.25), ("count", 42.0)],
)
async def test_aggregates_are_computed_server_side(mode, expected):
    """std maps onto mystats' "stdev" and count onto "eventCount"."""
    client = MagicMock()
    client.mystats.MyStats.return_value.data = _stats_frame(
        {"TEST:PV": [1.5, 0.5, 2.5, 0.25, 42.0]},
        ["mean", "min", "max", "stdev", "eventCount"],
    )
    connector = _connected(client)

    data = await connector.get_data(["TEST:PV"], _START, _END, precision_ms=60_000, processing=mode)

    assert client.query.MyStatsQuery.called
    assert not client.query.IntervalQuery.called
    assert data["value"].tolist() == [expected]


@pytest.mark.asyncio
async def test_bin_width_becomes_a_bin_count_for_mystats():
    """mystats takes a number of bins, not a width."""
    client = MagicMock()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": [1.5]}, ["mean"])
    connector = _connected(client)

    await connector.get_data(["TEST:PV"], _START, _END, precision_ms=60_000, processing="mean")

    # A one-hour window in 60-second bins.
    assert client.query.MyStatsQuery.call_args[1]["num_bins"] == 60


@pytest.mark.asyncio
async def test_median_falls_back_to_client_side_binning():
    """MYA computes no median, so raw events are fetched and binned locally."""
    client = MagicMock()
    client.interval.Interval.return_value.data = _events([1.0, 5.0, 3.0, 100.0], [0, 1, 2, 45])
    connector = _connected(client)

    data = await connector.get_data(
        ["TEST:PV"], _START, _END, precision_ms=30 * 60 * 1000, processing="median"
    )

    assert client.query.IntervalQuery.called
    assert not client.query.MyStatsQuery.called
    # median(1, 5, 3) in the first half-hour bin, median(100) in the second.
    assert data["value"].tolist() == [3.0, 100.0]


@pytest.mark.asyncio
async def test_aggregate_on_a_non_numeric_channel_is_refused():
    """An enum channel aggregated server-side would be a lie; name it instead."""
    client = MagicMock()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": ["CW MODE (DC)"]}, ["mean"])
    connector = _connected(client)

    with pytest.raises(ValueError, match="non-numeric"):
        await connector.get_data(["TEST:PV"], _START, _END, precision_ms=60_000, processing="mean")


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_processing_mode_is_refused():
    connector = _connected(MagicMock())
    with pytest.raises(ValueError, match="Unknown processing mode"):
        await connector.get_data(["TEST:PV"], _START, _END, precision_ms=60_000, processing="bogus")


@pytest.mark.asyncio
async def test_aggregate_at_full_resolution_is_refused():
    connector = _connected(MagicMock())
    with pytest.raises(ValueError, match="requires precision_ms > 0"):
        await connector.get_data(["TEST:PV"], _START, _END, precision_ms=0, processing="mean")


@pytest.mark.asyncio
async def test_reversed_window_is_refused():
    connector = _connected(MagicMock())
    with pytest.raises(ValueError, match="end_date must be after start_date"):
        await connector.get_data(["TEST:PV"], _END, _START)


@pytest.mark.asyncio
async def test_non_datetime_bounds_are_refused():
    connector = _connected(MagicMock())
    with pytest.raises(TypeError, match="start_date must be a datetime"):
        await connector.get_data(["TEST:PV"], "2026-01-01", _END)


# --------------------------------------------------------------------------------------
# Metadata and availability
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_metadata_reports_an_archived_channel():
    client = MagicMock()
    client.channel.Channel.return_value.matches = [{"name": "TEST:PV"}]
    connector = _connected(client)

    meta = await connector.get_metadata("TEST:PV")

    assert meta.channel == "TEST:PV"
    assert meta.is_archived is True


@pytest.mark.asyncio
async def test_get_metadata_reports_an_unknown_channel():
    client = MagicMock()
    client.channel.Channel.return_value.matches = []
    connector = _connected(client)

    meta = await connector.get_metadata("TEST:NOPE")

    assert meta.is_archived is False


@pytest.mark.asyncio
async def test_get_metadata_refuses_a_pattern():
    """A SQL wildcard in the address matches many channels; that is an error."""
    client = MagicMock()
    client.channel.Channel.return_value.matches = [{"name": "TEST:PV1"}, {"name": "TEST:PV2"}]
    connector = _connected(client)

    with pytest.raises(ValueError, match="more than one channel"):
        await connector.get_metadata("TEST:PV%")


@pytest.mark.asyncio
async def test_check_availability_reports_each_channel():
    """The sweep reports hits and misses; coverage probing depends on it."""
    client = MagicMock()
    matches = {"TEST:PV1": [{"name": "TEST:PV1"}], "TEST:PV2": []}
    client.channel.Channel.side_effect = lambda query: MagicMock(
        matches=matches[query.pattern], run=MagicMock()
    )
    client.query.ChannelQuery.side_effect = lambda **kw: MagicMock(pattern=kw["pattern"])
    connector = _connected(client)

    assert await connector.check_availability(["TEST:PV1", "TEST:PV2"]) == {
        "TEST:PV1": True,
        "TEST:PV2": False,
    }
