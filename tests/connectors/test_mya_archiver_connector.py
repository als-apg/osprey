"""
Unit tests for MYAArchiverConnector.

All tests mock jlab_archiver_client, so no installed JLab environment or
reachable myquery server is required.

Timestamps arrive from myquery as epoch milliseconds (``unix_timestamps_ms``),
so the fixtures below build integer indices -- the shape the client library
hands back once it has been told not to parse them.
"""

import sys
import time
import types
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from osprey_connectors.archiver.mya_archiver_connector import (
    MYAArchiverConnector,
    _like_literal,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_END = datetime(2026, 1, 1, 1, 0, 0, tzinfo=UTC)  # 1-hour window

_URLS = {
    "interval": "https://myquery.example.edu/myquery/interval",
    "mystats": "https://myquery.example.edu/myquery/mystats",
    "channel": "https://myquery.example.edu/myquery/channel",
}


def _millis(moment: datetime) -> int:
    """A UTC instant as myquery's epoch milliseconds."""
    return int(pd.Timestamp(moment).value // 10**6)


def _connected(client, timezone="UTC"):
    """A connected connector wired to a mock jlab_archiver_client.

    ``connect()`` is bypassed so the tests do not have to mock the client's
    config object; the attributes it would have set are set directly. The
    timezone is pinned to UTC so expected timestamps are readable.
    """
    connector = MYAArchiverConnector()
    connector._connected = True
    connector._client = client
    connector._urls = dict(_URLS)
    connector._timeout = 60
    connector._deployment = "ops"
    connector._timezone = __import__("zoneinfo").ZoneInfo(timezone)
    return connector


def _events(values, minutes, name="TEST:PV"):
    """An interval-endpoint series: real events at epoch-millisecond stamps."""
    index = [_millis(datetime(2026, 1, 1, 0, m, tzinfo=UTC)) for m in minutes]
    return pd.Series(values, index=index, name=name)


def _stats_frame(columns, stats, at=_START):
    """A mystats-endpoint frame: (timestamp, stat) MultiIndex, channels across.

    Args:
        columns: Mapping of channel name to the values, stat-major.
        stats: The stat names, in the order the values are given.
        at: The instant labelling the single bin.
    """
    index = pd.MultiIndex.from_product([[_millis(at)], stats], names=["timestamp", "stat"])
    return pd.DataFrame(columns, index=index)


def _fake_client():
    """A client mock whose runners accept the per-call ``url=`` argument."""
    client = MagicMock()
    client.interval.Interval.side_effect = None
    return client


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
    """`archiver.type: mya_archiver` must resolve without a dotted path."""
    from osprey_connectors import types
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

    register_builtin_connectors()
    assert ConnectorFactory._archiver_connectors[types.MYA_ARCHIVER] is MYAArchiverConnector
    assert types.MYA_ARCHIVER in types.CLI_ARCHIVER_TYPES


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_data", "get_metadata", "check_availability"])
async def test_every_read_refuses_when_not_connected(method):
    """After disconnect() every entry point reports the same thing.

    Without the guard, get_metadata and check_availability reach a None client
    and raise AttributeError from a worker thread.
    """
    connector = MYAArchiverConnector()
    args = {
        "get_data": (["TEST:PV"], _START, _END),
        "get_metadata": ("TEST:PV",),
        "check_availability": (["TEST:PV"],),
    }[method]

    with pytest.raises(RuntimeError, match="not connected"):
        await getattr(connector, method)(*args)


# --------------------------------------------------------------------------------------
# connect() -- settings scoping and defaults
# --------------------------------------------------------------------------------------


class _FakeLibraryConfig:
    """Stand-in for jlab_archiver_client's process-global config singleton."""

    def __init__(self):
        self.protocol = "https"
        self.myquery_server = "epicsweb.jlab.org"
        self.interval_path = "/myquery/interval"
        self.mystats_path = "/myquery/mystats"
        self.channel_path = "/myquery/channel"
        self.set_calls = []

    def set(self, **kwargs):
        self.set_calls.append(kwargs)


def _fake_module(settings):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(config=settings),
        query=MagicMock(),
        interval=MagicMock(),
        mystats=MagicMock(),
        channel=MagicMock(),
    )


@pytest.mark.asyncio
async def test_connect_scopes_its_server_per_call_not_process_wide():
    """The library's global config must not be mutated.

    `config.set()` writes a process-global singleton: a second connector with
    no overrides would inherit this one's server, and two connected instances
    would clobber each other. The overrides become per-call URLs instead.
    """
    settings = _FakeLibraryConfig()
    connector = MYAArchiverConnector()

    with patch.dict(sys.modules, {"jlab_archiver_client": _fake_module(settings)}):
        await connector.connect({"myquery_server": "myquery.example.edu", "protocol": "http"})

    assert settings.set_calls == []
    assert settings.myquery_server == "epicsweb.jlab.org"
    assert connector._urls["interval"] == "http://myquery.example.edu/myquery/interval"
    assert connector._urls["mystats"] == "http://myquery.example.edu/myquery/mystats"
    assert connector._urls["channel"] == "http://myquery.example.edu/myquery/channel"


@pytest.mark.asyncio
async def test_connect_keeps_the_library_defaults_when_nothing_is_named():
    """Every setting is optional; the client's own server is the fallback."""
    settings = _FakeLibraryConfig()
    connector = MYAArchiverConnector()

    with patch.dict(sys.modules, {"jlab_archiver_client": _fake_module(settings)}):
        await connector.connect({})

    assert connector._urls["mystats"] == "https://epicsweb.jlab.org/myquery/mystats"
    assert connector._deployment == "ops"


@pytest.mark.asyncio
async def test_an_explicitly_empty_timeout_still_bounds_the_wait():
    """`timeout:` left blank in YAML is None, which wait_for treats as forever."""
    settings = _FakeLibraryConfig()
    connector = MYAArchiverConnector()

    with patch.dict(sys.modules, {"jlab_archiver_client": _fake_module(settings)}):
        await connector.connect({"timeout": None, "deployment": None})

    assert connector._timeout == 60
    assert connector._deployment == "ops"


# --------------------------------------------------------------------------------------
# Timestamps -- epoch milliseconds, and the client floor that makes them work
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_endpoints_are_asked_for_epoch_milliseconds():
    """A naive wall clock cannot name an instant in the fall-back hour."""
    client = _fake_client()
    client.interval.Interval.return_value.data = _events([1.0], [0])
    connector = _connected(client)
    await connector.get_data(["TEST:PV"], _START, _END, precision_ms=0)
    assert client.query.IntervalQuery.call_args[1]["unix_timestamps_ms"] is True

    client = _fake_client()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": [1.5]}, ["mean"])
    connector = _connected(client)
    await connector.get_data(["TEST:PV"], _START, _END, precision_ms=60_000, processing="mean")
    assert client.query.MyStatsQuery.call_args[1]["unix_timestamps_ms"] is True


@pytest.mark.asyncio
async def test_the_repeated_fall_back_hour_keeps_its_true_instants():
    """1 Nov 2026 01:00-02:00 happens twice in America/New_York.

    Read as a naive wall clock, the second (standard-time) pass lands an hour
    early and the series stops being monotonic. Epoch milliseconds have no
    ambiguous hour, so the instants survive the round trip intact.
    """
    truth = pd.to_datetime(
        [
            "2026-11-01T04:30Z",  # 00:30 EDT
            "2026-11-01T05:15Z",  # 01:15 EDT, first pass
            "2026-11-01T05:45Z",
            "2026-11-01T06:15Z",  # 01:15 EST, second pass
            "2026-11-01T06:45Z",
            "2026-11-01T07:30Z",  # 02:30 EST
        ],
        utc=True,
    )
    client = _fake_client()
    client.interval.Interval.return_value.data = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=[_millis(t) for t in truth], name="TEST:PV"
    )
    connector = _connected(client, timezone="America/New_York")

    data = await connector.get_data(
        ["TEST:PV"],
        datetime(2026, 11, 1, 3, 0, tzinfo=UTC),
        datetime(2026, 11, 1, 9, 0, tzinfo=UTC),
        precision_ms=0,
    )

    assert list(data["timestamp"]) == list(truth)
    assert data["timestamp"].is_monotonic_increasing
    assert data["value"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


@pytest.mark.asyncio
async def test_a_client_library_that_ignores_the_flag_is_refused():
    """Before 4.0.1 the flag was accepted and then ignored.

    The client parsed the epoch milliseconds as nanoseconds, dating every
    sample to January 1970 -- with no error. Serving that is worse than
    failing, so a non-integer index is refused by name.
    """
    client = _fake_client()
    client.interval.Interval.return_value.data = pd.Series(
        [1.0, 2.0],
        index=pd.to_datetime(["1970-01-01 00:29:24.572400", "1970-01-01 00:29:24.576000"]),
        name="TEST:PV",
    )
    connector = _connected(client)

    with pytest.raises(RuntimeError, match=r"4\.0\.1"):
        await connector.get_data(["TEST:PV"], _START, _END, precision_ms=0)


# --------------------------------------------------------------------------------------
# raw -- the interval endpoint
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_uses_the_interval_endpoint_and_returns_a_long_frame():
    client = _fake_client()
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
async def test_the_endpoint_url_is_passed_per_call():
    """Not read from the library's global config at request time."""
    client = _fake_client()
    client.interval.Interval.return_value.data = _events([1.0], [0])
    connector = _connected(client)

    await connector.get_data(["TEST:PV"], _START, _END, precision_ms=0)

    assert client.interval.Interval.call_args[1]["url"] == _URLS["interval"]


@pytest.mark.asyncio
async def test_raw_decimates_at_the_requested_bin_keeping_true_timestamps():
    """A bin keeps its last real sample, at the time it was actually recorded."""
    client = _fake_client()
    client.interval.Interval.return_value.data = _events([1.0, 2.0, 3.0, 4.0], [0, 5, 30, 35])
    connector = _connected(client)

    data = await connector.get_data(["TEST:PV"], _START, _END, precision_ms=30 * 60 * 1000)

    # Two 30-minute bins; the last sample of each survives, at its own timestamp.
    assert data["value"].tolist() == [2.0, 4.0]
    assert [ts.minute for ts in data["timestamp"]] == [5, 35]


@pytest.mark.asyncio
async def test_each_channel_contributes_only_its_own_samples():
    """No shared index: channels with different event times are not padded."""
    client = _fake_client()
    per_channel = {
        "TEST:PV1": _events([1.0, 2.0], [0, 10], name="TEST:PV1"),
        "TEST:PV2": _events([9.0], [45], name="TEST:PV2"),
    }
    client.interval.Interval.side_effect = lambda query, url=None: MagicMock(
        data=per_channel[query.channel], run=MagicMock()
    )
    client.query.IntervalQuery.side_effect = lambda **kw: MagicMock(channel=kw["channel"])
    connector = _connected(client)

    data = await connector.get_data(["TEST:PV1", "TEST:PV2"], _START, _END, precision_ms=0)

    assert data["channel"].tolist() == ["TEST:PV1", "TEST:PV1", "TEST:PV2"]
    assert data["value"].tolist() == [1.0, 2.0, 9.0]


@pytest.mark.asyncio
async def test_channel_with_no_data_contributes_no_rows():
    client = _fake_client()
    client.interval.Interval.return_value.data = pd.Series(dtype="float64")
    connector = _connected(client)

    data = await connector.get_data(["TEST:PV"], _START, _END, precision_ms=0)

    assert len(data) == 0
    assert list(data.columns) == ["timestamp", "channel", "value"]
    assert str(data["timestamp"].dtype) == "datetime64[ns, UTC]"


# --------------------------------------------------------------------------------------
# The prior point -- MYA records changes, not samples
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_prior_point_is_requested_and_stamped_at_the_window_start():
    """A setpoint last changed before the window still has a known value.

    Without the prior point MYA answers nothing for it and the agent reports
    "no data" for a channel whose value is perfectly well known.
    """
    client = _fake_client()
    prior = _millis(_START - timedelta(hours=9))
    client.interval.Interval.return_value.data = pd.Series([7.0], index=[prior], name="TEST:PV")
    connector = _connected(client)

    data = await connector.get_data(["TEST:PV"], _START, _END, precision_ms=0)

    assert client.query.IntervalQuery.call_args[1]["prior_point"] is True
    assert data["value"].tolist() == [7.0]
    # Reported at the window start whose value it states, not nine hours before
    # it, where it would sit outside the range the caller asked for.
    assert list(data["timestamp"]) == [pd.Timestamp(_START)]


@pytest.mark.asyncio
async def test_a_real_sample_at_the_window_start_outranks_the_prior_point():
    """Both land on the same instant; the recorded one is the better witness."""
    client = _fake_client()
    client.interval.Interval.return_value.data = pd.Series(
        [7.0, 9.0, 11.0],
        index=[_millis(_START - timedelta(hours=9)), _millis(_START), _millis(_END)],
        name="TEST:PV",
    )
    connector = _connected(client)

    data = await connector.get_data(["TEST:PV"], _START, _END, precision_ms=0)

    assert data["value"].tolist() == [9.0, 11.0]
    assert list(data["timestamp"]) == [pd.Timestamp(_START), pd.Timestamp(_END)]


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
    client = _fake_client()
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
    client = _fake_client()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": [1.5]}, ["mean"])
    connector = _connected(client)

    await connector.get_data(["TEST:PV"], _START, _END, precision_ms=60_000, processing="mean")

    # A one-hour window in 60-second bins.
    assert client.query.MyStatsQuery.call_args[1]["num_bins"] == 60
    assert client.mystats.MyStats.call_count == 1


@pytest.mark.asyncio
async def test_a_width_that_does_not_divide_the_window_adds_a_partial_bin():
    """25-minute bins over an hour: two whole bins, then the ragged 10 minutes.

    Serving 20-minute bins instead would be a width the caller never asked
    for; dropping the remainder would discard real data from inside the window.
    Every other shipped archiver returns the partial final bin, which is what
    pandas' resample produces for the same request.
    """
    client = _fake_client()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": [1.5]}, ["mean"])
    connector = _connected(client)

    await connector.get_data(
        ["TEST:PV"], _START, _END, precision_ms=25 * 60 * 1000, processing="mean"
    )

    assert client.mystats.MyStats.call_count == 2
    whole, tail = client.query.MyStatsQuery.call_args_list
    assert whole[1]["num_bins"] == 2
    assert whole[1]["start"] == datetime(2026, 1, 1, 0, 0)
    assert whole[1]["end"] == datetime(2026, 1, 1, 0, 50)
    assert tail[1]["num_bins"] == 1
    assert tail[1]["start"] == datetime(2026, 1, 1, 0, 50)
    assert tail[1]["end"] == datetime(2026, 1, 1, 1, 0)


@pytest.mark.asyncio
async def test_a_window_narrower_than_one_bin_is_a_single_partial_bin():
    client = _fake_client()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": [1.5]}, ["mean"])
    connector = _connected(client)

    await connector.get_data(
        ["TEST:PV"], _START, _END, precision_ms=90 * 60 * 1000, processing="mean"
    )

    assert client.mystats.MyStats.call_count == 1
    assert client.query.MyStatsQuery.call_args[1]["num_bins"] == 1


@pytest.mark.asyncio
async def test_the_bin_count_is_computed_across_a_dst_transition():
    """Naive local arithmetic would make this window an hour short."""
    client = _fake_client()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": [1.5]}, ["mean"])
    connector = _connected(client, timezone="America/New_York")

    # Six real hours spanning the fall-back; the wall clock advances only five.
    await connector.get_data(
        ["TEST:PV"],
        datetime(2026, 11, 1, 3, 0, tzinfo=UTC),
        datetime(2026, 11, 1, 9, 0, tzinfo=UTC),
        precision_ms=60 * 60 * 1000,
        processing="mean",
    )

    assert client.query.MyStatsQuery.call_args[1]["num_bins"] == 6


@pytest.mark.asyncio
async def test_aggregate_on_a_non_numeric_channel_is_refused():
    """An enum channel aggregated server-side would be a lie; name it instead."""
    client = _fake_client()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": ["CW MODE (DC)"]}, ["mean"])
    connector = _connected(client)

    with pytest.raises(ValueError, match="non-numeric"):
        await connector.get_data(["TEST:PV"], _START, _END, precision_ms=60_000, processing="mean")


@pytest.mark.asyncio
async def test_median_falls_back_to_client_side_binning():
    """MYA computes no median, so raw events are fetched and binned locally."""
    client = _fake_client()
    client.interval.Interval.return_value.data = _events([1.0, 5.0, 3.0, 100.0], [0, 1, 2, 45])
    connector = _connected(client)

    data = await connector.get_data(
        ["TEST:PV"], _START, _END, precision_ms=30 * 60 * 1000, processing="median"
    )

    assert client.query.IntervalQuery.called
    assert not client.query.MyStatsQuery.called
    # median(1, 5, 3) in the first half-hour bin, median(100) in the second.
    assert data["value"].tolist() == [3.0, 100.0]


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_processing_mode_is_refused():
    connector = _connected(_fake_client())
    with pytest.raises(ValueError, match="Unknown processing mode"):
        await connector.get_data(["TEST:PV"], _START, _END, precision_ms=60_000, processing="bogus")


@pytest.mark.asyncio
async def test_aggregate_at_full_resolution_is_refused():
    connector = _connected(_fake_client())
    with pytest.raises(ValueError, match="requires precision_ms > 0"):
        await connector.get_data(["TEST:PV"], _START, _END, precision_ms=0, processing="mean")


@pytest.mark.asyncio
async def test_reversed_window_is_refused():
    connector = _connected(_fake_client())
    with pytest.raises(ValueError, match="end_date must be after start_date"):
        await connector.get_data(["TEST:PV"], _END, _START)


@pytest.mark.asyncio
async def test_non_datetime_bounds_are_refused():
    connector = _connected(_fake_client())
    with pytest.raises(TypeError, match="start_date must be a datetime"):
        await connector.get_data(["TEST:PV"], "2026-01-01", _END)


# --------------------------------------------------------------------------------------
# Metadata and availability
# --------------------------------------------------------------------------------------


def test_channel_names_are_quoted_against_sql_like():
    """Verified against epicsweb: the server honours backslash escapes.

    `_` is a single-character wildcard there, and accelerator names are full of
    underscores -- `IGLK100HVPSkVolt_` matches `IGLK100HVPSkVolts`.
    """
    assert _like_literal("IGLK100Preset_Volt") == r"IGLK100Preset\_Volt"
    assert _like_literal("A%B") == r"A\%B"
    assert _like_literal("IGLK100HVPSkVolts") == "IGLK100HVPSkVolts"
    # The escape character itself is quoted first, or it would escape an escape.
    assert _like_literal("A\\B") == r"A\\B"


@pytest.mark.asyncio
async def test_get_metadata_quotes_the_address_before_looking_it_up():
    client = _fake_client()
    client.channel.Channel.return_value.matches = [{"name": "IGLK100Preset_Volt"}]
    connector = _connected(client)

    meta = await connector.get_metadata("IGLK100Preset_Volt")

    assert client.query.ChannelQuery.call_args[1]["pattern"] == r"IGLK100Preset\_Volt"
    assert meta.is_archived is True
    assert meta.channel == "IGLK100Preset_Volt"


@pytest.mark.asyncio
async def test_get_metadata_reports_an_unknown_channel():
    client = _fake_client()
    client.channel.Channel.return_value.matches = []
    connector = _connected(client)

    meta = await connector.get_metadata("TEST:NOPE")

    assert meta.is_archived is False


@pytest.mark.asyncio
async def test_get_metadata_refuses_a_pattern():
    """A SQL wildcard in the address is a caller error, not a lookup miss.

    Quoting means the query itself can no longer match siblings, so the
    give-away is the `%` in the address rather than the number of matches.
    `_` cannot serve here -- it is legal in a channel name.
    """
    connector = _connected(_fake_client())

    with pytest.raises(ValueError, match="looks like a pattern"):
        await connector.get_metadata("TEST:PV%")


@pytest.mark.asyncio
async def test_check_availability_reports_each_channel():
    """The sweep reports hits and misses; coverage probing depends on it."""
    client = _fake_client()
    matches = {"TEST:PV1": [{"name": "TEST:PV1"}], "TEST:PV2": []}
    client.channel.Channel.side_effect = lambda query, url=None: MagicMock(
        matches=matches[query.pattern], run=MagicMock()
    )
    client.query.ChannelQuery.side_effect = lambda **kw: MagicMock(pattern=kw["pattern"])
    connector = _connected(client)

    assert await connector.check_availability(["TEST:PV1", "TEST:PV2"]) == {
        "TEST:PV1": True,
        "TEST:PV2": False,
    }


@pytest.mark.asyncio
async def test_check_availability_bounds_each_channel_not_the_whole_sweep():
    """A wide list must not time out as a unit and discard every answer.

    Ten lookups at 50 ms each exceed the 300 ms timeout in total but sit well
    inside it individually, so this passes only if the bound is per channel.
    """
    client = _fake_client()
    client.channel.Channel.side_effect = lambda query, url=None: MagicMock(
        matches=[{"name": query.pattern}], run=lambda: time.sleep(0.05)
    )
    client.query.ChannelQuery.side_effect = lambda **kw: MagicMock(pattern=kw["pattern"])
    connector = _connected(client)
    connector._timeout = 0.3

    channels = [f"TEST:PV{n}" for n in range(10)]
    assert await connector.check_availability(channels) == dict.fromkeys(channels, True)
