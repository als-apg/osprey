"""
Unit tests for MYAArchiverConnector.

All tests mock jlab_archiver_client, so no installed JLab environment or
reachable myquery server is required.

Timestamps arrive from myquery as epoch milliseconds (``unix_timestamps_ms``),
so the fixtures below build integer indices -- the shape the client library
hands back once it has been told not to parse them.
"""

import asyncio
import sys
import types
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from osprey_connectors.archiver import ArchiveCoverage
from osprey_connectors.archiver._mya_archiver_connector import (
    MYAArchiverConnector,
    _like_literal,
)
from osprey_connectors.errors import ArchiveQueryError
from osprey_connectors.service_answers import BackendSpec, Unavailable

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


def _spec(settings: dict | None = None) -> BackendSpec:
    return BackendSpec(
        kind="archive", type="mya_archiver", target="live", settings=settings or {}, rehearsal=False
    )


def _built(settings: dict | None = None, settings_module=None) -> MYAArchiverConnector:
    """An archive built from *settings* against a fake client library."""
    module = _fake_module(settings_module or _FakeLibraryConfig())
    with patch.dict(sys.modules, {"jlab_archiver_client": module}):
        return MYAArchiverConnector(_spec(settings))


def _no_answer(answer: object, cause: str) -> str:
    """*answer*'s reason, after asserting it is ``Unavailable`` with *cause*."""
    assert isinstance(answer, Unavailable), answer
    assert answer.cause == cause, answer
    return answer.reason


def _connected(client, timezone="UTC"):
    """An archive wired to a mock jlab_archiver_client.

    It is built against a fake library config, then pointed at *client* and the
    fixed endpoint URLs, so the tests do not have to mock the client's config
    object per call. The timezone is pinned to UTC so expected timestamps are
    readable.
    """
    connector = _built({"timeout_s": 60, "timezone": timezone})
    connector._client = client
    connector._urls = dict(_URLS)
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

    A connector whose history lacks `processing` raises TypeError on every
    call, which is how this connector's first version failed in the field.
    """
    import inspect

    from osprey_connectors.archiver import ChannelArchive

    base = inspect.signature(ChannelArchive.history)
    impl = inspect.signature(MYAArchiverConnector.history)
    assert list(impl.parameters) == list(base.parameters)


@pytest.mark.asyncio
async def test_a_closed_archive_still_answers():
    """myquery is stateless HTTP: closing holds nothing, so a later read still asks."""
    client = _fake_client()
    client.channel.Channel.return_value.matches = [{"name": "TEST:PV"}]
    connector = _connected(client)

    await connector.aclose()
    await connector.aclose()

    coverage = await connector.coverage(["TEST:PV"])
    assert coverage["TEST:PV"].is_archived is True


# --------------------------------------------------------------------------------------
# Building -- settings scoping and defaults
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
async def test_building_scopes_its_server_per_call_not_process_wide():
    """The library's global config must not be mutated.

    `config.set()` writes a process-global singleton: a second connector with
    no overrides would inherit this one's server, and two connected instances
    would clobber each other. The overrides become per-call URLs instead.
    """
    settings = _FakeLibraryConfig()
    connector = _built({"myquery_server": "myquery.example.edu", "protocol": "http"}, settings)

    assert settings.set_calls == []
    assert settings.myquery_server == "epicsweb.jlab.org"
    assert connector._urls["interval"] == "http://myquery.example.edu/myquery/interval"
    assert connector._urls["mystats"] == "http://myquery.example.edu/myquery/mystats"
    assert connector._urls["channel"] == "http://myquery.example.edu/myquery/channel"


@pytest.mark.asyncio
async def test_building_keeps_the_library_defaults_when_nothing_is_named():
    """Every setting is optional; the client's own server is the fallback."""
    connector = _built({})

    assert connector._urls["mystats"] == "https://epicsweb.jlab.org/myquery/mystats"
    assert connector._deployment == "ops"


@pytest.mark.asyncio
async def test_an_explicitly_empty_timeout_still_bounds_the_wait():
    """`timeout_s:` left blank in YAML is None, which wait_for treats as forever."""
    connector = _built({"timeout_s": None, "deployment": None})

    assert connector._timeout == 60
    assert connector._deployment == "ops"


@pytest.mark.asyncio
async def test_timeout_s_sets_the_wait():
    connector = _built({"timeout_s": 15})

    assert connector._timeout == 15


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth",
    [{"token_env": "ARCHIVER_TOKEN"}, {"username": "u", "password_env": "ARCHIVER_PASSWORD"}],
    ids=["token", "password"],
)
async def test_a_login_is_refused_by_name(auth):
    with pytest.raises(ValueError, match=r"`archiver\.settings\.auth`.*jlab-archiver-client"):
        _built({"auth": auth})


@pytest.mark.asyncio
async def test_a_ca_bundle_is_refused_by_name():
    with pytest.raises(ValueError, match=r"`archiver\.settings\.tls`.*jlab-archiver-client"):
        _built({"tls": {"ca_bundle": "/etc/ssl/certs/site-ca.pem"}})


@pytest.mark.asyncio
async def test_the_refusal_needs_no_client_library():
    with (
        patch.dict(sys.modules, {"jlab_archiver_client": None}),
        pytest.raises(ValueError, match=r"`archiver\.settings\.auth`"),
    ):
        MYAArchiverConnector(_spec({"auth": {"token_env": "ARCHIVER_TOKEN"}}))


@pytest.mark.asyncio
async def test_building_without_the_client_library_names_it():
    with (
        patch.dict(sys.modules, {"jlab_archiver_client": None}),
        pytest.raises(ImportError, match=r"jlab-archiver-client>=4\.0\.1"),
    ):
        MYAArchiverConnector(_spec())


@pytest.mark.asyncio
async def test_a_flat_timeout_is_refused_naming_timeout_s():
    with pytest.raises(ValueError, match=r"`archiver\.settings\.timeout_s`"):
        _built({"timeout": 60})


# --------------------------------------------------------------------------------------
# Timestamps -- epoch milliseconds, and the client floor that makes them work
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_endpoints_are_asked_for_epoch_milliseconds():
    """A naive wall clock cannot name an instant in the fall-back hour."""
    client = _fake_client()
    client.interval.Interval.return_value.data = _events([1.0], [0])
    connector = _connected(client)
    await connector.history(["TEST:PV"], start_at=_START, end_at=_END, precision_ms=0)
    assert client.query.IntervalQuery.call_args[1]["unix_timestamps_ms"] is True

    client = _fake_client()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": [1.5]}, ["mean"])
    connector = _connected(client)
    await connector.history(
        ["TEST:PV"], start_at=_START, end_at=_END, precision_ms=60_000, processing="mean"
    )
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

    data = await connector.history(
        ["TEST:PV"],
        start_at=datetime(2026, 11, 1, 3, 0, tzinfo=UTC),
        end_at=datetime(2026, 11, 1, 9, 0, tzinfo=UTC),
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

    answer = await connector.history(["TEST:PV"], start_at=_START, end_at=_END, precision_ms=0)
    assert "jlab-archiver-client>=4.0.1" in _no_answer(answer, "misconfigured")


# --------------------------------------------------------------------------------------
# raw -- the interval endpoint
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_uses_the_interval_endpoint_and_returns_a_long_frame():
    client = _fake_client()
    client.interval.Interval.return_value.data = _events([1.0, 2.0, 3.0], [0, 20, 40])
    connector = _connected(client)

    data = await connector.history(["TEST:PV"], start_at=_START, end_at=_END, precision_ms=0)

    assert client.query.IntervalQuery.called
    # mysampler forward-fills onto a shared grid, which the contract forbids.
    assert not client.query.MySamplerQuery.called
    assert list(data.columns) == ["timestamp", "address", "value"]
    assert str(data["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert data["value"].tolist() == [1.0, 2.0, 3.0]
    assert data["address"].tolist() == ["TEST:PV"] * 3


@pytest.mark.asyncio
async def test_the_endpoint_url_is_passed_per_call():
    """Not read from the library's global config at request time."""
    client = _fake_client()
    client.interval.Interval.return_value.data = _events([1.0], [0])
    connector = _connected(client)

    await connector.history(["TEST:PV"], start_at=_START, end_at=_END, precision_ms=0)

    assert client.interval.Interval.call_args[1]["url"] == _URLS["interval"]


@pytest.mark.asyncio
async def test_raw_decimates_at_the_requested_bin_keeping_true_timestamps():
    """A bin keeps its last real sample, at the time it was actually recorded."""
    client = _fake_client()
    client.interval.Interval.return_value.data = _events([1.0, 2.0, 3.0, 4.0], [0, 5, 30, 35])
    connector = _connected(client)

    data = await connector.history(
        ["TEST:PV"], start_at=_START, end_at=_END, precision_ms=30 * 60 * 1000
    )

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

    data = await connector.history(
        ["TEST:PV1", "TEST:PV2"], start_at=_START, end_at=_END, precision_ms=0
    )

    assert data["address"].tolist() == ["TEST:PV1", "TEST:PV1", "TEST:PV2"]
    assert data["value"].tolist() == [1.0, 2.0, 9.0]


@pytest.mark.asyncio
async def test_channel_with_no_data_contributes_no_rows():
    client = _fake_client()
    client.interval.Interval.return_value.data = pd.Series(dtype="float64")
    connector = _connected(client)

    data = await connector.history(["TEST:PV"], start_at=_START, end_at=_END, precision_ms=0)

    assert len(data) == 0
    assert list(data.columns) == ["timestamp", "address", "value"]
    assert str(data["timestamp"].dtype) == "datetime64[ns, UTC]"


# --------------------------------------------------------------------------------------
# The prior point -- MYA records changes, not samples
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_prior_point_is_requested_and_kept_at_its_recorded_time():
    """A setpoint last changed before the window still has a known value.

    Without the prior point MYA answers nothing for it and the agent reports
    "no data" for a channel whose value is perfectly well known. It is kept at
    the instant it was recorded -- before the window -- not relabelled to the
    window start, which would fabricate a timestamp nothing was recorded at.
    """
    client = _fake_client()
    prior = _START - timedelta(hours=9)
    client.interval.Interval.return_value.data = pd.Series(
        [7.0, 9.0], index=[_millis(prior), _millis(_START + timedelta(minutes=5))], name="TEST:PV"
    )
    connector = _connected(client)

    data = await connector.history(["TEST:PV"], start_at=_START, end_at=_END, precision_ms=0)

    assert client.query.IntervalQuery.call_args[1]["prior_point"] is True
    assert data["value"].tolist() == [7.0, 9.0]
    assert list(data["timestamp"]) == [
        pd.Timestamp(prior),
        pd.Timestamp(_START + timedelta(minutes=5)),
    ]


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

    data = await connector.history(
        ["TEST:PV"], start_at=_START, end_at=_END, precision_ms=60_000, processing=mode
    )

    assert client.query.MyStatsQuery.called
    assert not client.query.IntervalQuery.called
    assert data["value"].tolist() == [expected]


@pytest.mark.asyncio
async def test_bin_width_becomes_a_bin_count_for_mystats():
    """mystats takes a number of bins, not a width."""
    client = _fake_client()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": [1.5]}, ["mean"])
    connector = _connected(client)

    await connector.history(
        ["TEST:PV"], start_at=_START, end_at=_END, precision_ms=60_000, processing="mean"
    )

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

    await connector.history(
        ["TEST:PV"], start_at=_START, end_at=_END, precision_ms=25 * 60 * 1000, processing="mean"
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

    await connector.history(
        ["TEST:PV"], start_at=_START, end_at=_END, precision_ms=90 * 60 * 1000, processing="mean"
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
    await connector.history(
        ["TEST:PV"],
        start_at=datetime(2026, 11, 1, 3, 0, tzinfo=UTC),
        end_at=datetime(2026, 11, 1, 9, 0, tzinfo=UTC),
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

    with pytest.raises(ArchiveQueryError, match="non-numeric") as refused:
        await connector.history(
            ["TEST:PV"], start_at=_START, end_at=_END, precision_ms=60_000, processing="mean"
        )
    assert (refused.value.parameter, refused.value.address) == ("processing", "TEST:PV")


@pytest.mark.asyncio
async def test_median_falls_back_to_client_side_binning():
    """MYA computes no median, so raw events are fetched and binned locally."""
    client = _fake_client()
    client.interval.Interval.return_value.data = _events([1.0, 5.0, 3.0, 100.0], [0, 1, 2, 45])
    connector = _connected(client)

    data = await connector.history(
        ["TEST:PV"], start_at=_START, end_at=_END, precision_ms=30 * 60 * 1000, processing="median"
    )

    assert client.query.IntervalQuery.called
    assert not client.query.MyStatsQuery.called
    # median(1, 5, 3) in the first half-hour bin, median(100) in the second.
    assert data["value"].tolist() == [3.0, 100.0]


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("precision_ms", "processing", "match", "parameter"),
    [
        pytest.param(
            60_000, "bogus", "Unknown processing mode", "processing", id="unknown-processing-mode"
        ),
        pytest.param(
            0,
            "mean",
            "requires precision_ms > 0",
            "precision_ms",
            id="aggregate-at-full-resolution",
        ),
    ],
)
async def test_an_unusable_processing_request_is_refused(
    precision_ms, processing, match, parameter
):
    connector = _connected(_fake_client())
    with pytest.raises(ArchiveQueryError, match=match) as refused:
        await connector.history(
            ["TEST:PV"],
            start_at=_START,
            end_at=_END,
            precision_ms=precision_ms,
            processing=processing,
        )
    assert refused.value.parameter == parameter


@pytest.mark.asyncio
async def test_reversed_window_is_refused():
    connector = _connected(_fake_client())
    with pytest.raises(ArchiveQueryError, match="end_at must be after start_at") as refused:
        await connector.history(["TEST:PV"], start_at=_END, end_at=_START)
    assert refused.value.parameter == "end_at"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised", "cause", "reason"),
    [
        (
            ConnectionRefusedError("refused"),
            "unreachable",
            "Cannot connect to the MYA archiver. Check connectivity and configuration.",
        ),
        (
            OSError("connection reset by peer"),
            "unreachable",
            "Network issue reaching the MYA archiver: connection reset by peer",
        ),
        (RuntimeError("myquery broke"), "failed", "the MYA archiver read failed: myquery broke"),
    ],
    ids=["refused", "connection-words", "anything-else"],
)
async def test_a_failed_read_answers_why(raised, cause, reason):
    """A read myquery gives no answer to is ``Unavailable``, never raised."""
    client = _fake_client()
    client.interval.Interval.return_value.run.side_effect = raised
    connector = _connected(client)

    answer = await connector.history(["TEST:PV"], start_at=_START, end_at=_END, precision_ms=0)
    assert _no_answer(answer, cause) == reason


@pytest.mark.asyncio
async def test_a_read_that_outlasts_its_budget_answers_timeout():
    client = _fake_client()
    client.interval.Interval.return_value.run.side_effect = lambda: __import__("time").sleep(1)
    connector = _connected(client)

    answer = await connector.history(
        ["TEST:PV"], start_at=_START, end_at=_END, precision_ms=0, timeout_s=0.01
    )
    assert _no_answer(answer, "timeout") == "MYA archiver request timed out after 0.01s"


@pytest.mark.asyncio
async def test_a_missing_statistic_answers_bad_reply():
    client = _fake_client()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": [1.0]}, ["min"])
    connector = _connected(client)

    answer = await connector.history(
        ["TEST:PV"], start_at=_START, end_at=_END, precision_ms=60 * 60 * 1000, processing="mean"
    )
    assert _no_answer(answer, "bad_reply") == (
        "MYA returned no 'mean' statistic for processing='mean'"
    )


@pytest.mark.asyncio
async def test_non_datetime_bounds_are_refused():
    connector = _connected(_fake_client())
    with pytest.raises(TypeError, match="start_at must be a datetime"):
        await connector.history(["TEST:PV"], start_at="2026-01-01", end_at=_END)


# --------------------------------------------------------------------------------------
# Coverage
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
async def test_coverage_quotes_the_address_before_looking_it_up():
    client = _fake_client()
    client.channel.Channel.return_value.matches = [{"name": "IGLK100Preset_Volt"}]
    connector = _connected(client)

    coverage = await connector.coverage(["IGLK100Preset_Volt"])

    assert client.query.ChannelQuery.call_args[1]["pattern"] == r"IGLK100Preset\_Volt"
    # MYA states no bounds.
    assert coverage == {
        "IGLK100Preset_Volt": ArchiveCoverage(address="IGLK100Preset_Volt", is_archived=True)
    }


@pytest.mark.asyncio
async def test_coverage_reports_an_unknown_channel():
    client = _fake_client()
    client.channel.Channel.return_value.matches = []
    connector = _connected(client)

    coverage = await connector.coverage(["TEST:NOPE"])

    assert coverage["TEST:NOPE"].is_archived is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("matches", "reason"),
    [
        (
            [{"name": "TEST:PV"}, {"name": "TEST:PV2"}],
            "Archiver matched more than one channel for 'TEST:PV': ['TEST:PV', 'TEST:PV2'].",
        ),
        (
            [{"name": "TEST:OTHER"}],
            "Archiver returned the wrong channel. Requested: TEST:PV. Received: TEST:OTHER.",
        ),
    ],
    ids=["ambiguous", "wrong-channel"],
)
async def test_coverage_of_a_lookup_that_matched_otherwise_is_unknown(matches, reason):
    """A lookup that matched another channel cannot say, and names what it matched."""
    client = _fake_client()
    client.channel.Channel.return_value.matches = matches
    connector = _connected(client)

    coverage = await connector.coverage(["TEST:PV"])

    assert coverage["TEST:PV"] == ArchiveCoverage(
        address="TEST:PV", is_archived=None, unknown_reason=reason
    )


@pytest.mark.asyncio
async def test_coverage_of_a_failed_lookup_answers_failed():
    client = _fake_client()
    client.channel.Channel.return_value.run.side_effect = RuntimeError("myquery broke")
    connector = _connected(client)

    answer = await connector.coverage(["TEST:PV"])

    assert _no_answer(answer, "failed") == "the MYA channel lookup failed: RuntimeError"


@pytest.mark.asyncio
async def test_coverage_refuses_a_pattern():
    """A SQL wildcard in the address is a caller error, not a lookup miss.

    Quoting means the query itself can no longer match siblings, so the
    give-away is the `%` in the address rather than the number of matches.
    `_` cannot serve here -- it is legal in a channel name.
    """
    connector = _connected(_fake_client())

    with pytest.raises(ArchiveQueryError, match="looks like a pattern") as refused:
        await connector.coverage(["TEST:PV%"])
    assert (refused.value.parameter, refused.value.address) == ("addresses", "TEST:PV%")


@pytest.mark.asyncio
async def test_coverage_reports_each_channel():
    """The sweep reports hits and misses; coverage probing depends on it."""
    client = _fake_client()
    matches = {"TEST:PV1": [{"name": "TEST:PV1"}], "TEST:PV2": []}
    client.channel.Channel.side_effect = lambda query, url=None: MagicMock(
        matches=matches[query.pattern], run=MagicMock()
    )
    client.query.ChannelQuery.side_effect = lambda **kw: MagicMock(pattern=kw["pattern"])
    connector = _connected(client)

    coverage = await connector.coverage(["TEST:PV1", "TEST:PV2"])
    assert {address: entry.is_archived for address, entry in coverage.items()} == {
        "TEST:PV1": True,
        "TEST:PV2": False,
    }


@pytest.mark.asyncio
async def test_coverage_bounds_each_channel_not_the_whole_sweep():
    """A wide list must not time out as a unit and discard every answer.

    Each lookup spends 600 ms of a 1 s timeout, so any two outrun it together
    but none does alone, and this passes only if the bound is per channel. The
    loop's clock moves only when a lookup moves it: how long a worker thread
    waits to be scheduled on a busy runner spends none of the budget.
    """
    loop = asyncio.get_running_loop()
    now = [loop.time()]

    def lookup():
        now[0] += 0.6

    client = _fake_client()
    client.channel.Channel.side_effect = lambda query, url=None: MagicMock(
        matches=[{"name": query.pattern}], run=lookup
    )
    client.query.ChannelQuery.side_effect = lambda **kw: MagicMock(pattern=kw["pattern"])
    connector = _connected(client)
    connector._timeout = 1.0

    channels = [f"TEST:PV{n}" for n in range(10)]
    with patch.object(loop, "time", lambda: now[0]):
        coverage = await connector.coverage(channels)
    assert not isinstance(coverage, Unavailable), coverage
    assert {address: entry.is_archived for address, entry in coverage.items()} == dict.fromkeys(
        channels, True
    )


@pytest.mark.asyncio
async def test_server_side_and_client_side_bins_share_one_grid():
    """`mean` goes to mystats, `median` bins locally — both cut from the window.

    mystats is handed the window start and cuts from there. `resample` used to
    cut from midnight of the first sample's day, so one connector answered the
    same window on two grids depending on the processing mode, visible whenever
    the window opened off the midnight lattice.
    """
    start = datetime(2026, 1, 1, 0, 17, tzinfo=UTC)
    end = datetime(2026, 1, 1, 2, 17, tzinfo=UTC)
    hour_ms = 60 * 60 * 1000

    client = _fake_client()
    client.mystats.MyStats.return_value.data = _stats_frame({"TEST:PV": [1.5]}, ["mean"], at=start)
    connector = _connected(client)
    await connector.history(
        ["TEST:PV"], start_at=start, end_at=end, precision_ms=hour_ms, processing="mean"
    )

    # The server is told to cut from the window start.
    assert client.query.MyStatsQuery.call_args_list[0][1]["start"] == datetime(2026, 1, 1, 0, 17)

    client = _fake_client()
    client.interval.Interval.return_value.data = pd.Series(
        [1.0, 2.0],
        index=[
            _millis(datetime(2026, 1, 1, 0, 30, tzinfo=UTC)),
            _millis(datetime(2026, 1, 1, 1, 30, tzinfo=UTC)),
        ],
        name="TEST:PV",
    )
    connector = _connected(client)
    data = await connector.history(
        ["TEST:PV"], start_at=start, end_at=end, precision_ms=hour_ms, processing="median"
    )

    # And the local binning cuts from the same instant, not from midnight.
    assert [str(t)[11:16] for t in data["timestamp"]] == ["00:17", "01:17"]
