"""Tests for EPICS Archiver Appliance connector."""

import base64
import json
import logging
import urllib.error
import urllib.request
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from osprey_connectors.archiver import ArchiveCoverage
from osprey_connectors.archiver._epics_archiver_connector import EPICSArchiverConnector
from osprey_connectors.archiver._timerange import Processing
from osprey_connectors.errors import ArchiveQueryError, NaiveDatetimeError
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.service_answers import BackendSpec, Unavailable
from osprey_connectors.service_targets import service_targets
from osprey_connectors_tests._harness import isolated_connector_registries
from osprey_connectors_tests._loopback_https import Reply, loopback_pair


def _spec(settings: dict) -> BackendSpec:
    return BackendSpec(
        kind="archive", type="epics_archiver", target="live", settings=settings, rehearsal=False
    )


def _archive(settings: dict) -> EPICSArchiverConnector:
    """The archive *settings* configure, built as the factory builds it."""
    return EPICSArchiverConnector(_spec(settings))


def _no_answer(answer: object, cause: str) -> str:
    """*answer*'s reason, after asserting it is ``Unavailable`` with *cause*."""
    assert isinstance(answer, Unavailable), answer
    assert answer.cause == cause, answer
    return answer.reason


def _make_open_response(payload: list) -> MagicMock:
    """Return a context-manager mock that yields a file-like object with JSON payload."""
    body = json.dumps(payload).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _archiver_payload(pv: str, points: list) -> list:
    """Build a minimal archiver JSON payload for a single PV."""
    return [
        {
            "meta": {"name": pv},
            "data": [{"secs": s, "nanos": n, "val": v} for s, n, v in points],
        }
    ]


@pytest.fixture
def captured_urls():
    """Patch the connector's opener to record each request URL and answer with no data."""
    urls: list[str] = []

    def mock_open(req, timeout=None):  # noqa: ARG001 - stands in for the opener, whose caller names timeout
        urls.append(req.full_url)
        return _make_open_response([])

    with patch("urllib.request.OpenerDirector.open", side_effect=mock_open):
        yield urls


class TestBuildAndClose:
    """The archive reads its settings when it is built; there is no connect step."""

    @pytest.mark.asyncio
    async def test_build_success(self):
        """A valid block builds an archive naming the appliance."""
        connector = _archive({"url": "https://archiver.example.com"})

        assert connector._url == "https://archiver.example.com"
        assert connector.source.kind == "archive"
        assert connector.source.service == "epics_archiver"

        await connector.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("config_timeout", "expected"), [(None, 60), (120, 120)], ids=["default", "configured"]
    )
    async def test_connect_timeout(self, config_timeout, expected):
        """The timeout is 60 s unless the config names one."""
        config = {"url": "https://archiver.example.com"}
        if config_timeout is not None:
            config["timeout_s"] = config_timeout
        connector = _archive(config)

        assert connector._timeout == expected

        await connector.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", [{}, {"url": ""}], ids=["missing", "empty"])
    async def test_build_without_url_raises_value_error(self, config):
        """Building raises ValueError when the URL is missing or an empty string."""
        with pytest.raises(ValueError, match="archiver URL is required"):
            _archive(config)

    @pytest.mark.asyncio
    async def test_close_is_idempotent_and_the_archive_still_answers(self, captured_urls):
        """Nothing is held open: closing twice is safe, and a later read still asks."""
        connector = _archive({"url": "https://archiver.example.com"})

        await connector.aclose()
        await connector.aclose()
        frame = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert isinstance(frame, pd.DataFrame)
        assert len(captured_urls) == 1


class TestGetDataMethod:
    """Tests for get_data method."""

    @pytest.mark.asyncio
    async def test_get_data_single_pv_correct_values(self):
        """A single-PV fetch returns the canonical long-format DataFrame with the
        correct values, in timestamp order."""
        points = [(1704067200, 0, 1.0), (1704067201, 0, 2.0), (1704067202, 0, 3.0)]
        response = _make_open_response(_archiver_payload("PV:X", points))

        with patch("urllib.request.OpenerDirector.open", return_value=response):
            connector = _archive({"url": "https://archiver.example.com"})

            df = await connector.history(
                ["PV:X"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 1, 1, tzinfo=UTC),
            )

            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["timestamp", "address", "value"]
            assert df["timestamp"].dtype == "datetime64[ns, UTC]"
            assert df["value"].dtype == "float64"
            assert (df["address"] == "PV:X").all()
            assert list(df["value"]) == [1.0, 2.0, 3.0]

            await connector.aclose()

    @pytest.mark.asyncio
    async def test_get_data_string_valued_pv_round_trips(self):
        """An mbbi/DBR_STRING-style PV archives a string ``val``; it must come
        back unchanged, not raise."""
        points = [(1704067200, 0, "CW"), (1704067201, 0, "CW"), (1704067202, 0, "STANDBY")]
        response = _make_open_response(_archiver_payload("RF:MODE", points))

        with patch("urllib.request.OpenerDirector.open", return_value=response):
            connector = _archive({"url": "https://archiver.example.com"})

            df = await connector.history(
                ["RF:MODE"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 1, 1, tzinfo=UTC),
            )

            assert (df["address"] == "RF:MODE").all()
            assert list(df["value"]) == ["CW", "CW", "STANDBY"]

            await connector.aclose()

    @pytest.mark.asyncio
    async def test_get_data_multi_pv_returns_rows_for_each_channel(self):
        """Test that multi-PV fetch returns long-format rows for each channel."""
        points = [(1704067200 + i, 0, float(i)) for i in range(5)]

        call_count = [0]

        def mock_open(req, timeout=None):  # noqa: ARG001 - stands in for the opener, whose caller names timeout
            idx = call_count[0]
            call_count[0] += 1
            pv = "PV:1" if idx == 0 else "PV:2"
            return _make_open_response(_archiver_payload(pv, points))

        with patch("urllib.request.OpenerDirector.open", side_effect=mock_open):
            connector = _archive({"url": "https://archiver.example.com"})

            df = await connector.history(
                ["PV:1", "PV:2"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            )

            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["timestamp", "address", "value"]
            assert set(df["address"]) == {"PV:1", "PV:2"}
            assert (df["address"] == "PV:1").sum() == 5
            assert (df["address"] == "PV:2").sum() == 5

            await connector.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [[], [{"meta": {"name": "BEAM:CURRENT"}, "data": []}]],
        ids=["no_series", "series_without_data"],
    )
    async def test_get_data_empty_payload_returns_typed_empty_frame(self, payload):
        """An archiver answer of ``[]`` or ``[{meta, data: []}]`` gives the typed empty frame."""
        response = _make_open_response(payload)

        with patch("urllib.request.OpenerDirector.open", return_value=response):
            connector = _archive({"url": "https://archiver.example.com"})

            df = await connector.history(
                ["BEAM:CURRENT"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 1, 1, tzinfo=UTC),
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            assert list(df.columns) == ["timestamp", "address", "value"]
            assert df["timestamp"].dtype == "datetime64[ns, UTC]"
            assert df["value"].dtype == "float64"

            await connector.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bound,start,end",
        [
            ("start_at", "2024-01-01", datetime(2024, 1, 2, tzinfo=UTC)),
            ("end_at", datetime(2024, 1, 1, tzinfo=UTC), "2024-01-02"),
        ],
        ids=["start", "end"],
    )
    async def test_non_datetime_bound_is_rejected(self, bound, start, end):
        """get_data raises TypeError naming whichever bound is not a datetime."""
        connector = _archive({"url": "https://archiver.example.com"})

        with pytest.raises(TypeError, match=f"{bound} must be a datetime object"):
            await connector.history(
                ["BEAM:CURRENT"],
                start_at=start,
                end_at=end,
            )

        await connector.aclose()


class TestMultiPVLongFormat:
    """Each channel in a multi-PV response keeps its own real timestamps."""

    @pytest.mark.asyncio
    async def test_multi_pv_each_channel_keeps_its_own_timestamps(self):
        """Channels sampled at different instants are not forced onto a shared index."""
        base = 1704067200
        pv1_points = [(base, 0, 1.0), (base + 2, 0, 2.0)]
        pv2_points = [(base + 1, 0, 10.0), (base + 3, 0, 20.0)]

        call_count = [0]

        def mock_open(req, timeout=None):  # noqa: ARG001 - stands in for the opener, whose caller names timeout
            idx = call_count[0]
            call_count[0] += 1
            payload = (
                _archiver_payload("PV:A", pv1_points)
                if idx == 0
                else _archiver_payload("PV:B", pv2_points)
            )
            return _make_open_response(payload)

        with patch("urllib.request.OpenerDirector.open", side_effect=mock_open):
            connector = _archive({"url": "https://archiver.example.com"})

            df = await connector.history(
                ["PV:A", "PV:B"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 1, 0, 0, 10, tzinfo=UTC),
                precision_ms=1000,
            )

            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["timestamp", "address", "value"]
            assert len(df) == 4
            assert set(df["address"]) == {"PV:A", "PV:B"}

            expected_a = list(pd.to_datetime([base, base + 2], unit="s", utc=True))
            expected_b = list(pd.to_datetime([base + 1, base + 3], unit="s", utc=True))

            a_timestamps = list(df.loc[df["address"] == "PV:A", "timestamp"])
            b_timestamps = list(df.loc[df["address"] == "PV:B", "timestamp"])

            assert a_timestamps == expected_a
            assert b_timestamps == expected_b
            # No shared/aligned grid was fabricated between them.
            assert set(a_timestamps).isdisjoint(set(b_timestamps))

            await connector.aclose()


class TestGetDataErrorHandling:
    """Tests for error handling in get_data method."""

    @pytest.mark.asyncio
    async def test_url_error_answers_unreachable(self):
        """A urllib.error.URLError is the appliance not being reached."""
        with patch(
            "urllib.request.OpenerDirector.open",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            connector = _archive({"url": "https://archiver.example.com"})

            answer = await connector.history(
                ["BEAM:CURRENT"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 2, tzinfo=UTC),
            )
            reason = _no_answer(answer, "unreachable")
            assert reason.startswith("Cannot connect to archiver at https://archiver.example.com")

            await connector.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("connect_timeout", "match", "timeout"),
        [
            pytest.param(0.01, "timed out", 0.01, id="short-timeout"),
            # timeout=0 should trigger immediate timeout, not fall back to self._timeout=60
            pytest.param(60, "after 0s", 0, id="zero-is-respected-not-falsy"),
        ],
    )
    async def test_an_elapsed_timeout_answers_timeout(self, connect_timeout, match, timeout):
        """A read that outlasts its budget answers ``timeout``, naming the budget."""
        import asyncio

        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(10)

        with patch("asyncio.to_thread", side_effect=slow_fetch):
            connector = _archive(
                {"url": "https://archiver.example.com", "timeout_s": connect_timeout}
            )

            answer = await connector.history(
                ["BEAM:CURRENT"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 2, tzinfo=UTC),
                timeout_s=timeout,
            )
            assert match in _no_answer(answer, "timeout")

            await connector.aclose()

    @pytest.mark.asyncio
    async def test_connection_refused_answers_unreachable(self):
        """A ConnectionRefusedError is the appliance not being reached."""
        with patch(
            "urllib.request.OpenerDirector.open",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            connector = _archive({"url": "https://archiver.example.com"})

            answer = await connector.history(
                ["BEAM:CURRENT"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 2, tzinfo=UTC),
            )
            assert "Cannot connect to the archiver" in _no_answer(answer, "unreachable")

            await connector.aclose()

    @pytest.mark.asyncio
    async def test_generic_connection_failure_answers_unreachable(self):
        """A generic exception with 'connection' in its message is the appliance not reached."""
        with patch(
            "urllib.request.OpenerDirector.open",
            side_effect=Exception("Connection timed out: could not reach server"),
        ):
            connector = _archive({"url": "https://archiver.example.com"})

            answer = await connector.history(
                ["BEAM:CURRENT"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 2, tzinfo=UTC),
            )
            assert "Network connectivity issue" in _no_answer(answer, "unreachable")

            await connector.aclose()

    @pytest.mark.asyncio
    async def test_any_other_failure_answers_failed(self):
        """A failure none of the causes names is ``failed``, not raised."""
        with patch(
            "urllib.request.OpenerDirector.open", side_effect=Exception("the appliance broke")
        ):
            connector = _archive({"url": "https://archiver.example.com"})

            answer = await connector.history(
                ["BEAM:CURRENT"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 2, tzinfo=UTC),
            )
            assert "the appliance broke" in _no_answer(answer, "failed")

            await connector.aclose()

    @pytest.mark.asyncio
    async def test_waveform_pv_answers_unsupported(self):
        """An array-valued val (waveform PV) is a question this archive cannot answer."""
        payload = [
            {"meta": {"name": "CAM:IMAGE"}, "data": [{"secs": 1, "nanos": 0, "val": [1, 2, 3]}]}
        ]
        response = _make_open_response(payload)

        with patch("urllib.request.OpenerDirector.open", return_value=response):
            connector = _archive({"url": "https://archiver.example.com"})

            answer = await connector.history(
                ["CAM:IMAGE"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 2, tzinfo=UTC),
            )
            # The PV is named as it was asked for: wrapped in the bin's operator.
            assert _no_answer(answer, "unsupported") == (
                "Waveform PVs not supported: lastSample_1(CAM:IMAGE)"
            )

            await connector.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [{"not": "a list"}, [{"data": [{"secs": "x", "nanos": 0, "val": 1}]}]],
        ids=["not_a_list", "bad_sample"],
    )
    async def test_a_malformed_reply_answers_bad_reply(self, payload):
        """A reply that is not the retrieval format names the PV and answers ``bad_reply``."""
        with patch("urllib.request.OpenerDirector.open", return_value=_make_open_response(payload)):
            connector = _archive({"url": "https://archiver.example.com"})

            answer = await connector.history(
                ["CAM:IMAGE"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 2, tzinfo=UTC),
                precision_ms=0,
            )
            assert _no_answer(answer, "bad_reply").startswith(
                "Unexpected archiver reply for CAM:IMAGE"
            )

            await connector.aclose()


class TestCoverage:
    """What the archive says of whether a channel is archived."""

    @pytest.mark.asyncio
    async def test_coverage_does_not_claim_a_channel_archived(self, captured_urls):
        """The connector does not ask the appliance, so membership is unknown, with why."""
        connector = _archive({"url": "https://archiver.example.com"})

        channels = ["PV:1", "PV:2", "PV:3"]
        coverage = await connector.coverage(channels)

        assert list(coverage) == channels
        for pv in channels:
            assert coverage[pv] == ArchiveCoverage(
                address=pv,
                is_archived=None,
                unknown_reason=(
                    "the EPICS Archiver Appliance connector does not ask the appliance "
                    "whether this channel is archived"
                ),
            )
        assert captured_urls == []

        await connector.aclose()

    @pytest.mark.asyncio
    async def test_coverage_of_no_channel_is_empty(self):
        connector = _archive({"url": "https://archiver.example.com"})

        assert await connector.coverage([]) == {}

        await connector.aclose()

    @pytest.mark.asyncio
    async def test_the_appliance_reports_no_extent_and_has_no_probe(self, captured_urls):
        connector = _archive({"url": "https://archiver.example.com"})

        assert _no_answer(await connector.extent(), "unsupported") == (
            "this archiver does not report what it holds"
        )
        _no_answer(await connector.probe(), "unsupported")
        assert captured_urls == []


class TestBuildSideEffects:
    """Building the archive must be metadata-only: no third-party client, no stdout.

    Historical context (als-profiles GitLab #8): an earlier implementation
    constructed ``archivertools.ArchiverClient`` inside ``connect()``, whose
    ``__init__`` ICMP-pinged the server (fails in containers where ICMP is
    blocked but HTTP 17668 is open) and printed to stdout (corrupts MCP
    stdio transport). The archive now talks direct HTTP at fetch time;
    these tests pin that building it never regresses to build-time
    side effects.
    """

    @pytest.mark.asyncio
    async def test_connect_needs_no_archiver_client_library(self):
        """Building succeeds with archivertools entirely absent."""
        with patch.dict("sys.modules", {"archivertools": None}):
            connector = _archive({"url": "http://archiver.example.com:17668"})

            assert isinstance(connector, EPICSArchiverConnector)
            await connector.aclose()

    @pytest.mark.asyncio
    async def test_connect_does_not_pollute_stdout(self, capfd):
        """Nothing may print to stdout while the archive is built — MCP stdio safety.

        ``capfd`` captures file descriptor 1, so a write from a C extension or a
        subprocess is caught as well as one through ``sys.stdout``.
        """
        capfd.readouterr()

        connector = _archive({"url": "http://archiver.example.com:17668"})

        out = capfd.readouterr().out
        assert out == "", f"stdout was polluted while the archive was built: {out!r}"
        await connector.aclose()

    @pytest.mark.asyncio
    async def test_connect_performs_no_network_io(self):
        """Building must not open HTTP connections (fetch-time only)."""
        with patch(
            "urllib.request.OpenerDirector.open",
            side_effect=AssertionError("network I/O at build"),
        ):
            connector = _archive({"url": "http://archiver.example.com:17668"})

            assert isinstance(connector, EPICSArchiverConnector)
            await connector.aclose()


class TestFactoryIntegration:
    """Tests for factory integration."""

    @pytest.fixture(autouse=True)
    def setup_factory(self):
        """Register EPICS archiver connector and undo it afterward.

        These tests only need their own registration present, so the
        registries are bracketed rather than cleared: snapshot/restore drops
        this registration on teardown while leaving registrations made
        elsewhere in the process intact.
        """
        with isolated_connector_registries():
            ConnectorFactory.register_service("archive", "epics_archiver", EPICSArchiverConnector)
            yield

    @pytest.mark.asyncio
    async def test_factory_creates_epics_archiver_connector(self):
        """The factory builds the archive the config's own-name block configures."""
        config = {
            "archiver": {
                "type": "epics_archiver",
                "epics_archiver": {"url": "https://archiver.example.com", "timeout_s": 30},
            }
        }
        spec = service_targets(config).archive["live"]
        assert spec is not None

        connector = await ConnectorFactory.create_service(spec)

        assert isinstance(connector, EPICSArchiverConnector)
        assert connector._timeout == 30

        await connector.aclose()

    @pytest.mark.asyncio
    async def test_factory_with_missing_url_raises_error(self):
        """The factory propagates the archive's own ValueError for a missing URL."""
        with pytest.raises(ValueError, match="archiver URL is required"):
            await ConnectorFactory.create_service(_spec({}))


class TestQueryWindowTimezone:
    """The wire format is UTC; a caller's zone must be converted, not relabeled.

    Regression: the connector used to strftime a literal 'Z' onto the caller's
    wall-clock digits, shifting every query by the caller's UTC offset.
    """

    _LA = ZoneInfo("America/Los_Angeles")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("start", "end"),
        [
            # tz-aware, non-UTC: converted, not relabeled.
            (
                datetime(2026, 7, 30, 10, 0, 0, tzinfo=_LA),
                datetime(2026, 7, 30, 11, 0, 0, tzinfo=_LA),
            ),
            # already UTC: survives untouched.
            (
                datetime(2026, 7, 30, 17, 0, 0, tzinfo=UTC),
                datetime(2026, 7, 30, 18, 0, 0, tzinfo=UTC),
            ),
        ],
        ids=["aware_non_utc", "aware_utc"],
    )
    async def test_window_reaches_the_archiver_as_the_same_utc_instant(
        self, captured_urls, start, end
    ):
        connector = _archive({"url": "https://archiver.example.com"})

        await connector.history(["SR:DCCT"], start_at=start, end_at=end, precision_ms=1000)

        # 10:00 PDT is 17:00 UTC; 11:00 PDT is 18:00 UTC.
        assert "from=2026-07-30T17%3A00%3A00.000Z" in captured_urls[0]
        assert "to=2026-07-30T18%3A00%3A00.000Z" in captured_urls[0]

        await connector.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bound", ["start_at", "end_at"])
    async def test_a_naive_bound_is_refused_before_the_archiver_is_asked(
        self, captured_urls, bound
    ):
        """A bare wall clock names no instant; the package does not guess its zone."""
        window = {
            "start_at": datetime(2026, 7, 30, 17, 0, 0, tzinfo=UTC),
            "end_at": datetime(2026, 7, 30, 18, 0, 0, tzinfo=UTC),
        }
        window[bound] = window[bound].replace(tzinfo=None)
        connector = _archive({"url": "https://archiver.example.com"})

        with pytest.raises(NaiveDatetimeError, match=bound) as refused:
            await connector.history(["SR:DCCT"], **window, precision_ms=1000)

        assert refused.value.parameter == bound
        assert captured_urls == []
        await connector.aclose()


class TestProcessingModes:
    """The requested aggregation must reach the archiver as an operator."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("raw", "lastSample_60"),
            ("mean", "mean_60"),
            ("min", "min_60"),
            ("max", "max_60"),
            ("median", "median_60"),
            ("std", "std_60"),
            ("count", "count_60"),
        ],
    )
    async def test_mode_selects_the_archiver_operator(self, captured_urls, mode, expected):
        connector = _archive({"url": "https://archiver.example.com"})

        await connector.history(
            ["SR:DCCT"],
            start_at=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
            end_at=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
            precision_ms=60_000,
            processing=mode,
        )

        assert expected in captured_urls[0]

        await connector.aclose()

    @pytest.mark.asyncio
    async def test_unknown_mode_raises_archive_query_error(self):
        connector = _archive({"url": "https://archiver.example.com"})

        with pytest.raises(ArchiveQueryError, match="Unknown processing mode") as refused:
            await connector.history(
                ["SR:DCCT"],
                start_at=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                end_at=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                precision_ms=60_000,
                processing="p99",
            )
        assert refused.value.parameter == "processing"

        await connector.aclose()


class TestNonNumericAggregation:
    """Only ``raw`` is valid for an enum/status channel — on every backend.

    Regression: EPICS pushed the aggregation to the appliance without enforcing
    this, so a mean over a string PV returned raw values labelled as means.
    """

    @staticmethod
    def _connector_returning(payload: list):
        return patch(
            "urllib.request.OpenerDirector.open", return_value=_make_open_response(payload)
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["mean", "min", "max", "median", "std", "count"])
    async def test_aggregate_over_string_channel_raises_naming_it(self, mode):
        payload = _archiver_payload("SR:MODE", [(1000, 0, "CW"), (1060, 0, "STANDBY")])

        with self._connector_returning(payload):
            connector = _archive({"url": "https://archiver.example.com"})

            with pytest.raises(ArchiveQueryError, match="SR:MODE") as refused:
                await connector.history(
                    ["SR:MODE"],
                    start_at=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                    end_at=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                    precision_ms=60_000,
                    processing=mode,
                )
            assert refused.value.parameter == "processing"
            assert refused.value.address == "SR:MODE"

            await connector.aclose()

    @pytest.mark.asyncio
    async def test_raw_over_string_channel_is_allowed(self):
        """``raw`` never aggregates, so an enum channel passes through untouched."""
        payload = _archiver_payload("SR:MODE", [(1000, 0, "CW"), (1060, 0, "STANDBY")])

        with self._connector_returning(payload):
            connector = _archive({"url": "https://archiver.example.com"})

            df = await connector.history(
                ["SR:MODE"],
                start_at=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                end_at=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                precision_ms=60_000,
                processing="raw",
            )

            assert df["value"].tolist() == ["CW", "STANDBY"]

            await connector.aclose()

    @pytest.mark.asyncio
    async def test_empty_channel_does_not_fail_an_aggregate_query(self):
        """A channel that matched nothing has nothing to reject."""
        with self._connector_returning([]):
            connector = _archive({"url": "https://archiver.example.com"})

            df = await connector.history(
                ["SR:DCCT"],
                start_at=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                end_at=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                precision_ms=60_000,
                processing="mean",
            )

            assert df.empty

            await connector.aclose()


class TestSubSecondPrecision:
    """A bin width the appliance cannot express is rejected, not rounded.

    Regression: the width was floored with ``max(1, precision_ms // 1000)``,
    silently serving a 500 ms or 1500 ms request at 1 s.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("precision_ms", [1, 500, 999, 1500, 2500])
    async def test_non_whole_second_precision_is_rejected(self, precision_ms):
        connector = _archive({"url": "https://archiver.example.com"})

        with pytest.raises(ArchiveQueryError, match="whole number of seconds") as refused:
            await connector.history(
                ["SR:DCCT"],
                start_at=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                end_at=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                precision_ms=precision_ms,
                processing="raw",
            )
        assert refused.value.parameter == "precision_ms"

        await connector.aclose()

    @pytest.mark.asyncio
    async def test_rejection_follows_resolve_processing_rather_than_a_second_copy_of_the_rule(
        self, monkeypatch, captured_urls
    ):
        """Which widths are expressible is decided once, in ``resolve_processing``.

        The helper is forced to reject a width EPICS' own arithmetic would wave
        through; the request must be refused, not downgraded to a bare PV name.
        """
        monkeypatch.setattr(
            "osprey_connectors.archiver._epics_archiver_connector.resolve_processing",
            lambda processing, precision_ms, window_start: Processing(
                mode=processing,
                precision_ms=precision_ms,
                epics_operator=None,
                origin=pd.Timestamp(window_start),
            ),
        )

        connector = _archive({"url": "https://archiver.example.com"})

        with pytest.raises(ArchiveQueryError, match="whole number of seconds"):
            await connector.history(
                ["SR:DCCT"],
                start_at=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                end_at=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                precision_ms=60_000,
                processing="mean",
            )

        assert captured_urls == []

        await connector.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("processing", "precision_ms", "expected"),
        [
            ("mean", 1000, "mean_1"),
            ("mean", 60_000, "mean_60"),
            ("mean", 3600_000, "mean_3600"),
            # raw at a non-60s width: raw's operator width must track precision_ms.
            ("raw", 5000, "lastSample_5"),
        ],
    )
    async def test_whole_second_precision_renders_the_operator(
        self, captured_urls, processing, precision_ms, expected
    ):
        connector = _archive({"url": "https://archiver.example.com"})

        await connector.history(
            ["SR:DCCT"],
            start_at=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
            end_at=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
            precision_ms=precision_ms,
            processing=processing,
        )

        assert len(captured_urls) == 1
        assert expected in captured_urls[0]

        await connector.aclose()

    @pytest.mark.asyncio
    async def test_full_resolution_is_still_allowed(self, captured_urls):
        """precision_ms <= 0 is full resolution, not a sub-second bin width."""
        connector = _archive({"url": "https://archiver.example.com"})

        await connector.history(
            ["SR:DCCT"],
            start_at=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
            end_at=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
            precision_ms=0,
            processing="raw",
        )

        # Bare PV name, no operator wrapper.
        assert "lastSample" not in captured_urls[0]

        await connector.aclose()


class TestRetrievalPath:
    """The retrieval servlet's mount point is configurable.

    A bare appliance serves it at ``/retrieval``; a facility that fronts the
    appliance with a reverse proxy may publish it under another prefix. The
    connector must build the same request either way.
    """

    _WINDOW = {
        "start_at": datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
        "end_at": datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
    }

    @pytest.mark.asyncio
    async def test_default_is_the_appliance_root(self, captured_urls):
        """No ``retrieval_path`` means the appliance's own ``/retrieval`` servlet."""
        connector = _archive({"url": "https://archiver.example.com"})

        await connector.history(["SR:DCCT"], **self._WINDOW)

        assert captured_urls[0].startswith(
            "https://archiver.example.com/retrieval/data/getData.json?"
        )

        await connector.aclose()

    @pytest.mark.asyncio
    async def test_proxied_prefix_replaces_the_default(self, captured_urls):
        """A proxy that renames ``/retrieval`` is reached under its own prefix."""
        connector = _archive(
            {"url": "https://controls-web.example.com", "retrieval_path": "/archappl_retrieve"}
        )

        await connector.history(["SR:DCCT"], **self._WINDOW)

        assert captured_urls[0].startswith(
            "https://controls-web.example.com/archappl_retrieve/data/getData.json?"
        )
        assert "/retrieval/" not in captured_urls[0]

        await connector.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("url", "path"),
        [
            ("https://controls-web.example.com/", "/archappl_retrieve"),
            ("https://controls-web.example.com", "archappl_retrieve"),
            ("https://controls-web.example.com/", "archappl_retrieve/"),
        ],
        ids=["trailing_slash_on_url", "no_leading_slash", "both"],
    )
    async def test_slashes_are_normalized(self, captured_urls, url, path):
        """Neither a trailing slash on ``url`` nor a missing one on the path
        doubles or drops a separator."""
        connector = _archive({"url": url, "retrieval_path": path})

        await connector.history(["SR:DCCT"], **self._WINDOW)

        assert captured_urls[0].startswith(
            "https://controls-web.example.com/archappl_retrieve/data/getData.json?"
        )

        await connector.aclose()


class TestLoginAndTrust:
    """A real loopback appliance on a throwaway CA: the login and the trust reach the wire."""

    _WINDOW = {
        "start_at": datetime(2024, 1, 1, tzinfo=UTC),
        "end_at": datetime(2024, 1, 2, tzinfo=UTC),
    }
    _SECRET = "tok-3f9a-not-a-real-secret"
    _DATA = "/retrieval/data/getData.json"

    @pytest.fixture
    def appliance(self, tmp_path):
        with loopback_pair(tmp_path) as pair:
            pair.https.routes[self._DATA] = Reply(body=_archiver_payload("SR:DCCT", [(1, 0, 1.5)]))
            yield pair

    def _config(self, appliance, **extra):
        return {
            "url": appliance.https.url,
            "tls": {"ca_bundle": str(appliance.ca_pem)},
            **extra,
        }

    @pytest.mark.asyncio
    async def test_a_token_reaches_the_appliance(self, appliance, monkeypatch):
        monkeypatch.setenv("ARCHIVER_TEST_TOKEN", self._SECRET)
        connector = _archive(self._config(appliance, auth={"token_env": "ARCHIVER_TEST_TOKEN"}))

        frame = await connector.history(["SR:DCCT"], **self._WINDOW)

        assert appliance.https.authorizations() == [f"Bearer {self._SECRET}"]
        assert len(frame) == 1
        await connector.aclose()

    @pytest.mark.asyncio
    async def test_a_password_login_reaches_the_appliance_as_basic(self, appliance, monkeypatch):
        monkeypatch.setenv("ARCHIVER_TEST_PASSWORD", self._SECRET)
        connector = _archive(
            self._config(
                appliance,
                auth={"username": "reader", "password_env": "ARCHIVER_TEST_PASSWORD"},
            )
        )

        await connector.history(["SR:DCCT"], **self._WINDOW)

        (header,) = appliance.https.authorizations()
        scheme, encoded = header.split(" ", 1)
        assert scheme == "Basic"
        assert base64.b64decode(encoded).decode() == f"reader:{self._SECRET}"
        await connector.aclose()

    @pytest.mark.asyncio
    async def test_no_login_sends_no_authorization_header(self, appliance):
        connector = _archive(self._config(appliance))

        await connector.history(["SR:DCCT"], **self._WINDOW)

        assert appliance.https.seen == [(self._DATA, None)]
        await connector.aclose()

    @pytest.mark.asyncio
    async def test_every_request_goes_through_the_build_time_opener(self, appliance):
        appliance.https.routes[self._DATA] = Reply(body=[])
        with patch.object(urllib.request, "urlopen", side_effect=AssertionError("bypassed")):
            connector = _archive(self._config(appliance))
            await connector.history(["SR:DCCT", "SR:TUNE"], **self._WINDOW)
            await connector.aclose()

        assert len(appliance.https.seen) == 2

    @pytest.mark.asyncio
    async def test_an_unset_credential_variable_is_refused_at_build(self, monkeypatch):
        monkeypatch.delenv("ARCHIVER_TEST_TOKEN", raising=False)
        with pytest.raises(ConnectionError) as exc:
            _archive(
                {
                    "url": "https://archiver.example.org",
                    "auth": {"token_env": "ARCHIVER_TEST_TOKEN"},
                }
            )
        assert "archiver.settings.auth.token_env" in str(exc.value)
        assert "ARCHIVER_TEST_TOKEN" in str(exc.value)

    @pytest.mark.asyncio
    async def test_the_secret_never_reaches_the_log(self, appliance, monkeypatch, caplog):
        monkeypatch.setenv("ARCHIVER_TEST_TOKEN", self._SECRET)
        caplog.set_level(logging.DEBUG, logger="epics_archiver_connector")
        connector = _archive(self._config(appliance, auth={"token_env": "ARCHIVER_TEST_TOKEN"}))
        await connector.history(["SR:DCCT"], **self._WINDOW)
        await connector.aclose()

        assert caplog.records, "the connector logged nothing at DEBUG"
        assert self._SECRET not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", [401, 403])
    async def test_a_refused_login_names_the_key_not_reachability(
        self, appliance, monkeypatch, code
    ):
        appliance.https.routes[self._DATA] = Reply(status=code)
        monkeypatch.setenv("ARCHIVER_TEST_TOKEN", self._SECRET)
        connector = _archive(self._config(appliance, auth={"token_env": "ARCHIVER_TEST_TOKEN"}))

        message = _no_answer(await connector.history(["SR:DCCT"], **self._WINDOW), "refused")
        assert "`archiver.settings.auth.token_env`" in message
        assert f"HTTP {code}" in message
        assert "ARCHIVER_TEST_TOKEN" in message
        assert "Cannot connect" not in message
        assert "Network connectivity issue" not in message
        assert self._SECRET not in message
        await connector.aclose()

    @pytest.mark.asyncio
    async def test_an_appliance_that_asks_for_a_login_names_the_keys(self, appliance):
        appliance.https.routes[self._DATA] = Reply(status=401)
        connector = _archive(self._config(appliance))

        message = _no_answer(await connector.history(["SR:DCCT"], **self._WINDOW), "refused")
        assert "asks for a login (HTTP 401)" in message
        assert "`archiver.settings.auth` names none" in message
        assert "Network connectivity issue" not in message
        await connector.aclose()

    @pytest.mark.asyncio
    async def test_an_untrusted_certificate_names_tls_ca_bundle(self, appliance):
        connector = _archive({"url": appliance.https.url})

        answer = await connector.history(["SR:DCCT"], **self._WINDOW)
        message = _no_answer(answer, "misconfigured")
        assert "does not trust" in message
        assert "`archiver.settings.tls.ca_bundle`" in message
        assert "Network connectivity issue" not in message
        await connector.aclose()

    @pytest.mark.asyncio
    async def test_other_http_errors_keep_their_wording(self, appliance):
        appliance.https.routes.pop(self._DATA)
        connector = _archive(self._config(appliance))

        answer = await connector.history(["SR:DCCT"], **self._WINDOW)
        assert _no_answer(answer, "bad_reply").startswith("Cannot connect to archiver at")
        await connector.aclose()

    @pytest.mark.asyncio
    async def test_a_flat_timeout_is_refused_naming_timeout_s(self):
        with pytest.raises(ValueError, match=r"`archiver\.settings\.timeout_s`"):
            _archive({"url": "https://archiver.example.org", "timeout": 60})
