"""Behavioral tests for the EPICS control-system connector.

These tests drive the connector's real code paths — connect error/name-server
handling, the read path and its facility-timezone timestamp, the confirm flow
and its outcomes, the fail-closed write guard, and subscription plumbing —
with an injected fake ``aioca`` (``tests/connectors/_epics_fakes.py``) so no
real Channel Access is required. Gateway selection lives in
``test_epics_gateway_selection.py``.

Convention (matching PR #270): inject a fake client and assert on the
concrete payload — outcome word, observed value, alarm fields, env vars,
refusal reason — never merely that a call "didn't raise".
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import aioca
import numpy
import pytest

from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    WriteOutcome,
    raise_for_write_result,
)
from osprey_connectors.control_system.epics_connector import (
    EPICSConnector,
    _ChannelSubscription,
)
from tests.connectors._epics_fakes import (
    DBR_CHAR,
    DBR_DOUBLE,
    DBR_ENUM,
    DBR_LONG,
    DBR_SHORT,
    DBR_STRING,
    ECA_DISCONN,
    ECA_NOWTACCESS,
    ECA_TIMEOUT,
    FakeAioca,
    FakeChannel,
    ca_reading,
    clean_epics_env,  # noqa: F401 - fixture, used by name
    fake_aioca,  # noqa: F401 - fixture, used by name
    install_fake_aioca,
    writes_enabled,  # noqa: F401 - fixture, used by name
)
from tests.connectors._epics_fakes import (
    ca_connector as _connector,
)
from tests.connectors._epics_fakes import (
    patch_writes_enabled as _patch_writes_enabled,
)

_GATEWAYS = {"read_only": {"address": "ro", "port": 5064}}


def _channel_connector(value=None, *, ctrl=None, put=None, name="SR:CH", **kwargs):
    """A connector over one fake channel ``name`` answering with ``value``."""
    fake = FakeAioca({name: FakeChannel(value, ctrl=ctrl, put=put)})
    return _connector(aioca=fake, **kwargs)


class TestTimeoutKey:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    async def test_connect_reads_timeout_s(self, monkeypatch):
        _patch_writes_enabled(monkeypatch, False)

        connector = EPICSConnector()
        await connector.connect({"timeout_s": 7.5, "gateways": _GATEWAYS})

        assert connector._timeout == 7.5

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    async def test_connect_defaults_to_five_seconds(self, monkeypatch):
        _patch_writes_enabled(monkeypatch, False)

        connector = EPICSConnector()
        await connector.connect({"gateways": _GATEWAYS})

        assert connector._timeout == 5.0

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    async def test_the_old_timeout_key_is_refused(self, monkeypatch):
        _patch_writes_enabled(monkeypatch, False)

        connector = EPICSConnector()
        connector._connector_type = "epics"
        with pytest.raises(ValueError, match="renamed to timeout_s"):
            await connector.connect({"timeout": 7.5, "gateways": _GATEWAYS})

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    @pytest.mark.parametrize("bad", [0, -1, "five", True, float("nan"), float("inf")])
    async def test_a_timeout_s_that_is_not_a_positive_number_is_refused(self, monkeypatch, bad):
        _patch_writes_enabled(monkeypatch, False)

        connector = EPICSConnector()
        connector._connector_type = "epics"
        with pytest.raises(ValueError, match="control_system.connector.epics.timeout_s"):
            await connector.connect({"timeout_s": bad, "gateways": _GATEWAYS})

        assert connector._connected is False
        assert connector._epics_configured is False
        assert "EPICS_CA_ADDR_LIST" not in os.environ


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


class TestConnect:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_missing_aioca_raises_with_install_hint(self, monkeypatch):
        """A missing aioca raises ImportError naming the pip install command."""
        monkeypatch.setitem(sys.modules, "aioca", None)

        connector = EPICSConnector()
        with pytest.raises(ImportError, match="pip install aioca"):
            await connector.connect({"gateways": {}})

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_connect_points_ca_at_the_environment_it_routes_by(self, monkeypatch):
        """The CA loop is pointed at the EPICS_CA_* settings this connector routes by."""
        _patch_writes_enabled(monkeypatch, False)
        install_fake_aioca(monkeypatch)

        connector = EPICSConnector()
        await connector.connect({"gateways": {"read_only": {"address": "gw", "port": 5070}}})

        assert connector._ca.pointed_at == [
            {
                "EPICS_CA_ADDR_LIST": "gw",
                "EPICS_CA_SERVER_PORT": "5070",
                "EPICS_CA_NAME_SERVERS": None,
                "EPICS_CA_AUTO_ADDR_LIST": "NO",
            }
        ]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    async def test_name_server_branch_sets_and_clears_env(self, monkeypatch):
        """use_name_server routes via EPICS_CA_NAME_SERVERS and clears CA_ADDR_LIST."""
        _patch_writes_enabled(monkeypatch, False)

        connector = EPICSConnector()
        await connector.connect(
            {
                "gateways": {
                    "read_only": {
                        "address": "tunnel.example.com",
                        "port": 5074,
                        "use_name_server": True,
                    }
                }
            }
        )

        assert os.environ["EPICS_CA_NAME_SERVERS"] == "tunnel.example.com:5074"
        assert "EPICS_CA_ADDR_LIST" not in os.environ
        assert os.environ["EPICS_CA_AUTO_ADDR_LIST"] == "NO"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    @pytest.mark.parametrize("off", ["false", "False", "0", "no", ""])
    async def test_a_resolved_off_placeholder_routes_by_address_list(self, monkeypatch, off):
        """``${USE_NS:-false}`` resolves to a string, which must not pick name servers."""
        _patch_writes_enabled(monkeypatch, False)

        connector = EPICSConnector()
        await connector.connect(
            {"gateways": {"read_only": {"address": "gw", "port": 5064, "use_name_server": off}}}
        )

        assert os.environ["EPICS_CA_ADDR_LIST"] == "gw"
        assert "EPICS_CA_NAME_SERVERS" not in os.environ

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    @pytest.mark.parametrize("on", ["true", "True", "1", "yes"])
    async def test_a_resolved_on_placeholder_routes_by_name_server(self, monkeypatch, on):
        _patch_writes_enabled(monkeypatch, False)

        connector = EPICSConnector()
        await connector.connect(
            {"gateways": {"read_only": {"address": "gw", "port": 5064, "use_name_server": on}}}
        )

        assert os.environ["EPICS_CA_NAME_SERVERS"] == "gw:5064"
        assert "EPICS_CA_ADDR_LIST" not in os.environ

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    async def test_an_unreadable_use_name_server_refuses_to_connect(self, monkeypatch):
        _patch_writes_enabled(monkeypatch, False)

        connector = EPICSConnector()
        with pytest.raises(ValueError, match="use_name_server"):
            await connector.connect(
                {
                    "gateways": {
                        "read_only": {"address": "gw", "port": 5064, "use_name_server": "maybe"}
                    }
                }
            )

        assert "EPICS_CA_ADDR_LIST" not in os.environ
        assert "EPICS_CA_NAME_SERVERS" not in os.environ

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    async def test_limits_validator_initialized_when_config_present(self, monkeypatch):
        """A configured limits validator is stored on the connector after connect."""
        _patch_writes_enabled(monkeypatch, False)
        sentinel = MagicMock(name="limits_validator")
        monkeypatch.setattr(
            "osprey_connectors.control_system.limits_validator.LimitsValidator.from_config",
            classmethod(lambda cls, *, connector_type=None, target=None: sentinel),
        )

        connector = EPICSConnector()
        await connector.connect({"gateways": {"read_only": {"address": "ro", "port": 5064}}})

        assert connector._limits_validator is sentinel
        assert connector._connected is True


# ---------------------------------------------------------------------------
# read_channel error / timestamp fallback paths
# ---------------------------------------------------------------------------


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_unsubscribes_and_forgets_control_fields(self):
        """disconnect() closes subscriptions and drops the cached CTRL readings."""
        monitor = MagicMock()
        connector = _connector()
        connector._subscriptions = {"sub1": _ChannelSubscription("ca", monitor)}
        connector._ca_ctrl = {"A": object()}

        await connector.disconnect()

        monitor.close.assert_called_once()  # via unsubscribe()
        assert connector._ca_ctrl == {}
        assert connector._subscriptions == {}
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_a_ca_subscription_is_closed_on_the_ca_loop(self):
        """aioca monitors belong to the CA loop, so their close is handed to it."""
        monitor = MagicMock()
        ca_loop = MagicMock()
        subscription = _ChannelSubscription("ca", monitor, ca_loop=ca_loop)

        subscription.close()
        subscription.close()  # idempotent

        ca_loop.call_soon.assert_called_once()
        monitor.close.assert_not_called()  # the loop runs it, not this thread
        (release,) = ca_loop.call_soon.call_args.args
        release()
        monitor.close.assert_called_once_with()

    def test_the_release_forgets_the_reopener_and_closes_the_current_handle(self):
        """A repoint between close() and the release swapped the handle: the new one closes."""
        first, reopened = MagicMock(), MagicMock()
        ca_loop = MagicMock()
        reopen = MagicMock()
        subscription = _ChannelSubscription("ca", first, ca_loop=ca_loop, reopen=reopen)

        subscription.close()
        subscription.handle = reopened
        (release,) = ca_loop.call_soon.call_args.args
        release()

        ca_loop.discard_reopener.assert_called_once_with(reopen)
        reopened.close.assert_called_once_with()
        first.close.assert_not_called()


class TestReadChannel:
    @pytest.mark.asyncio
    async def test_unconnected_pv_raises_connection_error(self):
        """A channel that never connects surfaces as ConnectionError with the timeout."""
        connector = _connector(aioca=FakeAioca())

        with pytest.raises(ConnectionError, match="Failed to connect to PV 'SR:NOPE'"):
            await connector.read_channel("SR:NOPE", timeout=0.5)

    @pytest.mark.asyncio
    async def test_a_get_that_times_out_on_a_connected_channel_raises_timeout(self):
        """Connected but unanswered is a timeout, never a reading of None."""
        connector = _channel_connector(aioca.CANothing("SR:CH", ECA_TIMEOUT))

        with pytest.raises(TimeoutError, match="SR:CH"):
            await connector.read_channel("SR:CH", timeout=0.5)

    @pytest.mark.asyncio
    async def test_a_get_lost_to_a_disconnect_raises_connection_error(self):
        connector = _channel_connector(aioca.CANothing("SR:CH", ECA_DISCONN))

        with pytest.raises(ConnectionError, match="SR:CH"):
            await connector.read_channel("SR:CH", timeout=0.5)

    @pytest.mark.asyncio
    async def test_read_channel_timestamp_is_facility_tz_aware(self, monkeypatch):
        """The channel's own timestamp is rendered in the facility zone, not UTC or the box's.

        A naive ``datetime.fromtimestamp(ts)`` without a zone would fail this.
        """
        monkeypatch.setattr(
            "osprey_connectors.control_system.epics_connector.get_facility_timezone",
            lambda: ZoneInfo("Asia/Tokyo"),  # UTC+9, no DST
        )
        connector = _channel_connector(ca_reading(1.23, timestamp=1_750_000_000.0))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.timestamp.tzinfo is not None
        assert result.timestamp.utcoffset().total_seconds() == 9 * 3600
        assert result.metadata.timestamp.utcoffset().total_seconds() == 9 * 3600

    @pytest.mark.asyncio
    async def test_missing_timestamp_falls_back_to_now(self, monkeypatch):
        """When the reading carries no timestamp, the read stamps a facility-tz 'now'."""
        monkeypatch.setattr(
            "osprey_connectors.control_system.epics_connector.get_facility_timezone",
            lambda: ZoneInfo("Asia/Tokyo"),
        )
        connector = _channel_connector(ca_reading(3.14, timestamp=0))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.value == 3.14
        assert result.timestamp.tzinfo is not None
        assert result.timestamp.utcoffset().total_seconds() == 9 * 3600

    @pytest.mark.asyncio
    async def test_the_value_leaves_as_a_plain_python_type(self):
        """aioca's reading classes stay inside the connector."""
        connector = _channel_connector(ca_reading(2.5))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert type(result.value) is float
        assert result.value == 2.5

    @pytest.mark.asyncio
    async def test_an_array_leaves_as_a_plain_ndarray(self):
        import numpy

        connector = _channel_connector(ca_reading([1.0, 2.0, 3.0]))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert type(result.value) is numpy.ndarray
        assert result.value.tolist() == [1.0, 2.0, 3.0]
        assert result.metadata.raw_metadata["count"] == 3

    @pytest.mark.asyncio
    async def test_units_and_precision_come_from_the_ctrl_fields(self):
        connector = _channel_connector(
            ca_reading(1.0), ctrl=ca_reading(1.0, units="kV", precision=4)
        )

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.units == "kV"
        assert result.metadata.precision == 4
        assert result.metadata.raw_metadata["type"] == "time_double"

    @pytest.mark.asyncio
    async def test_ctrl_fields_are_fetched_once_per_channel(self):
        """They describe the record, not the reading; every read still gets the value fresh."""
        connector = _channel_connector(ca_reading(1.0))

        await connector.read_channel("SR:CH", timeout=1.0)
        await connector.read_channel("SR:CH", timeout=1.0)

        formats = [c[2]["format"] for c in connector._aioca.calls if c[0] == "caget"]
        assert formats.count(aioca.FORMAT_CTRL) == 1
        assert formats.count(aioca.FORMAT_TIME) == 2

    @pytest.mark.asyncio
    async def test_a_failed_ctrl_fetch_still_returns_the_reading_and_is_retried(self):
        connector = _channel_connector(ca_reading(4.0), ctrl=aioca.CANothing("SR:CH", ECA_TIMEOUT))

        first = await connector.read_channel("SR:CH", timeout=1.0)
        await connector.read_channel("SR:CH", timeout=1.0)

        assert first.value == 4.0
        assert first.metadata.units == ""
        formats = [c[2]["format"] for c in connector._aioca.calls if c[0] == "caget"]
        assert formats.count(aioca.FORMAT_CTRL) == 2  # not cached while it fails

    @pytest.mark.asyncio
    async def test_read_multiple_drops_failures(self, monkeypatch):
        """read_multiple_channels returns only the channels that read successfully."""
        good = ChannelValue(value=1.0, timestamp=None, metadata=ChannelMetadata())

        async def fake_read(addr, timeout=None):  # noqa: ARG001 - the control-system connector interface fixes this signature
            if addr == "BAD":
                raise ConnectionError("nope")
            return good

        connector = _connector()
        monkeypatch.setattr(connector, "read_channel", fake_read)

        result = await connector.read_multiple_channels(["GOOD", "BAD"])

        assert set(result) == {"GOOD"}
        assert result["GOOD"] is good


# ---------------------------------------------------------------------------
# write_channel — the confirm flow
# ---------------------------------------------------------------------------


def _write_connector(
    *, observed=5.0, put=None, limits=None, reading=None, ctrl=None, connected=True
):
    """A connector whose put answers ``put`` and whose channel then holds ``observed``.

    The confirming read runs for real, all the way down to the fake ``caget``,
    so the channel — never a patched ``read_channel`` — is what every
    confirm-flow test steers. ``put`` is ``None`` for an acknowledged put or an
    ECA code; ``reading`` overrides the whole TIME-form answer, an
    ``aioca.CANothing`` included.
    """
    channel = FakeChannel(
        reading if reading is not None else ca_reading(observed),
        ctrl=ctrl,
        put=put,
        connected=connected,
    )
    return _connector(aioca=FakeAioca({"SR:CH": channel}), limits_validator=limits)


def _reads_after_put(connector) -> list:
    calls = connector._aioca.calls
    put_at = next(i for i, c in enumerate(calls) if c[0] == "caput")
    return [c for c in calls[put_at:] if c[0] == "caget"]


@pytest.mark.usefixtures("writes_enabled")
class TestWriteConfirmation:
    """One confirm flow, five outcomes — the same five every connector reports.

    A write is *confirmed* when the channel it wrote now holds the value sent,
    exactly. There is no tolerance and no second verdict: the outcome word is
    the whole result, and no consumer re-derives one.
    """

    @pytest.mark.asyncio
    async def test_a_channel_holding_the_value_sent_is_confirmed(self):
        connector = _write_connector(observed=5.0)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == pytest.approx(5.0)
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_a_channel_holding_a_different_value_is_a_mismatch(self):
        """A clamped or rounded setpoint is reported, not tolerated."""
        connector = _write_connector(observed=4.7)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed_value == pytest.approx(4.7)
        # Both numbers are on the result; the raise path names them from there,
        # and an error_message here would suppress that wording.
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_a_put_whose_callback_times_out_is_unconfirmed_and_not_re_read(self):
        """The channel connected and the put went out; its callback never came.

        The outcome is unknown, not a success to go and confirm. No confirming
        read is made, because a read that raced the record's own processing
        would report whatever the channel held a moment ago as the outcome.
        """
        connector = _write_connector(put=ECA_TIMEOUT)

        result = await connector.write_channel("SR:CH", 5.0, confirm=True)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.observed_value is None
        assert "did not acknowledge" in result.error_message
        assert _reads_after_put(connector) == []

    @pytest.mark.asyncio
    async def test_a_confirming_read_that_raises_is_unconfirmed(self):
        """The value was sent; what the channel holds is unknown, not wrong."""
        connector = _write_connector(reading=TimeoutError("ca timeout"))

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.observed_value is None
        assert "ca timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_a_confirming_read_that_times_out_is_unconfirmed_not_a_mismatch(self):
        """An unanswered get is a timeout, never an observation of ``None``."""
        connector = _write_connector(reading=aioca.CANothing("SR:CH", ECA_TIMEOUT))

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.observed_value is None
        assert "timed out" in result.error_message

    @pytest.mark.asyncio
    async def test_a_channel_that_never_connects_is_failed_and_nothing_is_sent(self):
        """aioca reports a connect timeout and a put-callback timeout with one code.

        Connecting first is what tells them apart: here nothing left the
        process, so the write failed outright and nothing is left to confirm.
        """
        connector = _write_connector(connected=False)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.FAILED
        assert "no value was sent" in result.error_message
        assert result.observed_value is None
        assert connector._aioca.puts == []

    @pytest.mark.asyncio
    async def test_a_put_denied_by_access_security_is_refused(self):
        connector = _write_connector(put=ECA_NOWTACCESS)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "CONTROL_SYSTEM_REFUSED"
        assert "Write access denied" in result.error_message
        assert _reads_after_put(connector) == []

    @pytest.mark.asyncio
    async def test_any_other_put_failure_is_raised_as_unknown(self):
        """A lost circuit mid-put leaves the outcome unknown; no outcome word claims it."""
        connector = _write_connector(put=ECA_DISCONN)

        with pytest.raises(ConnectionError, match="SR:CH"):
            await connector.write_channel("SR:CH", 5.0)

    @pytest.mark.asyncio
    async def test_confirm_false_checks_nothing(self):
        connector = _write_connector(observed=9.9)

        result = await connector.write_channel("SR:CH", 5.0, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.observed_value is None
        assert result.error_message is None
        # Not even a read: the channel disagreeing is not this write's verdict.
        assert _reads_after_put(connector) == []

    @pytest.mark.asyncio
    async def test_a_mismatch_names_both_values_for_display(self):
        """``notes`` is display-only — nothing classifies a write by parsing it."""
        connector = _write_connector(observed=4.7)

        result = await connector.write_channel("SR:CH", 5.0)

        assert "4.7" in result.notes
        assert "5.0" in result.notes


def _typed_write_connector(datatype, *, observed=5, count=None):
    """A write connector whose channel is of field type ``datatype``.

    ``cainfo`` answers the field type and element count of the channel's
    reading, which is also what the confirming read returns.
    """
    reading = ca_reading(observed, datatype=datatype, count=count)
    return _connector(aioca=FakeAioca({"SR:CH": FakeChannel(reading)}))


@pytest.mark.usefixtures("writes_enabled")
class TestWriteExactness:
    """A value goes out exactly as written, or not at all.

    The IOC converts a number into the channel's field type without checking
    it, so a soft IOC stored 1.5 sent to a ``longout`` as 1 and 2**31 as
    -2147483648, and with confirmation off nothing reported either. Such a
    write is refused before ``caput``.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("confirm", [True, False])
    @pytest.mark.parametrize(
        ("datatype", "value", "reason"),
        [
            (DBR_LONG, 1.5, "not a whole number"),
            (DBR_LONG, "1.9", "not a numeral an integer field reads"),
            (DBR_LONG, 2**31, "outside the channel's integer range"),
            (DBR_LONG, "2147483648", "outside the channel's integer range"),
            (DBR_SHORT, 40000, "outside the channel's integer range"),
            (DBR_CHAR, 300, "outside the channel's integer range"),
            (DBR_CHAR, -1, "outside the channel's integer range"),
            (DBR_LONG, [1.5, 2], "not a whole number"),
            (DBR_LONG, [2**31], "outside the channel's integer range"),
            (DBR_LONG, numpy.array([1.0, 2.5]), "not a whole number"),
            (DBR_LONG, float("nan"), "not a whole number"),
            (DBR_LONG, float("inf"), "not a whole number"),
            (DBR_ENUM, 1.5, "not a whole number"),
            (DBR_ENUM, -1, "outside the channel's integer range"),
        ],
    )
    async def test_a_value_the_integer_channel_cannot_hold_is_refused_unsent(
        self, datatype, value, reason, confirm
    ):
        connector = _typed_write_connector(datatype)

        result = await connector.write_channel("SR:CH", value, confirm=confirm)

        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "VALIDATION_ERROR"
        assert reason in result.error_message
        assert connector._aioca.puts == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("datatype", "value", "sent"),
        [
            (DBR_LONG, 2.0, 2),
            (DBR_LONG, "0x10", 16),
            (DBR_LONG, -(2**31), -(2**31)),
            (DBR_SHORT, True, 1),
            (DBR_CHAR, 255, 255),
            (DBR_LONG, [2.0, 3], [2, 3]),
            (DBR_LONG, numpy.array([2.0, 3.0]), [2, 3]),
            (DBR_ENUM, 1, 1),
            (DBR_DOUBLE, 1.5, 1.5),
        ],
    )
    async def test_a_value_the_channel_holds_exactly_is_sent(self, datatype, value, sent):
        """A whole float bound for an integer field goes out as the integer it is."""
        connector = _typed_write_connector(datatype)

        await connector.write_channel("SR:CH", value, confirm=False)

        [(_name, put_value, _wait)] = connector._aioca.puts
        assert put_value == sent
        assert type(put_value) is type(sent)

    @pytest.mark.asyncio
    async def test_text_for_a_char_array_is_sent_as_a_string(self):
        connector = _typed_write_connector(DBR_CHAR, observed=[0] * 32)

        await connector.write_channel("SR:CH", "text for a char array", confirm=False)

        [(_name, put_value, _wait)] = connector._aioca.puts
        assert put_value == "text for a char array\0"
        caput = next(call for call in connector._aioca.calls if call[0] == "caput")
        assert caput[2]["datatype"] == FakeAioca.DBR_CHAR_STR

    @pytest.mark.asyncio
    async def test_a_channel_that_never_connects_fails_with_nothing_sent(self):
        connector = _write_connector(connected=False)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.FAILED
        assert "no value was sent" in result.error_message
        assert connector._aioca.puts == []

    @pytest.mark.asyncio
    async def test_text_a_char_array_holds_is_confirmed(self):
        """Text goes to a char array as its bytes plus a NUL, and a confirming
        read returns those bytes, never the text."""
        held = [ord(c) for c in "hello"] + [0, 0]
        connector = _typed_write_connector(DBR_CHAR, observed=held)

        result = await connector.write_channel("SR:CH", "hello")

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == "hello"

    @pytest.mark.asyncio
    async def test_text_a_char_array_cut_short_is_a_mismatch(self):
        held = [ord(c) for c in "hel"] + [0, 0, 0, 0]
        connector = _typed_write_connector(DBR_CHAR, observed=held)

        result = await connector.write_channel("SR:CH", "hello")

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed_value == "hel"


@pytest.mark.usefixtures("writes_enabled")
class TestConfirmingRead:
    """What the confirming read and the put around it do."""

    @pytest.mark.asyncio
    async def test_the_confirming_read_goes_to_the_ioc(self):
        """The read after the put is a get of its own, never a cached answer."""
        connector = _write_connector(observed=5.0)

        await connector.write_channel("SR:CH", 5.0)

        assert [c[2]["format"] for c in _reads_after_put(connector)][0] == aioca.FORMAT_TIME

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    @pytest.mark.parametrize(("value", "expected"), [("false", False), ("True", True)])
    async def test_fresh_reads_reads_a_resolved_placeholder_for_what_it_spells(
        self, monkeypatch, value, expected
    ):
        """``${VAR:-false}`` resolves to the string ``"false"``; the key still loads."""
        _patch_writes_enabled(monkeypatch, False)
        connector = EPICSConnector()
        await connector.connect({"fresh_reads": value, "gateways": _GATEWAYS})

        assert connector._fresh_reads is expected

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
    async def test_fresh_reads_refuses_text_that_spells_neither_on_nor_off(self, monkeypatch):
        """A typo is refused rather than read as off, though the key now changes nothing."""
        _patch_writes_enabled(monkeypatch, False)
        connector = EPICSConnector()
        connector._connector_type = "epics"

        with pytest.raises(ValueError, match=r"control_system\.connector\.epics\.fresh_reads"):
            await connector.connect(
                {"fresh_reads": "ture", "gateways": {"read_only": {"address": "ro", "port": 5064}}}
            )

    @pytest.mark.asyncio
    async def test_a_confirming_put_waits_for_the_ioc_callback(self):
        """Put-callback is the protocol's acknowledgement that the put landed."""
        connector = _write_connector(observed=5.0)

        await connector.write_channel("SR:CH", 5.0)

        assert connector._aioca.puts == [("SR:CH", 5.0, True)]

    @pytest.mark.asyncio
    async def test_an_unconfirmed_put_does_not_wait(self):
        connector = _write_connector(observed=5.0)

        await connector.write_channel("SR:CH", 5.0, confirm=False)

        assert connector._aioca.puts == [("SR:CH", 5.0, False)]

    @pytest.mark.asyncio
    async def test_the_put_connects_first(self):
        connector = _write_connector(observed=5.0)

        await connector.write_channel("SR:CH", 5.0)

        names = [c[0] for c in connector._aioca.calls]
        assert names.index("connect") < names.index("caput")

    @pytest.mark.asyncio
    async def test_an_enum_label_written_as_text_is_confirmed_by_its_index(self):
        """An mbbo takes "ON" and reads back 1; that is the same state.

        EPICS is the only connector that reports an ``enum_label``, and this is
        what it is for: without it the comparison would see ``"ON" != 1`` and
        report a mismatch on a write the machine took exactly as sent.
        """
        connector = _write_connector(
            reading=ca_reading(1, datatype=DBR_ENUM), ctrl=ca_reading(1, enums=("OFF", "ON"))
        )

        result = await connector.write_channel("SR:CH", "ON")

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == 1

    @pytest.mark.asyncio
    async def test_the_observed_value_keeps_the_type_the_channel_reports(self):
        """A string reading stays a string — nothing is coerced into a number."""
        connector = _write_connector(reading=ca_reading("ON", datatype=DBR_STRING))

        result = await connector.write_channel("SR:CH", "ON")

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == "ON"
        assert result.observed_number is None


@pytest.mark.usefixtures("writes_enabled")
class TestConfirmResolution:
    """``confirm=None`` is "no opinion", and never means ``False``."""

    @pytest.mark.asyncio
    async def test_an_omitted_confirm_takes_the_limits_database_policy(self):
        limits = MagicMock()
        limits.resolve_confirm.return_value = False
        connector = _write_connector(observed=9.9, limits=limits)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.UNREQUESTED
        limits.resolve_confirm.assert_called_once_with("SR:CH")

    @pytest.mark.asyncio
    async def test_a_connector_without_a_validator_confirms(self):
        """No limits database means no policy to read — the fleet default holds."""
        connector = _write_connector(observed=5.0, limits=None)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.CONFIRMED

    @pytest.mark.asyncio
    async def test_an_explicit_confirm_false_is_an_answer_not_an_omission(self):
        """A declined confirmation must not be re-resolved back into a check."""
        limits = MagicMock()
        limits.resolve_confirm.return_value = True
        connector = _write_connector(observed=5.0, limits=limits)

        result = await connector.write_channel("SR:CH", 5.0, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        limits.resolve_confirm.assert_not_called()


# ---------------------------------------------------------------------------
# write_channel — fail-closed guard
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("writes_enabled")
class TestWriteFailClosed:
    @pytest.mark.asyncio
    async def test_validation_error_refuses_write_without_caput(self):
        """A non-limits validation error fails closed: refused, and no caput."""
        limits = MagicMock()
        limits.validate.side_effect = RuntimeError("db unreadable")
        connector = _write_connector(limits=limits)

        result = await connector.write_channel("SR:CH", 1.0, confirm=False)

        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "VALIDATION_ERROR"
        assert result.error_message is not None
        assert connector._aioca.puts == []

    @pytest.mark.asyncio
    async def test_limits_violation_propagates(self):
        """A ChannelLimitsViolationError from validate is raised, not swallowed."""
        from osprey_connectors.errors import ChannelLimitsViolationError

        limits = MagicMock()
        limits.validate.side_effect = ChannelLimitsViolationError(
            channel_address="SR:CH",
            value=1.0,
            violation_type="MAX_EXCEEDED",
            violation_reason="too big",
        )
        connector = _write_connector(limits=limits)

        with pytest.raises(ChannelLimitsViolationError):
            await connector.write_channel("SR:CH", 1.0, confirm=False)

        assert connector._aioca.puts == []


# ---------------------------------------------------------------------------
# The max_step reader
# ---------------------------------------------------------------------------


class TestCurrentValueReader:
    @pytest.mark.asyncio
    async def test_reads_the_channel_as_a_plain_value(self):
        connector = _channel_connector(ca_reading(3.5))
        read_current = connector._current_value_reader()

        assert await asyncio.to_thread(read_current, "SR:CH") == 3.5

    @pytest.mark.asyncio
    async def test_a_channel_that_does_not_answer_reads_as_none(self):
        """``None`` fails the step check closed."""
        connector = _connector(aioca=FakeAioca())
        read_current = connector._current_value_reader()

        assert await asyncio.to_thread(read_current, "SR:NOPE") is None

    @pytest.mark.asyncio
    async def test_the_read_uses_the_step_read_timeout(self):
        connector = _channel_connector(ca_reading(3.5))
        connector._step_read_timeout = 0.75
        read_current = connector._current_value_reader()

        await asyncio.to_thread(read_current, "SR:CH")

        assert connector._aioca.calls[-1] == (
            "caget",
            "SR:CH",
            {"format": aioca.FORMAT_RAW, "timeout": 0.75, "datatype": None},
        )


# ---------------------------------------------------------------------------
# subscribe / unsubscribe / validate_channel / get_metadata
# ---------------------------------------------------------------------------


async def _subscribed(connector, name="SR:CH"):
    """Subscribe, and return (received list, the fake monitor's callback)."""
    received: list = []
    await connector.subscribe(name, received.append)
    monitor = connector._aioca.monitors[-1]
    return received, monitor.callback


class TestSubscribe:
    @pytest.mark.asyncio
    async def test_a_monitor_update_converts_to_channel_value(self, monkeypatch):
        """aioca's monitor update is adapted into a facility-tz ChannelValue."""
        monkeypatch.setattr(
            "osprey_connectors.control_system.epics_connector.get_facility_timezone",
            lambda: ZoneInfo("Asia/Tokyo"),
        )
        connector = _channel_connector(ctrl=ca_reading(0.0, units="A"))
        received, fire = await _subscribed(connector)
        await asyncio.sleep(0)  # let the background CTRL fetch land

        fire(ca_reading(7.0, timestamp=1_750_000_000.0))
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert len(received) == 1
        assert received[0].value == 7.0
        assert received[0].metadata.units == "A"
        assert received[0].timestamp.utcoffset().total_seconds() == 9 * 3600

    @pytest.mark.asyncio
    async def test_the_monitor_asks_for_timestamps_and_disconnect_notices(self):
        connector = _channel_connector()

        await connector.subscribe("SR:CH", lambda _value: None)

        call = next(c for c in connector._aioca.calls if c[0] == "camonitor")
        assert call[2] == {"format": aioca.FORMAT_TIME, "notify_disconnect": True}

    @pytest.mark.asyncio
    async def test_a_disconnect_notice_never_reaches_the_subscriber(self):
        connector = _channel_connector()
        received, fire = await _subscribed(connector)

        fire(aioca.CANothing("SR:CH", ECA_DISCONN))
        await asyncio.sleep(0.01)

        assert received == []

    @pytest.mark.asyncio
    async def test_unsubscribe_closes_the_monitor(self):
        connector = _channel_connector()
        sub_id = await connector.subscribe("SR:CH", lambda _value: None)

        await connector.unsubscribe(sub_id)

        assert connector._aioca.monitors[-1].closed is True
        assert sub_id not in connector._subscriptions

    def test_an_update_after_the_subscribers_loop_closed_closes_the_monitor(self):
        """osprey.runtime's loop ends with its call; aioca must not be left to print."""
        connector = _channel_connector()
        asyncio.run(connector.subscribe("SR:CH", lambda _value: None))
        monitor = connector._aioca.monitors[-1]

        monitor.callback(ca_reading(7.0))  # raising here is what aioca prints

        assert monitor.closed is True
        # Nor does a later repoint bring it back for a loop that is gone.
        connector._ca.repoint()
        assert connector._aioca.monitors == [monitor]


class TestSubscriptionAcrossARepoint:
    """aioca closes every monitor on a repoint without telling its callback."""

    @pytest.mark.asyncio
    async def test_a_repoint_reopens_the_monitor_and_updates_keep_arriving(self):
        connector = _channel_connector()
        received, _fire = await _subscribed(connector)
        first = connector._aioca.monitors[-1]

        connector._ca.repoint()

        reopened = connector._aioca.monitors[-1]
        assert reopened is not first
        assert (reopened.name, reopened.callback) == ("SR:CH", first.callback)
        call = [c for c in connector._aioca.calls if c[0] == "camonitor"][-1]
        assert call[2] == {"format": aioca.FORMAT_TIME, "notify_disconnect": True}
        reopened.callback(ca_reading(3.0))
        await asyncio.sleep(0.01)
        assert [value.value for value in received] == [3.0]

    @pytest.mark.asyncio
    async def test_unsubscribe_after_a_repoint_closes_the_reopened_monitor(self):
        connector = _channel_connector()
        sub_id = await connector.subscribe("SR:CH", lambda _value: None)
        connector._ca.repoint()
        reopened = connector._aioca.monitors[-1]

        await connector.unsubscribe(sub_id)

        assert reopened.closed is True
        assert connector._ca.reopeners == []
        connector._ca.repoint()  # an unsubscribed monitor stays closed
        assert connector._aioca.monitors[-1] is reopened

    @pytest.mark.asyncio
    async def test_each_subscription_is_reopened(self):
        connector = _channel_connector()
        await connector.subscribe("SR:CH", lambda _value: None)
        await connector.subscribe("SR:OTHER", lambda _value: None)

        connector._ca.repoint()

        assert [m.name for m in connector._aioca.monitors] == [
            "SR:CH",
            "SR:OTHER",
            "SR:CH",
            "SR:OTHER",
        ]


class _StalledCaLoop:
    """A CA loop whose answer never comes back: the caller is left to give up.

    ``opens_first`` decides whether the queued open runs before the caller
    gives up or only after its release has run.
    """

    def __init__(self, inline, *, opens_first: bool):
        self._inline = inline
        self._opens_first = opens_first
        self.queued = []

    async def run(self, coro):
        if self._opens_first:
            await coro
        else:
            self.queued.append(coro)
        await asyncio.Event().wait()

    def call_soon(self, callback, *args):
        callback(*args)

    def add_reopener(self, reopen):
        self._inline.add_reopener(reopen)

    def discard_reopener(self, reopen):
        self._inline.discard_reopener(reopen)


class TestSubscribeCancelled:
    """A caller that bounds the open must not leave a monitor behind it."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("opens_first", [True, False])
    async def test_no_monitor_is_left_open(self, opens_first):
        connector = _channel_connector()
        reopeners = connector._ca
        connector._ca = _StalledCaLoop(reopeners, opens_first=opens_first)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(connector.subscribe("SR:CH", lambda _value: None), 0.05)
        for coro in connector._ca.queued:  # the open the loop ran late
            await coro

        # Opened before the caller gave up, it is closed; queued, it never opens.
        assert [m.closed for m in connector._aioca.monitors] == ([True] if opens_first else [])
        assert reopeners.reopeners == []
        assert connector._subscriptions == {}


class TestValidateChannelAndMetadata:
    @pytest.mark.asyncio
    async def test_get_metadata_returns_read_metadata(self, monkeypatch):
        meta = ChannelMetadata(units="kV")
        value = ChannelValue(value=1.0, timestamp=None, metadata=meta)
        connector = _connector()
        monkeypatch.setattr(connector, "read_channel", AsyncMock(return_value=value))

        assert await connector.get_metadata("SR:CH") is meta

    @pytest.mark.asyncio
    async def test_validate_channel_probes_with_the_configured_timeout(self, monkeypatch):
        """The probe uses the connector's own ``timeout``, not a literal of its own."""
        value = ChannelValue(value=1.0, timestamp=None, metadata=ChannelMetadata())
        read = AsyncMock(return_value=value)
        connector = _connector(timeout=17.0)
        monkeypatch.setattr(connector, "read_channel", read)

        assert await connector.validate_channel("SR:CH") is True
        assert read.await_args.kwargs["timeout"] == 17.0

    @pytest.mark.asyncio
    async def test_validate_channel_false_on_read_error(self, monkeypatch):
        connector = _connector()
        monkeypatch.setattr(
            connector, "read_channel", AsyncMock(side_effect=ConnectionError("no route"))
        )

        assert await connector.validate_channel("SR:CH") is False


# ---------------------------------------------------------------------------
# Channel Access alarm names (read + subscribe)
# ---------------------------------------------------------------------------


class TestChannelAccessAlarmNames:
    """CA reports alarm status as an int; the connector emits the EPICS name.

    ``ChannelMetadata.alarm_status`` is declared ``str | None`` and PVAccess
    already emitted names, so the raw CA code was both the wrong type and
    unreadable downstream. The code itself is not lost — it stays in
    ``raw_metadata["status"]`` next to the severity.
    """

    @pytest.mark.asyncio
    async def test_read_reports_alarm_name_not_code(self):
        connector = _channel_connector(ca_reading(1.0, status=3, severity=2))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == "HIHI"  # not the integer 3

    @pytest.mark.asyncio
    async def test_read_reports_healthy_alarm_by_name(self):
        connector = _channel_connector(ca_reading(1.0, status=0, severity=0))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == "NO_ALARM"

    @pytest.mark.asyncio
    async def test_read_keeps_the_raw_code_beside_the_severity(self):
        connector = _channel_connector(ca_reading(1.0, status=5, severity=1))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == "LOLO"
        assert result.metadata.raw_metadata["status"] == 5
        assert result.metadata.raw_metadata["severity"] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("code", "name"), [(11, "HW_LIMIT"), (17, "UDF"), (21, "WRITE_ACCESS")]
    )
    async def test_the_names_match_epics_alarm_h(self, code, name):
        connector = _channel_connector(ca_reading(1.0, status=code, severity=1))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == name

    @pytest.mark.asyncio
    async def test_unmappable_code_reads_as_unknown(self):
        """An out-of-range code must not raise — it degrades to UNKNOWN."""
        connector = _channel_connector(ca_reading(1.0, status=99, severity=3))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == "UNKNOWN"
        assert result.metadata.raw_metadata["status"] == 99  # raw code still recorded

    @pytest.mark.asyncio
    async def test_subscribe_callback_reports_alarm_name(self, monkeypatch):
        """The monitor path maps codes exactly like the read path."""
        monkeypatch.setattr(
            "osprey_connectors.control_system.epics_connector.get_facility_timezone",
            lambda: ZoneInfo("UTC"),
        )
        connector = _channel_connector()
        received, fire = await _subscribed(connector)

        fire(ca_reading(7.0, status=4, severity=1))
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert received[0].metadata.alarm_status == "HIGH"
        assert received[0].metadata.alarm_severity == 1
        assert received[0].metadata.raw_metadata["status"] == 4
        assert received[0].metadata.raw_metadata["severity"] == 1


class TestReadAlarmSeverity:
    """Reads carry the typed severity, not just the alarm name.

    ``ChannelMetadata.alarm_severity`` follows the write-path convention
    (``ChannelWriteResult.alarm_severity``): 0 healthy, higher is worse,
    ``None`` = the control system reported no severity. Before this field the
    severity reached only ``raw_metadata``, which the ``channel_read`` tool
    never surfaces — so the agent could see MAJOR-vs-MINOR after a write but
    not on the read it based the write on.
    """

    @pytest.mark.asyncio
    async def test_read_reports_the_typed_severity(self):
        connector = _channel_connector(ca_reading(1.0, status=3, severity=2))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_severity == 2

    @pytest.mark.asyncio
    async def test_a_reported_healthy_severity_stays_zero(self):
        connector = _channel_connector(ca_reading(1.0, status=0, severity=0))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_severity == 0  # not None: healthy is a report

    @pytest.mark.asyncio
    async def test_an_unreported_severity_reads_as_none(self):
        connector = _channel_connector(ca_reading(1.0, status=0, severity=None))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_severity is None


# ---------------------------------------------------------------------------
# Channel Access enum labels (read + subscribe)
# ---------------------------------------------------------------------------

_MODES = ("OFFLINE", "STANDBY", "ACQUIRING", "FAULT")


def _enum_connector(value=2, labels=_MODES):
    """A connector over an mbbi: an index, and the labels its CTRL fields carry."""
    return _channel_connector(
        ca_reading(value, datatype=DBR_ENUM),
        ctrl=ca_reading(0, datatype=DBR_ENUM, enums=labels),
        name="SR:MODE",
    )


class TestChannelAccessEnumLabels:
    """An mbbi/bi/bo read answers with its index *and* the state that index means.

    The index stays the value — the machine-readable half, and the same type
    PVAccess reports for the same record — so the labels are carried beside it
    rather than in place of it. They come from the channel's CTRL fields, a
    round trip of its own, so every failure of that fetch degrades to "no
    labels" and never to a failed read: a reading with an index and no label is
    still a correct answer, a raised read is not.
    """

    @pytest.mark.asyncio
    async def test_enum_read_reports_the_index_and_its_labels(self):
        connector = _enum_connector(value=2)

        result = await connector.read_channel("SR:MODE", timeout=1.0)

        assert result.value == 2  # the index, not "ACQUIRING"
        assert result.metadata.enum_label == "ACQUIRING"
        assert result.metadata.enum_labels == ["OFFLINE", "STANDBY", "ACQUIRING", "FAULT"]
        assert result.metadata.raw_metadata["type"] == "time_enum"

    @pytest.mark.asyncio
    async def test_index_zero_resolves_to_its_label_not_to_nothing(self):
        """A bi at 0 is a state, not a falsy miss."""
        connector = _enum_connector(value=0, labels=("OFF", "ON"))

        result = await connector.read_channel("SR:MODE", timeout=1.0)

        assert result.value == 0
        assert result.metadata.enum_label == "OFF"

    @pytest.mark.asyncio
    async def test_a_non_enum_read_leaves_both_fields_unset(self):
        """The fields are how a consumer tells an enum channel from a numeric one."""
        connector = _channel_connector(ca_reading(7.25))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.value == 7.25
        assert result.metadata.enum_labels is None
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_a_failed_label_fetch_still_returns_the_reading(self):
        """The CTRL fetch is a round trip to the IOC, and it is allowed to fail."""
        connector = _channel_connector(
            ca_reading(2, datatype=DBR_ENUM),
            ctrl=aioca.CANothing("SR:MODE", ECA_TIMEOUT),
            name="SR:MODE",
        )

        result = await connector.read_channel("SR:MODE", timeout=1.0)

        assert result.value == 2  # the read is not lost with the labels
        assert result.metadata.enum_labels is None
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_unreported_labels_leave_the_fields_unset(self):
        connector = _channel_connector(
            ca_reading(2, datatype=DBR_ENUM), ctrl=ca_reading(0), name="SR:MODE"
        )

        result = await connector.read_channel("SR:MODE", timeout=1.0)

        assert result.value == 2
        assert result.metadata.enum_labels is None
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_an_index_past_the_label_list_keeps_the_list(self):
        """An unresolvable index loses its label, not the states it could not name."""
        connector = _enum_connector(value=9, labels=("OFF", "ON"))

        result = await connector.read_channel("SR:MODE", timeout=1.0)

        assert result.value == 9
        assert result.metadata.enum_labels == ["OFF", "ON"]
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_subscribe_callback_reports_the_label_once_ctrl_fields_arrive(self, monkeypatch):
        monkeypatch.setattr(
            "osprey_connectors.control_system.epics_connector.get_facility_timezone",
            lambda: ZoneInfo("UTC"),
        )
        connector = _enum_connector()
        received, fire = await _subscribed(connector, "SR:MODE")
        await asyncio.sleep(0)  # the background CTRL fetch

        fire(ca_reading(3, datatype=DBR_ENUM))
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert received[0].value == 3
        assert received[0].metadata.enum_label == "FAULT"
        assert received[0].metadata.enum_labels == list(_MODES)

    @pytest.mark.asyncio
    async def test_an_update_before_the_labels_arrive_is_delivered_anyway(self):
        """An update that beats the CTRL fetch carries the index alone."""
        connector = _enum_connector()
        received, fire = await _subscribed(connector, "SR:MODE")

        fire(ca_reading(1, datatype=DBR_ENUM))  # before the fetch has run
        await asyncio.sleep(0.01)

        assert received[0].value == 1
        assert received[0].metadata.enum_label is None
        assert received[0].metadata.enum_labels is None


# ---------------------------------------------------------------------------
# write_channel — alarm state reported by the confirming read
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("writes_enabled")
class TestConfirmingReadAlarmReporting:
    """The confirming read carries the channel's alarm state into the result.

    Alarm state is *information* on a write — reported beside the outcome,
    never a reason to raise and never a reason to withhold a confirmation.
    ``None`` means "not reported" and stays distinct from a reported healthy
    channel (severity ``0``).
    """

    @pytest.mark.asyncio
    async def test_healthy_confirming_read_reports_severity_zero(self):
        connector = _write_connector(reading=ca_reading(5.0, status=0, severity=0))

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.alarm_status == "NO_ALARM"
        assert result.alarm_severity == 0
        assert result.alarm_severity is not None

    @pytest.mark.asyncio
    async def test_a_confirmed_write_reports_a_major_alarm_and_still_returns(self):
        """The channel took the value; that it is also in alarm is a second fact.

        Both facts are needed, and neither replaces the other — so the raise
        path returns a confirmed result whatever its alarm severity.
        """
        connector = _write_connector(reading=ca_reading(5.0, status=3, severity=2))

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.alarm_status == "HIHI"
        assert result.alarm_severity == 2
        assert raise_for_write_result(result) is result

    @pytest.mark.asyncio
    async def test_a_mismatch_carries_the_alarm_state_too(self):
        connector = _write_connector(reading=ca_reading(9.9, status=3, severity=2))

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.alarm_status == "HIHI"
        assert result.alarm_severity == 2

    @pytest.mark.asyncio
    async def test_an_unconfirmed_write_claims_no_alarm_state(self):
        """Nothing was read, so no alarm state can be claimed."""
        connector = _write_connector(reading=TimeoutError("ca timeout"))

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.alarm_status is None
        assert result.alarm_severity is None

    @pytest.mark.asyncio
    async def test_notes_text_never_feeds_the_structured_fields(self):
        """Notes and messages are display-only: their wording moves no field.

        The exception message is echoed into ``error_message``; wording it to
        look like a healthy confirmed reading must not change the outcome or
        manufacture an alarm state.
        """
        connector = _write_connector(
            reading=TimeoutError("confirmed NO_ALARM severity 0 value 5.0")
        )

        result = await connector.write_channel("SR:CH", 5.0)

        assert "confirmed NO_ALARM" in result.error_message  # the wording did land
        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.alarm_status is None
        assert result.alarm_severity is None
        assert result.observed_value is None
