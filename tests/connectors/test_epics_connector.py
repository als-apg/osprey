"""Behavioral tests for the EPICS control-system connector.

These tests drive the connector's real code paths — connect() and its client
import, the Channel Access read and its cached display metadata, the
facility-timezone timestamp, the confirm flow and its outcomes, the confirming
put's deadline, the fail-closed write guard, and subscription plumbing — over
a fake ``pvaccess`` module (``tests/connectors/_epics_fakes.py``), so no real
Channel Access is required. Gateway selection lives in
``test_epics_gateway_selection.py``; the same paths against real soft IOCs live
in ``test_epics_soft_ioc.py``.

Convention (matching PR #270): inject a fake client and assert on the concrete
payload — outcome word, observed value, alarm fields, env vars, the pvRequest
a call carried, refusal reason — never merely that a call "didn't raise".
"""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from osprey.connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    WriteOutcome,
    raise_for_write_result,
)
from osprey.connectors.control_system.epics_connector import EPICSConnector
from tests.connectors._epics_fakes import (
    CA_DISPLAY_REQUEST,
    CA_READ_REQUEST,
    CONFIRMING_PUT_REQUEST,
    FakePvaccess,
    FakePvaException,
    ca_connector,
    clean_epics_env,  # noqa: F401 - fixture, used by name
    enum_record,
    install_fake_pvaccess,
    record,
    writes_enabled,  # noqa: F401 - fixture, used by name
)
from tests.connectors._epics_fakes import (
    patch_writes_enabled as _patch_writes_enabled,
)

TZ_PATCH = "osprey.connectors.control_system.epics_connector.get_facility_timezone"


def _served(address="SR:CH", fields=None, **connector_kwargs):
    """A connector whose fake client serves ``fields`` (default: an analog record)."""
    pvaccess = FakePvaccess()
    pvaccess.serve(address, fields if fields is not None else record(1.0))
    return ca_connector(pvaccess, **connector_kwargs)


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


class TestConnect:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_missing_pvapy_raises_with_install_hint(self, monkeypatch):
        """A missing pvapy raises ImportError naming the pip install command."""
        monkeypatch.setitem(sys.modules, "pvaccess", None)

        connector = EPICSConnector()
        with pytest.raises(ImportError, match="pip install pvapy"):
            await connector.connect({"gateways": {}})

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_connect_holds_the_client_it_imported_and_opens_no_channel(self, monkeypatch):
        """pvapy reads the EPICS environment at the first channel of a provider.

        connect() writes that environment, so it must not open a channel of
        its own — one opened here, before the variables below it are set,
        would pin the process to whatever the environment said before.
        """
        _patch_writes_enabled(monkeypatch, False)
        pvaccess = install_fake_pvaccess(monkeypatch)

        connector = EPICSConnector()
        await connector.connect({"gateways": {"read_only": {"address": "ro", "port": 5064}}})

        assert connector._pvaccess is pvaccess
        assert pvaccess.channels == []
        assert os.environ["EPICS_CA_ADDR_LIST"] == "ro"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_name_server_branch_sets_and_clears_env(self, monkeypatch):
        """use_name_server routes via EPICS_CA_NAME_SERVERS and clears CA_ADDR_LIST."""
        _patch_writes_enabled(monkeypatch, False)
        install_fake_pvaccess(monkeypatch)

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
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_limits_validator_initialized_when_config_present(self, monkeypatch):
        """A configured limits validator is stored on the connector after connect."""
        _patch_writes_enabled(monkeypatch, False)
        install_fake_pvaccess(monkeypatch)
        sentinel = MagicMock(name="limits_validator")
        monkeypatch.setattr(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            classmethod(lambda cls, *, connector_type=None, target=None: sentinel),
        )

        connector = EPICSConnector()
        await connector.connect({"gateways": {"read_only": {"address": "ro", "port": 5064}}})

        assert connector._limits_validator is sentinel
        assert connector._connected is True


# ---------------------------------------------------------------------------
# disconnect()
# ---------------------------------------------------------------------------


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_stops_monitors_and_drops_cached_channels(self):
        """Monitors stop first; cached channels and display metadata are forgotten."""
        connector = _served()
        pvaccess = connector._pvaccess
        await connector.read_channel("SR:CH", timeout=1.0)
        sub_id = await connector.subscribe("SR:CH", lambda value: None)
        name = connector._subscriptions[sub_id].name

        await connector.disconnect()

        ops = [(entry["op"], entry["request"]) for entry in pvaccess.log[-2:]]
        assert ops == [("stopMonitor", None), ("unsubscribe", name)]
        assert connector._subscriptions == {}
        assert connector._channels == {}
        assert connector._ca_displays == {}
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_a_second_disconnect_does_nothing(self):
        connector = _served()
        await connector.subscribe("SR:CH", lambda value: None)
        await connector.disconnect()
        logged = len(connector._pvaccess.log)

        await connector.disconnect()

        assert len(connector._pvaccess.log) == logged


# ---------------------------------------------------------------------------
# read_channel
# ---------------------------------------------------------------------------


class TestReadChannel:
    @pytest.mark.asyncio
    async def test_unreachable_channel_raises_connection_error(self):
        """pvapy's "timed out" is re-raised as the stdlib ConnectionError, with the budget."""
        connector = ca_connector(FakePvaccess())

        with pytest.raises(ConnectionError, match="CA channel 'SR:NOPE'") as excinfo:
            await connector.read_channel("SR:NOPE", timeout=0.5)

        assert "timeout after 0.5s" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, FakePvaException)

    @pytest.mark.asyncio
    async def test_an_unconnected_connector_is_a_connection_error(self):
        """No client was ever loaded: the read cannot even be attempted."""
        with pytest.raises(ConnectionError, match="not connected"):
            await EPICSConnector().read_channel("SR:CH", timeout=0.5)

    @pytest.mark.asyncio
    async def test_an_unrecognized_client_error_propagates_unchanged(self):
        """Only the classified "timed out" text is a connection failure."""
        pvaccess = FakePvaccess()
        error = FakePvaException("Invalid pvRequest")

        def refuse(_request):
            raise error

        pvaccess.get_hooks["SR:CH"] = refuse
        connector = ca_connector(pvaccess)

        with pytest.raises(FakePvaException) as excinfo:
            await connector.read_channel("SR:CH", timeout=0.5)

        assert excinfo.value is error

    @pytest.mark.asyncio
    async def test_the_read_asks_for_value_alarm_and_timestamp_under_the_timeout(self):
        """pvapy's CA provider drops the timestamp when display rides along."""
        connector = _served()

        await connector.read_channel("SR:CH", timeout=1.25)

        reads = connector._pvaccess.calls("get", request=CA_READ_REQUEST)
        assert len(reads) == 1
        assert reads[0]["provider"] == "CA"
        assert reads[0]["timeout"] == 1.25

    @pytest.mark.asyncio
    async def test_read_channel_timestamp_is_facility_tz_aware(self, monkeypatch):
        """The record's own timestamp is rendered in the facility zone, not UTC or the box's.

        A naive ``datetime.fromtimestamp(ts)`` without a zone would fail this.
        """
        monkeypatch.setattr(TZ_PATCH, lambda: ZoneInfo("Asia/Tokyo"))  # UTC+9, no DST
        connector = _served(fields=record(1.23, seconds=1_750_000_000, nanoseconds=500_000_000))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.timestamp.tzinfo is not None
        assert result.timestamp.utcoffset().total_seconds() == 9 * 3600
        assert result.timestamp.timestamp() == pytest.approx(1_750_000_000.5)
        assert result.metadata.timestamp.utcoffset().total_seconds() == 9 * 3600

    @pytest.mark.asyncio
    async def test_a_never_processed_record_is_stamped_now_not_1970(self, monkeypatch):
        """secondsPastEpoch 0 is "no timestamp": the read stamps a facility-tz now."""
        monkeypatch.setattr(TZ_PATCH, lambda: ZoneInfo("Asia/Tokyo"))
        connector = _served(fields=record(3.14, seconds=0))

        before = time.time()
        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.value == 3.14
        assert result.timestamp.utcoffset().total_seconds() == 9 * 3600
        assert result.timestamp.timestamp() >= before

    @pytest.mark.asyncio
    async def test_one_channel_is_reused_across_reads(self):
        """The same address and provider reuse one cached pvapy Channel."""
        connector = _served()

        await connector.read_channel("SR:CH", timeout=1.0)
        await connector.read_channel("SR:CH", timeout=1.0)

        assert len(connector._pvaccess.channels_for("SR:CH")) == 1

    @pytest.mark.asyncio
    async def test_display_metadata_is_fetched_once_and_cached(self):
        """Units and precision are record configuration: one round trip per channel."""
        connector = _served(fields=record(1.0, units="mA", fmt="F9.3"))

        first = await connector.read_channel("SR:CH", timeout=1.0)
        second = await connector.read_channel("SR:CH", timeout=1.0)

        assert len(connector._pvaccess.calls("get", request=CA_DISPLAY_REQUEST)) == 1
        assert len(connector._pvaccess.calls("get", request=CA_READ_REQUEST)) == 2
        for reading in (first, second):
            assert reading.metadata.units == "mA"
            assert reading.metadata.precision == 3

    @pytest.mark.asyncio
    async def test_a_failed_display_fetch_keeps_the_reading_and_is_retried(self):
        """Losing a value for want of its units would be the wrong trade."""
        pvaccess = FakePvaccess()
        pvaccess.serve("SR:CH", record(2.5, units="A"))
        failures = [TimeoutError("display timed out")]

        def flaky_display(request):
            if request == CA_DISPLAY_REQUEST and failures:
                raise failures.pop()
            return None  # everything else is served normally

        pvaccess.get_hooks["SR:CH"] = flaky_display
        connector = ca_connector(pvaccess)

        first = await connector.read_channel("SR:CH", timeout=1.0)
        second = await connector.read_channel("SR:CH", timeout=1.0)

        assert first.value == 2.5
        assert first.metadata.units == ""
        assert second.metadata.units == "A"
        assert len(pvaccess.calls("get", request=CA_DISPLAY_REQUEST)) == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fmt", "precision"),
        [("F9.3", 3), ("F8.2", 2), ("E12.5", 5), ("I12", None), ("", None)],
    )
    async def test_precision_is_read_from_the_display_format(self, fmt, precision):
        """pvapy's CA provider carries PREC only inside ``format``."""
        connector = _served(fields=record(1.0, fmt=fmt))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.precision == precision

    @pytest.mark.asyncio
    async def test_display_limits_and_description_are_mapped(self):
        """Channel Access carries no DESC: an empty description is "not reported"."""
        connector = _served(fields=record(1.0, limit_low=-10.0, limit_high=10.0))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.display_low == -10.0
        assert result.metadata.display_high == 10.0
        assert result.metadata.description is None
        assert result.metadata.raw_metadata["provider"] == "ca"

    @pytest.mark.asyncio
    async def test_read_multiple_drops_failures(self, monkeypatch):
        """read_multiple_channels returns only the channels that read successfully."""
        good = ChannelValue(value=1.0, timestamp=None, metadata=ChannelMetadata())

        async def fake_read(addr, timeout=None):  # noqa: ARG001 - the control-system connector interface fixes this signature
            if addr == "BAD":
                raise ConnectionError("nope")
            return good

        connector = ca_connector()
        monkeypatch.setattr(connector, "read_channel", fake_read)

        result = await connector.read_multiple_channels(["GOOD", "BAD"])

        assert set(result) == {"GOOD"}
        assert result["GOOD"] is good


# ---------------------------------------------------------------------------
# write_channel — the confirm flow
# ---------------------------------------------------------------------------


def _write_connector(*, observed=5.0, fields=None, put_hook=None, get_hook=None, limits=None):
    """A connector whose channel SR:CH holds ``observed`` and takes every put.

    The confirming read runs for real, all the way down to the fake channel's
    ``get``, so the fake client — never a patched ``read_channel`` — is what
    every confirm-flow test steers. ``put_hook`` replaces the put (to clamp,
    stall or fail it); ``get_hook`` can intercept any get (return ``None`` to
    let it through).
    """
    pvaccess = FakePvaccess()
    pvaccess.serve("SR:CH", fields if fields is not None else record(observed))
    if put_hook is not None:
        pvaccess.put_hooks["SR:CH"] = put_hook
    if get_hook is not None:
        pvaccess.get_hooks["SR:CH"] = get_hook
    return ca_connector(pvaccess, limits_validator=limits)


def _clamping_connector(held):
    """A connector whose record takes every put but ends up holding ``held``."""
    connector = _write_connector()
    served = connector._pvaccess.served["SR:CH"]
    connector._pvaccess.put_hooks["SR:CH"] = lambda _sent, _request: served.update(value=held)
    return connector


def _reads(connector):
    """The confirming (or ordinary) reads the connector made."""
    return connector._pvaccess.calls("get", request=CA_READ_REQUEST)


@pytest.mark.usefixtures("writes_enabled")
class TestWriteConfirmation:
    """One confirm flow, and the outcomes every connector reports.

    A write is *confirmed* when the channel it wrote now holds the value sent,
    exactly. There is no tolerance and no second verdict: the outcome word is
    the whole result, and no consumer re-derives one.
    """

    @pytest.mark.asyncio
    async def test_a_channel_holding_the_value_sent_is_confirmed(self):
        connector = _write_connector(observed=0.0)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == pytest.approx(5.0)
        assert result.error_message is None
        assert len(_reads(connector)) == 1

    @pytest.mark.asyncio
    async def test_a_channel_holding_a_different_value_is_a_mismatch(self):
        """A clamped or rounded setpoint is reported, not tolerated."""
        connector = _clamping_connector(4.7)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed_value == pytest.approx(4.7)
        # Both numbers are on the result; the raise path names them from there,
        # and an error_message here would suppress that wording.
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_a_put_whose_callback_outlasts_the_timeout_is_unconfirmed_and_not_re_read(
        self,
    ):
        """The value was sent and nothing has said the IOC took it.

        The outcome is unknown, not a success to go and confirm. No confirming
        read is made, because a read that raced the record's own processing
        would report whatever the channel held a moment ago as the outcome of
        this write.
        """
        connector = _write_connector(put_hook=lambda _sent, _request: time.sleep(0.6))

        start = time.monotonic()
        result = await connector.write_channel("SR:CH", 5.0, confirm=True, timeout=0.2)
        elapsed = time.monotonic() - start

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.observed_value is None
        assert "did not acknowledge" in result.error_message
        assert elapsed < 0.55  # the deadline, not the put, ended the wait
        assert _reads(connector) == []
        assert len(connector._pvaccess.calls("put")) == 1  # it WAS sent

    @pytest.mark.asyncio
    async def test_a_deadline_passing_while_still_connecting_sends_nothing(self):
        """ "Sent" is claimed only once it is true.

        The channel is connected (by introspection) before the put is issued; a
        deadline that passes during that connect is a known non-write —
        ``failed``, nothing sent — and the put that would have followed it is
        abandoned, never issued late.
        """
        connector = _write_connector()
        connector._pvaccess.introspection_hooks["SR:CH"] = lambda: time.sleep(0.5)

        result = await connector.write_channel("SR:CH", 5.0, confirm=True, timeout=0.15)
        await asyncio.sleep(0.6)  # let the stalled connect finish on its worker

        assert result.outcome is WriteOutcome.FAILED
        assert "nothing was sent" in result.error_message
        assert connector._pvaccess.calls("put") == []

    @pytest.mark.asyncio
    async def test_a_confirming_read_that_raises_is_unconfirmed(self):
        """The value was sent; what the channel holds is unknown, not wrong."""

        def failing_read(request):
            if request == CA_READ_REQUEST:
                raise TimeoutError("ca timeout")
            return None

        connector = _write_connector(get_hook=failing_read)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.observed_value is None
        assert "ca timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_a_confirming_read_with_no_value_is_unconfirmed_not_a_mismatch(self):
        """A reading that carries no value is not a reading of the setpoint.

        Compared against the setpoint, ``None`` would not match, and the write
        would be reported as a mismatch carrying an ``observed_value`` of
        ``None`` — an observation the machine never made. The channel's value
        is unknown, which is what ``unconfirmed`` means.
        """
        connector = _clamping_connector(None)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.observed_value is None
        assert "timed out" in result.error_message

    @pytest.mark.asyncio
    async def test_confirm_false_checks_nothing(self):
        connector = _write_connector(observed=9.9)

        result = await connector.write_channel("SR:CH", 5.0, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.observed_value is None
        assert result.error_message is None
        # Not even a read: the channel disagreeing is not this write's verdict.
        assert _reads(connector) == []

    @pytest.mark.asyncio
    async def test_a_mismatch_names_both_values_for_display(self):
        """``notes`` is display-only — nothing classifies a write by parsing it."""
        connector = _clamping_connector(4.7)

        result = await connector.write_channel("SR:CH", 5.0)

        assert "4.7" in result.notes
        assert "5.0" in result.notes


@pytest.mark.usefixtures("writes_enabled")
class TestThePut:
    """What goes over the wire, and in what order."""

    @pytest.mark.asyncio
    async def test_a_confirming_put_waits_for_the_ioc_callback(self):
        """``record[block=true]`` is the put-callback: the protocol's acknowledgement."""
        connector = _write_connector()

        await connector.write_channel("SR:CH", 5.0)

        (put,) = connector._pvaccess.calls("put")
        assert put["request"] == CONFIRMING_PUT_REQUEST
        assert put["provider"] == "CA"

    @pytest.mark.asyncio
    async def test_an_unconfirmed_put_does_not_wait(self):
        connector = _write_connector()

        await connector.write_channel("SR:CH", 5.0, confirm=False)

        (put,) = connector._pvaccess.calls("put")
        assert put["request"] is None

    @pytest.mark.asyncio
    async def test_a_confirming_put_connects_before_it_is_issued(self):
        connector = _write_connector()

        await connector.write_channel("SR:CH", 5.0)

        ops = [(entry["op"], entry["request"]) for entry in connector._pvaccess.log]
        assert ops[:2] == [("introspect", None), ("put", CONFIRMING_PUT_REQUEST)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("value", "sent"),
        [
            pytest.param(np.float64(2.5), 2.5, id="numpy-scalar"),
            pytest.param(np.array([1.0, 2.0]), [1.0, 2.0], id="numpy-array"),
            pytest.param((1, 2, 3), [1, 2, 3], id="tuple"),
            pytest.param("ON", "ON", id="string"),
        ],
    )
    async def test_the_value_is_reduced_to_a_type_pvapy_accepts(self, value, sent):
        """pvapy's put overloads take Python scalars, str and list — nothing else."""
        connector = _write_connector()

        await connector.write_channel("SR:CH", value, confirm=False)

        (put,) = connector._pvaccess.calls("put")
        assert put["value"] == sent
        assert type(put["value"]) is type(sent)

    @pytest.mark.asyncio
    async def test_an_enum_label_written_as_text_is_confirmed_by_its_index(self):
        """An mbbo takes "ON" and reads back 1; that is the same state.

        EPICS is the only connector that reports an ``enum_label``, and this is
        what it is for: without it the comparison would see ``"ON" != 1`` and
        report a mismatch on a write the machine took exactly as sent.
        """
        connector = _write_connector(fields=enum_record(0, ("OFF", "ON")))

        result = await connector.write_channel("SR:CH", "ON")

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == 1

    @pytest.mark.asyncio
    async def test_the_observed_value_keeps_the_type_the_channel_reports(self):
        """A string reading stays a string — nothing is coerced into a number."""
        connector = _write_connector(fields=record("OFF"))

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
    async def test_validation_error_refuses_write_without_a_put(self):
        """A non-limits validation error fails closed: refused, and no put."""
        limits = MagicMock()
        limits.validate.side_effect = RuntimeError("db unreadable")
        connector = _write_connector(limits=limits)

        result = await connector.write_channel("SR:CH", 1.0, confirm=False)

        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "VALIDATION_ERROR"
        assert result.error_message is not None
        assert connector._pvaccess.calls("put") == []

    @pytest.mark.asyncio
    async def test_limits_violation_propagates(self):
        """A ChannelLimitsViolationError from validate is raised, not swallowed."""
        from osprey.errors import ChannelLimitsViolationError

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

        assert connector._pvaccess.calls("put") == []


# ---------------------------------------------------------------------------
# subscribe / unsubscribe / validate_channel / get_metadata
# ---------------------------------------------------------------------------


class TestSubscribe:
    @pytest.mark.asyncio
    async def test_updates_arrive_as_facility_tz_channel_values(self, monkeypatch):
        """The first update is the current value; later ones follow the record."""
        monkeypatch.setattr(TZ_PATCH, lambda: ZoneInfo("Asia/Tokyo"))
        connector = _served(fields=record(1.0, units="A"))
        received = []

        await connector.subscribe("SR:CH", received.append)
        (monitor,) = [c for c in connector._pvaccess.channels_for("SR:CH") if c.monitoring]
        monitor.fire(record(7.0, units="A"))
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert [value.value for value in received] == [1.0, 7.0]
        assert received[1].metadata.units == "A"
        assert received[1].timestamp.utcoffset().total_seconds() == 9 * 3600

    @pytest.mark.asyncio
    async def test_a_monitor_runs_on_a_channel_of_its_own(self):
        """A pvapy Channel runs one monitor, so the shared read channel is never used."""
        connector = _served()
        await connector.read_channel("SR:CH", timeout=1.0)

        sub_id = await connector.subscribe("SR:CH", lambda value: None)

        read_channel = connector._channels[("SR:CH", False)].channel
        monitor_channel = connector._subscriptions[sub_id].channel
        assert monitor_channel is not read_channel
        assert monitor_channel.monitor_request == CA_READ_REQUEST
        assert monitor_channel.provider == "CA"

    @pytest.mark.asyncio
    async def test_units_come_from_the_cached_display_not_the_update(self):
        """A CA monitor cannot carry ``display``; the reads' cached metadata fills it."""
        connector = _served(fields=record(1.0, units="mA", fmt="F9.3"))
        received = []

        await connector.subscribe("SR:CH", received.append)
        await asyncio.sleep(0.01)

        assert received[0].metadata.units == "mA"
        assert received[0].metadata.precision == 3
        assert len(connector._pvaccess.calls("get", request=CA_DISPLAY_REQUEST)) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_the_monitor_and_its_subscriber(self):
        connector = _served()
        received = []
        sub_id = await connector.subscribe("SR:CH", received.append)
        subscription = connector._subscriptions[sub_id]

        await connector.unsubscribe(sub_id)
        subscription.channel.fire(record(9.0))
        await asyncio.sleep(0.01)

        assert subscription.closed is True
        assert subscription.channel.subscribers == {}
        assert [value.value for value in received] == [1.0]
        assert sub_id not in connector._subscriptions


class TestValidateChannelAndMetadata:
    @pytest.mark.asyncio
    async def test_get_metadata_returns_read_metadata(self, monkeypatch):
        meta = ChannelMetadata(units="kV")
        value = ChannelValue(value=1.0, timestamp=None, metadata=meta)
        connector = ca_connector()
        monkeypatch.setattr(connector, "read_channel", AsyncMock(return_value=value))

        assert await connector.get_metadata("SR:CH") is meta

    @pytest.mark.asyncio
    async def test_validate_channel_probes_with_the_configured_timeout(self, monkeypatch):
        """The probe uses the connector's own ``timeout``, not a literal of its own."""
        value = ChannelValue(value=1.0, timestamp=None, metadata=ChannelMetadata())
        read = AsyncMock(return_value=value)
        connector = ca_connector(timeout=17.0)
        monkeypatch.setattr(connector, "read_channel", read)

        assert await connector.validate_channel("SR:CH") is True
        assert read.await_args.kwargs["timeout"] == 17.0

    @pytest.mark.asyncio
    async def test_validate_channel_false_on_read_error(self, monkeypatch):
        connector = ca_connector()
        monkeypatch.setattr(
            connector, "read_channel", AsyncMock(side_effect=ConnectionError("no route"))
        )

        assert await connector.validate_channel("SR:CH") is False

    @pytest.mark.asyncio
    async def test_an_unreachable_channel_validates_false(self):
        assert await ca_connector(FakePvaccess(), timeout=0.2).validate_channel("SR:NOPE") is False


# ---------------------------------------------------------------------------
# Channel Access alarm names (read + subscribe)
# ---------------------------------------------------------------------------


class TestChannelAccessAlarmNames:
    """The alarm status is reported by NAME, taken from ``alarm.message``.

    pvapy's CA provider spells the CA status there (``'HIHI'``, ``'UDF'``), so
    no status-code table is involved. ``alarm.status`` — the normative-type
    status, not the CA code — stays in ``raw_metadata`` beside the severity.
    """

    @pytest.mark.asyncio
    async def test_read_reports_the_alarm_by_name(self):
        connector = _served(fields=record(7.0, severity=2, status=1, message="HIHI"))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == "HIHI"

    @pytest.mark.asyncio
    async def test_a_healthy_ca_record_is_named_no_alarm(self):
        """The CA provider sends an EMPTY message for a healthy record."""
        connector = _served(fields=record(1.0, severity=0, message=""))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == "NO_ALARM"

    @pytest.mark.asyncio
    async def test_read_keeps_the_raw_fields_beside_the_name(self):
        connector = _served(fields=record(1.0, severity=1, status=1, message="LOLO"))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        raw = result.metadata.raw_metadata
        assert result.metadata.alarm_status == "LOLO"
        assert raw["status"] == 1
        assert raw["severity"] == 1
        assert raw["alarm_message"] == "LOLO"

    @pytest.mark.asyncio
    async def test_subscribe_callback_reports_alarm_name(self, monkeypatch):
        """The monitor path maps alarms exactly like the read path."""
        monkeypatch.setattr(TZ_PATCH, lambda: ZoneInfo("UTC"))
        connector = _served()
        received = []

        sub_id = await connector.subscribe("SR:CH", received.append)
        connector._subscriptions[sub_id].channel.fire(
            record(7.0, severity=1, status=1, message="HIGH")
        )
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert received[-1].metadata.alarm_status == "HIGH"
        assert received[-1].metadata.alarm_severity == 1
        assert received[-1].metadata.raw_metadata["severity"] == 1


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
        connector = _served(fields=record(7.0, severity=2, message="HIHI"))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_severity == 2

    @pytest.mark.asyncio
    async def test_a_reported_healthy_severity_stays_zero(self):
        connector = _served(fields=record(1.0, severity=0))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_severity == 0  # not None: healthy is a report

    @pytest.mark.asyncio
    async def test_an_unreported_severity_reads_as_none(self):
        fields = record(1.0)
        del fields["alarm"]  # the record answered its value but carried no alarm
        connector = _served(fields=fields)

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_severity is None
        assert result.metadata.alarm_status is None


# ---------------------------------------------------------------------------
# Channel Access enum labels (read + subscribe)
# ---------------------------------------------------------------------------

MODES = ("OFFLINE", "STANDBY", "ACQUIRING", "FAULT")


class TestChannelAccessEnumLabels:
    """An mbbi/bi/bo read answers with its index *and* the state that index means.

    The index stays the value — the machine-readable half, and the same type
    PVAccess reports for the same record — so the labels are carried beside it
    rather than in place of it. pvapy delivers the choices with the value, in
    the same NTEnum shape a PVA server sends, so there is no second fetch to
    fail; every odd shape of that list still degrades to "no labels" and
    never to a failed read.
    """

    @pytest.mark.asyncio
    async def test_enum_read_reports_the_index_and_its_labels(self):
        connector = _served(fields=enum_record(2, MODES))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.value == 2  # the index, not "ACQUIRING"
        assert result.metadata.enum_label == "ACQUIRING"
        assert result.metadata.enum_labels == list(MODES)
        assert result.metadata.raw_metadata["nt_type"] == "NTEnum"

    @pytest.mark.asyncio
    async def test_index_zero_resolves_to_its_label_not_to_nothing(self):
        """A bi at 0 is a state, not a falsy miss."""
        connector = _served(fields=enum_record(0, ("OFF", "ON")))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.value == 0
        assert result.metadata.enum_label == "OFF"

    @pytest.mark.asyncio
    async def test_an_enum_record_reads_without_display_metadata(self):
        """pvapy serves no ``display`` for an enum; the read does not need one."""
        connector = _served(fields=enum_record(1, ("OFF", "ON")))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.units == ""
        assert result.metadata.precision is None
        assert result.metadata.enum_label == "ON"

    @pytest.mark.asyncio
    async def test_a_non_enum_read_leaves_both_fields_unset(self):
        """The fields are how a consumer tells an enum channel from a numeric one."""
        connector = _served(fields=record(7.25))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.value == 7.25
        assert result.metadata.enum_labels is None
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_unreported_labels_leave_the_fields_unset(self):
        """An empty choices list names nothing; the index alone is still the answer."""
        connector = _served(fields=enum_record(2, ()))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.value == 2
        assert result.metadata.enum_labels is None
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_an_index_past_the_label_list_keeps_the_list(self):
        """An unresolvable index loses its label, not the states it could not name."""
        connector = _served(fields=enum_record(9, ("OFF", "ON")))

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.value == 9
        assert result.metadata.enum_labels == ["OFF", "ON"]
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_subscribe_callback_reports_the_label(self, monkeypatch):
        """A monitor update carries its choices too, and maps like a read."""
        monkeypatch.setattr(TZ_PATCH, lambda: ZoneInfo("UTC"))
        connector = _served(fields=enum_record(0, MODES))
        received = []

        sub_id = await connector.subscribe("SR:CH", received.append)
        connector._subscriptions[sub_id].channel.fire(enum_record(3, MODES))
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert [value.value for value in received] == [0, 3]
        assert received[-1].metadata.enum_label == "FAULT"
        assert received[-1].metadata.enum_labels == list(MODES)


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
        connector = _write_connector(fields=record(5.0, severity=0))

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
        connector = _write_connector(fields=record(5.0, severity=2, message="HIHI"))

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.alarm_status == "HIHI"
        assert result.alarm_severity == 2
        assert raise_for_write_result(result) is result

    @pytest.mark.asyncio
    async def test_a_mismatch_carries_the_alarm_state_too(self):
        connector = _write_connector(fields=record(9.9, severity=2, message="HIHI"))
        served = connector._pvaccess.served["SR:CH"]
        connector._pvaccess.put_hooks["SR:CH"] = lambda _sent, _request: served.update(value=9.9)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.alarm_status == "HIHI"
        assert result.alarm_severity == 2

    @pytest.mark.asyncio
    async def test_an_unconfirmed_write_claims_no_alarm_state(self):
        """Nothing was read, so no alarm state can be claimed."""

        def failing_read(request):
            if request == CA_READ_REQUEST:
                raise TimeoutError("ca timeout")
            return None

        connector = _write_connector(get_hook=failing_read)

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

        def failing_read(request):
            if request == CA_READ_REQUEST:
                raise TimeoutError("confirmed NO_ALARM severity 0 value 5.0")
            return None

        connector = _write_connector(get_hook=failing_read)

        result = await connector.write_channel("SR:CH", 5.0)

        assert "confirmed NO_ALARM" in result.error_message  # the wording did land
        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.alarm_status is None
        assert result.alarm_severity is None
        assert result.observed_value is None


# ---------------------------------------------------------------------------
# Review fixes: exact writes, one confirming put per channel, first-use
# serialization, FAILED when nothing was sent, char arrays, backstops
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("writes_enabled")
class TestExactNumericWrites:
    """pvapy's scalar put keeps six significant digits; a typed PvObject keeps all."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", [1.2345678901234567, 499654321.5, 1.0000001])
    @pytest.mark.parametrize("confirm", [True, False])
    async def test_a_float_goes_out_as_a_pvobject_of_the_channels_own_type(self, value, confirm):
        connector = _write_connector(observed=0.0)

        await connector.write_channel("SR:CH", value, confirm=confirm)

        (put,) = connector._pvaccess.calls("put")
        assert put["pv_type"] == "DOUBLE"
        assert put["value"] == value  # the full double, not its six-digit text
        assert connector._pvaccess.served["SR:CH"]["value"] == value

    @pytest.mark.asyncio
    async def test_a_whole_float_to_an_integer_channel_goes_out_as_an_int(self):
        connector = _write_connector(fields=record(0))

        result = await connector.write_channel("SR:CH", 123456789.0)

        (put,) = connector._pvaccess.calls("put")
        assert put["value"] == 123456789
        assert type(put["value"]) is int
        assert result.outcome is WriteOutcome.CONFIRMED

    @pytest.mark.asyncio
    async def test_a_fraction_to_an_integer_channel_is_refused_unsent(self):
        connector = _write_connector(fields=record(0))

        result = await connector.write_channel("SR:CH", 1.5)

        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "VALIDATION_ERROR"
        assert connector._pvaccess.calls("put") == []

    @pytest.mark.asyncio
    async def test_a_float_to_a_string_channel_is_spelled_exactly(self):
        connector = _write_connector(fields=record("x"))

        await connector.write_channel("SR:CH", 1.2345678901234567, confirm=False)

        (put,) = connector._pvaccess.calls("put")
        assert put["value"] == "1.2345678901234567"


@pytest.mark.usefixtures("writes_enabled")
class TestOneConfirmingPutPerChannel:
    """A retry never enters pvapy behind a pending put-callback on the same channel."""

    @pytest.mark.asyncio
    async def test_a_retry_behind_a_pending_put_fails_unsent_within_its_deadline(self):
        connector = _write_connector(put_hook=lambda _sent, _request: time.sleep(0.8))
        connector._pvaccess.serve("SR:OTHER", record(2.0))

        first = await connector.write_channel("SR:CH", 5.0, confirm=True, timeout=0.2)
        start = time.monotonic()
        retry, other = await asyncio.gather(
            connector.write_channel("SR:CH", 5.0, confirm=True, timeout=0.2),
            connector.read_channel("SR:OTHER", timeout=1.0),
        )
        elapsed = time.monotonic() - start

        assert first.outcome is WriteOutcome.UNCONFIRMED
        assert retry.outcome is WriteOutcome.FAILED
        assert "still waiting" in retry.error_message
        assert elapsed < 0.4
        assert other.value == 2.0
        assert len(connector._pvaccess.calls("put")) == 1  # the retry was never sent
        await asyncio.sleep(0.8)  # let the first put finish on its worker

    @pytest.mark.asyncio
    async def test_a_put_after_the_pending_one_settles_goes_through(self):
        connector = _write_connector(put_hook=lambda _sent, _request: time.sleep(0.3))

        await connector.write_channel("SR:CH", 5.0, confirm=True, timeout=0.1)
        second = await connector.write_channel("SR:CH", 5.0, confirm=True, timeout=1.0)

        assert second.outcome is not WriteOutcome.FAILED
        assert len(connector._pvaccess.calls("put")) == 2


@pytest.mark.usefixtures("writes_enabled")
class TestNothingSentIsFailed:
    """A channel that never connected is a known non-write: FAILED, and a batch goes on."""

    @pytest.mark.asyncio
    async def test_an_unreachable_channel_is_a_failed_write(self):
        connector = _write_connector()

        result = await connector.write_channel("SR:NOPE", 5.0, timeout=0.2)

        assert result.outcome is WriteOutcome.FAILED
        assert "nothing was sent" in result.error_message
        assert result.refusal_reason is None
        assert connector._pvaccess.calls("put") == []

    @pytest.mark.asyncio
    async def test_a_dead_channel_mid_batch_keeps_every_row(self):
        connector = _write_connector(observed=0.0)
        connector._pvaccess.serve("SR:LAST", record(0.0))

        results = await connector.write_multiple_channels(
            [("SR:CH", 5.0), ("SR:NOPE", 1.0), ("SR:LAST", 3.0)], timeout=0.2, confirm=True
        )

        assert [r.outcome for r in results] == [
            WriteOutcome.CONFIRMED,
            WriteOutcome.FAILED,
            WriteOutcome.CONFIRMED,
        ]
        assert connector._pvaccess.served["SR:LAST"]["value"] == 3.0


class TestFirstUseIsSerialized:
    """pvapy hangs or fails when many threads make a fresh Channel's first call at once."""

    @pytest.mark.asyncio
    async def test_concurrent_first_reads_all_succeed(self):
        connector = _served()
        pvaccess = connector._pvaccess
        in_flight = {"n": 0, "max": 0}

        def first_use_is_fragile(_request):
            (channel,) = pvaccess.channels_for("SR:CH")
            if channel.connected:
                return None
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
            try:
                if in_flight["n"] > 1:
                    raise FakePvaException("Channel SR:CH timed out.")
                time.sleep(0.05)
                return None
            finally:
                in_flight["n"] -= 1

        pvaccess.get_hooks["SR:CH"] = first_use_is_fragile

        results = await asyncio.gather(*[connector.read_channel("SR:CH") for _ in range(14)])

        assert [r.value for r in results] == [1.0] * 14
        assert in_flight["max"] == 1

    @pytest.mark.asyncio
    async def test_a_monitor_retries_display_metadata_that_first_failed(self, monkeypatch):
        monkeypatch.setattr(
            "osprey_connectors.control_system.epics_connector._DISPLAY_RETRY_S", 0.0
        )
        connector = _served(fields=record(1.0, units="mA"))
        failures = {"left": 1}

        def display_fails_once(request):
            if request == CA_DISPLAY_REQUEST and failures["left"]:
                failures["left"] -= 1
                raise FakePvaException("Channel SR:CH timed out.")
            return None

        connector._pvaccess.get_hooks["SR:CH"] = display_fails_once
        received = []

        await connector.subscribe("SR:CH", received.append)
        (monitor,) = [c for c in connector._pvaccess.channels_for("SR:CH") if c.monitoring]
        await asyncio.sleep(0.1)  # the retry the first update scheduled
        monitor.fire(record(2.0, units="mA"))
        await asyncio.sleep(0.05)

        assert received[0].metadata.units == ""  # the fetch had failed
        assert received[-1].metadata.units == "mA"  # ... and was retried


@pytest.mark.usefixtures("writes_enabled")
class TestCharArrays:
    """A Channel Access char waveform: text in as bytes, bytes out unsigned."""

    @staticmethod
    def _char_connector(nelm=16):
        connector = _write_connector(fields=record(np.array([], dtype=np.int8)))
        if nelm is not None:
            connector._pvaccess.serve("SR:CH.NELM", record(float(nelm)))
        return connector

    @pytest.mark.asyncio
    async def test_text_is_written_as_nul_terminated_bytes_and_confirmed_as_text(self):
        connector = self._char_connector()
        served = connector._pvaccess.served["SR:CH"]
        connector._pvaccess.put_hooks["SR:CH"] = lambda sent, _request: served.update(
            value=np.array(sent, dtype=np.int8)
        )

        result = await connector.write_channel("SR:CH", "hello")

        (put,) = connector._pvaccess.calls("put")
        assert put["value"] == [*b"hello", 0]
        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == "hello"

    @pytest.mark.asyncio
    async def test_text_is_cut_to_nelm(self):
        connector = self._char_connector(nelm=4)

        await connector.write_channel("SR:CH", "hello", confirm=False)

        (put,) = connector._pvaccess.calls("put")
        assert put["value"] == list(b"hell")

    @pytest.mark.asyncio
    async def test_unsigned_bytes_go_out_signed(self):
        connector = self._char_connector()

        await connector.write_channel("SR:CH", [200, 65, 255], confirm=False)

        (put,) = connector._pvaccess.calls("put")
        assert put["value"] == [-56, 65, -1]

    @pytest.mark.asyncio
    async def test_a_ca_char_array_reads_unsigned(self):
        connector = _served(fields=record(np.array([-56, 65, -1], dtype=np.int8)))

        value = (await connector.read_channel("SR:CH")).value

        assert value.dtype == np.uint8
        assert value.tolist() == [200, 65, 255]

    @pytest.mark.asyncio
    async def test_a_pva_byte_array_stays_as_served(self):
        address = "SR:CAM1:IMAGE"
        pvaccess = FakePvaccess()
        pvaccess.serve(address, record(np.array([-56, 65], dtype=np.int8)))
        connector = ca_connector(pvaccess, globs=("SR:CAM*:IMAGE",))

        value = (await connector.read_channel(address)).value

        assert value.tolist() == [-56, 65]


class TestBackstops:
    """No offloaded pvapy call is awaited forever."""

    @pytest.mark.asyncio
    async def test_a_read_that_never_returns_is_a_connection_error(self):
        connector = _served(timeout=0.1)
        connector._pvaccess.get_hooks["SR:CH"] = lambda _request: time.sleep(2.0)

        start = time.monotonic()
        with pytest.raises(ConnectionError, match="did not complete"):
            await connector.read_channel("SR:CH")

        assert time.monotonic() - start < 1.6  # the 1.3 s backstop, not the 2 s hang
