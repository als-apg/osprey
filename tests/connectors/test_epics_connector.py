"""Behavioral tests for the EPICS control-system connector.

No general EPICS connector test existed before this file (see the note in
``test_epics_connector_timezone.py``); gateway selection and the read timestamp
were the only covered paths. These tests drive the connector's remaining real
code paths — libca configuration, connect error/name-server handling, the
confirm flow and its five outcomes, the fail-closed write guard, and
subscription plumbing — with an injected fake ``_epics`` so no real Channel
Access is required.

Convention (matching ``test_epics_connector_timezone.py`` and PR #270): inject a
fake ``_epics`` and assert on the concrete payload — outcome word, observed
value, alarm fields, env vars, refusal reason — never merely that a call
"didn't raise".
"""

import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    WriteOutcome,
    raise_for_write_result,
)
from osprey.connectors.control_system.epics_connector import (
    EPICSConnector,
    _ChannelSubscription,
    _configure_pyepics_libca,
)

EPICS_VARS = [
    "EPICS_CA_ADDR_LIST",
    "EPICS_CA_SERVER_PORT",
    "EPICS_CA_NAME_SERVERS",
    "EPICS_CA_AUTO_ADDR_LIST",
]


@pytest.fixture
def clean_epics_env(monkeypatch):
    """Snapshot EPICS_* env vars so connect()'s direct os.environ writes are restored."""
    for var in EPICS_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


def _patch_writes_enabled(monkeypatch, enabled: bool):
    def fake_get_config_value(key, default=None):
        if key == "control_system.writes_enabled":
            return enabled
        return default

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)


def _connector(*, epics=None, limits_validator=None, timeout=5.0):
    """Build a connector that skips connect() by injecting its runtime state."""
    connector = EPICSConnector()
    connector._epics = epics if epics is not None else MagicMock()
    connector._limits_validator = limits_validator
    connector._timeout = timeout
    connector._connected = True
    connector._epics_configured = True
    return connector


@pytest.fixture
def writes_enabled(monkeypatch):
    """Enable the base-class writes gate so write_channel reaches its real body.

    ControlSystemConnector.__init_subclass__ wraps write_channel with a
    _writes_enabled pre-check that is False in a config-less test env; these
    write-path tests are about what happens *after* that gate opens.
    """
    monkeypatch.setattr(EPICSConnector, "_writes_enabled", property(lambda self: True))


# ---------------------------------------------------------------------------
# _configure_pyepics_libca
# ---------------------------------------------------------------------------


class TestConfigurePyepicsLibca:
    def test_explicit_override_is_left_untouched(self, monkeypatch):
        """An operator's PYEPICS_LIBCA always wins — the helper returns early."""
        monkeypatch.setenv("PYEPICS_LIBCA", "/operator/libca.so")

        _configure_pyepics_libca()

        assert os.environ["PYEPICS_LIBCA"] == "/operator/libca.so"

    def test_sets_libca_from_epicscorelibs_when_unset(self, monkeypatch):
        """When unset, the helper points PYEPICS_LIBCA at epicscorelibs' libca."""
        monkeypatch.delenv("PYEPICS_LIBCA", raising=False)
        fake_path = types.ModuleType("epicscorelibs.path")
        fake_path.get_lib = lambda name: f"/fake/{name}/libca.so"
        fake_pkg = types.ModuleType("epicscorelibs")
        fake_pkg.path = fake_path
        monkeypatch.setitem(sys.modules, "epicscorelibs", fake_pkg)
        monkeypatch.setitem(sys.modules, "epicscorelibs.path", fake_path)

        _configure_pyepics_libca()

        assert os.environ["PYEPICS_LIBCA"] == "/fake/ca/libca.so"

    def test_no_op_when_epicscorelibs_absent(self, monkeypatch):
        """epicscorelibs missing -> PYEPICS_LIBCA stays unset (pyepics resolves itself)."""
        monkeypatch.delenv("PYEPICS_LIBCA", raising=False)
        # Block both the package and the submodule: in an env where EPICS is
        # installed, `epicscorelibs.path` is already cached in sys.modules, so
        # nulling only the parent would not stop `from epicscorelibs.path import`.
        monkeypatch.setitem(sys.modules, "epicscorelibs", None)
        monkeypatch.setitem(sys.modules, "epicscorelibs.path", None)

        _configure_pyepics_libca()

        assert "PYEPICS_LIBCA" not in os.environ


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


class TestConnect:
    @pytest.mark.asyncio
    async def test_missing_pyepics_raises_with_install_hint(self, monkeypatch, clean_epics_env):
        """A missing pyepics raises ImportError naming the pip install command."""
        monkeypatch.setitem(sys.modules, "epics", None)

        connector = EPICSConnector()
        with pytest.raises(ImportError, match="pip install pyepics"):
            await connector.connect({"gateways": {}})

    @pytest.mark.asyncio
    async def test_name_server_branch_sets_and_clears_env(self, monkeypatch, clean_epics_env):
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
    async def test_limits_validator_initialized_when_config_present(
        self, monkeypatch, clean_epics_env
    ):
        """A configured limits validator is stored on the connector after connect."""
        _patch_writes_enabled(monkeypatch, False)
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
# read_channel error / timestamp fallback paths
# ---------------------------------------------------------------------------


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_unsubscribes_and_clears_cache(self):
        """disconnect() drops subscriptions and best-effort-disconnects cached PVs."""
        sub_pv = MagicMock()
        cached_ok = MagicMock()
        cached_bad = MagicMock()
        cached_bad.disconnect.side_effect = RuntimeError("already gone")
        connector = _connector()
        connector._subscriptions = {"sub1": _ChannelSubscription("ca", sub_pv)}
        connector._pv_cache = {"A": cached_ok, "B": cached_bad}

        await connector.disconnect()

        sub_pv.clear_callbacks.assert_called_once()  # via unsubscribe()
        cached_ok.disconnect.assert_called_once()  # error on cached_bad is swallowed
        assert connector._pv_cache == {}
        assert connector._subscriptions == {}
        assert connector._connected is False


class TestReadChannel:
    @pytest.mark.asyncio
    async def test_unconnected_pv_raises_connection_error(self):
        """A PV that never connects surfaces as ConnectionError with the timeout."""
        pv = MagicMock()
        pv.wait_for_connection.return_value = False
        pv.connected = False
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _connector(epics=epics)

        with pytest.raises(ConnectionError, match="Failed to connect to PV 'SR:NOPE'"):
            await connector.read_channel("SR:NOPE", timeout=0.5)

    @pytest.mark.asyncio
    async def test_missing_timestamp_falls_back_to_now(self, monkeypatch):
        """When the PV reports no timestamp, the read stamps a facility-tz 'now'."""
        tokyo = __import__("zoneinfo").ZoneInfo("Asia/Tokyo")
        monkeypatch.setattr(
            "osprey.connectors.control_system.epics_connector.get_facility_timezone",
            lambda: tokyo,
        )
        pv = MagicMock()
        pv.wait_for_connection.return_value = True
        pv.connected = True
        pv.get.return_value = 3.14
        pv.timestamp = 0  # falsy -> now() branch
        pv.units = "mm"
        pv.status = 0
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.value == 3.14
        assert result.timestamp.tzinfo is not None
        assert result.timestamp.utcoffset().total_seconds() == 9 * 3600

    @pytest.mark.asyncio
    async def test_pv_cache_reused_across_reads(self, monkeypatch):
        """The same channel reuses its cached PV object instead of recreating it."""
        monkeypatch.setattr(
            "osprey.connectors.control_system.epics_connector.get_facility_timezone",
            lambda: __import__("zoneinfo").ZoneInfo("UTC"),
        )
        pv = MagicMock()
        pv.wait_for_connection.return_value = True
        pv.connected = True
        pv.get.return_value = 1.0
        pv.timestamp = 1_750_000_000.0
        pv.units = ""
        pv.status = 0
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _connector(epics=epics)

        await connector.read_channel("SR:CH", timeout=1.0)
        await connector.read_channel("SR:CH", timeout=1.0)

        epics.PV.assert_called_once()  # created on first read, cached for the second

    @pytest.mark.asyncio
    async def test_read_multiple_drops_failures(self, monkeypatch):
        """read_multiple_channels returns only the channels that read successfully."""
        good = ChannelValue(value=1.0, timestamp=None, metadata=ChannelMetadata())

        async def fake_read(addr, timeout=None):
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


def _readback_pv(value, *, status=0, severity=0, pv_type="time_double", labels=None):
    """A fake pyepics PV standing in for the channel a write confirms against."""
    pv = _connected_pv(value=value, status=status, severity=severity)
    pv.type = pv_type
    if labels is not None:
        pv.enum_strs = labels
    return pv


def _write_connector(*, observed=5.0, caput=True, limits=None, pv=None, read_error=None):
    """A connector whose caput answers ``caput`` and whose channel holds ``observed``.

    The confirming read runs for real, all the way down to ``pv.get()``, so the
    injected fake ``_epics.PV`` — never a patched ``read_channel`` — is what
    every confirm-flow test steers. Patching the read would hide the one thing
    the confirming read exists to do differently from an ordinary read.
    """
    epics = MagicMock()
    epics.caput.return_value = caput
    if pv is None:
        pv = _readback_pv(observed)
    if read_error is not None:
        pv.get.side_effect = read_error
    epics.PV.return_value = pv
    return _connector(epics=epics, limits_validator=limits)


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
    async def test_a_confirming_read_that_raises_is_unconfirmed(self):
        """The value was sent; what the channel holds is unknown, not wrong."""
        connector = _write_connector(read_error=TimeoutError("ca timeout"))

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.observed_value is None
        assert "ca timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_a_confirming_read_that_times_out_is_unconfirmed_not_a_mismatch(self):
        """pyepics answers a timed-out ``get`` with ``None`` instead of raising.

        Compared against the setpoint that ``None`` would not match, and the
        write would be reported as a mismatch carrying an ``observed_value`` of
        ``None`` — an observation the machine never made. The channel's value is
        unknown, which is what ``unconfirmed`` means.
        """
        connector = _write_connector(observed=None)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.observed_value is None
        assert "timed out" in result.error_message

    @pytest.mark.asyncio
    async def test_a_put_the_control_system_did_not_take_is_failed(self):
        connector = _write_connector(caput=False)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.FAILED
        assert result.error_message is not None
        assert result.observed_value is None
        # Nothing was written, so there is nothing to confirm.
        connector._epics.PV.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirm_false_checks_nothing(self):
        connector = _write_connector(observed=9.9)

        result = await connector.write_channel("SR:CH", 5.0, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.observed_value is None
        assert result.error_message is None
        # Not even a read: the channel disagreeing is not this write's verdict.
        connector._epics.PV.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_mismatch_names_both_values_for_display(self):
        """``notes`` is display-only — nothing classifies a write by parsing it."""
        connector = _write_connector(observed=4.7)

        result = await connector.write_channel("SR:CH", 5.0)

        assert "4.7" in result.notes
        assert "5.0" in result.notes


@pytest.mark.usefixtures("writes_enabled")
class TestConfirmingRead:
    """What the confirming read does differently from an ordinary read."""

    @pytest.mark.asyncio
    async def test_the_confirming_read_bypasses_the_monitor_cache(self):
        """pyepics' auto-monitor cache can still hold the pre-write value.

        The put-callback says the IOC processed the write, not that a cached
        subscription update has arrived. Under confirm-by-default, comparing a
        stale cached reading against the setpoint would report a MISMATCH the
        machine never had — so the confirming read always goes to the wire.
        """
        pv = _readback_pv(5.0)
        connector = _write_connector(pv=pv)

        await connector.write_channel("SR:CH", 5.0)

        assert pv.get.call_args.kwargs["use_monitor"] is False

    @pytest.mark.asyncio
    async def test_an_ordinary_read_still_uses_the_monitor_cache(self):
        """Only confirmation pays for a fresh get; plain reads keep the cache."""
        pv = _readback_pv(5.0)
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _connector(epics=epics)

        await connector.read_channel("SR:CH", timeout=1.0)

        assert pv.get.call_args.kwargs.get("use_monitor", True) is True

    @pytest.mark.asyncio
    async def test_a_confirming_put_waits_for_the_ioc_callback(self):
        """Put-callback is the protocol's acknowledgement that the put landed."""
        connector = _write_connector(observed=5.0)

        await connector.write_channel("SR:CH", 5.0)

        assert connector._epics.caput.call_args.kwargs["wait"] is True

    @pytest.mark.asyncio
    async def test_an_unconfirmed_put_does_not_wait(self):
        connector = _write_connector(observed=5.0)

        await connector.write_channel("SR:CH", 5.0, confirm=False)

        assert connector._epics.caput.call_args.kwargs["wait"] is False

    @pytest.mark.asyncio
    async def test_an_enum_label_written_as_text_is_confirmed_by_its_index(self):
        """An mbbo takes "ON" and reads back 1; that is the same state.

        EPICS is the only connector that reports an ``enum_label``, and this is
        what it is for: without it the comparison would see ``"ON" != 1`` and
        report a mismatch on a write the machine took exactly as sent.
        """
        pv = _readback_pv(1, pv_type="time_enum", labels=("OFF", "ON"))
        connector = _write_connector(pv=pv)

        result = await connector.write_channel("SR:VALVE", "ON")

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == 1

    @pytest.mark.asyncio
    async def test_the_observed_value_keeps_the_type_the_channel_reports(self):
        """A string reading stays a string — nothing is coerced into a number."""
        pv = _readback_pv("ON", pv_type="time_string")
        connector = _write_connector(pv=pv)

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
        epics = MagicMock()
        limits = MagicMock()
        limits.validate.side_effect = RuntimeError("db unreadable")
        connector = _connector(epics=epics, limits_validator=limits)

        result = await connector.write_channel("SR:CH", 1.0, confirm=False)

        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "VALIDATION_ERROR"
        assert result.error_message is not None
        epics.caput.assert_not_called()

    @pytest.mark.asyncio
    async def test_limits_violation_propagates(self):
        """A ChannelLimitsViolationError from validate is raised, not swallowed."""
        from osprey.errors import ChannelLimitsViolationError

        epics = MagicMock()
        limits = MagicMock()
        limits.validate.side_effect = ChannelLimitsViolationError(
            channel_address="SR:CH",
            value=1.0,
            violation_type="MAX_EXCEEDED",
            violation_reason="too big",
        )
        connector = _connector(epics=epics, limits_validator=limits)

        with pytest.raises(ChannelLimitsViolationError):
            await connector.write_channel("SR:CH", 1.0, confirm=False)

        epics.caput.assert_not_called()


# ---------------------------------------------------------------------------
# subscribe / unsubscribe / validate_channel / get_metadata
# ---------------------------------------------------------------------------


class TestSubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_registers_pv_and_returns_id(self):
        pv = MagicMock()
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _connector(epics=epics)

        sub_id = await connector.subscribe("SR:CH", lambda v: None)

        assert sub_id.startswith("SR:CH_")
        assert connector._subscriptions[sub_id].handle is pv

    @pytest.mark.asyncio
    async def test_epics_callback_converts_to_channel_value(self, monkeypatch):
        """The pyepics callback is adapted into a facility-tz ChannelValue."""
        tokyo = __import__("zoneinfo").ZoneInfo("Asia/Tokyo")
        monkeypatch.setattr(
            "osprey.connectors.control_system.epics_connector.get_facility_timezone",
            lambda: tokyo,
        )
        pv = MagicMock()
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _connector(epics=epics)
        received = []

        await connector.subscribe("SR:CH", received.append)

        # Grab the wrapper pyepics would call and fire it as CA would.
        epics_callback = epics.PV.call_args.kwargs["callback"]
        epics_callback(pvname="SR:CH", value=7.0, timestamp=1_750_000_000.0, units="A")
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert len(received) == 1
        assert received[0].value == 7.0
        assert received[0].metadata.units == "A"
        assert received[0].timestamp.utcoffset().total_seconds() == 9 * 3600

    @pytest.mark.asyncio
    async def test_unsubscribe_clears_and_removes(self):
        pv = MagicMock()
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _connector(epics=epics)
        sub_id = await connector.subscribe("SR:CH", lambda v: None)

        await connector.unsubscribe(sub_id)

        pv.clear_callbacks.assert_called_once()
        assert sub_id not in connector._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_unknown_id_is_noop(self):
        connector = _connector()
        # Must not raise for an id that was never registered.
        await connector.unsubscribe("does-not-exist")


class TestValidateChannelAndMetadata:
    @pytest.mark.asyncio
    async def test_get_metadata_returns_read_metadata(self, monkeypatch):
        meta = ChannelMetadata(units="kV")
        value = ChannelValue(value=1.0, timestamp=None, metadata=meta)
        connector = _connector()
        monkeypatch.setattr(connector, "read_channel", AsyncMock(return_value=value))

        assert await connector.get_metadata("SR:CH") is meta

    @pytest.mark.asyncio
    async def test_validate_channel_true_on_successful_read(self, monkeypatch):
        value = ChannelValue(value=1.0, timestamp=None, metadata=ChannelMetadata())
        connector = _connector()
        monkeypatch.setattr(connector, "read_channel", AsyncMock(return_value=value))

        assert await connector.validate_channel("SR:CH") is True

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


def _connected_pv(*, status=0, severity=0, value=1.0):
    """A fake pyepics PV that reports a value and an alarm state."""
    pv = MagicMock()
    pv.wait_for_connection.return_value = True
    pv.connected = True
    pv.get.return_value = value
    pv.timestamp = 1_750_000_000.0
    pv.units = "mA"
    pv.precision = 3
    pv.status = status
    pv.severity = severity
    return pv


class TestChannelAccessAlarmNames:
    """CA reports alarm status as an int; the connector emits the EPICS name.

    ``ChannelMetadata.alarm_status`` is declared ``str | None`` and PVAccess
    already emitted names, so the raw CA code was both the wrong type and
    unreadable downstream. The code itself is not lost — it stays in
    ``raw_metadata["status"]`` next to the severity.
    """

    @pytest.mark.asyncio
    async def test_read_reports_alarm_name_not_code(self):
        epics = MagicMock()
        epics.PV.return_value = _connected_pv(status=3, severity=2)
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == "HIHI"  # not the integer 3

    @pytest.mark.asyncio
    async def test_read_reports_healthy_alarm_by_name(self):
        epics = MagicMock()
        epics.PV.return_value = _connected_pv(status=0, severity=0)
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == "NO_ALARM"

    @pytest.mark.asyncio
    async def test_read_keeps_the_raw_code_beside_the_severity(self):
        epics = MagicMock()
        epics.PV.return_value = _connected_pv(status=5, severity=1)
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == "LOLO"
        assert result.metadata.raw_metadata["status"] == 5
        assert result.metadata.raw_metadata["severity"] == 1

    @pytest.mark.asyncio
    async def test_unmappable_code_reads_as_unknown(self):
        """An out-of-range code must not raise — it degrades to UNKNOWN."""
        epics = MagicMock()
        epics.PV.return_value = _connected_pv(status=99, severity=3)
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:CH", timeout=1.0)

        assert result.metadata.alarm_status == "UNKNOWN"
        assert result.metadata.raw_metadata["status"] == 99  # raw code still recorded

    @pytest.mark.asyncio
    async def test_subscribe_callback_reports_alarm_name(self, monkeypatch):
        """The monitor path maps codes exactly like the read path."""
        monkeypatch.setattr(
            "osprey.connectors.control_system.epics_connector.get_facility_timezone",
            lambda: __import__("zoneinfo").ZoneInfo("UTC"),
        )
        epics = MagicMock()
        epics.PV.return_value = MagicMock()
        connector = _connector(epics=epics)
        received = []

        await connector.subscribe("SR:CH", received.append)
        epics_callback = epics.PV.call_args.kwargs["callback"]
        epics_callback(
            pvname="SR:CH", value=7.0, timestamp=1_750_000_000.0, units="A", status=4, severity=1
        )
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert received[0].metadata.alarm_status == "HIGH"
        assert received[0].metadata.raw_metadata["status"] == 4
        assert received[0].metadata.raw_metadata["severity"] == 1


# ---------------------------------------------------------------------------
# Channel Access enum labels (read + subscribe)
# ---------------------------------------------------------------------------


def _enum_pv(*, value=2, labels=("OFFLINE", "STANDBY", "ACQUIRING", "FAULT"), pv_type="time_enum"):
    """A fake pyepics PV for an mbbi: an index, and the labels it indexes into."""
    pv = _connected_pv(value=value)
    pv.type = pv_type
    pv.enum_strs = labels
    return pv


class TestChannelAccessEnumLabels:
    """An mbbi/bi/bo read answers with its index *and* the state that index means.

    The index stays the value — the machine-readable half, and the same type
    PVAccess reports for the same record — so the labels are carried beside it
    rather than in place of it. Fetching them costs a ``get_ctrlvars`` round
    trip to the IOC, so every failure mode of that fetch degrades to "no
    labels" and never to a failed read: a reading with an index and no label is
    still a correct answer, a raised read is not.
    """

    @pytest.mark.asyncio
    async def test_enum_read_reports_the_index_and_its_labels(self):
        epics = MagicMock()
        epics.PV.return_value = _enum_pv(value=2)
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:MODE", timeout=1.0)

        assert result.value == 2  # the index, not "ACQUIRING"
        assert result.metadata.enum_label == "ACQUIRING"
        assert result.metadata.enum_labels == ["OFFLINE", "STANDBY", "ACQUIRING", "FAULT"]

    @pytest.mark.asyncio
    async def test_index_zero_resolves_to_its_label_not_to_nothing(self):
        """A bi at 0 is a state, not a falsy miss."""
        epics = MagicMock()
        epics.PV.return_value = _enum_pv(value=0, labels=("OFF", "ON"))
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:SHUTTER", timeout=1.0)

        assert result.value == 0
        assert result.metadata.enum_label == "OFF"

    @pytest.mark.asyncio
    async def test_the_plain_enum_type_spelling_is_recognized_too(self):
        """pyepics spells it "enum" or "time_enum" depending on the PV's form."""
        epics = MagicMock()
        epics.PV.return_value = _enum_pv(value=1, pv_type="enum")
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:MODE", timeout=1.0)

        assert result.metadata.enum_label == "STANDBY"

    @pytest.mark.asyncio
    async def test_a_non_enum_read_leaves_both_fields_unset(self):
        """The fields are how a consumer tells an enum channel from a numeric one."""
        epics = MagicMock()
        epics.PV.return_value = _connected_pv(value=7.25)
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:CURRENT", timeout=1.0)

        assert result.value == 7.25
        assert result.metadata.enum_labels is None
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_a_failed_label_fetch_still_returns_the_reading(self):
        """get_ctrlvars is a round trip to the IOC, and it is allowed to fail."""
        pv = _enum_pv(value=2)
        type(pv).enum_strs = property(
            lambda self: (_ for _ in ()).throw(TimeoutError("ctrlvars timed out"))
        )
        epics = MagicMock()
        epics.PV.return_value = pv
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:MODE", timeout=1.0)

        assert result.value == 2  # the read is not lost with the labels
        assert result.metadata.enum_labels is None
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_unreported_labels_leave_the_fields_unset(self):
        """A PV whose ctrlvars have never been fetched reports enum_strs as None."""
        epics = MagicMock()
        epics.PV.return_value = _enum_pv(value=2, labels=None)
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:MODE", timeout=1.0)

        assert result.value == 2
        assert result.metadata.enum_labels is None
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_an_index_past_the_label_list_keeps_the_list(self):
        """An unresolvable index loses its label, not the states it could not name."""
        epics = MagicMock()
        epics.PV.return_value = _enum_pv(value=9, labels=("OFF", "ON"))
        connector = _connector(epics=epics)

        result = await connector.read_channel("SR:MODE", timeout=1.0)

        assert result.value == 9
        assert result.metadata.enum_labels == ["OFF", "ON"]
        assert result.metadata.enum_label is None

    @pytest.mark.asyncio
    async def test_subscribe_callback_reports_the_label_from_its_kwargs(self, monkeypatch):
        """pyepics hands the monitor callback the PV's whole arg set, enum_strs included."""
        monkeypatch.setattr(
            "osprey.connectors.control_system.epics_connector.get_facility_timezone",
            lambda: __import__("zoneinfo").ZoneInfo("UTC"),
        )
        epics = MagicMock()
        epics.PV.return_value = MagicMock()
        connector = _connector(epics=epics)
        received = []

        await connector.subscribe("SR:MODE", received.append)
        epics_callback = epics.PV.call_args.kwargs["callback"]
        epics_callback(
            pvname="SR:MODE",
            value=3,
            timestamp=1_750_000_000.0,
            enum_strs=("OFFLINE", "STANDBY", "ACQUIRING", "FAULT"),
        )
        await asyncio.sleep(0.01)  # let call_soon_threadsafe flush

        assert received[0].value == 3
        assert received[0].metadata.enum_label == "FAULT"
        assert received[0].metadata.enum_labels == [
            "OFFLINE",
            "STANDBY",
            "ACQUIRING",
            "FAULT",
        ]

    @pytest.mark.asyncio
    async def test_subscribe_callback_without_labels_delivers_the_update_anyway(self, monkeypatch):
        """Until ctrlvars are fetched pyepics passes enum_strs=None; the update still lands."""
        monkeypatch.setattr(
            "osprey.connectors.control_system.epics_connector.get_facility_timezone",
            lambda: __import__("zoneinfo").ZoneInfo("UTC"),
        )
        epics = MagicMock()
        epics.PV.return_value = MagicMock()
        connector = _connector(epics=epics)
        received = []

        await connector.subscribe("SR:MODE", received.append)
        epics_callback = epics.PV.call_args.kwargs["callback"]
        epics_callback(pvname="SR:MODE", value=1, timestamp=1_750_000_000.0, enum_strs=None)
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
        pv = _readback_pv(5.0, status=0, severity=0)
        connector = _write_connector(pv=pv)

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
        pv = _readback_pv(5.0, status=3, severity=2)
        connector = _write_connector(pv=pv)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.alarm_status == "HIHI"
        assert result.alarm_severity == 2
        assert raise_for_write_result(result) is result

    @pytest.mark.asyncio
    async def test_a_mismatch_carries_the_alarm_state_too(self):
        pv = _readback_pv(9.9, status=3, severity=2)
        connector = _write_connector(pv=pv)

        result = await connector.write_channel("SR:CH", 5.0)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.alarm_status == "HIHI"
        assert result.alarm_severity == 2

    @pytest.mark.asyncio
    async def test_an_unconfirmed_write_claims_no_alarm_state(self):
        """Nothing was read, so no alarm state can be claimed."""
        connector = _write_connector(read_error=TimeoutError("ca timeout"))

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
            read_error=TimeoutError("confirmed NO_ALARM severity 0 value 5.0")
        )

        result = await connector.write_channel("SR:CH", 5.0)

        assert "confirmed NO_ALARM" in result.error_message  # the wording did land
        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.alarm_status is None
        assert result.alarm_severity is None
        assert result.observed_value is None
