"""Behavioral tests for the EPICS control-system connector.

No general EPICS connector test existed before this file (see the note in
``test_epics_connector_timezone.py``); gateway selection and the read timestamp
were the only covered paths. These tests drive the connector's remaining real
code paths — libca configuration, connect error/name-server handling, the
write-verification result matrix (none/callback/readback), the fail-closed
write guard, and subscription plumbing — with an injected fake ``_epics`` so no
real Channel Access is required.

Convention (matching ``test_epics_connector_timezone.py`` and PR #270): inject a
fake ``_epics`` and assert on the concrete payload — verification level, notes,
env vars, refusal reason — never merely that a call "didn't raise".
"""

import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.connectors.control_system.base import ChannelMetadata, ChannelValue
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
            classmethod(lambda cls: sentinel),
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
# write_channel — verification result matrix
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("writes_enabled")
class TestWriteVerification:
    @pytest.mark.asyncio
    async def test_none_success(self):
        epics = MagicMock()
        epics.caput.return_value = True
        connector = _connector(epics=epics)

        result = await connector.write_channel("SR:CH", 1.0, verification_level="none")

        assert result.success is True
        assert result.verification.level == "none"
        assert result.verification.verified is False
        assert "No verification requested" in result.verification.notes
        # none path must not wait on an IOC callback.
        assert epics.caput.call_args.kwargs["wait"] is False

    @pytest.mark.asyncio
    async def test_none_failure(self):
        epics = MagicMock()
        epics.caput.return_value = False
        connector = _connector(epics=epics)

        result = await connector.write_channel("SR:CH", 1.0, verification_level="none")

        assert result.success is False
        assert "Write command failed" in result.verification.notes
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_callback_success(self):
        epics = MagicMock()
        epics.caput.return_value = True
        connector = _connector(epics=epics)

        result = await connector.write_channel("SR:CH", 1.0, verification_level="callback")

        assert result.success is True
        assert result.verification.verified is True
        assert "IOC callback confirmed" in result.verification.notes
        assert epics.caput.call_args.kwargs["wait"] is True

    @pytest.mark.asyncio
    async def test_callback_failure(self):
        epics = MagicMock()
        epics.caput.return_value = False
        connector = _connector(epics=epics)

        result = await connector.write_channel("SR:CH", 1.0, verification_level="callback")

        assert result.success is False
        assert result.verification.verified is False
        assert "IOC callback failed or timeout" in result.verification.notes

    @pytest.mark.asyncio
    async def test_readback_verified_within_tolerance(self, monkeypatch):
        epics = MagicMock()
        epics.caput.return_value = True
        connector = _connector(epics=epics)
        readback = ChannelValue(value=5.0005, timestamp=None, metadata=ChannelMetadata())
        monkeypatch.setattr(connector, "read_channel", AsyncMock(return_value=readback))

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        assert result.success is True
        assert result.verification.verified is True
        assert result.verification.readback_value == pytest.approx(5.0005)
        assert result.verification.tolerance_used == 0.01

    @pytest.mark.asyncio
    async def test_readback_mismatch_reports_unverified(self, monkeypatch):
        epics = MagicMock()
        epics.caput.return_value = True
        connector = _connector(epics=epics)
        readback = ChannelValue(value=9.9, timestamp=None, metadata=ChannelMetadata())
        monkeypatch.setattr(connector, "read_channel", AsyncMock(return_value=readback))

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        # The write command itself succeeded; verification did not.
        assert result.success is True
        assert result.verification.verified is False
        assert "Readback mismatch" in result.verification.notes

    @pytest.mark.asyncio
    async def test_readback_caput_failure(self, monkeypatch):
        """A failed caput on the readback path returns failure without reading back."""
        epics = MagicMock()
        epics.caput.return_value = False
        connector = _connector(epics=epics)
        read = AsyncMock()
        monkeypatch.setattr(connector, "read_channel", read)

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        assert result.success is False
        assert "Write command failed" in result.verification.notes
        read.assert_not_called()  # no readback attempted when the write itself failed

    @pytest.mark.asyncio
    async def test_readback_exception_is_non_fatal(self, monkeypatch):
        """A readback that raises leaves the write successful but unverified."""
        epics = MagicMock()
        epics.caput.return_value = True
        connector = _connector(epics=epics)
        monkeypatch.setattr(
            connector, "read_channel", AsyncMock(side_effect=TimeoutError("ca timeout"))
        )

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        assert result.success is True
        assert result.verification.verified is False
        assert "Readback failed" in result.verification.notes
        assert "ca timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_auto_verification_level_from_global_config(self, monkeypatch):
        """With no override and no per-channel config, the global default level is used."""
        epics = MagicMock()
        epics.caput.return_value = True
        connector = _connector(epics=epics, limits_validator=None)

        def fake_get_config_value(key, default=None):
            if key == "control_system.write_verification.default_level":
                return "none"
            return default

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)

        result = await connector.write_channel("SR:CH", 2.0)  # no verification_level

        assert result.verification.level == "none"

    @pytest.mark.asyncio
    async def test_explicit_tolerance_survives_auto_level(self, monkeypatch):
        """An explicit tolerance is kept even when the level is auto-resolved."""
        epics = MagicMock()
        epics.caput.return_value = True
        connector = _connector(epics=epics, limits_validator=None)
        readback = ChannelValue(value=5.0, timestamp=None, metadata=ChannelMetadata())
        monkeypatch.setattr(connector, "read_channel", AsyncMock(return_value=readback))

        def fake_get_config_value(key, default=None):
            if key == "control_system.write_verification.default_level":
                return "readback"
            return default

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)

        # verification_level is None (auto), but tolerance is explicitly provided.
        result = await connector.write_channel("SR:CH", 5.0, tolerance=0.25)

        assert result.verification.level == "readback"
        assert result.verification.tolerance_used == 0.25


# ---------------------------------------------------------------------------
# write_channel — fail-closed guard
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("writes_enabled")
class TestWriteFailClosed:
    @pytest.mark.asyncio
    async def test_invalid_level_rejected_before_any_caput(self):
        epics = MagicMock()
        connector = _connector(epics=epics)

        with pytest.raises(ValueError, match="Invalid verification_level"):
            await connector.write_channel("SR:CH", 1.0, verification_level="bogus")

        epics.caput.assert_not_called()

    @pytest.mark.asyncio
    async def test_validation_error_refuses_write_without_caput(self):
        """A non-limits validation error fails closed: refused, blocked, no caput."""
        epics = MagicMock()
        limits = MagicMock()
        limits.validate.side_effect = RuntimeError("db unreadable")
        connector = _connector(epics=epics, limits_validator=limits)

        result = await connector.write_channel("SR:CH", 1.0, verification_level="none")

        assert result.success is False
        assert result.blocked is True
        assert result.refusal_reason == "VALIDATION_ERROR"
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
            await connector.write_channel("SR:CH", 1.0, verification_level="none")

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
# write_channel — readback alarm reporting and failure classification
# ---------------------------------------------------------------------------


def _readback(value, *, alarm=None, severity=None):
    """A readback ChannelValue, optionally carrying alarm metadata."""
    raw = {} if severity is None else {"severity": severity}
    return ChannelValue(
        value=value,
        timestamp=None,
        metadata=ChannelMetadata(alarm_status=alarm, raw_metadata=raw),
    )


def _write_connector(monkeypatch, *, readback=None, raises=None, limits=None):
    """A connector whose caput succeeds, with ``read_channel`` stubbed.

    Every write test below needs the same setup and differs only in what the
    readback does, so that is the only thing left at the call site.
    """
    epics = MagicMock()
    epics.caput.return_value = True
    connector = _connector(epics=epics, limits_validator=limits)
    if raises is not None:
        monkeypatch.setattr(connector, "read_channel", AsyncMock(side_effect=raises))
    elif readback is not None:
        monkeypatch.setattr(connector, "read_channel", AsyncMock(return_value=readback))
    return connector


@pytest.mark.usefixtures("writes_enabled")
class TestReadbackAlarmReporting:
    """The readback carries the channel's alarm state into the result.

    These are the *structured* fields a consumer classifies a write from;
    ``notes`` stays display-only and is never parsed.
    """

    @pytest.mark.asyncio
    async def test_healthy_readback_reports_severity_zero(self, monkeypatch):
        """Severity 0 is a reported healthy value — not "unreported" (None)."""
        connector = _write_connector(
            monkeypatch, readback=_readback(5.0, alarm="NO_ALARM", severity=0)
        )

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        assert result.verification.verified is True
        assert result.verification.readback_alarm_status == "NO_ALARM"
        assert result.verification.readback_alarm_severity == 0
        assert result.verification.readback_alarm_severity is not None
        assert result.verification.failure_kind is None

    @pytest.mark.asyncio
    async def test_matching_readback_with_major_alarm_stays_verified(self, monkeypatch):
        """The value matched, so the write verified — but the alarm is reported.

        Suppressing the alarm here is exactly the narration hole #465 describes:
        the caller needs both facts, not one of them.
        """
        connector = _write_connector(monkeypatch, readback=_readback(5.0, alarm="HIHI", severity=2))

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        assert result.verification.verified is True
        assert result.verification.readback_alarm_status == "HIHI"
        assert result.verification.readback_alarm_severity == 2
        assert result.verification.failure_kind is None

    @pytest.mark.asyncio
    async def test_absent_alarm_metadata_leaves_both_fields_null(self, monkeypatch):
        """A connector that reports no alarm state says so with None, not 0."""
        connector = _write_connector(monkeypatch, readback=_readback(5.0))

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        assert result.verification.verified is True
        assert result.verification.readback_alarm_status is None
        assert result.verification.readback_alarm_severity is None

    @pytest.mark.asyncio
    async def test_active_alarm_mismatch_reports_alarm_without_failure_kind(self, monkeypatch):
        """A value mismatch is not a failure *kind*: the readback itself worked."""
        connector = _write_connector(monkeypatch, readback=_readback(9.9, alarm="HIHI", severity=2))

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        assert result.success is True
        assert result.verification.verified is False
        assert result.verification.readback_alarm_status == "HIHI"
        assert result.verification.readback_alarm_severity == 2
        assert result.verification.failure_kind is None

    @pytest.mark.asyncio
    async def test_plain_mismatch_leaves_failure_kind_null(self, monkeypatch):
        connector = _write_connector(
            monkeypatch, readback=_readback(9.9, alarm="NO_ALARM", severity=0)
        )

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        assert result.verification.verified is False
        assert result.verification.failure_kind is None

    @pytest.mark.asyncio
    async def test_readback_exception_sets_failure_kind(self, monkeypatch):
        """A readback that raised is classified, not left indistinguishable."""
        connector = _write_connector(monkeypatch, raises=TimeoutError("ca timeout"))

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        assert result.verification.failure_kind == "readback_failed"
        assert result.verification.verified is False
        # Nothing was read, so no alarm state can be claimed.
        assert result.verification.readback_alarm_status is None
        assert result.verification.readback_alarm_severity is None

    @pytest.mark.asyncio
    async def test_notes_text_never_feeds_the_structured_fields(self, monkeypatch):
        """Notes are display-only: their wording changes nothing a consumer reads.

        The exception message is echoed verbatim into ``notes``; wording it to
        look like a healthy verified readback must not move a single field.
        """
        connector = _write_connector(
            monkeypatch, raises=TimeoutError("verified NO_ALARM severity 0 tolerance ok")
        )

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.01
        )

        assert "verified NO_ALARM" in result.verification.notes  # the wording did land
        assert result.verification.verified is False
        assert result.verification.readback_alarm_status is None
        assert result.verification.readback_alarm_severity is None
        assert result.verification.failure_kind == "readback_failed"


# ---------------------------------------------------------------------------
# write_channel — tolerance resolution under an explicit level
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("writes_enabled")
class TestExplicitLevelToleranceResolution:
    """Naming the level must not silently discard the configured tolerance.

    Passing ``verification_level="readback"`` used to skip config resolution
    entirely, falling back to 0.001 *absolute* — far tighter than the
    percentage tolerances shipped for setpoint channels, so ordinary writes
    reported as mismatches.
    """

    @pytest.mark.asyncio
    async def test_explicit_level_still_resolves_the_limits_db_tolerance(self, monkeypatch):
        limits = MagicMock()
        limits.get_verification_config.return_value = ("readback", 0.05)
        connector = _write_connector(monkeypatch, readback=_readback(5.04), limits=limits)

        # Explicit level, no tolerance: 0.04 is inside the configured 0.05 but
        # far outside the 0.001 absolute fallback.
        result = await connector.write_channel("SR:CH", 5.0, verification_level="readback")

        assert result.verification.tolerance_used == 0.05
        assert result.verification.verified is True

    @pytest.mark.asyncio
    async def test_explicit_tolerance_still_wins_over_the_limits_db(self, monkeypatch):
        limits = MagicMock()
        limits.get_verification_config.return_value = ("readback", 0.05)
        connector = _write_connector(monkeypatch, readback=_readback(5.0), limits=limits)

        result = await connector.write_channel(
            "SR:CH", 5.0, verification_level="readback", tolerance=0.5
        )

        assert result.verification.tolerance_used == 0.5

    @pytest.mark.asyncio
    async def test_explicit_level_is_not_overridden_by_the_limits_db_level(self, monkeypatch):
        """Only the tolerance is resolved — the caller's level is authoritative."""
        limits = MagicMock()
        limits.get_verification_config.return_value = ("none", None)
        connector = _write_connector(monkeypatch, readback=_readback(5.0), limits=limits)

        result = await connector.write_channel("SR:CH", 5.0, verification_level="readback")

        assert result.verification.level == "readback"

    @pytest.mark.asyncio
    async def test_explicit_none_level_does_not_consult_the_config(self, monkeypatch):
        """Tolerance is meaningless without a readback: no resolution happens."""
        limits = MagicMock()
        limits.get_verification_config.side_effect = AssertionError(
            "verification config resolved for a level that performs no readback"
        )
        connector = _write_connector(monkeypatch, limits=limits)

        result = await connector.write_channel("SR:CH", 1.0, verification_level="none")

        assert result.verification.level == "none"
