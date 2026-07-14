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

import osprey.connectors.control_system.epics_connector as epics_module
from osprey.connectors.control_system.base import ChannelMetadata, ChannelValue
from osprey.connectors.control_system.epics_connector import (
    EPICSConnector,
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
        connector._subscriptions = {"sub1": sub_pv}
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
        assert connector._subscriptions[sub_id] is pv

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


# Keep a reference to the module import so a wildcard-style unused-import lint
# never trims it; connect() patches attributes on this module by path.
assert epics_module.EPICSConnector is EPICSConnector
