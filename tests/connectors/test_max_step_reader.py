"""``max_step`` is measured through the connector that is doing the write.

The shared ``LimitsValidator`` owns no control-system client. It used to
``import epics`` and ``caget`` the address inside the step check, which meant
the one optional safety limit that needs a fresh read only ever worked on
Channel Access — and there, over the process-wide ``EPICS_CA_*`` environment
rather than the client the write itself goes through.

Each connector now hands the validator its own fresh-read primitive. These
tests hold each one to reading through its OWN client, and hold the simulator
— which no Channel Access read could ever have measured — to enforcing the
limit for real.
"""

import sys
from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.control_system.base import WriteOutcome
from osprey.connectors.control_system.epics_connector import EPICSConnector
from osprey.connectors.control_system.limits_validator import (
    DEFAULT_STEP_READ_TIMEOUT_SECONDS,
    ChannelLimitsConfig,
    LimitsValidator,
    step_read_timeout_seconds,
)
from osprey.connectors.control_system.mock_connector import MockConnector
from osprey.errors import ChannelLimitsViolationError

_DOOCS_LIMITS_PATCH = "osprey.connectors.control_system.doocs_connector.LimitsValidator.from_config"
_DOOCS_TZ_PATCH = "osprey.connectors.control_system.doocs_connector.get_facility_timezone"
_TANGO_LIMITS_PATCH = "osprey.connectors.control_system.tango_connector.LimitsValidator.from_config"
_TANGO_TZ_PATCH = "osprey.connectors.control_system.tango_connector.get_facility_timezone"

CURRENT = 10.0


def _writes_enabled(key, default=None):
    if key == "control_system.writes_enabled":
        return True
    return default


def _step_validator(channel: str, max_step: float = 5.0) -> LimitsValidator:
    limits = {
        channel: ChannelLimitsConfig(
            channel_address=channel, min_value=0.0, max_value=100.0, max_step=max_step
        )
    }
    return LimitsValidator(limits, {"allow_unlisted_channels": False}, {})


# ---------------------------------------------------------------------------
# Each connector's reader answers from its own client
# ---------------------------------------------------------------------------


async def test_the_simulator_reads_its_own_store(monkeypatch):
    monkeypatch.setattr("osprey.utils.config.get_config_value", _writes_enabled)
    connector = MockConnector()
    await connector.connect({"response_delay_ms": 0, "noise_level": 0.0})
    connector._state["SIM:CHANNEL:SP"] = CURRENT

    assert connector._current_value_reader()("SIM:CHANNEL:SP") == CURRENT

    await connector.disconnect()


def test_the_epics_connector_reads_with_the_client_it_connected_with():
    connector = EPICSConnector()
    connector._epics = MagicMock()
    connector._epics.caget.return_value = CURRENT

    assert connector._current_value_reader()("SR:CH") == CURRENT
    connector._epics.caget.assert_called_once()
    assert connector._epics.caget.call_args.args[0] == "SR:CH"


def test_the_epics_connector_refuses_to_read_a_pva_routed_address():
    """A PVA address must never be read with the CA client.

    ``write_channel`` refuses those before validation runs; this is the same
    rule stated where the read is made. ``None`` is what the step check treats
    as "could not read", so the write fails closed.
    """
    connector = EPICSConnector()
    connector._epics = MagicMock()
    connector._pva_channel_globs = ["PVA:*"]

    assert connector._current_value_reader()("PVA:IMAGE") is None
    connector._epics.caget.assert_not_called()


async def test_the_doocs_connector_reads_with_doocs4py():
    eq_data = MagicMock()
    eq_data.get_data.return_value = CURRENT
    doocs4py = MagicMock()
    doocs4py.__version__ = "2.0.0"
    doocs4py.names.return_value = [("FACILITY", "XFEL")]
    doocs4py.get.return_value = eq_data

    with (
        patch.dict(sys.modules, {"doocs4py": doocs4py}),
        patch(_DOOCS_LIMITS_PATCH, return_value=None),
        patch(_DOOCS_TZ_PATCH, return_value=UTC),
    ):
        from osprey.connectors.control_system.doocs_connector import DOOCSConnector

        conn = DOOCSConnector()
        await conn.connect({})

        assert conn._current_value_reader()("FAC/DEV/LOC/PROP") == CURRENT
        doocs4py.get.assert_called_once_with("FAC/DEV/LOC/PROP")

        await conn.disconnect()


async def test_the_tango_connector_reads_with_its_device_proxy():
    attribute = MagicMock()
    attribute.value = CURRENT
    proxy = MagicMock()
    proxy.read_attribute.return_value = attribute

    tango = MagicMock()
    tango.__version__ = "10.0.0"
    tango.DeviceProxy.return_value = proxy
    tango.Database.return_value.get_info.return_value = "TANGO Database sys/database/2"

    with (
        patch.dict(sys.modules, {"tango": tango}),
        patch(_TANGO_LIMITS_PATCH, return_value=None),
        patch(_TANGO_TZ_PATCH, return_value=UTC),
    ):
        from osprey.connectors.control_system.tango_connector import TangoConnector

        conn = TangoConnector()
        await conn.connect({})

        assert conn._current_value_reader()("sr/power_supply/ps01/Current") == CURRENT
        proxy.read_attribute.assert_called_once_with("Current")

        await conn.disconnect()


# ---------------------------------------------------------------------------
# ... and the limit is now enforceable off Channel Access
# ---------------------------------------------------------------------------


async def test_max_step_blocks_an_oversized_step_on_the_simulator(monkeypatch):
    """The defect, from the operator's end: max_step on a non-CA connector.

    No Channel Access read could ever answer for a simulated channel, so this
    write used to be refused as unverifiable however small the step was.
    """
    monkeypatch.setattr("osprey.utils.config.get_config_value", _writes_enabled)
    connector = MockConnector()
    await connector.connect({"response_delay_ms": 0, "noise_level": 0.0})
    connector._state["SIM:CHANNEL:SP"] = CURRENT
    connector._limits_validator = _step_validator("SIM:CHANNEL:SP", max_step=5.0)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        await connector.write_channel("SIM:CHANNEL:SP", 90.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert connector._state["SIM:CHANNEL:SP"] == CURRENT

    await connector.disconnect()


async def test_max_step_lets_a_small_step_through_on_the_simulator(monkeypatch):
    monkeypatch.setattr("osprey.utils.config.get_config_value", _writes_enabled)
    connector = MockConnector()
    await connector.connect({"response_delay_ms": 0, "noise_level": 0.0})
    connector._state["SIM:CHANNEL:SP"] = CURRENT
    connector._limits_validator = _step_validator("SIM:CHANNEL:SP", max_step=5.0)

    result = await connector.write_channel("SIM:CHANNEL:SP", 12.0)

    assert result.outcome is WriteOutcome.CONFIRMED
    assert connector._state["SIM:CHANNEL:SP"] == 12.0

    await connector.disconnect()


# ---------------------------------------------------------------------------
# The fresh-read budget is a connector key, not a framework constant
#
# How long that read takes is a property of the facility's control network —
# a gateway two hops away answers slower than a soft IOC on this host — so it
# is authored beside the connector's own `timeout`. Running out of budget
# answers None, which refuses the write, so raising it buys a slow channel
# room and never a weaker check.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    [None, {}, {"timeout": 5.0}, {"step_read_timeout_s": 0}, {"step_read_timeout_s": "2"}],
    ids=["absent-block", "empty", "other-keys", "zero", "string"],
)
def test_an_unstated_or_unusable_budget_takes_the_default(block) -> None:
    """A typo in a tuning key must not be what takes writes down."""
    assert step_read_timeout_seconds(block) == DEFAULT_STEP_READ_TIMEOUT_SECONDS


def test_an_unusable_budget_is_warned_about_by_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A silent fallback leaves an operator with a key that reads as accepted."""
    with caplog.at_level("WARNING"):
        assert (
            step_read_timeout_seconds({"step_read_timeout_s": -1}, "epics")
            == DEFAULT_STEP_READ_TIMEOUT_SECONDS
        )

    assert "control_system.connector.epics.step_read_timeout_s" in caplog.text
    assert "-1" in caplog.text


def test_a_block_declaring_nothing_warns_about_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unset is the shipped posture, not a misconfiguration."""
    with caplog.at_level("WARNING"):
        step_read_timeout_seconds({"timeout": 5.0}, "epics")

    assert caplog.text == ""


def test_a_declared_budget_is_read() -> None:
    assert step_read_timeout_seconds({"step_read_timeout_s": 0.5}) == 0.5
    assert step_read_timeout_seconds({"step_read_timeout_s": 10}) == 10.0


async def test_the_epics_connector_reads_with_the_budget_its_block_declares() -> None:
    """The connector's own block is where the budget comes from."""
    connector = EPICSConnector()
    connector._epics = MagicMock()
    connector._epics.caget.return_value = CURRENT
    connector._step_read_timeout = step_read_timeout_seconds({"step_read_timeout_s": 0.5})

    connector._current_value_reader()("SR:CH")

    assert connector._epics.caget.call_args.kwargs["timeout"] == 0.5


def test_an_unconnected_connector_still_has_a_budget() -> None:
    """The reader is usable before connect, and fails closed just as quickly."""
    assert EPICSConnector()._step_read_timeout == DEFAULT_STEP_READ_TIMEOUT_SECONDS
