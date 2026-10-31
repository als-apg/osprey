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

from unittest.mock import MagicMock

import pytest

from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.control_system._limits_validator import (
    DEFAULT_STEP_READ_TIMEOUT_SECONDS,
    ChannelLimitsConfig,
    LimitsValidator,
    step_read_timeout_seconds,
)
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.errors import ChannelLimitsViolationError
from osprey_connectors_tests._epics_fakes import FakeChannel, ca_connector, ca_reading
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from tests.connectors._driver_fakes import driven_through
from tests.facility.served_tree import in_process_config, served_tree

CURRENT = 10.0


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch, tmp_path):
    """The login's switch on for the simulator target, connectors stamped with it.

    With :func:`writes_enabled_config` serving a write tier that allows, that is
    everything the reference monitor's verdict asks before a write reaches the
    transport; a test that disarms serves a tier that does not.
    """
    return arm_switch(monkeypatch, tmp_path / "armed-data")


def _step_validator(channel: str, max_step: float = 5.0) -> LimitsValidator:
    limits = {
        channel: ChannelLimitsConfig(
            channel_address=channel,
            min_value=0.0,
            max_value=100.0,
            max_step=max_step,
            writable=True,
        )
    }
    return LimitsValidator(limits, {"mode": "exclusive"}, {})


# ---------------------------------------------------------------------------
# Each connector's reader answers from its own client
# ---------------------------------------------------------------------------


async def test_the_simulator_reads_its_own_store(monkeypatch, tmp_path):
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
    connector = VAInProcessConnector()
    await connector.connect(
        in_process_config(served_tree(tmp_path, ["SIM:CHANNEL:SP"]), response_delay_ms=0)
    )
    connector._composite.set({"SIM:CHANNEL:SP": CURRENT})

    assert connector.current_value_reader()("SIM:CHANNEL:SP") == CURRENT

    await connector.disconnect()


def _ca_connector_holding(value) -> EPICSConnector:
    """An EPICS connector whose Channel Access client answers ``value`` for SR:CH."""
    return ca_connector(channels={"SR:CH": FakeChannel(ca_reading(value))})


def test_the_epics_connector_reads_with_the_client_it_connected_with():
    connector = _ca_connector_holding(CURRENT)

    assert connector.current_value_reader()("SR:CH") == CURRENT
    assert [c[:2] for c in connector._aioca.calls] == [("caget", "SR:CH")]


def test_the_epics_connector_refuses_to_read_a_pva_routed_address():
    """A PVA address must never be read with the CA client.

    ``write_channel`` refuses those before validation runs; this is the same
    rule stated where the read is made. ``None`` is what the step check treats
    as "could not read", so the write fails closed.
    """
    connector = _ca_connector_holding(CURRENT)
    connector._pva_channel_globs = ["PVA:*"]

    assert connector.current_value_reader()("PVA:IMAGE") is None
    assert connector._aioca.calls == []


async def test_the_doocs_connector_reads_with_doocs4py():
    eq_data = MagicMock()
    eq_data.get_data.return_value = CURRENT
    doocs4py = MagicMock()
    doocs4py.__version__ = "2.0.0"
    doocs4py.names.return_value = [("FACILITY", "XFEL")]
    doocs4py.get.return_value = eq_data

    with driven_through("doocs4py", doocs4py):
        from osprey_connectors.control_system._doocs_connector import DOOCSConnector

        conn = DOOCSConnector()
        await conn.connect({})

        assert conn.current_value_reader()("FAC/DEV/LOC/PROP") == CURRENT
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

    with driven_through("tango", tango):
        from osprey_connectors.control_system._tango_connector import TangoConnector

        conn = TangoConnector()
        await conn.connect({})

        assert conn.current_value_reader()("sr/power_supply/ps01/Current") == CURRENT
        proxy.read_attribute.assert_called_once_with("Current")

        await conn.disconnect()


# ---------------------------------------------------------------------------
# ... and the limit is now enforceable off Channel Access
# ---------------------------------------------------------------------------


async def test_max_step_blocks_an_oversized_step_on_the_simulator(monkeypatch, tmp_path):
    """The defect, from the operator's end: max_step on a non-CA connector.

    No Channel Access read could ever answer for a simulated channel, so this
    write used to be refused as unverifiable however small the step was.
    """
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
    connector = VAInProcessConnector()
    await connector.connect(
        in_process_config(served_tree(tmp_path, ["SIM:CHANNEL:SP"]), response_delay_ms=0)
    )
    connector._composite.set({"SIM:CHANNEL:SP": CURRENT})
    connector._limits_validator = _step_validator("SIM:CHANNEL:SP", max_step=5.0)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        await connector.write_channel("SIM:CHANNEL:SP", 90.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert connector._composite.held(["SIM:CHANNEL:SP"])["SIM:CHANNEL:SP"] == CURRENT

    await connector.disconnect()


async def test_max_step_lets_a_small_step_through_on_the_simulator(monkeypatch, tmp_path):
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
    connector = VAInProcessConnector()
    await connector.connect(
        in_process_config(served_tree(tmp_path, ["SIM:CHANNEL:SP"]), response_delay_ms=0)
    )
    connector._composite.set({"SIM:CHANNEL:SP": CURRENT})
    connector._limits_validator = _step_validator("SIM:CHANNEL:SP", max_step=5.0)

    result = await connector.write_channel("SIM:CHANNEL:SP", 12.0)

    assert result.outcome is WriteOutcome.CONFIRMED
    assert connector._composite.held(["SIM:CHANNEL:SP"])["SIM:CHANNEL:SP"] == 12.0

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


def test_the_epics_connector_reads_with_the_budget_its_block_declares() -> None:
    """The connector's own block is where the budget comes from."""
    connector = _ca_connector_holding(CURRENT)
    connector._step_read_timeout = step_read_timeout_seconds({"step_read_timeout_s": 0.5})

    connector.current_value_reader()("SR:CH")

    assert connector._aioca.calls[-1][2]["timeout"] == 0.5


def test_an_unconnected_connector_still_has_a_budget() -> None:
    """The reader is usable before connect, and fails closed just as quickly."""
    assert EPICSConnector()._step_read_timeout == DEFAULT_STEP_READ_TIMEOUT_SECONDS
