"""Gateway selection for the EPICS connector, and the role it keeps.

EPICS Channel Access uses one process-wide CA context, so the connector points
at a single gateway. A read-only gateway rejects writes, so a connector whose
writes the write verdict allows must route through the write-capable gateway.
Selection is defense-in-depth: ``write_access`` is used only when the verdict
allows *and* it is configured; otherwise the connector stays on ``read_only``,
so a machine whose switch is off also has its writes rejected at the network
layer. The one exception is a machine reached only through a write gateway: it
reads through that gateway while off, and every write is refused in software.
A readonly run never selects the write gateway, so on such a machine it refuses
to connect.

``connect()`` stores the role it selected. A write the verdict allows is refused
``switching`` when a write gateway is configured and was not selected — the
switch turned on after the connector chose its gateway — and is never refused
for that reason when the block configures no write gateway.

The verdict is armed the way the monitor reads it (``osprey_connectors_tests._write_fakes``):
a section whose tier allows, a connector stamped with the simulator target, and
the login's switch for it in the record.
"""

import os

import pytest

import osprey_connectors.control_system._epics_connector as epics_connector_module
from osprey_connectors.control_system._base import ControlSystemConnector, WriteOutcome
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.control_system._va_connector import VirtualAcceleratorConnector
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors.types import WRITES_KEY
from osprey_connectors_tests._epics_fakes import (  # noqa: F401 - fixtures, used by name
    FakeChannel,
    InlineCaLoop,
    clean_epics_env,
    install_fake_aioca,
)
from osprey_connectors_tests._harness._control_context_fixtures import write_control_context
from osprey_connectors_tests._write_fakes import (
    ARMED_SECTION,
    ARMED_TARGET,
    arm_switch,
    config_reader,
    turn_switch,
)

# connect() runs against a stand-in aioca and an inline CA loop: the real loop
# would claim this process's Channel Access environment for every later test.
pytestmark = [pytest.mark.usefixtures("aioca", "clean_epics_env"), pytest.mark.asyncio]

READ = {"address": "ro.example.com", "port": 5064}
WRITE = {"address": "wr.example.com", "port": 5084}
SP = "SR:CORR:1:SP"


def _both_gateways():
    return {"read_only": dict(READ), "write_access": dict(WRITE)}


def _serve(monkeypatch, section: dict) -> None:
    """Serve one ``control_system:`` section to every config read."""
    monkeypatch.setattr("osprey_connectors.config.get_config_value", config_reader(section))


@pytest.fixture
def record_root(tmp_path):
    return tmp_path / "agent-data"


def _armed(monkeypatch, root, *, on: bool, section: dict | None = None):
    """Serve *section* (tier allows by default) and stamp the simulator target.

    *on* writes the login's switch on for it; off writes an ownerless record
    with every switch off.
    """
    _serve(monkeypatch, dict(ARMED_SECTION) if section is None else section)
    arm_switch(monkeypatch, root, ARMED_TARGET if on else None)
    monkeypatch.setattr(ControlSystemConnector, "_control_target", ARMED_TARGET)
    if not on:
        write_control_context(root, target=ARMED_TARGET, owned_by=None)


@pytest.fixture
def aioca(monkeypatch):
    """The stand-in aioca behind ``connect()``, answering for :data:`SP`."""
    return install_fake_aioca(monkeypatch, {SP: FakeChannel()})


# -- Selection follows the verdict --------------------------------------------


async def test_write_access_gateway_used_when_the_verdict_allows(monkeypatch, record_root):
    _armed(monkeypatch, record_root, on=True)

    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5084"
    assert connector._selected_role == "write_access"
    assert connector._write_gateway_configured is True


async def test_read_only_gateway_used_while_the_switch_is_off(monkeypatch, record_root):
    _armed(monkeypatch, record_root, on=False)

    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5064"
    assert connector._selected_role == "read_only"


async def test_a_tier_that_never_writes_stays_on_read_only(monkeypatch, record_root):
    _armed(monkeypatch, record_root, on=True, section={"writes": "never"})

    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"


async def test_readonly_run_stays_on_read_only_gateway(monkeypatch, record_root):
    """A read-only run never routes through the write gateway, whatever the switch says."""
    _armed(monkeypatch, record_root, on=True)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5064"


async def test_readwrite_run_uses_write_gateway(monkeypatch, record_root):
    _armed(monkeypatch, record_root, on=True)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readwrite")

    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"


async def test_an_unstamped_connector_stays_on_read_only(monkeypatch, record_root):
    """No stamped machine: the verdict refuses ``target_unknown``."""
    _armed(monkeypatch, record_root, on=True)
    monkeypatch.setattr(ControlSystemConnector, "_control_target", None)

    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"


async def test_va_connector_inherits_the_selection(monkeypatch, record_root):
    """The VA connector adds gateway-port filling, not a selection of its own."""
    _armed(monkeypatch, record_root, on=True)
    connector = VirtualAcceleratorConnector()
    connector._connector_type = "virtual_accelerator"

    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5084"


async def test_warns_when_the_verdict_allows_but_no_write_gateway(monkeypatch, record_root):
    _armed(monkeypatch, record_root, on=True)
    warnings: list[str] = []
    monkeypatch.setattr(
        epics_connector_module.logger,
        "warning",
        lambda msg, *a, **k: warnings.append(str(msg)),
    )
    connector = EPICSConnector()
    connector._connector_type = "virtual_accelerator"

    await connector.connect({"gateways": {"read_only": dict(READ)}})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"
    assert connector._selected_role == "read_only"
    assert connector._write_gateway_configured is False
    assert any(f"{WRITES_KEY}: allowed" in w for w in warnings), warnings


# -- A machine with only a write gateway ---------------------------------------


async def test_a_write_gateway_only_machine_reads_through_it_while_off(
    monkeypatch, record_root, aioca
):
    _armed(monkeypatch, record_root, on=False)

    connector = EPICSConnector()
    await connector.connect({"gateways": {"write_access": dict(WRITE)}})
    reading = await connector.read_channel(SP)
    result = await connector.write_channel(SP, 2.0)

    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"
    assert connector._selected_role == "write_access"
    assert reading.value == 1.0
    assert result.outcome is WriteOutcome.REFUSED
    assert refusal_phrase("switch_off", "simulator") in result.error_message
    assert aioca.puts == []


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
async def test_a_readonly_run_on_a_write_gateway_only_machine_refuses_to_connect(
    monkeypatch, record_root, on
):
    """A readonly run never reads through the write gateway, so it has nothing to connect to."""
    _armed(monkeypatch, record_root, on=on)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    loop = InlineCaLoop()
    monkeypatch.setattr(epics_connector_module, "get_ca_loop", lambda: loop)

    connector = EPICSConnector()
    with pytest.raises(ConnectionError) as caught:
        await connector.connect({"gateways": {"write_access": dict(WRITE)}})

    assert "no read-only gateway" in str(caught.value)
    assert "deployment admin" in str(caught.value)
    # A connect error, not a write refusal: the chip cannot fix it.
    assert refusal_phrase("readonly_run") not in str(caught.value)
    # Refused before Channel Access was pointed anywhere: no ambient-network fallback.
    assert loop.pointed_at == []
    assert "EPICS_CA_ADDR_LIST" not in os.environ
    assert connector._selected_role is None


async def test_a_readonly_run_with_no_gateways_still_connects(monkeypatch, record_root):
    """A gateway-less block (the queue worker's config) is not a write-gateway-only one."""
    _armed(monkeypatch, record_root, on=True)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    connector = EPICSConnector()
    await connector.connect({"timeout_s": 5.0})

    assert connector._selected_role is None
    assert connector._write_gateway_configured is False


async def test_a_write_gateway_only_machine_writes_once_on(monkeypatch, record_root, aioca):
    _armed(monkeypatch, record_root, on=True)

    connector = EPICSConnector()
    await connector.connect({"gateways": {"write_access": dict(WRITE)}})
    result = await connector.write_channel(SP, 2.0, confirm=False)

    assert result.outcome is not WriteOutcome.REFUSED, result.error_message
    assert aioca.puts == [(SP, 2.0, False)]


# -- The stored role -----------------------------------------------------------


async def test_a_connector_lagging_the_switch_refuses_switching(monkeypatch, record_root, aioca):
    """Connected while off, then turned on: the read gateway cannot carry the write."""
    _armed(monkeypatch, record_root, on=False)
    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    turn_switch(ARMED_TARGET, "asked")
    result = await connector.write_channel(SP, 2.0)

    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason == "WRITES_DISABLED"
    assert refusal_phrase("switching", "simulator") in result.error_message
    assert aioca.puts == []


async def test_a_lagging_connector_turned_off_again_refuses_switch_off(monkeypatch, record_root):
    _armed(monkeypatch, record_root, on=False)
    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    turn_switch(ARMED_TARGET, "asked")
    turn_switch(ARMED_TARGET, "off")
    result = await connector.write_channel(SP, 2.0)

    assert refusal_phrase("switch_off", "simulator") in result.error_message


async def test_reconnecting_after_the_turn_on_selects_the_write_gateway(
    monkeypatch, record_root, aioca
):
    _armed(monkeypatch, record_root, on=False)
    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    turn_switch(ARMED_TARGET, "asked")
    await connector.connect({"gateways": _both_gateways()})
    result = await connector.write_channel(SP, 2.0, confirm=False)

    assert connector._selected_role == "write_access"
    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"
    assert result.outcome is not WriteOutcome.REFUSED, result.error_message
    assert aioca.puts == [(SP, 2.0, False)]


async def test_a_connector_on_the_write_gateway_refuses_after_turn_off(
    monkeypatch, record_root, aioca
):
    _armed(monkeypatch, record_root, on=True)
    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    turn_switch(ARMED_TARGET, "off")
    result = await connector.write_channel(SP, 2.0)

    assert refusal_phrase("switch_off", "simulator") in result.error_message
    assert aioca.puts == []


async def test_no_write_gateway_is_never_refused_switching(monkeypatch, record_root, aioca):
    """Connected while off on a read-only block, then turned on: it writes through what it has."""
    _armed(monkeypatch, record_root, on=False)
    connector = EPICSConnector()
    await connector.connect({"gateways": {"read_only": dict(READ)}})

    turn_switch(ARMED_TARGET, "asked")
    result = await connector.write_channel(SP, 2.0, confirm=False)

    assert result.outcome is not WriteOutcome.REFUSED, result.error_message
    assert aioca.puts == [(SP, 2.0, False)]


async def test_a_batch_on_a_lagging_connector_refuses_every_operation(
    monkeypatch, record_root, aioca
):
    _armed(monkeypatch, record_root, on=False)
    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    turn_switch(ARMED_TARGET, "asked")
    results = await connector.write_multiple_channels([(SP, 1.0), (SP, 2.0)])

    assert [r.outcome for r in results] == [WriteOutcome.REFUSED, WriteOutcome.REFUSED]
    assert all(refusal_phrase("switching", "simulator") in r.error_message for r in results)
    assert aioca.puts == []
