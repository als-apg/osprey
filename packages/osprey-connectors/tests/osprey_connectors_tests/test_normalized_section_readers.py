"""Runtime readers on a normalized ``control_system`` section.

A build normalizes the retired ``writes_enabled`` leaves into the tier's
``control_system.writes`` key, so a rendered document carries ``writes:
allowed`` (or ``never``) and no ``writes_enabled`` anywhere. Every reader that
decides runtime behaviour must answer from that key: a normalized ``allowed``
render, with the machine's switch on, still selects the write gateway and
writes, and a readonly run on a write-gateway-only block still refuses.

Each section below is normalized: it states ``writes`` and no retired leaf.
"""

import os

import pytest

import osprey_connectors.control_system._base as connector_base
import osprey_connectors.control_system._epics_connector as epics_connector_module
from osprey_connectors.control_system._base import ControlSystemConnector, WriteOutcome
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.ipc._verification import derive_endpoints
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors.types import EPICS, TARGET_LIVE, WRITES_KEY
from osprey_connectors_tests._epics_fakes import (  # noqa: F401 - fixtures, used by name
    FakeChannel,
    InlineCaLoop,
    clean_epics_env,
    install_fake_aioca,
)
from osprey_connectors_tests._harness._control_context_fixtures import write_control_context
from osprey_connectors_tests._write_fakes import ARMED_TARGET, arm_switch, config_reader

READ = {"address": "ro.example.com", "port": 5064}
WRITE = {"address": "wr.example.com", "port": 5084}
SP = "SR:CORR:1:SP"


def _normalized(writes: str, **extra) -> dict:
    """A normalized section: the tier key stated, no ``writes_enabled`` anywhere."""
    section = {"writes": writes, **extra}
    assert "writes_enabled" not in repr(section)
    return section


ALLOWED = _normalized(
    "allowed",
    type=EPICS,
    connector={EPICS: {"gateways": {"read_only": dict(READ), "write_access": dict(WRITE)}}},
)
NEVER = _normalized("never", type=EPICS, connector={EPICS: {}})


def _serve(monkeypatch, section: dict) -> None:
    monkeypatch.setattr("osprey_connectors.config.get_config_value", config_reader(section))


@pytest.fixture
def record_root(tmp_path):
    return tmp_path / "agent-data"


@pytest.fixture
def aioca(monkeypatch):
    return install_fake_aioca(monkeypatch, {SP: FakeChannel()})


# -- The reference monitor's verdict --------------------------------------------


def test_an_allowed_render_lets_the_monitor_pass_the_tier(monkeypatch, record_root):
    # Arrange
    arm_switch(monkeypatch, record_root, ARMED_TARGET)
    _serve(monkeypatch, ALLOWED)

    # Act
    verdict = connector_base._monitor_verdict(ARMED_TARGET)

    # Assert
    assert verdict.allowed, verdict.reason


@pytest.mark.parametrize("target", [ARMED_TARGET, None], ids=["stamped", "unstamped"])
def test_a_never_render_refuses_on_the_tier(monkeypatch, record_root, target):
    # Arrange
    arm_switch(monkeypatch, record_root, ARMED_TARGET)
    _serve(monkeypatch, NEVER)

    # Act
    verdict = connector_base._monitor_verdict(target)

    # Assert
    assert (verdict.allowed, verdict.reason) == (False, "tier_never")


def test_a_never_render_refusal_speaks_the_tier_phrase(monkeypatch, record_root):
    """The refusal is the contract's tier phrase and names no retired leaf."""
    # Arrange
    arm_switch(monkeypatch, record_root, ARMED_TARGET)
    _serve(monkeypatch, NEVER)
    verdict = connector_base._monitor_verdict(ARMED_TARGET)

    # Act
    result = connector_base._verdict_refusal(SP, 1.0, verdict)

    # Assert
    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason == "WRITES_DISABLED"
    assert refusal_phrase("tier_never", verdict.kind) in result.error_message
    assert "writes_enabled" not in result.error_message


def test_an_allowed_render_names_no_tier_reason(monkeypatch, record_root):
    """With the tier allowing and the switch off, only the record's reason is spoken."""
    # Arrange
    arm_switch(monkeypatch, record_root, None)
    write_control_context(record_root, target=ARMED_TARGET, owned_by=None)
    _serve(monkeypatch, ALLOWED)
    verdict = connector_base._monitor_verdict(ARMED_TARGET)

    # Act
    result = connector_base._verdict_refusal(SP, 1.0, verdict)

    # Assert
    assert result.outcome is WriteOutcome.REFUSED
    assert verdict.reason not in (None, "tier_never")
    assert refusal_phrase("tier_never", verdict.kind) not in result.error_message
    assert WRITES_KEY not in result.error_message


# The IPC config reader of the tier is gone: it was ``tier_allows`` over the config's
# ``control_system`` section, target-blind. ``test_tier_writes.py``
# (``test_literal_allowed_arms``, ``test_any_other_value_of_the_new_key_is_never``)
# pins that answer, and the case table's ``tier-*`` rows pin it through the verdict.


# -- Gateway-role selection -----------------------------------------------------


def test_an_allowed_render_derives_the_write_gateway():
    derivation = derive_endpoints({"control_system": ALLOWED}, TARGET_LIVE, readonly_run=False)

    assert derivation.selected_role == "write_access"


def test_a_never_render_derives_the_read_gateway():
    section = dict(ALLOWED, writes="never")

    derivation = derive_endpoints({"control_system": section}, TARGET_LIVE, readonly_run=False)

    assert derivation.selected_role == "read_only"


def test_a_readonly_run_on_an_allowed_render_derives_the_read_gateway():
    derivation = derive_endpoints({"control_system": ALLOWED}, TARGET_LIVE, readonly_run=True)

    assert derivation.selected_role == "read_only"


# -- The connector, end to end --------------------------------------------------


def _armed(monkeypatch, root, *, on: bool) -> None:
    """Serve a normalized allowed section and stamp the simulator target."""
    _serve(monkeypatch, _normalized("allowed"))
    arm_switch(monkeypatch, root, ARMED_TARGET if on else None)
    monkeypatch.setattr(ControlSystemConnector, "_control_target", ARMED_TARGET)
    if not on:
        write_control_context(root, target=ARMED_TARGET, owned_by=None)


@pytest.mark.usefixtures("clean_epics_env")
@pytest.mark.asyncio
async def test_an_allowed_render_with_the_switch_on_selects_the_write_gateway_and_writes(
    monkeypatch, record_root, aioca
):
    # Arrange
    _armed(monkeypatch, record_root, on=True)
    connector = EPICSConnector()

    # Act
    await connector.connect({"gateways": {"read_only": dict(READ), "write_access": dict(WRITE)}})
    result = await connector.write_channel(SP, 2.0, confirm=False)

    # Assert
    assert connector._selected_role == "write_access"
    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"
    assert result.outcome is not WriteOutcome.REFUSED, result.error_message
    assert aioca.puts == [(SP, 2.0, False)]


@pytest.mark.usefixtures("clean_epics_env", "aioca")
@pytest.mark.asyncio
async def test_a_readonly_run_on_a_write_gateway_only_block_still_refuses(monkeypatch, record_root):
    # Arrange
    _armed(monkeypatch, record_root, on=True)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    loop = InlineCaLoop()
    monkeypatch.setattr(epics_connector_module, "get_ca_loop", lambda: loop)
    connector = EPICSConnector()

    # Act / Assert
    with pytest.raises(ConnectionError) as caught:
        await connector.connect({"gateways": {"write_access": dict(WRITE)}})

    assert "no read-only gateway" in str(caught.value)
    assert loop.pointed_at == []
    assert connector._selected_role is None
