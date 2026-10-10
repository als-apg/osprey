"""Where the simulator runs, and the one wire fact every check reads from it.

``control_system.type: virtual_accelerator`` is the simulator in both venues;
``control_system.connector.virtual_accelerator.serving`` says whether it is
served from its container or in this process. Every check that asks "does this
speak Channel Access" or "does this dial a network" reads
:func:`~osprey_connectors.types.connector_transport`, never the type word.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from osprey.connectors import types
from osprey.connectors.types import (
    DOOCS,
    EPICS,
    IN_PROCESS,
    LIVE_STANDIN,
    SERVED,
    TANGO,
    TARGET_LIVE,
    TARGET_VA,
    TRANSPORT_CA,
    TRANSPORT_DOOCS,
    TRANSPORT_IN_PROCESS,
    TRANSPORT_TANGO,
    VIRTUAL_ACCELERATOR,
    baseline_target,
    connector_transport,
    resolve_serving,
    resolve_target,
    speaks_channel_access,
    switch_capable,
    talks_to_network,
)

DOTTED = "my_pkg.module.MyConnector"


def _va(serving: str | None = None, **blocks) -> dict:
    block = {} if serving is None else {"serving": serving}
    return {"type": VIRTUAL_ACCELERATOR, "connector": {VIRTUAL_ACCELERATOR: block, **blocks}}


# (section, serving, transport, speaks CA, talks to network)
ROWS = [
    pytest.param(None, IN_PROCESS, TRANSPORT_IN_PROCESS, False, False, id="no-section"),
    pytest.param({}, IN_PROCESS, TRANSPORT_IN_PROCESS, False, False, id="empty"),
    pytest.param({"type": None}, IN_PROCESS, TRANSPORT_IN_PROCESS, False, False, id="blank-type"),
    pytest.param(
        {"connector": {VIRTUAL_ACCELERATOR: {"serving": SERVED}}},
        IN_PROCESS,
        TRANSPORT_IN_PROCESS,
        False,
        False,
        id="no-type-leaf-unread",
    ),
    pytest.param(_va(), SERVED, TRANSPORT_CA, True, True, id="va-default-served"),
    pytest.param(
        _va(IN_PROCESS), IN_PROCESS, TRANSPORT_IN_PROCESS, False, False, id="va-in-process"
    ),
    pytest.param(_va(SERVED), SERVED, TRANSPORT_CA, True, True, id="va-served"),
    pytest.param({"type": EPICS}, SERVED, TRANSPORT_CA, True, True, id="epics"),
    pytest.param({"type": LIVE_STANDIN}, SERVED, TRANSPORT_CA, True, True, id="live-standin"),
    pytest.param({"type": DOOCS}, SERVED, TRANSPORT_DOOCS, False, True, id="doocs"),
    pytest.param({"type": TANGO}, SERVED, TRANSPORT_TANGO, False, True, id="tango"),
    pytest.param({"type": DOTTED}, SERVED, None, False, True, id="dotted"),
]


@pytest.mark.parametrize(("section", "serving", "transport", "ca", "network"), ROWS)
def test_the_venue_decides_the_wire(section, serving, transport, ca, network):
    assert resolve_serving(section) == serving
    assert connector_transport(section) == transport
    assert speaks_channel_access(section) is ca
    assert talks_to_network(section) is network


def test_an_unknown_serving_value_is_refused():
    with pytest.raises(ValueError, match=r"serving.*'served'.*'in_process'"):
        resolve_serving(_va("container"))


def test_the_type_argument_overrides_the_section_type():
    """A target's type, not the section's own, picks the row."""
    section = {"type": EPICS, "connector": {VIRTUAL_ACCELERATOR: {}}}
    assert connector_transport(section, VIRTUAL_ACCELERATOR) == TRANSPORT_CA


@pytest.mark.parametrize(
    "section", [None, {}, _va(IN_PROCESS)], ids=["no-type", "empty", "va-in-process"]
)
def test_the_in_process_simulator_baselines_on_va(section):
    assert baseline_target(section) == TARGET_VA


def test_in_process_simulator_beside_a_live_block_is_switchable():
    section = _va(IN_PROCESS, epics={"gateways": {"read_only": {"address": "gw", "port": 5064}}})
    assert switch_capable(section) is True
    assert resolve_target(section, TARGET_LIVE) == EPICS


def test_retired_message_names_the_new_spelling():
    message = types.retired_type_message("mock")
    assert "`mock` is retired" in message
    assert "control_system.connector.virtual_accelerator.serving: in_process" in message
    assert (
        "osprey set connector=virtual_accelerator "
        "config.control_system.connector.virtual_accelerator.serving=in_process"
    ) in message
    assert types.RETIRED_CONTROL_SYSTEM_TYPES == {"mock": VIRTUAL_ACCELERATOR}


# ---------------------------------------------------------------------------
# The factory builds the venue the leaf names, and stamps its wire
# ---------------------------------------------------------------------------


@pytest.fixture
def builtin_connectors():
    from osprey_connectors.factory import isolated_connector_registries, register_builtin_connectors

    with isolated_connector_registries(clear=True):
        register_builtin_connectors()
        yield


@pytest.mark.usefixtures("builtin_connectors")
@pytest.mark.parametrize(
    ("config", "class_name", "transport"),
    [
        pytest.param(
            _va(IN_PROCESS), "VAInProcessConnector", TRANSPORT_IN_PROCESS, id="in-process"
        ),
        pytest.param({}, "VAInProcessConnector", TRANSPORT_IN_PROCESS, id="no-type"),
        pytest.param(_va(), "VirtualAcceleratorConnector", TRANSPORT_CA, id="va-alone"),
        pytest.param(
            {
                "type": VIRTUAL_ACCELERATOR,
                "connector": {VIRTUAL_ACCELERATOR: {"timeout_s": 5.0}, EPICS: {}},
            },
            "VirtualAcceleratorConnector",
            TRANSPORT_CA,
            id="va-target-on-a-live-deployment",
        ),
    ],
)
def test_factory_builds_the_venue_the_leaf_names(config, class_name, transport):
    from osprey_connectors.factory import ConnectorFactory

    connector, _type_config = ConnectorFactory.build_control_system_connector(config)

    assert type(connector).__name__ == class_name
    assert connector.transport == transport
    assert connector._connector_type == VIRTUAL_ACCELERATOR


@pytest.mark.usefixtures("builtin_connectors")
def test_factory_refuses_mock_with_the_new_spelling():
    from osprey_connectors.factory import ConnectorFactory

    with pytest.raises(ValueError) as caught:
        ConnectorFactory.build_control_system_connector({"type": "mock"})

    assert str(caught.value) == types.retired_type_message("mock")


def test_an_unbuilt_connector_names_no_transport():
    from osprey_connectors.control_system.va_in_process_connector import VAInProcessConnector

    assert VAInProcessConnector().transport is None


# ---------------------------------------------------------------------------
# No reader names the wire by its type
# ---------------------------------------------------------------------------

#: The modules that may compare a type to the simulator's: the vocabulary that
#: computes the transport, the factory that picks the venue from it, the
#: derivation's port fill for the served venue, and the two hooks' literal
#: tables.
_TYPE_COMPARISON_HOMES = {
    "packages/osprey-connectors/src/osprey_connectors/types.py",
    "packages/osprey-connectors/src/osprey_connectors/factory.py",
    "packages/osprey-connectors/src/osprey_connectors/ipc/verification.py",
    "src/osprey/templates/claude_code/claude/hooks/osprey_target_state.py",
    "src/osprey/templates/claude_code/claude/hooks/osprey_approval.py",
}

_TYPE_COMPARISON = re.compile(
    r"==\s*(?:types\.)?VIRTUAL_ACCELERATOR\b|==\s*[\"']virtual_accelerator[\"']"
)


def test_no_reader_names_the_wire_by_type():
    """Every check asks the transport; none spells the type list it used to."""
    root = Path(__file__).resolve().parents[2]
    listed = subprocess.run(
        ["git", "ls-files", "src", "packages"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    offenders = []
    for relative in listed:
        if not relative.endswith(".py"):
            continue
        text = (root / relative).read_text(encoding="utf-8")
        if "CHANNEL_ACCESS_TYPES" in text:
            offenders.append(f"{relative}: CHANNEL_ACCESS_TYPES")
        if relative not in _TYPE_COMPARISON_HOMES and _TYPE_COMPARISON.search(text):
            offenders.append(f"{relative}: compares a type to the simulator's")
    assert offenders == []
