"""An address the facility does not hold is answered as not found, read or write.

The controls tools are driven end to end over the real in-process simulator: a
served tree holding ``A:SP`` and ``A:RB``, the connector the factory would build
for it, writes armed (a write tier that allows, the simulator's switch on) and no
limits validator. Each write is stamped as the approval hook stamps an approved
call, so the tool's stage 1 lets it through. Each test pins the exact JSON
the agent reads, so no envelope for an unknown address can name an internal
error, a control-system denial or a connection failure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import yaml

from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey.mcp_server.control_system.server_context import initialize_server_context
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.errors import ChannelNotFoundError
from osprey_connectors.write_target import KIND_BANNERS
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from tests._approval_stamp_fixture import ApprovalStamps
from tests.facility.served_tree import in_process_config, served_tree
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
    hook_error_class_map,
)

UNKNOWN = "NO:SUCH:CHANNEL"

ACTIVE_TARGET = {"name": "va", "label": KIND_BANNERS["simulator"]}

SUGGESTIONS = [
    "Nothing was read from or written to this address.",
    "Look the channel up among the facility's channels (the channel finder, where this "
    "deployment has one) instead of guessing an address.",
]


@pytest.fixture
async def served(tmp_path, monkeypatch):
    """The controls tools over a real in-process simulator serving ``A:SP`` and ``A:RB``.

    The login's switch is on for the simulator and the connector is stamped with it
    (:func:`arm_switch`); with a write tier that allows, the reference monitor lets the
    write through to the simulator, which answers for the unknown address.
    """
    monkeypatch.chdir(tmp_path)
    arm_switch(monkeypatch, tmp_path / "armed-data")
    view = served_tree(tmp_path / "served", ["A:SP"], readings=["A:RB"])
    control_system = {
        "type": "virtual_accelerator",
        "writes": "allowed",
        "connector": {"virtual_accelerator": in_process_config(view, response_delay_ms=0)},
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump({"control_system": control_system}))
    initialize_server_context()
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)

    connector = VAInProcessConnector()
    await connector.connect(in_process_config(view, response_delay_ms=0))
    with (
        patch(
            "osprey_connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ),
        patch(
            "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
    ):
        yield
    await connector.disconnect()


async def _read(channels: list[str]):
    from osprey.mcp_server.control_system.tools.channel_read import channel_read

    return await get_tool_fn(channel_read)(channels=channels)


async def _write(operations: list[dict]):
    """One approved ``channel_write`` call: both approval stamps filed, the call bound to them."""
    from osprey.mcp_server.control_system import target_state
    from osprey.mcp_server.control_system.tools.channel_write import (
        _read_target_binding,
        channel_write,
    )

    state = target_state.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    binding = _read_target_binding()
    target, generation = binding if binding is not None else (None, None)
    with ApprovalStamps(state).approved(
        "channel_write", operations=operations, target=target, generation=generation
    ):
        return await get_tool_fn(channel_write)(operations=operations)


@pytest.mark.usefixtures("served")
async def test_a_read_of_an_unknown_address_is_not_found():
    with assert_raises_error() as raised:
        await _read([UNKNOWN])

    assert raised["envelope"] == {
        "error": True,
        "error_type": "not_found",
        "error_message": (
            f"Unknown channel during channel_read (active target: {ACTIVE_TARGET['label']}): "
            f"{UNKNOWN} is not a channel of this facility"
        ),
        "suggestions": SUGGESTIONS,
        "details": {
            "channel": UNKNOWN,
            "active_target": ACTIVE_TARGET,
            "missing": {
                UNKNOWN: f"ChannelNotFoundError: {UNKNOWN} is not a channel of this facility"
            },
        },
    }


@pytest.mark.usefixtures("served")
async def test_a_mixed_read_lists_the_unknown_address_with_its_type():
    data = extract_response_dict(await _read(["A:RB", UNKNOWN]))

    assert data["summary"]["missing"] == {
        UNKNOWN: f"ChannelNotFoundError: {UNKNOWN} is not a channel of this facility"
    }
    assert "A:RB" in data["summary"]["readings"]


@pytest.mark.usefixtures("served")
async def test_a_write_to_an_unknown_address_is_not_found():
    with assert_raises_error() as raised:
        await _write([{"channel": UNKNOWN, "value": 1.5}])

    envelope = raised["envelope"]
    (row,) = envelope["details"].pop("results")
    assert (row["channel"], row["outcome"], row["refusal_reason"]) == (
        UNKNOWN,
        "refused",
        "CHANNEL_NOT_FOUND",
    )
    assert envelope == {
        "error": True,
        "error_type": "not_found",
        "error_message": (
            f"Unknown channel during channel_write (active target: {ACTIVE_TARGET['label']}): "
            f"{UNKNOWN} is not a channel of this facility"
        ),
        "suggestions": SUGGESTIONS,
        "details": {"channel": UNKNOWN, "active_target": ACTIVE_TARGET},
    }
    assert not any("access security" in line for line in envelope["suggestions"])


@pytest.mark.usefixtures("served")
async def test_a_mixed_write_reports_the_unknown_row_and_keeps_the_written_one():
    data = extract_response_dict(
        await _write([{"channel": "A:SP", "value": 1.5}, {"channel": UNKNOWN, "value": 1.5}])
    )

    rows = {row["channel"]: row for row in data["summary"]["results"]}
    assert rows["A:SP"]["outcome"] == "confirmed"
    unknown = rows[UNKNOWN]
    assert unknown["outcome"] == "refused"
    assert unknown["refusal_reason"] == "CHANNEL_NOT_FOUND"
    assert unknown["error"] == f"{UNKNOWN} is not a channel of this facility"


async def test_a_history_read_of_an_unknown_address_is_not_found():
    with assert_raises_error() as raised:
        async with connector_error_handler("archiver_read", "archiver"):
            raise ChannelNotFoundError("X")

    assert raised["envelope"] == {
        "error": True,
        "error_type": "not_found",
        "error_message": "Unknown channel during archiver_read: X is not a channel of this facility",
        "suggestions": SUGGESTIONS,
        "details": {"channel": "X"},
    }


def test_the_hook_classes_the_answer_as_data():
    assert hook_error_class_map()["not_found"] == "Data"
