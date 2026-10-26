"""Which write tools are the facility's: the rendered ``facility_write_tools`` set.

A facility write tool writes to ``live`` whatever the session points at. The
set is rendered verbatim from the registry's one producer: exact names for a
server that declares ``writes:``, the whole-server matcher for one with no
list. ``control_system.write_tools`` extras the render appended are facility
tools too. A declaring server's other tools are in no write list, so they are
reads. A render from before the key falls back to the whole-server matchers in
``write_tools``.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey_connectors.write_target import (
    TARGET_LIVE,
    _is_facility_tool,
    write_target_for_tool,
)

DECLARED_WRITE = "mcp__plc__set_value"
DECLARED_READ = "mcp__plc__get_value"
WHOLE_SERVER_WRITE = "mcp__magnets__set_current"
WHOLE_SERVER_READ = "mcp__magnets__read_current"
EXTRA = "mcp__orbit__apply_correction"
FRAMEWORK_WRITE = "mcp__controls__channel_write"

HOOK_CONFIG: dict[str, Any] = {
    "server_prefixes": ["mcp__controls__", "mcp__magnets__", "mcp__orbit__", "mcp__plc__"],
    "write_tools": [FRAMEWORK_WRITE, "mcp__magnets__.*", DECLARED_WRITE, EXTRA],
    "control_system_write_tools": [EXTRA],
    "facility_write_tools": [DECLARED_WRITE, "mcp__magnets__.*"],
}

PRE_KEY: dict[str, Any] = {k: v for k, v in HOOK_CONFIG.items() if k != "facility_write_tools"}

CONFIG = {"control_system": {"type": "epics", "writes": "allowed"}}


@pytest.mark.parametrize(
    ("tool", "facility"),
    [
        (DECLARED_WRITE, True),
        (DECLARED_READ, False),
        (WHOLE_SERVER_WRITE, True),
        (WHOLE_SERVER_READ, True),
        (EXTRA, True),
        (FRAMEWORK_WRITE, False),
    ],
)
def test_the_rendered_set_decides(tool, facility):
    assert _is_facility_tool(tool, HOOK_CONFIG) is facility


def test_an_empty_set_names_no_whole_server_tool():
    """The key present and empty: the render has no custom server, whatever the shape."""
    hooks = {**HOOK_CONFIG, "facility_write_tools": []}

    assert _is_facility_tool(WHOLE_SERVER_WRITE, hooks) is False
    assert _is_facility_tool(EXTRA, hooks) is True


def test_a_render_without_the_key_reads_whole_server_matchers():
    assert _is_facility_tool(WHOLE_SERVER_WRITE, PRE_KEY) is True
    assert _is_facility_tool(EXTRA, PRE_KEY) is True
    assert _is_facility_tool(FRAMEWORK_WRITE, PRE_KEY) is False


def test_non_string_entries_name_nothing():
    hooks = {**HOOK_CONFIG, "facility_write_tools": [None, 3, {"x": 1}]}

    assert _is_facility_tool(DECLARED_WRITE, hooks) is False


@pytest.mark.parametrize("tool", [DECLARED_WRITE, WHOLE_SERVER_WRITE, WHOLE_SERVER_READ, EXTRA])
def test_a_facility_write_is_placed_on_live(tool, tmp_path):
    placed = write_target_for_tool(
        CONFIG, tool, {}, hook_config=HOOK_CONFIG, record_path=tmp_path / "none.json"
    )

    assert placed == TARGET_LIVE


def test_a_declaring_servers_read_writes_nothing(tmp_path):
    placed = write_target_for_tool(
        CONFIG, DECLARED_READ, {}, hook_config=HOOK_CONFIG, record_path=tmp_path / "none.json"
    )

    assert placed is None
