"""Tests for the CLI tool-inventory recorder.

None of these starts a real CLI or touches the network: each feeds literal
stream-json lines or name lists to the functions the probe and the check are built
from, or points the probe at a stand-in script.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# scripts/ is not a package, so the recorder is loaded by path and not registered
# in sys.modules.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "cli_tool_inventory.py"
_spec = importlib.util.spec_from_file_location("cli_tool_inventory", _MODULE_PATH)
assert _spec and _spec.loader
inventory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inventory)


def _init(tools: list[str]) -> str:
    return json.dumps({"type": "system", "subtype": "init", "tools": tools})


def test_init_tools_reads_the_first_init_message():
    lines = [
        "not json at all",
        json.dumps({"type": "system", "subtype": "hook_started"}),
        _init(["Write", "Bash", "Read"]),
        _init(["Other"]),
    ]
    assert inventory.init_tools(lines) == ["Bash", "Read", "Write"]


def test_init_tools_drops_mcp_names():
    lines = [_init(["Bash", "mcp__controls__channel_read", "Read"])]
    assert inventory.init_tools(lines) == ["Bash", "Read"]


def test_init_tools_without_an_init_message_raises():
    with pytest.raises(ValueError):
        inventory.init_tools(["noise", json.dumps({"type": "assistant"})])


def test_stale_names_lists_recorded_names_the_build_lacks():
    assert inventory.stale_names({"Bash", "MultiEdit"}, {"Bash", "Read"}) == ["MultiEdit"]


def test_render_inventory_round_trips():
    lists = {"tools": ["Bash", "Read"], "remote_config_tools": ["Bash", "Monitor", "Read"]}
    text = inventory.render_inventory("9.9.9", lists)
    assert json.loads(text) == {"cli_version": "9.9.9", **lists}
    assert text.endswith("}\n") and not text.endswith("\n\n")


def test_the_second_probe_reaches_remote_configuration():
    assert set(inventory.PROBES) == {"tools", "remote_config_tools"}
    assert inventory.PROBES["tools"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    remote = inventory.PROBES["remote_config_tools"]
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in remote
    assert remote["ANTHROPIC_BASE_URL"] == inventory.PROBE_ENV["ANTHROPIC_BASE_URL"]


def test_probe_of_a_silent_build_reports_a_timeout(tmp_path, monkeypatch):
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexec sleep 30\n")
    cli.chmod(0o755)
    monkeypatch.setattr(inventory, "INIT_TIMEOUT_S", 0.5)
    with pytest.raises(inventory.InventoryError, match="no system/init message within 0.5 s"):
        inventory.probe(cli)
