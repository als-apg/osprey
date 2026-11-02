"""The agent's `setup_patch` refuses the write ceiling `control_system.writes`.

The ceiling only moves through the profile or a tier file. The agent's own
config patch tool must refuse it — whatever the value, before anything on disk
is touched — exactly as it refused the retired ``writes_enabled`` leaves.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.workspace.tools import setup as setup_tools
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

SETUP_MOD = "osprey.mcp_server.workspace.tools.setup"


@pytest.fixture
def render(tmp_path):
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "control_system": {
                    "type": "virtual_accelerator",
                    "connector": {"virtual_accelerator": {"serving": "in_process"}},
                    "writes": "never",
                },
                "ui": {"theme": "light"},
            }
        )
    )
    with patch(f"{SETUP_MOD}.resolve_config_path", return_value=tmp_path / "config.yml"):
        yield tmp_path


@pytest.mark.parametrize("value", ["allowed", "never", "true"], ids=["allowed", "never", "other"])
async def test_setup_patch_refuses_the_write_ceiling(render: Path, value):
    before = (render / "config.yml").read_bytes()
    fn = get_tool_fn(setup_tools.setup_patch)

    with (
        patch(f"{SETUP_MOD}.notify_agent_activity_async"),
        assert_raises_error(error_type="protected_key") as ctx,
    ):
        await fn(file="config.yml", key_path="control_system.writes", value=value)

    message = ctx["envelope"]["error_message"]
    assert "control_system.writes" in message, message
    assert (render / "config.yml").read_bytes() == before
    assert json.dumps(ctx["envelope"]).count(f'"{value}"') == 0


def test_cold_note_names_the_new_key():
    notes = setup_tools._COLD_CHANGE_NOTES["config.yml"]
    assert "control_system.writes" in notes
    assert "control_system.writes_enabled" not in notes
