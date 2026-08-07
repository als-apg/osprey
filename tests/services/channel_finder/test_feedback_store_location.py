"""Feedback stores live under ``_agent_data/``, never in the build-owned ``data/``.

The hierarchical feedback store and the pending-review store are written while
the agent runs. A project's ``data/`` tree is re-rendered from the profile on
every build and checksummed into the manifest, so runtime writes there read as
project drift and are erased by ``osprey build --force`` — taking the operator's
accumulated feedback with them. These tests pin the shipped defaults: the config
templates, the app fallback, and the capture hook must all agree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from osprey.interfaces.channel_finder.app import FEEDBACK_DIR

SRC = Path(__file__).resolve().parents[3] / "src" / "osprey"

CONFIG_TEMPLATES = (
    SRC / "templates/apps/control_assistant/config.yml.j2",
    SRC / "templates/apps/channel_finder_standalone/config.yml.j2",
)
CAPTURE_HOOK = SRC / "templates/claude_code/claude/hooks/osprey_cf_feedback_capture.py"


def test_app_default_is_under_the_agent_data_root():
    assert FEEDBACK_DIR.startswith("_agent_data/")


@pytest.mark.parametrize("template", CONFIG_TEMPLATES, ids=lambda p: p.parent.name)
def test_config_template_default_is_under_the_agent_data_root(template):
    store_paths = re.findall(r"^\s*store_path:\s*(\S+)", template.read_text(), re.MULTILINE)

    assert store_paths, f"no feedback store_path found in {template}"
    for path in store_paths:
        assert path.startswith("_agent_data/"), f"{template} points a runtime writer at {path}"


@pytest.mark.parametrize("template", CONFIG_TEMPLATES, ids=lambda p: p.parent.name)
def test_config_template_yaml_still_parses(template):
    """The relocation is a value change, not a structural one."""
    rendered = re.sub(r"{%.*?%}", "", template.read_text(), flags=re.DOTALL)
    rendered = re.sub(r"{{.*?}}", "placeholder", rendered)

    assert yaml.safe_load(rendered) is not None


def test_capture_hook_writes_under_the_agent_data_root():
    source = CAPTURE_HOOK.read_text()

    assert '"_agent_data", "feedback"' in source
    assert '"data", "feedback"' not in source
