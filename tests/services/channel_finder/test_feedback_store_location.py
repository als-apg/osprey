"""Feedback stores live under the agent-data root, never in build-owned ``data/``.

The hierarchical feedback store and the pending-review store are written while
the agent runs. A project's ``data/`` tree is re-rendered from the profile on
every build and checksummed into the manifest, so runtime writes there read as
project drift and are erased by ``osprey build`` — taking the operator's
accumulated feedback with them. These tests pin the shipped defaults: the config
templates, the app fallback, and the capture hook must all agree.

Every assertion is written against
:data:`~osprey.utils.workspace.DEFAULT_AGENT_DATA_BASE_DIR` rather than a
literal path, because that constant is what the three producers resolve. A test
naming the directory itself would keep passing against a stale spelling while
the writers moved on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from osprey.interfaces.channel_finder.app import FEEDBACK_DIR
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

SRC = Path(__file__).resolve().parents[3] / "src" / "osprey"

CONFIG_TEMPLATES = (
    SRC / "templates/apps/control_assistant/config.yml.j2",
    SRC / "templates/apps/channel_finder_standalone/config.yml.j2",
)
CAPTURE_HOOK = SRC / "templates/claude_code/claude/hooks/osprey_cf_feedback_capture.py"


def test_app_default_is_under_the_agent_data_root():
    assert FEEDBACK_DIR.startswith(f"{DEFAULT_AGENT_DATA_BASE_DIR}/")


@pytest.mark.parametrize("template", CONFIG_TEMPLATES, ids=lambda p: p.parent.name)
def test_config_template_default_is_under_the_agent_data_root(template):
    store_paths = re.findall(r"^\s*store_path:\s*(\S+)", template.read_text(), re.MULTILINE)

    assert store_paths, f"no feedback store_path found in {template}"
    for path in store_paths:
        assert path.startswith(f"{DEFAULT_AGENT_DATA_BASE_DIR}/"), (
            f"{template} points a runtime writer at {path}"
        )


@pytest.mark.parametrize("template", CONFIG_TEMPLATES, ids=lambda p: p.parent.name)
def test_config_template_yaml_still_parses(template):
    """The relocation is a value change, not a structural one."""
    rendered = re.sub(r"{%.*?%}", "", template.read_text(), flags=re.DOTALL)
    rendered = re.sub(r"{{.*?}}", "placeholder", rendered)

    assert yaml.safe_load(rendered) is not None


def test_capture_hook_writes_under_the_agent_data_root():
    """The hook composes its store path from the root it imports, not a literal.

    Asserted on the composition rather than on a rendered path because the hook
    is a standalone script copied into a project: it never imports this test's
    view of the world, and its own import of the constant is the thing that has
    to stay true. The fallback spelling is checked too — it is the branch taken
    when the hook runs with osprey off the path, and a stale one there would put
    the store somewhere nothing reads.

    The root the composition is anchored on matters as much as the tail: a hook
    runs with its working directory at the render, so anchoring on that (rather
    than on the repo root ``get_repo_root`` derives) would put the store two
    directories deep inside a zone every build deletes.
    """
    source = CAPTURE_HOOK.read_text()

    assert "from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR" in source
    assert '_AGENT_DATA_ROOT = "' + DEFAULT_AGENT_DATA_BASE_DIR + '"' in source
    assert "repo_root = get_repo_root(hook_input)" in source
    assert 'os.path.join(repo_root, _AGENT_DATA_ROOT, "feedback"' in source
    assert '"data", "feedback"' not in source
