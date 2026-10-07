"""Feedback stores live under the agent-data root, never in build-owned ``data/``.

The hierarchical feedback store and the pending-review store are written while
the agent runs. A project's ``data/`` tree is re-rendered from the profile on
every build and checksummed into the manifest, so runtime writes there read as
project drift and are erased by ``osprey build`` — taking the operator's
accumulated feedback with them. These tests pin the shipped defaults: the config
template, the app fallback, and the capture hook must all agree.

The three producers follow ``agent_data.base_dir``. Every default is asserted
against :data:`~osprey.utils.workspace.DEFAULT_AGENT_DATA_BASE_DIR` rather than a
literal path, because that constant is what the key falls back to. A test naming
the directory itself would keep passing against a stale spelling while the
writers moved on.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from osprey.interfaces.channel_finder.app import feedback_dir
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

SRC = Path(__file__).resolve().parents[3] / "src" / "osprey"

# The framework template is the one shipped config template that writes a
# feedback store_path. The packaged app templates that used to carry their own
# copy are gone; a deployment states the rest in its profile's `config:` block.
CONFIG_TEMPLATE = SRC / "templates/project/config.yml.j2"
CAPTURE_HOOK = SRC / "templates/claude_code/claude/hooks/osprey_cf_feedback_capture.py"


def test_app_fallback_follows_the_agent_data_root():
    assert feedback_dir(None) == f"{DEFAULT_AGENT_DATA_BASE_DIR}/feedback"
    assert (
        feedback_dir({"agent_data": {"base_dir": "custom/agent_data"}})
        == "custom/agent_data/feedback"
    )


def test_config_template_default_is_under_the_agent_data_root():
    store_paths = re.findall(r"^\s*store_path:\s*(\S+)", CONFIG_TEMPLATE.read_text(), re.MULTILINE)

    assert store_paths, f"no feedback store_path found in {CONFIG_TEMPLATE}"
    for path in store_paths:
        assert path.startswith(f"{DEFAULT_AGENT_DATA_BASE_DIR}/"), (
            f"{CONFIG_TEMPLATE} points a runtime writer at {path}"
        )


def test_config_template_yaml_still_parses():
    """The relocation is a value change, not a structural one.

    Rendered for real, in the mode that emits the block: the hierarchical
    pipeline is the one that writes a feedback store.
    """
    from osprey.cli.templates.manager import _enable_flags
    from tests._config_render_context import MINIMAL_CONFIG_CONTEXT, render_config

    context = {
        **MINIMAL_CONFIG_CONTEXT,
        "channel_finder_mode": "hierarchical",
        "default_pipeline": "hierarchical",
        **_enable_flags("hierarchical"),
    }
    rendered = render_config(**context)

    assert yaml.safe_load(rendered) is not None
    assert "store_path" in rendered


def test_capture_hook_writes_under_the_agent_data_root():
    """The hook composes its store path from the configured agent-data root.

    Asserted on the composition rather than on a rendered path because the hook
    is a standalone script copied into a project: it never imports this test's
    view of the world. It resolves ``agent_data.base_dir`` through
    ``osprey_hook_log.repo_agent_data_root``, and the fallback literal that
    helper uses is pinned in ``osprey_hook_log.py`` by
    ``tests/hooks/test_osprey_hook_log.py::test_agent_data_fallback_literal_is_the_framework_default``.

    The root the composition is anchored on matters as much as the tail: a hook
    runs with its working directory at the render, so anchoring on that (rather
    than on the repo root ``get_repo_root`` derives) would put the store two
    directories deep inside a zone every build deletes.
    """
    source = CAPTURE_HOOK.read_text()

    assert "repo_agent_data_root(hook_input)" in source
    assert '"feedback", "pending_reviews.json"' in source
    assert "DEFAULT_AGENT_DATA_BASE_DIR" not in source
    assert "repo_root = get_repo_root(hook_input)" in source
    assert '"data", "feedback"' not in source
