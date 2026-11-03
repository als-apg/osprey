"""The file-tool deny on the control-context store, ``<agent-data root>/control_target``.

The store holds each identity's control-context record and its writes switch, so
an agent able to edit it could arm its own writes. settings.json denies the file
tools there with two ``Edit(path)`` rules (Claude Code checks Edit, Write and
NotebookEdit against ``Edit`` rules only): the absolute ``//`` form from the root
the runtime resolves, and the relative form spelled like the render's other
agent-data rules. The section renders after the kill switch and is never
filtered, so no profile ``remove_deny`` lifts it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from osprey.cli.templates.claude_code import (
    _rendered_deny_list,
    config_derived_context,
    store_deny,
)
from osprey.cli.templates.manager import TemplateManager
from osprey_connectors.posture_store import STATE_DIR_NAME
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR, agent_data_base_dir
from tests._preset_data import bundle_data_root

#: The project root a container render names: the root the terminal runs at,
#: which exists only inside the container.
IN_CONTAINER_ROOT = "/app/project"


def _absolute_rule(root: Path | str, base_dir: str = DEFAULT_AGENT_DATA_BASE_DIR) -> str:
    return f"Edit(/{Path(root) / base_dir / STATE_DIR_NAME}/**)"


def _relative_rule(base_dir: str = DEFAULT_AGENT_DATA_BASE_DIR) -> str:
    return f"Edit({base_dir}/{STATE_DIR_NAME}/**)"


def _settings(project: Path) -> dict:
    return json.loads((project / ".claude" / "settings.json").read_text())


def _project(tmp_path: Path, name: str, fields: dict | None = None) -> Path:
    """A rendered control-assistant project, regenerated with *fields* applied."""
    from osprey.utils.config_writer import config_update_fields
    from tests.cli._template_projects import create_project_with_manifest
    from tests.cli.test_profile_to_claude_contract import _apply_preset_config

    manager = TemplateManager()
    project = create_project_with_manifest(
        manager,
        project_name=name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
        data_root=bundle_data_root("control_assistant"),
    )
    _apply_preset_config(manager, project, "control-assistant")
    if fields:
        config_update_fields(project / "config.yml", fields)
    manager.regenerate_claude_code(project)
    return project


# --- the rules themselves --------------------------------------------------------


def test_a_render_below_the_root_names_the_store_beside_it(tmp_path):
    """The agent runs in ``build/``; the store sits under the repo root above it.

    A config naming no ``project_root`` anchors on the repo the render sits in,
    so the absolute rule names ``<repo>/<base>/control_target`` and not a
    directory under ``build/`` that the relative form alone would point at.
    """
    repo = tmp_path / "repo"
    render = repo / "build"
    render.mkdir(parents=True)

    rules = store_deny({}, render)

    assert rules == [_absolute_rule(repo.resolve()), _relative_rule()], rules
    assert str(render.resolve() / DEFAULT_AGENT_DATA_BASE_DIR) not in rules[0], (
        "the absolute rule must name the store beside the render, not inside it"
    )


def test_an_in_container_terminal_render_names_the_container_root(tmp_path):
    """A terminal render's config names the root the container runs at.

    The rule is evaluated inside the container, so it must name that root even
    though the render is written on the host, where the root does not exist.
    """
    config = {"project_root": IN_CONTAINER_ROOT}

    rules = store_deny(config, tmp_path)

    assert rules[0] == _absolute_rule(IN_CONTAINER_ROOT), rules
    assert str(tmp_path) not in rules[0], "a host path must not leak into a container rule"
    assert rules[1] == _relative_rule()


def test_a_relocated_agent_data_root_moves_both_rules(tmp_path):
    """``agent_data.base_dir`` is the one key naming the root; both forms follow it."""
    config = {"project_root": IN_CONTAINER_ROOT, "agent_data": {"base_dir": "state/agent"}}

    assert store_deny(config, tmp_path) == [
        _absolute_rule(IN_CONTAINER_ROOT, "state/agent"),
        _relative_rule("state/agent"),
    ]


def test_an_unspellable_root_is_refused(tmp_path):
    """A wildcard in the root would widen the deny past the store; refuse the render."""
    with pytest.raises(ValueError, match="cannot spell"):
        store_deny({"project_root": "/srv/odd*root"}, tmp_path)


def test_both_render_paths_carry_the_store_deny(tmp_path):
    """The rules come from config_derived_context, which both render paths read."""
    config = {"project_root": IN_CONTAINER_ROOT}

    ctx = config_derived_context(config, tmp_path)

    assert ctx["store_deny"] == store_deny(config, tmp_path)


# --- the deny list -----------------------------------------------------------------


def test_remove_deny_cannot_lift_the_store_deny(tmp_path):
    """The lint's mirror of the deny array keeps the store rules whatever remove_deny says."""
    rules = store_deny({}, tmp_path)
    ctx = {
        "deny_defaults": ["Bash", "Edit"],
        "facility_permissions": {"deny": list(rules), "remove_deny": list(rules)},
        "killswitch_deny": ["mcp__controls__channel_write"],
        "store_deny": rules,
    }

    deny = _rendered_deny_list(ctx)

    assert deny == ["Bash", "Edit", "mcp__controls__channel_write", *rules], deny


def test_a_store_rule_the_profile_already_denies_renders_once(tmp_path):
    rules = store_deny({}, tmp_path)
    ctx = {
        "deny_defaults": ["Edit"],
        "facility_permissions": {"deny": [rules[1]]},
        "store_deny": rules,
    }

    assert _rendered_deny_list(ctx) == ["Edit", rules[1], rules[0]]


# --- the rendered settings.json -----------------------------------------------------


def test_a_single_user_host_render_denies_the_store_last(tmp_path):
    """A host render (no web terminals) anchors on its own repo and renders both rules.

    They come after the kill switch, last in the array, and the template's array
    is exactly the one the build lint reasons about.
    """
    project = _project(tmp_path, "store-deny-host", {"modules.web_terminals.enabled": False})
    config = yaml.safe_load((project / "config.yml").read_text())
    rules = store_deny(config, project)
    deny = _settings(project)["permissions"]["deny"]

    assert deny[-2:] == rules, deny
    assert rules[1] == _relative_rule(agent_data_base_dir(config))
    assert rules[0].startswith("Edit(//"), rules


def test_a_profile_remove_deny_does_not_lift_the_rendered_store_deny(tmp_path):
    """The lift attempt end to end: remove_deny naming both rules leaves both rendered."""
    from osprey_connectors.workspace import resolve_shared_data_root

    probe = _project(tmp_path / "probe", "store-deny-probe")
    config = yaml.safe_load((probe / "config.yml").read_text())
    rules = store_deny(config, probe)
    assert rules[0] == f"Edit(/{resolve_shared_data_root(config) / STATE_DIR_NAME}/**)"

    project = _project(
        tmp_path / "lift",
        "store-deny-probe",
        {"claude_code.permissions.remove_deny": rules},
    )
    deny = _settings(project)["permissions"]["deny"]
    lifted = store_deny(yaml.safe_load((project / "config.yml").read_text()), project)

    for rule in lifted:
        assert deny.count(rule) == 1, f"remove_deny lifted or duplicated {rule}: {deny}"


def test_only_edit_path_rules_are_rendered_for_the_store(tmp_path):
    """Claude Code checks Write and NotebookEdit against ``Edit(path)`` rules.

    A ``Write(path)`` or ``NotebookEdit(path)`` rule grants or denies nothing and
    is reported as a misconfiguration at every agent start.
    """
    project = _project(tmp_path, "store-deny-tools")
    permissions = _settings(project)["permissions"]

    for section in ("allow", "deny", "ask"):
        stray = [e for e in permissions[section] if e.startswith(("Write(", "NotebookEdit("))]
        assert not stray, f"{section} carries non-Edit path rules: {stray}"
    assert sum(STATE_DIR_NAME in e for e in permissions["deny"]) == 2
