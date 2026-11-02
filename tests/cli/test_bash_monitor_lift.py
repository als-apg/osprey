"""PR-12: on an arming tier, lifting ``Bash`` or ``Monitor`` is a build refusal.

A shell on a render whose ``control_system.writes`` is ``asked`` or ``allowed`` is
an arming path: it reaches the write switch, the connectors and the dispatcher wire around
every write gate. So ``claude_code.permissions.remove_deny`` lifting either
shell built-in fails the build there, even when the profile declares its own
PreToolUse gate for it, and the refusal names the tier's word. On a ``never``
render the same lift keeps today's declared-gate warning.

Covered on the root render (a regen over a project's own config) and on a tier
render (a real ``osprey build`` of the exemplar, whose ``readonly`` tier is
``never`` and whose ``readwrite`` tier is spelled each arming word in turn).
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import chdir
from pathlib import Path

import pytest
import yaml

from osprey.agent_runner.tool_names import SHELL_TOOLS
from osprey.cli.templates.manager import TemplateManager
from osprey_connectors.errors import BuildProfileError
from tests._preset_data import bundle_data_root
from tests.cli.test_build_cmd import _create_project, _ship_declared_pre_hook
from tests.fixtures.lifecycle_repo import (
    EXEMPLAR_DIRNAME,
    build_exemplar_repo,
    preserved_environ,
)

SHELLS = list(SHELL_TOOLS)
#: Both words arm a shell: `asked` governs the tools that prompt, not the shell.
ARMING = ("asked", "allowed")


def test_the_shell_set_is_bash_and_monitor_declared_once() -> None:
    """One shell set, read by both the roster guard and the PR-12 refusal.

    A render that lifts ``Monitor`` alone has a shell (it runs commands in the
    background), so both gates must count it; reading one declaration is what
    keeps them from drifting apart.
    """
    from osprey.cli.templates import claude_code
    from osprey.deployment.web_terminals import roster

    assert SHELL_TOOLS == ("Bash", "Monitor")
    assert claude_code.SHELL_TOOLS is SHELL_TOOLS
    assert roster.SHELL_TOOLS is SHELL_TOOLS
    assert not hasattr(claude_code, "ARMING_SHELL_TOOLS")
    assert not hasattr(roster, "SHELL_DENY_ENTRIES")


def _root_project(tmp_path: Path, name: str, writes: str, permissions: dict) -> tuple:
    """A hello-world project whose root config sets the tier and the permissions."""
    manager = TemplateManager()
    project = _create_project(
        manager,
        project_name=name,
        output_dir=tmp_path,
        data_bundle="hello_world",
        data_root=bundle_data_root("hello_world"),
    )
    manager.generate_manifest(
        project,
        name,
        recorded_preset="hello-world",
        artifacts=manager._effective_artifacts("hello_world", None),
        preset_name="hello-world",
    )
    config = yaml.safe_load((project / "config.yml").read_text())
    config.setdefault("control_system", {})["writes"] = writes
    config.setdefault("claude_code", {})["permissions"] = permissions
    (project / "config.yml").write_text(yaml.dump(config))
    return manager, project


# --- root render ------------------------------------------------------------


@pytest.mark.parametrize("word", ARMING)
@pytest.mark.parametrize("tool", SHELLS)
def test_root_arming_tier_refuses_a_lifted_shell_even_with_a_declared_gate(
    tmp_path: Path, tool: str, word: str
) -> None:
    manager, project = _root_project(
        tmp_path, f"pr12-root-{tool.lower()}", word, {"remove_deny": [tool]}
    )
    _ship_declared_pre_hook(project, f"facility_{tool.lower()}_gate.py", tool)

    with pytest.raises(BuildProfileError) as excinfo:
        manager.regenerate_claude_code(project)

    message = str(excinfo.value)
    assert tool in message
    assert "arming path" in message
    assert f"control_system.writes: {word}" in message
    assert f"On an {word!r} tier" in message
    assert "remove_deny" in message
    assert f"pr12-root-{tool.lower()}" in message
    # Refused before anything is written: no settings.json lifts the shell.
    settings = project / ".claude" / "settings.json"
    if settings.exists():
        assert tool in json.loads(settings.read_text())["permissions"]["deny"]


@pytest.mark.parametrize("word", ARMING)
@pytest.mark.parametrize("tool", SHELLS)
def test_root_arming_tier_refuses_a_lifted_ungated_shell_with_the_pr12_reason(
    tmp_path: Path, tool: str, word: str
) -> None:
    """With no gate at all the refusal is still PR-12's, not the generic one."""
    manager, project = _root_project(
        tmp_path, f"pr12-ungated-{tool.lower()}", word, {"remove_deny": [tool]}
    )

    with pytest.raises(BuildProfileError, match="arming path"):
        manager.regenerate_claude_code(project)


@pytest.mark.parametrize("tool", SHELLS)
def test_root_never_keeps_the_declared_gate_warning(tmp_path: Path, caplog, tool: str) -> None:
    manager, project = _root_project(
        tmp_path, f"pr12-never-{tool.lower()}", "never", {"remove_deny": [tool]}
    )
    _ship_declared_pre_hook(project, f"facility_{tool.lower()}_gate.py", tool)

    with caplog.at_level(logging.WARNING):
        manager.regenerate_claude_code(project)

    settings = json.loads((project / ".claude" / "settings.json").read_text())
    assert tool not in settings["permissions"]["deny"]
    warnings = [
        r.message for r in caplog.records if tool in r.message and "permissionDecision" in r.message
    ]
    assert warnings, [r.message for r in caplog.records]
    assert not any("arming path" in r.message for r in caplog.records)


@pytest.mark.parametrize("tool", SHELLS)
def test_root_never_ungated_lift_keeps_the_no_gate_refusal(tmp_path: Path, tool: str) -> None:
    manager, project = _root_project(
        tmp_path, f"pr12-never-ungated-{tool.lower()}", "never", {"remove_deny": [tool]}
    )

    with pytest.raises(BuildProfileError) as excinfo:
        manager.regenerate_claude_code(project)

    assert "with no gate" in str(excinfo.value)
    assert "arming path" not in str(excinfo.value)


@pytest.mark.parametrize("word", ARMING)
def test_root_arming_tier_without_a_lift_builds_with_both_shells_denied(
    tmp_path: Path, word: str
) -> None:
    manager, project = _root_project(tmp_path, "pr12-no-lift", word, {})

    manager.regenerate_claude_code(project)

    deny = json.loads((project / ".claude" / "settings.json").read_text())["permissions"]["deny"]
    for tool in SHELLS:
        assert tool in deny


@pytest.mark.parametrize("word", ARMING)
def test_root_arming_tier_lifting_a_non_shell_is_not_pr12(
    tmp_path: Path, caplog, word: str
) -> None:
    """EnterWorktree writes to disk but runs no shell: it keeps the warning."""
    manager, project = _root_project(
        tmp_path, "pr12-worktree", word, {"remove_deny": ["EnterWorktree", "WebSearch"]}
    )
    _ship_declared_pre_hook(project, "facility_worktree_gate.py", "EnterWorktree")

    with caplog.at_level(logging.WARNING):
        manager.regenerate_claude_code(project)

    deny = json.loads((project / ".claude" / "settings.json").read_text())["permissions"]["deny"]
    assert "EnterWorktree" not in deny
    assert "WebSearch" not in deny
    assert "Bash" in deny and "Monitor" in deny


# --- tier render ------------------------------------------------------------


def _lift_in_tier(repo: Path, tier: str, tool: str) -> None:
    """Add ``claude_code.permissions.remove_deny: [tool]`` to a tier's config block."""
    path = repo / "tiers" / f"{tier}.yml"
    text = path.read_text(encoding="utf-8")
    assert text.count("\nconfig:\n") == 1, path
    path.write_text(
        text.replace(
            "\nconfig:\n",
            f"\nconfig:\n  claude_code.permissions.remove_deny:\n    - {tool}\n",
        ),
        encoding="utf-8",
    )


def _spell_tier(repo: Path, tier: str, word: str) -> None:
    """State *word* as a tier file's ``control_system.writes``."""
    path = repo / "tiers" / f"{tier}.yml"
    text, count = re.subn(
        r"^  control_system\.writes: \w+$",
        f"  control_system.writes: {word}",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert count == 1, path
    path.write_text(text, encoding="utf-8")


#: The exemplar profile's dispatch tier line (the only two-space ``tier:``).
_DISPATCH_TIER_LINE = "\n  tier: readonly\n"


def _build_with_lift(
    base: Path, tier: str, tool: str, *, untiered_dispatch: bool = False, word: str | None = None
):
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command

    base.mkdir(parents=True, exist_ok=True)
    with preserved_environ():
        os.environ["OSPREY_AUDIT_IDENTITY"] = "builder"
        os.environ.pop("OSPREY_TERMINAL_USER", None)
        os.environ.pop("OSPREY_CONTROL_CONTEXT_TREE", None)
        repo = build_exemplar_repo(base / EXEMPLAR_DIRNAME, seed_env=True)
        _lift_in_tier(repo, tier, tool)
        if word is not None:
            _spell_tier(repo, tier, word)
        if untiered_dispatch:
            # A dispatch tier may lift no base deny at all; take the dispatch
            # worker off the tier so the lint under test is what decides.
            profile = repo / "profile.yml"
            text = profile.read_text(encoding="utf-8")
            assert text.count(_DISPATCH_TIER_LINE) == 1
            profile.write_text(text.replace(_DISPATCH_TIER_LINE, "\n"), encoding="utf-8")
        with chdir(repo):
            result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    return result


def _failure_text(result, caplog) -> str:
    """The CLI prints a summary; the build error itself goes to the log."""
    return "\n".join([result.output, *(r.getMessage() for r in caplog.records)])


@pytest.mark.slow
@pytest.mark.parametrize("word", ARMING)
@pytest.mark.parametrize("tool", SHELLS)
def test_arming_tier_render_refuses_a_lifted_shell(
    tmp_path: Path, caplog, tool: str, word: str
) -> None:
    """The readwrite tier arms writes: lifting a shell there fails the build, and
    the refusal names that tier's render and its word."""
    with caplog.at_level(logging.WARNING):
        result = _build_with_lift(tmp_path, "readwrite", tool, word=word)

    assert result.exit_code != 0, result.output
    text = _failure_text(result, caplog)
    assert "arming path" in text, text
    assert "'als-exemplar-readwrite'" in text, text
    assert f"lifts {tool!r}" in text, text
    assert f"control_system.writes: {word}" in text, text


@pytest.mark.slow
def test_never_tier_render_lift_is_not_pr12(tmp_path: Path, caplog) -> None:
    """The readonly tier is ``never``: an ungated lift there is the ordinary
    no-gate refusal, never PR-12's."""
    with caplog.at_level(logging.WARNING):
        result = _build_with_lift(tmp_path, "readonly", "Bash", untiered_dispatch=True)

    text = _failure_text(result, caplog)
    assert "arming path" not in text, text
    assert result.exit_code != 0, result.output
    assert "with no gate" in text, text
