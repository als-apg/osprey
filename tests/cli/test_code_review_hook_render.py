"""The rendered ``code-review`` hook rule: its deadline, and that it gates nothing.

The rule's deadline comes from config, in both its ``timeout`` and its ``--deadline``
argument. The hook allows the edit whenever review is off, so the write-tool lint must
not take its ``NotebookEdit`` matcher as the tool's gate.

The real-build half goes through ``osprey build``: a rule rendered with no hook file
behind it would pass every unit test here and still be dead in a project.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.agent_runner.tool_names import DENY_DEFAULTS
from osprey.cli.init_cmd import init
from osprey.cli.templates.claude_code import (
    _build_framework_hook_rules,
    _lint_write_tools_are_gated,
    config_derived_context,
)
from osprey.services.code_review.settings import DEFAULT_PROMPT
from osprey_connectors.errors import BuildProfileError
from tests.cli._lifecycle_build import invoke_build

HOOK_FILE = "osprey_code_review.py"
SCRIPT = f'"$CLAUDE_PROJECT_DIR/.claude/hooks/{HOOK_FILE}"'


def _only_hook(rules: list[dict]) -> dict:
    assert len(rules) == 1
    hooks = rules[0]["hooks"]
    assert len(hooks) == 1
    return hooks[0]


# ---------------------------------------------------------------------------
# config_derived_context: the deadline a config asks for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "deadline"),
    [
        pytest.param({}, 90, id="no-section"),
        pytest.param({"code_review": {"enabled": True}}, 90, id="enabled-default-limit"),
        pytest.param({"code_review": {"enabled": True, "time_limit": 120}}, 150, id="limit-120"),
        pytest.param({"code_review": {"enabled": True, "time_limit": 0.5}}, 31, id="rounds-up"),
        # A disabled section is not validated, and its limit is not in force.
        pytest.param({"code_review": {"enabled": False, "time_limit": 120}}, 90, id="disabled"),
    ],
)
def test_the_deadline_follows_the_config(tmp_path: Path, config: dict, deadline: int) -> None:
    assert config_derived_context(config, tmp_path)["code_review_deadline_s"] == deadline


@pytest.mark.parametrize(
    "section",
    [
        pytest.param("yes", id="not-a-mapping"),
        pytest.param({"enabled": "yes"}, id="enabled-not-a-bool"),
        pytest.param({"enabled": True, "time_limit": "soon"}, id="bad-limit"),
    ],
)
def test_a_malformed_section_renders_the_default_deadline(tmp_path: Path, section: Any) -> None:
    """Refusing a malformed section is the build check's job; a render still completes."""
    context = config_derived_context({"code_review": section}, tmp_path)

    assert context["code_review_deadline_s"] == 90


# ---------------------------------------------------------------------------
# _build_framework_hook_rules: the rule
# ---------------------------------------------------------------------------


def test_the_default_deadline_renders_90_when_none_is_passed() -> None:
    pre, post = _build_framework_hook_rules(["code-review"])

    assert post == []
    assert pre[0]["matcher"] == "NotebookEdit"
    hook = _only_hook(pre)
    assert hook["timeout"] == 90
    assert hook["command"] == f"python3 {SCRIPT} --deadline 90"


def test_a_deadline_renders_in_both_the_timeout_and_the_argument() -> None:
    pre, _ = _build_framework_hook_rules(["code-review"], code_review_deadline_s=150)

    hook = _only_hook(pre)
    assert hook["timeout"] == 150
    assert hook["command"] == f"python3 {SCRIPT} --deadline 150"


def test_every_other_hook_renders_unchanged() -> None:
    """Rules for the other hooks are identical whatever the review's deadline is."""
    others = ["memory-guard", "writes-check", "approval", "limits", "error-guidance"]

    expected = _build_framework_hook_rules(others)
    pre, post = _build_framework_hook_rules([*others, "code-review"], code_review_deadline_s=150)

    assert [r for r in pre if HOOK_FILE not in r["hooks"][0]["command"]] == expected[0]
    assert post == expected[1]
    for rule in [*expected[0], *expected[1]]:
        assert "--deadline" not in rule["hooks"][0]["command"]


# ---------------------------------------------------------------------------
# _lint_write_tools_are_gated: the rule is not a gate
# ---------------------------------------------------------------------------


def _lint_ctx() -> dict[str, Any]:
    """A render that denies ``Write``, so the lint reaches ``NotebookEdit``.

    ``Write`` is checked first and is not in the deny floor, so without this every
    profile lacking memory-guard is refused on ``Write`` whether or not the review
    rule counts as a gate.
    """
    return {"deny_defaults": [*DENY_DEFAULTS, "Write"], "facility_permissions": {}}


def test_the_review_rule_does_not_gate_notebook_edit() -> None:
    pre, _ = _build_framework_hook_rules(["code-review"])

    with pytest.raises(BuildProfileError, match="'NotebookEdit' with no gate"):
        _lint_write_tools_are_gated(_lint_ctx(), pre)


def test_a_profile_declared_review_rule_does_not_gate_notebook_edit() -> None:
    """Wiring the same script by hand does not turn it into a gate."""
    ctx = _lint_ctx()
    ctx["declared_hooks"] = {
        "PreToolUse": [
            {
                "matcher": "NotebookEdit",
                "hooks": [{"type": "command", "command": f"python3 {SCRIPT}", "timeout": 90}],
            }
        ]
    }

    with pytest.raises(BuildProfileError, match="'NotebookEdit' with no gate"):
        _lint_write_tools_are_gated(ctx, [])


def test_memory_guard_still_gates_notebook_edit_beside_the_review_rule() -> None:
    pre, _ = _build_framework_hook_rules(["memory-guard", "code-review"])

    _lint_write_tools_are_gated(_lint_ctx(), pre)


def test_a_review_rule_that_shares_an_entry_with_another_hook_keeps_that_gate() -> None:
    """Only a rule that runs nothing but the review is discounted."""
    ctx = _lint_ctx()
    ctx["declared_hooks"] = {
        "PreToolUse": [
            {
                "matcher": "NotebookEdit",
                "hooks": [
                    {"type": "command", "command": f"python3 {SCRIPT}"},
                    {"type": "command", "command": "/opt/facility/notebook-gate"},
                ],
            }
        ]
    }

    _lint_write_tools_are_gated(ctx, [])


def test_notebook_edit_may_still_be_denied_outright() -> None:
    pre, _ = _build_framework_hook_rules(["code-review"])
    ctx = _lint_ctx()
    ctx["deny_defaults"] = [*ctx["deny_defaults"], "NotebookEdit"]

    _lint_write_tools_are_gated(ctx, pre)


# ---------------------------------------------------------------------------
# Real builds
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scaffold(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One hello-world scaffold; each test builds its own copy of it."""
    repo = tmp_path_factory.mktemp("code-review-render") / "scaffold"
    result = CliRunner().invoke(init, [str(repo), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output
    return repo


@pytest.fixture
def project(scaffold: Path, tmp_path: Path) -> Callable[..., Path]:
    """Copy the scaffold and select the hook.

    Returns:
        A callable taking the review section (``None`` for no section); it returns the
        repo unbuilt.
    """

    def _project(section: dict | None = None) -> Path:
        repo = tmp_path / "repo"
        shutil.copytree(scaffold, repo)
        profile_path = repo / "profile.yml"
        profile = yaml.safe_load(profile_path.read_text("utf-8"))
        profile["hooks"] = [*profile["hooks"], "code-review"]
        if section is not None:
            profile["config"]["code_review"] = section
        profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), "utf-8")
        return repo

    return _project


def _review_hook(repo: Path) -> dict:
    """The one rendered ``NotebookEdit`` hook that runs the review."""
    settings = json.loads((repo / "build" / ".claude" / "settings.json").read_text("utf-8"))
    found = [
        hook
        for rule in settings["hooks"]["PreToolUse"]
        if rule["matcher"] == "NotebookEdit"
        for hook in rule["hooks"]
        if HOOK_FILE in hook["command"]
    ]
    assert len(found) == 1, found
    return found[0]


def _assert_rendered(repo: Path, deadline: int) -> None:
    assert (repo / "build" / ".claude" / "hooks" / HOOK_FILE).is_file()
    hook = _review_hook(repo)
    assert hook["timeout"] == deadline
    assert hook["command"].endswith(f"{SCRIPT} --deadline {deadline}")
    # settings.json.j2 replaced the `python3 ` token with the project's interpreter.
    assert hook["command"].startswith('"/')
    assert not hook["command"].startswith("python3 ")


def test_a_build_with_no_section_renders_the_file_and_the_default_rule(
    project: Callable[..., Path],
) -> None:
    repo = project()

    result = invoke_build(repo)

    assert result.exit_code == 0, result.output
    _assert_rendered(repo, 90)


def test_a_build_with_review_enabled_renders_the_same_rule(
    project: Callable[..., Path],
) -> None:
    repo = project({"enabled": True})
    prompt = repo / DEFAULT_PROMPT
    if not prompt.exists():
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text("Review the code.\n", encoding="utf-8")

    result = invoke_build(repo)

    assert result.exit_code == 0, result.output
    _assert_rendered(repo, 90)


def test_a_build_with_a_time_limit_renders_it_in_the_rule(
    project: Callable[..., Path],
) -> None:
    repo = project({"enabled": True, "time_limit": 120})

    result = invoke_build(repo)

    assert result.exit_code == 0, result.output
    _assert_rendered(repo, 150)
