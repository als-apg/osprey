"""The maintainer skills quote the workflows and the validator as they are.

The release skill's manual fallback stands in for ``release.yml`` when a
publish has to be finished by hand, and the panel skill's head is copied
verbatim into every new panel. Both are followed literally, so each quoted
build line, test invocation and HTML head is held to the thing it quotes.
"""

from __future__ import annotations

import json
import re

from osprey.interfaces.design_system.panels.validator import PanelRule, assert_valid_panel
from tests.docs.test_environment_variable_page import _REPO_ROOT

_SKILLS = _REPO_ROOT / "plugins" / "osprey" / "skills"
_RELEASE_SKILL = (_SKILLS / "release" / "SKILL.md").read_text(encoding="utf-8")
_PANEL_SKILL = (_SKILLS / "panel" / "SKILL.md").read_text(encoding="utf-8")
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"

_BUILD_LINE = re.compile(r"^\s*(uv build(?: --package [\w-]+)?)\b", re.M)


def _fenced(text: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", text, re.S)


def test_the_manual_fallback_builds_what_the_release_workflow_builds() -> None:
    workflow = (_WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    built = set(_BUILD_LINE.findall(workflow))
    assert built, "release.yml no longer spells its `uv build` lines one per line"
    fallback = next(block for block in _fenced(_RELEASE_SKILL, "bash") if "vYYYY.M.P &&" in block)
    missing = sorted(built - set(_BUILD_LINE.findall(fallback)))
    assert not missing, f"the release skill's framework fallback never runs: {missing}"


def test_the_quoted_e2e_invocation_is_the_one_ci_runs() -> None:
    ci = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    quoted = re.search(r"It runs `(pytest tests/e2e/ [^`]+)`", _RELEASE_SKILL)
    assert quoted, "the release skill no longer quotes the e2e-tests invocation"
    assert f"uv run {quoted.group(1)} \\" in ci, f"ci.yml does not run `{quoted.group(1)}`"
    counted = re.search(r"\b[\w-]+ `--ignore` flags", _RELEASE_SKILL)
    assert not counted, (
        f"{counted.group(0)!r}: the ignore list grows with every dedicated lane, "
        "so a count in prose goes stale — name the list, not its length"
    )


def test_the_panel_head_passes_the_validator_it_is_checked_with(tmp_path) -> None:
    (tmp_path / "index.html").write_text(_fenced(_PANEL_SKILL, "html")[0], encoding="utf-8")
    manifest = next(block for block in _fenced(_PANEL_SKILL, "json") if '"entry"' in block)
    (tmp_path / "manifest.json").write_text(manifest, encoding="utf-8")
    assert json.loads(manifest)["entry"] == "index.html"
    assert_valid_panel(tmp_path)


def test_the_panel_rule_table_lists_every_rule() -> None:
    listed = set(re.findall(r"^\| `([A-Z_]+)` \|", _PANEL_SKILL, re.M))
    assert listed == {rule.name for rule in PanelRule}
