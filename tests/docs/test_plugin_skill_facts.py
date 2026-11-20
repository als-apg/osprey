"""The maintainer skills quote the workflows and the validator as they are.

The release skill's manual fallback stands in for ``release.yml`` when a
publish has to be finished by hand, and the panel skill's head is copied
verbatim into every new panel. Both are followed literally, so each quoted
build line, test invocation and HTML head is held to the thing it quotes.
"""

from __future__ import annotations

import json
import re

from osprey.interfaces.design_system.panels.page_ready import DESIGN_SYSTEM_STATIC
from osprey.interfaces.design_system.panels.validator import PanelRule, assert_valid_panel
from tests.docs.test_environment_variable_page import _REPO_ROOT
from tests.interfaces.test_embedded_identity_contract import FRAMED_ONLY_RULE

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


def test_every_panel_skill_page_and_sheet_passes_the_validator(tmp_path) -> None:
    pages = _fenced(_PANEL_SKILL, "html")
    sheets = _fenced(_PANEL_SKILL, "css")
    assert pages, "the panel skill no longer fences its head as an html block"
    manifest = next(block for block in _fenced(_PANEL_SKILL, "json") if '"entry"' in block)
    assert json.loads(manifest)["entry"] == "index.html"
    for block in [*pages, *sheets]:
        assert FRAMED_ONLY_RULE not in block, "the tile bar is a panel's one header"
    for number, page in enumerate(pages):
        panel = tmp_path / f"page-{number}"
        panel.mkdir()
        (panel / "index.html").write_text(page, encoding="utf-8")
        (panel / "manifest.json").write_text(manifest, encoding="utf-8")
        assert_valid_panel(panel)
    for number, sheet in enumerate(sheets):
        panel = tmp_path / f"sheet-{number}"
        panel.mkdir()
        (panel / "index.html").write_text(pages[0], encoding="utf-8")
        (panel / "manifest.json").write_text(manifest, encoding="utf-8")
        (panel / "panel.css").write_text(sheet, encoding="utf-8")
        assert_valid_panel(panel)


def test_every_page_ready_wait_names_the_writers_two_end_values() -> None:
    writer = (DESIGN_SYSTEM_STATIC / "js" / "page-ready.js").read_text(encoding="utf-8")
    ready = re.search(r"setAttribute\(PAGE_READY_ATTRIBUTE, '([^']+)'\)", writer)
    failed = re.search(r"const FAILED = '([^']+)';", writer)
    assert ready and failed, "page-ready.js no longer spells its two end values"
    for skill in ("panel", "build-app"):
        text = (_SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
        assert f"`{ready.group(1)}` or `{failed.group(1)}`" in text, skill
    page = (
        _REPO_ROOT / "docs" / "source" / "how-to" / "apps" / "build-your-own-app.rst"
    ).read_text(encoding="utf-8")
    assert f"``{ready.group(1)}`` or ``{failed.group(1)}``" in page


def test_the_panel_rule_table_lists_every_rule() -> None:
    listed = set(re.findall(r"^\| `([A-Z_]+)` \|", _PANEL_SKILL, re.M))
    assert listed == {rule.name for rule in PanelRule}
