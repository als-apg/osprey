"""The Review tab's route: the code review section as the next review reads it.

Read by every tier; the warning states what a review does while the section or
the rules file cannot be used, which is pinned against the reviewer's own
behaviour in ``osprey.services.code_review.review.decide``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.code_review import (
    RULES_UNUSABLE_BLOCK,
    RULES_UNUSABLE_FLAG,
    router,
)
from osprey.interfaces.web_terminal.routes.scaffold import router as scaffold_router
from osprey.services.code_review.contract import ReviewFailure
from osprey.services.code_review.review import ACTION_PROCEED, ACTION_REFUSE, decide

from .conftest import bare_route_app

RULES = "# Code reviewer\n\n## Rules\n\n- Read `SR:BEAM:USER_MODE` first.\n"
PROMPT = "data/code_review/reviewer.md"


def _project(tmp_path: Path, section: str, rules: str | None = RULES) -> Path:
    """A repo with a render at ``build/`` and, optionally, the reviewer's rules."""
    render = tmp_path / "build"
    render.mkdir(parents=True)
    (render / "config.yml").write_text(section, encoding="utf-8")
    if rules is not None:
        prompt = tmp_path / PROMPT
        prompt.parent.mkdir(parents=True)
        prompt.write_text(rules, encoding="utf-8")
    return render


def _get(tmp_path: Path, section: str, rules: str | None = RULES, *, admin: bool = True) -> dict:
    render = _project(tmp_path, section, rules)
    app = bare_route_app(router, config_path=render / "config.yml", config_panel_enabled=admin)
    response = TestClient(app).get("/api/code-review")
    assert response.status_code == 200
    return response.json()


ON = "code_review:\n  enabled: true\n"


def test_on_serves_the_rules_the_reviewer_reads(tmp_path):
    body = _get(tmp_path, ON + "  time_limit: 45\n")
    assert body["enabled"] is True
    assert body["level"] == "block"
    assert body["time_limit"] == 45
    assert body["prompt"] == PROMPT
    assert body["rules"] == RULES
    assert body["warning"] is None


def test_off_is_shown_with_its_values_and_no_warning(tmp_path):
    body = _get(tmp_path, "code_review:\n  enabled: false\n  time_limit: 90\n", rules=None)
    assert body["enabled"] is False
    assert body["time_limit"] == 90
    assert body["rules"] is None
    # Off, a missing file is no review's concern.
    assert body["warning"] is None


def test_absent_section_is_off(tmp_path):
    body = _get(tmp_path, "web: {}\n")
    assert body["enabled"] is False
    assert body["rules"] == RULES
    assert body["warning"] is None


def test_every_tier_reads_it_and_only_the_admin_tier_is_told_it_may_change(tmp_path):
    assert _get(tmp_path, ON, admin=False)["can_change"] is False
    assert _get(tmp_path / "admin", ON, admin=True)["can_change"] is True


@pytest.mark.parametrize(
    "prompt",
    ["data/code_review/missing.md", "../outside.md", "build/config.yml"],
    ids=["missing", "climbs-out", "build-zone"],
)
def test_unusable_rules_under_block_say_code_is_blocked(tmp_path, prompt):
    body = _get(tmp_path, ON + f"  prompt: {prompt}\n")
    assert body["rules"] is None
    head = RULES_UNUSABLE_BLOCK.split("({reason})")[0]
    assert body["warning"].startswith(head)
    assert "code is blocked" in body["warning"]
    assert prompt in body["warning"]
    # The reviewer: a prompt failure under block refuses the code.
    assert decide(ReviewFailure("prompt", "x"), "block").action == ACTION_REFUSE


def test_unusable_rules_under_flag_say_code_runs_unreviewed(tmp_path):
    body = _get(tmp_path, ON + "  level: flag\n", rules=None)
    assert body["warning"].startswith(RULES_UNUSABLE_FLAG.split("({reason})")[0])
    assert "runs unreviewed" in body["warning"]
    assert decide(ReviewFailure("prompt", "x"), "flag").action == ACTION_PROCEED


@pytest.mark.parametrize(
    ("level", "template", "consequence"),
    [
        ("block", RULES_UNUSABLE_BLOCK, "code is blocked"),
        ("flag", RULES_UNUSABLE_FLAG, "runs unreviewed"),
    ],
)
def test_empty_rules_are_treated_as_missing(tmp_path, level, template, consequence):
    body = _get(tmp_path, ON + f"  level: {level}\n", rules="  \n")
    assert body["rules"] is None
    assert body["warning"].startswith(template.split("({reason})")[0])
    assert "is empty" in body["warning"]
    assert consequence in body["warning"]


def test_unreadable_section_says_every_review_blocks(tmp_path):
    body = _get(tmp_path, "code_review:\n  enabled: 'yes'\n")
    assert body["enabled"] is None
    assert body["rules"] is None
    assert "every review fails" in body["warning"]
    assert "code is blocked" in body["warning"]
    assert "the profile's `config:` block" in body["warning"]
    # The reviewer: a config failure has no level, and refuses.
    assert decide(ReviewFailure("config", "x"), None).action == ACTION_REFUSE


def test_the_safety_tab_no_longer_lists_the_rules(tmp_path):
    render = _project(tmp_path, ON)
    app = bare_route_app(scaffold_router, project_cwd=str(render))
    names = [a["name"] for a in TestClient(app).get("/api/scaffold").json()["artifacts"]]
    assert "code-review/rules" not in names
