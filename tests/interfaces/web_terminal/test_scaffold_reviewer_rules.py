"""The code reviewer's rules in the Safety tab: listed and read, never written.

The rules are deployment data, not a build artifact, so the gallery serves them
through the same list/get routes read-only, and only while code review is on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.scaffold import router
from osprey.interfaces.web_terminal.scaffold_gallery_service import (
    REVIEWER_RULES_NAME,
    ScaffoldGalleryService,
)

from .conftest import bare_route_app

RULES = "# Code reviewer\n\n## Rules\n\n- Read `SR:BEAM:USER_MODE` first.\n"


def _project(tmp_path: Path, section: str, rules: str | None = RULES) -> Path:
    """A repo with a render at ``build/`` and, optionally, the reviewer's rules."""
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text(section, encoding="utf-8")
    if rules is not None:
        prompt = tmp_path / "data" / "code_review" / "reviewer.md"
        prompt.parent.mkdir(parents=True)
        prompt.write_text(rules, encoding="utf-8")
    return render


ON = "code_review:\n  enabled: true\n"


def _rules_entries(render: Path) -> list[dict]:
    artifacts = ScaffoldGalleryService(render).list_artifacts()
    return [a for a in artifacts if a["name"] == REVIEWER_RULES_NAME]


def test_listed_read_only_when_code_review_is_on(tmp_path):
    (entry,) = _rules_entries(_project(tmp_path, ON))
    assert entry["category"] == "code-review"
    assert entry["output_path"] == "data/code_review/reviewer.md"
    assert entry["language"] == "markdown"
    assert entry["read_only"] is True
    assert entry["kind"] == "data"
    # No framework copy exists, so the panel must not offer a diff against one.
    assert entry["custom"] is True


def test_content_is_the_file_the_reviewer_reads(tmp_path):
    content = ScaffoldGalleryService(_project(tmp_path, ON)).get_content(REVIEWER_RULES_NAME)
    assert content == {"content": RULES, "source": "user-owned", "language": "markdown"}


def test_configured_prompt_is_the_one_listed(tmp_path):
    render = _project(tmp_path, ON.replace("true", "true\n  prompt: data/rules.md"), rules=None)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "rules.md").write_text("mine\n", encoding="utf-8")
    (entry,) = _rules_entries(render)
    assert entry["output_path"] == "data/rules.md"
    assert ScaffoldGalleryService(render).get_content(REVIEWER_RULES_NAME)["content"] == "mine\n"


@pytest.mark.parametrize(
    "section",
    [
        "",
        "code_review:\n  enabled: false\n",
        "web: {}\n",
    ],
    ids=["no-config", "off", "no-section"],
)
def test_absent_while_code_review_is_off(tmp_path, section):
    render = _project(tmp_path, section)
    assert _rules_entries(render) == []
    with pytest.raises(KeyError):
        ScaffoldGalleryService(render).get_content(REVIEWER_RULES_NAME)


@pytest.mark.parametrize(
    "prompt",
    [
        "data/code_review/missing.md",
        "../outside.md",
        "/etc/hosts",
        "build/config.yml",
    ],
    ids=["missing", "climbs-out", "absolute", "build-zone"],
)
def test_absent_when_the_reviewer_would_refuse_the_prompt(tmp_path, prompt):
    render = _project(tmp_path, f"code_review:\n  enabled: true\n  prompt: {prompt}\n")
    assert _rules_entries(render) == []
    with pytest.raises(KeyError):
        ScaffoldGalleryService(render).get_content(REVIEWER_RULES_NAME)


@pytest.fixture
def client(tmp_path):
    render = _project(tmp_path, ON)
    app = bare_route_app(router, project_cwd=str(render), scaffold_write_enabled=True)
    return TestClient(app)


def test_routes_list_and_get_the_rules(client):
    names = [a["name"] for a in client.get("/api/scaffold").json()["artifacts"]]
    assert REVIEWER_RULES_NAME in names
    body = client.get(f"/api/scaffold/{REVIEWER_RULES_NAME}").json()
    assert body["content"] == RULES


def test_routes_offer_no_write_to_the_rules(client, tmp_path):
    """Even with gallery writes on, no verb reaches the reviewer's file."""
    prompt = tmp_path / "data" / "code_review" / "reviewer.md"
    assert client.post(f"/api/scaffold/{REVIEWER_RULES_NAME}/claim").status_code == 404
    saved = client.put(f"/api/scaffold/{REVIEWER_RULES_NAME}/override", json={"content": "x"})
    assert saved.status_code == 404
    assert client.get(f"/api/scaffold/{REVIEWER_RULES_NAME}/diff").status_code == 404
    assert prompt.read_text(encoding="utf-8") == RULES
