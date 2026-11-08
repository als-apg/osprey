"""``osprey build`` copies an application's declared skills into each render that shows it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.web_terminals.render_naming import render_project
from tests.cli._lifecycle_build import invoke_build
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo, drop_applications

KNOB: dict[str, Any] = {
    "name": "knob",
    "containers": {
        "knob": {
            "runtime": "osprey",
            "command": ["python", "-m", "knob_app.server"],
            "health": "/health",
        },
    },
    "skills": ["knob-tune"],
}

SKILL_MD = "---\nname: knob-tune\ndescription: Tune the knob.\n---\n\nTurn it gently.\n"


def _skill_dir(repo: Path, skill: str) -> Path:
    return repo / "apps" / "knob" / "skills" / skill


def _rendered_skills(repo: Path, login: str) -> Path:
    project, _ = render_project(EXEMPLAR_DIRNAME, login)
    return repo / "build" / project / ".claude" / "skills"


@pytest.fixture(scope="module")
def skills_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo = build_exemplar_repo(tmp_path_factory.mktemp("skills") / EXEMPLAR_DIRNAME)
    drop_applications(repo)
    app_dir = repo / "apps" / "knob"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.yml").write_text(yaml.safe_dump(KNOB, sort_keys=False), encoding="utf-8")
    tune = _skill_dir(repo, "knob-tune")
    (tune / "references").mkdir(parents=True)
    (tune / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (tune / "references" / "gain.md").write_text("gain table\n", encoding="utf-8")
    # A skill directory the manifest does not declare: it stays behind.
    stray = _skill_dir(repo, "knob-stray")
    stray.mkdir(parents=True)
    (stray / "SKILL.md").write_text("---\nname: knob-stray\ndescription: d\n---\n", "utf-8")
    with (repo / "profile.yml").open("a", encoding="utf-8") as profile:
        profile.write("\n" + yaml.safe_dump({"apps": {"knob": {}}}, sort_keys=False))
    (repo / "tiers" / "quiet.yml").write_text(
        "name: Quiet\nconfig:\n  control_system.writes: asked\napps:\n  knob:\n    shown: false\n",
        encoding="utf-8",
    )
    (repo / "tiers" / "asker.yml").write_text(
        "name: Asker\nconfig:\n  control_system.writes: asked\n", encoding="utf-8"
    )
    result = invoke_build(repo)
    assert result.exit_code == 0, result.output
    return repo


def test_a_showing_login_carries_the_declared_skill_tree(skills_repo: Path) -> None:
    skills = _rendered_skills(skills_repo, "asker")
    tree = skills / "knob-tune"
    copied = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*"))

    assert copied == ["SKILL.md", "references", "references/gain.md"]
    assert (skills / "knob-tune" / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD
    assert (skills / "knob-tune" / "references" / "gain.md").read_text() == "gain table\n"


def test_an_undeclared_skill_directory_is_not_copied(skills_repo: Path) -> None:
    assert not (_rendered_skills(skills_repo, "asker") / "knob-stray").exists()


def test_a_login_that_hides_the_application_carries_none_of_its_skills(skills_repo: Path) -> None:
    skills = _rendered_skills(skills_repo, "quiet")
    assert not (skills / "knob-tune").exists()
    assert not (skills / "knob-stray").exists()
