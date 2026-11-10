"""An application's agent skills: declared in app.yml, checked, and copied into each render that shows it."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.apps.errors import AppManifestError
from osprey.apps.loader import _refuse_missing_skills
from osprey.apps.manifest import AppManifest, parse_manifest
from osprey.apps.projection import login_skills
from osprey.apps.rules import AppSelection, LoginSelection
from osprey.cli.build_cmd import _copy_app_skills
from osprey_connectors.errors import BuildProfileError


def _manifest(name: str, skills: list[str]) -> AppManifest:
    data = {
        "name": name,
        "containers": {name: {"runtime": "osprey", "health": "/health"}},
        "skills": skills,
    }
    return parse_manifest(data, app=name)


def _login(**apps: bool) -> LoginSelection:
    return LoginSelection(
        audience="person",
        apps={n: AppSelection(shown=s) for n, s in apps.items()},
        tier="asked",
        shared_cards=(),
    )


def _skill(root: Path, app: str, skill: str) -> Path:
    path = root / "apps" / app / "skills" / skill / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(f"---\nname: {skill}\ndescription: d\n---\n", encoding="utf-8")
    return path


def test_skills_default_to_none():
    data = {"name": "knob", "containers": {"knob": {"runtime": "osprey", "health": "/health"}}}
    assert parse_manifest(data, app="knob").skills == []


def test_a_skill_name_is_a_slug():
    with pytest.raises(AppManifestError):
        _manifest("knob", ["Knob Skill"])


def test_a_listed_skill_needs_its_skill_file(tmp_path: Path):
    with pytest.raises(AppManifestError, match="apps/knob/skills/knob/SKILL.md does not exist"):
        _refuse_missing_skills(tmp_path, (_manifest("knob", ["knob"]),))


def test_a_listed_skill_with_its_skill_file_passes(tmp_path: Path):
    _skill(tmp_path, "knob", "knob")
    _refuse_missing_skills(tmp_path, (_manifest("knob", ["knob"]),))


def test_two_applications_cannot_carry_one_skill_name(tmp_path: Path):
    _skill(tmp_path, "knob", "corrector")
    _skill(tmp_path, "dial", "corrector")
    manifests = (_manifest("knob", ["corrector"]), _manifest("dial", ["corrector"]))
    with pytest.raises(AppManifestError, match="already carries a skill named 'corrector'"):
        _refuse_missing_skills(tmp_path, manifests)


def test_a_render_carries_the_skills_of_the_applications_it_shows():
    manifests = [_manifest("knob", ["knob", "knob-tune"]), _manifest("dial", ["dial"])]
    assert login_skills(manifests, _login(knob=True, dial=False)) == [
        ("knob", "knob"),
        ("knob", "knob-tune"),
    ]


def test_an_application_the_login_does_not_name_carries_no_skill():
    assert login_skills([_manifest("knob", ["knob"])], _login()) == []


def test_the_skill_is_copied_into_the_render(tmp_path: Path):
    source = _skill(tmp_path, "knob", "knob")
    (source.parent / "reference.md").write_text("ref", encoding="utf-8")
    render = tmp_path / "build"
    copied = _copy_app_skills(tmp_path, render, [_manifest("knob", ["knob"])], _login(knob=True))
    assert copied == ["knob"]
    assert (render / ".claude/skills/knob/SKILL.md").read_text() == source.read_text()
    assert (render / ".claude/skills/knob/reference.md").read_text() == "ref"


def test_a_skill_directory_the_manifest_does_not_declare_is_not_copied(tmp_path: Path):
    _skill(tmp_path, "knob", "knob")
    _skill(tmp_path, "knob", "stray")
    render = tmp_path / "build"
    copied = _copy_app_skills(tmp_path, render, [_manifest("knob", ["knob"])], _login(knob=True))
    assert copied == ["knob"]
    assert not (render / ".claude/skills/stray").exists()


def test_a_hidden_application_copies_nothing(tmp_path: Path):
    _skill(tmp_path, "knob", "knob")
    render = tmp_path / "build"
    copied = _copy_app_skills(tmp_path, render, [_manifest("knob", ["knob"])], _login(knob=False))
    assert copied == []
    assert not (render / ".claude/skills/knob").exists()


def test_a_skill_name_the_render_already_has_is_refused(tmp_path: Path):
    _skill(tmp_path, "knob", "diagnose")
    render = tmp_path / "build"
    (render / ".claude/skills/diagnose").mkdir(parents=True)
    with pytest.raises(BuildProfileError, match="already has one by that name"):
        _copy_app_skills(tmp_path, render, [_manifest("knob", ["diagnose"])], _login(knob=True))
