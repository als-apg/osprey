"""The application's agent skill: present, named for the app, and accepted by the loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from orbit_bump import APP_NAME

from osprey.apps.loader import _refuse_missing_skills, load
from osprey.profiles import presets

PRESETS_ROOT = Path(presets.__file__).parent
SKILL = PRESETS_ROOT / "apps" / APP_NAME / "skills" / APP_NAME / "SKILL.md"


def _frontmatter(text: str) -> dict:
    assert text.startswith("---\n")
    head, _, _ = text[4:].partition("\n---\n")
    return yaml.safe_load(head)


def test_the_loader_accepts_the_preset_app_with_its_skill():
    manifests = load(PRESETS_ROOT, [APP_NAME])
    assert [m.name for m in manifests] == [APP_NAME]
    assert manifests[0].skills == [APP_NAME]
    _refuse_missing_skills(PRESETS_ROOT, tuple(manifests))


def test_the_skill_frontmatter_names_it_and_says_when_to_trigger():
    meta = _frontmatter(SKILL.read_text(encoding="utf-8"))
    assert meta["name"] == APP_NAME
    description = meta["description"].lower()
    for phrase in ("bump the orbit", "local bump", "move bpm"):
        assert phrase in description


def test_the_skill_covers_both_tools_and_every_outcome():
    body = SKILL.read_text(encoding="utf-8")
    for word in (
        "plan_bump",
        "apply_bump",
        "from golden",
        "not_applicable",
        "partially_applied",
        "confirmed",
        "mismatch",
        "refused",
        "failed",
        "unconfirmed",
        "not_attempted",
        "Stale",
        "Writes off",
        "Unarmed login",
    ):
        assert word in body, word


def test_the_skill_names_no_protocol_or_facility_prefix():
    body = SKILL.read_text(encoding="utf-8")
    for word in ("MCP", "HTTP", "POST ", "GET ", "/api/", "ALS", "SR:"):
        assert word not in body, word
