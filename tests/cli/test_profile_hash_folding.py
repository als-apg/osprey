"""The profile hash covers every file the build reads, and personas resolve their root.

``compute_profile_hash`` is what the deploy-side staleness advisory compares
against, so anything a build consumes has to be inside it. Beyond the ``data:``
tree and ``overlay:`` sources pinned in ``test_profile_material_hash.py``, that
means the profile's convention directories (``rules/``, ``skills/`` and the rest
of the mapping table, including the ``project/`` verbatim mirror) and its
``triggers.yml`` — a rule the agent reads or a trigger a dispatcher fires on is
as much build input as a channel database.

Personas are the second half. A ``personas/<name>.yml`` file holds only a delta,
so its hash is meaningless without the root it merges over: the hash resolves the
same implicit merge the build does, anchored at the profile root, which is what
makes an edit to the root — its YAML, its data tree, or its convention
directories — mark every persona's project stale too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli import build_profile_presets
from osprey.cli.build_profile import compute_preset_hash, compute_profile_hash
from osprey.cli.profile_conventions import CONVENTION_SOURCES


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def profile(tmp_path):
    """A materialized-looking profile root with a data tree."""
    root = tmp_path / "facility"
    _write(root / "data" / "channels.json", '{"channels": []}\n')
    _write(root / "profile.yml", "name: Facility\napp_template: hello_world\ndata: data\n")
    return root


@pytest.fixture
def persona(profile):
    """A persona delta under the profile root."""
    return _write(profile / "personas" / "reader.yml", "name: Reader\ntier: 1\n")


# ── Convention material ──────────────────────────────────────────────────────


# One representative artifact per convention category, in the shape that
# category expects — the fold is shape-blind, but a realistic tree is the one
# a facility will actually have.
_CONVENTION_ARTIFACTS: dict[str, str] = {
    "rules": "rules/safety.md",
    "skills": "skills/orbit/SKILL.md",
    "agents": "agents/orbit-writer.md",
    "commands": "commands/scan.md",
    "output-styles": "output-styles/terse.md",
    "hooks": "hooks/facility_guard.py",
    "web-terminal-context": "web-terminal-context/alice/notes.md",
    "mcp_servers": "mcp_servers/facility/server.py",
    "services": "services/archiver/compose.yml",
    "project": "project/docs/runbook.md",
}


def test_every_convention_category_is_represented():
    """A new convention directory must be added to the fixtures below.

    Without this the parametrized tests would silently stop covering a category
    the mapping table gained.
    """
    assert sorted(_CONVENTION_ARTIFACTS) == sorted(CONVENTION_SOURCES)


@pytest.mark.parametrize("category", sorted(_CONVENTION_ARTIFACTS))
def test_adding_a_convention_artifact_changes_the_hash(profile, category):
    """A profile that gains an artifact is a profile the project no longer matches."""
    before = compute_profile_hash(profile / "profile.yml")

    _write(profile / _CONVENTION_ARTIFACTS[category], "original\n")

    assert compute_profile_hash(profile / "profile.yml") != before


@pytest.mark.parametrize("category", sorted(_CONVENTION_ARTIFACTS))
def test_editing_a_convention_artifact_changes_the_hash(profile, category):
    """Editing the contents counts, not just the file list."""
    artifact = _write(profile / _CONVENTION_ARTIFACTS[category], "original\n")
    before = compute_profile_hash(profile / "profile.yml")

    artifact.write_text("edited\n", encoding="utf-8")

    assert compute_profile_hash(profile / "profile.yml") != before


@pytest.mark.parametrize("category", sorted(_CONVENTION_ARTIFACTS))
def test_removing_a_convention_artifact_changes_the_hash(profile, category):
    """A deleted artifact moves the hash the same way an added one does."""
    artifact = _write(profile / _CONVENTION_ARTIFACTS[category], "original\n")
    before = compute_profile_hash(profile / "profile.yml")

    artifact.unlink()

    assert compute_profile_hash(profile / "profile.yml") != before


def test_triggers_file_is_folded(profile):
    """``triggers.yml`` is build input — the dispatcher fires on what it says."""
    triggers = _write(profile / "triggers.yml", "triggers: []\n")
    before = compute_profile_hash(profile / "profile.yml")

    triggers.write_text("triggers:\n  - name: alarm\n", encoding="utf-8")

    assert compute_profile_hash(profile / "profile.yml") != before


def test_unrelated_file_does_not_change_the_hash(profile):
    """A profile is a directory people document; prose is not build input."""
    before = compute_profile_hash(profile / "profile.yml")

    _write(profile / "README.md", "# Facility profile\n")

    assert compute_profile_hash(profile / "profile.yml") == before


def test_hash_is_stable_across_repeated_calls(profile):
    """Nothing in the fold depends on walk order or wall-clock state."""
    _write(profile / "rules" / "safety.md", "no writes above 1 A\n")

    assert compute_profile_hash(profile / "profile.yml") == compute_profile_hash(
        profile / "profile.yml"
    )


def test_bundled_presets_do_not_fold_their_shared_directory(tmp_path, monkeypatch):
    """Presets share one package directory, so its neighbours are not their material.

    Folding there would couple every preset's hash to every other preset's
    directories — an edit to the multi-user demo's seeded context would report
    drift on a hello-world deployment.
    """
    presets = tmp_path / "presets"
    _write(presets / "base.yml", "name: base\napp_template: hello_world\n")
    rules = _write(presets / "rules" / "safety.md", "original\n")
    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: presets)
    before = compute_preset_hash("base")

    rules.write_text("edited\n", encoding="utf-8")

    assert before is not None
    assert compute_preset_hash("base") == before


# ── Persona deltas ───────────────────────────────────────────────────────────


def test_persona_hash_differs_from_its_root(profile, persona):
    """A persona builds a different project than its root, so it hashes differently."""
    assert compute_profile_hash(persona) != compute_profile_hash(profile / "profile.yml")


def test_editing_the_delta_changes_the_persona_hash(profile, persona):
    """The persona's own layer is part of what it resolves to."""
    before = compute_profile_hash(persona)

    persona.write_text("name: Reader\ntier: 3\n", encoding="utf-8")

    assert compute_profile_hash(persona) != before


def test_editing_the_root_profile_changes_the_persona_hash(profile, persona):
    """The root is half of what a persona resolves to — an edit there reaches it."""
    before = compute_profile_hash(persona)

    _write(
        profile / "profile.yml",
        "name: Facility\napp_template: hello_world\ndata: data\nchannel_finder_mode: in_context\n",
    )

    assert compute_profile_hash(persona) != before


def test_editing_the_root_data_tree_changes_the_persona_hash(profile, persona):
    """A persona reads the root's data tree, not one of its own."""
    before = compute_profile_hash(persona)

    _write(profile / "data" / "channels.json", '{"channels": ["BEAM:CURRENT"]}\n')

    assert compute_profile_hash(persona) != before


def test_editing_a_root_convention_dir_changes_the_persona_hash(profile, persona):
    """Convention artifacts reach persona projects through the same root."""
    _write(profile / "rules" / "safety.md", "original\n")
    before = compute_profile_hash(persona)

    _write(profile / "rules" / "safety.md", "edited\n")

    assert compute_profile_hash(persona) != before


def test_persona_material_anchors_at_the_root_not_at_personas(profile, persona):
    """A tree beside the delta is not build input — the anchor is the root."""
    before = compute_profile_hash(persona)

    _write(profile / "personas" / "rules" / "stray.md", "not a convention directory\n")

    assert compute_profile_hash(persona) == before


def test_two_personas_under_one_root_hash_differently(profile, persona):
    """The delta is part of the digest, so siblings do not collide."""
    other = _write(profile / "personas" / "writer.yml", "name: Writer\ntier: 3\n")

    assert compute_profile_hash(other) != compute_profile_hash(persona)


def test_persona_inherits_the_roots_declarations(profile, persona):
    """The merge really happens: a root key the delta omits still shapes the hash.

    Pinned by contrast with a delta that spells the same key itself — if the
    root were ignored, the two would hash identically.
    """
    _write(
        profile / "profile.yml",
        "name: Facility\napp_template: hello_world\ndata: data\ntier: 3\n",
    )
    inheriting = _write(profile / "personas" / "inherits.yml", "name: Reader\n")
    overriding = _write(profile / "personas" / "overrides.yml", "name: Reader\ntier: 1\n")

    assert compute_profile_hash(inheriting) != compute_profile_hash(overriding)


def test_persona_delta_exclude_is_applied(profile, persona):
    """``exclude:`` subtracts from the merged result, so it moves the hash."""
    _write(
        profile / "profile.yml",
        "name: Facility\napp_template: hello_world\ndata: data\nskills:\n  - orbit\n  - scan\n",
    )
    plain = _write(profile / "personas" / "plain.yml", "name: Reader\n")
    excluding = _write(
        profile / "personas" / "excluding.yml",
        "name: Reader\nexclude:\n  skills:\n    - scan\n",
    )

    assert compute_profile_hash(excluding) != compute_profile_hash(plain)


def test_persona_without_a_root_has_no_hash(tmp_path):
    """An orphan delta cannot be built, so there is nothing to compare against."""
    orphan = _write(tmp_path / "stray" / "personas" / "reader.yml", "name: Reader\n")

    assert compute_profile_hash(orphan) is None
