"""The profile hash covers every file the build reads, and personas resolve their root.

``compute_profile_hash`` is what the deploy-side staleness advisory compares
against, so anything a build consumes has to be inside it. Beyond the ``data:``
tree and ``overlay:`` sources pinned in ``test_profile_material_hash.py``, that
means the profile's convention directories (``rules/``, ``skills/`` and the rest
of the mapping table, including the ``project/`` verbatim mirror) and the
trigger file its ``dispatch:`` block names — a rule the agent reads or a trigger
a dispatcher fires on is as much build input as a channel database.

Personas are the second half. A ``personas/<name>.yml`` file holds only a delta,
so its hash is meaningless without the root it merges over: the hash resolves the
same implicit merge the build does, anchored at the profile root, which is what
makes an edit to the root — its YAML, its data tree, or its convention
directories — mark every persona's project stale too.

That relationship runs both ways: the deltas in ``personas/`` are folded into
the ROOT profile's hash as well, so editing one is drift for the deployment
that hosts those personas. It has to be, because a persona's rendered project
is re-rendered only when it is absent — a build wiping ``build/`` is what
re-arms it, and nothing would call for that build if the delta were outside the
fingerprint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli import build_profile_presets
from osprey.cli.build_profile import compute_preset_hash, compute_profile_hash
from osprey.cli.build_profile_merge import profile_triggers_material
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
    return _write(profile / "personas" / "reader.yml", "name: Reader\nmodel: claude-haiku-4-5\n")


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


def _name_triggers(profile: Path, triggers: str) -> None:
    """Give the fixture profile a ``dispatch:`` block naming ``triggers``."""
    path = profile / "profile.yml"
    path.write_text(
        path.read_text(encoding="utf-8") + f"dispatch:\n  triggers: {triggers}\n",
        encoding="utf-8",
    )


def test_triggers_file_is_folded(profile):
    """``triggers.yml`` is build input — the dispatcher fires on what it says."""
    _name_triggers(profile, "triggers.yml")
    triggers = _write(profile / "triggers.yml", "triggers: []\n")
    before = compute_profile_hash(profile / "profile.yml")

    triggers.write_text("triggers:\n  - name: alarm\n", encoding="utf-8")

    assert compute_profile_hash(profile / "profile.yml") != before


def test_the_trigger_file_dispatch_names_is_folded(profile):
    """A trigger file nested beside the profile is build input wherever it lives."""
    _name_triggers(profile, "triggers/x.yml")
    triggers = _write(profile / "triggers" / "x.yml", "triggers: []\n")
    before = compute_profile_hash(profile / "profile.yml")

    triggers.write_text("triggers:\n  - name: alarm\n", encoding="utf-8")

    assert compute_profile_hash(profile / "profile.yml") != before


def test_a_triggers_file_no_dispatch_block_names_is_not_folded(profile):
    """A ``triggers.yml`` nothing reads is not build input."""
    triggers = _write(profile / "triggers.yml", "triggers: []\n")
    before = compute_profile_hash(profile / "profile.yml")

    triggers.write_text("triggers:\n  - name: alarm\n", encoding="utf-8")

    assert compute_profile_hash(profile / "profile.yml") == before


def test_a_bundled_trigger_file_is_never_folded(profile, tmp_path, monkeypatch):
    """A bundled trigger file moves with the package, never with the profile."""
    bundled_dir = tmp_path / "bundled"
    bundled = _write(bundled_dir / "bundled.yml", "triggers: []\n")
    monkeypatch.setattr(build_profile_presets, "_triggers_dir", lambda: bundled_dir)
    _name_triggers(profile, "bundled.yml")
    before = compute_profile_hash(profile / "profile.yml")

    bundled.write_text("triggers:\n  - name: upgraded\n", encoding="utf-8")

    assert compute_profile_hash(profile / "profile.yml") == before


def test_a_profile_file_shadowing_a_bundled_name_is_folded(profile, tmp_path, monkeypatch):
    """The profile's own copy is what the build copies, so it is what is folded."""
    bundled_dir = tmp_path / "bundled"
    _write(bundled_dir / "shared.yml", "triggers: []\n")
    monkeypatch.setattr(build_profile_presets, "_triggers_dir", lambda: bundled_dir)
    _name_triggers(profile, "shared.yml")
    local = _write(profile / "shared.yml", "triggers: []\n")
    before = compute_profile_hash(profile / "profile.yml")

    local.write_text("triggers:\n  - name: local\n", encoding="utf-8")

    assert compute_profile_hash(profile / "profile.yml") != before


@pytest.mark.parametrize("spelling", ["triggers.yml", "./triggers.yml"])
def test_the_default_trigger_spelling_folds_the_root_file(profile, spelling):
    """The default spelling keys and folds the root file exactly as before."""
    _write(profile / "triggers.yml", "triggers: []\n")
    resolved = {"dispatch": {"triggers": spelling}}

    key, path = profile_triggers_material(resolved, profile) or ("", Path())

    assert key == "triggers.yml"
    assert path.resolve() == (profile / "triggers.yml").resolve()


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


def test_editing_a_delta_marks_the_ROOT_profile_stale(profile, persona):
    """The gap this closes: a persona edit that no verb ever noticed.

    ``personas/`` is source zone, and a persona's rendered project is build
    output — but ``up`` re-renders one only when its directory is *absent*, and
    what re-arms that is a build wiping ``build/``. If the delta were outside
    the fingerprint, editing it would leave the deploy-side check calling the
    build clean, so nothing would refuse, nothing would rebuild, and the
    superseded render would keep going into that persona's image.
    """
    before = compute_profile_hash(profile / "profile.yml")

    persona.write_text("name: Reader\nmodel: claude-opus-5-5\n", encoding="utf-8")

    assert compute_profile_hash(profile / "profile.yml") != before


@pytest.mark.parametrize("change", ["added", "removed"])
@pytest.mark.usefixtures("persona")
def test_the_set_of_deltas_is_part_of_the_root_hash(profile, change):
    """A persona the catalog can reference is build input the moment it exists."""
    other = profile / "personas" / "writer.yml"
    if change == "removed":
        other.write_text("name: Writer\n", encoding="utf-8")
    before = compute_profile_hash(profile / "profile.yml")

    if change == "removed":
        other.unlink()
    else:
        other.write_text("name: Writer\n", encoding="utf-8")

    assert compute_profile_hash(profile / "profile.yml") != before


def test_a_profile_with_no_deltas_hashes_as_it_did_before_personas_were_folded(profile):
    """Backward compatibility, stated as the property that guarantees it.

    A repo with no ``personas/`` must keep the hash its build stamped, or every
    such deployment would read as drifted the moment it upgraded. The fold
    contributes exactly nothing unless a delta is there to fold — an absent
    directory, an empty one, and one holding only material that cannot be built
    all leave the digest identical — so "no deltas" is byte-for-byte the
    pre-fold hash.
    """
    before = compute_profile_hash(profile / "profile.yml")
    assert not (profile / "personas").exists()

    (profile / "personas").mkdir()
    assert compute_profile_hash(profile / "profile.yml") == before

    _write(profile / "personas" / ".DS_Store", "editor noise\n")
    assert compute_profile_hash(profile / "profile.yml") == before

    _write(profile / "personas" / "nested" / "notes.md", "not a delta\n")
    assert compute_profile_hash(profile / "profile.yml") == before


def test_a_sibling_delta_moves_a_personas_own_hash(profile, persona):
    """The accepted over-report, pinned so it is a decision rather than a drift.

    A delta anchors at the profile root, so the root's whole persona set is its
    material too. Editing one delta therefore marks its siblings stale — an
    idempotent rebuild, deliberately preferred over subtracting siblings and
    risking a persona that no longer matches its source.
    """
    sibling = _write(profile / "personas" / "writer.yml", "name: Writer\nmodel: claude-haiku-4-5\n")
    before = compute_profile_hash(persona)

    sibling.write_text("name: Writer\nmodel: claude-opus-5-5\n", encoding="utf-8")

    assert compute_profile_hash(persona) != before


def test_persona_hash_differs_from_its_root(profile, persona):
    """A persona builds a different project than its root, so it hashes differently."""
    assert compute_profile_hash(persona) != compute_profile_hash(profile / "profile.yml")


@pytest.mark.usefixtures("profile")
def test_editing_the_delta_changes_the_persona_hash(persona):
    """The persona's own layer is part of what it resolves to."""
    before = compute_profile_hash(persona)

    persona.write_text("name: Reader\nmodel: claude-opus-5-5\n", encoding="utf-8")

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
    other = _write(profile / "personas" / "writer.yml", "name: Writer\nmodel: claude-opus-5-5\n")

    assert compute_profile_hash(other) != compute_profile_hash(persona)


@pytest.mark.usefixtures("persona")
def test_persona_inherits_the_roots_declarations(profile):
    """The merge really happens: a root key the delta omits still shapes the hash.

    Pinned by contrast with a delta that spells the same key itself — if the
    root were ignored, the two would hash identically.
    """
    _write(
        profile / "profile.yml",
        "name: Facility\napp_template: hello_world\ndata: data\nmodel: claude-opus-5-5\n",
    )
    inheriting = _write(profile / "personas" / "inherits.yml", "name: Reader\n")
    overriding = _write(
        profile / "personas" / "overrides.yml", "name: Reader\nmodel: claude-haiku-4-5\n"
    )

    assert compute_profile_hash(inheriting) != compute_profile_hash(overriding)


@pytest.mark.usefixtures("persona")
def test_persona_delta_exclude_is_applied(profile):
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
