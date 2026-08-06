"""The profile content hash covers the files a profile points at, not just its YAML.

``compute_profile_hash`` is what the deploy-side staleness advisory compares
against, so anything a build consumes has to be inside it. A profile's ``data:``
tree and its ``overlay:`` sources are consumed verbatim by the build, which means
a facility that edits a channel database or a persona file changes what a project
would be built from while the profile YAML itself is untouched — invisible to a
hash over the YAML alone.

The digest is pinned to a deterministic shape: every regular file contributes its
posix path relative to the source plus the SHA-256 of its bytes, entries sorted by
that path, directories skipped. Sorting removes filesystem walk order and posix
separators remove the platform, so two identical trees hash identically no matter
where or on what they live.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_profile import compute_profile_hash


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def data_profile(tmp_path):
    """A profile directory carrying its own ``data/`` tree."""
    root = tmp_path / "facility"
    _write(root / "data" / "channel_databases" / "als.json", '{"channels": []}\n')
    _write(root / "data" / "docs" / "notes.md", "notes\n")
    return _write(root / "profile.yml", "name: Facility\napp_template: hello_world\ndata: data\n")


def test_data_tree_edit_moves_the_hash(data_profile):
    """The case the advisory exists for: same YAML, different facility data."""
    before = compute_profile_hash(data_profile)
    _write(data_profile.parent / "data" / "channel_databases" / "als.json", '{"channels": [1]}\n')
    assert compute_profile_hash(data_profile) != before


def test_data_tree_addition_moves_the_hash(data_profile):
    before = compute_profile_hash(data_profile)
    _write(data_profile.parent / "data" / "channel_databases" / "extra.json", "{}\n")
    assert compute_profile_hash(data_profile) != before


def test_unchanged_data_tree_hashes_the_same(data_profile):
    """Re-hashing an untouched tree must not manufacture drift on every deploy."""
    assert compute_profile_hash(data_profile) == compute_profile_hash(data_profile)


def test_identical_trees_in_different_locations_hash_the_same(tmp_path):
    """Only content and relative layout may reach the digest — never a host path.

    Two copies of the same profile under different parents are the same build
    input, and the deterministic algorithm (posix relative paths, sorted) is what
    guarantees a rebuild elsewhere reads as current rather than as drift.
    """
    hashes = set()
    for location in ("alpha", "beta"):
        root = tmp_path / location
        _write(root / "data" / "b" / "second.json", "second\n")
        _write(root / "data" / "a" / "first.json", "first\n")
        profile = _write(root / "profile.yml", "name: Same\ndata: data\n")
        hashes.add(compute_profile_hash(profile))
    assert len(hashes) == 1


def test_empty_directory_does_not_move_the_hash(data_profile):
    """Directories contribute nothing — only the files under them."""
    before = compute_profile_hash(data_profile)
    (data_profile.parent / "data" / "empty_subdir").mkdir()
    assert compute_profile_hash(data_profile) == before


def test_shared_parent_data_tree_marks_sibling_profiles_stale(tmp_path):
    """Persona profiles share one ``../data`` tree; an edit must reach all of them.

    Sibling profiles live in their own directories beside a single data tree they
    all declare as ``../data``. Anchoring on the profile directory alone would
    resolve nothing there, so each persona project would keep deploying against a
    tree that has since changed.
    """
    _write(tmp_path / "data" / "channel_databases" / "als.json", '{"channels": []}\n')
    profiles = [
        _write(tmp_path / user / "profile.yml", f"name: {user}\ndata: ../data\n")
        for user in ("alice", "bob")
    ]
    before = [compute_profile_hash(p) for p in profiles]
    assert all(h is not None for h in before)

    _write(tmp_path / "data" / "channel_databases" / "als.json", '{"channels": [1]}\n')
    assert [compute_profile_hash(p) for p in profiles] != before


def test_missing_data_tree_still_hashes(tmp_path):
    """A profile naming a tree that isn't there is unbuildable, not unhashable.

    Returning ``None`` here would read as "cannot compare" and silence the
    version check alongside it.
    """
    profile = _write(tmp_path / "profile.yml", "name: Gone\ndata: nowhere\n")
    assert compute_profile_hash(profile) is not None


def test_overlay_file_edit_moves_the_hash(tmp_path):
    _write(tmp_path / "persona.md", "you are an operator\n")
    profile = _write(
        tmp_path / "profile.yml",
        "name: Overlaid\noverlay:\n  persona.md: .claude/rules/persona.md\n",
    )
    before = compute_profile_hash(profile)
    _write(tmp_path / "persona.md", "you are a physicist\n")
    assert compute_profile_hash(profile) != before


def test_overlay_directory_source_is_recursed(tmp_path):
    """A directory overlay copies its whole subtree, so the hash must see it all."""
    _write(tmp_path / "skills" / "diagnose" / "SKILL.md", "steps\n")
    profile = _write(
        tmp_path / "profile.yml",
        "name: Overlaid\noverlay:\n  skills: .claude/skills\n",
    )
    before = compute_profile_hash(profile)
    _write(tmp_path / "skills" / "diagnose" / "SKILL.md", "different steps\n")
    assert compute_profile_hash(profile) != before


def test_missing_overlay_source_still_hashes(tmp_path):
    profile = _write(
        tmp_path / "profile.yml",
        "name: Overlaid\noverlay:\n  absent.md: .claude/absent.md\n",
    )
    assert compute_profile_hash(profile) is not None


def test_profile_without_file_inputs_is_unaffected_by_neighbouring_files(tmp_path):
    """No ``data:``/``overlay:`` means no tree material — the YAML alone decides.

    This is what keeps the bundled presets' pinned digests stable: they declare
    no file inputs, so nothing beside them can move their hash.
    """
    profile = _write(tmp_path / "profile.yml", "name: Plain\napp_template: hello_world\n")
    before = compute_profile_hash(profile)
    _write(tmp_path / "data" / "stray.json", "{}\n")
    _write(tmp_path / "unrelated.md", "text\n")
    assert compute_profile_hash(profile) == before
