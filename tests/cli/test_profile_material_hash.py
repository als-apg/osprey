"""The profile content hash covers the files a profile points at, not just its YAML.

``compute_profile_hash`` is what the deploy-side staleness advisory compares
against, so anything a build consumes has to be inside it. A profile's ``data:``
tree is consumed verbatim by the build, which means a facility that edits a
channel database changes what a project would be built from while the profile
YAML itself is untouched — invisible to a hash over the YAML alone.

The convention directories and ``triggers.yml`` are folded on the same
principle; they are pinned separately in ``test_profile_hash_folding.py``.

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
    return _write(root / "profile.yml", "name: Facility\ndata: data\n")


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


def test_runtime_minted_material_does_not_move_the_hash(data_profile):
    """#716: material ``osprey up`` mints under ``data/.runtime/`` is not build input.

    The start itself writes CURVE keypairs into the data tree, so folding them
    would make every successful start mark its own build OUT OF DATE — a
    scaffolded boot unit then refuses to start after any of them. The reserved
    ``.runtime/`` subpath is spelled literally here, not through the constant,
    because the spelling IS the contract: the fold and the writer must agree on
    these exact bytes.
    """
    before = compute_profile_hash(data_profile)

    _write(
        data_profile.parent / "data" / ".runtime" / "bluesky_curve" / "bridge" / "proxy.key_secret",
        "minted-by-osprey-up\n",
    )

    assert compute_profile_hash(data_profile) == before


def test_authored_data_still_moves_the_hash_when_runtime_material_exists(data_profile):
    """The carve-out is surgical: authored ``data/`` edits beside it still count."""
    _write(
        data_profile.parent / "data" / ".runtime" / "bluesky_curve" / "bridge" / "proxy.key_secret",
        "minted-by-osprey-up\n",
    )
    before = compute_profile_hash(data_profile)

    _write(data_profile.parent / "data" / "docs" / "notes.md", "edited\n")

    assert compute_profile_hash(data_profile) != before


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


def test_shared_parent_data_tree_marks_profiles_outside_a_repository_stale(tmp_path):
    """The hash folds a data tree wherever it sits, for a profile outside any repository.

    Profiles outside any repository are hashed like any other: each names the
    tree beside it as ``../data``, and an edit to that tree moves every hash
    that folds it.
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


def test_profile_without_file_inputs_is_unaffected_by_neighbouring_files(tmp_path):
    """No ``data:`` means no tree material — the YAML alone decides.

    This is what keeps the bundled presets' pinned digests stable: they declare
    no file inputs, so nothing beside them can move their hash.
    """
    profile = _write(tmp_path / "profile.yml", "name: Plain\n")
    before = compute_profile_hash(profile)
    _write(tmp_path / "data" / "stray.json", "{}\n")
    _write(tmp_path / "unrelated.md", "text\n")
    assert compute_profile_hash(profile) == before


def test_inputs_folded_in_order_give_the_fold_digest(tmp_path):
    """The fold is exactly a walk over the material and overlay entries, in order."""
    import hashlib

    from osprey.cli.build_profile_document import _read_profile_document
    from osprey.cli.build_profile_merge import (
        ProfileInputRole,
        _fold_profile_material,
        _fold_source_tree,
        profile_inputs,
        resolve_profile_document,
    )
    from osprey.cli.profile_root import layer_files
    from osprey.cli.variant_selection import VARIANT_SETTING_FILENAME, VARIANT_SETTING_KEY

    root = tmp_path / "facility"
    _write(root / "data" / "channels.json", "{}\n")
    _write(root / "data" / ".runtime" / "minted.key", "runtime\n")
    _write(root / "rules" / "safety.md", "rule\n")
    _write(root / "skills" / "scan" / "SKILL.md", "skill\n")
    _write(root / "triggers.yml", "triggers: []\n")
    _write(root / "tiers" / "operator.yml", "name: operator\n")
    _write(root / "tiers" / "expert.yml", "name: expert\n")
    _write(root / "apps" / "logbook.yml", "name: logbook\n")
    _write(root / "profiles" / "teststand.yml", "# teststand\n")
    _write(root / VARIANT_SETTING_FILENAME, f"{VARIANT_SETTING_KEY}=teststand\n")
    profile = _write(
        root / "profile.yml", "name: Facility\ndata: data\ndispatch:\n  triggers: triggers.yml\n"
    )
    document = resolve_profile_document(_read_profile_document(profile), profile, warn=False)

    inputs = profile_inputs(document)
    folded = [entry for entry in inputs if entry.role is not ProfileInputRole.DOCUMENT]
    labels = [entry.label for entry in folded]
    assert labels[0] == "data"
    assert labels[-4:] == ["tier", "tier", "app", "variant"]
    assert "triggers" in labels
    assert [entry.path for entry in folded if entry.label in ("tier", "app")] == [
        path for _kind, path in layer_files(root)
    ]
    assert [entry.role for entry in inputs] == sorted(
        (entry.role for entry in inputs), key=list(ProfileInputRole).index
    )

    expected = hashlib.sha256()
    for entry in folded:
        _fold_source_tree(
            expected, entry.label, entry.path, skip_runtime_output=entry.label == "data"
        )
    actual = hashlib.sha256()
    _fold_profile_material(actual, inputs)
    assert actual.hexdigest() == expected.hexdigest()


def test_a_label_disagreeing_with_the_role_fails_at_construction(tmp_path):
    """A label is ``None`` exactly for a document; anything else never exists."""
    from osprey.cli.build_profile_merge import ProfileInput, ProfileInputRole

    with pytest.raises(ValueError):
        ProfileInput("x", None, tmp_path, ProfileInputRole.MATERIAL)
    with pytest.raises(ValueError):
        ProfileInput("profile.yml", "data", tmp_path, ProfileInputRole.DOCUMENT)
    with pytest.raises(ValueError):
        ProfileInput("", "data", tmp_path, ProfileInputRole.MATERIAL)


def test_the_data_key_is_the_tree_the_profile_names(tmp_path):
    """For a profile outside any repository, the data key is the tree as spelled."""
    from osprey.cli.build_profile_document import _read_profile_document
    from osprey.cli.build_profile_merge import profile_inputs, resolve_profile_document

    _write(tmp_path / "data" / "channels.json", "{}\n")
    profile = _write(tmp_path / "alice" / "profile.yml", "name: alice\ndata: ../data\n")
    document = resolve_profile_document(_read_profile_document(profile), profile, warn=False)

    data = [entry for entry in profile_inputs(document) if entry.label == "data"]

    assert [entry.key for entry in data] == ["../data/"]
    assert data[0].path == (tmp_path / "data").resolve()


def test_resolved_document_records_the_repo_local_extends_files(tmp_path):
    """The chain's non-preset files are recorded nearest first; a bundled preset is not."""
    from osprey.cli.build_profile_document import _read_profile_document
    from osprey.cli.build_profile_merge import resolve_profile_document

    root = tmp_path / "facility"
    _write(root / "site-base.yml", "extends: control-assistant\nname: Site\n")
    profile = _write(root / "profile.yml", "extends: site-base.yml\nname: Facility\n")

    document = resolve_profile_document(_read_profile_document(profile), profile, warn=False)

    assert document.extends_files == ((root / "site-base.yml").resolve(),)


def test_the_documents_are_the_root_profile_and_its_extends_files(tmp_path):
    """Every document the resolver read is an input; a bundled preset is not."""
    from osprey.cli.build_profile_document import _read_profile_document
    from osprey.cli.build_profile_merge import (
        ProfileInputRole,
        profile_inputs,
        resolve_profile_document,
    )

    root = tmp_path / "facility"
    _write(root / "site-base.yml", "extends: control-assistant\nname: Site\n")
    profile = _write(root / "profile.yml", "extends: site-base.yml\nname: Facility\n")
    document = resolve_profile_document(_read_profile_document(profile), profile, warn=False)

    documents = [
        entry for entry in profile_inputs(document) if entry.role is ProfileInputRole.DOCUMENT
    ]

    assert [entry.path for entry in documents] == [
        root / "profile.yml",
        *document.extends_files,
    ]
    assert [entry.key for entry in documents] == ["profile.yml", "site-base.yml"]


def _service_profile(root: Path, template: str) -> Path:
    _write(root / "svc" / "docker-compose.yml.j2", "services: {}\n")
    return _write(
        root / "profile.yml", f"name: Facility\nservices:\n  svc:\n    template: {template}\n"
    )


def test_a_profile_service_template_is_a_material_input(tmp_path):
    """A profile-relative template is material; a bundled ``osprey.<name>`` one is not."""
    from osprey.cli.build_profile_document import _read_profile_document
    from osprey.cli.build_profile_merge import (
        ProfileInputRole,
        profile_inputs,
        resolve_profile_document,
    )

    def service_entries(profile: Path) -> list:
        document = resolve_profile_document(_read_profile_document(profile), profile, warn=False)
        return [
            entry
            for entry in profile_inputs(document)
            if entry.label is not None and entry.label.startswith("services.")
        ]

    root = tmp_path / "facility"
    entries = service_entries(_service_profile(root, "svc"))

    assert [(entry.key, entry.label, entry.path, entry.role) for entry in entries] == [
        ("svc/", "services.svc.template", root / "svc", ProfileInputRole.MATERIAL)
    ]
    assert service_entries(_service_profile(tmp_path / "bundled", "osprey.postgresql")) == []


def test_editing_an_in_repo_service_template_moves_the_hash(tmp_path):
    profile = _service_profile(tmp_path / "facility", "svc")
    before = compute_profile_hash(profile)

    _write(profile.parent / "svc" / "docker-compose.yml.j2", "services:\n  svc: {}\n")

    assert compute_profile_hash(profile) != before
