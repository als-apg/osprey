"""A mounted machine root is a profile input, folded and staged as machine facts.

``machine: <dir>`` names the facility directory of a machine repository mounted
inside the deployment repository (a submodule or a nested clone). The profile
hash folds it, so a machine edit marks the build stale; the repository walk
covers it, so a link escaping the repository is refused; and image staging
copies it while keeping the mount's own repository files (its ``.git``,
profile, ``build/``, ``var/``) out of the image.

Every repository here is ``git init``-ed, so its root holds ``.git`` the way a
real deployment's does.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from osprey.cli.build_cmd import (
    _image_build_inputs,
    _image_machine_input,
    _stage_source_zone,
)
from osprey.cli.build_profile_document import _read_profile_document
from osprey.cli.build_profile_merge import (
    MACHINE_INPUT_LABEL,
    _fold_profile_material,
    _fold_source_tree,
    inputs_outside_repository,
    profile_inputs,
    refuse_inputs_outside_repository,
    resolve_profile_document,
)
from osprey.utils.workspace import PROFILE_FILENAME
from osprey_connectors.errors import BuildProfileError
from tests._builds import init_project

pytestmark = [pytest.mark.slow]


def _git_init(directory: Path) -> None:
    subprocess.run(["git", "init", "-q", str(directory)], check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A freshly initialised deployment repository whose root holds ``.git``."""
    root = init_project(tmp_path / "parent", "hello-world", "demo")
    _git_init(root)
    return root


def _write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _machine_tree(root: Path) -> Path:
    """A minimal machine facility directory at *root*."""
    _write(root / "identity.yaml", "code: m\nname: M\n")
    _write(root / "records" / "bpm.yaml", "- name: BPM1\n")
    return root


def _mount(repo: Path, top: str = "vendor/x", *, versioned: bool = True) -> Path:
    """A machine repository at ``repo/<top>`` with its facility at ``data/facility``.

    It carries its own repository files: a profile, ``build/``, ``var/``,
    ``.claude/``, ``.env`` and a sibling data directory, as a clone would.
    """
    mount = repo / top
    mount.mkdir(parents=True)
    if versioned:
        _git_init(mount)
    _write(mount / PROFILE_FILENAME, "name: machine-repo\n")
    _write(mount / "build" / "out.txt")
    _write(mount / "var" / "state.txt")
    _write(mount / ".claude" / "settings.json", "{}\n")
    _write(mount / ".env", "SECRET=1\n")
    _write(mount / "README.md", "the machine\n")
    _write(mount / "data" / "other" / "x.yaml")
    machine = _machine_tree(mount / "data" / "facility")
    _set_machine(repo, f"{top}/data/facility")
    return machine


def _set_machine(repo: Path, spelling: str) -> None:
    profile = repo / PROFILE_FILENAME
    profile.write_text(
        f"machine: {spelling}\n{profile.read_text(encoding='utf-8')}", encoding="utf-8"
    )


def _document(repo: Path):
    profile = repo / PROFILE_FILENAME
    return resolve_profile_document(_read_profile_document(profile), profile, warn=False)


def _hash(repo: Path) -> str:
    digest = hashlib.sha256()
    _fold_profile_material(digest, profile_inputs(_document(repo)))
    return digest.hexdigest()


def _stage(repo: Path, tmp_path: Path) -> Path:
    image = tmp_path / "image"
    image.mkdir()
    _stage_source_zone(
        repo,
        image,
        build_inputs=_image_build_inputs(repo),
        machine=_image_machine_input(repo),
    )
    return image


def _staged(image: Path) -> set[str]:
    return {path.relative_to(image).as_posix() for path in image.rglob("*")}


# ---------------------------------------------------------------------------
# The input and the hash
# ---------------------------------------------------------------------------


def test_a_mounted_machine_is_a_material_input_labelled_machine(repo: Path) -> None:
    machine = _mount(repo)

    entries = [e for e in profile_inputs(_document(repo)) if e.label == MACHINE_INPUT_LABEL]

    assert len(entries) == 1
    assert entries[0].key == "vendor/x/data/facility/"
    assert entries[0].path.resolve() == machine.resolve()
    assert entries[0].role.value == "material"


def test_a_profile_without_machine_lists_no_machine_input(repo: Path) -> None:
    assert all(e.label != MACHINE_INPUT_LABEL for e in profile_inputs(_document(repo)))
    assert _image_machine_input(repo) is None


def test_a_machine_edit_moves_the_profile_hash(repo: Path) -> None:
    machine = _mount(repo)
    before = _hash(repo)

    _write(machine / "records" / "bpm.yaml", "- name: BPM2\n")

    assert _hash(repo) != before


def test_a_new_machine_file_moves_the_profile_hash(repo: Path) -> None:
    machine = _mount(repo)
    before = _hash(repo)

    _write(machine / "decks" / "lattice.yaml")

    assert _hash(repo) != before


@pytest.mark.parametrize(
    "relative",
    [
        "vendor/x/.git/HEAD-extra",
        "vendor/x/build/out.txt",
        "vendor/x/var/state.txt",
        "vendor/x/.env",
        "vendor/x/data/facility/.git",
        "vendor/x/data/facility/records/.git/config",
        "vendor/x/data/facility/knowledge/notes.md",
    ],
)
def test_edits_outside_the_machine_facts_do_not_move_the_hash(repo: Path, relative: str) -> None:
    _mount(repo)
    before = _hash(repo)

    _write(repo / relative, "changed\n")

    assert _hash(repo) == before


def test_the_fold_skips_vcs_metadata_at_any_depth(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    _write(tree / "a.yaml")

    def digest() -> str:
        d = hashlib.sha256()
        _fold_source_tree(d, "data", tree)
        return d.hexdigest()

    before = digest()
    _write(tree / ".git" / "config")
    _write(tree / "sub" / "deeper" / ".git", "gitdir: elsewhere\n")
    assert digest() == before

    _write(tree / ".gitignore", "*.tmp\n")
    assert digest() != before


def test_a_tree_without_nested_git_folds_as_before(tmp_path: Path) -> None:
    """The fold of an ordinary tree is today's: path NUL sha NUL, sorted."""
    tree = tmp_path / "tree"
    _write(tree / "b.yaml", "b\n")
    _write(tree / "a" / "c.yaml", "c\n")

    expected = hashlib.sha256()
    expected.update(b"data\0")
    for rel, text in (("a/c.yaml", b"c\n"), ("b.yaml", b"b\n")):
        expected.update(rel.encode() + b"\0")
        expected.update(hashlib.sha256(text).hexdigest().encode() + b"\0")

    actual = hashlib.sha256()
    _fold_source_tree(actual, "data", tree)
    assert actual.hexdigest() == expected.hexdigest()


# ---------------------------------------------------------------------------
# The repository boundary
# ---------------------------------------------------------------------------


def test_a_symlinked_machine_escaping_the_repository_is_refused(repo: Path, tmp_path: Path) -> None:
    outside = _machine_tree(tmp_path / "outside" / "facility")
    (repo / "machine-link").symlink_to(outside, target_is_directory=True)
    _set_machine(repo, "machine-link")
    document = _document(repo)

    outside_entries = inputs_outside_repository(document)

    assert [o.entry.label for o in outside_entries] == [MACHINE_INPUT_LABEL]
    with pytest.raises(BuildProfileError, match="machine-link"):
        refuse_inputs_outside_repository(document)


def test_a_link_inside_the_machine_escaping_the_repository_is_refused(
    repo: Path, tmp_path: Path
) -> None:
    machine = _mount(repo)
    escape = tmp_path / "escape"
    escape.mkdir()
    (machine / "linked").symlink_to(escape, target_is_directory=True)

    outside_entries = inputs_outside_repository(_document(repo))

    assert [(o.entry.label, o.link) for o in outside_entries] == [(MACHINE_INPUT_LABEL, "linked")]


# ---------------------------------------------------------------------------
# Image staging
# ---------------------------------------------------------------------------


def test_staging_keeps_the_machine_and_drops_the_mount_repository(
    repo: Path, tmp_path: Path
) -> None:
    _mount(repo)

    staged = _staged(_stage(repo, tmp_path))

    assert "vendor/x/data/facility/identity.yaml" in staged
    assert "vendor/x/data/facility/records/bpm.yaml" in staged
    assert PROFILE_FILENAME in staged
    for dropped in (
        ".git",
        "vendor/x/.git",
        f"vendor/x/{PROFILE_FILENAME}",
        "vendor/x/build",
        "vendor/x/var",
        "vendor/x/.claude",
        "vendor/x/.env",
        "vendor/x/README.md",
        "vendor/x/data/other",
    ):
        assert dropped not in staged, dropped
    assert not any(part == ".git" for path in staged for part in path.split("/"))


def test_staging_a_top_level_submodule_mount(repo: Path, tmp_path: Path) -> None:
    """A submodule's ``.git`` is a file; the mount top is found all the same."""
    mount = repo / "als"
    _write(mount / ".git", "gitdir: ../.git/modules/als\n")
    _write(mount / PROFILE_FILENAME, "name: als\n")
    _write(mount / "build" / "x.txt")
    _machine_tree(mount / "data" / "facility")
    _set_machine(repo, "als/data/facility")

    staged = _staged(_stage(repo, tmp_path))

    assert "als/data/facility/identity.yaml" in staged
    assert "als/.git" not in staged
    assert f"als/{PROFILE_FILENAME}" not in staged
    assert "als/build" not in staged


def test_staging_a_machine_root_that_is_its_own_mount_top(repo: Path, tmp_path: Path) -> None:
    machine = _machine_tree(repo / "facility-repo")
    _git_init(machine)
    _write(machine / "decks" / "lattice.yaml")
    _set_machine(repo, "facility-repo")

    staged = _staged(_stage(repo, tmp_path))

    assert "facility-repo/identity.yaml" in staged
    assert "facility-repo/decks/lattice.yaml" in staged
    assert "facility-repo/.git" not in staged


def test_an_unversioned_machine_directory_keeps_its_siblings(repo: Path, tmp_path: Path) -> None:
    _mount(repo, "vendor/plain", versioned=False)

    staged = _staged(_stage(repo, tmp_path))

    assert "vendor/plain/data/facility/identity.yaml" in staged
    assert "vendor/plain/README.md" in staged
    assert "vendor/plain/data/other/x.yaml" in staged
    assert f"vendor/plain/{PROFILE_FILENAME}" in staged
    assert "vendor/plain/.env" not in staged


def test_staging_skips_nested_vcs_metadata_without_a_machine(repo: Path, tmp_path: Path) -> None:
    _write(repo / "data" / "vendored" / ".git" / "config")
    _write(repo / "data" / "vendored" / "keep.yaml")

    staged = _staged(_stage(repo, tmp_path))

    assert "data/vendored/keep.yaml" in staged
    assert "data/vendored/.git" not in staged
    assert PROFILE_FILENAME in staged
    assert ".git" not in staged
