"""The two facility roots and the predicates every walk of the machine root uses.

A deployment either has one facility tree playing both roles, or mounts a
machine root beside its content root. These tests pin the path arithmetic in
``osprey.facility.roots``: which relative paths are version-control metadata,
and which are machine entries, single-tree and mounted.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path, PurePath, PurePosixPath

import pytest

from osprey.facility.roots import (
    DEPLOYMENT_CONTENT_DIRS,
    FacilityRoots,
    is_machine_entry,
    is_vcs_metadata,
)

SINGLE = FacilityRoots.single(Path("/repo/data/facility"))
MOUNTED = FacilityRoots(
    machine=Path("/repo/machine/data/facility"), content=Path("/repo/data/facility")
)


def test_the_content_directories_are_exactly_the_three_deployment_ones() -> None:
    assert DEPLOYMENT_CONTENT_DIRS == ("knowledge", "ariel", "benchmarks")


def test_single_names_one_tree_in_both_roles() -> None:
    roots = FacilityRoots.single(Path("/x/facility"))
    assert roots.machine == roots.content == Path("/x/facility")
    assert not roots.mounted


def test_distinct_roots_are_mounted() -> None:
    assert MOUNTED.mounted


def test_roots_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        MOUNTED.machine = Path("/elsewhere")  # type: ignore[misc]


@pytest.mark.parametrize(
    "rel",
    [".git", ".git/config", ".git/objects/ab/cd", "imported/x/.git", "a/b/.git/HEAD"],
)
def test_git_metadata_at_any_depth_is_vcs_metadata(rel: str) -> None:
    assert is_vcs_metadata(PurePosixPath(rel))


@pytest.mark.parametrize(
    "rel",
    [
        ".gitignore",
        ".github",
        ".github/workflows/ci.yml",
        "knowledge/.gitkeep",
        "records.yaml",
        "git/x",
        "a.git",
        "imported/x/.gitattributes",
    ],
)
def test_lookalikes_are_not_vcs_metadata(rel: str) -> None:
    assert not is_vcs_metadata(PurePosixPath(rel))


def test_vcs_metadata_accepts_any_pure_path() -> None:
    assert is_vcs_metadata(PurePath(".git") / "config")


@pytest.mark.parametrize("roots", [SINGLE, MOUNTED])
def test_vcs_metadata_is_never_a_machine_entry(roots: FacilityRoots) -> None:
    assert not is_machine_entry(roots, PurePosixPath(".git"))
    assert not is_machine_entry(roots, PurePosixPath("imported/x/.git/HEAD"))


@pytest.mark.parametrize("name", DEPLOYMENT_CONTENT_DIRS)
def test_a_single_tree_keeps_its_content_directories(name: str) -> None:
    assert is_machine_entry(SINGLE, PurePosixPath(name))
    assert is_machine_entry(SINGLE, PurePosixPath(name) / "a.yaml")


@pytest.mark.parametrize("name", DEPLOYMENT_CONTENT_DIRS)
def test_a_mounted_machine_root_skips_top_level_content_directories(name: str) -> None:
    assert not is_machine_entry(MOUNTED, PurePosixPath(name))
    assert not is_machine_entry(MOUNTED, PurePosixPath(name) / "sub" / "a.yaml")


def test_a_nested_content_name_stays_a_machine_entry_when_mounted() -> None:
    assert is_machine_entry(MOUNTED, PurePosixPath("imported/x/knowledge/a.yaml"))


@pytest.mark.parametrize(
    "rel",
    [
        "identity.yaml",
        "records.yaml",
        "scenarios/warm.yaml",
        "knowledge.yaml",
        "ariel-notes/a.md",
        ".gitignore",
    ],
)
def test_ordinary_machine_files_are_machine_entries(rel: str) -> None:
    assert is_machine_entry(MOUNTED, PurePosixPath(rel))
    assert is_machine_entry(SINGLE, PurePosixPath(rel))
