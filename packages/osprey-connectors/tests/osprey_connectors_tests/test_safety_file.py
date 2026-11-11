"""``safety_file.atomic_replace`` replaces a file whole, through a name nobody can predict."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors import safety_file
from osprey_connectors.safety_file import atomic_replace


def test_a_link_planted_at_every_predictable_name_is_never_followed(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"old\n")
    victim = tmp_path / "victim"
    victim.write_text("untouched", encoding="utf-8")
    planted = [
        path.with_name(name)
        for name in (
            f"{path.name}.tmp",
            f".{path.name}.tmp",
            f".{path.name}.{os.getpid()}.tmp",
            f"{path.name}.{os.getpid()}.tmp",
        )
    ]
    for link in planted:
        link.symlink_to(victim)

    atomic_replace(path, b"new\n", mode=0o664)

    assert path.read_bytes() == b"new\n"
    assert not path.is_symlink()
    assert victim.read_text(encoding="utf-8") == "untouched"
    assert all(link.is_symlink() for link in planted)
    leftovers = sorted(p for p in path.parent.iterdir() if p.name.endswith(".tmp"))
    assert leftovers == sorted(planted)


def test_a_failing_replace_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "profile"
    folder.mkdir()
    path = folder / ".env"
    path.write_bytes(b"old")

    def refuse(*_args: Any) -> None:
        raise OSError("replace refused")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="replace refused"):
        atomic_replace(path, b"new", mode=0o644)

    assert path.read_bytes() == b"old"
    assert [p.name for p in folder.iterdir()] == [".env"]


@pytest.mark.parametrize("mode", [0o664, 0o644, 0o600])
def test_the_result_has_the_mode_asked_for_whatever_the_umask(tmp_path: Path, mode: int) -> None:
    path = tmp_path / "file"
    previous = os.umask(0o077)
    try:
        atomic_replace(path, b"data", mode=mode)
    finally:
        os.umask(previous)

    assert path.read_bytes() == b"data"
    assert path.stat().st_mode & 0o777 == mode


def _refuse_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the disk refuse to flush a regular file, as some mounts do."""
    real_fsync = os.fsync

    def refuse(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("flush refused")
        real_fsync(fd)

    monkeypatch.setattr(safety_file.os, "fsync", refuse)


def test_a_refused_flush_fails_the_replace_when_sync_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"old")
    _refuse_fsync(monkeypatch)

    with pytest.raises(OSError, match="flush refused"):
        atomic_replace(path, b"new", mode=0o640, sync="required")

    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["file"]


def test_a_refused_flush_still_replaces_the_file_when_sync_is_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"old")
    _refuse_fsync(monkeypatch)

    atomic_replace(path, b"new", mode=0o640, sync="best_effort")

    assert path.read_bytes() == b"new"
    assert path.stat().st_mode & 0o777 == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["file"]


# -- ensure_shared_dir --------------------------------------------------------


@pytest.fixture
def owner_only_umask():
    """Umask 077, the one under which a directory left at the umask shuts the group out."""
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.usefixtures("owner_only_umask")
def test_every_level_created_is_exactly_the_shared_dir_mode(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    os.chmod(root, 0o2750)
    leaf = root / "tree" / "alice"

    safety_file.ensure_shared_dir(leaf)

    assert stat.S_IMODE(leaf.stat().st_mode) == safety_file.SHARED_DIR_MODE == 0o2770
    assert stat.S_IMODE(leaf.parent.stat().st_mode) == 0o2770
    assert stat.S_IMODE(root.stat().st_mode) == 0o2750, "an existing level is not this writer's"


def test_a_created_level_takes_the_group_of_the_location(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    os.chmod(root, 0o2770)

    safety_file.ensure_shared_dir(root / "alice")

    assert (root / "alice").stat().st_gid == root.stat().st_gid


def test_an_existing_directory_keeps_the_mode_it_was_provisioned_with(tmp_path: Path) -> None:
    leaf = tmp_path / "alice"
    leaf.mkdir()
    os.chmod(leaf, 0o750)

    safety_file.ensure_shared_dir(leaf)

    assert stat.S_IMODE(leaf.stat().st_mode) == 0o750


def test_a_refused_mode_raises_unless_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: Any) -> None:
        raise PermissionError("chmod refused")

    monkeypatch.setattr(os, "fchmod", refuse)
    with pytest.raises(PermissionError, match="chmod refused"):
        safety_file.ensure_shared_dir(tmp_path / "a")

    safety_file.ensure_shared_dir(tmp_path / "b", modes="best_effort")
    assert (tmp_path / "b").is_dir()


def _another_writer_makes_first(
    monkeypatch: pytest.MonkeyPatch, contested: Path, mode: int = 0o750
) -> None:
    """Have *contested* appear, made by someone else, just before this call makes it."""
    real_mkdir = os.mkdir

    def racing_mkdir(path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path) == contested:
            real_mkdir(path)
            os.chmod(path, mode)
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", racing_mkdir)


@pytest.mark.usefixtures("owner_only_umask")
def test_a_leaf_another_writer_made_first_still_leaves_the_levels_above_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaf = tmp_path / "tree" / "alice"
    _another_writer_makes_first(monkeypatch, leaf)

    safety_file.ensure_shared_dir(leaf)

    assert stat.S_IMODE(leaf.parent.stat().st_mode) == 0o2770, "this writer made it"
    assert stat.S_IMODE(leaf.stat().st_mode) == 0o750, "the other writer's, as it made it"


@pytest.mark.usefixtures("owner_only_umask")
def test_an_intermediate_another_writer_made_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under another uid a mode set on that level would raise EPERM; it is never tried.
    middle = tmp_path / "tree"
    leaf = middle / "alice"
    _another_writer_makes_first(monkeypatch, middle)

    safety_file.ensure_shared_dir(leaf)

    assert stat.S_IMODE(middle.stat().st_mode) == 0o750
    assert stat.S_IMODE(leaf.stat().st_mode) == 0o2770


@pytest.mark.parametrize("modes", ["required", "best_effort"])
def test_a_level_swapped_for_a_link_is_never_chmodded_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modes: str
) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    os.chmod(victim, 0o700)
    leaf = tmp_path / "alice"
    real_mkdir = os.mkdir

    def swapping_mkdir(path: Any, *args: Any, **kwargs: Any) -> None:
        real_mkdir(path, *args, **kwargs)
        os.rmdir(path)
        os.symlink(victim, path)

    monkeypatch.setattr(os, "mkdir", swapping_mkdir)
    if modes == "required":
        with pytest.raises(OSError):
            safety_file.ensure_shared_dir(leaf, modes=modes)
    else:
        safety_file.ensure_shared_dir(leaf, modes=modes)

    assert stat.S_IMODE(victim.stat().st_mode) == 0o700


def test_the_shared_file_modes_are_named() -> None:
    assert safety_file.SHARED_FILE_MODE == 0o640
    assert safety_file.SHARED_LOCK_MODE == 0o660
