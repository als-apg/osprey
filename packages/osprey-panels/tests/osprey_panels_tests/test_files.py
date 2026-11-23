"""``_files.atomic_replace`` replaces a file whole, through a name nobody can predict."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Any

import pytest

from osprey_panels import _files, emit
from osprey_panels._files import atomic_replace


def test_a_link_planted_at_every_predictable_name_is_never_followed(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
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

    atomic_replace(path, b"new\n", mode=0o644)

    assert path.read_bytes() == b"new\n"
    assert not path.is_symlink()
    assert victim.read_text(encoding="utf-8") == "untouched"
    assert all(link.is_symlink() for link in planted)
    leftovers = sorted(p for p in path.parent.iterdir() if p.name.endswith(".tmp"))
    assert leftovers == sorted(planted)


def test_a_failing_replace_leaves_no_temporary_file_and_the_old_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "screen.json"
    path.write_bytes(b"old")

    def refuse(*_args: Any) -> None:
        raise OSError("replace refused")

    monkeypatch.setattr(_files.os, "replace", refuse)
    with pytest.raises(OSError, match="replace refused"):
        atomic_replace(path, b"new", mode=0o644)

    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["screen.json"]


def test_a_failing_write_leaves_no_temporary_file_and_the_old_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "screen.json"
    path.write_bytes(b"old")
    real_write = os.write
    calls = 0

    def write_once_then_fail(fd: int, data: Any) -> int:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("disk full")
        return real_write(fd, bytes(data[:2]))

    monkeypatch.setattr(_files.os, "write", write_once_then_fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_replace(path, b"new bytes", mode=0o644)

    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["screen.json"]


def test_a_refused_flush_fails_the_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"old")
    real_fsync = os.fsync

    def refuse(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("flush refused")
        real_fsync(fd)

    monkeypatch.setattr(_files.os, "fsync", refuse)
    with pytest.raises(OSError, match="flush refused"):
        atomic_replace(path, b"new", mode=0o644)

    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["file"]


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


def test_a_short_write_is_completed_not_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file"
    real_write = os.write

    def one_byte(fd: int, data: Any) -> int:
        return real_write(fd, bytes(data[:1]))

    monkeypatch.setattr(_files.os, "write", one_byte)
    atomic_replace(path, b"whole content", mode=0o644)

    assert path.read_bytes() == b"whole content"


@pytest.mark.parametrize(
    ("name", "pattern"),
    [
        ("rollup-x" + emit.SCREEN_SUFFIX, emit._TEMPORARY),
        ("index.json", emit._INDEX_TEMPORARY),
    ],
)
def test_the_temporary_name_is_the_one_the_writer_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, pattern: Any
) -> None:
    """A run killed mid-write leaves a temporary that ``emit`` recognises as its own."""
    seen: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def recording(*args: Any, **kwargs: Any) -> tuple[int, str]:
        fd, temporary = real_mkstemp(*args, **kwargs)
        seen.append(Path(temporary).name)
        return fd, temporary

    monkeypatch.setattr(_files.tempfile, "mkstemp", recording)
    atomic_replace(tmp_path / name, b"{}", mode=0o644)

    assert len(seen) == 1
    assert pattern.fullmatch(seen[0]), seen[0]


def test_the_writer_sets_its_file_mode_through_the_vendored_replace(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "index.json"
    previous = os.umask(0o077)
    try:
        emit._atomic(path, b"{}")
    finally:
        os.umask(previous)

    assert path.read_bytes() == b"{}"
    assert path.stat().st_mode & 0o777 == emit._FILE_MODE
