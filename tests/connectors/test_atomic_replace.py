"""``atomic_replace`` replaces a file whole, through a name nobody can predict."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.dotenv import atomic_replace


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
