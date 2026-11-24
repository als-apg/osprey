"""One file of a view is written once for the renders that agree on what it follows from.

A view is shared whole only between renders that agree on every config key its
writer reads. :func:`~osprey.facility.views.write_once` shares a single file
between renders that differ in those keys and agree on the ones the file itself
follows from.
"""

from __future__ import annotations

from pathlib import Path

from osprey.facility.views import ViewInputs, write_once
from osprey.facility.views.outputs import WrittenViews
from osprey_facility.records import FacilityRecords


def _inputs(tmp_path: Path, written: WrittenViews | None) -> ViewInputs:
    return ViewInputs(
        records=FacilityRecords({"channels": []}),
        rendered_config={},
        facility_dir=tmp_path,
        served=[],
        written=written,
    )


class _Writer:
    """Writes ``data`` and counts the times it was asked to."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.calls = 0

    def __call__(self, target: Path) -> None:
        self.calls += 1
        target.write_bytes(self.data)


def test_a_second_render_with_the_same_key_gets_a_copy(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, {})
    write = _Writer(b"bytes")

    write_once(inputs, tmp_path / "first", ("name", 1), write)
    write_once(inputs, tmp_path / "second", ("name", 1), write)

    assert write.calls == 1
    assert (tmp_path / "second").read_bytes() == b"bytes"
    assert (tmp_path / "first").stat().st_ino != (tmp_path / "second").stat().st_ino


def test_another_key_is_written_itself(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, {})
    write_once(inputs, tmp_path / "first", ("name", 1), _Writer(b"one"))
    other = _Writer(b"two")

    write_once(inputs, tmp_path / "second", ("name", 2), other)

    assert other.calls == 1
    assert (tmp_path / "second").read_bytes() == b"two"


def test_a_file_changed_since_it_was_written_is_not_copied(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, {})
    write = _Writer(b"bytes")
    write_once(inputs, tmp_path / "first", "name", write)
    (tmp_path / "first").write_bytes(b"edited")

    write_once(inputs, tmp_path / "second", "name", write)
    write_once(inputs, tmp_path / "third", "name", write)

    assert write.calls == 2
    assert (tmp_path / "second").read_bytes() == b"bytes"
    assert (tmp_path / "third").read_bytes() == b"bytes"


def test_a_file_removed_since_it_was_written_is_written_again(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, {})
    write = _Writer(b"bytes")
    write_once(inputs, tmp_path / "first", "name", write)
    (tmp_path / "first").unlink()

    write_once(inputs, tmp_path / "second", "name", write)

    assert write.calls == 2
    assert (tmp_path / "second").read_bytes() == b"bytes"


def test_the_same_target_is_written_again(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, {})
    write_once(inputs, tmp_path / "file", "name", _Writer(b"old"))

    write_once(inputs, tmp_path / "file", "name", _Writer(b"new"))

    assert (tmp_path / "file").read_bytes() == b"new"


def test_a_render_outside_a_build_always_writes(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, None)
    write = _Writer(b"bytes")

    write_once(inputs, tmp_path / "first", "name", write)
    write_once(inputs, tmp_path / "second", "name", write)

    assert write.calls == 2
