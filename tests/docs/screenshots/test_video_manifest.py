"""Unit tests for the demo-video manifest committed next to the posters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from docs.screenshots import video_manifest as vm


def _file(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _entry(tmp_path: Path, theme: str = "dark") -> dict:
    return vm.entry(
        osprey_version="2026.9.0",
        recorded_at=1790500000.0,
        duration_s=58.24,
        real_session_s=307.0,
        speed=16.0,
        mp4=_file(tmp_path, f"osprey-demo-{theme}.mp4", b"mp4-" + theme.encode()),
        poster=_file(tmp_path, f"osprey-demo-{theme}-poster.jpg", b"jpg-" + theme.encode()),
    )


def test_an_entry_records_the_take_and_both_files_by_name_and_hash(tmp_path) -> None:
    e = _entry(tmp_path)
    assert e["osprey_version"] == "2026.9.0"
    assert e["recorded_at"] == "2026-09-27T09:06:40+00:00"
    assert (e["duration_s"], e["real_session_s"], e["speed"]) == (58.24, 307.0, 16.0)
    assert e["mp4"] == {
        "name": "osprey-demo-dark.mp4",
        "sha256": hashlib.sha256(b"mp4-dark").hexdigest(),
    }
    assert e["poster"]["name"] == "osprey-demo-dark-poster.jpg"


def test_a_new_take_is_not_uploaded_until_the_release_is_named(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    vm.set_release(path, "docs-media-v2026.9.0")
    vm.put_theme(path, "dark", _entry(tmp_path))
    data = json.loads(path.read_text())
    assert data["release"] is None
    assert set(data["themes"]) == {"dark"}
    vm.set_release(path, "docs-media-v2026.9.0")
    vm.put_theme(path, "light", _entry(tmp_path, "light"))
    assert vm.load(path)["release"] is None
    assert set(vm.load(path)["themes"]) == {"dark", "light"}


def test_a_missing_manifest_reads_as_empty(tmp_path) -> None:
    assert vm.load(tmp_path / "none.json") == {"release": None, "themes": {}}


@pytest.mark.parametrize("field", vm.ENTRY_FIELDS)
def test_every_missing_field_is_named(tmp_path, field) -> None:
    e = _entry(tmp_path)
    del e[field]
    assert vm.missing_fields(e) == [field]


def test_a_file_without_its_hash_is_a_missing_field(tmp_path) -> None:
    e = _entry(tmp_path)
    del e["poster"]["sha256"]
    assert vm.missing_fields(e) == ["poster.sha256"]


@pytest.mark.parametrize(
    "version", ["2026.9.0", "2026.10.12", "2026.9.0a1", "2026.9.0b4", "2026.9.0rc2"]
)
def test_the_release_is_named_for_the_osprey_version(version) -> None:
    assert vm.release_name(version) == f"docs-media-v{version}"


@pytest.mark.parametrize(
    "version",
    [
        "",
        "v2026.9.0",
        "2026.9",
        "2026.9.0b",
        "2026.9.0beta4",
        "2026.9.0b4.post1",
        "2026.9.0.dev3",
        "latest",
    ],
)
def test_a_malformed_version_is_refused(version) -> None:
    with pytest.raises(ValueError):
        vm.release_name(version)


def test_sha256_streams_the_file(tmp_path) -> None:
    path = _file(tmp_path, "x.bin", b"abc" * 100_000)
    assert vm.sha256(path) == hashlib.sha256(b"abc" * 100_000).hexdigest()
