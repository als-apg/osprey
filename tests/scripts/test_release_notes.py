"""The release body is the tag's whole CHANGELOG section, wherever it sits."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "release_notes", Path(__file__).resolve().parents[2] / "scripts" / "ci" / "release_notes.py"
)
assert _spec and _spec.loader
notes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(notes)

CHANGELOG = """# Changelog

## [Unreleased]

## [2026.10.1] - 2026-10-30

### Fixed

- the newer fix

## [2026.10.0] - 2026-10-20

### Added

- the first bullet
- the last bullet

"""


def test_the_last_section_keeps_its_last_bullet():
    assert notes.section(CHANGELOG, "2026.10.0") == [
        "## [2026.10.0] - 2026-10-20",
        "",
        "### Added",
        "",
        "- the first bullet",
        "- the last bullet",
    ]


def test_a_middle_section_stops_before_the_next_heading():
    assert notes.section(CHANGELOG, "2026.10.1")[-1] == "- the newer fix"


@pytest.mark.parametrize("version", ["2026x10x0", "2026.10", "2026.10.0b1", "026.10.0"])
def test_the_heading_is_matched_literally(version):
    """A dot is not a wildcard, and a prefix of a version is not the version."""
    assert notes.section(CHANGELOG, version) == []


def test_a_missing_section_points_at_the_file(tmp_path, capsys):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    assert notes.main([str(changelog), "2027.1.0"]) == 0
    assert capsys.readouterr().out == f"See {changelog} for details.\n"


def test_main_prints_the_section(tmp_path, capsys):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    assert notes.main([str(changelog), "2026.10.0"]) == 0
    assert capsys.readouterr().out.endswith("- the last bullet\n")
