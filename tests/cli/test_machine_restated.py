"""Tests for the ``machine-restated`` stop of a mounted deployment.

While a profile mounts a machine root (``machine:``), the deployment's own
facility tree (``<data>/facility``) holds only the deployment's content
directories. Any other entry restates a machine fact, so validate and build
stop on it with a message naming the deployment file, the machine file it
restates (or the machine root when there is none) and the remedy.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from osprey.cli.profile_conventions import machine_restated_violation
from osprey.facility.errors import FacilityBuildError
from osprey.facility.roots import DEPLOYMENT_CONTENT_DIRS, FacilityRoots


def _write(path: Path, text: str = "x: 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def roots(tmp_path: Path) -> FacilityRoots:
    """A mounted deployment: ``machine/`` beside ``data/facility``."""
    machine = tmp_path / "machine"
    _write(machine / "identity.yaml", "code: m\nname: M\n")
    _write(machine / "records" / "x.yaml")
    content = tmp_path / "data" / "facility"
    content.mkdir(parents=True)
    return FacilityRoots(machine=machine, content=content)


def test_clean_content_root_passes(roots: FacilityRoots) -> None:
    assert machine_restated_violation(roots) is None


def test_restated_record_stops_naming_both_files(roots: FacilityRoots) -> None:
    _write(roots.content / "records" / "x.yaml")

    error = machine_restated_violation(roots)

    assert isinstance(error, FacilityBuildError)
    assert error.kind == "machine-restated"
    message = error.format_message()
    assert "data/facility/records/x.yaml" in message
    assert "machine/records/x.yaml" in message
    assert "fix:" in message
    assert error.sources == ("data/facility/records/x.yaml",)


def test_stray_limits_stops_naming_the_machine_root(roots: FacilityRoots) -> None:
    _write(roots.content / "limits.yaml", "limits: []\n")

    error = machine_restated_violation(roots)

    assert isinstance(error, FacilityBuildError)
    assert error.kind == "machine-restated"
    message = error.format_message()
    assert "data/facility/limits.yaml" in message
    # The machine has no limits.yaml, so the message names its root instead.
    assert "machine/limits.yaml" not in message
    assert "machine/" in message


def test_stray_limits_names_the_machine_copy_when_it_exists(roots: FacilityRoots) -> None:
    _write(roots.machine / "limits.yaml", "limits: []\n")
    _write(roots.content / "limits.yaml", "limits: []\n")

    error = machine_restated_violation(roots)

    assert error is not None
    assert "machine/limits.yaml" in error.format_message()


def test_content_directories_and_dot_entries_pass(roots: FacilityRoots) -> None:
    for name in DEPLOYMENT_CONTENT_DIRS:
        _write(roots.content / name / "a.yaml")
    _write(roots.content / "knowledge" / "notes.md", "# notes\n")
    _write(roots.content / "ariel" / "vocabulary.yaml")
    _write(roots.content / "benchmarks" / "queries.json", "[]\n")
    _write(roots.content / ".DS_Store", "")
    _write(roots.content / ".hidden" / "records" / "x.yaml")

    assert set(DEPLOYMENT_CONTENT_DIRS) == {"knowledge", "ariel", "benchmarks"}
    assert machine_restated_violation(roots) is None


def test_empty_restated_directory_passes(roots: FacilityRoots) -> None:
    (roots.content / "records").mkdir()
    _write(roots.content / "records" / ".DS_Store", "")

    assert machine_restated_violation(roots) is None


def test_first_offender_is_deterministic(roots: FacilityRoots) -> None:
    _write(roots.content / "scenarios" / "b.yaml")
    _write(roots.content / "classes" / "a.yaml")
    _write(roots.content / "limits.yaml")

    error = machine_restated_violation(roots)

    assert error is not None
    assert error.sources == ("data/facility/classes/a.yaml",)


def test_stated_probe_channel_is_not_its_concern(roots: FacilityRoots) -> None:
    """D4: the probe and freshness channels, ARIEL vocabulary and benchmark
    dataset are deployment settings, so the stop reads only the content root's
    files and never a stated setting."""
    assert list(inspect.signature(machine_restated_violation).parameters) == ["roots"]
    _write(roots.content / "ariel" / "vocabulary.yaml")
    _write(roots.content / "benchmarks" / "queries.json", "[]\n")

    assert machine_restated_violation(roots) is None


def test_unmounted_deployment_never_stops(tmp_path: Path) -> None:
    facility = tmp_path / "data" / "facility"
    _write(facility / "identity.yaml", "code: m\nname: M\n")
    _write(facility / "records" / "x.yaml")
    _write(facility / "limits.yaml", "limits: []\n")

    assert machine_restated_violation(FacilityRoots.single(facility)) is None


def test_missing_content_root_passes(tmp_path: Path) -> None:
    machine = tmp_path / "machine"
    _write(machine / "identity.yaml", "code: m\nname: M\n")
    roots = FacilityRoots(machine=machine, content=tmp_path / "data" / "facility")

    assert machine_restated_violation(roots) is None
