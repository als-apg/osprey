"""The panel tests' checked-in facility document is what the build writes.

``packages/osprey-panels/tests/osprey_panels_tests/fixture/facility.json``
stands in for building the fixture tree (``panel_fixture/``) in every panel
session. These tests rebuild it from the tree with the description's own checks
only and compare byte for byte, so a change to the tree or to the build that
leaves the document behind fails here, naming the regeneration command. The
rebuild needs neither the panels nor the framework, so it is a package test; the
document stays beside the panel tests that read it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from osprey_facility_tests import rebuild_panel_fixture as script
from osprey_facility_tests._harness import REPO_ROOT

SCRIPT = Path(script.__file__).resolve()
COMMAND = (
    "uv run python packages/osprey-facility/tests/osprey_facility_tests/rebuild_panel_fixture.py"
)
FIXTURE_DOCUMENT = (
    REPO_ROOT / "packages" / "osprey-panels" / "tests" / "osprey_panels_tests" / "fixture"
) / "facility.json"


def test_the_command_names_the_script_where_it_is() -> None:
    assert script.COMMAND == COMMAND
    assert (REPO_ROOT / COMMAND.split()[-1]).resolve() == SCRIPT
    assert script.FIXTURE_DOCUMENT == FIXTURE_DOCUMENT
    assert (script.FIXTURE_TREE / "documents.yaml").is_file()


def test_the_checked_in_document_is_the_fixture_tree_built_byte_for_byte() -> None:
    assert FIXTURE_DOCUMENT.is_file(), f"{FIXTURE_DOCUMENT} is missing: run {COMMAND}"
    assert FIXTURE_DOCUMENT.read_bytes() == script.rebuilt(), (
        f"{FIXTURE_DOCUMENT.relative_to(REPO_ROOT)} is stale: run {COMMAND}"
    )


def test_a_stale_document_fails_the_check_naming_the_command_and_passes_once_regenerated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = tmp_path / "facility.json"
    stale.write_bytes(FIXTURE_DOCUMENT.read_bytes().replace(b'"XR01:QF1"', b'"XR01:QF9"', 1))

    assert script.main(["--check", "--path", str(stale)]) == 1
    assert f"is stale; run {COMMAND}" in capsys.readouterr().out

    assert script.main(["--path", str(stale)]) == 0
    assert stale.read_bytes() == script.rebuilt()
    assert script.main(["--check", "--path", str(stale)]) == 0


def test_a_missing_document_fails_the_check_naming_the_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert script.main(["--check", "--path", str(tmp_path / "facility.json")]) == 1
    assert f"is missing; run {COMMAND}" in capsys.readouterr().out


def test_the_script_runs_as_the_command_names_it() -> None:
    result = subprocess.run(
        [sys.executable, COMMAND.split()[-1], "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
