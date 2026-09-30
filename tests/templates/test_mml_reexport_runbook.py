"""The runbook for re-running the MATLAB Middle Layer export on a MATLAB host.

The shipped README carries one section that tells a facility, or the owner of the
committed SPEAR3 and NSLS-II exports, how a fresh export is produced and what to
do with it. The two fixture READMEs point at that section rather than at a
private note.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MML_DIR = REPO_ROOT / "src" / "osprey" / "templates" / "apps" / "control_assistant" / "data" / "mml"
README = MML_DIR / "README.md"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"
FIXTURE_READMES = (FIXTURES / "spear3" / "README.md", FIXTURES / "nsls2" / "README.md")

SECTION_TITLE = "Re-running the export on a MATLAB host"
SECTION_ANCHOR = "re-running-the-export-on-a-matlab-host"

#: Every file one ``mml_export`` 2.1.0 run writes, by suffix.
EXPORT_SUFFIXES = (
    ".ao.json",
    ".ad.json",
    ".va.json",
    ".response.json",
    ".lattice.mat",
    ".model.json",
)


def _section(text: str, title: str) -> str:
    """The body of the ``##`` section named *title*, up to the next ``##``."""
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"no '## {title}' section"
    return match.group(1)


@pytest.fixture(scope="module")
def runbook() -> str:
    return _section(README.read_text(encoding="utf-8"), SECTION_TITLE)


def test_section_exists() -> None:
    assert f"## {SECTION_TITLE}\n" in README.read_text(encoding="utf-8")


def test_runbook_runs_one_matlab_per_sub_machine_with_setpathmml_first(runbook: str) -> None:
    assert "one MATLAB" in runbook
    assert "setpathmml" in runbook
    assert "before anything else" in runbook
    assert "switch2sim" in runbook
    assert "mml_export(" in runbook


def test_runbook_compiles_the_bundled_at(runbook: str) -> None:
    assert "atmexall" in runbook
    assert "simulators/at2.0" in runbook


def test_runbook_names_the_two_spear3_symlinks(runbook: str) -> None:
    assert "machine/SPEAR3 -> Spear3" in runbook
    assert "SPEAR3physdata.mat -> Spear3physdata.mat" in runbook


def test_runbook_names_the_labca_warning_and_simulator_mode(runbook: str) -> None:
    assert "LabCA" in runbook
    assert "simulator mode" in runbook


def test_runbook_names_the_six_files(runbook: str) -> None:
    assert "six files" in runbook
    for suffix in EXPORT_SUFFIXES:
        assert suffix in runbook, suffix


def test_runbook_states_the_owner_step(runbook: str) -> None:
    assert "all six files" in runbook
    for key in ("_export.exporter", "timestamp", "matlab"):
        assert f"`{key}`" in runbook, key
    assert "file counts" in runbook


@pytest.mark.parametrize("readme", FIXTURE_READMES, ids=lambda p: p.parent.name)
def test_fixture_readme_points_at_the_runbook(readme: Path) -> None:
    text = readme.read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)#]+README\.md)#" + SECTION_ANCHOR + r"\)", text)
    assert links, f"{readme} does not link the runbook section"
    for link in links:
        assert (readme.parent / link).resolve() == README.resolve(), link


def test_no_fixture_readme_mentions_a_session_path() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in FIXTURES.rglob("README.md")
        if ".claude/" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
