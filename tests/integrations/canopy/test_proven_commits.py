"""The CANOPY this OSPREY is proven against is declared once, and judged in one place.

``osprey.integrations.canopy.proven`` names one CANOPY commit and one
PyDMConverter commit, reads which commits are installed, and words the one line
every reader prints. The contract in this directory proves something about that
pair only when the installed distributions are those commits, so one
``canopy``-marked test asks the judgement; everything else here runs without
CANOPY, against fake distributions.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests import _canopy_contract as contract

from osprey.integrations.canopy import PROVEN_COMMITS, proven

OTHER_COMMIT = contract.OTHER_COMMIT
CANOPY_PIN = PROVEN_COMMITS["canopy"]
PYDM_PIN = PROVEN_COMMITS["PyDMConverter"]
PROVEN_MODULE = Path(proven.__file__)


class _FakeDistribution:
    def __init__(self, version: str, direct_url: object = None):
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        if filename != "direct_url.json" or self._direct_url is None:
            return None
        if isinstance(self._direct_url, str):
            return self._direct_url
        return json.dumps(self._direct_url)


def _install(monkeypatch: pytest.MonkeyPatch, **distributions: _FakeDistribution) -> None:
    """Make *distributions* the only ones ``importlib.metadata`` finds."""

    def distribution(name: str) -> _FakeDistribution:
        try:
            return distributions[name]
        except KeyError:
            raise importlib.metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)


def _git_install(commit: str) -> dict:
    return {
        "url": "https://example.org/canopy.git",
        "vcs_info": {"vcs": "git", "commit_id": commit},
    }


def _at(commit: str) -> _FakeDistribution:
    """A git install recording *commit*, whose version label names another."""
    return _FakeDistribution("0.1.dev476+ge5e62dad7", _git_install(commit))


EDITABLE_AT_THE_PIN = _FakeDistribution(
    f"0.1.dev476+g{CANOPY_PIN[:9]}",
    {"url": "file:///src/canopy", "dir_info": {"editable": True}},
)


# --- the declaration --------------------------------------------------------------------


def test_the_proven_commits_name_canopy_and_its_converter_and_nothing_else() -> None:
    assert sorted(PROVEN_COMMITS) == ["PyDMConverter", "canopy"]


def test_the_package_re_exports_the_modules_declaration() -> None:
    assert PROVEN_COMMITS is proven.PROVEN_COMMITS


@pytest.mark.parametrize("distribution", ["canopy", "PyDMConverter"])
def test_a_proven_commit_is_a_full_commit_id(distribution: str) -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", PROVEN_COMMITS[distribution])


# --- the files copied from CANOPY -------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
CANOPY_COPIES = Path(__file__).parent / "fixtures" / "canopy-copies.json"

#: Every CANOPY file this repository keeps a byte-identical copy of, by OSPREY path.
COPIED_FROM_CANOPY = {
    "tests/integrations/canopy/fixtures/ir-map.json": "tests/canopy_mcp/fixtures/ir-map.json",
    "tests/integrations/canopy/fixtures/grammar-vectors.json": (
        "tests/canopy_mcp/fixtures/grammar-vectors.json"
    ),
    "tests/integrations/canopy/fixtures/listing-vectors.json": (
        "tests/canopy_mcp/fixtures/listing-vectors.json"
    ),
    "tests/integrations/canopy/fixtures/match-vectors.json": (
        "tests/canopy_mcp/fixtures/collections/match-vectors.json"
    ),
    "src/osprey/templates/claude_code/claude/skills/canopy-panel-builder/SKILL.md": (
        "src/canopy_mcp/skill/SKILL.md"
    ),
    "src/osprey/templates/claude_code/claude/skills/canopy-operator/SKILL.md": (
        "src/canopy_mcp/skill/operator/SKILL.md"
    ),
}


def _copies_record() -> dict:
    return json.loads(CANOPY_COPIES.read_text())


def _copies_name_pin() -> bool:
    """Whether the copies were taken at the CANOPY commit this OSPREY is proven against."""
    return _copies_record()["canopy"] == PROVEN_COMMITS["canopy"]


def _git_blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _copy_matches(entry: dict, root: Path) -> bool:
    """Whether the OSPREY-side copy under *root* is the blob the record names."""
    return _git_blob_id(root / entry["osprey_path"]) == entry["blob"]


def test_the_canopy_copies_were_taken_at_the_proven_canopy() -> None:
    record = _copies_record()
    assert sorted(record) == ["canopy", "copies"]
    assert re.fullmatch(r"[0-9a-f]{40}", record["canopy"])
    assert _copies_name_pin()


def test_the_canopy_copies_were_taken_at_the_proven_canopy__mutation_moves_the_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(proven.PROVEN_COMMITS, "canopy", OTHER_COMMIT)
    assert not _copies_name_pin()


def test_the_record_names_every_copy_by_both_paths_and_its_blob() -> None:
    copies = _copies_record()["copies"]
    assert all(sorted(entry) == ["blob", "canopy_path", "osprey_path"] for entry in copies)
    assert all(re.fullmatch(r"[0-9a-f]{40}", entry["blob"]) for entry in copies)
    assert {entry["osprey_path"]: entry["canopy_path"] for entry in copies} == COPIED_FROM_CANOPY


@pytest.mark.parametrize("osprey_path", sorted(COPIED_FROM_CANOPY))
def test_each_copy_is_the_blob_the_record_names(osprey_path: str) -> None:
    (entry,) = [e for e in _copies_record()["copies"] if e["osprey_path"] == osprey_path]
    assert _copy_matches(entry, REPO_ROOT)


@pytest.mark.parametrize("osprey_path", sorted(COPIED_FROM_CANOPY))
def test_each_copy_is_the_blob_the_record_names__mutation_edits_the_copy(
    osprey_path: str, tmp_path: Path
) -> None:
    (entry,) = [e for e in _copies_record()["copies"] if e["osprey_path"] == osprey_path]
    edited = tmp_path / osprey_path
    edited.parent.mkdir(parents=True)
    edited.write_bytes((REPO_ROOT / osprey_path).read_bytes() + b"\n")
    assert not _copy_matches(entry, tmp_path)


# --- reading an installed commit --------------------------------------------------------


def test_a_git_install_names_its_recorded_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, canopy=_FakeDistribution("0.4.0", _git_install(OTHER_COMMIT)))
    assert proven.installed("canopy") == proven.Held(commit=OTHER_COMMIT)


def test_the_recorded_commit_wins_over_the_version_label(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDistribution(f"0.1.dev476+g{CANOPY_PIN[:9]}", _git_install(OTHER_COMMIT))
    _install(monkeypatch, canopy=fake)
    assert proven.installed("canopy").commit == OTHER_COMMIT


def test_an_editable_install_records_no_commit_whatever_its_label_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, canopy=EDITABLE_AT_THE_PIN)
    assert proven.installed("canopy") == proven.Held(reason=proven.EDITABLE)


@pytest.mark.parametrize(
    "direct_url",
    [
        None,
        {"url": "file:///src/canopy", "dir_info": {}},
        {"url": "file:///src/canopy", "dir_info": {"editable": False}},
        {"url": "file:///src/canopy", "vcs_info": {"vcs": "git"}},
        {"url": "file:///src/canopy", "vcs_info": {"commit_id": ""}},
        {"url": "file:///src/canopy", "vcs_info": {"commit_id": 7}},
        {"url": "file:///src/canopy", "vcs_info": "git"},
        ["not", "an", "object"],
        "{not json",
    ],
    ids=[
        "no-record",
        "path",
        "path-not-editable",
        "vcs-without-commit",
        "empty-commit",
        "commit-not-a-string",
        "vcs-not-an-object",
        "record-not-an-object",
        "malformed",
    ],
)
def test_an_install_without_a_recorded_commit_has_none_whatever_its_label_says(
    monkeypatch: pytest.MonkeyPatch, direct_url: object
) -> None:
    _install(monkeypatch, canopy=_FakeDistribution(f"0.1.dev476+g{CANOPY_PIN[:9]}", direct_url))
    assert proven.installed("canopy") == proven.Held(reason=proven.NO_RECORD)


def test_a_distribution_that_is_not_installed_has_no_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    assert proven.installed("canopy") == proven.Held(reason=proven.NOT_INSTALLED)


# --- comparing with the pin -------------------------------------------------------------


@pytest.mark.parametrize(
    ("commit", "expected"),
    [
        (CANOPY_PIN, True),
        (CANOPY_PIN[:9], False),
        (CANOPY_PIN[:1], False),
        ("", False),
        (OTHER_COMMIT, False),
        (CANOPY_PIN + "0", False),
        (PYDM_PIN, False),
    ],
    ids=["whole", "prefix", "one-hex", "empty", "other", "longer", "the-other-pin"],
)
def test_is_pin_is_whole_equality(commit: str, expected: bool) -> None:
    assert proven.is_pin("canopy", commit) is expected


# --- reading the version string an artefact carries -------------------------------------


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (f"0.1.dev476+g{CANOPY_PIN[:9]}", True),
        (f"0.1.dev476+g{CANOPY_PIN}", True),
        (f"0.1.dev476+g{CANOPY_PIN[:1]}", True),
        ("0.0.2.dev146+ge5e62dad7", False),
        (f"0.1.dev476+g{CANOPY_PIN[1:9]}", False),
        (f"0.1.dev476+g{CANOPY_PIN}0", False),
        (f"0.1.dev476+g{CANOPY_PIN[:9]}.d20261011", False),
        ("0.1.dev476+d20261011", False),
        ("0.1.dev476+gnothex", False),
        ("0.1.dev476+g", False),
        ("0.4.0", False),
        ("0.4.0+local", False),
    ],
)
def test_a_version_names_the_pin_by_a_clean_label_that_prefixes_it(
    version: str, expected: bool
) -> None:
    assert proven.version_names_pin("canopy", version) is expected


# --- the judgement ----------------------------------------------------------------------


def test_both_at_the_pin_is_proven(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, canopy=_at(CANOPY_PIN), PyDMConverter=_at(PYDM_PIN))
    judgement = proven.judge()
    assert judgement.state == proven.PROVEN
    assert [(r.distribution, r.at_pin) for r in judgement.readings] == [
        ("canopy", True),
        ("PyDMConverter", True),
    ]


def test_neither_installed_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)
    assert proven.judge().state == proven.ABSENT


def test_one_missing_is_unproven(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, canopy=_at(CANOPY_PIN))
    assert proven.judge().state == proven.UNPROVEN


@pytest.mark.parametrize("off", ["canopy", "PyDMConverter"])
def test_one_off_the_pin_is_unproven(monkeypatch: pytest.MonkeyPatch, off: str) -> None:
    installs = {"canopy": _at(CANOPY_PIN), "PyDMConverter": _at(PYDM_PIN)}
    installs[off] = _at(OTHER_COMMIT)
    _install(monkeypatch, **installs)
    judgement = proven.judge()
    assert judgement.state == proven.UNPROVEN
    assert {r.distribution: r.at_pin for r in judgement.readings}[off] is False


def test_an_editable_canopy_whose_label_names_the_pin_is_unproven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, canopy=EDITABLE_AT_THE_PIN, PyDMConverter=_at(PYDM_PIN))
    assert proven.judge().state == proven.UNPROVEN


@pytest.mark.canopy
def test_the_installed_canopy_is_the_proven_one() -> None:
    """The pins are what is installed."""
    assert proven.judge().state == "proven", proven.line(proven.judge(), None)


# --- the line ---------------------------------------------------------------------------

PROVEN_AGAINST = (
    f"this OSPREY is proven against canopy {CANOPY_PIN[:12]}, PyDMConverter {PYDM_PIN[:12]}"
)


#: The platform the venue tests stand in, so a line's text is the same on every host.
_VENUE = ("Linux", "x86_64")


@pytest.fixture
def on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proven.platform, "system", lambda: _VENUE[0])
    monkeypatch.setattr(proven.platform, "machine", lambda: _VENUE[1])


PAIR = f"canopy {CANOPY_PIN}, PyDMConverter {PYDM_PIN}"


@pytest.mark.usefixtures("on_linux")
@pytest.mark.parametrize(
    ("ran", "expected"),
    [
        (True, f"CANOPY contract proven against {PAIR} on Linux x86_64"),
        (False, f"CANOPY contract failed against {PAIR} on Linux x86_64"),
        (None, f"CANOPY at the proven commits: {PAIR}"),
    ],
    ids=["passed", "failed", "probe"],
)
def test_the_line_of_a_proven_install(
    monkeypatch: pytest.MonkeyPatch, ran: bool | None, expected: str
) -> None:
    _install(monkeypatch, canopy=_at(CANOPY_PIN), PyDMConverter=_at(PYDM_PIN))
    assert proven.line(proven.judge(), ran) == expected


def test_the_venue_is_this_interpreters_platform() -> None:
    import platform

    assert proven.venue() == f"{platform.system()} {platform.machine()}"


@pytest.mark.usefixtures("on_linux")
@pytest.mark.parametrize("passed", [True, False])
def test_the_attestation_is_the_line_of_a_proven_install(
    monkeypatch: pytest.MonkeyPatch, passed: bool
) -> None:
    """A caller that judged the installs proven elsewhere words the line from
    the pins alone, as the installed environment would."""
    _install(monkeypatch)
    attested = proven.attestation(passed)
    _install(monkeypatch, canopy=_at(CANOPY_PIN), PyDMConverter=_at(PYDM_PIN))
    assert attested == proven.line(proven.judge(), passed)


@pytest.mark.parametrize("ran", [True, False, None])
def test_the_line_of_an_unproven_install_names_what_is_held(
    monkeypatch: pytest.MonkeyPatch, ran: bool | None
) -> None:
    _install(monkeypatch, canopy=EDITABLE_AT_THE_PIN, PyDMConverter=_at(OTHER_COMMIT))
    assert proven.line(proven.judge(), ran) == (
        "CANOPY contract proves nothing here: canopy editable, records no commit, "
        f"PyDMConverter {OTHER_COMMIT[:12]}; {PROVEN_AGAINST}"
    )


def test_the_line_of_an_install_missing_one_names_it_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, canopy=_FakeDistribution("1.0", {"url": "file:///w", "dir_info": {}}))
    assert proven.line(proven.judge(), None) == (
        "CANOPY contract proves nothing here: canopy installed without a recorded commit, "
        f"PyDMConverter not installed; {PROVEN_AGAINST}"
    )


@pytest.mark.parametrize("ran", [True, False, None])
def test_the_line_of_an_absent_install(monkeypatch: pytest.MonkeyPatch, ran: bool | None) -> None:
    _install(monkeypatch)
    assert proven.line(proven.judge(), ran) == (
        f"CANOPY contract not run here: CANOPY is not installed; {PROVEN_AGAINST}"
    )


# --- the command line -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("installs", "code"),
    [
        ({"canopy": _at(CANOPY_PIN), "PyDMConverter": _at(PYDM_PIN)}, 0),
        ({"canopy": _at(OTHER_COMMIT), "PyDMConverter": _at(PYDM_PIN)}, 1),
        ({}, 2),
    ],
    ids=["proven", "unproven", "absent"],
)
def test_main_prints_the_probe_line_and_exits_by_the_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    installs: dict[str, _FakeDistribution],
    code: int,
) -> None:
    _install(monkeypatch, **installs)
    assert proven.main([]) == code
    assert capsys.readouterr().out == proven.line(proven.judge(), None) + "\n"


def test_main_pins_prints_both_commits(capsys: pytest.CaptureFixture[str]) -> None:
    assert proven.main(["--pins"]) == 0
    assert capsys.readouterr().out == f"canopy={CANOPY_PIN}\nPyDMConverter={PYDM_PIN}\n"


def test_the_module_runs_by_path_with_nothing_but_the_standard_library(tmp_path: Path) -> None:
    """CI reads the pins with the runner's own interpreter before anything is installed."""
    done = subprocess.run(
        [sys.executable, "-I", "-S", str(PROVEN_MODULE), "--pins"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == f"canopy={CANOPY_PIN}\nPyDMConverter={PYDM_PIN}\n"


def _run_in_a_bare_site(
    tmp_path: Path, module: str, installed: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    """Run ``python -S -m module`` over a site holding only this package's
    stdlib-only modules and distributions recording ``installed`` commits."""
    site = tmp_path / "site"
    package = site / "osprey/integrations/canopy"
    package.mkdir(parents=True)
    for init in ("osprey", "osprey/integrations"):
        (site / init / "__init__.py").write_text("", encoding="utf-8")
    for name in ("__init__.py", "proven.py", "__main__.py"):
        (package / name).write_bytes((PROVEN_MODULE.parent / name).read_bytes())
    for name, commit in installed.items():
        info = site / f"{name}-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
        )
        record = {"url": "file:///clone", "vcs_info": {"vcs": "git", "commit_id": commit}}
        (info / "direct_url.json").write_text(json.dumps(record), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-S", "-m", module, *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(site)},
        check=False,
    )


@pytest.mark.parametrize(
    ("installed", "code"),
    [
        ({"canopy": CANOPY_PIN, "PyDMConverter": PYDM_PIN}, 0),
        ({"canopy": OTHER_COMMIT, "PyDMConverter": PYDM_PIN}, 1),
        ({}, 2),
    ],
    ids=["proven", "unproven", "absent"],
)
def test_the_package_command_judges_with_a_clean_stderr(
    tmp_path: Path, installed: dict[str, str], code: int
) -> None:
    done = _run_in_a_bare_site(tmp_path, "osprey.integrations.canopy", installed)
    assert done.returncode == code, done.stderr
    assert done.stdout.startswith("CANOPY ")
    assert done.stdout.count("\n") == 1 and done.stdout.endswith("\n")
    assert done.stderr == ""


def test_the_package_command_prints_the_pins(tmp_path: Path) -> None:
    done = _run_in_a_bare_site(tmp_path, "osprey.integrations.canopy", {}, "--pins")
    assert done.returncode == 0, done.stderr
    assert done.stdout == f"canopy={CANOPY_PIN}\nPyDMConverter={PYDM_PIN}\n"
    assert done.stderr == ""


def test_the_package_command_with_a_clean_stderr__mutation_runs_the_submodule(
    tmp_path: Path,
) -> None:
    done = _run_in_a_bare_site(tmp_path, "osprey.integrations.canopy.proven", {}, "--pins")
    assert "RuntimeWarning" in done.stderr
    assert done.returncode == 0
    assert done.stdout == f"canopy={CANOPY_PIN}\nPyDMConverter={PYDM_PIN}\n"


# --- what a run says --------------------------------------------------------------------


class _Item:
    def __init__(self, marked: bool):
        self.marked = marked

    def get_closest_marker(self, name: str) -> object | None:
        return object() if self.marked and name == contract.MARKER else None


class _Report:
    def __init__(self, nodeid: str, *keywords: str, marked: bool = False):
        self.nodeid = nodeid
        self.keywords = dict.fromkeys((nodeid.rsplit("::", 1)[-1], *keywords), 1)
        contract.stamp(_Item(marked), self)


def _config(markexpr: str = "") -> SimpleNamespace:
    return SimpleNamespace(option=SimpleNamespace(markexpr=markexpr))


TEST_A = "tests/integrations/canopy/test_a.py::test_a"
TEST_B = "tests/integrations/canopy/test_b.py::test_b"

PROVEN_LINE = f"CANOPY contract proven against {PAIR} on Linux x86_64"


def _at_the_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, canopy=_at(CANOPY_PIN), PyDMConverter=_at(PYDM_PIN))
    monkeypatch.setattr(proven, "venue", lambda: " ".join(_VENUE))


def test_a_run_whose_canopy_tests_passed_at_the_pins_says_proven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _at_the_pins(monkeypatch)
    stats = {"passed": [_Report(TEST_A), _Report(TEST_B, "canopy", marked=True)]}
    assert contract.summary_lines(_config(), stats) == [PROVEN_LINE]


@pytest.mark.parametrize("category", ["failed", "error"])
def test_a_run_with_a_failed_canopy_test_at_the_pins_says_failed(
    monkeypatch: pytest.MonkeyPatch, category: str
) -> None:
    _at_the_pins(monkeypatch)
    stats = {"passed": [_Report(TEST_A)], category: [_Report(TEST_B, "canopy", marked=True)]}
    assert contract.summary_lines(_config(), stats) == [
        f"CANOPY contract failed against {PAIR} on Linux x86_64"
    ]


def test_a_failed_unmarked_test_does_not_fail_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _at_the_pins(monkeypatch)
    stats = {"passed": [_Report(TEST_B, "canopy", marked=True)], "failed": [_Report(TEST_A)]}
    assert contract.summary_lines(_config(), stats) == [PROVEN_LINE]


def test_an_editable_canopy_whose_label_names_the_pin_proves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, canopy=EDITABLE_AT_THE_PIN, PyDMConverter=_at(PYDM_PIN))
    stats = {"passed": [_Report(TEST_B, "canopy", marked=True)]}
    assert contract.summary_lines(_config(), stats) == [
        f"CANOPY contract proves nothing here: canopy editable, records no commit, "
        f"PyDMConverter {PYDM_PIN[:12]}; {PROVEN_AGAINST}"
    ]


def test_a_commit_that_cannot_be_read_proves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, canopy=_FakeDistribution("0.4.0"))
    stats = {"error": [_Report(TEST_B, "canopy", marked=True)]}
    assert contract.summary_lines(_config(), stats) == [
        "CANOPY contract proves nothing here: canopy installed without a recorded commit, "
        f"PyDMConverter not installed; {PROVEN_AGAINST}"
    ]


def test_a_run_without_a_canopy_test_names_no_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    _at_the_pins(monkeypatch)
    stats = {"passed": [_Report(TEST_A)], "warnings": [object()], "error": [object()]}
    assert contract.summary_lines(_config(), stats) == []


def test_a_deselected_canopy_test_is_not_one_that_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    _at_the_pins(monkeypatch)
    stats = {"passed": [_Report(TEST_A)], "deselected": [_Report(TEST_B, "canopy", marked=True)]}
    assert contract.summary_lines(_config("not canopy"), stats) == [contract.NOT_EXERCISED_TEXT]


def test_an_unmarked_test_under_the_canopy_directory_is_not_one_that_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _at_the_pins(monkeypatch)
    stats = {"passed": [_Report(TEST_A, "test_a.py", "canopy", "integrations", "tests")]}
    assert contract.summary_lines(_config(), stats) == []


def test_a_marked_test_outside_a_canopy_directory_is_one_that_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _at_the_pins(monkeypatch)
    nodeid = "tests/other/test_c.py::test_c"
    stats = {"passed": [_Report(nodeid, marked=True)]}
    assert contract.summary_lines(_config(), stats) == [PROVEN_LINE]


def _report_from_a_worker(pytestconfig: pytest.Config, marked: bool) -> pytest.TestReport:
    """A report of TEST_B as the controller of a parallel run receives it from a worker."""
    report = pytest.TestReport(
        nodeid=TEST_B,
        location=("tests/integrations/canopy/test_b.py", 0, "test_b"),
        keywords={"test_b": 1, "canopy": 1},
        outcome="passed",
        longrepr=None,
        when="call",
    )
    contract.stamp(_Item(marked), report)
    hook = pytestconfig.hook
    wire = json.loads(
        json.dumps(hook.pytest_report_to_serializable(config=pytestconfig, report=report))
    )
    return hook.pytest_report_from_serializable(config=pytestconfig, data=wire)


def test_a_canopy_test_run_by_a_worker_process_is_one_that_ran(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config
) -> None:
    _at_the_pins(monkeypatch)
    stats = {"passed": [_report_from_a_worker(pytestconfig, marked=True)]}
    assert contract.summary_lines(_config(), stats) == [PROVEN_LINE]


def test_an_unmarked_test_run_by_a_worker_process_is_not_one_that_ran(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config
) -> None:
    _at_the_pins(monkeypatch)
    stats = {"passed": [_report_from_a_worker(pytestconfig, marked=False)]}
    assert contract.summary_lines(_config(), stats) == []
