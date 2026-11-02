"""Drift of the materialized login files from the bundled source each names.

``osprey init`` writes every login as an explicit file under ``tiers/`` with a
``provenance:`` block naming the bundled source it was made from, exactly as
``profile.yml`` names its preset. ``osprey validate`` compares each of them
with what ``init`` would write from that source today, so a facility edit to a
shared login is reported the way an edit to the profile is —
and a file whose source no longer ships is named rather than crashed on.

The fixture is one real ``osprey init`` of ``control-assistant``, copied per
test so each test edits its own repo the way an operator would.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_drift import DriftReport, preset_drift_report
from osprey.cli.init_cmd import init
from osprey.cli.validate_cmd import validate

PROFILE = "profile.yml"
LOGBOOK = Path("tiers") / "logbook.yml"
READONLY = Path("tiers") / "readonly.yml"
EXAMPLE_ARTIFACT = "artifact_server.example_artifact: false"


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real materialization of the control-assistant preset."""
    root = tmp_path_factory.mktemp("layer-drift") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", "control-assistant", "--no-git"])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def repo(materialized: Path, tmp_path: Path) -> Path:
    """A private copy of the materialization for this test to edit."""
    target = tmp_path / "demo"
    shutil.copytree(materialized, target)
    return target


def _report(repo: Path, target: Path | None = None) -> DriftReport:
    """Run the lint the way ``osprey validate`` does for *target* (default: the root)."""
    profile_file = target or repo / PROFILE
    profile, _ = resolve_build_profile(profile_file, None)
    return preset_drift_report(profile_file, profile.provenance)


def _validate(*args: str):
    return CliRunner().invoke(validate, list(args))


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} must occur exactly once in {path}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _drift_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if "preset drift" in line]


def test_fresh_init_repo_reports_no_drift(repo: Path) -> None:
    assert (repo / LOGBOOK).is_file() and (repo / READONLY).is_file()
    report = _report(repo)
    assert report.findings == []
    assert report.stale_markers == []
    result = _validate(str(repo / PROFILE))
    assert result.exit_code == 0, result.output
    assert _drift_lines(result.output) == []
    for layer in (LOGBOOK, READONLY):
        result = _validate(str(repo / layer))
        assert result.exit_code == 0, result.output
        assert _drift_lines(result.output) == []


def test_every_login_file_is_paired(repo: Path) -> None:
    """Each login file in the repo is compared — none is silently skipped."""
    layers = sorted(
        path.relative_to(repo).as_posix()
        for dirname in ("tiers",)
        for path in (repo / dirname).glob("*.yml")
    )
    assert len(layers) == 6, "the flagship preset materializes six logins"
    for relative in layers:
        path = repo / relative
        text = path.read_text(encoding="utf-8")
        path.write_text(text.rstrip("\n") + "\nzz_unknown_key: 1\n", encoding="utf-8")
    subjects = sorted(
        finding.profile_ref.split(":")[0]
        for finding in _report(repo).findings
        if finding.subject == "zz_unknown_key"
    )
    assert subjects == layers


def test_one_edited_shared_login_key_is_one_drift_line(repo: Path) -> None:
    _edit(repo / LOGBOOK, EXAMPLE_ARTIFACT, "artifact_server.example_artifact: true")
    report = _report(repo)
    assert len(report.unmarked) == 1
    finding = report.unmarked[0]
    assert finding.kind == "value"
    assert finding.subject == "config.artifact_server.example_artifact"
    assert finding.profile_ref.startswith("tiers/logbook.yml:")

    result = _validate(str(repo / PROFILE))
    assert result.exit_code == 0, result.output  # a value difference is a note
    lines = _drift_lines(result.output)
    assert len(lines) == 1, result.output
    assert "tiers/logbook.yml" in lines[0]
    assert "artifact_server.example_artifact" in lines[0]


def test_validating_the_shared_login_file_reports_that_login_only(repo: Path) -> None:
    _edit(repo / LOGBOOK, EXAMPLE_ARTIFACT, "artifact_server.example_artifact: true")
    _edit(repo / READONLY, "control_system.writes: never", "control_system.writes: asked")
    refs = [f.profile_ref for f in _report(repo, repo / LOGBOOK).unmarked]
    assert len(refs) == 1 and refs[0].startswith("tiers/logbook.yml:")
    refs = [f.profile_ref for f in _report(repo, repo / READONLY).unmarked]
    assert len(refs) == 1 and refs[0].startswith("tiers/readonly.yml:")


def test_structural_shared_login_drift_is_refused(repo: Path) -> None:
    _edit(repo / LOGBOOK, "  - logbook-deep-research", "")
    result = _validate(str(repo / PROFILE))
    assert result.exit_code == 2, result.output
    assert "tiers/logbook.yml" in result.output
    assert "logbook-deep-research" in result.output

    passed = _validate(str(repo / PROFILE), "--drift=warn")
    assert passed.exit_code == 0, passed.output
    assert len(_drift_lines(passed.output)) == 1


def test_a_marker_claims_a_layer_difference(repo: Path) -> None:
    _edit(
        repo / LOGBOOK,
        EXAMPLE_ARTIFACT,
        "# DEVIATION: our gallery shows the example\n  artifact_server.example_artifact: true",
    )
    report = _report(repo)
    assert report.unmarked == []
    assert [f.marked for f in report.findings] == [True]


def test_pairing_follows_provenance_not_the_file_name(repo: Path) -> None:
    """A copy of a login under another name still pairs with the source it names."""
    shutil.copy(repo / LOGBOOK, repo / "tiers" / "night-shift.yml")
    assert _report(repo).findings == []


def test_source_that_no_longer_ships_is_reported_not_crashed_on(repo: Path) -> None:
    _edit(repo / LOGBOOK, "  preset: logbook", "  preset: retired-app")
    report = _report(repo)
    assert report.note is not None
    assert "tiers/logbook.yml" in report.note and "retired-app" in report.note
    assert report.findings == []

    result = _validate(str(repo / PROFILE))
    assert result.exit_code == 0, result.output
    assert "retired-app" in result.output
    assert "tiers/logbook.yml" in result.output


def test_a_moved_on_source_is_noted(repo: Path) -> None:
    text = (repo / LOGBOOK).read_text(encoding="utf-8")
    old = next(line for line in text.splitlines() if line.startswith("  preset_hash:"))
    _edit(repo / LOGBOOK, old, "  preset_hash: sha256:" + "0" * 64)
    report = _report(repo)
    assert report.note is not None and "tiers/logbook.yml" in report.note
    assert report.findings == []


def test_a_layer_without_provenance_is_not_compared(repo: Path) -> None:
    path = repo / LOGBOOK
    text = path.read_text(encoding="utf-8")
    head = text[: text.index("provenance:")]
    path.write_text(head + "zz_unknown_key: 1\n", encoding="utf-8")
    assert _report(repo).findings == []
