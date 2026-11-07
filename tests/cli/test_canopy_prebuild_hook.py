"""The demo preset's ``canopy-payload`` pre_build step.

The step stages the prebuilt CANOPY payload each terminal's own CANOPY runs
from, and is the phase's only step. These tests run the phase through
:func:`_run_lifecycle_phase` exactly as ``build_cmd`` does — repo root as the
working directory, an empty stage directory as ``project_path``. The staging itself is tested with its module.

The step stays skipped here (``OSPREY_CANOPY_PAYLOAD_CHECK=0``, suite-wide in
``tests/conftest.py``). That switch does not skip everything: the step first
checks the deployment's screen library against its write-binding report,
whatever the switch says. The exemplar repo carries the preset's screen
library with its write-binding report, as ``osprey init`` puts it into a
deployment, so the check passes on a fresh library; the last section swaps it
for a library of its own, with and without a report.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from osprey.cli.build_lifecycle import _SHELL_METACHARACTERS, _run_lifecycle_phase
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.integrations.canopy import LIBRARY_SUBPATH, library
from osprey_connectors.errors import BuildProfileError


@pytest.fixture
def repo(lifecycle_repo: Path) -> Path:
    """The exemplar repo ``osprey init --preset demo`` materializes."""
    return lifecycle_repo


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    path = tmp_path / "stage"
    path.mkdir()
    return path


def _pre_build_steps(repo: Path) -> list:
    profile, _ = resolve_build_profile(repo / "profile.yml", None)
    return list(profile.lifecycle.pre_build)


def _run(repo: Path, stage: Path) -> None:
    """Run the phase exactly as ``build_cmd`` does for ``pre_build``."""
    _run_lifecycle_phase(
        "pre_build", _pre_build_steps(repo), repo, stage, venv=None, repo_root=repo
    )


# ---------------------------------------------------------------------------
# The materialized hook
# ---------------------------------------------------------------------------


def test_the_materialized_profile_carries_the_payload_step_alone(repo: Path) -> None:
    steps = _pre_build_steps(repo)
    assert [step.name for step in steps] == ["canopy-payload"]
    assert [step.timeout for step in steps] == [30]


def test_the_payload_step_runs_the_staging_module_without_a_shell(repo: Path) -> None:
    """The interpreter running ``osprey`` runs the module; no shell is involved."""
    (step,) = _pre_build_steps(repo)
    assert step.run == '"{python}" -m osprey.integrations.canopy.stage'
    assert not any(meta in step.run for meta in _SHELL_METACHARACTERS)


def test_the_suite_opts_out_of_the_payload_step() -> None:
    assert os.environ.get("OSPREY_CANOPY_PAYLOAD_CHECK") == "0"


def test_an_unset_payload_passes_the_step_once_it_is_on(
    repo: Path, stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The build then leaves the panel out (``build_profile_canopy``); the step stages nothing."""
    monkeypatch.setenv("OSPREY_CANOPY_PAYLOAD_CHECK", "1")
    monkeypatch.delenv("OSPREY_CANOPY_PAYLOAD", raising=False)
    _run(repo, stage)
    assert not (repo / "canopy_payload").exists()


def test_a_named_payload_that_is_missing_refuses_the_build(
    repo: Path, stage: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSPREY_CANOPY_PAYLOAD_CHECK", "1")
    monkeypatch.setenv("OSPREY_CANOPY_PAYLOAD", str(tmp_path / "nowhere"))
    with pytest.raises(BuildProfileError, match="canopy-payload"):
        _run(repo, stage)
    assert not (repo / "canopy_payload").exists()


# ---------------------------------------------------------------------------
# The screen library, checked by the payload step before its own switch
# ---------------------------------------------------------------------------


def _make_library(repo: Path, *, report: bool) -> Path:
    """Replace the exemplar's library with a one-screen library of the test's own."""
    root = repo / LIBRARY_SUBPATH
    shutil.rmtree(root)
    (root / "screens" / "sr").mkdir(parents=True)
    (root / "screens" / "sr" / "overview.screen.json").write_bytes(b"{}\n")
    for path in [root.parent.parent, root.parent, root, *root.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    if report:
        judged = library.judge(library.scan(root))
        document = {
            "generator": {"schema": "canopy-library-report/1"},
            "files": {
                "digest": judged.digest,
                "list": [{"path": path, "sha256": sha} for path, sha in judged.files],
            },
        }
        (root / library.REPORT_JSON).write_text(json.dumps(document) + "\n", encoding="utf-8")
        (root / library.REPORT_JSON).chmod(0o644)
    return root


def test_the_exemplar_repos_library_is_fresh(repo: Path, stage: Path) -> None:
    """The library the exemplar ships matches its report, so the phase passes."""
    assert (repo / LIBRARY_SUBPATH / library.REPORT_JSON).is_file()
    answer = library.library_is_fresh(repo / LIBRARY_SUBPATH)
    assert answer.fresh, answer.lines
    assert answer.lines[0].startswith("report is current (")
    _run(repo, stage)


def test_a_library_with_its_report_passes_the_phase(repo: Path, stage: Path) -> None:
    _make_library(repo, report=True)
    _run(repo, stage)


def test_a_library_without_a_report_stops_the_build_though_staging_is_off(
    repo: Path, stage: Path
) -> None:
    assert os.environ.get("OSPREY_CANOPY_PAYLOAD_CHECK") == "0"
    root = _make_library(repo, report=False)
    with pytest.raises(BuildProfileError) as excinfo:
        _run(repo, stage)
    message = str(excinfo.value)
    assert "canopy-payload" in message
    assert "no report: the library holds 1 files and no LIBRARY-REPORT.json" in message
    assert f"python -m canopy_mcp.report {root} --served-at library --write" in message
    assert not (repo / "canopy_payload").exists()


def test_a_library_changed_after_its_report_stops_the_build(repo: Path, stage: Path) -> None:
    root = _make_library(repo, report=True)
    (root / "screens" / "sr" / "overview.screen.json").write_bytes(b'{"edited": true}\n')
    with pytest.raises(BuildProfileError, match="changed: screens/sr/overview.screen.json"):
        _run(repo, stage)
