"""The once-per-sub-machine rule of the MATLAB parity fixture, without MATLAB.

``test_mml_export_parity`` skips as a whole where there is no MATLAB, so the
rule its ``export`` fixture relies on is held here with a stand-in export.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.templates.test_mml_export_parity import CASES, Case, once_per_case


def test_a_finished_export_runs_once(tmp_path: Path) -> None:
    calls: list[str] = []

    def export_case(case: Case, workdir: Path) -> Path:
        calls.append(case.key)
        return workdir / "ao.json"

    run = once_per_case(tmp_path, export_case)

    assert run(CASES[0]) == run(CASES[0]) == tmp_path / CASES[0].key / "ao.json"
    assert calls == [CASES[0].key]


def test_a_failed_export_is_raised_to_every_later_request(tmp_path: Path) -> None:
    calls: list[str] = []

    def export_case(case: Case, workdir: Path) -> Path:
        calls.append(case.key)
        assert workdir.is_dir()
        raise AssertionError(f"exporting {case.key} failed (exit 1)")

    run = once_per_case(tmp_path, export_case)

    for _ in range(3):
        with pytest.raises(AssertionError, match="failed \\(exit 1\\)"):
            run(CASES[0])
    assert calls == [CASES[0].key]


def test_one_failure_leaves_the_other_sub_machines_alone(tmp_path: Path) -> None:
    def export_case(case: Case, workdir: Path) -> Path:
        if case is CASES[0]:
            raise AssertionError("first sub-machine failed")
        return workdir / "ao.json"

    run = once_per_case(tmp_path, export_case)

    with pytest.raises(AssertionError, match="first sub-machine failed"):
        run(CASES[0])
    assert run(CASES[1]) == tmp_path / CASES[1].key / "ao.json"
