"""``OPTIMIZER_RESULTS_DIR`` moves where runs are written and read.

The deployment points it at the mounted results volume; unset, runs land in
``./results`` as they always have.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.results import (
    RESULTS_DIR_ENV,
    initialize_results_dir,
    results_root,
    scan_results_directory,
)


def test_unset_means_the_working_directorys_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RESULTS_DIR_ENV, raising=False)
    assert results_root() == Path("results")


def test_a_run_is_written_under_the_configured_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(RESULTS_DIR_ENV, str(tmp_path / "volume"))
    config = OptimizationConfig(environment_name="sim", control_target="va", objective_name="obj")

    run_dir = initialize_results_dir(config, "job-1")

    assert run_dir.parent == tmp_path / "volume" / "sim"
    assert scan_results_directory() == [f"sim/{run_dir.name}"]


def test_an_explicit_directory_still_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(RESULTS_DIR_ENV, str(tmp_path / "volume"))
    config = OptimizationConfig(environment_name="sim", control_target="va", objective_name="obj")

    run_dir = initialize_results_dir(config, "job-1", tmp_path / "elsewhere")

    assert run_dir.parent == tmp_path / "elsewhere"
