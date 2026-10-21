"""End-to-end tests for each worker's main() CLI entry point.

Each worker is invoked as ``python -m ...workers.<name> <job> <output>``.
These tests drive main() directly over a job file, with a mocked lattice
loader returning a real in-memory FODO lattice and tiny compute settings, then
assert the output carries the job's key and job id and its data matches the
schema the figure adapters expect.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from osprey.interfaces.lattice_dashboard.workers import _base
from osprey.interfaces.lattice_dashboard.workers import chromaticity as chrom_mod
from osprey.interfaces.lattice_dashboard.workers import da as da_mod
from osprey.interfaces.lattice_dashboard.workers import footprint as fp_mod
from osprey.interfaces.lattice_dashboard.workers import lma as lma_mod
from osprey.interfaces.lattice_dashboard.workers import optics as optics_mod
from osprey.interfaces.lattice_dashboard.workers import resonance as res_mod


@pytest.fixture
def run_worker(tmp_path, make_fodo, monkeypatch):
    """Return a helper that runs a worker main() against a real FODO ring."""

    def _run(main_fn, settings=None, baseline_overrides=None, prepared=None):
        job = {
            "figure": "x",
            "key": "k" * 64,
            "job_id": 7,
            "deck": "/fake.json",
            "deck_sha256": "d" * 64,
            "prepared": prepared or {"solve": "periodic", "twiss_in": None},
            "families": {},
            "overrides": {},
            "baseline_overrides": baseline_overrides,
            "settings": settings,
            "periodicity": 1,
        }
        job_path = tmp_path / "jobs" / "7.json"
        job_path.parent.mkdir()
        job_path.write_text(json.dumps(job))
        output_path = tmp_path / "figures" / "out.json"

        # A fresh lattice per load so baseline + current don't alias/mutate each other.
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: make_fodo())
        monkeypatch.setattr(_base.sys, "argv", ["prog", str(job_path), str(output_path)])

        main_fn()
        payload = json.loads(output_path.read_text())
        assert payload["key"] == job["key"]
        assert payload["job_id"] == 7
        assert payload["deck_sha256"] == job["deck_sha256"]
        return payload["data"]

    return _run


def test_optics_main(run_worker):
    raw = run_worker(optics_mod.main)
    assert set(raw) == {"s_pos", "beta_x", "beta_y", "eta_x", "baseline", "summary_updates"}
    assert len(raw["s_pos"]) == len(raw["beta_x"])
    assert all(np.isfinite(raw["beta_x"]))
    assert raw["baseline"] is None


def test_optics_main_publishes_summary_updates(run_worker):
    """The dashboard's summary reads this block for the figure's key."""
    raw = run_worker(optics_mod.main)
    assert set(raw["summary_updates"]) == {"tunes", "chromaticity", "beta_max"}
    assert raw["summary_updates"]["beta_max"][0] == pytest.approx(max(raw["beta_x"]))


def test_optics_main_with_baseline(run_worker):
    raw = run_worker(optics_mod.main, baseline_overrides={})
    assert raw["baseline"] is not None
    assert set(raw["baseline"]) == {"s_pos", "beta_x", "beta_y", "eta_x"}


def test_optics_main_single_pass_starts_from_the_prepared_twiss_in(run_worker):
    prepared = {
        "solve": "single_pass",
        "twiss_in": {"beta": [7.0, 3.0], "alpha": [0.0, 0.0], "closed_orbit": [0.0] * 6},
    }
    raw = run_worker(optics_mod.main, prepared=prepared)
    assert raw["beta_x"][0] == pytest.approx(7.0)
    assert set(raw["summary_updates"]) == {"beta_max"}


def test_chromaticity_main(run_worker):
    raw = run_worker(chrom_mod.main, settings={"n_steps": 5})
    assert set(raw) == {"dp", "nux", "nuy", "baseline"}
    assert len(raw["dp"]) == 5
    assert all(np.isfinite(raw["nux"]))


def test_chromaticity_main_with_baseline(run_worker):
    raw = run_worker(chrom_mod.main, settings={"n_steps": 5}, baseline_overrides={})
    assert raw["baseline"] is not None
    assert set(raw["baseline"]) == {"dp", "nux", "nuy"}


def test_resonance_main(run_worker):
    raw = run_worker(res_mod.main)
    assert set(raw) == {"nux", "nuy", "baseline_nux", "baseline_nuy"}
    assert isinstance(raw["nux"], float)
    assert 0.0 <= raw["nux"] <= 1.0
    assert raw["baseline_nux"] is None


def test_resonance_main_with_baseline(run_worker):
    raw = run_worker(res_mod.main, baseline_overrides={})
    assert raw["baseline_nux"] is not None
    assert isinstance(raw["baseline_nuy"], float)


def test_da_main(run_worker):
    settings = {"nturns": 32, "n_angles": 5, "amp_max_mm": 10.0, "n_bisect": 5}
    raw = run_worker(da_mod.main, settings=settings)
    assert set(raw) == {"da_x", "da_y", "area_mm2", "nturns", "baseline"}
    assert raw["nturns"] == 32
    assert raw["area_mm2"] >= 0
    assert len(raw["da_x"]) == 2 * 5 + 1


def test_lma_main(run_worker):
    settings = {"nturns": 32, "n_refpts": 6, "dp_max_pct": 4.0, "n_bisect": 5, "n_sectors": 1}
    raw = run_worker(lma_mod.main, settings=settings)
    assert set(raw) == {"s_pos", "dp_plus", "dp_minus", "lattice_elements", "n_sectors", "baseline"}
    assert raw["n_sectors"] == 1
    assert len(raw["s_pos"]) == len(raw["dp_plus"]) == len(raw["dp_minus"])


def test_lma_main_with_baseline(run_worker):
    settings = {"nturns": 16, "n_refpts": 4, "dp_max_pct": 3.0, "n_bisect": 4, "n_sectors": 1}
    raw = run_worker(lma_mod.main, settings=settings, baseline_overrides={})
    assert raw["baseline"] is not None
    assert set(raw["baseline"]) == {"s_pos", "dp_plus", "dp_minus"}


def test_da_main_with_baseline(run_worker):
    settings = {"nturns": 16, "n_angles": 4, "amp_max_mm": 8.0, "n_bisect": 4}
    raw = run_worker(da_mod.main, settings=settings, baseline_overrides={})
    assert raw["baseline"] is not None
    assert set(raw["baseline"]) == {"da_x", "da_y", "area_mm2"}


def test_footprint_main(run_worker):
    settings = {"n_amp": 3, "x_max_mm": 1.0, "y_max_mm": 0.5, "n_half": 32}
    raw = run_worker(fp_mod.main, settings=settings)
    assert set(raw) == {
        "nux",
        "nuy",
        "amps",
        "diffusion",
        "design_tune",
        "n_amp",
        "baseline",
        "baseline_tune",
    }
    assert raw["n_amp"] == 3
    assert len(raw["design_tune"]) == 2
    # Surviving particles produce aligned arrays
    assert len(raw["nux"]) == len(raw["nuy"]) == len(raw["diffusion"])


def test_footprint_main_with_baseline(run_worker):
    settings = {"n_amp": 3, "x_max_mm": 1.0, "y_max_mm": 0.5, "n_half": 32}
    raw = run_worker(fp_mod.main, settings=settings, baseline_overrides={})
    assert raw["baseline"] is not None
    assert set(raw["baseline"]) == {"nux", "nuy", "amps"}
    assert raw["baseline_tune"] is not None
