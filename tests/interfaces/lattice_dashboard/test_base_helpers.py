"""Tests for lattice dashboard worker shared helpers (_base.py).

Covers the plumbing every worker relies on: settings merge, arg parsing,
job loading, and lattice construction from a job with parameter overrides —
including the baseline loader.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from osprey.interfaces.lattice_dashboard.workers import _base
from osprey.interfaces.lattice_dashboard.workers._base import (
    load_baseline_lattice,
    load_job,
    load_lattice,
    load_settings,
    parse_args,
    prepared_twiss_in,
)


class TestLoadSettings:
    """load_settings merges the job's settings group over the group defaults."""

    def test_defaults_when_no_saved(self):
        merged = load_settings({"settings": None}, "da")
        # Full default set returned even with no settings in the job
        assert merged["nturns"] == 512
        assert merged["n_angles"] == 19

    def test_saved_overrides_defaults(self):
        job = {"settings": {"nturns": 1024}}
        merged = load_settings(job, "da")
        assert merged["nturns"] == 1024
        # Untouched keys keep defaults
        assert merged["n_angles"] == 19

    def test_unknown_keys_dropped(self):
        job = {"settings": {"bogus": 999, "nturns": 256}}
        merged = load_settings(job, "da")
        assert "bogus" not in merged
        assert merged["nturns"] == 256

    def test_unknown_group_returns_empty(self):
        assert load_settings({}, "no_such_group") == {}


class TestParseArgs:
    """parse_args reads job/output paths from argv."""

    def test_valid_args(self, monkeypatch):
        monkeypatch.setattr(_base.sys, "argv", ["prog", "/tmp/job.json", "/tmp/out.json"])
        job_path, output_path = parse_args()
        assert str(job_path) == "/tmp/job.json"
        assert str(output_path) == "/tmp/out.json"

    def test_wrong_arg_count_exits(self, monkeypatch):
        monkeypatch.setattr(_base.sys, "argv", ["prog", "only_one"])
        with pytest.raises(SystemExit):
            parse_args()


class TestLoadJob:
    """load_job round-trips JSON from disk."""

    def test_reads_json(self, tmp_path):
        payload = {"deck": "/x.json", "overrides": {"QF": 1.5}}
        p = tmp_path / "7.json"
        p.write_text(json.dumps(payload))
        assert load_job(p) == payload


class TestLoadRing:
    """load_lattice applies the job's family overrides to the loaded deck."""

    def test_override_applied_to_family(self, make_fodo, monkeypatch):
        ring = make_fodo()
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: ring)

        job = {"deck": "/fake.json", "overrides": {"QF": 2.5}, "families": {"QF": "K"}}
        result = load_lattice(job)

        qf_k = [e.K for e in result if e.FamName == "QF"]
        assert qf_k, "QF family should exist in the lattice"
        assert all(k == pytest.approx(2.5) for k in qf_k)

    def test_loads_the_jobs_deck(self, make_fodo, monkeypatch):
        loaded = []
        monkeypatch.setattr(
            _base.at, "load_lattice", lambda path: loaded.append(path) or make_fodo()
        )

        load_lattice({"deck": "/decks/SR.json", "overrides": {}, "families": {}})

        assert loaded == ["/decks/SR.json"]

    def test_no_overrides_leaves_ring_unchanged(self, make_fodo, monkeypatch):
        ring = make_fodo()
        baseline_k = next(e.K for e in ring if e.FamName == "QF")
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: ring)

        result = load_lattice({"deck": "/fake.json", "overrides": {}, "families": {}})

        qf_k = next(e.K for e in result if e.FamName == "QF")
        assert qf_k == pytest.approx(baseline_k)

    def test_missing_family_param_defaults_to_k(self, make_fodo, monkeypatch):
        ring = make_fodo()
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: ring)

        # No families entry → param defaults to "K"
        result = load_lattice({"deck": "/fake.json", "overrides": {"QD": -1.7}, "families": {}})

        qd_k = [e.K for e in result if e.FamName == "QD"]
        assert all(k == pytest.approx(-1.7) for k in qd_k)


class TestLoadBaselineRing:
    """load_baseline_lattice applies the job's baseline overrides."""

    def test_returns_none_when_no_baseline(self):
        assert load_baseline_lattice({"deck": "/x.json", "baseline_overrides": None}) is None

    def test_applies_baseline_overrides(self, make_fodo, monkeypatch):
        ring = make_fodo()
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: ring)

        job = {"deck": "/fake.json", "baseline_overrides": {"QF": 0.9}, "families": {"QF": "K"}}
        result = load_baseline_lattice(job)

        assert result is not None
        qf_k = [e.K for e in result if e.FamName == "QF"]
        assert all(k == pytest.approx(0.9) for k in qf_k)

    def test_empty_overrides_returns_ring(self, make_fodo, monkeypatch):
        ring = make_fodo()
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: ring)

        result = load_baseline_lattice(
            {"deck": "/fake.json", "baseline_overrides": {}, "families": {}}
        )
        assert result is not None
        assert isinstance(np.asarray(result.get_s_pos(len(result))), np.ndarray)


class TestPreparedTwissIn:
    def test_single_pass_gives_arrays(self):
        job = {"prepared": {"solve": "single_pass", "twiss_in": {"beta": [7.0, 3.0]}}}
        assert prepared_twiss_in(job)["beta"].tolist() == [7.0, 3.0]

    def test_periodic_gives_none(self):
        assert prepared_twiss_in({"prepared": {"solve": "periodic", "twiss_in": None}}) is None
