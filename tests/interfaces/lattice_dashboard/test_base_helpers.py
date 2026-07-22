"""Tests for lattice dashboard worker shared helpers (_base.py).

Covers the plumbing every worker relies on: settings merge, arg parsing,
state loading, and ring construction with parameter overrides — including
the baseline-ring loader.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from osprey.interfaces.lattice_dashboard.workers import _base
from osprey.interfaces.lattice_dashboard.workers._base import (
    load_baseline_ring,
    load_ring,
    load_settings,
    load_state,
    parse_args,
)


class TestLoadSettings:
    """load_settings merges saved values over group defaults."""

    def test_defaults_when_no_saved(self):
        merged = load_settings({}, "da")
        # Full default set returned even with empty state
        assert merged["nturns"] == 512
        assert merged["n_angles"] == 19

    def test_saved_overrides_defaults(self):
        state = {"settings": {"da": {"nturns": 1024}}}
        merged = load_settings(state, "da")
        assert merged["nturns"] == 1024
        # Untouched keys keep defaults
        assert merged["n_angles"] == 19

    def test_unknown_keys_dropped(self):
        state = {"settings": {"da": {"bogus": 999, "nturns": 256}}}
        merged = load_settings(state, "da")
        assert "bogus" not in merged
        assert merged["nturns"] == 256

    def test_unknown_group_returns_empty(self):
        assert load_settings({}, "no_such_group") == {}


class TestParseArgs:
    """parse_args reads state/output paths from argv."""

    def test_valid_args(self, monkeypatch):
        monkeypatch.setattr(_base.sys, "argv", ["prog", "/tmp/state.json", "/tmp/out.json"])
        state_path, output_path = parse_args()
        assert str(state_path) == "/tmp/state.json"
        assert str(output_path) == "/tmp/out.json"

    def test_wrong_arg_count_exits(self, monkeypatch):
        monkeypatch.setattr(_base.sys, "argv", ["prog", "only_one"])
        with pytest.raises(SystemExit):
            parse_args()


class TestLoadState:
    """load_state round-trips JSON from disk."""

    def test_reads_json(self, tmp_path):
        payload = {"base_lattice": "/x.mat", "overrides": {"QF": 1.5}}
        p = tmp_path / "state.json"
        p.write_text(json.dumps(payload))
        assert load_state(p) == payload


class TestLoadRing:
    """load_ring applies family overrides to the loaded ring."""

    def test_override_applied_to_family(self, make_fodo, monkeypatch):
        ring = make_fodo()
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: ring)

        state = {
            "base_lattice": "/fake.mat",
            "overrides": {"QF": 2.5},
            "families": {"QF": {"param": "K", "type": "quadrupole"}},
        }
        result = load_ring(state)

        qf_k = [e.K for e in result if e.FamName == "QF"]
        assert qf_k, "QF family should exist in ring"
        assert all(k == pytest.approx(2.5) for k in qf_k)

    def test_no_overrides_leaves_ring_unchanged(self, make_fodo, monkeypatch):
        ring = make_fodo()
        baseline_k = next(e.K for e in ring if e.FamName == "QF")
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: ring)

        state = {"base_lattice": "/fake.mat", "overrides": {}, "families": {}}
        result = load_ring(state)

        qf_k = next(e.K for e in result if e.FamName == "QF")
        assert qf_k == pytest.approx(baseline_k)

    def test_missing_family_param_defaults_to_k(self, make_fodo, monkeypatch):
        ring = make_fodo()
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: ring)

        # No families entry → param defaults to "K"
        state = {"base_lattice": "/fake.mat", "overrides": {"QD": -1.7}, "families": {}}
        result = load_ring(state)

        qd_k = [e.K for e in result if e.FamName == "QD"]
        assert all(k == pytest.approx(-1.7) for k in qd_k)


class TestLoadBaselineRing:
    """load_baseline_ring reads baseline.json alongside state.json."""

    def test_returns_none_when_no_baseline(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text("{}")
        assert load_baseline_ring(state_path, {"base_lattice": "/x.mat"}) is None

    def test_applies_baseline_overrides(self, tmp_path, make_fodo, monkeypatch):
        ring = make_fodo()
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: ring)

        state_path = tmp_path / "state.json"
        state_path.write_text("{}")
        (tmp_path / "baseline.json").write_text(json.dumps({"overrides": {"QF": 0.9}}))

        state = {
            "base_lattice": "/fake.mat",
            "families": {"QF": {"param": "K"}},
        }
        result = load_baseline_ring(state_path, state)

        assert result is not None
        qf_k = [e.K for e in result if e.FamName == "QF"]
        assert all(k == pytest.approx(0.9) for k in qf_k)

    def test_empty_overrides_returns_ring(self, tmp_path, make_fodo, monkeypatch):
        ring = make_fodo()
        monkeypatch.setattr(_base.at, "load_lattice", lambda path: ring)

        state_path = tmp_path / "state.json"
        state_path.write_text("{}")
        (tmp_path / "baseline.json").write_text(json.dumps({}))

        result = load_baseline_ring(state_path, {"base_lattice": "/fake.mat", "families": {}})
        assert result is not None
        assert isinstance(np.asarray(result.get_s_pos(len(result))), np.ndarray)
