"""Tests for the lattice dashboard's deck description and figure keys.

``describe_deck`` reads a deck's magnet families and its own summary numbers
without solving its optics; ``figure_key`` names one figure's inputs.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from osprey.interfaces.lattice_dashboard.state import (
    DEFAULT_SESSION,
    LatticeState,
    Selection,
    capabilities_for,
    describe_deck,
    figure_key,
    prepared_lists,
)
from osprey.simulation.engines.pyat import Prepared


def _make_mock_deck(n_elements: int = 10, energy: float = 2e9):
    ring = MagicMock()
    ring.__len__ = lambda self: n_elements
    ring.energy = energy
    ring.get_s_pos.return_value = np.array([100.0])
    ring.__iter__ = lambda self: iter([])
    return ring


class TestDescribeDeck:
    def test_summary_is_the_decks_own_numbers(self):
        mock_at = MagicMock()
        mock_at.load_lattice.return_value = _make_mock_deck()

        with patch.dict(sys.modules, {"at": mock_at}):
            families, summary = describe_deck("/fake/lattice.json")

        assert families == {}
        assert summary == {
            "energy_gev": 2.0,
            "circumference_m": 100.0,
            "periodicity": 1,
            "num_elements": 10,
        }
        mock_at.get_optics.assert_not_called()


_PREPARED = Prepared(solve="periodic", twiss_in=None, rest_mass_gev=0.000511, length_m=8.0)


def _key(**changes):
    inputs = {
        "deck_sha256": "d" * 64,
        "prepared": prepared_lists(_PREPARED),
        "settings": None,
        "overrides": {},
        "baseline_overrides": None,
    }
    inputs.update(changes)
    return figure_key("optics", **inputs)


class TestFigureKey:
    def test_the_same_inputs_give_the_same_key(self):
        assert _key(overrides={"QF": 1.0, "QD": -1.0}) == _key(overrides={"QD": -1.0, "QF": 1.0})

    @pytest.mark.parametrize(
        "change",
        [
            {"deck_sha256": "e" * 64},
            {"settings": {"n_steps": 5}},
            {"overrides": {"QF": 1.0}},
            {"baseline_overrides": {}},
            {"prepared": {**prepared_lists(_PREPARED), "rest_mass_gev": 0.938}},
        ],
    )
    def test_any_input_changes_the_key(self, change):
        assert _key(**change) != _key()

    def test_twiss_in_is_serialised_as_lists(self):
        prepared = Prepared(
            solve="single_pass",
            twiss_in={"beta": np.array([7.0, 3.0])},
            rest_mass_gev=0.000511,
            length_m=8.0,
        )
        assert prepared_lists(prepared)["twiss_in"] == {"beta": [7.0, 3.0]}


class TestStore:
    @pytest.fixture
    def state(self, tmp_path):
        state = LatticeState(tmp_path / "lattice")
        state.adopt(
            Selection(
                model="SR",
                status="ready",
                deck=tmp_path / "SR.json",
                deck_sha256="d" * 64,
                prepared=_PREPARED,
                capabilities=capabilities_for("periodic"),
                summary={"energy_gev": 2.0},
            ),
            reset=False,
        )
        return state

    def test_constructing_writes_no_file(self, tmp_path):
        LatticeState(tmp_path / "lattice")
        assert not (tmp_path / "lattice").exists()

    def test_the_unmodified_deck_is_stored_shared(self, state, tmp_path):
        key = state.figure_key("optics")
        assert state.figure_path("optics", key) == (
            tmp_path / "lattice" / "figures" / "shared" / "optics" / f"{key}.json"
        )

    def test_a_what_if_is_stored_in_the_session(self, state, tmp_path):
        state.set_param("QF", 1.1)
        key = state.figure_key("optics")
        assert state.figure_path("optics", key) == (
            tmp_path
            / "lattice"
            / "sessions"
            / DEFAULT_SESSION
            / "figures"
            / "optics"
            / f"{key}.json"
        )

    def test_prune_keeps_the_newest_keys(self, state):
        directory = state.figure_path("optics", "x").parent
        directory.mkdir(parents=True)
        for n in range(10):
            path = directory / f"{n}.json"
            path.write_text("{}")
            stamp = 1_000_000_000 + n
            os.utime(path, (stamp, stamp))

        state.prune("optics", kept=8)

        assert sorted(p.stem for p in directory.iterdir()) == [str(n) for n in range(2, 10)]

    def test_an_adopted_selection_sets_the_baseline_to_the_deck(self, state):
        assert state.get_baseline()["overrides"] == {}
        assert state.get_baseline()["summary"] == {"energy_gev": 2.0}
