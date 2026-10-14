"""What a model's engine stops a scenario on, beside the model's deck.

The facility build shows each scenario's engine-keyed faults on a model to
that model's engine, with the model's deck, when the engine's plug-in states
``scenario_stops``; every answer is one ``engine-invalid`` stop on the
scenario. The pyat engine answers one: a non-zero misalignment sigma on a deck
that holds a magnet of that class under a repeated name. A deck no such sigma
targets is never read, and a plug-in without the function is asked nothing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.facility import scenarios
from osprey.facility.errors import FacilityBuildError
from osprey.facility.scenarios import check_scenario_engines
from tests.facility._synthetic_trees import Deck, deck_tree, sr_deck, write_tree

pytest.importorskip("at")

PROJECT = "demo"

#: The stop a quadrupole sigma meets on a deck holding two quadrupoles named QX.
REPEATED_QUADRUPOLE = (
    "facility: engine-invalid: scenario shaken — faults.SR.quadrupole_dx_sigma draws one "
    "misalignment per quadrupole name, and the deck repeats QX (2 times); fix: give every "
    "quadrupole a unique name in the deck, or state no quadrupole sigma"
)


def _quadrupole(name: str) -> Any:
    return lambda at: at.Quadrupole(name, 0.1, 0.0)


def _tree(faults: dict[str, Any] | None, *names: str) -> dict[str, Any]:
    """The deck tree, its periodic deck ending in a quadrupole per name, and one scenario."""
    tree = deck_tree()
    tree["decks/sr.json"] = sr_deck(*(_quadrupole(name) for name in names))
    if faults is not None:
        tree["scenarios/shaken.yaml"] = {"faults": {"SR": faults}}
    return tree


def _build(tmp_path: Path, tree: dict[str, Any]) -> dict[str, Any]:
    from osprey.facility.build import build_facility

    return build_facility(write_tree(tmp_path / "data" / "facility", tree), project_name=PROJECT)


def _stop_line(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        _build(tmp_path, tree)
    return str(stopped.value.format_message())


# --- through the facility build --------------------------------------------------------


def test_a_sigma_on_a_deck_repeating_a_magnet_name_stops_the_build(tmp_path: Path) -> None:
    tree = _tree({"quadrupole_dx_sigma": 1.0e-4}, "QX", "QX")

    assert _stop_line(tmp_path, tree) == REPEATED_QUADRUPOLE


def test_the_same_deck_builds_when_no_scenario_states_a_sigma(tmp_path: Path) -> None:
    assert _build(tmp_path, _tree(None, "QX", "QX"))["scenarios"] == []


def test_a_seed_alone_on_that_deck_builds(tmp_path: Path) -> None:
    assert _build(tmp_path, _tree({"misalignment_seed": 7}, "QX", "QX"))


def test_a_zero_sigma_on_that_deck_builds(tmp_path: Path) -> None:
    assert _build(tmp_path, _tree({"quadrupole_dx_sigma": 0.0}, "QX", "QX"))


def test_a_sigma_of_another_magnet_class_on_that_deck_builds(tmp_path: Path) -> None:
    assert _build(tmp_path, _tree({"sextupole_roll_sigma": 1.0e-3}, "QX", "QX"))


def test_a_sigma_on_a_deck_of_unique_magnet_names_builds(tmp_path: Path) -> None:
    """The deck's three drifts share one name; no sigma selects a drift."""
    assert _build(tmp_path, _tree({"quadrupole_dx_sigma": 1.0e-4}, "QX", "QY"))


def test_a_stopped_scenario_counts_as_one_the_build_stops_on(tmp_path: Path) -> None:
    from osprey.facility.validate import stale_scenarios

    tree = _tree({"quadrupole_dx_sigma": 1.0e-4}, "QX", "QX")
    facility = write_tree(tmp_path / "data" / "facility", tree)

    assert stale_scenarios(facility, project_name=PROJECT) == ["shaken"]


def test_an_unknown_fault_name_is_told_the_model_wide_names(tmp_path: Path) -> None:
    line = _stop_line(tmp_path, _tree({"ZZZZ": 1.0}))

    assert line == (
        "facility: engine-invalid: scenario shaken — faults.SR.ZZZZ names no channel and no "
        "pyat variable; fix: name a channel address or a pyat variable (<address>/<field>, or "
        "one of aperture_elliptical_x, aperture_elliptical_y, aperture_rectangular_x, "
        "aperture_rectangular_y, dipole_dx_sigma, dipole_dy_sigma, dipole_roll_sigma, "
        "misalignment_seed, quadrupole_dx_sigma, quadrupole_dy_sigma, quadrupole_roll_sigma, "
        "sextupole_dx_sigma, sextupole_dy_sigma, sextupole_roll_sigma)"
    )


# --- the pyat engine's answer ----------------------------------------------------------


def _saved(tmp_path: Path, *names: str) -> Path:
    path = tmp_path / "deck.json"
    Deck(lambda at: [at.Drift("D", 1.0), *(_quadrupole(name)(at) for name in names)]).save(path)
    return path


def test_no_sigma_reads_no_deck(tmp_path: Path) -> None:
    from osprey.simulation.engines import pyat

    absent = tmp_path / "absent.json"

    assert pyat.scenario_stops(absent, {}) == []
    assert pyat.scenario_stops(absent, {"misalignment_seed": 3, "BPM1:X/offset": 1.0}) == []
    assert pyat.scenario_stops(absent, {"dipole_dy_sigma": 0.0}) == []
    assert pyat.scenario_stops(absent, {"dipole_dy_sigma": "wide"}) == []


def test_a_magnet_class_stops_once_under_its_first_sigma(tmp_path: Path) -> None:
    from osprey.simulation.engines import pyat

    faults = {"quadrupole_roll_sigma": 1.0e-3, "quadrupole_dy_sigma": 1.0e-4}

    stops = pyat.scenario_stops(_saved(tmp_path, "QX", "QX"), faults)

    assert [name for name, _detail, _remedy in stops] == ["quadrupole_dy_sigma"]


def test_more_repeated_names_than_are_listed_are_counted(tmp_path: Path) -> None:
    from osprey.simulation.engines import pyat

    deck = _saved(tmp_path, *("QA", "QB", "QC", "QD", "QE") * 2, "QC")

    ((name, detail, remedy),) = pyat.scenario_stops(deck, {"quadrupole_dx_sigma": 1.0})

    assert name == "quadrupole_dx_sigma"
    assert detail == (
        "draws one misalignment per quadrupole name, and the deck repeats QA (2 times), "
        "QB (2 times), QC (3 times) and 2 more"
    )
    assert remedy == "give every quadrupole a unique name in the deck, or state no quadrupole sigma"


def test_a_name_a_magnet_shares_with_another_element_is_repeated(tmp_path: Path) -> None:
    from osprey.simulation.engines import pyat

    path = tmp_path / "deck.json"
    Deck(lambda at: [at.Drift("QX", 1.0), at.Quadrupole("QX", 0.1, 0.0)]).save(path)

    ((_name, detail, _remedy),) = pyat.scenario_stops(path, {"quadrupole_dx_sigma": 1.0})

    assert detail.endswith("the deck repeats QX (2 times)")


# --- what the build asks of any engine -------------------------------------------------


def _document(tmp_path: Path, faults: dict[str, Any], *, deck: bool = True) -> dict[str, Any]:
    if deck:
        (tmp_path / "m.deck").write_text("", encoding="utf-8")
    model = {"name": "M", "engine": "stub", "wiring": [{"address": "A:SP"}]}
    return {
        "channels": [{"id": "A:SP", "role": "setpoint"}],
        "models": [{**model, "deck": "m.deck"} if deck else model],
        "scenarios": [{"name": "odd", "faults": {"M": faults}}],
    }


def _plugged(monkeypatch: pytest.MonkeyPatch, **names: Any) -> list[tuple[Path, dict[str, Any]]]:
    """Register a stub plug-in stating ``names``; returns what its ``scenario_stops`` was shown."""
    shown: list[tuple[Path, dict[str, Any]]] = []
    if "scenario_stops" in names:
        answer = names["scenario_stops"]

        def scenario_stops(deck: Path, faults: dict[str, Any]) -> list[tuple[str, str, str]]:
            shown.append((deck, dict(faults)))
            return answer

        names["scenario_stops"] = scenario_stops
    plugin = SimpleNamespace(fault_variables=lambda wiring: {"wobble": _slot()}, **names)
    monkeypatch.setattr(scenarios, "_plugin", lambda model: plugin)
    return shown


def _slot() -> Any:
    return SimpleNamespace(address=None, field="wobble", value_range=None, options=None)


def test_a_plug_in_without_the_function_is_asked_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plugged(monkeypatch)

    assert check_scenario_engines(_document(tmp_path, {"wobble": 1.0}), tmp_path) == []


def test_each_answer_is_one_stop_on_the_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = [("wobble", "asks more than the deck holds", "ask less")]
    shown = _plugged(monkeypatch, scenario_stops=answer)

    (stop,) = check_scenario_engines(_document(tmp_path, {"wobble": 1.0, "A:SP": 2.0}), tmp_path)

    assert shown == [(tmp_path / "m.deck", {"wobble": 1.0})]
    assert stop.sources == ("scenarios/odd.yaml",)
    assert str(stop.format_message()) == (
        "facility: engine-invalid: scenario odd — faults.M.wobble asks more than the deck holds; "
        "fix: ask less"
    )


def test_a_scenario_keying_only_channels_is_not_shown_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown = _plugged(monkeypatch, scenario_stops=[("wobble", "never", "never")])

    assert check_scenario_engines(_document(tmp_path, {"A:SP": 2.0}), tmp_path) == []
    assert shown == []


def test_a_model_without_a_deck_is_not_shown_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown = _plugged(monkeypatch, scenario_stops=[("wobble", "never", "never")])

    assert check_scenario_engines(_document(tmp_path, {"wobble": 1.0}, deck=False), tmp_path) == []
    assert shown == []
