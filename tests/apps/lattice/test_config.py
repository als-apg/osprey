"""The deployment's choices: decks, engines, the analyses each model runs, and their bounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lattice_app.catalog import ANALYSES
from lattice_app.config import DEFAULT_DEADLINE_S, load_settings


def _config(**over: Any) -> dict[str, Any]:
    return {"models": [{"name": "sr", "deck": "decks/sr.lte"}], **over}


def test_a_model_defaults_to_pyat_and_every_analysis(tmp_path: Path) -> None:
    settings = load_settings(_config(), tmp_path)

    [model] = settings.models
    assert model.engine.name == "pyat"
    assert model.deck == tmp_path / "decks" / "sr.lte"
    assert model.label == "sr"
    assert model.offered() == list(ANALYSES)
    assert model.setups["da"].settings == dict(ANALYSES["da"].settings)
    assert model.setups["da"].deadline_s == DEFAULT_DEADLINE_S
    assert settings.max_workers == 2
    assert settings.precompute is False


def test_the_facility_chooses_which_analyses_run_their_settings_and_their_bound(
    tmp_path: Path,
) -> None:
    config = _config(
        deadline_s=600,
        analyses={
            "da": {"settings": {"nturns": 128, "n_angles": 9, "n_bisect": 8}, "deadline_s": 1200},
            "footprint": {"enabled": False},
        },
    )

    [model] = load_settings(config, tmp_path).models

    assert "footprint" not in model.offered()
    da = model.setups["da"]
    assert (da.settings["nturns"], da.settings["n_angles"], da.settings["n_bisect"]) == (128, 9, 8)
    # A setting the facility does not name keeps its default.
    assert da.settings["amp_max_mm"] == ANALYSES["da"].settings["amp_max_mm"]
    assert da.deadline_s == 1200
    assert model.setups["optics"].deadline_s == 600


def test_a_model_overrides_the_shared_block(tmp_path: Path) -> None:
    config = {
        "analyses": {"lma": {"settings": {"nturns": 256}}},
        "models": [
            {"name": "sr", "deck": "sr.lte"},
            {
                "name": "booster",
                "deck": "/abs/booster.lte",
                "energies_gev": [0.425, 6],
                "analyses": {"lma": {"enabled": False}, "da": {"settings": {"nturns": 64}}},
            },
        ],
    }

    sr, booster = load_settings(config, tmp_path).models

    assert sr.setups["lma"].settings["nturns"] == 256
    assert "lma" not in booster.offered()
    assert booster.setups["da"].settings["nturns"] == 64
    assert booster.deck == Path("/abs/booster.lte")
    assert booster.energies_gev == (0.425, 6.0)


def test_an_engine_offers_only_what_it_computes(tmp_path: Path) -> None:
    config = {
        "engines": {
            "aps": {"command": ["python", "-m", "aps.engine"], "analyses": ["optics", "lma"]}
        },
        "models": [
            {"name": "a", "deck": "a.lte", "engine": "aps"},
            {"name": "b", "deck": "b.lte", "engine": "elegant"},
        ],
    }

    a, b = load_settings(config, tmp_path).models

    assert a.engine.command == ("python", "-m", "aps.engine")
    assert a.offered() == ["optics", "lma"]
    assert b.offered() == ["optics", "da", "lma"]


def test_no_models_block_offers_no_model_in_a_bare_directory(tmp_path: Path) -> None:
    assert load_settings({}, tmp_path).models == ()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"models": {}}, "models is not a list"),
        ({"models": [{"deck": "a.lte"}]}, "models[0].name is missing"),
        ({"models": [{"name": "a"}]}, "models[0].deck is missing"),
        (
            {"models": [{"name": "a", "deck": "a.lte", "engine": "mad"}]},
            "models[0].engine names no engine mad",
        ),
        (
            {"models": [{"name": "a", "deck": "a"}, {"name": "a", "deck": "b"}]},
            "models names a more than once",
        ),
        (
            {"models": [{"name": "a", "deck": "a", "energies_gev": [0]}]},
            "models[0].energies_gev is not a list of positive energies",
        ),
        (_config(analyses={"orbit": {}}), "analyses names no analysis orbit"),
        (_config(deadline_s=0), "deadline_s is not a positive number of seconds"),
        (_config(max_workers=0), "max_workers is not a positive whole number"),
        (_config(engines={"x": {"command": "run"}}), "engines.x.command is not a list"),
    ],
)
def test_a_config_the_application_cannot_run_is_named(
    tmp_path: Path, config: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_settings(config, tmp_path)
