"""``web.tile_bar.controls`` refused at build when the tile bar cannot draw it.

:func:`~osprey.cli.build_profile_tile_bar.tile_bar_control_errors` judges every
spelling of the key in a profile's merged ``config:`` block against the ids
the terminal draws (:mod:`osprey.interfaces.web_terminal.tile_bar`).
:meth:`~osprey.cli.build_profile.BuildProfile.validate` reports the result
with every other fault of the profile, so ``osprey build`` and ``osprey
validate`` stop on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_profile import BuildProfile
from osprey.cli.build_profile_tile_bar import tile_bar_control_errors
from osprey_connectors.errors import BuildProfileError

#: The ids every refusal names as the ones to use.
_ALLOWED = "ask_agent, reload, help"


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({}, id="absent"),
        pytest.param({"web.tile_bar.controls": None}, id="no_value"),
        pytest.param({"web.tile_bar.controls": []}, id="empty"),
        pytest.param({"web.tile_bar.controls": ["help", "reload", "ask_agent"]}, id="every_id"),
        pytest.param({"web.tile_bar.controls": ["reload", "reload"]}, id="repeated_id"),
        pytest.param(None, id="config_not_a_mapping"),
    ],
)
def test_a_value_the_bar_can_draw_passes(config) -> None:
    assert tile_bar_control_errors(config) == []


def test_an_unknown_id_is_refused_with_the_ids_to_use() -> None:
    (error,) = tile_bar_control_errors({"web.tile_bar.controls": ["ask_agent", "relaod"]})
    assert error.startswith("`web.tile_bar.controls` names 'relaod',")
    assert _ALLOWED in error
    assert "'ask_agent'" not in error, "a good id is not named as a bad one"


def test_every_unknown_id_is_named_in_one_message() -> None:
    (error,) = tile_bar_control_errors({"web.tile_bar.controls": ["agent", "refresh"]})
    assert "names 'agent', 'refresh'," in error


@pytest.mark.parametrize("value", ["reload", True, {"reload": True}])
def test_a_value_that_is_not_a_list_is_refused(value) -> None:
    (error,) = tile_bar_control_errors({"web.tile_bar.controls": value})
    assert f"is {value!r}, not a list." in error
    assert _ALLOWED in error


@pytest.mark.parametrize(
    ("config", "written"),
    [
        pytest.param({"web.tile_bar.controls": ["relaod"]}, "web.tile_bar.controls", id="dotted"),
        pytest.param(
            {"web.tile_bar": {"controls": ["relaod"]}},
            "web.tile_bar: controls",
            id="prefix_over_mapping",
        ),
        pytest.param(
            {"web": {"tile_bar": {"controls": ["relaod"]}}}, "web: tile_bar: controls", id="nested"
        ),
    ],
)
def test_every_spelling_reaches_the_check_and_is_named(config, written) -> None:
    (error,) = tile_bar_control_errors(config)
    assert error.startswith(f"`{written}` names 'relaod',")


@pytest.mark.usefixtures("facility_data_tree")
def test_validate_refuses_an_unknown_id(tmp_path: Path) -> None:
    profile = BuildProfile(
        name="x", data="data", config={"web.tile_bar.controls": ["ask_agent", "relaod"]}
    )
    with pytest.raises(BuildProfileError) as excinfo:
        profile.validate(tmp_path)
    message = str(excinfo.value)
    assert message.startswith("Build profile validation failed:")
    assert "`web.tile_bar.controls` names 'relaod'," in message
    assert _ALLOWED in message


@pytest.mark.usefixtures("facility_data_tree")
def test_validate_accepts_ids_the_bar_draws(tmp_path: Path) -> None:
    profile = BuildProfile(name="x", data="data", config={"web.tile_bar.controls": ["reload"]})
    profile.validate(tmp_path)
