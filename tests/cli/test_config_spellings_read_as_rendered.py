"""A ``config:`` leaf stated in more than one spelling reads as the render writes it.

A profile's ``config:`` block can reach one leaf dotted (``simulation.tick_s``)
and nested (``simulation: {tick_s: ...}``), or through a dotted prefix holding a
mapping. The render applies the keys in order, so the spelling written last in
key order is the value it writes. The build's value checks read the block
through ``_expand_dotted``, which keeps that same spelling for every leaf, while
every path any spelling addresses stays present in the tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_load import _parse_profile
from osprey.cli.build_profile_merge import resolve_profile_document
from osprey.cli.build_profile_presets import _load_preset_raw, list_tiers
from osprey.facility.errors import FacilityBuildError
from osprey.utils.config_writer import config_read, config_update_fields

_TICK_ZERO_LINE = (
    "facility: profile-invalid: path simulation.tick_s — simulation.tick_s must be a "
    "number of seconds greater than 0 (got 0); fix: set `simulation.tick_s` to a "
    "number of seconds greater than 0"
)


def _walk(tree: Any, dotted: str) -> Any:
    node = tree
    for part in dotted.split("."):
        node = node[part]
    return node


@pytest.mark.parametrize(
    ("config", "dotted"),
    [
        pytest.param(
            {"simulation": {"tick_s": 0.5}, "simulation.tick_s": 0},
            "simulation.tick_s",
            id="nested-then-dotted",
        ),
        pytest.param(
            {"simulation.tick_s": 0, "simulation": {"tick_s": 0.5}},
            "simulation.tick_s",
            id="dotted-then-nested",
        ),
        pytest.param({"a.b": {"c": 1}, "a.b.c": 2}, "a.b.c", id="prefix-then-dotted"),
        pytest.param({"a.b.c": 2, "a.b": {"c": 1}}, "a.b.c", id="dotted-then-prefix"),
        pytest.param({"a": {"b": {"c": [1]}}, "a.b.c": [2]}, "a.b.c", id="nested-list-then-dotted"),
    ],
)
def test_every_leaf_stated_twice_reads_what_the_render_writes(
    tmp_path, config: dict[str, Any], dotted: str
) -> None:
    path = tmp_path / "config.yml"
    path.write_text("{}\n")
    config_update_fields(path, config)
    rendered = _walk(config_read(path), dotted)
    assert _walk(_expand_dotted(config), dotted) == rendered


def test_a_zero_tick_stated_after_a_valid_nested_tick_stops_the_profile() -> None:
    with pytest.raises(FacilityBuildError) as caught:
        _parse_profile({"config": {"simulation": {"tick_s": 0.5}, "simulation.tick_s": 0}})
    assert str(caught.value) == _TICK_ZERO_LINE


def test_a_valid_tick_stated_after_a_zero_one_parses() -> None:
    _parse_profile({"config": {"simulation.tick_s": 0, "simulation": {"tick_s": 0.5}}})


def test_a_start_set_stated_twice_is_checked_as_the_render_keeps_it() -> None:
    with pytest.raises(FacilityBuildError) as caught:
        _parse_profile(
            {
                "config": {
                    "simulation": {"default_scenarios": ["a"]},
                    "simulation.default_scenarios": "a",
                }
            }
        )
    assert str(caught.value).startswith(
        "facility: profile-invalid: path simulation.default_scenarios — "
        "simulation.default_scenarios must be a list of scenario names, got 'a'"
    )


@pytest.mark.parametrize(
    "layer",
    [f"tiers/{name}" for name in list_tiers()],
)
def test_a_child_preset_that_turns_the_web_terminals_off_reads_off(
    layer: str, tmp_path: Path
) -> None:
    """Every tier and app render is attached: its web-terminal module reads off
    although the host spells ``modules.web_terminals`` as a dotted prefix
    holding a mapping with ``enabled: true``."""
    host, _host_path = _load_preset_raw("control-assistant")
    assert _expand_dotted(host["config"])["modules"]["web_terminals"]["enabled"] is True
    (tmp_path / "profile.yml").write_text(yaml.safe_dump(host), encoding="utf-8")
    raw, _path = _load_preset_raw(layer)
    layer_path = tmp_path / f"{layer}.yml"
    layer_path.parent.mkdir()
    layer_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    document = resolve_profile_document(dict(raw), layer_path, warn=False)

    assert document.is_layer
    assert _expand_dotted(document.raw["config"])["modules"]["web_terminals"]["enabled"] is False


def test_a_path_one_spelling_states_stays_addressed_when_a_later_mapping_omits_it() -> None:
    simulation = _expand_dotted(
        {"simulation.default_scenarios": ["x"], "simulation": {"tick_s": 1.0}}
    )["simulation"]
    assert simulation["default_scenarios"] == ["x"]
    assert simulation["tick_s"] == 1.0
