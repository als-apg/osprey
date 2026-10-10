"""Spans place each device of the two-model demo from its own model's deck.

The built control-assistant demo holds two deck-bearing models: the periodic
``SR`` and the single-pass transfer line ``LINE``. Every LINE device takes its
s from the LINE deck and its place ``LINE`` from the LINE span; no device is
placed by a span of a model other than the one that gave its s, though the
two models' s ranges overlap. No group, and no middle-layer Family derived
from the build, holds devices or channels of both models.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

pytestmark = [pytest.mark.slow]

#: The demo's models that have a deck.
MODELS = ("LINE", "SR")


def _model_of(record_id: str) -> str | None:
    """The model a demo device id or channel address names, by its leading segment."""
    for model in MODELS:
        if record_id.startswith((f"{model}/", f"{model}:")):
            return model
    return None


def _span_of(place: str, places: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The span of ``place`` or of its deepest ancestor that has one."""
    parts = place.split("/")
    for depth in range(len(parts), 0, -1):
        span = places.get("/".join(parts[:depth]), {}).get("span")
        if span:
            return span
    return None


def _models(members: Iterable[str]) -> set[str | None]:
    return {_model_of(member) for member in members}


def _cells(node: Mapping[str, Any]) -> Iterator[tuple[str, list[str]]]:
    """Every (Field, ChannelNames) of a middle-layer Family node."""
    for key, child in node.items():
        if key.startswith("_") or not isinstance(child, Mapping):
            continue
        if "ChannelNames" in child:
            yield key, list(child["ChannelNames"])
        else:
            yield from _cells(child)


@pytest.fixture(scope="module")
def facility(built_control_assistant: BuiltProject) -> dict[str, Any]:
    return built_control_assistant.facility


@pytest.fixture(scope="module")
def middle_layer(built_control_assistant: BuiltProject) -> dict[str, Any]:
    from osprey.facility.views.channel_finder import middle_layer_document

    document, _left_out, _keyed = middle_layer_document(built_control_assistant.facility)
    return document


def test_every_line_device_is_placed_by_the_line_span(facility: dict[str, Any]) -> None:
    line = [device for device in facility["devices"] if _model_of(device["id"]) == "LINE"]
    assert len(line) == 20
    assert {
        (device["model"], device["place"], device["provenance"]["place_from"]) for device in line
    } == {("LINE", "LINE", "span")}
    assert all(isinstance(device["s"], float) for device in line)


def test_the_models_s_ranges_overlap(facility: dict[str, Any]) -> None:
    s = {
        model: [device["s"] for device in facility["devices"] if device.get("model") == model]
        for model in MODELS
    }
    assert min(s["SR"]) < max(s["LINE"])


def test_every_span_placed_device_lies_in_a_span_of_its_own_model(
    facility: dict[str, Any],
) -> None:
    places = {place["id"]: place for place in facility["places"]}
    placed = [
        device for device in facility["devices"] if device["provenance"].get("place_from") == "span"
    ]
    crossed = [
        device["id"]
        for device in placed
        if (_span_of(device["place"], places) or {}).get("model") != device["model"]
    ]
    assert {device["model"] for device in placed} == set(MODELS)
    assert crossed == []


def test_no_device_of_one_model_sits_in_a_place_of_the_other(facility: dict[str, Any]) -> None:
    crossed = [
        device["id"]
        for device in facility["devices"]
        if device.get("model") and _model_of(f"{device['place']}/") not in (device["model"], None)
    ]
    assert crossed == []


def test_no_group_mixes_line_and_sr_devices(facility: dict[str, Any]) -> None:
    mixed = [
        group["id"]
        for group in facility["groups"]
        if {"LINE", "SR"} <= _models(group.get("members", []))
    ]
    line = sorted(
        group["id"] for group in facility["groups"] if _models(group["members"]) == {"LINE"}
    )
    assert mixed == []
    assert line == ["LINE/BPM", "LINE/HCM", "LINE/VCM"]


def test_no_middle_layer_family_mixes_line_and_sr_channels(
    middle_layer: dict[str, Any],
) -> None:
    mixed = [
        (system, family, field)
        for system, families in middle_layer.items()
        if system != "schema"
        for family, node in families.items()
        if not family.startswith("_")
        for field, addresses in _cells(node)
        if {"LINE", "SR"} <= _models(addresses)
    ]
    line_cells = [
        (system, family)
        for system, families in middle_layer.items()
        if system != "schema"
        for family, node in families.items()
        if not family.startswith("_")
        for _field, addresses in _cells(node)
        if "LINE" in _models(addresses)
    ]
    assert mixed == []
    assert line_cells
    assert {system for system, _family in line_cells} == {"LINE"}


def test_the_line_device_list_counts_its_own_devices(middle_layer: dict[str, Any]) -> None:
    setups = {
        family: node["_setup"]
        for family, node in middle_layer["LINE"].items()
        if not family.startswith("_")
    }
    assert sorted(setups) == ["BPM", "HCM", "Quadrupole", "VCM"]
    assert all(
        common.startswith("LINE ") for setup in setups.values() for common in setup["CommonNames"]
    )
    assert sum(len(setup["DeviceList"]) for setup in setups.values()) == 20
