"""A registered kind's members and its own sub-block in the measurement file.

Every member role is named once, under ``groups:`` or ``instruments:``, whoever
declared it: the roles the pyAML view ships and the roles a listed kind
declares alike, each in the map its sort says. A kind's own sub-block,
``kind_blocks.<kind>``, holds only what is the kind's own and is validated by
the pydantic model the kind declares, before its ``build_stops`` hook runs.

The kind here is a declaration put into the registry for one test: a monitor
group, a stepped setpoint group and two instruments of its own, beside the
shipped corrector roles.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from osprey.facility import kinds
from osprey.facility.errors import FacilityBuildError
from osprey.facility.render import render_facility_outputs
from osprey.facility.views.facts import FACTS_FILE, hook_measurement
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors, with_rf

pytest.importorskip("pyaml")

KIND = "turn_kind"

#: The kind's own roles and their sorts.
OWN_SORTS = {
    "turn_bpm": "monitors",
    "kick": "setpoints",
    "inject": "instrument",
    "status": "instrument",
}

DECLARATION = dataclasses.replace(
    kinds.ORM,
    name=KIND,
    tool=None,
    view_section=None,
    members={
        "turn_bpm": "wired",
        "kick": "wired",
        "hcor": "wired",
        "inject": "exists",
        "status": "optional",
    },
    member_sorts=OWN_SORTS,
    stepped_roles=("kick",),
    corrector_roles=(),
    tool_keys=(),
    single_pass=False,
    description="a kind with roles of its own",
)


def _register(monkeypatch: pytest.MonkeyPatch, *declarations: kinds.KindDeclaration) -> None:
    """Every built-in kind and ``declarations``, as a deployment package would register them."""
    built_in = kinds.builtin_kinds()
    found = {**built_in, **{d.name: d for d in declarations}}
    sources = {
        **dict.fromkeys(built_in, "built-in"),
        **{d.name: "package turn-kind" for d in declarations},
    }
    monkeypatch.setattr(kinds, "_load", lambda: (found, sources))


@pytest.fixture
def turn_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kind registered for the test's duration."""
    _register(monkeypatch, DECLARATION)


def _tree(**sr: Any) -> dict[str, Any]:
    """SR with correctors and the RF; its file lists the kind and names its roles."""
    tree = with_correctors(with_rf(measured_tree()))
    measurement = tree["measurement/SR.yaml"]
    measurement["kinds"] = [KIND]
    measurement["groups"] |= {"turn_bpm": "SR/BPM", "kick": "SR/Q"}
    measurement["instruments"] |= {"inject": "SR:TUNE:Y"}
    measurement |= sr
    return tree


def _stop(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path / "tree", tree)
    return str(stopped.value.format_message())


def _facts(tmp_path: Path, tree: dict[str, Any]) -> dict[str, Any]:
    document, facility = built_document(tmp_path / "tree", tree)
    render = tmp_path / "render"
    render.mkdir()
    render_facility_outputs(render, document, {}, facility)
    loaded: dict[str, Any] = json.loads((render / "data" / FACTS_FILE).read_text("utf-8"))
    return loaded


# --- the declaration ---------------------------------------------------------------


def test_a_member_the_view_does_not_ship_states_its_sort():
    unsorted = dataclasses.replace(DECLARATION, member_sorts={})

    with pytest.raises(kinds.KindRegistryError) as refused:
        kinds._declaration(unsorted, KIND, "package turn-kind")

    assert str(refused.value) == (
        f"measurement kind {KIND} (package turn-kind): member turn_bpm states no sort; "
        "fix: state member_sorts['turn_bpm'] as one of monitors, setpoints, instrument"
    )


def test_a_shipped_member_keeps_the_sort_the_view_gives_it():
    resorted = dataclasses.replace(DECLARATION, member_sorts={**OWN_SORTS, "hcor": "instrument"})

    with pytest.raises(kinds.KindRegistryError, match="the pyAML view names hcor setpoints"):
        kinds._declaration(resorted, KIND, "package turn-kind")


def test_the_sorts_are_the_shipped_ones_and_each_listed_kind_s_own():
    assert kinds.member_sorts([]) == dict(kinds.SHIPPED_SORTS)
    assert kinds.member_sorts([DECLARATION]) == {**kinds.SHIPPED_SORTS, **OWN_SORTS}


# --- the measurement file's maps ---------------------------------------------------

_with_kind = pytest.mark.usefixtures("turn_kind")


@_with_kind
def test_a_role_a_listed_kind_declares_is_named_in_groups_and_instruments(tmp_path):
    facts = _facts(tmp_path, _tree())

    members = facts["measurement_models"]["SR"]["kinds"][KIND]["members"]
    assert members["turn_bpm"] == {"requirement": "wired", "addresses": ["BPM1:X", "BPM1:Y"]}
    assert members["kick"] == {"requirement": "wired", "addresses": ["QD:SP", "QF:SP"]}
    assert members["inject"] == {"requirement": "exists", "addresses": ["SR:TUNE:Y"]}
    assert members["status"] == {"requirement": "optional", "addresses": []}


@_with_kind
def test_the_approval_block_lists_the_addresses_of_every_stepped_role(tmp_path):
    block = hook_measurement(_facts(tmp_path, _tree()), tmp_path)["SR"]

    assert block["kinds"] == {KIND: {"kick": "kick"}}
    assert block["groups"]["kick"] == ["QD:SP", "QF:SP"]
    assert "turn_bpm" not in block["groups"]


def test_the_approval_block_lists_the_channel_of_a_stepped_instrument_role(tmp_path, monkeypatch):
    _register(monkeypatch, dataclasses.replace(DECLARATION, stepped_roles=("inject", "kick")))
    HANDED.clear()

    block = hook_measurement(_facts(tmp_path, _tree()), tmp_path)["SR"]

    assert block["kinds"] == {KIND: {"inject": "inject", "kick": "kick"}}
    assert block["groups"]["inject"] == ["SR:TUNE:Y"]
    assert block["groups"]["kick"] == ["QD:SP", "QF:SP"]
    assert "status" not in block["groups"]


@_with_kind
def test_a_role_no_listed_kind_declares_stops_the_build(tmp_path):
    tree = _tree(kinds=["trm"])

    assert _stop(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — `groups.kick` names a role no listed kind "
        "declares; fix: remove `groups.kick`, or list a kind that "
        "declares kick in `kinds`"
    )


@_with_kind
def test_a_role_named_in_the_other_map_stops_the_build(tmp_path):
    tree = _tree()
    tree["measurement/SR.yaml"]["groups"]["status"] = "SR/BPM"

    assert _stop(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — `groups.status` names a role whose kind "
        "names it under `instruments`; fix: move `groups.status` to `instruments.status`"
    )


@_with_kind
def test_a_wired_monitor_role_holding_no_monitor_stops_the_build(tmp_path):
    tree = _tree()
    tree["measurement/SR.yaml"]["groups"]["turn_bpm"] = "SR/Q"

    assert _stop(tmp_path, tree) == (
        f"facility: reference-missing: measurement SR — kind {KIND} needs `groups.turn_bpm`; "
        "group SR/Q holds no monitor model SR wires; fix: name a "
        "group holding a monitor model SR wires as `groups.turn_bpm`"
    )


@_with_kind
def test_an_instrument_named_as_a_list_stops_the_build(tmp_path):
    tree = _tree()
    tree["measurement/SR.yaml"]["instruments"]["inject"] = ["SR:TUNE:Y"]

    assert _stop(tmp_path, tree) == (
        "facility: source-invalid: path models.1.measurement.instruments.inject — "
        "measurement/SR.yaml `instruments.inject` is not a channel address; fix: write `instruments.inject` in measurement/SR.yaml "
        "as a channel address"
    )
