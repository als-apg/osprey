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
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from osprey.facility import kinds
from osprey.facility.errors import FacilityBuildError
from osprey.facility.render import render_facility_outputs
from osprey.facility.views.facts import FACTS_FILE, hook_measurement
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors, with_rf

KIND = "turn_kind"


class TurnBlock(BaseModel):
    """The kind's own sub-block: a trigger sample and an ordered chain of step ids."""

    model_config = ConfigDict(extra="forbid")

    trigger_sample: int = Field(ge=0)
    steps: list[str] = []


def _stops(document: Mapping[str, Any], block: Any) -> Iterator[kinds.KindStop]:
    """Refuse a step named ``refuse``; record what the hook was handed."""
    del document
    HANDED.append(block)
    for step in block.get("steps") or []:
        if step == "refuse":
            yield kinds.KindStop("step refuse is refused", "drop step refuse")


def _facts_writer(document: Mapping[str, Any], block: Mapping[str, Any]) -> dict[str, Any]:
    del document
    return {"trigger_sample": block["trigger_sample"], "chain": list(block.get("steps") or [])}


#: Every block the kind's ``build_stops`` hook was handed, in order.
HANDED: list[Any] = []

BLOCK = {"trigger_sample": 3, "steps": ["arm", "fire"]}

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
    measurement_schema=TurnBlock,
    build_stops=_stops,
    facts_writer=_facts_writer,
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
    HANDED.clear()


def _tree(**sr: Any) -> dict[str, Any]:
    """SR with correctors and the RF; its file lists the kind and names its roles."""
    tree = with_correctors(with_rf(measured_tree()))
    measurement = tree["measurement/SR.yaml"]
    measurement["kinds"] = [KIND]
    measurement["groups"] |= {"turn_bpm": "SR/BPM", "kick": "SR/Q"}
    measurement["instruments"] |= {"inject": "SR:TUNE:Y"}
    measurement["kind_blocks"] = {KIND: dict(BLOCK)}
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


# --- the kind's own sub-block ------------------------------------------------------


def _sr_measurement(document: dict[str, Any]) -> dict[str, Any]:
    (sr,) = [model for model in document["models"] if model["name"] == "SR"]
    measurement: dict[str, Any] = sr["measurement"]
    return measurement


@_with_kind
def test_the_sub_block_reaches_the_build_stops_hook_and_the_per_model_block(tmp_path):
    facts = _facts(tmp_path, _tree())

    assert HANDED == [BLOCK]
    assert facts["measurement_models"]["SR"]["kinds"][KIND]["block"] == BLOCK
    assert "block" not in facts["measurement_models"]["SR"]["kinds"].get("trm", {})


@_with_kind
def test_a_block_the_kinds_schema_refuses_stops_the_build_before_its_hook(tmp_path):
    from osprey.facility.validate import _measurement_stops

    document, _ = built_document(tmp_path / "tree", _tree())
    HANDED.clear()
    _sr_measurement(document)["kind_blocks"][KIND] = {"trigger_sample": -1, "steps": ["refuse"]}

    (stop,) = _measurement_stops(document)
    assert stop.detail == (
        f"`kind_blocks.{KIND}.trigger_sample: Input should be greater than or equal to 0`"
    )
    assert stop.remedy == f"correct `kind_blocks.{KIND}` as kind {KIND}'s measurement schema states"
    assert HANDED == []


@_with_kind
def test_a_block_the_schema_admits_is_judged_by_the_kinds_hook(tmp_path):
    tree = _tree(kind_blocks={KIND: {"trigger_sample": 3, "steps": ["refuse"]}})

    assert _stop(tmp_path, tree) == (
        f"facility: value-invalid: measurement SR — kind {KIND}: step refuse is refused; "
        "fix: drop step refuse"
    )


def test_a_facts_writer_that_fails_on_an_admitted_block_stops_the_build(tmp_path, monkeypatch):
    def failing(document: Mapping[str, Any], block: Mapping[str, Any]) -> dict[str, Any]:
        del document, block
        raise KeyError("chain")

    _register(monkeypatch, dataclasses.replace(DECLARATION, facts_writer=failing))

    assert _stop(tmp_path, _tree()) == (
        f"facility: value-invalid: measurement SR — kind {KIND}'s facts writer fails on "
        "this file's block: KeyError: 'chain'; fix: make the facts writer of kind "
        f"{KIND} (package turn-kind) accept every block its measurement schema admits"
    )


def _raising_stops(document: Mapping[str, Any], block: Any) -> Iterator[kinds.KindStop]:
    del document
    yield kinds.KindStop("first", "first")
    raise KeyError(block["absent"])


@pytest.mark.parametrize(
    ("hook", "cause"),
    [
        pytest.param(_raising_stops, "KeyError: 'absent'", id="raises"),
        pytest.param(
            lambda document, block: ["not a stop"],
            "AttributeError: 'str' object has no attribute 'remedy'",
            id="yields-no-stop",
        ),
    ],
)
def test_a_build_stops_hook_that_fails_stops_the_build_with_one_line(
    tmp_path, monkeypatch, hook, cause
):
    _register(monkeypatch, dataclasses.replace(DECLARATION, build_stops=hook))

    assert _stop(tmp_path, _tree()) == (
        f"facility: value-invalid: measurement SR — kind {KIND}'s build_stops hook fails on "
        f"this file's block: {cause}; fix: make the build_stops hook of kind {KIND} "
        "(package turn-kind) accept every block its measurement schema admits"
    )


@_with_kind
def test_an_absent_block_is_judged_as_an_empty_one(tmp_path):
    tree = _tree()
    del tree["measurement/SR.yaml"]["kind_blocks"]

    assert _stop(tmp_path, tree) == (
        f"facility: value-invalid: measurement SR — `kind_blocks.{KIND}.trigger_sample: "
        f"Field required`; fix: correct `kind_blocks.{KIND}` as kind {KIND}'s measurement "
        "schema states"
    )


@_with_kind
def test_a_block_of_a_kind_the_file_does_not_list_stops_the_build(tmp_path):
    tree = _tree()
    tree["measurement/SR.yaml"]["kind_blocks"]["orm"] = {}

    assert _stop(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — `kind_blocks.orm` is the block of a kind "
        "the file does not list; fix: remove `kind_blocks.orm`, or list orm in `kinds`"
    )


@_with_kind
def test_a_block_of_a_kind_with_no_schema_stops_the_build(tmp_path):
    tree = _tree(kinds=[KIND, "trm"])
    tree["measurement/SR.yaml"]["kind_blocks"]["trm"] = {"anything": 1}

    assert _stop(tmp_path, tree) == (
        "facility: value-invalid: measurement SR — kind trm declares no measurement schema, "
        "so it takes no block; fix: remove `kind_blocks.trm`"
    )


@_with_kind
def test_kind_blocks_is_a_mapping(tmp_path):
    tree = _tree(kind_blocks=[KIND])

    assert _stop(tmp_path, tree) == (
        "facility: source-invalid: path models.1.measurement.kind_blocks — "
        "measurement/SR.yaml `kind_blocks` is not a mapping of kind to a mapping; fix: write "
        "`kind_blocks` in measurement/SR.yaml as a mapping of kind to a mapping"
    )
