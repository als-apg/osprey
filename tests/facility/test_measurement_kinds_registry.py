"""The measurement-kind registry: one entry-point group, built-in and deployment kinds alike.

The five built-in kinds are entry points of the ``osprey-framework``
distribution; a deployment package registers its own kind the same way. The
fixture package under ``fixtures/registry_kind/`` is put on ``sys.path`` with
its ``dist-info`` for each case that needs it, the shape an installed wheel
leaves. Every case reads what the build wrote: the per-model kind block in the
facts, the deployment-wide ``measurement_kinds`` table, the approval hook's
block, the measure tool's argument resolution and its advertised schema.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from osprey.facility import kinds
from osprey.facility.errors import FacilityBuildError
from osprey.facility.render import render_facility_outputs
from osprey.facility.views.facts import FACTS_FILE, hook_measurement
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors, with_rf
from tests.facility.fixtures.registry_kind import REGISTRY_KIND, install

pytest.importorskip("pyaml")

BUILT_INS = ("chromaticity_monitor", "crm", "dispersion", "orm", "trm")
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


@pytest.fixture
def registry_kind(monkeypatch: pytest.MonkeyPatch) -> kinds.KindDeclaration:
    """The fixture package installed for this test; its declaration."""
    install(monkeypatch)
    return kinds.registered()[REGISTRY_KIND]


def _tree(**sr: Any) -> dict[str, Any]:
    """SR with correctors, the RF and the fixture kind listed; ``sr`` edits its file."""
    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"] |= {"kinds": [REGISTRY_KIND], "corrector_delta": 1.0e-5, **sr}
    return tree


def _render(tmp_path: Path, tree: dict[str, Any]) -> Path:
    document, facility = built_document(tmp_path / "tree", tree)
    render = tmp_path / "render"
    render.mkdir()
    render_facility_outputs(render, document, {}, facility)
    (render / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    return render


def _facts(render: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((render / "data" / FACTS_FILE).read_text("utf-8"))
    return loaded


def _stop(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path / "tree", tree)
    return str(stopped.value.format_message())


# --- the registry ------------------------------------------------------------------


def test_the_built_in_kinds_are_entry_points_of_the_framework_distribution():
    group = tomllib.loads(PYPROJECT.read_text("utf-8"))["project"]["entry-points"][
        kinds.ENTRY_POINT_GROUP
    ]

    assert sorted(group) == list(BUILT_INS)
    assert tuple(kinds.builtin_kinds()) == BUILT_INS
    assert all(kinds.kind_source(name) == "built-in" for name in BUILT_INS)


def test_every_built_in_plan_is_one_the_script_runs():
    for name, declaration in kinds.builtin_kinds().items():
        assert kinds.plan_problems(declaration.measure_plan) == [], name


def test_a_package_registers_a_kind_through_the_entry_point_group(registry_kind):
    assert registry_kind.name == REGISTRY_KIND
    assert kinds.kind_source(REGISTRY_KIND) == "package registry-kind-fixture"
    assert REGISTRY_KIND not in kinds.builtin_kinds()
    assert list(kinds.registered()) == [*BUILT_INS, REGISTRY_KIND]


def test_a_kind_named_like_a_built_in_stops_the_build_naming_both_sources(tmp_path, monkeypatch):
    install(monkeypatch, "collision")

    assert _stop(tmp_path, measured_tree()) == (
        "facility: value-invalid: measurement kind orm — built-in declares a kind of the "
        "same name; declared by package registry-kind-collision; fix: give the deployment's "
        "kind a name no other kind has"
    )


def test_a_kind_naming_a_hook_osprey_does_not_offer_stops_the_build(tmp_path, monkeypatch):
    install(monkeypatch, "unknown_hook")

    message = _stop(tmp_path, measured_tree())

    assert message.startswith(
        "facility: value-invalid: measurement kind hooked_kind — it names hook 'on_measure', "
        "which OSPREY does not offer; declared by package registry-kind-hooks; "
        "fix: name only these: name, tool,"
    )


def test_a_plan_word_osprey_does_not_offer_is_refused():
    plan = {**kinds.ORM.measure_plan, "fit": "sweep", "replay": True}

    problems = kinds.plan_problems(plan)

    assert "its measure plan names 'replay', which OSPREY does not offer" in problems
    assert "its measure plan fits by 'sweep', not one of magnet-group, rf, monitor" in problems


def test_a_measurement_file_naming_an_unregistered_kind_stops_the_build(tmp_path):
    message = _stop(tmp_path, _tree())

    assert message == (
        "facility: reference-missing: measurement SR — kind registry_kind is not registered "
        "under osprey.facility.kinds by any installed package; fix: install the kind's "
        "package, or remove registry_kind from `kinds`"
    )


# --- members, single pass and the stepped roles ------------------------------------


def test_members_are_judged_by_their_requirement(tmp_path, registry_kind):
    """``wired`` must be wired, ``exists`` only present, ``optional`` may be absent."""
    assert registry_kind.members["tune"] == "exists"
    assert registry_kind.members["rf"] == "optional"
    unwired = _tree(instruments={"tune": "LQ:SP"})
    no_rf = _tree(instruments={"tune": "SR:TUNE:X"})
    no_hcor = _tree()
    del no_hcor["measurement/SR.yaml"]["groups"]["hcor"]

    built_document(tmp_path / "unwired", unwired)
    built_document(tmp_path / "no-rf", no_rf)
    assert _stop(tmp_path / "no-hcor", no_hcor) == (
        "facility: reference-missing: measurement SR — kind registry_kind needs `groups.hcor`, "
        "which the file does not name; fix: name `groups.hcor`, or remove registry_kind from "
        "`kinds`"
    )


def test_a_single_pass_kind_runs_on_a_single_pass_model(tmp_path, registry_kind):
    assert registry_kind.single_pass
    tree = measured_tree()
    tree["measurement/LINE.yaml"] |= {
        "kinds": ["orm", REGISTRY_KIND],
        "instruments": {"tune": "SR:TUNE:X"},
    }
    document, _ = built_document(tmp_path / "ok", tree)
    (line,) = [model for model in document["models"] if model["name"] == "LINE"]
    assert line["measurement"]["kinds"] == ["orm", REGISTRY_KIND]

    tree["measurement/LINE.yaml"] |= {"kinds": ["trm"], "quad_delta": 0.001}
    assert _stop(tmp_path / "stop", tree).endswith(
        "fix: remove trm from `kinds`; a single_pass model allows orm and registry_kind alone"
    )


def test_the_per_model_block_carries_the_declared_argument_set(tmp_path, registry_kind):
    facts = _facts(_render(tmp_path, _tree()))
    block = facts["measurement_models"]["SR"]["kinds"][REGISTRY_KIND]

    assert list(facts["measurement_models"]["SR"]["kinds"]) == [REGISTRY_KIND]
    assert block["stepped_roles"] == ["hcor", "vcor"]
    assert block["corrector_roles"] == ["hcor", "vcor"]
    assert (block["matrix"], block["takes_by"], block["single_pass"]) == (True, False, True)
    assert block["tool"] == {"type": registry_kind.tool.type, "name": registry_kind.tool.name}
    assert block["view_files"] == []
    assert block["measure_plan"]["result"] == "matrix"
    assert block["members"]["hcor"] == {"requirement": "wired", "addresses": ["SCOR:H:SP"]}
    assert block["members"]["rf"] == {"requirement": "optional", "addresses": ["RF:SP"]}
    assert facts["measurement_kinds"][REGISTRY_KIND] == kinds.kind_table_row(registry_kind)


def test_tool_keys_round_trip_through_the_sdk(tmp_path, registry_kind):
    from osprey.apps import sdk

    facts = sdk.read_facts(_render(tmp_path, _tree()))

    assert sdk.tool_keys(facts, "SR", REGISTRY_KIND) == list(registry_kind.tool_keys)


def test_the_kinds_tool_section_is_its_view_section(tmp_path, registry_kind):
    import yaml

    render = _render(tmp_path, _tree())
    configuration = yaml.safe_load(
        (render / "data" / "pyaml" / "SR" / "configuration.yaml").read_text("utf-8")
    )
    (section,) = [d for d in configuration["devices"] if d["name"] == registry_kind.tool.name]

    assert section == {
        "type": registry_kind.tool.type,
        "name": registry_kind.tool.name,
        "bpm_array_name": "SR_BPM",
        "hcorr_array_name": "SR_HCM",
        "vcorr_array_name": "SR_VCM",
        "corrector_delta": 1.0e-5,
    }


# --- no row ------------------------------------------------------------------------


def _declared(monkeypatch: pytest.MonkeyPatch, **changes: Any) -> None:
    """The registry with the fixture kind changed by ``changes``."""
    declared = {**kinds.registered()}
    declared[REGISTRY_KIND] = dataclasses.replace(declared[REGISTRY_KIND], **changes)
    monkeypatch.setattr(kinds, "registered", lambda: dict(declared))


def test_a_kind_with_no_tool_writes_no_tool_section(tmp_path, monkeypatch, registry_kind):
    _declared(monkeypatch, tool=None)
    render = _render(tmp_path, _tree())

    text = (render / "data" / "pyaml" / "SR" / "configuration.yaml").read_text("utf-8")
    assert registry_kind.tool.name not in text
    assert _facts(render)["measurement_models"]["SR"]["kinds"][REGISTRY_KIND]["tool"] is None


@pytest.mark.usefixtures("registry_kind")
def test_a_kind_with_no_stepped_role_writes_no_stepped_entry(tmp_path, monkeypatch):
    _declared(monkeypatch, stepped_roles=())
    facts = _facts(_render(tmp_path, _tree()))

    assert REGISTRY_KIND in facts["measurement_models"]["SR"]["kinds"]
    assert REGISTRY_KIND not in hook_measurement(facts, tmp_path)["SR"]["kinds"]


# --- the kind's own sub-block, hooks and writers --------------------------------------

_BLOCK = {
    "trigger_sample": 12,
    "settings": [
        {
            "id": "gain",
            "label": "Gain",
            "description": "Monitor gain",
            "scope": "device",
            "signal": "gain",
            "timeout_s": 2.0,
        }
    ],
    "steps": [
        {"id": "arm", "label": "Arm", "description": "Arm the chain", "default": True},
        {
            "id": "inject",
            "label": "Inject",
            "description": "Request one shot",
            "default": True,
            "command": {
                "address": "SCOR:H:SP",
                "value": 1,
                "timeout_s": 5.0,
                "done": {"address": "BPM1:X", "value": 0},
            },
        },
    ],
}


def test_the_sub_block_is_validated_by_the_kinds_pydantic_model(registry_kind):
    assert kinds.block_problems(registry_kind, _BLOCK) == []
    assert kinds.block_problems(registry_kind, {**_BLOCK, "trigger_sample": -1}) == [
        "registry_kind.trigger_sample: Input should be greater than or equal to 0"
    ]
    assert kinds.block_problems(kinds.ORM, {"anything": 1}) == []


def _with_block(document: dict[str, Any]) -> dict[str, Any]:
    (sr,) = [model for model in document["models"] if model["name"] == "SR"]
    sr["measurement"]["kind_blocks"] = {REGISTRY_KIND: _BLOCK}
    return sr


@pytest.mark.usefixtures("registry_kind")
def test_a_command_step_on_an_unmarked_address_stops_the_build(tmp_path):
    from osprey.facility.validate import _measurement_stops

    document, _ = built_document(tmp_path, _tree())
    sr = _with_block(document)

    (stop,) = _measurement_stops(document)
    assert stop.detail == (
        "kind registry_kind: step inject writes SCOR:H:SP, which the facility does not mark command"
    )

    (entry,) = [e for e in sr["wiring"] if e["address"] == "SCOR:H:SP"]
    entry["marks"] = ["command"]
    assert list(_measurement_stops(document)) == []


def test_the_facts_writer_and_the_settings_transform_read_the_block(registry_kind):
    assert registry_kind.facts_writer is not None
    assert registry_kind.settings_from_measurement is not None

    assert registry_kind.facts_writer({}, _BLOCK) == {
        "chain": ["arm", "inject"],
        "trigger_sample": 12,
        "settings": ["gain"],
    }
    assert registry_kind.settings_from_measurement({"turns": 8, "solve": "periodic"}, _BLOCK) == {
        "turns": 12 + 64,
        "solve": "periodic",
    }


# --- what the runtime readers make of it ----------------------------------------------


@pytest.mark.usefixtures("registry_kind")
def test_the_approval_block_steps_the_kinds_corrector_groups(tmp_path):
    facts = _facts(_render(tmp_path, _tree()))

    block = hook_measurement(facts, tmp_path)["SR"]

    assert block["kinds"][REGISTRY_KIND] == {"hcor": "SR_HCM", "vcor": "SR_VCM"}
    assert block["groups"]["hcor"] == ["SCOR:H:SP"]
    assert block["groups"]["vcor"] == ["SCOR:V:SP"]
    assert block["corrector_roles"][REGISTRY_KIND] == ["hcor", "vcor"]
    assert REGISTRY_KIND not in block["by_kinds"]


def test_the_measure_tool_resolves_and_advertises_a_registered_kind(tmp_path, registry_kind):
    from fastmcp import FastMCP

    from osprey.mcp_server.python_executor.tools import pyaml_measure
    from osprey.mcp_server.python_executor.tools._pyaml_measure_args import resolve_measure_args

    render = _render(tmp_path, _tree())
    args = resolve_measure_args(
        REGISTRY_KIND,
        "SR",
        ["SCOR:V:SP"],
        None,
        None,
        config_path=render / "config.yml",
        timeout=60,
    )
    server = FastMCP("probe")
    pyaml_measure.register(server, render)
    (tool,) = asyncio.run(server.list_tools())

    assert args.correctors == ("SCOR:V:SP",)
    assert args.plan == json.loads(json.dumps(dict(registry_kind.measure_plan)))
    assert args.matrix is True
    assert REGISTRY_KIND in tool.parameters["properties"]["kind"]["enum"]
