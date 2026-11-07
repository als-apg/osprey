"""The facts and the hook block a render records for its pyAML views.

``facility_facts.json`` holds one ``measurement_models`` record per pyAML view
the render wrote, ``{path, sha256, files}`` of the bytes on disk with the
model's kind block, and never ``facility.json``; the rendered
``hook_config.json`` carries the ``measurement`` block, ``{kinds, groups, rf,
by_kinds, corrector_roles, view_sha256}`` per model, read from the facts alone,
with the same digest; ``pyaml_view_present`` is true exactly when a view was
written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.facility.kinds import registered
from osprey.facility.records import FacilityRecords
from osprey.facility.render import FACILITY_FILE, render_facility_outputs
from osprey.facility.views.facts import FACTS_FILE, hook_measurement
from osprey.facility.views.pyaml import RF_PLANT_TYPE, measurement_groups
from tests.facility._pyaml_trees import built_document, measured_tree, with_rf

pytest.importorskip("pyaml")

_HOOK_CONFIG_TEMPLATE = "claude_code/claude/hooks/hook_config.json.j2"


def _render(tmp_path: Path, tree: dict[str, Any], config: dict[str, Any]) -> Path:
    document, facility = built_document(tmp_path / "tree", tree)
    render = tmp_path / "render"
    render.mkdir()
    render_facility_outputs(render, document, config, facility)
    return render


def _facts(render: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((render / "data" / FACTS_FILE).read_text("utf-8"))
    return loaded


def _context(render: Path, config: dict[str, Any]) -> dict[str, Any]:
    # A framework-default render pins writes `never` with python's read/write
    # tools on, which the write-gate lint refuses unless writes-check is selected.
    (render / ".osprey-manifest.json").write_text(
        json.dumps({"artifacts": {"hooks": ["writes-check"]}}), encoding="utf-8"
    )
    manager = TemplateManager()
    return claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, render, {"project_name": "demo", **config}
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_each_record_is_the_written_view(tmp_path: Path) -> None:
    render = _render(tmp_path, measured_tree(), {})
    records = {
        model: {key: record[key] for key in ("path", "sha256", "files")}
        for model, record in _facts(render)["measurement_models"].items()
    }
    view = render / "data" / "pyaml"
    assert records == {
        "LINE": {
            "path": "data/pyaml/LINE/configuration.yaml",
            "sha256": _sha(view / "LINE" / "configuration.yaml"),
            "files": {},
        },
        "SR": {
            "path": "data/pyaml/SR/configuration.yaml",
            "sha256": _sha(view / "SR" / "configuration.yaml"),
            "files": {
                "lattice.json": _sha(view / "SR" / "lattice.json"),
                "trm.json": _sha(view / "SR" / "trm.json"),
            },
        },
    }


def test_each_record_carries_its_models_kind_block(tmp_path: Path) -> None:
    render = _render(tmp_path, with_rf(measured_tree()), {})
    document = json.loads((render / FACILITY_FILE).read_text("utf-8"))
    facts = _facts(render)
    sr = facts["measurement_models"]["SR"]
    groups = measurement_groups(FacilityRecords(document), "SR")
    trm = registered()["trm"]

    assert sr["instruments"] == {"rf": "RF:SP", "tune": "SR:TUNE:X"}
    assert sr["rf"] == ["RF:SP"]
    assert set(sr["names"]) == {"bpm", "quad", "rf", "tune"}
    assert sr["kinds"] == {
        "trm": {
            "members": {
                "quad": {"requirement": "wired", "addresses": groups["quad"]},
                "tune": {"requirement": "wired", "addresses": ["SR:TUNE:X"]},
            },
            "stepped_roles": ["quad"],
            "tool_keys": ["quad_delta"],
            "tool": {"type": trm.tool.type, "name": trm.tool.name} if trm.tool else None,
            "single_pass": False,
            "matrix": True,
            "takes_by": True,
            "corrector_roles": [],
            "view_files": ["trm.json"],
            "measure_plan": json.loads(json.dumps(dict(trm.measure_plan))),
            "description": trm.description,
        }
    }
    assert set(facts["measurement_kinds"]) == set(registered())
    assert facts["measurement_kinds"]["orm"]["corrector_roles"] == ["hcor", "vcor"]


def test_the_facility_file_records_no_view(tmp_path: Path) -> None:
    render = _render(tmp_path, measured_tree(), {})
    facility = (render / FACILITY_FILE).read_text("utf-8")
    assert "measurement_models" not in facility
    assert "data/pyaml" not in facility


def test_the_rendered_hook_block_lists_kinds_groups_and_the_same_digest(tmp_path: Path) -> None:
    render = _render(tmp_path, measured_tree(), {})
    document = json.loads((render / FACILITY_FILE).read_text("utf-8"))
    records = _facts(render)["measurement_models"]
    ctx = _context(render, {})
    assert ctx["pyaml_view_present"] is True

    manager = TemplateManager()
    hook_config = json.loads(
        manager.jinja_env.get_template(_HOOK_CONFIG_TEMPLATE).render(
            {**ctx, "servers": [], "control_system_write_tools": []}
        )
    )

    line = measurement_groups(FacilityRecords(document), "LINE")
    sr = measurement_groups(FacilityRecords(document), "SR")
    assert hook_config["measurement"] == {
        "LINE": {
            "kinds": {"orm": {"hcor": "LINE_HCM", "vcor": "LINE_VCM"}},
            "groups": {role: line[role] for role in ("bpm", "hcor", "vcor")},
            "rf": [],
            "by_kinds": [],
            "corrector_roles": {"orm": ["hcor", "vcor"]},
            "view_sha256": records["LINE"]["sha256"],
        },
        "SR": {
            "kinds": {"trm": {"quad": "SR_Q"}},
            "groups": {"quad": sr["quad"]},
            "rf": [],
            "by_kinds": ["trm"],
            "corrector_roles": {},
            "view_sha256": records["SR"]["sha256"],
        },
    }


def test_a_texture_only_render_records_no_view(tmp_path: Path) -> None:
    config = {"simulation": {"models": []}}
    render = _render(tmp_path, measured_tree(), config)
    assert _facts(render)["measurement_models"] == {}
    assert not (render / "data" / "pyaml").exists()
    ctx = _context(render, config)
    assert ctx["pyaml_view_present"] is False
    assert ctx["measurement"] == {}


def test_one_group_named_for_both_planes_gives_the_block_both_plane_arrays(
    tmp_path: Path,
) -> None:
    tree = measured_tree()
    tree["measurement/LINE.yaml"]["groups"]["vcor"] = "LINE/HCM"
    render = _render(tmp_path, tree, {})
    line = hook_measurement(_facts(render), render)["LINE"]
    assert line["kinds"] == {"orm": {"hcor": "LINE_HCM_h", "vcor": "LINE_HCM_v"}}
    assert line["groups"] == {
        "bpm": ["LBPM:X", "LBPM:Y"],
        "hcor": ["LCOR:H:SP"],
        "vcor": ["LCOR:V:SP"],
    }


def test_the_block_names_the_rf_setpoint_the_views_rf_plant_steps(tmp_path: Path) -> None:
    render = _render(tmp_path, with_rf(measured_tree()), {})
    block = hook_measurement(_facts(render), render)
    assert block["SR"]["rf"] == ["RF:SP"]
    assert block["LINE"]["rf"] == []


def _with_crm(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with SR's sextupole SX wired, named ``groups.sext`` and ``crm`` listed."""
    tree["records/devices.yaml"].append({"id": "SR/SX", "class": "Sextupole"})
    tree["records/channels.yaml"].append(
        {"id": "SX:SP", "role": "setpoint", "on": {"device": "SR/SX"}}
    )
    tree["records/groups.yaml"].append({"id": "SR/S", "members": ["SR/SX"]})
    (sr,) = [model for model in tree["models.yaml"] if model["name"] == "SR"]
    sr["wiring"].append(
        {"address": "SX:SP", "element": "SX", "engine": {"attribute": "PolynomB", "index": 2}}
    )
    tree["measurement/SR.yaml"]["groups"]["sext"] = "SR/S"
    tree["measurement/SR.yaml"]["kinds"] = ["trm", "crm"]
    tree["measurement/SR.yaml"]["sextu_delta"] = 0.01
    return tree


@pytest.mark.parametrize(
    "tree", [measured_tree, lambda: _with_crm(with_rf(measured_tree()))], ids=["base", "crm"]
)
def test_the_block_lists_each_listed_kind_with_every_role_it_steps(
    tmp_path: Path, tree: Any
) -> None:
    built = tree()
    render = _render(tmp_path, built, {})
    block = hook_measurement(_facts(render), render)
    for model, entry in block.items():
        listed = built[f"measurement/{model}.yaml"]["kinds"]
        assert set(entry["kinds"]) == set(listed)
        for kind, stepped in entry["kinds"].items():
            assert set(stepped) == set(registered()[kind].stepped_roles)


def test_the_block_names_the_views_rf_plant_for_a_kind_that_steps_rf(tmp_path: Path) -> None:
    render = _render(tmp_path, _with_crm(with_rf(measured_tree())), {})
    configuration = yaml.safe_load(
        (render / "data" / "pyaml" / "SR" / "configuration.yaml").read_text("utf-8")
    )
    (plant,) = [
        device["name"] for device in configuration["devices"] if device.get("type") == RF_PLANT_TYPE
    ]
    block = hook_measurement(_facts(render), render)
    assert block["SR"]["kinds"]["crm"] == {"rf": plant, "sext": "SR_S"}


def test_the_block_is_read_from_the_facts_alone(tmp_path: Path) -> None:
    """A launch-time regeneration reads no view and no facility file: the facts decide."""
    render = _render(tmp_path, _with_crm(with_rf(measured_tree())), {})
    facts = _facts(render)

    block = hook_measurement(facts, render)

    assert hook_measurement(facts, tmp_path / "absent") == block
    assert block["SR"]["by_kinds"] == ["crm", "trm"]
    assert block["SR"]["view_sha256"] == facts["measurement_models"]["SR"]["sha256"]
    block["SR"]["groups"].clear()
    assert hook_measurement(facts, render)["SR"]["groups"]
