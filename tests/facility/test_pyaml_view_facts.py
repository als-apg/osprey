"""The facts and the hook block a render records for its pyAML views.

``facility_facts.json`` holds one ``measurement_models`` record per pyAML view
the render wrote, ``{path, sha256, files}`` of the bytes on disk, and never
``facility.json``; the rendered ``hook_config.json`` carries the
``measurement`` block, ``{kinds, groups, view_sha256}`` per model, with the
same digest; ``pyaml_view_present`` is true exactly when a view was written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.facility.render import FACILITY_FILE, render_facility_outputs
from osprey.facility.views.facts import FACTS_FILE, hook_measurement
from osprey.facility.views.pyaml import measurement_groups
from tests.facility._pyaml_trees import built_document, measured_tree

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
    manager = TemplateManager()
    return claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, render, {"project_name": "demo", **config}
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_each_record_is_the_written_view(tmp_path: Path) -> None:
    render = _render(tmp_path, measured_tree(), {})
    records = _facts(render)["measurement_models"]
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

    assert hook_config["measurement"] == {
        "LINE": {
            "kinds": {"orm": ["LINE_BPM", "LINE_HCM", "LINE_VCM"]},
            "groups": measurement_groups(document, "LINE"),
            "view_sha256": records["LINE"]["sha256"],
        },
        "SR": {
            "kinds": {"trm": ["SR_Q"]},
            "groups": measurement_groups(document, "SR"),
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
    assert line["kinds"] == {"orm": ["LINE_BPM", "LINE_HCM_h", "LINE_HCM_v"]}
    assert line["groups"] == {
        "bpm": ["LBPM:X", "LBPM:Y"],
        "hcor": ["LCOR:H:SP"],
        "vcor": ["LCOR:V:SP"],
        "quad": ["LQ:SP"],
    }
