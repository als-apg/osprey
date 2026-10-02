"""The agent facts view: what every render carries as ``data/facility_facts.*``.

``render_facility_outputs`` writes ``facility_facts.json`` (the identity, the
place levels, each device class with its count, aliases and families, each
model with whether the render serves it, and the channel count) and
``facility_facts.md`` (the same facts as one page) into each render's
``data/``. The agent context reads the name and the facts from that file, and
reads a render without one as a facility with no sources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.facility import TEXTURE
from osprey.facility.build import build_facility
from osprey.facility.render import render_facility_outputs
from osprey.facility.validate import vocabulary
from osprey.facility.views.facts import (
    FACTS_FILE,
    FACTS_PAGE,
    FACTS_SCHEMA,
    FACTS_TEMPLATE,
    facts_document,
    hook_measurement,
    read_facts,
    render_facts_page,
    zero_source_facts,
)
from tests.facility._synthetic_trees import BPM, QUAD, plain_tree, write_tree

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

# xdist_group("built_control_assistant"): every module reading the session's one
# control-assistant build shares a worker, so the build runs once per run.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("built_control_assistant")]

FACTS = f"data/{FACTS_FILE}"
PAGE = f"data/{FACTS_PAGE}"

ZERO_MODELS = [{"name": TEXTURE, "engine": TEXTURE, "served": True, "solve": None}]


def _built_facts(built: BuiltProject) -> dict[str, Any]:
    facts: dict[str, Any] = json.loads((built.build_dir / FACTS).read_bytes())
    return facts


def _context(render_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manager = TemplateManager()
    return claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, render_root, config
    )


def _rendered_config(built: BuiltProject) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load(
        (built.build_dir / "config.yml").read_text(encoding="utf-8")
    )
    return config


# --- the written files ---------------------------------------------------------------


def test_every_render_writes_the_facts_and_their_page(
    built_control_assistant: BuiltProject,
) -> None:
    assert built_control_assistant.outputs
    for outputs in built_control_assistant.outputs:
        facts = json.loads(outputs.files[FACTS])
        assert sorted(facts) == [
            "channel_count",
            "device_classes",
            "identity",
            "measurement_models",
            "models",
            "place_levels",
            "schema",
            "snapshot",
        ]
        assert facts["schema"] == FACTS_SCHEMA
        page = outputs.files[PAGE].decode("utf-8")
        assert page == render_facts_page(facts)
        assert page.startswith(f"schema: {FACTS_SCHEMA}\n")
        assert page.endswith("\n") and not page.endswith("\n\n")


def test_the_facts_are_the_facility_files(built_control_assistant: BuiltProject) -> None:
    facility = built_control_assistant.facility
    facts = _built_facts(built_control_assistant)

    assert facts["identity"] == {
        "code": facility["identity"]["code"],
        "name": built_control_assistant.repo.name,
        "description": None,
    }
    assert facts["place_levels"] == ["machine", "sector"]
    assert facts["channel_count"] == len(facility["channels"])
    assert facts["measurement_models"] == {}
    assert facts["snapshot"] is None
    assert facts["models"] == [
        {"name": "SR", "engine": "pyat", "served": True, "solve": "periodic"},
        {"name": TEXTURE, "engine": TEXTURE, "served": True, "solve": None},
    ]


def test_each_device_class_carries_its_count_aliases_and_families(
    built_control_assistant: BuiltProject,
) -> None:
    facility = built_control_assistant.facility
    classes = _built_facts(built_control_assistant)["device_classes"]

    class_of = {device["id"]: device["class"] for device in facility["devices"]}
    assert list(classes) == sorted(set(class_of.values()))
    authored: dict[str, list[str]] = {}
    for row in vocabulary()["classes"]:
        authored.setdefault(row["name"], []).extend(row.get("aliases") or [])
    for row in facility["classes"]:
        authored.setdefault(row["class"], []).extend(row.get("aliases") or [])
    for name, entry in classes.items():
        members = {device for device, cls in class_of.items() if cls == name}
        assert entry == {
            "count": len(members),
            "aliases": sorted(authored.get(name, [])),
            "families": sorted(
                group["id"] for group in facility["groups"] if members & set(group["members"])
            ),
        }
    assert classes["Quadrupole"]["aliases"]
    assert classes["Quadrupole"]["families"]
    assert {"BPM", "PM"} <= set(classes["BeamPositionMonitor"]["aliases"])


def test_two_renders_differing_in_served_models_write_different_facts(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    served = {}
    for name, models in (("all", None), ("none", [TEXTURE])):
        render_dir = tmp_path / name
        render_dir.mkdir()
        config = {"project_name": "demo", "simulation": {"models": models}}
        render_facility_outputs(
            render_dir,
            built_control_assistant.facility,
            config,
            built_control_assistant.facility_dir,
        )
        facts = json.loads((render_dir / FACTS).read_bytes())
        served[name] = {model["name"]: model["served"] for model in facts["models"]}

    assert served == {
        "all": {"SR": True, TEXTURE: True},
        "none": {"SR": False, TEXTURE: True},
    }


# --- facility-added classes ----------------------------------------------------------


def test_a_facility_added_class_reaches_the_facts_with_its_aliases(tmp_path: Path) -> None:
    tree = plain_tree()
    declared = {
        "class": "SkewQuad",
        "parent": QUAD,
        "aliases": ["Skew Quad", " skew quad ", "skew"],
    }
    tree["classes.yaml"] = [declared, {"class": "Spare", "parent": QUAD}]
    devices = tree["records/devices.yaml"]
    quad = next(device for device in devices if device["class"] == QUAD)
    quad["class"] = "SkewQuad"

    document = build_facility(write_tree(tmp_path / "facility", tree), project_name="p")
    classes = facts_document(document, [TEXTURE], "p")["device_classes"]

    assert declared in document["classes"]
    assert list(classes) == sorted([BPM, "SkewQuad", "Spare"])
    assert classes["SkewQuad"]["count"] == 1
    assert classes["SkewQuad"]["aliases"] == ["Skew Quad", "skew"]
    assert classes["Spare"] == {"count": 0, "aliases": [], "families": []}
    assert QUAD not in classes


# --- zero sources --------------------------------------------------------------------


def test_a_facility_with_no_sources_has_the_zero_source_facts(tmp_path: Path) -> None:
    document = build_facility(tmp_path / "absent", project_name="my proj")

    facts = facts_document(document, [TEXTURE], "my proj")

    assert facts == zero_source_facts({"code": "my_proj", "name": "my proj", "description": None})
    assert facts["device_classes"] == {}
    assert facts["models"] == ZERO_MODELS
    assert facts["measurement_models"] == {}
    assert "This facility's build holds no device class.\n" in render_facts_page(facts)


def test_a_render_with_no_facts_file_is_read_as_zero_sources(tmp_path: Path) -> None:
    assert read_facts(tmp_path, "1st-lab") == zero_source_facts(
        {"code": "x1st_lab", "name": "1st-lab", "description": None}
    )


def test_a_render_with_only_a_facility_file_takes_its_identity(tmp_path: Path) -> None:
    identity = {"code": "demo", "name": "Demo Lab", "description": "A demonstration."}
    (tmp_path / "facility.json").write_text(
        json.dumps({"schema": "osprey.facility.facility/1", "identity": identity}),
        encoding="utf-8",
    )

    assert read_facts(tmp_path, "other-project") == zero_source_facts(identity)


def test_an_unreadable_facts_file_is_read_as_zero_sources(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / FACTS_FILE).write_text("{", encoding="utf-8")

    assert read_facts(tmp_path, "lab")["device_classes"] == {}


def test_a_facts_file_missing_a_key_is_read_as_zero_sources(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    partial = zero_source_facts({"code": "other", "name": "Other", "description": None})
    del partial["measurement_models"]
    (tmp_path / "data" / FACTS_FILE).write_text(json.dumps(partial), encoding="utf-8")

    assert read_facts(tmp_path, "lab") == zero_source_facts(
        {"code": "lab", "name": "lab", "description": None}
    )


# --- the agent context ---------------------------------------------------------------


def test_an_unbuilt_project_context_carries_the_zero_source_facts(tmp_path: Path) -> None:
    ctx = _context(tmp_path, {"project_name": "lab", "facility": {"name": "Configured"}})

    assert ctx["facility_facts"] == zero_source_facts(
        {"code": "lab", "name": "lab", "description": None}
    )
    assert ctx["facility_name"] == "lab"
    assert ctx["pyaml_view_present"] is False
    assert ctx["measurement"] == {}


def test_the_built_context_reads_the_written_facts(built_control_assistant: BuiltProject) -> None:
    facts = _built_facts(built_control_assistant)

    ctx = _context(built_control_assistant.build_dir, _rendered_config(built_control_assistant))

    assert ctx["facility_facts"] == facts
    assert ctx["facility_name"] == facts["identity"]["name"]
    assert ctx["pyaml_view_present"] is False
    assert ctx["measurement"] == {}


def test_the_context_renders_the_builds_page_byte_for_byte(
    built_control_assistant: BuiltProject,
) -> None:
    manager = TemplateManager()
    ctx = claude_code.build_claude_code_context(
        manager.template_root,
        manager.jinja_env,
        built_control_assistant.build_dir,
        _rendered_config(built_control_assistant),
    )

    rendered = manager.jinja_env.get_template(FACTS_TEMPLATE).render(**ctx)

    assert rendered.encode("utf-8") == (built_control_assistant.build_dir / PAGE).read_bytes()


def test_a_dry_run_regeneration_after_the_build_changes_no_file(
    built_control_assistant: BuiltProject,
) -> None:
    from osprey.deployment.status_display import _artifact_drift

    drift = _artifact_drift(
        built_control_assistant.repo,
        built_control_assistant.build_dir,
        _rendered_config(built_control_assistant),
    )

    assert drift is not None
    assert drift["changed"] == []


def test_the_measurement_block_names_each_measurement_view() -> None:
    facts = zero_source_facts({"code": "lab", "name": "lab", "description": None})
    facts["measurement_models"] = {
        "SR": {"path": "data/pyaml/SR", "sha256": "ab", "files": {"SR.yaml": "cd"}}
    }

    assert hook_measurement(facts) == {"SR": {"view_sha256": "ab"}}
