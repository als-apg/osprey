"""The pyat-specialist reads the simulator view and exists only beside a served deck.

``config_derived_context`` names the render's served deck-bearing models as
``served_deck_models``: the models the agent facts list as served, engine other
than ``texture``, whose facility-file record names a deck. The agent's registry
entry is conditioned on that list, so a render serving none of them renders no
agent file and a CLAUDE.md that never names the agent. On a render that serves
one, the agent loads each such model from its copy under
``data/simulator/decks/`` and reads channel wiring from
``data/simulator/variables.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests._builds import BuiltProject
from tests._facility_file import write_facility_views

from osprey.cli.templates.claude_code import config_derived_context
from osprey.facility import FACILITY_FILE, TEXTURE
from osprey.facility.views.facts import FACTS_FILE, zero_source_facts

AGENT = Path(".claude") / "agents" / "pyat-specialist.md"


def _model(name: str, engine: str, served: bool) -> dict:
    return {"name": name, "engine": engine, "served": served, "solve": None}


def test_served_deck_models_are_the_served_physics_models_with_a_deck(tmp_path: Path) -> None:
    facts = zero_source_facts({"code": "demo", "name": "demo", "description": None})
    facts["models"] = [
        _model("SR", "pyat", True),
        _model("BTS", "pyat", True),
        _model("LTB", "pyat", False),
        _model("AR", "pyat", True),
        _model(TEXTURE, TEXTURE, True),
    ]
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / FACTS_FILE).write_text(json.dumps(facts), encoding="utf-8")
    facility = {
        "models": [
            {"name": "AR", "engine": "pyat", "deck": "decks/AR.json"},
            {"name": "BTS", "engine": "pyat"},
            {"name": "LTB", "engine": "pyat", "deck": "decks/LTB.json"},
            {"name": "SR", "engine": "pyat", "deck": "decks/SR.json"},
        ]
    }
    (tmp_path / FACILITY_FILE).write_text(json.dumps(facility), encoding="utf-8")

    assert config_derived_context({}, tmp_path)["served_deck_models"] == ["AR", "SR"]


def test_a_render_with_no_facts_serves_no_deck(tmp_path: Path) -> None:
    assert config_derived_context({}, tmp_path)["served_deck_models"] == []


def test_the_demo_agent_loads_the_served_deck_and_reads_the_variables_file(
    built_control_assistant: BuiltProject,
) -> None:
    text = (built_control_assistant.build_dir / AGENT).read_text(encoding="utf-8")

    assert "`SR`" in text
    assert "data/simulator/decks/SR.json" in text
    assert "at.load_lattice(" in text
    assert "data/simulator/variables.json" in text
    assert "va_bindings.json" not in text
    assert "osprey.simulation" not in text


@pytest.fixture(scope="module")
def texture_only_render(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The control-assistant bundle rendered with ``simulation.models: [texture]``.

    The facility file still holds the deck-bearing ``SR``; the render serves
    only ``texture``.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.cli.templates.manager import TemplateManager
    from osprey.utils.config_writer import config_update_fields

    manager = TemplateManager()
    bundle = "control_assistant"
    project = manager.create_project(
        project_name="texture-only",
        output_dir=tmp_path_factory.mktemp("texture-only"),
        data_bundle=bundle,
        context={"channel_finder_mode": "hierarchical"},
        data_root=Path(manager.template_root) / "apps" / bundle / "data",
    )
    profile, _profile_dir = resolve_build_profile(None, preset="control-assistant")
    config_update_fields(project / "config.yml", profile.config)
    config_update_fields(project / "config.yml", {"simulation.models": [TEXTURE]})
    manager.generate_manifest(
        project,
        "texture-only",
        "control-assistant",
        {},
        artifacts=manager._effective_artifacts(bundle, None),
    )
    write_facility_views(project, bundle)
    manager.regenerate_claude_code(project)
    return project


def test_a_texture_only_render_lists_the_deck_model_as_not_served(
    texture_only_render: Path,
) -> None:
    facts = json.loads((texture_only_render / "data" / FACTS_FILE).read_text(encoding="utf-8"))
    served = {model["name"]: model["served"] for model in facts["models"]}
    assert served == {"SR": False, TEXTURE: True}


def test_a_texture_only_render_has_no_pyat_specialist(texture_only_render: Path) -> None:
    assert not (texture_only_render / AGENT).exists()
    claude_md = (texture_only_render / "CLAUDE.md").read_text(encoding="utf-8")
    assert "pyat-specialist" not in claude_md
