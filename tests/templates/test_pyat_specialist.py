"""The pyat-specialist reads the simulator view and exists only beside a served deck.

``config_derived_context`` names the render's served decks as ``served_decks``:
one ``{model, path}`` for each model the agent facts list as served, engine
other than ``texture``, whose facility-file record names a deck, ``path`` being
the file the simulator view writes for it. The agent's registry entry is
conditioned on that list, so a render serving none of them renders no agent
file and a CLAUDE.md that never names the agent. On a render that serves one,
the agent loads each such deck by that path and reads channel wiring from
``data/simulator/variables.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import pytest
from tests._builds import BuiltProject
from tests._facility_file import write_facility_views
from tests._preset_data import bundle_data_root

from osprey.cli.templates.claude_code import config_derived_context
from osprey.facility import FACILITY_FILE, TEXTURE
from osprey.facility.views.facts import FACTS_FILE, zero_source_facts

AGENT = Path(".claude") / "agents" / "pyat-specialist.md"


def _model(name: str, engine: str, served: bool) -> dict:
    return {"name": name, "engine": engine, "served": served, "solve": None}


def test_served_decks_are_the_served_physics_models_with_a_deck(tmp_path: Path) -> None:
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
            {"name": "AR", "engine": "pyat", "deck": "decks/ar_lattice.mat"},
            {"name": "BTS", "engine": "pyat"},
            {"name": "LTB", "engine": "pyat", "deck": "decks/LTB.json"},
            {"name": "SR", "engine": "pyat", "deck": "decks/SR.json"},
        ]
    }
    (tmp_path / FACILITY_FILE).write_text(json.dumps(facility), encoding="utf-8")

    assert config_derived_context({}, tmp_path)["served_decks"] == [
        {"model": "AR", "path": "data/simulator/decks/AR.mat"},
        {"model": "SR", "path": "data/simulator/decks/SR.json"},
    ]


def test_a_render_with_no_facts_serves_no_deck(tmp_path: Path) -> None:
    assert config_derived_context({}, tmp_path)["served_decks"] == []


def test_the_agent_loads_a_mat_deck_by_the_file_the_view_writes(tmp_path: Path) -> None:
    from osprey.cli.templates.manager import TemplateManager

    facts = zero_source_facts({"code": "demo", "name": "demo", "description": None})
    facts["models"] = [_model("SR", "pyat", True)]
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / FACTS_FILE).write_text(json.dumps(facts), encoding="utf-8")
    facility = {"models": [{"name": "SR", "engine": "pyat", "deck": "decks/sr_lattice.mat"}]}
    (tmp_path / FACILITY_FILE).write_text(json.dumps(facility), encoding="utf-8")
    context = config_derived_context({}, tmp_path)
    context["enabled_agents"] = {"pyat-specialist"}

    text = (
        TemplateManager()
        .jinja_env.get_template("claude_code/claude/agents/pyat-specialist.md.j2")
        .render(**context)
    )

    assert "- `SR`: `data/simulator/decks/SR.mat`" in text
    assert 'at.load_lattice("data/simulator/decks/SR.mat")' in text
    assert ".json" not in text.split("## The Models You Load")[1].split("## Import Surface")[0]


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


class _TextureOnlyRender(NamedTuple):
    project: Path
    warnings: list[str]


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture(scope="module")
def texture_only(tmp_path_factory: pytest.TempPathFactory) -> _TextureOnlyRender:
    """The control-assistant bundle rendered with ``simulation.models: [texture]``.

    The facility file still holds the deck-bearing ``SR``; the render serves
    only ``texture``, and ``claude_code.agents.pyat-specialist.enabled`` is
    switched on by hand. The warnings the render logs are collected.
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
        data_root=bundle_data_root(bundle),
    )
    profile, _profile_dir = resolve_build_profile(None, preset="control-assistant")
    config_update_fields(project / "config.yml", profile.config)
    config_update_fields(
        project / "config.yml",
        {
            "simulation.models": [TEXTURE],
            "claude_code.agents.pyat-specialist.enabled": True,
        },
    )
    manager.generate_manifest(
        project,
        "texture-only",
        "control-assistant",
        {},
        artifacts=manager._effective_artifacts(bundle, None),
    )
    write_facility_views(project, bundle)
    collect = _Collect()
    registry_logger = logging.getLogger("osprey.registry.mcp")
    registry_logger.addHandler(collect)
    try:
        manager.regenerate_claude_code(project)
    finally:
        registry_logger.removeHandler(collect)
    return _TextureOnlyRender(project, collect.lines)


@pytest.fixture(scope="module")
def texture_only_render(texture_only: _TextureOnlyRender) -> Path:
    return texture_only.project


def test_a_texture_only_render_lists_the_deck_model_as_not_served(
    texture_only_render: Path,
) -> None:
    facts = json.loads((texture_only_render / "data" / FACTS_FILE).read_text(encoding="utf-8"))
    served = {model["name"]: model["served"] for model in facts["models"]}
    assert served == {"LINE": False, "SR": False, TEXTURE: True}


def test_a_texture_only_render_has_no_pyat_specialist(texture_only_render: Path) -> None:
    assert not (texture_only_render / AGENT).exists()
    claude_md = (texture_only_render / "CLAUDE.md").read_text(encoding="utf-8")
    assert "pyat-specialist" not in claude_md


def test_a_hand_enabled_agent_with_no_served_deck_is_left_out_with_one_line(
    texture_only: _TextureOnlyRender,
) -> None:
    assert not (texture_only.project / AGENT).exists()
    lines = [line for line in texture_only.warnings if "pyat-specialist" in line]
    assert lines == ["Agent 'pyat-specialist' is left out: it needs a served model with a deck"]
