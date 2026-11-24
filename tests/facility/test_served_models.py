"""The served list changes the simulator view only.

``simulation.models`` picks the physics models the simulator serves. The
facility file and the graph are the same under every served list; a model left
unserved hands its channels to the texture, which holds each setpoint at its
wiring default. The facility file follows its sources: a changed seed changes
it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from osprey.facility.views.outputs import render_facility_outputs
from osprey.services.graph_store.seeder import ttl_sha256
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.simulation import composite as composite_module
from osprey_facility import TEXTURE
from osprey_facility.build import build_facility
from osprey_facility.render import FACILITY_FILE
from tests._builds import FRAMEWORK_CHECKS

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = [pytest.mark.slow]

GRAPH = "data/graph/facility.ttl"
VARIABLES = "data/simulator/variables.json"
SEEDS = "simulation/seeds.yaml"

#: The channel whose seed ``test_a_changed_seed_changes_the_facility_file`` moves: the first
#: one ``seeds.yaml`` states a noise of 0.005 for, its noise as stated and as moved.
RESEEDED = "BR:DIAG:BPM:01:POSITION:X"
STATED_NOISE = "noise:\n    absolute: 0.005\n"
RESEEDED_NOISE = "noise:\n    absolute: 0.006\n"


def _render(root: Path, document: Any, config: dict[str, Any], facility_dir: Path) -> Path:
    root.mkdir(parents=True)
    render_facility_outputs(root, document, config, facility_dir)
    return root


@pytest.fixture(scope="module")
def texture_only(tmp_path_factory: pytest.TempPathFactory, built_demo: BuiltProject) -> Path:
    """The shared build re-rendered with the texture alone served, once for the module.

    Its readers only read the render.
    """
    config = yaml.safe_load((built_demo.build_dir / "config.yml").read_text(encoding="utf-8"))
    config.setdefault("simulation", {})["models"] = [TEXTURE]
    root = tmp_path_factory.mktemp("texture-only") / "texture-only"
    return _render(root, built_demo.facility, config, built_demo.facility_dir)


def _sr_magnet_defaults(render_dir: Path) -> dict[str, float]:
    variables = json.loads((render_dir / VARIABLES).read_text(encoding="utf-8"))
    (sr,) = [model for model in variables["models"] if model["name"] == "SR"]
    return {
        str(entry["address"]): float(entry["default"])
        for entry in sr["wiring"]
        if entry.get("direction") == "write" and str(entry["address"]).startswith("SR:MAG:")
    }


def test_a_texture_only_list_leaves_the_facility_file_and_the_graph_alone(
    texture_only: Path, built_demo: BuiltProject
) -> None:
    render_dir = texture_only
    built_graph = (built_demo.build_dir / GRAPH).read_text(encoding="utf-8")

    assert (render_dir / FACILITY_FILE).read_bytes() == built_demo.facility_raw
    assert ttl_sha256((render_dir / GRAPH).read_text(encoding="utf-8")) == ttl_sha256(built_graph)


async def test_a_texture_only_in_process_simulator_holds_every_sr_magnet_at_its_wiring_default(
    texture_only: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    render_dir = texture_only
    defaults = _sr_magnet_defaults(render_dir)
    connector = VAInProcessConnector()
    await connector.connect(
        {"simulator_view": str(render_dir / "data" / "simulator"), "response_delay_ms": 0}
    )

    read = await connector.read_multiple_channels(sorted(defaults))

    assert defaults
    assert connector._composite is not None and connector._composite.models == []
    assert {address: value.value for address, value in read.items()} == defaults
    await connector.disconnect()


def _channel(document: dict[str, Any], address: str) -> dict[str, Any]:
    (channel,) = [c for c in document["channels"] if c["id"] == address]
    found: dict[str, Any] = channel
    return found


def test_a_changed_seed_changes_the_facility_file(tmp_path: Path, built_demo: BuiltProject) -> None:
    # The authored side is the shared build's own facility file: the build makes it from the
    # same tree with the same project name.
    facility_dir = tmp_path / "reseeded" / "data" / "facility"
    shutil.copytree(built_demo.facility_dir, facility_dir)
    seeds = facility_dir / SEEDS
    text = seeds.read_text(encoding="utf-8")
    assert STATED_NOISE in text
    seeds.write_text(text.replace(STATED_NOISE, RESEEDED_NOISE, 1), encoding="utf-8")
    document = build_facility(facility_dir, project_name="demo", extra_checks=FRAMEWORK_CHECKS)
    render_dir = _render(tmp_path / "reseeded" / "render", document, {}, facility_dir)

    authored = built_demo.facility
    reseeded = json.loads((render_dir / FACILITY_FILE).read_bytes())
    assert _channel(authored, RESEEDED)["simulation"]["noise"] == {"absolute": 0.005}
    assert _channel(reseeded, RESEEDED)["simulation"]["noise"] == {"absolute": 0.006}
    # Nothing but that one seed moves: with it put back, the two files are equal.
    _channel(reseeded, RESEEDED)["simulation"]["noise"] = {"absolute": 0.005}
    assert reseeded == authored


def _measurement_surface(render_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """What a render's pyAML view brings: the view, the tools, the skill and the prompt."""
    import asyncio

    from fastmcp import FastMCP

    from osprey.cli.templates.claude_code import config_derived_context
    from osprey.cli.templates.manager import TemplateManager
    from osprey.mcp_server.python_executor.server import register_optional_tools

    ctx = config_derived_context(config, render_dir)
    server = FastMCP("probe")
    register_optional_tools(server, render_dir)
    env = TemplateManager().jinja_env
    skill = env.get_template("claude_code/claude/skills/pyaml-tools/SKILL.md.j2").render(ctx)
    prompt = env.get_template("claude_code/CLAUDE.md.j2").render(
        ctx, enabled_servers={"python"}, agents=[], agent_data_root="var/agent_data"
    )
    return {
        "view": (render_dir / "data" / "pyaml").exists(),
        "present": ctx["pyaml_view_present"],
        "tools": {tool.name for tool in asyncio.run(server.list_tools())},
        "skill": skill.strip() != "",
        "prompt": "## Measurements" in prompt,
    }


def test_a_texture_only_render_carries_no_pyaml_view_tool_skill_or_prompt_section(
    texture_only: Path, built_demo: BuiltProject
) -> None:
    config = yaml.safe_load((built_demo.build_dir / "config.yml").read_text(encoding="utf-8"))
    texture = _measurement_surface(texture_only, config)
    served = _measurement_surface(built_demo.build_dir, config)

    assert texture == {
        "view": False,
        "present": False,
        "tools": set(),
        "skill": False,
        "prompt": False,
    }
    assert served == {
        "view": True,
        "present": True,
        "tools": {"pyaml_measure", "model_response_matrix"},
        "skill": True,
        "prompt": True,
    }
