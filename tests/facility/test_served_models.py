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

from osprey.connectors.control_system.va_in_process_connector import VAInProcessConnector
from osprey.facility import TEXTURE
from osprey.facility.build import build_facility
from osprey.facility.render import FACILITY_FILE, render_facility_outputs
from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256
from osprey_connectors.simulation import composite as composite_module

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = [pytest.mark.slow]

GRAPH = "data/graph/facility.ttl"
VARIABLES = "data/simulator/variables.json"
SEEDS = "seeds.yaml"


def _render(root: Path, document: Any, config: dict[str, Any], facility_dir: Path) -> Path:
    root.mkdir(parents=True)
    render_facility_outputs(root, document, config, facility_dir)
    return root


def _texture_only(tmp_path: Path, built: BuiltProject) -> Path:
    config = yaml.safe_load((built.build_dir / "config.yml").read_text(encoding="utf-8"))
    config.setdefault("simulation", {})["models"] = [TEXTURE]
    return _render(tmp_path / "texture-only", built.facility, config, built.facility_dir)


def _sr_magnet_defaults(render_dir: Path) -> dict[str, float]:
    variables = json.loads((render_dir / VARIABLES).read_text(encoding="utf-8"))
    (sr,) = [model for model in variables["models"] if model["name"] == "SR"]
    return {
        str(entry["address"]): float(entry["default"])
        for entry in sr["wiring"]
        if entry.get("direction") == "write" and str(entry["address"]).startswith("SR:MAG:")
    }


def test_a_texture_only_list_leaves_the_facility_file_and_the_graph_alone(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    render_dir = _texture_only(tmp_path, built_control_assistant)
    built_graph = (built_control_assistant.build_dir / GRAPH).read_text(encoding="utf-8")

    assert (render_dir / FACILITY_FILE).read_bytes() == built_control_assistant.facility_raw
    assert ttl_sha256((render_dir / GRAPH).read_text(encoding="utf-8")) == ttl_sha256(built_graph)


async def test_a_texture_only_in_process_simulator_holds_every_sr_magnet_at_its_wiring_default(
    tmp_path: Path, built_control_assistant: BuiltProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    render_dir = _texture_only(tmp_path, built_control_assistant)
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


def test_a_changed_seed_changes_the_facility_file(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    rendered: dict[str, bytes] = {}
    for name in ("authored", "reseeded"):
        facility_dir = tmp_path / name / "data" / "facility"
        shutil.copytree(built_control_assistant.facility_dir, facility_dir)
        if name == "reseeded":
            seeds = facility_dir / SEEDS
            text = seeds.read_text(encoding="utf-8")
            stated = "noise:\n    absolute: 0.005\n"
            assert stated in text
            seeds.write_text(
                text.replace(stated, "noise:\n    absolute: 0.006\n", 1), encoding="utf-8"
            )
        document = build_facility(facility_dir, project_name="demo")
        render_dir = _render(tmp_path / name / "render", document, {}, facility_dir)
        rendered[name] = (render_dir / FACILITY_FILE).read_bytes()

    assert rendered["authored"] != rendered["reseeded"]
