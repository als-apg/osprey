"""The simulator view's reader: ``osprey_connectors.simulation.view``.

``SimulatorView`` opens the view a build wrote, refuses a file of another
schema with a rebuild hint, and answers every question a consumer asks of it
from the stamped wiring records and the channel entries.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pytest

from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    PLANES,
    REFRESH,
    ROLES,
    SCENARIOS_DIR,
    TEXTURE,
    VARIABLES_FILE,
    VIEW_RELPATH,
    Binding,
    Channel,
    Model,
    NoSimulatorView,
    SimulatorView,
    ViewSchemaError,
)

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = [pytest.mark.slow]

PREFIX = f"{VIEW_RELPATH}/"


@pytest.fixture
def demo_view(built_control_assistant: BuiltProject, tmp_path: Path) -> Path:
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith(PREFIX):
            target = tmp_path / "simulator" / name[len(PREFIX) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return tmp_path / "simulator"


def _raw(view_dir: Path, name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((view_dir / name).read_text(encoding="utf-8"))
    return document


def test_every_accessor_answers(demo_view: Path) -> None:
    view = SimulatorView.open(demo_view)
    variables = _raw(demo_view, VARIABLES_FILE)
    addresses = _raw(demo_view, ADDRESSES_FILE)

    assert view.path == demo_view
    assert view.render_root is None
    assert view.code == variables["code"]
    assert [model.name for model in view.models()] == [m["name"] for m in variables["models"]]
    assert all(isinstance(model, Model) for model in view.models())
    assert view.served() == ("LINE", "SR", TEXTURE)
    assert [model.name for model in view.physics_models()] == ["LINE", "SR"]
    assert view.channels() == tuple(addresses["channels"])
    assert dict(view.status_addresses()) == {
        "LINE": f"{view.code}:SIM:LINE:STATUS",
        "SR": f"{view.code}:SIM:SR:STATUS",
    }
    assert isinstance(view.status_addresses(), MappingProxyType)

    sr = view.model("SR")
    assert sr.engine == "pyat"
    assert sr.served is True
    assert sr.deck == demo_view / "decks" / "SR.json"
    assert sr.deck.is_file()
    assert dict(sr.settings) == {"pyat": {"solve": "periodic"}}
    wiring = {model["name"]: model["wiring"] for model in variables["models"]}
    assert len(sr.bindings) == len(wiring["SR"]) > 0
    line = view.model("LINE")
    assert dict(line.settings)["pyat"]["solve"] == "single_pass"
    assert len(line.bindings) == len(wiring["LINE"]) > 0
    assert view.model(TEXTURE).bindings == ()
    with pytest.raises(KeyError):
        view.model("NOPE")

    bindings = view.bindings()
    assert bindings == line.bindings + sr.bindings
    assert {binding.role for binding in bindings} == set(ROLES)
    assert {binding.plane for binding in bindings} <= {*PLANES, None}
    assert {binding.refresh for binding in bindings} <= set(REFRESH)
    first = bindings[0]
    assert isinstance(first, Binding)
    assert view.binding(first.address) == first
    assert dict(first.record) == wiring["LINE"][0]
    assert isinstance(first.record, MappingProxyType)

    channel = view.channel(first.address)
    assert isinstance(channel, Channel)
    assert channel.owner == "LINE"
    assert view.channel(sr.bindings[0].address).owner == "SR"
    textured = next(address for address in view.channels() if view.binding(address) is None)
    assert view.channel(textured).owner == TEXTURE

    seeds = _raw(demo_view, "seeds.json")["seeds"]
    seeded = next(iter(seeds))
    assert view.seed(seeded) is not None
    assert dict(view.seed(seeded) or {}).keys() == seeds[seeded].keys()
    assert view.seed("NO:SUCH:ADDRESS") is None
    assert [scenario["name"] for scenario in view.scenarios()] == [
        s["name"] for s in _raw(demo_view, "scenarios.json")["scenarios"]
    ]
    assert view.scenario_dir("nominal") == demo_view / SCENARIOS_DIR / "nominal"
    assert view.document(VARIABLES_FILE)["schema"] == variables["schema"]


def test_a_channels_motion_envelope_follows_its_seed_and_the_active_scenarios(
    demo_view: Path,
) -> None:
    from osprey_connectors.simulation.envelope import active_envelopes, motion_envelope

    view = SimulatorView.open(demo_view)
    seeds = _raw(demo_view, "seeds.json")["seeds"]
    scenarios = {entry["name"]: entry for entry in _raw(demo_view, "scenarios.json")["scenarios"]}
    noisy = next(address for address, seed in sorted(seeds.items()) if seed.get("drift"))
    live = active_envelopes(seeds, scenarios, ["rf-thermal-live"])
    replaced = sorted(scenarios["rf-thermal-live"]["noise"])[0]

    assert view.motion_envelope(noisy) == motion_envelope(seeds[noisy]) > 0.0
    assert view.motion_envelope("NO:SUCH:CHANNEL") == 0.0
    assert view.motion_envelope(replaced, active=("rf-thermal-live",)) == live[replaced]
    assert live[replaced] != motion_envelope(seeds.get(replaced))


def test_monitor_bindings_of_a_plane_are_the_element_reads_on_that_axis(demo_view: Path) -> None:
    view = SimulatorView.open(demo_view)
    models = sorted(_raw(demo_view, VARIABLES_FILE)["models"], key=lambda m: m["name"])
    expected = [
        record["address"]
        for model in models
        for record in model["wiring"]
        if record["role"] == "monitor" and record["plane"] == "x"
    ]

    found = [binding.address for binding in view.bindings(role="monitor", plane="x")]

    assert found == expected
    assert found


def test_bindings_filter_by_node(demo_view: Path) -> None:
    view = SimulatorView.open(demo_view)
    bound = next(
        binding for binding in view.bindings() if view.channel(binding.address).on is not None
    )
    (node,) = view.channel(bound.address).on.values()  # type: ignore[union-attr]

    on_node = view.bindings(node=node)

    assert bound in on_node
    for binding in on_node:
        assert node in view.channel(binding.address).on.values()  # type: ignore[union-attr]


@pytest.mark.parametrize("schema", ["osprey.facility.simulator/1", None])
def test_a_variables_file_of_another_schema_asks_for_a_rebuild(
    demo_view: Path, schema: str | None
) -> None:
    variables = _raw(demo_view, VARIABLES_FILE)
    if schema is None:
        del variables["schema"]
    else:
        variables["schema"] = schema
    (demo_view / VARIABLES_FILE).write_text(json.dumps(variables), encoding="utf-8")
    view = SimulatorView.open(demo_view)

    with pytest.raises(ViewSchemaError, match="rebuild"):
        view.models()


def test_a_wiring_record_without_its_description_asks_for_a_rebuild() -> None:
    with pytest.raises(ViewSchemaError, match="rebuild"):
        Binding.from_record("SR", {"id": "SR/A", "address": "A", "direction": "read"})


def test_a_missing_view_is_no_view_and_find_says_none(tmp_path: Path) -> None:
    with pytest.raises(NoSimulatorView, match="run osprey build"):
        SimulatorView.open(tmp_path / "nowhere")

    assert SimulatorView.find(tmp_path) is None


def test_find_still_refuses_a_view_of_another_schema(tmp_path: Path) -> None:
    view_dir = tmp_path / VIEW_RELPATH
    view_dir.mkdir(parents=True)
    (view_dir / ADDRESSES_FILE).write_text(json.dumps({"schema": "old"}), encoding="utf-8")

    with pytest.raises(ViewSchemaError, match="rebuild"):
        SimulatorView.find(tmp_path)


def test_a_repo_and_its_render_open_the_same_view(built_control_assistant: BuiltProject) -> None:
    repo = built_control_assistant.repo

    of_repo = SimulatorView.of_project(repo)
    of_render = SimulatorView.of_render(repo / "build")

    assert of_repo.path == of_render.path == repo / "build" / VIEW_RELPATH
    assert of_repo.render_root == repo / "build"
    assert SimulatorView.of_project(repo / "build").path == of_repo.path
    assert SimulatorView.path_for_project(repo) == of_repo.path
    assert of_repo.channels() == of_render.channels()


def test_the_reader_adds_only_the_standard_library_and_loads_no_model() -> None:
    probe = (
        "import json, sys\n"
        "import osprey_connectors.simulation\n"
        "before = set(sys.modules)\n"
        "import osprey_connectors.simulation.view\n"
        "added = sorted(\n"
        "    name for name in set(sys.modules) - before\n"
        "    if name.split('.')[0] not in sys.stdlib_module_names\n"
        "    and name.split('.')[0] != 'osprey_connectors'\n"
        ")\n"
        "roots = {name.split('.')[0] for name in sys.modules}\n"
        "print(json.dumps([added, sorted(roots & {'numpy', 'lume', 'at', 'osprey'})]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert json.loads(result.stdout) == [[], []]
