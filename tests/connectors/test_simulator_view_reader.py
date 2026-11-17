"""The simulator view's reader: ``osprey_connectors.simulation.view``.

``SimulatorView`` opens the view a build wrote, refuses a file of another
schema with a rebuild hint, and answers every question a consumer asks of it
from the stamped wiring records and the channel entries.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from osprey_connectors.simulation import TEXTURE
from osprey_connectors.simulation.records import (
    VIEW_DOCUMENTS,
    ChannelRecord,
    ModelRecord,
    ScenarioRecord,
    SeedRecord,
    VariablesDocument,
    WiringRecord,
)
from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    ENGINE_ROLES,
    PLANES,
    REFRESH,
    SCENARIOS_DIR,
    VARIABLES_FILE,
    VIEW_RELPATH,
    NoSimulatorViewError,
    SimulatorView,
    ViewSchemaError,
    document_bytes,
    view_path,
)

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = [pytest.mark.slow]

PREFIX = f"{VIEW_RELPATH}/"


@pytest.fixture
def demo_view(built_demo: BuiltProject, tmp_path: Path) -> Path:
    for name, data in built_demo.outputs[0].files.items():
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
    assert all(isinstance(model, ModelRecord) for model in view.models())
    assert view.served() == ("BTS", "LINE", "LTB", "SR", TEXTURE)
    assert [model.name for model in view.physics_models()] == ["BTS", "LINE", "LTB", "SR"]
    assert view.channels() == tuple(addresses["channels"])
    assert dict(view.status_addresses()) == {
        name: f"{view.code}:SIM:{name}:STATUS" for name in ("BTS", "LINE", "LTB", "SR")
    }
    assert isinstance(view.status_addresses(), MappingProxyType)

    sr = view.model("SR")
    assert sr.engine == "pyat"
    assert sr.served is True
    assert sr.deck == "decks/SR.json"
    deck = view.deck_path(sr)
    assert deck == demo_view / "decks" / "SR.json"
    assert deck.is_file()
    assert view.deck_path(view.model(TEXTURE)) is None
    assert sr.engine_settings() == {"pyat": {"solve": "periodic"}}
    wiring = {model["name"]: model["wiring"] for model in variables["models"]}
    assert len(sr.wiring) == len(wiring["SR"]) > 0
    line = view.model("LINE")
    assert line.engine_settings()["pyat"]["solve"] == "single_pass"
    assert len(line.wiring) == len(wiring["LINE"]) > 0
    assert view.model(TEXTURE).wiring == ()
    with pytest.raises(KeyError):
        view.model("NOPE")

    bindings = view.bindings()
    assert bindings == tuple(b for model in view.physics_models() for b in model.wiring)
    assert {binding.engine_role for binding in bindings} == set(ENGINE_ROLES)
    assert {binding.plane for binding in bindings} <= {*PLANES, None}
    assert {binding.refresh for binding in bindings} <= set(REFRESH)
    first = bindings[0]
    owner = view.physics_models()[0].name
    assert isinstance(first, WiringRecord)
    assert view.binding(first.address) == first
    assert first.engine_entry() == wiring[owner][0]
    with pytest.raises(ValidationError):
        first.address = "elsewhere"  # type: ignore[misc]

    channel = view.channel(first.address)
    assert isinstance(channel, ChannelRecord)
    assert channel.owner == owner
    assert view.channel(line.wiring[0].address).owner == "LINE"
    assert view.channel(sr.wiring[0].address).owner == "SR"
    textured = next(address for address in view.channels() if view.binding(address) is None)
    assert view.channel(textured).owner == TEXTURE

    seeds = _raw(demo_view, "seeds.json")["seeds"]
    seeded = next(iter(seeds))
    assert view.seed(seeded) is not None
    seed = view.seed(seeded)
    assert seed is not None
    assert seed.model_dump(exclude_unset=True).keys() == seeds[seeded].keys()
    assert view.seed("NO:SUCH:ADDRESS") is None
    assert [scenario.name for scenario in view.scenarios()] == [
        s["name"] for s in _raw(demo_view, "scenarios.json")["scenarios"]
    ]
    assert view.scenario_dir("nominal") == demo_view / SCENARIOS_DIR / "nominal"
    assert view.document(VariablesDocument).schema_id == variables["schema"]


def test_a_channels_motion_envelope_follows_its_seed_and_the_active_scenarios(
    demo_view: Path,
) -> None:
    from osprey_connectors.simulation.envelope import active_envelopes, motion_envelope

    view = SimulatorView.open(demo_view)
    seeds = {
        address: SeedRecord.model_validate(seed)
        for address, seed in _raw(demo_view, "seeds.json")["seeds"].items()
    }
    scenarios = {
        entry["name"]: ScenarioRecord.model_validate(entry)
        for entry in _raw(demo_view, "scenarios.json")["scenarios"]
    }
    noisy = next(address for address, seed in sorted(seeds.items()) if seed.drift)
    live = active_envelopes(seeds, scenarios, ["rf-thermal-live"])
    replaced = sorted(scenarios["rf-thermal-live"].noise)[0]

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
        if record["engine_role"] == "monitor" and record["plane"] == "x"
    ]

    found = [binding.address for binding in view.bindings(engine_role="monitor", plane="x")]

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


@pytest.mark.parametrize("key", ["engine_role", "plane", "refresh", "direction", "id"])
def test_a_wiring_record_without_a_key_every_record_states_asks_for_a_rebuild(
    demo_view: Path, key: str
) -> None:
    variables = _raw(demo_view, VARIABLES_FILE)
    sr = next(model for model in variables["models"] if model["name"] == "SR")
    del sr["wiring"][0][key]
    (demo_view / VARIABLES_FILE).write_text(json.dumps(variables), encoding="utf-8")
    view = SimulatorView.open(demo_view)

    with pytest.raises(ViewSchemaError, match="rebuild") as refused:
        view.models()

    assert refused.value.key is not None
    assert refused.value.key.endswith(f"wiring.0.{key}")


def test_a_wiring_record_states_its_plane_even_when_it_has_none() -> None:
    stated = {
        "id": "SR/A",
        "address": "A",
        "direction": "read",
        "engine_role": "readback",
        "refresh": "pass",
    }

    with pytest.raises(ValidationError, match="plane"):
        WiringRecord.model_validate(stated)
    record = WiringRecord.model_validate({**stated, "plane": None})
    assert record.plane is None
    assert record.marks == ()
    assert record.engine_entry() == {**stated, "plane": None}


def test_each_engine_is_handed_the_wiring_and_settings_it_was_before(demo_view: Path) -> None:
    """The wiring dicts and settings every model's engine is built from, pinned at the base."""
    view = SimulatorView.open(demo_view)

    handed = {
        model.name: hashlib.sha256(
            json.dumps(
                {
                    "wiring": [record.engine_entry() for record in model.wiring],
                    "settings": model.engine_settings(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for model in view.models()
    }

    assert handed == {
        "BTS": "6c2c2271275b546505be62990be56b2bf2737793bd6142c0fbdd82792b1fd0d2",
        "LINE": "f4043f2998a233843d27cccaba5c6bdb93ab07605bf163a4e7d57085f011f268",
        "LTB": "5e967c0e009af7c1a292526a55439beda9f8543146f75d949ed849d3adf70cc3",
        "SR": "af27622ada5095a5b4379cdb0d97f4d219e23969a5b53c124ffe378a04e6690b",
        TEXTURE: "a1ed3f9cb28e83eb3c50d1fd0ea1905eb954c14b027eb79b9c9c4122a6d654bd",
    }


def test_every_document_writes_back_the_bytes_it_was_read_from(demo_view: Path) -> None:
    view = SimulatorView.open(demo_view)

    view.check()

    for kind in VIEW_DOCUMENTS:
        assert document_bytes(view.document(kind)) == (demo_view / kind.FILE).read_bytes()


def test_constructing_a_view_reads_nothing_and_check_reads_every_file(demo_view: Path) -> None:
    (demo_view / "seeds.json").unlink()

    unopened = SimulatorView(demo_view / "missing")
    assert unopened.path == demo_view / "missing"
    view = SimulatorView.open(demo_view)
    with pytest.raises(NoSimulatorViewError) as missing:
        view.check()
    assert missing.value.path == demo_view / "seeds.json"


def test_a_missing_view_is_no_view_and_find_says_none(tmp_path: Path) -> None:
    with pytest.raises(NoSimulatorViewError, match="run osprey build"):
        SimulatorView.open(tmp_path / "nowhere")

    assert SimulatorView.find(tmp_path) is None


def test_find_still_refuses_a_view_of_another_schema(tmp_path: Path) -> None:
    view_dir = tmp_path / VIEW_RELPATH
    view_dir.mkdir(parents=True)
    (view_dir / ADDRESSES_FILE).write_text(json.dumps({"schema": "old"}), encoding="utf-8")

    with pytest.raises(ViewSchemaError, match="rebuild"):
        SimulatorView.find(tmp_path)


def test_a_repo_and_its_render_open_the_same_view(built_demo: BuiltProject) -> None:
    repo = built_demo.repo

    of_repo = SimulatorView.of_project(repo)
    of_render = SimulatorView.of_render(repo / "build")

    assert of_repo.path == of_render.path == repo / "build" / VIEW_RELPATH
    assert of_repo.render_root == repo / "build"
    assert SimulatorView.of_project(repo / "build").path == of_repo.path
    assert view_path(repo) == SimulatorView.path_for_project(repo) == of_repo.path
    assert of_repo.channels() == of_render.channels()


#: What the reader imports beyond the standard library: pydantic, which its
#: document models are built on, and pydantic's own dependencies.
_READER_DEPENDENCIES = (
    "pydantic",
    "pydantic_core",
    "typing_extensions",
    "typing_inspection",
    "annotated_types",
)


def test_the_reader_adds_only_pydantic_and_loads_no_model() -> None:
    probe = (
        "import json, sys\n"
        "import osprey_connectors.simulation\n"
        "before = set(sys.modules)\n"
        "import osprey_connectors.simulation.view\n"
        "added = sorted(\n"
        "    name for name in set(sys.modules) - before\n"
        "    if name.split('.')[0] not in sys.stdlib_module_names\n"
        "    and not name.startswith('_sysconfigdata')\n"
        f"    and name.split('.')[0] not in {_READER_DEPENDENCIES!r}\n"
        "    and name.split('.')[0] != 'osprey_connectors'\n"
        ")\n"
        "roots = {name.split('.')[0] for name in sys.modules}\n"
        "print(json.dumps([added, sorted(roots & {'numpy', 'lume', 'at', 'osprey'})]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    )

    assert json.loads(result.stdout) == [[], []]
