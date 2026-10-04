"""The simulator view: what every render carries under ``data/simulator/``.

``render_facility_outputs`` writes ``served_models.json`` (the models the
render's ``simulation.models`` serves, texture last), ``addresses.json`` (every
channel address of the facility file and one status address per served physics
model), ``decks/<model>.json`` (a byte copy of each deck-bearing model's deck,
served or not), ``variables.json`` (every model with its wiring and every
channel with its write facts), ``seeds.json`` (each channel's seed record) and
``scenarios.json`` (every scenario's writes) into each render's
``data/simulator/``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.facility import TEXTURE
from osprey.facility.render import FACILITY_FILE, render_facility_outputs
from osprey.utils.workspace import BUILD_DIR_NAME

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

pytestmark = [pytest.mark.slow]

SERVED = "data/simulator/served_models.json"
ADDRESSES = "data/simulator/addresses.json"
DECK = "data/simulator/decks/SR.json"
VARIABLES = "data/simulator/variables.json"
SEEDS = "data/simulator/seeds.json"
SCENARIOS = "data/simulator/scenarios.json"
LIMITS = "data/channel_limits.json"
FACTS = ("data/facility_facts.json", "data/facility_facts.md")
BLUESKY = "data/bluesky_devices.yml"
GRAPH = "data/graph/facility.ttl"


def _render(
    tmp_path: Path, built: BuiltProject, config: dict[str, Any], name: str = "render"
) -> Path:
    render_dir = tmp_path / name
    render_dir.mkdir()
    render_facility_outputs(render_dir, built.facility, config, built.facility_dir)
    return render_dir


def test_every_render_writes_the_simulator_files_beside_the_limits(
    built_control_assistant: BuiltProject,
) -> None:
    assert built_control_assistant.outputs
    for outputs in built_control_assistant.outputs:
        assert sorted(set(outputs.files) - {BLUESKY}) == sorted(
            [
                FACILITY_FILE,
                ADDRESSES,
                DECK,
                SCENARIOS,
                SEEDS,
                SERVED,
                VARIABLES,
                LIMITS,
                *FACTS,
                GRAPH,
            ]
        )
    # The deployment's render runs a Bluesky lane; a persona without one has no view.
    assert BLUESKY in built_control_assistant.outputs[0].files


def test_addresses_are_the_facility_file_channels_and_the_served_status(
    built_control_assistant: BuiltProject,
) -> None:
    addresses = json.loads((built_control_assistant.build_dir / ADDRESSES).read_bytes())
    facility = built_control_assistant.facility

    assert addresses["schema"] == "osprey.facility.addresses/1"
    assert addresses["channels"] == sorted(channel["id"] for channel in facility["channels"])
    assert addresses["status"] == ["ca:SIM:SR:STATUS"]


def test_the_demo_serves_sr_then_texture(built_control_assistant: BuiltProject) -> None:
    served = json.loads((built_control_assistant.build_dir / SERVED).read_bytes())

    assert served == {"schema": "osprey.facility.served_models/1", "models": ["SR", TEXTURE]}


def test_the_deck_is_a_byte_copy_that_pyat_loads(built_control_assistant: BuiltProject) -> None:
    at = pytest.importorskip("at")
    copy = built_control_assistant.build_dir / DECK
    source = built_control_assistant.facility_dir / "decks" / "SR.json"

    assert hashlib.sha256(copy.read_bytes()).hexdigest() == (
        hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert len(at.load_lattice(str(copy))) == 802


@pytest.mark.parametrize("models", [[], [TEXTURE]])
def test_no_physics_serves_texture_alone_and_keeps_the_deck(
    tmp_path: Path, built_control_assistant: BuiltProject, models: list[str]
) -> None:
    render_dir = _render(tmp_path, built_control_assistant, {"simulation": {"models": models}})

    served = json.loads((render_dir / SERVED).read_bytes())
    addresses = json.loads((render_dir / ADDRESSES).read_bytes())
    assert served["models"] == [TEXTURE]
    assert addresses["status"] == []
    assert (render_dir / DECK).read_bytes() == (
        built_control_assistant.build_dir / DECK
    ).read_bytes()


def test_render_writes_are_sorted_and_deterministic(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    config = yaml.safe_load((built_control_assistant.build_dir / "config.yml").read_text())
    first = _render(tmp_path, built_control_assistant, config, "first")
    second = _render(tmp_path, built_control_assistant, config, "second")

    for relative in (SERVED, ADDRESSES, DECK, VARIABLES, SEEDS, SCENARIOS):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
        assert (first / relative).read_bytes() == (
            built_control_assistant.build_dir / relative
        ).read_bytes()


def test_two_mock_personas_differing_only_in_the_key_render_different_lists(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    from osprey.cli.build_profile_archiver import _expand_dotted
    from osprey.cli.build_profile_load import load_profile_document

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "profile.yml").write_text(
        "name: demo\ndata: data\nconfig:\n  control_system.type: mock\n  simulation.models: null\n"
    )
    (repo / "personas").mkdir()
    for name, models in (("physics", "[SR]"), ("plain", "[texture]")):
        (repo / "personas" / f"{name}.yml").write_text(
            f"name: {name}\nconfig:\n  simulation.models: {models}\n"
        )

    rendered = {}
    for name in ("physics", "plain"):
        document = load_profile_document(repo / "personas" / f"{name}.yml")
        config = _expand_dotted(document.profile.config)
        assert config["control_system"]["type"] == "mock"
        render_dir = _render(tmp_path, built_control_assistant, config, name)
        rendered[name] = (render_dir / SERVED).read_bytes()

    assert json.loads(rendered["physics"])["models"] == ["SR", TEXTURE]
    assert json.loads(rendered["plain"])["models"] == [TEXTURE]
    assert rendered["physics"] != rendered["plain"]


def test_an_omitted_view_is_named_on_stderr_alone(
    tmp_path: Path,
    built_control_assistant: BuiltProject,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osprey.facility import views

    stub = views.View(
        name="stub",
        path="stub",
        written_when=lambda _inputs: False,
        reason="stub.enabled",
        write=lambda _root, _inputs: pytest.fail("an omitted view is never written"),
    )
    monkeypatch.setattr(views, "VIEWS", (*views.VIEWS, stub))

    render_dir = _render(tmp_path, built_control_assistant, {})

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "view stub not written: stub.enabled" in captured.err
    assert not (render_dir / "data" / "stub").exists()
    assert (render_dir / SERVED).is_file()


def test_validate_names_an_unknown_served_model(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    from osprey.cli.main import cli

    source = built_control_assistant.repo
    repo = tmp_path / source.name
    shutil.copytree(
        source,
        repo,
        symlinks=True,
        ignore=lambda folder, _names: [BUILD_DIR_NAME] if Path(folder) == source else [],
    )
    profile = repo / "profile.yml"
    document = yaml.safe_load(profile.read_text(encoding="utf-8"))
    document.setdefault("config", {})["simulation.models"] = ["NOPE"]
    profile.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(cli, ["facility", "validate", "--repo", str(repo)])

    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr.startswith("facility: profile-invalid: path simulation.models — ")
    assert "`NOPE`" in result.stderr


MODEL_KEYS = {"name", "engine", "served", "settings", "deck", "wiring"}
CHANNEL_KEYS = {
    "address",
    "role",
    "pair",
    "value_type",
    "unit",
    "description",
    "writable",
    "value_range",
    "owner",
}
CHANNEL_OPTIONAL = {"options", "shape", "precision"}
SCENARIO_SLOTS = ("description", "drivers", "couple", "noise")
UNLISTED_SETPOINT = "SR:MAG:HCM:02:CURRENT:SP"


def _view(root: Path, relative: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((root / relative).read_bytes())
    return document


def _limits(mode: str) -> dict[str, Any]:
    return {"control_system": {"limits_checking": {"enabled": True, "mode": mode}}}


def _channel(variables: dict[str, Any], address: str) -> dict[str, Any]:
    (found,) = [channel for channel in variables["channels"] if channel["address"] == address]
    return found


def test_variables_carries_its_schema_and_key_sets(built_control_assistant: BuiltProject) -> None:
    variables = _view(built_control_assistant.build_dir, VARIABLES)
    facility = built_control_assistant.facility

    assert set(variables) == {"schema", "code", "models", "channels"}
    assert variables["schema"] == "osprey.facility.simulator/1"
    assert variables["code"] == facility["identity"]["code"]
    assert [model["name"] for model in variables["models"]] == sorted(
        model["name"] for model in facility["models"]
    )
    for model in variables["models"]:
        assert set(model) == MODEL_KEYS, model["name"]
    addresses = [channel["address"] for channel in variables["channels"]]
    assert addresses == sorted(str(channel["id"]) for channel in facility["channels"])
    for channel in variables["channels"]:
        assert CHANNEL_KEYS <= set(channel) <= CHANNEL_KEYS | CHANNEL_OPTIONAL, channel["address"]
        assert isinstance(channel["writable"], bool), channel["address"]


def test_the_demo_wiring_is_simulator_wiring_serialised(
    built_control_assistant: BuiltProject,
) -> None:
    from osprey.facility.views.simulator import simulator_wiring

    variables = _view(built_control_assistant.build_dir, VARIABLES)
    facility = built_control_assistant.facility
    models = {model["name"]: model for model in variables["models"]}

    for name, model in models.items():
        assert model["wiring"] == simulator_wiring(facility, name), name
    assert len(models["SR"]["wiring"]) > 0
    assert models["SR"] | {"wiring": None} == {
        "name": "SR",
        "engine": "pyat",
        "served": True,
        "settings": {"pyat": {"solve": "periodic"}},
        "deck": "decks/SR.json",
        "wiring": None,
    }


def test_a_model_without_wiring_records_has_empty_wiring(
    built_control_assistant: BuiltProject,
) -> None:
    variables = _view(built_control_assistant.build_dir, VARIABLES)
    (texture,) = [model for model in variables["models"] if model["name"] == TEXTURE]

    assert texture["wiring"] == []
    assert texture["deck"] is None
    assert texture["served"] is True


def test_an_unserved_model_is_listed_as_not_served(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    render_dir = _render(tmp_path, built_control_assistant, {"simulation": {"models": [TEXTURE]}})
    models = {m["name"]: m for m in _view(render_dir, VARIABLES)["models"]}

    assert models["SR"]["served"] is False
    assert models["SR"]["deck"] == "decks/SR.json"
    assert models[TEXTURE]["served"] is True


def test_channel_owner_is_the_wiring_model_else_texture(
    built_control_assistant: BuiltProject,
) -> None:
    variables = _view(built_control_assistant.build_dir, VARIABLES)
    wired = {
        entry["address"]: model["name"]
        for model in variables["models"]
        for entry in model["wiring"]
    }

    for channel in variables["channels"]:
        assert channel["owner"] == wired.get(channel["address"], TEXTURE), channel["address"]
    assert {channel["owner"] for channel in variables["channels"]} == {"SR", TEXTURE}


def test_channel_facts_come_from_the_facility_record(
    built_control_assistant: BuiltProject,
) -> None:
    variables = _view(built_control_assistant.build_dir, VARIABLES)
    records = {
        str(channel["id"]): channel for channel in built_control_assistant.facility["channels"]
    }

    for channel in variables["channels"]:
        record = records[channel["address"]]
        assert channel["role"] == record["role"]
        assert channel["value_type"] == record["value_type"]
        assert channel["unit"] == record.get("unit")
        assert channel["description"] == record.get("description")
        assert channel.get("options") == record.get("options")
        assert channel.get("shape") == record.get("shape")
        if record["role"] == "setpoint":
            assert channel["pair"] == record.get("pair", channel["address"])
        else:
            assert channel["pair"] is None


def test_writable_and_value_range_come_from_the_limits_records(
    built_control_assistant: BuiltProject,
) -> None:
    variables = _view(built_control_assistant.build_dir, VARIABLES)

    bounded = _channel(variables, "SR:MAG:HCM:01:CURRENT:SP")
    stepped = _channel(variables, "SR:RF:CAVITY:01:FREQUENCY:SP")
    blocked = _channel(variables, "SR:VAC:ION-PUMP:01:VOLTAGE:SP")
    assert (bounded["writable"], bounded["value_range"]) == (True, [-12.0, 12.0])
    assert (stepped["writable"], stepped["value_range"]) == (True, [500.0, 500.8])
    assert (blocked["writable"], blocked["value_range"]) == (False, None)
    for channel in variables["channels"]:
        if channel["role"] != "setpoint":
            assert channel["writable"] is False, channel["address"]


@pytest.mark.parametrize(
    ("config", "writable"),
    [(_limits("optional"), True), (_limits("exclusive"), False), ({}, False)],
    ids=["optional", "exclusive", "unstated"],
)
def test_a_setpoint_absent_from_limits_is_writable_only_under_optional(
    tmp_path: Path,
    built_control_assistant: BuiltProject,
    config: dict[str, Any],
    writable: bool,
) -> None:
    listed = {record["address"] for record in built_control_assistant.facility["limits"]["records"]}
    assert UNLISTED_SETPOINT not in listed

    render_dir = _render(tmp_path, built_control_assistant, config)
    channel = _channel(_view(render_dir, VARIABLES), UNLISTED_SETPOINT)

    assert channel["role"] == "setpoint"
    assert channel["writable"] is writable
    assert channel["value_range"] is None


def test_a_per_type_limits_block_decides_for_the_simulated_target(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    config = {
        "control_system": {
            "type": "mock",
            "limits_checking": {"enabled": True, "mode": "optional"},
            "connector": {"mock": {"limits_checking": {"enabled": True, "mode": "exclusive"}}},
        }
    }
    render_dir = _render(tmp_path, built_control_assistant, config)

    assert _channel(_view(render_dir, VARIABLES), UNLISTED_SETPOINT)["writable"] is False


def test_seeds_are_each_channel_seed_record(built_control_assistant: BuiltProject) -> None:
    seeds = _view(built_control_assistant.build_dir, SEEDS)
    expected = {
        str(channel["id"]): channel["simulation"]
        for channel in built_control_assistant.facility["channels"]
        if channel.get("simulation") is not None
    }

    assert set(seeds) == {"schema", "seeds"}
    assert seeds["schema"] == "osprey.facility.seeds/1"
    assert seeds["seeds"] == expected
    assert list(seeds["seeds"]) == sorted(expected)


def test_scenarios_carry_every_scenario_and_its_blocks(
    built_control_assistant: BuiltProject,
) -> None:
    scenarios = _view(built_control_assistant.build_dir, SCENARIOS)
    source = built_control_assistant.facility_dir / "scenarios"
    names = sorted(path.stem for path in source.glob("*.yaml"))

    assert set(scenarios) == {"schema", "scenarios"}
    assert scenarios["schema"] == "osprey.facility.scenarios/1"
    assert [entry["name"] for entry in scenarios["scenarios"]] == names
    for entry in scenarios["scenarios"]:
        authored = yaml.safe_load((source / f"{entry['name']}.yaml").read_text())
        assert entry["description"] == authored["description"], entry["name"]
        for block in ("overrides", "archiver", "logbook", "drivers", "couple", "noise"):
            assert entry.get(block) == authored.get(block), (entry["name"], block)
        faults = {model: fault["writes"] for model, fault in entry.get("faults", {}).items()}
        assert faults == authored.get("faults", {}), entry["name"]
    (live,) = [entry for entry in scenarios["scenarios"] if entry["name"] == "rf-thermal-live"]
    authored = yaml.safe_load((source / "rf-thermal-live.yaml").read_text())
    for block in ("drivers", "couple", "noise"):
        assert live[block] == authored[block]


def test_a_fault_on_an_unserved_model_is_inactive(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    served = {
        entry["name"]: entry
        for entry in _view(built_control_assistant.build_dir, SCENARIOS)["scenarios"]
    }
    render_dir = _render(tmp_path, built_control_assistant, {"simulation": {"models": [TEXTURE]}})
    unserved = {entry["name"]: entry for entry in _view(render_dir, SCENARIOS)["scenarios"]}

    assert "inactive" not in served["orm-dual-fault"]["faults"]["SR"]
    assert unserved["orm-dual-fault"]["faults"]["SR"]["inactive"] == "model not served"
    assert (
        unserved["orm-dual-fault"]["faults"]["SR"]["writes"]
        == served["orm-dual-fault"]["faults"]["SR"]["writes"]
    )
    assert "inactive" not in str(unserved["rf-thermal-live"])


def test_a_scenario_stating_none_of_the_optional_slots_carries_none_of_them(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    facility = copy.deepcopy(built_control_assistant.facility)
    facility["scenarios"] = [{"name": "bare", "overrides": {UNLISTED_SETPOINT: 1.0}}]
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    render_facility_outputs(render_dir, facility, {}, built_control_assistant.facility_dir)

    (entry,) = _view(render_dir, SCENARIOS)["scenarios"]
    assert entry == {"name": "bare", "overrides": {UNLISTED_SETPOINT: 1.0}}
    assert not set(SCENARIO_SLOTS) & set(entry)


def _render_channels(
    tmp_path: Path, built: BuiltProject, edit: dict[str, dict[str, Any]], drop: tuple[str, ...]
) -> dict[str, Any]:
    facility = copy.deepcopy(built.facility)
    for record in facility["channels"]:
        if str(record["id"]) in edit:
            for key in drop:
                record.pop(key, None)
            record.update(edit[str(record["id"])])
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    render_facility_outputs(render_dir, facility, {}, built.facility_dir)
    return _view(render_dir, VARIABLES)


def test_a_channel_omitting_role_and_value_type_takes_their_defaults(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    readback = next(
        str(record["id"])
        for record in built_control_assistant.facility["channels"]
        if record.get("role") == "readback"
    )
    variables = _render_channels(
        tmp_path, built_control_assistant, {readback: {}}, ("role", "value_type")
    )

    channel = _channel(variables, readback)
    assert (channel["role"], channel["value_type"], channel["pair"]) == ("readback", "float", None)


def test_a_setpoint_stating_a_null_pair_pairs_with_itself(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    variables = _render_channels(
        tmp_path, built_control_assistant, {UNLISTED_SETPOINT: {"pair": None}}, ()
    )

    assert _channel(variables, UNLISTED_SETPOINT)["pair"] == UNLISTED_SETPOINT
