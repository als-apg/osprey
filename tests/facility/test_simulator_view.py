"""The simulator view: what every render carries under ``data/simulator/``.

``render_facility_outputs`` writes ``served_models.json`` (the models the
render's ``simulation.models`` serves, texture last), ``addresses.json`` (every
channel address of the facility file and one status address per served physics
model), ``decks/<model><suffix>`` (a byte copy of each deck-bearing model's
deck under its own suffix, served or not), ``variables.json`` (every model with its wiring and every
channel with its write facts), ``seeds.json`` (each channel's seed record) and
``scenarios.json`` (every scenario's writes, with a byte copy of each file
its logbook entries attach under ``scenarios/<name>/``) into each render's
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
from osprey.utils.workspace import TIERS_DIRNAME
from osprey_connectors.simulation.view import SCENARIOS_SCHEMA, SEEDS_SCHEMA
from osprey_connectors.workspace import BUILD_DIR_NAME

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

pytestmark = [pytest.mark.slow]

SERVED = "data/simulator/served_models.json"
ADDRESSES = "data/simulator/addresses.json"
DECK = "data/simulator/decks/SR.json"
LINE_DECK = "data/simulator/decks/LINE.json"
#: The single-pass transfer lines' decks beside LINE's.
TRANSFER_DECKS = ("data/simulator/decks/BTS.json", "data/simulator/decks/LTB.json")
VARIABLES = "data/simulator/variables.json"
SEEDS = "data/simulator/seeds.json"
SCENARIOS = "data/simulator/scenarios.json"
SCENARIO_FILES = (
    "data/simulator/scenarios/bpm-polarity/plots/corrector_bump_test.png",
    "data/simulator/scenarios/nominal/plots/orbit_rms.json",
    "data/simulator/scenarios/rf-thermal/plots/cavity_temperatures.json",
)
LIMITS = "data/channel_limits.json"
FACTS = ("data/facility_facts.json", "data/facility_facts.md")
BLUESKY = "data/bluesky_devices.yml"
#: The hierarchical channel index, written only into a render that selects the
#: hierarchical channel-finder pipeline.
CHANNEL_INDEX = "data/channel_finder/hierarchical.json"
GRAPH = "data/graph/facility.ttl"
PYAML = (
    "data/pyaml/BTS/configuration.yaml",
    "data/pyaml/LINE/configuration.yaml",
    "data/pyaml/LTB/configuration.yaml",
    "data/pyaml/SR/configuration.yaml",
    "data/pyaml/SR/crm.json",
    "data/pyaml/SR/lattice.json",
    "data/pyaml/SR/trm.json",
)


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
        assert sorted(set(outputs.files) - {BLUESKY, CHANNEL_INDEX}) == sorted(
            [
                FACILITY_FILE,
                ADDRESSES,
                DECK,
                LINE_DECK,
                *TRANSFER_DECKS,
                SCENARIOS,
                *SCENARIO_FILES,
                SEEDS,
                SERVED,
                VARIABLES,
                LIMITS,
                *FACTS,
                GRAPH,
                *PYAML,
            ]
        )
    # The deployment's render runs a Bluesky lane; a render without one has no view.
    assert BLUESKY in built_control_assistant.outputs[0].files
    # Only the channel-finder app's render selects the hierarchical pipeline,
    # once in its own render and once in its image.
    assert [
        outputs.render_dir.name
        for outputs in built_control_assistant.outputs
        if CHANNEL_INDEX in outputs.files
    ] == ["demo-channel-finder"] * 2


def test_addresses_are_the_facility_file_channels_and_the_served_status(
    built_control_assistant: BuiltProject,
) -> None:
    addresses = json.loads((built_control_assistant.build_dir / ADDRESSES).read_bytes())
    facility = built_control_assistant.facility

    assert addresses["schema"] == "osprey.facility.addresses/1"
    assert addresses["channels"] == sorted(channel["id"] for channel in facility["channels"])
    assert addresses["status"] == [f"ca:SIM:{name}:STATUS" for name in ("BTS", "LINE", "LTB", "SR")]


def test_the_demo_serves_its_deck_models_then_texture(
    built_control_assistant: BuiltProject,
) -> None:
    served = json.loads((built_control_assistant.build_dir / SERVED).read_bytes())

    assert served == {
        "schema": "osprey.facility.served_models/1",
        "models": ["BTS", "LINE", "LTB", "SR", TEXTURE],
    }


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


def test_two_in_process_renders_differing_only_in_the_key_render_different_lists(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    from osprey.cli.build_profile_archiver import _expand_dotted
    from osprey.cli.build_profile_load import load_profile_document

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "profile.yml").write_text(
        "name: demo\ndata: data\nconfig:\n  control_system.type: virtual_accelerator\n"
        "  control_system.connector.virtual_accelerator.serving: in_process\n"
        "  simulation.models: null\n"
    )
    (repo / TIERS_DIRNAME).mkdir()
    for name, models in (("physics", "[SR]"), ("plain", "[texture]")):
        (repo / TIERS_DIRNAME / f"{name}.yml").write_text(
            f"name: {name}\nconfig:\n  simulation.models: {models}\n"
        )

    rendered = {}
    for name in ("physics", "plain"):
        document = load_profile_document(repo / TIERS_DIRNAME / f"{name}.yml")
        config = _expand_dotted(document.profile.config)
        assert config["control_system"]["type"] == "virtual_accelerator"
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
        written_when=lambda _inputs: (False, "stub.enabled"),
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
    "on",
}
CHANNEL_OPTIONAL = {"options", "shape", "precision"}
SCENARIO_SLOTS = ("description", "drivers", "couple", "noise", "still", "channel_faults")
BUILT_IN_STILL = {
    "name": "still",
    "description": "Every reading serves without drift, couplings or noise.",
    "still": "all",
}
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
    assert variables["schema"] == "osprey.facility.simulator/2"
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


#: The demo's served waveform channel: the SEK injection septum's current pulse.
SEK_WAVEFORM = "SR:MAG:SEPTUM:01:CURRENT:WAVEFORM"


def test_the_septum_waveform_is_a_texture_waveform_of_500_samples(
    built_control_assistant: BuiltProject,
) -> None:
    variables = _view(built_control_assistant.build_dir, VARIABLES)
    sek = _channel(variables, SEK_WAVEFORM)

    assert sek["value_type"] == "waveform"
    assert sek["shape"] == [500]
    assert sek["unit"] == "A"
    assert sek["role"] == "readback"
    assert sek["writable"] is False
    assert sek["owner"] == TEXTURE
    assert "500 samples" in sek["description"]


def test_the_septum_waveform_seed_is_its_pulse(built_control_assistant: BuiltProject) -> None:
    seeds = _view(built_control_assistant.build_dir, SEEDS)["seeds"]

    assert seeds[SEK_WAVEFORM] == {
        "pulse": {
            "driver": "SR:MAG:SEPTUM:01:CURRENT:SP",
            "enable": "SR:MAG:SEPTUM:01:STATUS:ON",
        }
    }
    assert seeds["SR:MAG:SEPTUM:01:CURRENT:SP"]["nominal"] == 3000.0


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


DESCRIPTION_KEYS = ("role", "plane", "refresh")


def test_every_wiring_entry_carries_the_engines_description(
    built_control_assistant: BuiltProject,
) -> None:
    from osprey.simulation.engines import pyat

    variables = _view(built_control_assistant.build_dir, VARIABLES)
    (sr,) = [model for model in variables["models"] if model["name"] == "SR"]

    roles = set()
    for entry in sr["wiring"]:
        record = {key: value for key, value in entry.items() if key not in DESCRIPTION_KEYS}
        described = pyat.describe(record)
        assert {key: entry[key] for key in DESCRIPTION_KEYS} == {
            key: described[key] for key in DESCRIPTION_KEYS
        }
        assert "kind" not in entry
        roles.add(entry["role"])
    assert roles == {"setpoint", "readback", "monitor", "output"}
    facility_sr = next(m for m in built_control_assistant.facility["models"] if m["name"] == "SR")
    assert {k: facility_sr["wiring"][0][k] for k in DESCRIPTION_KEYS} == {
        k: sr["wiring"][0][k] for k in DESCRIPTION_KEYS
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


def test_a_deck_keeps_its_suffix(tmp_path: Path, built_control_assistant: BuiltProject) -> None:
    import at

    facility_dir = tmp_path / "facility"
    shutil.copytree(built_control_assistant.facility_dir, facility_dir)
    source = facility_dir / "decks" / "SR.mat"
    at.save_lattice(at.load_lattice(str(facility_dir / "decks" / "SR.json")), str(source))
    facility = copy.deepcopy(built_control_assistant.facility)
    (sr,) = [model for model in facility["models"] if model["name"] == "SR"]
    sr["deck"] = "decks/SR.mat"
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    render_facility_outputs(render_dir, facility, {}, facility_dir)

    served = render_dir / "data" / "simulator" / "decks"
    assert sorted(path.name for path in served.iterdir()) == [
        "BTS.json",
        "LINE.json",
        "LTB.json",
        "SR.mat",
    ]
    assert (served / "SR.mat").read_bytes() == source.read_bytes()
    models = {m["name"]: m for m in _view(render_dir, VARIABLES)["models"]}
    assert models["SR"]["deck"] == "decks/SR.mat"


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
    assert {channel["owner"] for channel in variables["channels"]} == {
        "BTS",
        "LINE",
        "LTB",
        "SR",
        TEXTURE,
    }


def test_channel_on_is_the_facility_records_on(built_control_assistant: BuiltProject) -> None:
    variables = _view(built_control_assistant.build_dir, VARIABLES)
    records = {
        str(channel["id"]): channel for channel in built_control_assistant.facility["channels"]
    }

    for channel in variables["channels"]:
        assert channel["on"] == records[channel["address"]].get("on"), channel["address"]
    assert any(channel["on"] is not None for channel in variables["channels"])


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
            "type": "virtual_accelerator",
            "limits_checking": {"enabled": True, "mode": "optional"},
            "connector": {
                "virtual_accelerator": {
                    "serving": "in_process",
                    "limits_checking": {"enabled": True, "mode": "exclusive"},
                }
            },
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
    assert seeds["schema"] == SEEDS_SCHEMA
    assert seeds["seeds"] == expected
    assert list(seeds["seeds"]) == sorted(expected)


def test_scenarios_carry_every_scenario_and_its_blocks(
    built_control_assistant: BuiltProject,
) -> None:
    scenarios = _view(built_control_assistant.build_dir, SCENARIOS)
    source = built_control_assistant.facility_dir / "scenarios"
    names = sorted([*(path.stem for path in source.glob("*.yaml")), "still"])

    assert set(scenarios) == {"schema", "scenarios"}
    assert scenarios["schema"] == SCENARIOS_SCHEMA
    assert [entry["name"] for entry in scenarios["scenarios"]] == names
    for entry in scenarios["scenarios"]:
        if entry["name"] == "still":
            assert entry == BUILT_IN_STILL
            continue
        authored = yaml.safe_load((source / f"{entry['name']}.yaml").read_text())
        assert entry["description"] == authored["description"], entry["name"]
        for block in (
            "overrides",
            "archiver",
            "logbook",
            "drivers",
            "couple",
            "noise",
            "channel_faults",
        ):
            assert entry.get(block) == authored.get(block), (entry["name"], block)
        faults = {model: fault["writes"] for model, fault in entry.get("faults", {}).items()}
        assert faults == authored.get("faults", {}), entry["name"]
    (live,) = [entry for entry in scenarios["scenarios"] if entry["name"] == "rf-thermal-live"]
    authored = yaml.safe_load((source / "rf-thermal-live.yaml").read_text())
    for block in ("drivers", "couple", "noise"):
        assert live[block] == authored[block]


def test_each_attached_file_is_a_byte_copy_beside_the_scenarios(
    built_control_assistant: BuiltProject,
) -> None:
    source = built_control_assistant.facility_dir
    for relative in SCENARIO_FILES:
        copied = built_control_assistant.build_dir / relative
        assert (
            copied.read_bytes() == (source / relative.removeprefix("data/simulator/")).read_bytes()
        )


def test_a_render_leaves_out_the_copy_of_a_file_no_entry_attaches_any_more(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    """Once nominal's entries attach nothing, its copy is not written; the others still are.

    A rebuild replaces ``build/`` wholesale (``tests/cli/test_build_swap.py``,
    ``test_the_second_build_replaces_the_first_wholesale``), so the render decides.
    """
    facility = copy.deepcopy(built_control_assistant.facility)
    (nominal,) = [s for s in facility["scenarios"] if s["name"] == "nominal"]
    assert any(entry.get("attachments") for entry in nominal["logbook"])
    for entry in nominal["logbook"]:
        entry.pop("attachments", None)
    render_dir = tmp_path / "render"
    render_dir.mkdir()

    written = render_facility_outputs(
        render_dir, facility, {}, built_control_assistant.facility_dir
    )

    copies = [
        path.relative_to(render_dir).as_posix()
        for path in written
        if path.relative_to(render_dir).as_posix().startswith("data/simulator/scenarios/")
    ]
    assert copies == [relative for relative in SCENARIO_FILES if "/nominal/" not in relative]
    assert not (render_dir / "data/simulator/scenarios/nominal").exists()


def test_a_file_two_entries_attach_is_copied_and_listed_once(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    facility = copy.deepcopy(built_control_assistant.facility)
    (nominal,) = [s for s in facility["scenarios"] if s["name"] == "nominal"]
    (attached,) = [e for e in nominal["logbook"] if e.get("attachments")]
    twin = {**copy.deepcopy(attached), "entry_id": "TWIN"}
    nominal["logbook"].append(twin)
    render_dir = tmp_path / "render"
    render_dir.mkdir()

    written = render_facility_outputs(
        render_dir, facility, {}, built_control_assistant.facility_dir
    )

    copied = render_dir / "data/simulator/scenarios/nominal/plots/orbit_rms.json"
    assert written.count(copied) == 1


def test_a_withdrawn_scenario_is_neither_listed_nor_copied(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    from osprey.facility.views.simulator import withdraw_scenarios

    render_dir = _render(tmp_path, built_control_assistant, {})
    view = render_dir / "data/simulator"
    listed = [entry["name"] for entry in _view(render_dir, SCENARIOS)["scenarios"]]
    others = sorted(
        path for path in view.rglob("*") if path.is_file() and "scenarios/nominal" not in str(path)
    )
    before = {path: path.read_bytes() for path in others if path.name != "scenarios.json"}

    withdraw_scenarios(view, ["nominal"])

    document = _view(render_dir, SCENARIOS)
    assert document["schema"] == SCENARIOS_SCHEMA
    assert [entry["name"] for entry in document["scenarios"]] == [
        name for name in listed if name != "nominal"
    ]
    assert not (view / "scenarios" / "nominal").exists()
    assert {path: path.read_bytes() for path in before} == before


def test_withdrawing_an_unlisted_scenario_withdraws_nothing(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    from osprey.facility.views.simulator import withdraw_scenarios

    render_dir = _render(tmp_path, built_control_assistant, {})
    view = render_dir / "data/simulator"
    listing = (render_dir / SCENARIOS).read_bytes()

    with pytest.raises(KeyError, match="no-such-scenario"):
        withdraw_scenarios(view, ["nominal", "no-such-scenario"])

    assert (render_dir / SCENARIOS).read_bytes() == listing
    assert (view / "scenarios" / "nominal").is_dir()


def test_a_bad_attachment_in_any_scenario_stops_before_any_file_is_copied(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    facility = copy.deepcopy(built_control_assistant.facility)
    (vacuum,) = [s for s in facility["scenarios"] if s["name"] == "vacuum-burst"]
    vacuum["logbook"] = [
        {
            "entry_id": "E-1",
            "when": {"days_ago": 1, "time": "02:00:00"},
            "author": "A. Author",
            "title": "Title",
            "text": "Body",
            "attachments": [{"plot": "plots/absent.json"}],
        }
    ]
    render_dir = tmp_path / "render"
    render_dir.mkdir()

    with pytest.raises(ValueError, match="plots/absent.json"):
        render_facility_outputs(render_dir, facility, {}, built_control_assistant.facility_dir)

    assert not (render_dir / "data/simulator/scenarios").exists()


def test_an_attachment_naming_a_missing_file_is_refused(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    facility = copy.deepcopy(built_control_assistant.facility)
    entry = {
        "entry_id": "E-1",
        "when": {"days_ago": 1, "time": "02:00:00"},
        "author": "A. Author",
        "title": "Title",
        "text": "Body",
        "attachments": [{"plot": "plots/absent.json"}],
    }
    facility["scenarios"] = [{"name": "bare", "logbook": [entry]}]
    render_dir = tmp_path / "render"
    render_dir.mkdir()

    with pytest.raises(ValueError, match="plots/absent.json"):
        render_facility_outputs(render_dir, facility, {}, built_control_assistant.facility_dir)


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

    (entry,) = [e for e in _view(render_dir, SCENARIOS)["scenarios"] if e["name"] == "bare"]
    assert entry == {"name": "bare", "overrides": {UNLISTED_SETPOINT: 1.0}}
    assert not set(SCENARIO_SLOTS) & set(entry)


def _rendered_scenarios(
    tmp_path: Path, built: BuiltProject, scenarios: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    facility = copy.deepcopy(built.facility)
    facility["scenarios"] = scenarios
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    render_facility_outputs(render_dir, facility, {}, built.facility_dir)
    return _view(render_dir, SCENARIOS)["scenarios"]


def test_the_built_in_still_is_listed_when_no_file_defines_one(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    scenarios = _rendered_scenarios(
        tmp_path, built_control_assistant, [{"name": "zeta"}, {"name": "alpha"}]
    )

    assert scenarios == [{"name": "alpha"}, BUILT_IN_STILL, {"name": "zeta"}]


def test_channel_faults_are_carried_verbatim_on_a_texture_address(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    texture = next(
        record["address"]
        for record in _view(built_control_assistant.build_dir, VARIABLES)["channels"]
        if record["owner"] == TEXTURE and record["role"] == "readback"
    )
    scenario = {"name": "cut", "channel_faults": {texture: "disconnected"}}

    (entry,) = [
        e
        for e in _rendered_scenarios(tmp_path, built_control_assistant, [scenario])
        if e["name"] == "cut"
    ]
    assert entry == scenario


def test_a_files_still_replaces_the_built_in(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    own = {"name": "still", "description": "Monitors only.", "still": ["SR:DIAG:BPM:01:POSITION"]}

    assert _rendered_scenarios(tmp_path, built_control_assistant, [own]) == [own]


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


def test_a_float_channel_stating_precision_carries_it(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    variables = _render_channels(
        tmp_path, built_control_assistant, {UNLISTED_SETPOINT: {"precision": 3}}, ()
    )

    assert _channel(variables, UNLISTED_SETPOINT)["precision"] == 3


def test_a_setpoint_stating_a_null_pair_pairs_with_itself(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    variables = _render_channels(
        tmp_path, built_control_assistant, {UNLISTED_SETPOINT: {"pair": None}}, ()
    )

    assert _channel(variables, UNLISTED_SETPOINT)["pair"] == UNLISTED_SETPOINT


def _with_channel(built: BuiltProject, address: str) -> dict[str, Any]:
    facility = copy.deepcopy(built.facility)
    facility["channels"].append({"id": address})
    return facility


@pytest.mark.parametrize(
    "config",
    [
        {"deployed_services": ["archiver_recorder"]},
        {"services": {"archiver_recorder": {"path": "./services/archiver_recorder"}}},
    ],
    ids=["deployed", "projected"],
)
@pytest.mark.parametrize("address", ["BPM1.X", "$LAB:TEMP", "LAB%TEMP"])
def test_a_configured_recorder_serves_any_address(
    tmp_path: Path, built_control_assistant: BuiltProject, config: dict[str, Any], address: str
) -> None:
    facility = _with_channel(built_control_assistant, address)

    render_facility_outputs(tmp_path, facility, config, built_control_assistant.facility_dir)

    assert address in _view(tmp_path, ADDRESSES)["channels"]
