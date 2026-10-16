"""The trajectory service starts on one target with one command set, or refuses naming why.

Every start here runs against a rendered build: the facility file is built and
rendered, the config beside the render serves the simulator in process, and the
service reads the ``facility_facts.json`` and the limits file the build wrote.
Only the raw-put block is recorded rather than installed, because installing it
would patch the client libraries of the whole test process.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

pytest.importorskip("at")

from tests.facility._pyaml_trees import measured_tree, with_correctors
from tests.facility._synthetic_trees import with_shot_wiring, with_trajectory, write_tree
from tests.facility.served_tree import in_process_config

from osprey.services.trajectory import startup
from osprey.services.trajectory.startup import (
    GUARDED_RUN_DIR_ENV_VAR,
    MODEL_KEY,
    TARGET_KEY,
    StartupRefused,
    TrajectoryCommands,
    select_commands,
)
from osprey.utils.workspace import reset_config_cache
from osprey_connectors.posture_store import CONTROL_CONTEXT_TREE_ENV_VAR

if TYPE_CHECKING:
    from tests._builds import BuiltProject

MODEL = "SR"
ARM = "SR:BPM:ARM"
REQUEST = "SR:INJ:REQ"
BUSY = "SR:INJ:BUSY"
LIMITS = Path("data") / "channel_limits.json"
FACTS = Path("data") / "facility_facts.json"


def _tree() -> dict[str, Any]:
    tree = with_trajectory(with_correctors(measured_tree()), MODEL, "SR/BPM1", marker="M1")
    with_shot_wiring(tree, MODEL, "BPM1")
    tree["limits.yaml"] = {
        "records": [
            {"address": ARM, "min_value": 0, "max_value": 1, "confirm": False},
            {"address": REQUEST, "min_value": 0, "max_value": 1000, "confirm": False},
        ]
    }
    return tree


def _render(repo: Path) -> Path:
    from osprey.facility.build import build_facility
    from osprey.facility.render import render_facility_outputs

    facility = repo / "data" / "facility"
    write_tree(facility, _tree())
    render = repo / "build"
    render.mkdir()
    document = build_facility(facility, project_name="served")
    render_facility_outputs(render, document, _config(repo), facility, omitted_reported=set())
    return render


def _config(
    repo: Path,
    *,
    writes_enabled: bool = True,
    limits: bool = True,
    service: Any = None,
) -> dict[str, Any]:
    view = repo / "build" / "data" / "simulator"
    config: dict[str, Any] = {
        "project_root": str(repo),
        "control_system": {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": in_process_config(view, response_delay_ms=0)},
            "writes_enabled": writes_enabled,
            "limits_checking": {
                "enabled": limits,
                "database_path": LIMITS.as_posix(),
                "mode": "exclusive",
                "on_violation": "error",
            },
        },
    }
    if service is not None:
        config["services"] = {"trajectory": service}
    return config


def _deploy(repo: Path, monkeypatch: pytest.MonkeyPatch, **settings: Any) -> None:
    path = repo / "build" / "config.yml"
    path.write_text(yaml.safe_dump(_config(repo, **settings)), encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(path))
    reset_config_cache()


def _write_slots(repo: Path, slots: dict[str, Any]) -> None:
    path = repo / "build" / FACTS
    facts = json.loads(path.read_text(encoding="utf-8"))
    facts["trajectory_slots"] = slots
    path.write_text(json.dumps(facts), encoding="utf-8")


@pytest.fixture(autouse=True)
def armed(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Each install of the raw-put block, recorded instead of performed."""
    calls: list[int] = []
    monkeypatch.setattr(startup, "_RAW_PUT_ARMED", False)
    monkeypatch.setattr(startup, "_install_raw_put_block", lambda: calls.append(1))
    return calls


def _shared_dirs(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = repo / "_tree"
    tree.mkdir(exist_ok=True)
    monkeypatch.setenv(CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    monkeypatch.setenv(GUARDED_RUN_DIR_ENV_VAR, str(repo / "var" / "guarded_run"))


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A built deployment on the simulator in process, writes armed and limits checked."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "profile.yml").write_text("name: served\n", encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.delenv("OSPREY_CONTROL_TARGET", raising=False)
    _render(root)
    _deploy(root, monkeypatch)
    _shared_dirs(root, monkeypatch)
    yield root
    reset_config_cache()


async def _start() -> startup.TrajectoryStartup:
    return await startup.start()


async def _started() -> startup.TrajectoryStartup:
    started = await _start()
    await started.connector.disconnect()
    return started


# -- starting -------------------------------------------------------------------


async def test_starts_on_the_baseline_with_the_one_models_commands(
    repo: Path, armed: list[int]
) -> None:
    started = await _start()
    try:
        assert started.target == "va"
        assert started.connector._control_target == "va"
        assert started.connector._connector_type == "virtual_accelerator"
        assert started.commands == TrajectoryCommands(
            model=MODEL, arm=(ARM,), injection_request=REQUEST, injection_busy=BUSY
        )
        assert started.commands.commands == frozenset({ARM, REQUEST})
        assert started.writing_available
        assert started.guarded_run_dir == repo / "var" / "guarded_run" / "va"
        assert armed == [1]
    finally:
        await started.connector.disconnect()


@pytest.mark.usefixtures("repo")
async def test_the_raw_put_block_is_armed_once(armed: list[int]) -> None:
    await _started()
    await _started()

    assert armed == [1]


async def test_the_named_target_is_the_connectors(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _deploy(repo, monkeypatch, service={"target": "va", "model": MODEL})

    started = await _started()

    assert started.target == "va"
    assert started.commands is not None and started.commands.model == MODEL


async def test_no_trajectory_model_starts_with_writing_unavailable(
    repo: Path, armed: list[int]
) -> None:
    _write_slots(repo, {})

    started = await _started()

    assert started.commands is None
    assert not started.writing_available
    assert armed == [1]


# -- the target -----------------------------------------------------------------


@pytest.mark.parametrize("target", ["sim", "", "LIVE", 3])
async def test_a_target_outside_the_control_targets_is_refused(
    repo: Path, monkeypatch: pytest.MonkeyPatch, target: Any
) -> None:
    _deploy(repo, monkeypatch, service={"target": target})

    with pytest.raises(StartupRefused, match=TARGET_KEY):
        await _start()


async def test_live_on_a_deployment_with_no_live_machine_is_refused(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _deploy(repo, monkeypatch, service={"target": "live"})
    monkeypatch.setenv(GUARDED_RUN_DIR_ENV_VAR, str(repo / "var" / "guarded_run"))

    with pytest.raises(StartupRefused, match=TARGET_KEY):
        await _start()


# -- the limits file ------------------------------------------------------------


async def test_a_missing_limits_file_refuses_an_armed_checking_target(
    repo: Path, armed: list[int]
) -> None:
    missing = repo / "build" / LIMITS
    missing.unlink()

    with pytest.raises(StartupRefused) as refused:
        await _start()

    assert str(missing) in str(refused.value)
    assert armed == []


async def test_an_unconfigured_limits_file_refuses_an_armed_checking_target(repo: Path) -> None:
    config = _config(repo)
    del config["control_system"]["limits_checking"]["database_path"]
    path = repo / "build" / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    reset_config_cache()

    with pytest.raises(StartupRefused, match="database_path"):
        await _start()


@pytest.mark.parametrize(
    "settings", [{"writes_enabled": False}, {"limits": False}], ids=["read-only", "unchecked"]
)
async def test_a_missing_limits_file_is_no_fault_without_the_pairing(
    repo: Path, monkeypatch: pytest.MonkeyPatch, settings: dict[str, bool]
) -> None:
    (repo / "build" / LIMITS).unlink()
    _deploy(repo, monkeypatch, **settings)

    started = await _started()

    assert started.commands is not None


# -- the shared directories -----------------------------------------------------


@pytest.mark.parametrize("variable", [CONTROL_CONTEXT_TREE_ENV_VAR, GUARDED_RUN_DIR_ENV_VAR])
@pytest.mark.parametrize("value", [None, "  "], ids=["unset", "blank"])
@pytest.mark.usefixtures("repo")
async def test_an_unset_shared_directory_is_refused_naming_it(
    monkeypatch: pytest.MonkeyPatch,
    armed: list[int],
    variable: str,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)

    with pytest.raises(StartupRefused, match=variable):
        await _start()

    assert armed == []


async def test_a_guarded_run_dir_that_is_not_the_targets_parent_is_refused(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv(GUARDED_RUN_DIR_ENV_VAR, str(elsewhere))

    with pytest.raises(StartupRefused) as refused:
        await _start()

    message = str(refused.value)
    assert GUARDED_RUN_DIR_ENV_VAR in message
    assert str(elsewhere) in message
    assert str(repo / "var" / "guarded_run" / "va") in message


# -- the command set ------------------------------------------------------------


def _slots(prefix: str) -> dict[str, Any]:
    return {
        "arm": [f"{prefix}:BPM:ARM"],
        "armed": [f"{prefix}:BPM:ARMED"],
        "injection_request": f"{prefix}:INJ:REQ",
        "injection_busy": f"{prefix}:INJ:BUSY",
    }


def test_no_trajectory_model_selects_nothing() -> None:
    assert select_commands({"trajectory_slots": {}}, None) is None


def test_several_models_are_refused_naming_them() -> None:
    facts = {"trajectory_slots": {"SR": _slots("SR"), "BR": _slots("BR")}}

    with pytest.raises(StartupRefused) as refused:
        select_commands(facts, None)

    message = str(refused.value)
    assert "BR" in message and "SR" in message and MODEL_KEY in message


def test_the_named_model_is_selected_among_several() -> None:
    facts = {"trajectory_slots": {"SR": _slots("SR"), "BR": _slots("BR")}}

    selected = select_commands(facts, "BR")

    assert selected == TrajectoryCommands(
        model="BR",
        arm=("BR:BPM:ARM",),
        injection_request="BR:INJ:REQ",
        injection_busy="BR:INJ:BUSY",
    )
    assert selected.commands == frozenset({"BR:BPM:ARM", "BR:INJ:REQ"})


@pytest.mark.parametrize("slots", [{}, {"SR": _slots("SR")}], ids=["none", "another"])
def test_a_named_model_without_slots_is_refused_naming_it(slots: dict[str, Any]) -> None:
    with pytest.raises(StartupRefused, match="'BR'"):
        select_commands({"trajectory_slots": slots}, "BR")


@pytest.mark.parametrize(
    ("slot", "value"),
    [
        ("arm", []),
        ("arm", None),
        ("injection_request", None),
        ("injection_request", ""),
        ("injection_busy", None),
    ],
)
@pytest.mark.parametrize("named", [None, "SR"], ids=["only", "named"])
def test_an_empty_slot_of_the_selected_model_is_refused_naming_both(
    slot: str, value: Any, named: str | None
) -> None:
    slots = _slots("SR")
    if value is None:
        del slots[slot]
    else:
        slots[slot] = value

    with pytest.raises(StartupRefused) as refused:
        select_commands({"trajectory_slots": {"SR": slots}}, named)

    message = str(refused.value)
    assert "'SR'" in message and repr(slot) in message


async def test_several_models_refuse_the_start(repo: Path, armed: list[int]) -> None:
    _write_slots(repo, {"SR": _slots("SR"), "BR": _slots("BR")})

    with pytest.raises(StartupRefused, match=MODEL_KEY):
        await _start()

    assert armed == []


# -- the refusal ----------------------------------------------------------------


def test_a_refusal_exits_non_zero_with_its_reason() -> None:
    refusal = StartupRefused("the reason")

    assert isinstance(refusal, SystemExit)
    assert isinstance(refusal.code, str) and "the reason" in refusal.code


# -- the degraded example ring --------------------------------------------------


@pytest.mark.slow
async def test_starts_against_the_degraded_example_ring(
    built_control_assistant: BuiltProject,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    armed: list[int],
) -> None:
    from osprey.simulation.engines.pyat import NO_CLOSED_ORBIT
    from osprey_connectors.control_system.va_in_process_connector import simulation_state_dir
    from osprey_connectors.simulation import composite as composite_module
    from osprey_connectors.simulation import model_status

    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    root = tmp_path / "example"
    render = root / "build"
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith("data/"):
            target = render / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    shutil.copy(built_control_assistant.repo / "profile.yml", root / "profile.yml")
    view = render / "data" / "simulator"
    state = simulation_state_dir(view)
    state.mkdir(parents=True, exist_ok=True)
    (state / "active_scenarios").write_text("first-turn-errors\n", encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.delenv("OSPREY_CONTROL_TARGET", raising=False)
    _deploy(root, monkeypatch)
    _shared_dirs(root, monkeypatch)

    started = await _start()
    try:
        assert started.target == "va"
        assert started.commands is not None
        assert started.commands.model == MODEL
        assert started.commands.arm and started.commands.injection_request
        assert model_status(started.connector, MODEL) == NO_CLOSED_ORBIT
        assert armed == [1]
    finally:
        await started.connector.disconnect()
        reset_config_cache()
