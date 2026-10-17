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

from osprey.services.trajectory import constants, startup
from osprey.services.trajectory.constants import (
    ACTION_TIMEOUT_KEY,
    ARMED_TIMEOUT_KEY,
    CONTINUOUS_MAX_SHOTS_KEY,
    LOSS_SUM_CUTOFF_KEY,
    MAX_PAUSE_CEILING_S,
    RAW_WINDOW_TURNS_KEY,
    SEQUENCE_DIR_ENV_VAR,
    SEQUENCE_DIR_KEY,
    SEQUENCE_MOUNT,
    SHOT_SUM_THRESHOLD_KEY,
    VA_SEQUENCE_DIR,
)
from osprey.services.trajectory.startup import (
    GUARDED_RUN_DIR_ENV_VAR,
    MODEL_KEY,
    TARGET_KEY,
    StartupRefused,
    TrajectoryCommands,
    TrajectorySettings,
    resolve_sequence_dir,
    select_commands,
    service_settings,
)
from osprey.services.trajectory.step import ToolUnavailable
from osprey_connectors.posture_store import CONTROL_CONTEXT_TREE_ENV_VAR
from osprey_connectors.workspace import reset_config_cache

if TYPE_CHECKING:
    from tests._builds import BuiltProject

MODEL = "SR"
ARM = "SR:BPM:ARM"
ARMED = "SR:BPM:ARMED"
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
    monkeypatch.delenv(SEQUENCE_DIR_ENV_VAR, raising=False)
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
            model=MODEL, arm=(ARM,), injection_request=REQUEST, injection_busy=BUSY, armed=(ARMED,)
        )
        assert started.commands.commands == frozenset({ARM, REQUEST})
        assert started.writing_available
        assert started.guarded_run_dir == repo / "var" / "guarded_run" / "va"
        assert started.settings == TrajectorySettings()
        assert started.sequence_dir is None
        assert started.sequence is None
        # This tree's response matrix is all zero, so the slot names why there is no tool.
        assert isinstance(started.tool, ToolUnavailable)
        assert armed == [1]
    finally:
        await started.connector.disconnect()


async def test_the_tunable_keys_and_the_sequence_dir_reach_the_started_service(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _deploy(
        repo,
        monkeypatch,
        service={"armed_timeout_s": 2.5, "continuous_max_shots": 7, "sequence_dir": "seq"},
    )

    started = await _started()

    assert started.settings.armed_timeout_s == 2.5
    assert started.settings.continuous_max_shots == 7
    assert started.sequence_dir == repo / "seq"


async def test_the_mounted_sequence_dir_wins_over_the_key(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _deploy(repo, monkeypatch, service={"sequence_dir": "host/path"})
    monkeypatch.setenv(SEQUENCE_DIR_ENV_VAR, SEQUENCE_MOUNT)

    started = await _started()

    assert started.sequence_dir == Path(SEQUENCE_MOUNT)


async def test_the_service_connector_reads_fresh(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the service's own connector config gains ``fresh_reads``; the section is untouched."""
    from osprey_connectors.factory import ConnectorFactory

    seen: list[tuple[dict[str, Any], str]] = []

    async def create(config: dict[str, Any], *, control_target: str) -> object:
        seen.append((config, control_target))
        return object()

    monkeypatch.setattr(ConnectorFactory, "create_control_system_connector", create)
    section = _config(repo)["control_system"]
    original = json.loads(json.dumps(section))

    await startup._connect(section, "va")

    ((config, target),) = seen
    assert target == "va"
    assert config["type"] == "virtual_accelerator"
    assert config["connector"]["virtual_accelerator"]["fresh_reads"] is True
    assert (
        config["connector"]["virtual_accelerator"]["simulator_view"]
        == original["connector"]["virtual_accelerator"]["simulator_view"]
    )
    assert section == original


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


async def test_an_optional_posture_with_no_limits_file_starts(repo: Path) -> None:
    """Optional + no path is legal at service start (q-0001/1, q-0002)."""
    config = _config(repo)
    config["control_system"]["limits_checking"]["mode"] = "optional"
    del config["control_system"]["limits_checking"]["database_path"]
    path = repo / "build" / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    reset_config_cache()

    started = await _started()

    assert started.commands is not None


async def test_an_optional_posture_naming_a_missing_limits_file_refuses(repo: Path) -> None:
    """A configured path must name a file whatever the mode."""
    config = _config(repo)
    config["control_system"]["limits_checking"]["mode"] = "optional"
    path = repo / "build" / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    reset_config_cache()
    missing = repo / "build" / LIMITS
    missing.unlink()

    with pytest.raises(StartupRefused) as refused:
        await _start()

    assert str(missing) in str(refused.value)


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
        armed=("BR:BPM:ARMED",),
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
        ("armed", []),
        ("armed", None),
        ("armed", [""]),
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


def test_an_armed_flag_count_unlike_the_arm_count_is_refused() -> None:
    slots = _slots("SR")
    slots["armed"] = ["SR:BPM:ARMED", "SR:BPM2:ARMED"]

    with pytest.raises(StartupRefused) as refused:
        select_commands({"trajectory_slots": {"SR": slots}}, None)

    message = str(refused.value)
    assert "'SR'" in message and "'arm'" in message and "'armed'" in message


def test_the_armed_flags_are_watched_never_commanded() -> None:
    selected = select_commands({"trajectory_slots": {"SR": _slots("SR")}}, None)

    assert selected is not None
    assert selected.armed == ("SR:BPM:ARMED",)
    assert "SR:BPM:ARMED" not in selected.commands


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


# -- the tunable keys -----------------------------------------------------------


def _settings_of(block: dict[str, Any]) -> TrajectorySettings:
    return service_settings(
        {"control_system": {"type": "virtual_accelerator"}, "services": {"trajectory": block}}
    )[2]


def test_an_unstated_tunable_key_takes_its_default() -> None:
    settings = _settings_of({})

    assert settings == TrajectorySettings(
        loss_sum_cutoff=constants.DEFAULT_LOSS_SUM_CUTOFF,
        shot_sum_threshold=constants.DEFAULT_SHOT_SUM_THRESHOLD,
        armed_timeout_s=constants.DEFAULT_ARMED_TIMEOUT_S,
        raw_window_turns=constants.DEFAULT_RAW_WINDOW_TURNS,
        continuous_max_shots=constants.DEFAULT_CONTINUOUS_MAX_SHOTS,
        sequence_dir=None,
        action_timeout_s=constants.DEFAULT_ACTION_TIMEOUT_S,
    )


def test_the_va_example_sum_thresholds_are_the_defaults() -> None:
    """The VA's sum reads 1.0 with beam and 0 without; halfway is its example."""
    assert constants.DEFAULT_LOSS_SUM_CUTOFF == 0.5
    assert constants.DEFAULT_SHOT_SUM_THRESHOLD == 0.5


def test_stated_tunable_keys_are_read() -> None:
    settings = _settings_of(
        {
            "loss_sum_cutoff": 0,
            "shot_sum_threshold": "1.5",
            "armed_timeout_s": 3,
            "raw_window_turns": "40",
            "continuous_max_shots": 2,
            "sequence_dir": "./facility/sequence",
            "action_timeout_s": 12.5,
        }
    )

    assert settings == TrajectorySettings(
        loss_sum_cutoff=0.0,
        shot_sum_threshold=1.5,
        armed_timeout_s=3.0,
        raw_window_turns=40,
        continuous_max_shots=2,
        sequence_dir="./facility/sequence",
        action_timeout_s=12.5,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (LOSS_SUM_CUTOFF_KEY, -0.1),
        (LOSS_SUM_CUTOFF_KEY, True),
        (LOSS_SUM_CUTOFF_KEY, "high"),
        (SHOT_SUM_THRESHOLD_KEY, float("nan")),
        (ARMED_TIMEOUT_KEY, 0),
        (ARMED_TIMEOUT_KEY, float("inf")),
        (ACTION_TIMEOUT_KEY, -1),
        (ACTION_TIMEOUT_KEY, False),
        (RAW_WINDOW_TURNS_KEY, 0),
        (RAW_WINDOW_TURNS_KEY, 2.5),
        (RAW_WINDOW_TURNS_KEY, True),
        (CONTINUOUS_MAX_SHOTS_KEY, -3),
        (CONTINUOUS_MAX_SHOTS_KEY, "many"),
        (SEQUENCE_DIR_KEY, ""),
        (SEQUENCE_DIR_KEY, 3),
    ],
)
def test_an_unusable_tunable_key_is_refused_naming_it(key: str, value: Any) -> None:
    with pytest.raises(StartupRefused, match=key.replace(".", r"\.")):
        _settings_of({key.rsplit(".", 1)[1]: value})


async def test_an_unusable_tunable_key_refuses_the_start(
    repo: Path, monkeypatch: pytest.MonkeyPatch, armed: list[int]
) -> None:
    _deploy(repo, monkeypatch, service={"armed_timeout_s": 0})

    with pytest.raises(StartupRefused, match=ARMED_TIMEOUT_KEY):
        await _start()

    assert armed == []


# -- the sequence module directory ----------------------------------------------


def test_no_sequence_dir_named_resolves_to_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SEQUENCE_DIR_ENV_VAR, raising=False)

    assert resolve_sequence_dir(TrajectorySettings(), tmp_path) is None


def test_a_relative_sequence_dir_is_the_project_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SEQUENCE_DIR_ENV_VAR, raising=False)
    settings = TrajectorySettings(sequence_dir=VA_SEQUENCE_DIR)

    assert (
        resolve_sequence_dir(settings, tmp_path) == tmp_path / "build/services/trajectory/sequence"
    )


def test_an_absolute_sequence_dir_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SEQUENCE_DIR_ENV_VAR, raising=False)
    elsewhere = tmp_path / "elsewhere"

    settings = TrajectorySettings(sequence_dir=str(elsewhere))

    assert resolve_sequence_dir(settings, tmp_path / "root") == elsewhere


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_mount_variable_falls_back_to_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    monkeypatch.setenv(SEQUENCE_DIR_ENV_VAR, blank)

    assert (
        resolve_sequence_dir(TrajectorySettings(sequence_dir="seq"), tmp_path) == tmp_path / "seq"
    )


def test_the_va_default_names_the_package_dir_the_build_copies() -> None:
    """The injector's VA default is the template's ``sequence/``, as copied into the project."""
    from osprey.cli.build_injectors import _locate_pkg_services

    shipped = _locate_pkg_services() / "trajectory" / "sequence"
    assert shipped.is_dir()
    assert Path(VA_SEQUENCE_DIR).parts == ("build", "services", "trajectory", "sequence")


def test_the_compose_template_binds_the_fixed_target_it_exports() -> None:
    """The mount target and the variable the template spells are the constants the service reads."""
    from osprey.cli.build_injectors import _locate_pkg_services

    text = (_locate_pkg_services() / "trajectory" / "docker-compose.yml.j2").read_text()

    assert f"{SEQUENCE_DIR_ENV_VAR}: {SEQUENCE_MOUNT}\n" in text
    assert f"- {{{{ svc.sequence_dir }}}}:{SEQUENCE_MOUNT}:ro\n" in text


# -- the sequence module delivery, at build -------------------------------------


def _inject(tmp_path: Path, *, target: str | None, authored: dict[str, Any] | None = None) -> Any:
    from osprey.cli.build_injectors import _inject_trajectory
    from osprey.cli.build_profile_schema import TrajectoryConfig

    project = tmp_path / "build"
    project.mkdir()
    config: dict[str, Any] = {"control_system": {"type": "virtual_accelerator"}}
    if authored is not None:
        config["services"] = {"trajectory": authored}
    (project / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    _inject_trajectory(TrajectoryConfig(target=target), project)
    written = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
    return project, written["services"]["trajectory"]


@pytest.mark.parametrize("target", [None, "va"], ids=["baseline", "named"])
def test_the_build_defaults_a_va_target_to_the_shipped_module(
    tmp_path: Path, target: str | None
) -> None:
    project, block = _inject(tmp_path, target=target)

    assert block["sequence_dir"] == VA_SEQUENCE_DIR
    assert (project.parent / VA_SEQUENCE_DIR).is_dir()


def test_the_build_leaves_a_live_target_without_a_module(tmp_path: Path) -> None:
    _, block = _inject(tmp_path, target="live")

    assert "sequence_dir" not in block


def test_the_build_carries_the_authored_tunable_keys(tmp_path: Path) -> None:
    authored = {
        "loss_sum_cutoff": 120.0,
        "shot_sum_threshold": 80.0,
        "armed_timeout_s": 2.0,
        "raw_window_turns": 50,
        "continuous_max_shots": 10,
        "sequence_dir": "./facility/sequence",
        "action_timeout_s": 30.0,
    }

    _, block = _inject(tmp_path, target=None, authored=authored)

    assert {key: block[key] for key in authored} == authored


# -- the budgets ----------------------------------------------------------------


def test_the_stop_budget_is_one_pause_one_call_and_one_poll() -> None:
    budgets = constants.budgets(max_pause_s=2.0, call_timeout_s=5.0, action_timeout_s=30.0)

    assert budgets.stop_budget_s == pytest.approx(2.0 + 5.0 + constants.CHECKPOINT_POLL_S)
    assert budgets.action_budget_s == pytest.approx(30.0 + budgets.stop_budget_s)


def test_the_stop_grace_period_covers_shutdown_drain_and_disconnect() -> None:
    call = 5.0
    stop = MAX_PAUSE_CEILING_S + call + constants.CHECKPOINT_POLL_S

    grace = constants.stop_grace_period_s(call)

    assert isinstance(grace, int)
    assert grace >= 2 * stop + call
    assert grace == constants.stop_grace_period_s(call)
    assert constants.stop_grace_period_s(call + 10) > grace
