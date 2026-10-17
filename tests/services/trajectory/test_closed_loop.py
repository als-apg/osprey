"""The service threads the errored example ring to a full first turn, in process.

The built example project is deployed the way the shipped preset states it:
the simulator served in process, writes armed, and the shipped example
``limits.yaml`` checked ``optional``, so the commands and every unlisted
corrector pass and the one corrector the file bounds stays bounded. With the
``first-turn-errors`` scenario active the shipped seed loses the first turn
partway round the ring.

The loop is the one a panel drives: open a session, take a shot, apply the
step its post-shot hook installed, shoot again, until no BPM is lost. Every
part of it is the real one the service starts with: the deployment checks of
``start()``, the admission gate on a provisioned control-context tree, the
action worker, the run lock, ``command_write`` firing the arm and the shipped
VA sequence module firing the injection, the turn-by-turn extraction, the
pyAML correction tool built from the render, and the checked corrector writes.
Only the process-wide raw-put block is recorded rather than installed.

This test needs pyAML; ``tests/simulation/test_thread_first_turn_without_pyaml.py``
proves the different property that threading needs no middle layer.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

pytest.importorskip("at")

from tests._guarded_run import reset_run_state
from tests.facility.served_tree import in_process_config

from osprey.cli.build_injectors import _locate_pkg_services
from osprey.facility import FACILITY_FILE
from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY
from osprey.services.trajectory import startup as startup_module
from osprey.services.trajectory.acquisition import shot_action
from osprey.services.trajectory.actions import ActionLayer
from osprey.services.trajectory.admit import AdmitGate
from osprey.services.trajectory.app import shot_rows
from osprey.services.trajectory.apply import apply
from osprey.services.trajectory.constants import SEQUENCE_DIR_ENV_VAR
from osprey.services.trajectory.sessions import Sessions
from osprey.services.trajectory.startup import GUARDED_RUN_DIR_ENV_VAR, TrajectoryStartup
from osprey.services.trajectory.state import ShotResult
from osprey.services.trajectory.step import TrajectoryTool
from osprey.services.trajectory.store import TrajectoryStore
from osprey_connectors import posture_store
from osprey_connectors.identity import AUDIT_IDENTITY_ENV
from osprey_connectors.workspace import load_osprey_config, reset_config_cache

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = pytest.mark.slow

TARGET = "va"
OWNER = "alice"
KIND = "panel"
MODEL = "SR"
SCENARIO = "first-turn-errors"
#: The seed the scenario ships with.
SHIPPED_SEED = 11
#: The BPM (1-based, in beam order from injection) the first shot is lost by at the latest.
LATEST_LOSS_BPM = 36
#: The steps applied after which every BPM must see the first turn.
MAX_STEPS = 5
#: The rendered limits database the shipped example ``limits.yaml`` becomes.
LIMITS = "data/channel_limits.json"
#: The one corrector the shipped example ``limits.yaml`` bounds.
BOUNDED_CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
#: The shipped VA sequence module.
SHIPPED_SEQUENCE = _locate_pkg_services() / "trajectory" / "sequence"


def _config(view: Path) -> dict[str, Any]:
    """The preset's control-system posture: in process, writes armed, limits optional."""
    return {
        "control_system": {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": in_process_config(view, response_delay_ms=0)},
            "writes_enabled": True,
            "limits_checking": {
                "enabled": True,
                "database_path": LIMITS,
                "mode": "optional",
                "on_violation": "error",
            },
        },
    }


@pytest.fixture
def deployment(
    built_control_assistant: BuiltProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """The built example deployed in a fresh root with the scenario active; its root."""
    from osprey_connectors.control_system.va_in_process_connector import simulation_state_dir
    from osprey_connectors.simulation import composite as composite_module

    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_CONTEXT_DIR_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        posture_store.AGENT_DATA_ROOT_ENV_VAR,
        ENV_CONTROL_IDENTITY,
        AUDIT_IDENTITY_ENV,
        "OSPREY_EXECUTION_MODE",
        "OSPREY_CONTROL_TARGET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    # Installing the raw-put block would patch the client libraries of the whole test process.
    monkeypatch.setattr(startup_module, "_RAW_PUT_ARMED", False)
    monkeypatch.setattr(startup_module, "_install_raw_put_block", lambda: None)

    root = tmp_path / "example"
    render = root / "build"
    staged: set[str] = set()
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith("data/") or name == FACILITY_FILE:
            target = render / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            staged.add(name)
    assert {LIMITS, FACILITY_FILE} <= staged, "the build rendered no limits file or facility file"
    shutil.copy(built_control_assistant.repo / "profile.yml", root / "profile.yml")

    view = render / "data" / "simulator"
    state = simulation_state_dir(view)
    state.mkdir(parents=True, exist_ok=True)
    (state / "active_scenarios").write_text(f"{SCENARIO}\n", encoding="utf-8")

    config_path = render / "config.yml"
    config_path.write_text(yaml.safe_dump(_config(view)), encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    monkeypatch.setenv(SEQUENCE_DIR_ENV_VAR, str(SHIPPED_SEQUENCE))
    monkeypatch.chdir(root)
    reset_config_cache()
    reset_run_state(monkeypatch)

    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    monkeypatch.setenv(GUARDED_RUN_DIR_ENV_VAR, str(root / "var" / "guarded_run"))
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()
    reset_config_cache()


@pytest.fixture
async def started(deployment: Path) -> AsyncIterator[TrajectoryStartup]:  # noqa: ARG001
    """The service started by its own deployment checks."""
    service = await startup_module.start()
    try:
        yield service
    finally:
        await service.connector.disconnect()


def _lost_at(shot: ShotResult) -> int | None:
    """The 1-based BPM, in beam order, the shot is lost by; ``None`` for a full turn."""
    if not shot.lost:
        return None
    return shot.bpm_order.index(shot.lost[0]) + 1


async def test_the_service_threads_the_shipped_seed_to_a_full_first_turn(
    started: TrajectoryStartup, deployment: Path
) -> None:
    assert started.target == TARGET
    assert started.commands is not None and started.commands.model == MODEL
    assert started.sequence is not None, "the shipped VA sequence module was not loaded"
    assert isinstance(started.tool, TrajectoryTool), started.tool

    layer = ActionLayer(started, AdmitGate.from_config(load_osprey_config(), started.target))
    layer.start()
    sessions = Sessions(layer)
    store = TrajectoryStore(deployment / "state")
    rows = shot_rows(started)

    def build(hook: Any) -> Any:
        return shot_action(
            started, store, rows=rows, post_shot=hook, call_timeout_s=sessions.call_timeout_s
        )

    async def shoot(client_id: str) -> ShotResult:
        return await asyncio.wait_for(sessions.shot(OWNER, KIND, client_id, store, build), 120)

    lost_per_shot: list[int | None] = []
    moved_per_step: list[int] = []
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id

        # The limits the worker resolved are the shipped example file, enforced optionally.
        shot = await shoot(client_id)
        validator = started.connector._limits_validator
        assert validator is not None
        assert getattr(validator, "failsafe_reason", None) is None
        assert BOUNDED_CORRECTOR in validator.limits, "the shipped example limits were not loaded"
        assert shot.commands_fired, "the shot fired no command"
        assert shot.shots_included == 1, shot.shots_failed
        lost_per_shot.append(_lost_at(shot))

        first = lost_per_shot[0]
        assert first is not None, (
            f"{SCENARIO} with misalignment_seed {SHIPPED_SEED}: the uncorrected first turn "
            "is not lost"
        )
        assert first <= LATEST_LOSS_BPM, (
            f"{SCENARIO} with misalignment_seed {SHIPPED_SEED}: the uncorrected first turn is "
            f"lost at BPM {first}, later than BPM {LATEST_LOSS_BPM}"
        )

        while lost_per_shot[-1] is not None and len(moved_per_step) < MAX_STEPS:
            step = sessions.step.step
            assert step is not None, "the shot installed no step"
            assert step.shot_id == shot.shot_id
            assert not step.all_zero, f"the step for a lost turn moves nothing: {step.reason}"
            done = await asyncio.wait_for(
                apply(sessions, OWNER, KIND, client_id, {"step_digest": step.digest}), 120
            )
            assert done.name == "apply" and not done.stopped
            assert done.moved, "the apply moved no corrector"
            moved_per_step.append(len(done.moved))
            shot = await shoot(client_id)
            assert shot.shots_included == 1, shot.shots_failed
            lost_per_shot.append(_lost_at(shot))
    finally:
        await layer.close()

    assert lost_per_shot[-1] is None, (
        f"{SCENARIO} with misalignment_seed {SHIPPED_SEED}: after {len(moved_per_step)} "
        f"applied steps the first turn is still lost (lost at BPM per shot: {lost_per_shot})"
    )
    assert shot.lost == [] and shot.per_shot_lost_at == [None]
    assert shot.last_bpm == shot.bpm_order[-1]
    assert 1 <= len(moved_per_step) <= MAX_STEPS, lost_per_shot
