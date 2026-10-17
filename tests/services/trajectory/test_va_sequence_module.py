"""The VA sequence module the build ships: it loads, and its ``inject`` fires one shot.

The module is the file in the trajectory service template's ``sequence/``
directory, the one the build copies into a project and names as the sequence
directory of a VA target. Its ``inject`` runs through a real
:class:`~osprey.services.trajectory.sequence.SequenceHandle` on the in-process
VA connector, inside an action the real :class:`ActionLayer` runs under the run
lock, on a built and rendered project whose limits records state
``confirm: false`` for the commands. Nothing in the write path is patched.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("at")

from tests._guarded_run import reset_run_state
from tests.facility._pyaml_trees import measured_tree, with_correctors
from tests.facility._synthetic_trees import model, with_shot_wiring, with_trajectory, write_tree
from tests.facility.served_tree import in_process_config

from osprey.cli.build_injectors import _locate_pkg_services
from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY, command_write
from osprey.services.trajectory.actions import ActionLayer
from osprey.services.trajectory.admit import Admission, AdmitGate
from osprey.services.trajectory.constants import MAX_PAUSE_CEILING_S, VA_SEQUENCE_DIR
from osprey.services.trajectory.sequence import (
    SEQUENCE_MODULE_FILE,
    SequenceHandle,
    SequenceModule,
    load_sequence,
)
from osprey.services.trajectory.startup import TrajectoryCommands, TrajectoryStartup
from osprey_connectors import posture_store
from osprey_connectors.control_system.va_in_process_connector import VAInProcessConnector
from osprey_connectors.identity import AUDIT_IDENTITY_ENV
from osprey_connectors.workspace import load_osprey_config, reset_config_cache

TARGET = "va"
OWNER = "alice"
KIND = "panel"
MODEL = "SR"
ARM = "SR:BPM:ARM"
ARMED = "SR:BPM:ARMED"
REQUEST = "SR:INJ:REQ"
BUSY = "SR:INJ:BUSY"
COMMANDS = TrajectoryCommands(
    model=MODEL, arm=(ARM,), injection_request=REQUEST, injection_busy=BUSY
)
SHIPPED = _locate_pkg_services() / "trajectory" / "sequence"


# -- the shipped module loads ----------------------------------------------------


def _shipped() -> SequenceModule:
    module = load_sequence(SHIPPED, COMMANDS)
    assert module is not None
    return module


def test_the_shipped_module_is_the_file_the_loader_reads() -> None:
    assert (SHIPPED / SEQUENCE_MODULE_FILE).is_file()


def test_the_loader_accepts_the_shipped_module_with_no_slots_and_a_short_pause() -> None:
    module = _shipped()

    assert set(module.entry_points) == {"inject", "bpm_mode"}
    assert module.entry_point("bunch_clean") is None
    assert dict(module.metadata.slots) == {}
    assert dict(module.metadata.writes) == {}
    assert 0 < module.metadata.max_pause_s <= 1.0 <= MAX_PAUSE_CEILING_S


def test_the_build_copies_the_module_a_va_target_loads(tmp_path: Path) -> None:
    from osprey.cli.build_injectors import _inject_trajectory
    from osprey.cli.build_profile_schema import TrajectoryConfig

    project = tmp_path / "build"
    project.mkdir()
    (project / "config.yml").write_text(
        yaml.safe_dump({"control_system": {"type": "virtual_accelerator"}}), encoding="utf-8"
    )
    _inject_trajectory(TrajectoryConfig(target="va"), project)

    copied = tmp_path / VA_SEQUENCE_DIR
    assert (copied / SEQUENCE_MODULE_FILE).read_bytes() == (
        SHIPPED / SEQUENCE_MODULE_FILE
    ).read_bytes()
    module = load_sequence(copied, COMMANDS)
    assert module is not None
    assert set(module.entry_points) == {"inject", "bpm_mode"}


# -- inject fires one shot on the in-process VA ----------------------------------


def _tree() -> dict[str, Any]:
    """The periodic model measured by trajectory, its shot records wired, its commands limited."""
    tree = with_trajectory(with_correctors(measured_tree()), MODEL, "SR/BPM1", marker="M1")
    with_shot_wiring(tree, MODEL, "BPM1")
    model(tree, MODEL)["settings"]["pyat"]["injection"]["x"] = 1.0e-3
    tree["limits.yaml"] = {
        "records": [
            {"address": ARM, "min_value": 0, "max_value": 1, "confirm": False},
            {"address": REQUEST, "min_value": 0, "max_value": 1000, "confirm": False},
        ]
    }
    return tree


def _config(view: Path) -> dict[str, Any]:
    """The deployment's config: the simulator in process, writes armed, every unlisted channel refused."""
    return {
        "control_system": {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": in_process_config(view, response_delay_ms=0)},
            "writes_enabled": True,
            "limits_checking": {
                "enabled": True,
                "database_path": "data/channel_limits.json",
                "mode": "exclusive",
                "on_violation": "error",
            },
        }
    }


@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A built, rendered deployment this process reads, with a provisioned control-context tree."""
    from osprey.facility.build import build_facility
    from osprey.facility.render import render_facility_outputs

    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_CONTEXT_DIR_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        posture_store.AGENT_DATA_ROOT_ENV_VAR,
        ENV_CONTROL_IDENTITY,
        AUDIT_IDENTITY_ENV,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "deploy"
    facility = root / "data" / "facility"
    write_tree(facility, _tree())
    render = root / "build"
    render.mkdir(parents=True)
    view = render / "data" / "simulator"
    config = _config(view)
    document = build_facility(facility, project_name="served")
    render_facility_outputs(render, document, config, facility, omitted_reported=set())
    config_path = render / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    monkeypatch.chdir(root)
    reset_config_cache()
    reset_run_state(monkeypatch)

    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    posture_store.invalidate_cache()
    yield view
    posture_store.invalidate_cache()
    reset_config_cache()


@pytest.fixture
async def connector(deployment: Path) -> AsyncIterator[VAInProcessConnector]:
    """The in-process VA connector, stamped with the service's target as the factory stamps it."""
    va = VAInProcessConnector()
    va._control_target = TARGET
    await va.connect(in_process_config(deployment, response_delay_ms=0))
    yield va
    await va.disconnect()


@pytest.fixture
async def layer(connector: VAInProcessConnector) -> AsyncIterator[ActionLayer]:
    startup = TrajectoryStartup(
        target=TARGET,
        connector=connector,
        commands=COMMANDS,
        guarded_run_dir=Path("/unused"),
        sequence=_shipped(),
    )
    actions = ActionLayer(startup, AdmitGate.from_config(load_osprey_config(), TARGET))
    yield actions
    await actions.close()


def _shots(connector: VAInProcessConnector) -> int:
    return int(connector._composite._children[MODEL].model.shot_count)


async def _value(connector: VAInProcessConnector, address: str) -> float:
    return float((await connector.read_channel(address)).value)


def _run(layer: ActionLayer, entry_point: str, *, arm: bool) -> Any:
    """An action: arm (when asked), then run one entry point of the shipped module."""

    async def action(admission: Admission, stop: asyncio.Event) -> SequenceHandle:  # noqa: ARG001
        startup = layer.startup
        if arm:
            await command_write(
                startup.connector, list(COMMANDS.arm), 1, commands=COMMANDS.commands
            )
        handle = SequenceHandle(
            startup.connector,
            commands=startup.commands,
            metadata=startup.sequence.metadata,
            entry_point=entry_point,
            stop=stop,
        )
        await startup.sequence.entry_point(entry_point)(handle)
        return handle

    return action


async def test_inject_through_the_handle_fires_exactly_one_shot(
    connector: VAInProcessConnector, layer: ActionLayer
) -> None:
    assert _shots(connector) == 0
    before = await _value(connector, REQUEST)

    handle = await asyncio.wait_for(layer.submit(OWNER, KIND, _run(layer, "inject", arm=True)), 30)

    assert handle.commands_fired == [REQUEST]
    assert handle.moved == []
    assert _shots(connector) == 1
    assert await _value(connector, REQUEST) == before + 1
    # The shot consumed the arm.
    assert await _value(connector, ARMED) == 0


async def test_each_inject_counts_the_request_up_and_fires_one_more_shot(
    connector: VAInProcessConnector, layer: ActionLayer
) -> None:
    before = await _value(connector, REQUEST)

    for shot in (1, 2, 3):
        await asyncio.wait_for(layer.submit(OWNER, KIND, _run(layer, "inject", arm=True)), 30)

        assert _shots(connector) == shot
        assert await _value(connector, REQUEST) == before + shot


async def test_bpm_mode_writes_nothing_and_fires_no_shot(
    connector: VAInProcessConnector, layer: ActionLayer
) -> None:
    before = await _value(connector, REQUEST)

    handle = await asyncio.wait_for(
        layer.submit(OWNER, KIND, _run(layer, "bpm_mode", arm=False)), 30
    )

    assert handle.moved == []
    assert handle.commands_fired == []
    assert _shots(connector) == 0
    assert await _value(connector, REQUEST) == before
