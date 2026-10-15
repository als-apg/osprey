"""The write seam: an action the layer runs fires the model's arm through the real connector.

A test-only action, handed to :meth:`ActionLayer.submit` and registered on no
route, calls :func:`~osprey.runtime.guarded_run.command_write` on the selected
model's ``arm`` slot. It runs in the layer's real worker task, behind the real
admit gate on a real control-context tree, the real reachability probe, the
limits validator the worker rebuilds from the rendered config, the owner
binding and the real run lock. The connector is an in-process VA connected to a
built and rendered project whose limits records state ``confirm: false`` for
the commands, and its control target is stamped with the service's target, so
the connector's own per-write chip check judges the target the service drives.

Nothing in the write path is patched. In the second case the owner's record is
narrowed while ``admit`` is stubbed to pass, so the refusal can only come from
the connector's per-write chip check.
"""

from __future__ import annotations

import asyncio
import os
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

from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY, command_write
from osprey.services.trajectory.actions import ActionLayer
from osprey.services.trajectory.admit import Admission, AdmitGate
from osprey.services.trajectory.startup import TrajectoryCommands, TrajectoryStartup
from osprey_connectors import control_context, posture_store
from osprey_connectors.control_system.va_in_process_connector import VAInProcessConnector
from osprey_connectors.errors import ChannelWriteBlockedError
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


def _layer(connector: VAInProcessConnector, gate: Any = None) -> ActionLayer:
    startup = TrajectoryStartup(
        target=TARGET,
        connector=connector,
        commands=COMMANDS,
        guarded_run_dir=Path("/unused"),
    )
    if gate is None:
        gate = AdmitGate.from_config(load_osprey_config(), TARGET)
    return ActionLayer(startup, gate)


def _arm_action(layer: ActionLayer) -> Any:
    """The test-only action: fire the selected model's arm slot as a command write."""

    async def arm(admission: Admission, stop: asyncio.Event) -> dict[str, Any]:  # noqa: ARG001
        fired = await command_write(
            layer.startup.connector, list(COMMANDS.arm), 1, commands=COMMANDS.commands
        )
        return {"fired": fired, "task": asyncio.current_task(), "owner": admission.owner}

    return arm


def _narrow(owner: str) -> None:
    tree = Path(os.environ[posture_store.CONTROL_CONTEXT_TREE_ENV_VAR])
    control_context.write_record(
        control_context.ControlContext(target=TARGET, generation=1, posture={TARGET: "sandbox"}),
        path=tree / owner / control_context.RECORD_FILENAME,
    )
    posture_store.invalidate_cache()


async def _armed(connector: VAInProcessConnector) -> int:
    return int((await connector.read_channel(ARMED)).value)


class _PassingGate:
    """An admit gate that admits every owner, so only the connector can refuse."""

    def __init__(self) -> None:
        self.asked: list[tuple[Any, str]] = []

    def admit(self, owner: Any, kind: str) -> Admission:
        self.asked.append((owner, kind))
        return Admission(owner=owner, kind=kind, target=TARGET, generation=1, posture={})


async def test_an_action_fires_the_models_arm_through_the_worker_and_the_connector(
    connector: VAInProcessConnector,
) -> None:
    layer = _layer(connector)
    try:
        assert await _armed(connector) == 0

        result = await asyncio.wait_for(layer.submit(OWNER, KIND, _arm_action(layer)), 30)

        assert result["fired"] == [ARM]
        assert result["owner"] == OWNER
        assert result["task"] is layer.worker_task
        assert await _armed(connector) == 1
        # The validator the worker rebuilt for the service target is the enforcing one.
        validator = connector._limits_validator
        assert validator is not None
        assert getattr(validator, "failsafe_reason", None) is None
    finally:
        await layer.close()


async def test_a_narrowed_owner_admitted_anyway_is_refused_by_the_connectors_chip_check(
    connector: VAInProcessConnector,
) -> None:
    _narrow(OWNER)
    gate = _PassingGate()
    layer = _layer(connector, gate)
    try:
        with pytest.raises(ChannelWriteBlockedError) as caught:
            await asyncio.wait_for(layer.submit(OWNER, KIND, _arm_action(layer)), 30)

        assert gate.asked == [(OWNER, KIND)]
        assert caught.value.commands_fired == []
        assert caught.value.reason == "WRITES_DISABLED"
        # The chip check judged the target the service drives, from its stamp.
        assert f"writes are off for the '{TARGET}' control target" in str(caught.value)
        assert "control-target chip" in str(caught.value)
        assert await _armed(connector) == 0
    finally:
        await layer.close()


async def test_the_same_narrowing_is_refused_by_the_real_admit_before_the_connector(
    connector: VAInProcessConnector,
) -> None:
    from osprey.services.trajectory.admit import NARROWING, AdmitRefused

    _narrow(OWNER)
    layer = _layer(connector)
    try:
        with pytest.raises(AdmitRefused) as caught:
            await asyncio.wait_for(layer.submit(OWNER, KIND, _arm_action(layer)), 30)

        assert caught.value.code == NARROWING
        assert await _armed(connector) == 0
    finally:
        await layer.close()
