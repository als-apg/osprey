"""The example's ES1 stage moves as a positioner, on the in-process simulator.

The stage is built from the ``bluesky_devices.yml`` the demo build writes, so
the device is the one the worker would build, and driven by a real RunEngine
over the in-process simulator serving a copy of the demo's simulator view. The
simulator moves the stage's readback when its move command is written and
plays the command's status: ``MOVING`` for the ``for_s`` its seed states, then
``DONE``.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Coroutine, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from bluesky import RunEngine
from bluesky import plan_stubs as bps

from osprey.facility.views.bluesky import BLUESKY_DEVICES_FILE
from osprey.services.bluesky_bridge.devices._specs_from_file import specs_from_file
from osprey.services.bluesky_bridge.devices.connector import ConnectorPositioner, build_devices
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from tests._builds import BuiltProject
from tests.facility.served_tree import in_process_config

pytestmark = [pytest.mark.slow]

POSITION, POSITION_RB = "ES1:MOT:STAGE:01:POSITION:SP", "ES1:MOT:STAGE:01:POSITION:RB"
MOVE, MOVE_STATUS = "ES1:MOT:STAGE:01:MOVE:SP", "ES1:MOT:STAGE:01:STATUS:MOVE"
#: The simulator's tick in these tests.
TICK_S = 0.05


def _on_loop(run_engine: RunEngine, coroutine: Coroutine[Any, Any, Any], hang_s: float) -> Any:
    """Run ``coroutine`` on the RunEngine's loop, the connector's loop."""
    return asyncio.run_coroutine_threadsafe(coroutine, run_engine.loop).result(timeout=hang_s)


@dataclass
class _Stage:
    """The positioner, the RunEngine that drives it and the labels its status took."""

    run_engine: RunEngine
    positioner: ConnectorPositioner
    connector: VAInProcessConnector
    statuses: list[str]
    tolerance: float
    hang_s: float

    def on_loop(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        """Run ``coroutine`` on the RunEngine's loop."""
        return _on_loop(self.run_engine, coroutine, self.hang_s)


def _status_options(built: BuiltProject) -> list[str]:
    (status,) = [c for c in built.facility["channels"] if c["id"] == MOVE_STATUS]
    options: list[str] = status["options"]
    return options


@pytest.fixture
def stage(
    built_demo: BuiltProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_Stage]:
    arm_switch(monkeypatch, tmp_path / "armed-data")
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
    render = tmp_path / "build"
    view = render / "data" / "simulator"
    shutil.copytree(built_demo.build_dir / "data" / "simulator", view)
    (render / "config.yml").write_text(f"simulation:\n  tick_s: {TICK_S}\n")

    specs = specs_from_file(built_demo.build_dir / "data" / BLUESKY_DEVICES_FILE)
    (spec,) = [spec for spec in specs.settables if spec.name == POSITION]
    assert spec.started_by is not None and spec.settle_tolerance is not None
    # A hang ceiling: every call here ends within one act's bound.
    hang_s = 2 * spec.started_by.timeout_s
    run_engine = RunEngine(context_managers=[])
    connector = VAInProcessConnector()
    statuses: list[str] = []
    options = _status_options(built_demo)
    _on_loop(run_engine, connector.connect(in_process_config(view, response_delay_ms=0)), hang_s)
    _on_loop(
        run_engine,
        connector.subscribe(MOVE_STATUS, lambda reading: statuses.append(options[reading.value])),
        hang_s,
    )
    devices = _on_loop(
        run_engine, build_devices([spec], connector=connector, simulated=True), hang_s
    )
    positioner = devices[POSITION]
    assert isinstance(positioner, ConnectorPositioner)
    try:
        yield _Stage(run_engine, positioner, connector, statuses, spec.settle_tolerance, hang_s)
    finally:
        _on_loop(run_engine, connector.disconnect(), hang_s)


def _readback(stage: _Stage) -> float:
    return float(stage.on_loop(stage.connector.read_channel(POSITION_RB)).value)


def test_the_example_stage_moves_on_its_start_command(stage: _Stage) -> None:
    target = _readback(stage) + 1.0

    stage.run_engine(bps.mv(stage.positioner, target))

    assert abs(_readback(stage) - target) <= stage.tolerance
    moved = stage.statuses[1:]  # the first callback is the value before the move
    assert moved[: moved.index("DONE") + 1] == ["MOVING", "DONE"]


def test_the_example_stage_moved_to_where_it_is_passes(stage: _Stage) -> None:
    target = _readback(stage) + 1.0
    stage.run_engine(bps.mv(stage.positioner, target))

    stage.run_engine(bps.mv(stage.positioner, target))

    assert abs(_readback(stage) - target) <= stage.tolerance


def test_the_example_move_command_is_no_device(built_demo: BuiltProject) -> None:
    specs = specs_from_file(built_demo.build_dir / "data" / BLUESKY_DEVICES_FILE)
    names = {spec.name for spec in (*specs.settables, *specs.triggerables, *specs.readables)}

    assert MOVE not in names
    (spec,) = [spec for spec in specs.settables if spec.name == POSITION]
    assert spec.started_by is not None
    assert (spec.started_by.command, spec.started_by.status) == (MOVE, MOVE_STATUS)
