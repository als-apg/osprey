"""The example's ES1 detector, triggered and recorded by a real RunEngine.

The detector is built from the ``bluesky_devices.yml`` the demo build writes,
so it is the device the worker would build, and run by a real ``RunEngine``
through the catalog plan wrapper over the in-process simulator serving a copy
of the demo's simulator view. The simulator plays the acquire command's status
when the command is written: ``TRUE`` for the ``for_s`` its seed states, then
``FALSE``. The write goes through the connector's own gate, which answers for
the owner the queue item carries, read from a real control-context tree.

What a test waits on is what the product attests: the RunEngine's return, the
status the simulator serves, and the run's own documents.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
from collections.abc import Coroutine, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from bluesky import RunEngine
from bluesky import plans as bp
from bluesky.utils import FailedStatus, RunEngineInterrupted
from pydantic import BaseModel, ConfigDict

from osprey.facility.views.bluesky import BLUESKY_DEVICES_FILE
from osprey.services.bluesky_bridge import qserver_startup
from osprey.services.bluesky_bridge.devices._specs_from_file import specs_from_file
from osprey.services.bluesky_bridge.devices.connector import ConnectorDetector, build_devices
from osprey.services.bluesky_bridge.plan_fields import TriggeredChannel
from osprey_connectors.control_system._base import ControlSystemConnector
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG
from osprey_connectors_tests._write_fakes import ARMED_TARGET, config_reader
from tests._builds import BuiltProject
from tests.facility.served_tree import in_process_config
from tests.services.bluesky_bridge.test_bind_owner_runengine import (
    _OWNER,
    _UNNARROWED_OWNER,
    record_for,
    turn_off_for,
)

pytestmark = [pytest.mark.slow]

DETECTOR = "ES1-MCA01"
ACQUIRE, STATUS = "ES1:DET:MCA:01:ACQUIRE:SP", "ES1:DET:MCA:01:STATUS:ACQUIRE"
COUNTS, SPECTRUM = "ES1:DET:MCA:01:COUNTS:RB", "ES1:DET:MCA:01:SPECTRUM:WAVEFORM"
#: The simulator's tick in these tests.
TICK_S = 0.05
_PLAN_NAME = "count_detector"
#: The lane's ``control_system`` section: writes allowed, both logins allowed, so
#: only each owner's own switch decides a write.
_SECTION = {
    "type": "va",
    "writes": "allowed",
    "login_tiers": {_OWNER: "allowed", _UNNARROWED_OWNER: "allowed"},
}


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: TriggeredChannel = DETECTOR
    num: int = 1


def _count(devices: Any, params: Any) -> Iterator[Any]:
    """Trigger and record the declared detector ``num`` times."""
    yield from bp.count([devices[params.detector]], num=params.num)


@dataclass
class _Station:
    """The detector, the RunEngine that runs it and the labels its status took."""

    run_engine: RunEngine
    detector: ConnectorDetector
    statuses: list[str]
    busy: threading.Event
    idle_again: threading.Event
    hang_s: float

    def plan(self) -> Any:
        """The count, wrapped as the worker wraps a catalog plan."""
        spec = type(
            "_Spec",
            (),
            {
                "name": _PLAN_NAME,
                "schema": _Params,
                "plan": staticmethod(_count),
                "description": "Count the detector once.",
                "metadata": None,
            },
        )()
        return qserver_startup._make_plan_function(spec, devices={DETECTOR: self.detector})


def _on_loop(run_engine: RunEngine, coroutine: Coroutine[Any, Any, Any], hang_s: float) -> Any:
    """Run ``coroutine`` on the RunEngine's loop, the connector's loop."""
    return asyncio.run_coroutine_threadsafe(coroutine, run_engine.loop).result(timeout=hang_s)


@pytest.fixture
def station(
    built_demo: BuiltProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_Station]:
    from osprey_connectors import posture_store

    tree = tmp_path / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    monkeypatch.setattr(ControlSystemConnector, "_control_target", ARMED_TARGET)
    monkeypatch.setattr("osprey_connectors.config.get_config_value", config_reader(_SECTION))
    for owner in (_OWNER, _UNNARROWED_OWNER):
        record_for(tree, owner, on=True, target=ARMED_TARGET)

    render = tmp_path / "build"
    view = render / "data" / "simulator"
    shutil.copytree(built_demo.build_dir / "data" / "simulator", view)
    (render / "config.yml").write_text(f"simulation:\n  tick_s: {TICK_S}\n")

    (spec,) = specs_from_file(built_demo.build_dir / "data" / BLUESKY_DEVICES_FILE).detectors
    # A hang ceiling: every call here ends within one act's bound.
    hang_s = 2 * spec.timeout_s
    (status,) = [c for c in built_demo.facility["channels"] if c["id"] == STATUS]
    options: list[str] = status["options"]
    statuses: list[str] = []
    busy, idle_again = threading.Event(), threading.Event()

    def seen(reading: Any) -> None:
        label = options[reading.value]
        statuses.append(label)
        if label != status["idle"]:
            busy.set()
        elif busy.is_set():
            idle_again.set()

    run_engine = RunEngine(context_managers=[])
    connector = VAInProcessConnector()
    _on_loop(run_engine, connector.connect(in_process_config(view, response_delay_ms=0)), hang_s)
    _on_loop(run_engine, connector.subscribe(STATUS, seen), hang_s)
    devices = _on_loop(run_engine, build_devices(connector=connector, detectors=[spec]), hang_s)
    detector = devices[DETECTOR]
    assert isinstance(detector, ConnectorDetector)
    try:
        yield _Station(run_engine, detector, statuses, busy, idle_again, hang_s)
    finally:
        _on_loop(run_engine, connector.disconnect(), hang_s)
        posture_store._owner_var.set(posture_store.NO_OWNER)


def test_a_count_fires_the_acquire_waits_for_the_status_and_records_every_result(
    station: _Station,
) -> None:
    documents: list[tuple[str, dict[str, Any]]] = []

    station.run_engine(
        station.plan()(**{RESERVED_OWNER_KWARG: _OWNER}),
        lambda name, document: documents.append((name, document)),
    )

    (descriptor,) = [document for name, document in documents if name == "descriptor"]
    (event,) = [document for name, document in documents if name == "event"]
    assert descriptor["object_keys"] == {DETECTOR: [COUNTS, SPECTRUM]}
    assert descriptor["data_keys"][SPECTRUM]["shape"] == [1024]
    assert len(event["data"][SPECTRUM]) == 1024
    assert station.statuses[1:] == ["TRUE", "FALSE"]


def test_a_trigger_write_is_decided_for_the_bound_owner(station: _Station, tmp_path: Path) -> None:
    from osprey_connectors import posture_store

    turn_off_for(tmp_path / posture_store.STATE_DIR_NAME, _UNNARROWED_OWNER, target=ARMED_TARGET)

    with pytest.raises(FailedStatus, match="switch_off"):
        station.run_engine(station.plan()(**{RESERVED_OWNER_KWARG: _UNNARROWED_OWNER}))

    assert station.statuses[1:] == []


def test_an_abort_during_the_acquisition_writes_nothing_after_it(station: _Station) -> None:
    interrupted: list[BaseException] = []

    def run() -> None:
        try:
            station.run_engine(station.plan()(num=3, **{RESERVED_OWNER_KWARG: _OWNER}))
        except RunEngineInterrupted as exc:
            interrupted.append(exc)

    runner = threading.Thread(target=run)
    runner.start()
    assert station.busy.wait(station.hang_s), "the acquisition never started"
    station.run_engine.request_pause(defer=False)
    runner.join(station.hang_s)
    assert interrupted, "the pause did not interrupt the run"
    station.run_engine.abort("aborted during the acquisition")

    assert station.idle_again.wait(station.hang_s), "the acquisition did not finish by itself"
    assert station.statuses[1:] == ["TRUE", "FALSE"]
    assert station.run_engine.state == "idle"
