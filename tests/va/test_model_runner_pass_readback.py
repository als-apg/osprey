"""A write pass reads back only what its write can change.

An apply writes one setting per corrector, and a write that arrives alone is a
pass of its own, so a pass that read the whole roster back would put a full
publishing pass behind every such write. A physics setting reaches its own model
alone, and there moves itself, its readback and what the model computes from
its settings; a texture setpoint moves itself, its echo and the texture
channels derived from either. The first tests check that on the built example's
composite in process, through the helpers the runner uses: every value a
corrector, a quadrupole or a texture setpoint write changes is one of the
setpoint's records. The last checks that the runner reads back exactly that for
a pass of setpoints, and the usual roster for a pass that writes nothing; it
needs the serving package but no transport.
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from osprey.services.virtual_accelerator.serving.runner_config import (
    pass_readback,
    periodic_addresses,
    setting_records,
    write_records,
)
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.view import SimulatorView

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = pytest.mark.slow

T0 = 1_760_000_000.0

CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
CORRECTOR_READBACK = "SR:MAG:HCM:01:CURRENT:RB"
MONITOR = "SR:DIAG:BPM:01:POSITION:X"


@pytest.fixture
def view(built_demo: BuiltProject) -> SimulatorView:
    return SimulatorView.of_render(built_demo.build_dir)


@pytest.fixture
def composite(view: SimulatorView, tmp_path: Path) -> Composite:
    return Composite(view, state_dir=tmp_path / "state", clock=lambda: T0, model_log=False)


def _roster(composite: Composite, view: SimulatorView) -> list[str]:
    """What a write pass reads back before it is narrowed to a setting's records."""
    periodic = periodic_addresses(view)
    return [name for name in composite.supported_variables if name not in periodic]


def _same(before: Any, after: Any) -> bool:
    if isinstance(before, np.ndarray) or isinstance(after, np.ndarray):
        return bool(np.array_equal(np.asarray(before), np.asarray(after), equal_nan=True))
    if isinstance(before, float) and isinstance(after, float):
        return before == after or (np.isnan(before) and np.isnan(after))
    return bool(before == after)


def _changed_by(composite: Composite, roster: list[str], values: dict[str, Any]) -> set[str]:
    """The roster's addresses whose value ``values``'s write changed."""
    before = composite.get(roster)
    composite.set(values)
    after = composite.get(roster)
    return {name for name in roster if not _same(before[name], after[name])}


@pytest.mark.parametrize(
    ("setting", "readback"),
    [
        (CORRECTOR, CORRECTOR_READBACK),
        ("SR:MAG:VCM:10:CURRENT:SP", "SR:MAG:VCM:10:CURRENT:RB"),
        ("SR:MAG:QF:01:CURRENT:SP", "SR:MAG:QF:01:CURRENT:RB"),
    ],
)
def test_every_value_a_setting_write_changes_is_one_of_its_setting_records(
    composite: Composite, view: SimulatorView, setting: str, readback: str
) -> None:
    roster = _roster(composite, view)
    level = composite.get(setting)

    changed = _changed_by(composite, roster, {setting: level + 0.5})

    assert {setting, readback, MONITOR} <= changed
    assert changed <= pass_readback(setting_records(view), [setting])


def _texture_setpoints(composite: Composite, view: SimulatorView) -> list[str]:
    """The writable setpoints no physics model binds, in address order."""
    bound = setting_records(view)
    return sorted(
        address
        for address, variable in composite.supported_variables.items()
        if not variable.read_only and address not in bound
    )


def test_every_setpoint_has_write_records(composite: Composite, view: SimulatorView) -> None:
    records = write_records(view, composite.texture_reach())

    assert _texture_setpoints(composite, view)
    assert {
        address
        for address, variable in composite.supported_variables.items()
        if not variable.read_only
    } <= records.keys()


def test_every_value_a_texture_write_changes_is_one_of_its_write_records(
    composite: Composite, view: SimulatorView
) -> None:
    roster = _roster(composite, view)
    records = write_records(view, composite.texture_reach())

    for setpoint in _texture_setpoints(composite, view)[:25]:
        level = composite.get(setpoint)
        if not isinstance(level, float):
            continue
        low, high = composite.supported_variables[setpoint].value_range or (None, None)
        target = level + 0.5 if high is None or level + 0.5 <= high else level - 0.5

        changed = _changed_by(composite, roster, {setpoint: target})

        assert setpoint in changed
        assert changed <= pass_readback(records, [setpoint]), setpoint


def test_each_write_pass_reads_back_what_its_write_can_change(
    composite: Composite, view: SimulatorView, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip(
        "lume_pva_apg", reason="the served VA runs on lume-pva-apg, which ships for Linux only"
    )
    from lume_pva_apg.runner import Runner

    from osprey.services.virtual_accelerator.serving.runner import ModelRunner

    texture = _texture_setpoints(composite, view)[0]
    other = view.status_addresses()["BTS"]
    roster = sorted({CORRECTOR, "SR:DIAG:TUNE:X", texture, other})
    read: list[list[str]] = []
    monkeypatch.setattr(Runner, "_cycle_output_names", lambda _self: list(roster))
    monkeypatch.setattr(
        Runner, "_run_cycle", lambda self, _item: read.append(self._cycle_output_names())
    )
    runner = object.__new__(ModelRunner)
    runner.queue = queue.Queue()
    runner._periodic = periodic_addresses(view)
    runner._setting_records = write_records(view, composite.texture_reach())

    for values in ({CORRECTOR: 0.5}, {texture: 1.0}, {CORRECTOR: 0.5, texture: 1.0}, {}):
        ModelRunner._run_cycle(runner, {"values": values, "done": []})

    setting_pass, texture_pass, mixed_pass, empty_pass = read
    assert setting_pass == ["SR:DIAG:TUNE:X", CORRECTOR]
    assert texture_pass == [texture]
    assert mixed_pass == sorted({"SR:DIAG:TUNE:X", CORRECTOR, texture})
    assert empty_pass == roster
