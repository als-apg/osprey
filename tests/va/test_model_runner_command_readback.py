"""A write pass of commands alone reads back the shot records of the commands' models.

An arm writes one command per device, and every write is a pass of its own, so
a command pass that read the whole roster back would put one full publishing
pass per device between an arm and the injection that follows it. A command
writes no setting, so the shot records -- the commands, the armed flag and the
turn series -- are all such a pass can change. The first tests check that on the
built example's composite in process, through the helper the runner uses: every
value an arm or an injection changes is a shot record. The last checks that the
runner reads back exactly that for a command pass, and the usual roster for any
other; it needs the serving package but no transport.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("at")

from osprey.services.virtual_accelerator.serving.runner_config import (
    command_readback,
    periodic_addresses,
    shot_records,
)
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.view import SimulatorView

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = pytest.mark.slow

T0 = 1_760_000_000.0

ARM = "SR:DIAG:BPM:01:TBT:ARM"
ARMED = "SR:DIAG:BPM:01:TBT:ARMED"
REQUEST = "SR:DIAG:INJECTION:REQUEST"
SERIES = "SR:DIAG:BPM:01:TBT:SUM"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"


@pytest.fixture
def view(built_control_assistant: BuiltProject) -> SimulatorView:
    return SimulatorView.of_render(built_control_assistant.build_dir)


@pytest.fixture
def composite(view: SimulatorView, tmp_path: Path) -> Composite:
    return Composite(view, state_dir=tmp_path / "state", clock=lambda: T0, model_log=False)


def _roster(composite: Composite, view: SimulatorView) -> list[str]:
    """What a write pass that is not of commands alone reads back."""
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


def test_every_value_an_arm_and_an_injection_change_is_a_shot_record(
    composite: Composite, view: SimulatorView
) -> None:
    roster = _roster(composite, view)
    records = shot_records(view)

    armed = _changed_by(composite, roster, {ARM: 1})
    injected = _changed_by(composite, roster, {REQUEST: 1})

    assert ARMED in armed
    assert armed <= command_readback(records, [ARM])
    assert {ARMED, SERIES} <= injected
    assert injected <= command_readback(records, [REQUEST])


def test_a_command_pass_reads_back_its_shot_and_any_other_write_pass_the_roster(
    view: SimulatorView, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip(
        "lume_pva_apg", reason="the served VA runs on lume-pva-apg, which ships for Linux only"
    )
    from lume_pva_apg.runner import Runner

    from osprey.services.virtual_accelerator.serving.runner import ModelRunner

    roster = sorted({*command_readback(shot_records(view), [ARM]), CORRECTOR, "SR:DIAG:TUNE:X"})
    read: list[list[str]] = []
    monkeypatch.setattr(Runner, "_cycle_output_names", lambda _self: list(roster))
    monkeypatch.setattr(
        Runner, "_run_cycle", lambda self, _item: read.append(self._cycle_output_names())
    )
    runner = object.__new__(ModelRunner)
    runner._periodic = periodic_addresses(view)
    runner._shot_records = shot_records(view)

    for values in ({ARM: 1}, {ARM: 1, CORRECTOR: 0.5}, {}):
        ModelRunner._run_cycle(runner, {"values": values, "done": []})

    command_pass, setting_pass, empty_pass = read
    assert set(command_pass) == command_readback(shot_records(view), [ARM])
    assert setting_pass == roster
    assert empty_pass == roster
