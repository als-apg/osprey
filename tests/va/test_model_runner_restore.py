"""A failed pass writes back the settings it cached, never a command.

The model runner caches every settable value before a pass and, when the pass
fails after ``model.set`` landed, writes the cache back. A command acts once and
holds no setting: writing it back would act again, and a batch holding one
model's command beside another owner's channel is refused whole by the
composite, which would leave the run loop without a restore. These tests run the
runner's cache-and-restore sequence against the built example's composite in
process, through the same helpers the runner uses, so they run on any host.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("at")

from osprey.services.virtual_accelerator.serving.runner_config import (
    command_addresses,
    restorable_state,
)
from osprey_connectors.simulation.composite import Composite, WriteRefusedError
from osprey_connectors.simulation.view import SimulatorView

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = pytest.mark.slow

T0 = 1_760_000_000.0

ARM = "SR:DIAG:BPM:01:TBT:ARM"
REQUEST = "SR:DIAG:INJECTION:REQUEST"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"


@pytest.fixture
def view(built_control_assistant: BuiltProject) -> SimulatorView:
    return SimulatorView.of_render(built_control_assistant.build_dir)


@pytest.fixture
def composite(view: SimulatorView, tmp_path: Path) -> Composite:
    return Composite(view, state_dir=tmp_path / "state", clock=lambda: T0, model_log=False)


def _settable(composite: Composite) -> list[str]:
    """The names the runner caches before a pass: every variable not read-only."""
    return [name for name, var in composite.supported_variables.items() if not var.read_only]


def test_the_example_commands_are_the_arm_and_the_injection_request(
    view: SimulatorView,
) -> None:
    commands = command_addresses(view)

    assert {ARM, REQUEST} <= commands
    assert CORRECTOR not in commands


def test_a_state_keeps_every_value_but_a_command_and_is_left_as_it_was() -> None:
    state = {ARM: 1, REQUEST: 3, CORRECTOR: 0.5}

    restored = restorable_state(state, frozenset({ARM, REQUEST}))

    assert restored == {CORRECTOR: 0.5}
    assert state == {ARM: 1, REQUEST: 3, CORRECTOR: 0.5}


def test_the_whole_cached_state_after_a_command_write_is_refused(
    composite: Composite,
) -> None:
    composite.set({ARM: 1})
    cached = composite.get(_settable(composite))

    with pytest.raises(WriteRefusedError, match="a command is written with channels of its own"):
        composite.set(cached)


def test_a_failed_pass_after_a_shot_restores_the_settings_and_fires_nothing(
    composite: Composite, view: SimulatorView
) -> None:
    composite.set({ARM: 1})
    composite.set({REQUEST: 1})
    sr = composite._children["SR"].model
    assert sr.shot_count == 1

    cached = restorable_state(composite.get(_settable(composite)), command_addresses(view))
    before = composite.held([CORRECTOR])[CORRECTOR]
    written: list[dict[str, Any]] = []
    real_set = composite.set

    def recording_set(values: dict[str, Any]) -> None:
        written.append(dict(values))
        real_set(values)

    composite.set = recording_set  # type: ignore[method-assign]
    composite.set({CORRECTOR: before + 1.0})

    composite.set(cached)

    assert not command_addresses(view) & set(written[-1])
    assert composite.held([CORRECTOR])[CORRECTOR] == pytest.approx(before)
    assert sr.shot_count == 1
    composite.set({CORRECTOR: before + 0.5})
    assert composite.held([CORRECTOR])[CORRECTOR] == pytest.approx(before + 0.5)
