"""The built demo's wiring defaults against the virtual accelerator's bindings.

Every setpoint the demo's ``SR`` model wires carries a ``default`` in
``build/facility.json``: the value the deck holds, through the wiring's
calibration. For each setpoint the virtual accelerator's bindings name, it is
the binding's ``nominal``; the wired cavity's frequency is the deck cavity's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

# xdist_group("built_control_assistant"): every module reading the session's one
# control-assistant build shares a worker, so the build runs once per run.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("built_control_assistant")]

REPO_ROOT = Path(__file__).resolve().parents[2]
VA_BINDINGS = (
    REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation/va_bindings.json"
)

#: The one wired cavity setpoint and the frequency, in MHz, its deck cavity holds.
CAVITY_SETPOINT = ("SR:RF:CAVITY:01:FREQUENCY:SP", 500.41692828147894)


def _setpoint_defaults(document: dict[str, Any]) -> dict[str, float]:
    """Each setpoint ``SR`` wires, to its wiring ``default``."""
    setpoints = {c["id"] for c in document["channels"] if c["role"] == "setpoint"}
    [model] = [m for m in document["models"] if m["name"] == "SR"]
    return {w["address"]: w["default"] for w in model["wiring"] if w["address"] in setpoints}


def _binding_nominals() -> dict[str, float]:
    bindings = json.loads(VA_BINDINGS.read_text(encoding="utf-8"))["bindings"]
    return {b["setpoint_address"]: b["nominal"] for b in bindings if b["setpoint_address"]}


def test_every_wired_setpoint_defaults_to_its_binding_nominal(
    built_control_assistant: BuiltProject,
) -> None:
    defaults = _setpoint_defaults(built_control_assistant.facility)
    nominals = _binding_nominals()
    address, frequency = CAVITY_SETPOINT

    assert len(defaults) == 349
    assert set(defaults) - set(nominals) == {address}
    for setpoint, default in sorted(defaults.items()):
        expected = frequency if setpoint == address else nominals[setpoint]
        assert default == pytest.approx(expected, abs=1e-9), setpoint
