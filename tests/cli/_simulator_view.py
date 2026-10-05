"""A simulator view under a stubbed ``build/``, for the ``osprey sim`` CLI tests.

The view holds one physics model, ``SR``, served by the ``stub`` engine this
module registers, plus the texture. ``SR`` fails to build when an active
scenario seeds its ``poison`` fault, with :data:`SR_ERROR` as its error text.
No real engine runs, so a test reads model status without a lattice.
"""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.composite import ENGINE_GROUP
from tests._simulator_view import write_scenarios_view

#: The rendered config of a mock deployment serving the view.
MOCK_CONFIG = """\
claude_code:
  provider: anthropic
control_system:
  type: mock
"""

#: The error text ``SR`` fails with while a scenario seeds its ``poison`` fault.
SR_ERROR = "closed orbit is not finite at 3 monitors"

#: A scenario that fails ``SR``.
SR_BROKEN = {
    "name": "sr-broken",
    "description": "SR cannot find a closed orbit.",
    "faults": {"SR": {"writes": {"poison": 1.0}}},
}

#: The scenario every view lists.
NOMINAL = {"name": "nominal", "description": "Baseline machine."}

_BPM = "SR:DIAG:BPM:01:POSITION:X"
_TEXTURE_SP = "SR:VAC:PUMP:01:CURRENT:SP"


class _StubModel(LUMEModel):
    """One readback that always reads the same value."""

    def __init__(self) -> None:
        self._variables: dict[str, Variable] = {
            _BPM: ScalarVariable(name=_BPM, read_only=True),
        }

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def reset(self) -> None:
        pass

    def _set(self, values: dict[str, Any]) -> None:
        del values

    def _get(self, names: list[str]) -> dict[str, Any]:
        return dict.fromkeys(names, 0.0)


def _build(model: str, wiring: Any, deck: Any, settings: Any, active: Any = None) -> Any:
    del model, wiring, deck, settings
    if (active or {}).get("poison"):
        raise RuntimeError(SR_ERROR)
    return _StubModel()


_STUB = SimpleNamespace(build=_build)


def register_stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve engine ``stub`` from this module in the composite's entry-point lookup."""
    real = importlib.metadata.entry_points

    def entry_points(**selection: Any) -> Any:
        if selection == {"group": ENGINE_GROUP, "name": "stub"}:
            return [SimpleNamespace(load=lambda: _STUB)]
        return real(**selection)

    monkeypatch.setattr(composite_module, "metadata", SimpleNamespace(entry_points=entry_points))


def _channel(address: str, owner: str, role: str) -> dict[str, Any]:
    return {
        "address": address,
        "role": role,
        "pair": address if role == "setpoint" else None,
        "value_type": "float",
        "unit": None,
        "description": None,
        "writable": False,
        "value_range": None,
        "owner": owner,
    }


def write_simulator_view(
    build: Path, scenarios: Sequence[Mapping[str, Any]] = (NOMINAL, SR_BROKEN)
) -> Path:
    """Write the view into ``<build>/data/simulator`` and return its directory.

    Args:
        build: The render directory, ``<repo>/build``.
        scenarios: The scenarios ``scenarios.json`` lists, sorted by name there.

    Returns:
        The view directory.
    """
    channels = [_channel(_BPM, "SR", "readback"), _channel(_TEXTURE_SP, "texture", "setpoint")]
    documents = {
        "served_models.json": {"models": ["SR", "texture"]},
        "addresses.json": {
            "channels": sorted(channel["address"] for channel in channels),
            "status": ["ca:SIM:SR:STATUS"],
        },
        "variables.json": {
            "code": "ca",
            "models": [
                {
                    "name": "SR",
                    "engine": "stub",
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [{"id": "0", "address": _BPM, "direction": "read"}],
                },
                {
                    "name": "texture",
                    "engine": "texture",
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [],
                },
            ],
            "channels": sorted(channels, key=lambda channel: channel["address"]),
        },
        "seeds.json": {"seeds": {_TEXTURE_SP: {"nominal": 5.0}}},
    }
    view = build / "data" / "simulator"
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_text(json.dumps(document), encoding="utf-8")
    write_scenarios_view(
        build, {str(s["name"]): {k: v for k, v in s.items() if k != "name"} for s in scenarios}
    )
    return view
