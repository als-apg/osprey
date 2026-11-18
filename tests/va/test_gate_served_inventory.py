"""The served addresses of a view are composed once, and the image gate reads them there.

A virtual accelerator serves three classes of address its view's
``addresses.json`` lists: ``channels``, ``status`` and ``ioc_health``.
``served_addresses`` is the one composition of them, the model RPC's ``info``
lists exactly that, and ``scripts/va/build_and_boot_check.sh`` compares what
``info`` answers over the wire with the same function's reading of the view on
disk. A gate that spelled the classes out itself would fall behind the first
class added after it, and fail on a server that serves its view exactly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from osprey.services.virtual_accelerator.serving.model_surface import served_addresses
from osprey_connectors.simulation.view import SCHEMAS, SimulatorView
from tests._simulator_view import view_channel

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts/va/build_and_boot_check.sh"

SETPOINT = "T:SP"
STATUS = "T:SIM:M:STATUS"
HEALTH = [
    f"T:SIM:{model}:{signal}"
    for model in ("M", "texture")
    for signal in ("HEARTBEAT", "UPTIME", "ENGINEER")
]

#: An address class of ``addresses.json`` read by subscript: ``addresses["status"]``.
_CLASS_SUBSCRIPT = re.compile(r"""addresses\[["'](channels|status|ioc_health)["']\]""")


class _OlderView:
    """A view as a connectors release without ``ioc_health_by_model`` presents it."""

    def __init__(self, view: SimulatorView) -> None:
        self._view = view

    def __getattr__(self, name: str) -> Any:
        if name == "ioc_health_by_model":
            raise AttributeError(name)
        return getattr(self._view, name)


def _view(path: Path, *, health: bool = True) -> SimulatorView:
    """A view of one texture setpoint and a physics model ``M`` with no deck."""
    addresses: dict[str, Any] = {"channels": [SETPOINT], "status": [STATUS]}
    if health:
        addresses["ioc_health"] = list(HEALTH)
    documents = {
        "served_models.json": {"models": ["M", "texture"]},
        "addresses.json": addresses,
        "variables.json": {
            "code": "T",
            "models": [
                {
                    "name": name,
                    "engine": engine,
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [],
                }
                for name, engine in (("M", "stub"), ("texture", "texture"))
            ],
            "channels": [view_channel(SETPOINT, role="setpoint", writable=True)],
        },
        "seeds.json": {"seeds": {SETPOINT: {"nominal": 5.0}}},
        "scenarios.json": {"scenarios": [{"name": "nominal"}]},
    }
    directory = path / "simulator"
    directory.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (directory / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    return SimulatorView.open(directory)


def test_the_served_addresses_are_the_channels_then_status_then_ioc_health(
    tmp_path: Path,
) -> None:
    assert served_addresses(_view(tmp_path)) == [SETPOINT, STATUS, *HEALTH]


def test_a_view_without_the_field_serves_its_channels_and_status_alone(tmp_path: Path) -> None:
    assert served_addresses(_view(tmp_path, health=False)) == [SETPOINT, STATUS]


def test_a_view_without_the_accessor_serves_its_channels_and_status_alone(
    tmp_path: Path,
) -> None:
    assert served_addresses(_OlderView(_view(tmp_path))) == [SETPOINT, STATUS]


def test_the_gate_reads_the_served_addresses_from_the_one_composition() -> None:
    script = GATE.read_text(encoding="utf-8")

    assert "served_addresses(SimulatorView.open(view))" in script
    assert _CLASS_SUBSCRIPT.findall(script) == []
