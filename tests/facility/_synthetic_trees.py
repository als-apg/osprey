"""Minimal synthetic ``data/facility`` trees for the facility build's tests.

A tree is a mapping of path (relative to ``data/facility``) to content: a string
is written as it is, a ``Deck`` is saved as a pyAT lattice file, anything else
is dumped as YAML. Every factory returns a fresh tree, so a test may edit it.

Two base trees:

* ``plain_tree``: one quadrupole with a setpoint/readback pair and one BPM,
  no model;
* ``deck_tree``: the same devices wired into model ``SR``, a periodic deck,
  plus model ``LINE`` on a single-pass deck.

``with_trajectory`` gives one model of a tree that measures it the
``trajectory`` kind: the signal channels on a monitor device, the two injection
channels on the model's place, and the measurement file's keys.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "BPM",
    "LINE_TWISS",
    "QUAD",
    "READING",
    "SETTING",
    "TRAJECTORY_KEYS",
    "TURN_SAMPLES",
    "Deck",
    "deck_tree",
    "line_deck",
    "model",
    "plain_tree",
    "sr_deck",
    "trajectory_channels",
    "with_trajectory",
    "write_tree",
]

QUAD = "Quadrupole"
BPM = "BeamPositionMonitor"

#: The engine block of a quadrupole setpoint and of a BPM reading.
SETTING = {"attribute": "PolynomB", "index": 1}
READING = {"attribute": "PolynomB", "index": 0}

#: The initial conditions of the single-pass model.
LINE_TWISS = {"beta": [5.0, 3.0], "alpha": [0.0, 0.0]}

#: The samples each turn-by-turn channel of a trajectory tree holds.
TURN_SAMPLES = 32

#: The step and settle keys a trajectory tree's measurement file states.
TRAJECTORY_KEYS = {"corrector_delta": 1.0e-5, "alpha": 0.5, "n_discard": 0, "gain": 1.0}


@dataclass(frozen=True)
class Deck:
    """A lattice file, built from pyAT's element classes when written.

    Attributes:
        elements: Called with the ``at`` module; returns the elements in order.
    """

    elements: Callable[[Any], list[Any]]

    def save(self, path: Path) -> None:
        """Write the deck as a pyAT lattice file.

        Args:
            path: The file to write; its parents are created.
        """
        import at

        lattice = at.Lattice(self.elements(at), energy=3e9, particle="electron", periodicity=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        at.save_lattice(lattice, str(path))


def _sr_elements(at: Any) -> list[Any]:
    """QFA 0-0.25, M1, D, QD 1.25-1.55, D, M2, BPM1 at 2.55, D, M3, QFB 3.55-3.8."""
    return [
        at.Quadrupole("QFA", 0.25, 1.1),
        at.Marker("M1"),
        at.Drift("D", 1.0),
        at.Quadrupole("QD", 0.3, -1.0),
        at.Drift("D", 1.0),
        at.Marker("M2"),
        at.Monitor("BPM1"),
        at.Drift("D", 1.0),
        at.Marker("M3"),
        at.Quadrupole("QFB", 0.25, 1.1),
    ]


def sr_deck(*extra: Callable[[Any], Any]) -> Deck:
    """The periodic deck, with extra elements appended at its end.

    Its three drifts share the name ``D``; nothing wires them.

    Args:
        extra: Each called with the ``at`` module; returns one element.

    Returns:
        The deck.
    """
    return Deck(lambda at: [*_sr_elements(at), *(make(at) for make in extra)])


def line_deck() -> Deck:
    """The single-pass deck: M0 at 0, Q1 at 1.0-1.2, LBPM at 2.2."""
    return Deck(
        lambda at: [
            at.Marker("M0"),
            at.Drift("DL", 1.0),
            at.Quadrupole("Q1", 0.2, 0.9),
            at.Drift("DL2", 1.0),
            at.Monitor("LBPM"),
        ]
    )


def plain_tree() -> dict[str, Any]:
    """One quadrupole with a setpoint/readback pair and one BPM; no model."""
    return {
        "records/devices.yaml": [
            {"id": "SR/Q1", "class": QUAD},
            {"id": "SR/BPM1", "class": BPM},
        ],
        "records/channels.yaml": [
            {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB", "on": {"device": "SR/Q1"}},
            {"id": "Q1:RB", "on": {"device": "SR/Q1"}},
            {"id": "BPM1:X", "on": {"device": "SR/BPM1"}},
        ],
    }


def deck_tree() -> dict[str, Any]:
    """Model SR on a periodic deck and model LINE on a single-pass deck.

    SR wires the split quadrupole QF (slices QFA, QFB), QD (element) and the
    BPM1 readback; LINE wires LINE/Q1. Places SR and LINE span their models.
    """
    return {
        "records/places.yaml": [
            {"id": "SR", "span": {"model": "SR", "from_marker": "M1"}},
            {"id": "LINE", "span": {"model": "LINE", "from_marker": "M0"}},
        ],
        "records/devices.yaml": [
            {"id": "SR/QF", "class": QUAD},
            {"id": "SR/QD", "class": QUAD},
            {"id": "SR/BPM1", "class": BPM},
            {"id": "LINE/Q1", "class": QUAD},
        ],
        "records/channels.yaml": [
            {"id": "QF:SP", "role": "setpoint", "on": {"device": "SR/QF"}},
            {"id": "QD:SP", "role": "setpoint", "on": {"device": "SR/QD"}},
            {"id": "BPM1:X", "on": {"device": "SR/BPM1"}},
            {"id": "LQ:SP", "role": "setpoint", "on": {"device": "LINE/Q1"}},
        ],
        "models.yaml": [
            {
                "name": "SR",
                "engine": "pyat",
                "deck": "decks/sr.json",
                "wiring": [
                    {
                        "address": "QF:SP",
                        "slices": [{"element": "QFA"}, {"element": "QFB"}],
                        "engine": dict(SETTING),
                    },
                    {"address": "QD:SP", "element": "QD", "engine": dict(SETTING)},
                    {"address": "BPM1:X", "element": "BPM1", "engine": dict(READING)},
                ],
            },
            {
                "name": "LINE",
                "engine": "pyat",
                "deck": "decks/line.json",
                "settings": {"pyat": {"solve": "single_pass", "twiss_in": dict(LINE_TWISS)}},
                "wiring": [{"address": "LQ:SP", "element": "Q1", "engine": dict(SETTING)}],
            },
        ],
        "decks/sr.json": sr_deck(),
        "decks/line.json": line_deck(),
    }


def model(tree: Mapping[str, Any], name: str, file: str = "models.yaml") -> dict[str, Any]:
    """The model record named ``name`` in one models file of a tree.

    Args:
        tree: The tree.
        name: The model's name.
        file: The models file.

    Returns:
        The record itself, for the caller to edit.
    """
    return next(record for record in tree[file] if record["name"] == name)


def trajectory_channels(device: str, stem: str) -> list[dict[str, Any]]:
    """The five signal channels the ``trajectory`` kind asks of one monitor device.

    Args:
        device: The device's id.
        stem: The prefix of the five addresses.

    Returns:
        ``<stem>:TBT:X``, ``:TBT:Y`` and ``:TBT:SUM`` (waveforms of
        ``TURN_SAMPLES``), the command ``<stem>:ARM`` and the flag
        ``<stem>:ARMED``.
    """
    on = {"device": device}
    series = {"value_type": "waveform", "shape": [TURN_SAMPLES], "on": on}
    return [
        {"id": f"{stem}:TBT:X", "signal": "bpm_tbt_x", **series},
        {"id": f"{stem}:TBT:Y", "signal": "bpm_tbt_y", **series},
        {"id": f"{stem}:TBT:SUM", "signal": "bpm_tbt_sum", **series},
        {
            "id": f"{stem}:ARM",
            "role": "setpoint",
            "value_type": "int",
            "signal": "bpm_arm_command",
            "on": on,
        },
        {"id": f"{stem}:ARMED", "value_type": "int", "signal": "bpm_armed_status", "on": on},
    ]


def with_trajectory(
    tree: dict[str, Any], name: str, device: str, marker: str | None = None
) -> dict[str, Any]:
    """``tree`` with model ``name`` measuring the ``trajectory`` kind and nothing else.

    The model's measurement file must already name ``groups.bpm``,
    ``groups.hcor`` and ``groups.vcor``, and a place must carry the model's
    name. ``device`` gets the five signal channels under the stem
    ``<name>:BPM``; the place gets the request ``<name>:INJ:REQ``, which the
    model wires, and the flag ``<name>:INJ:BUSY``, which it does not.

    Args:
        tree: A tree whose model ``name`` has a measurement file.
        name: The model's name.
        device: The one member device of the model's ``groups.bpm`` group.
        marker: The deck marker a periodic model launches from; ``None`` for a
            single-pass model, which states no launch.

    Returns:
        ``tree``, edited in place.
    """
    request, busy = f"{name}:INJ:REQ", f"{name}:INJ:BUSY"
    tree["records/channels.yaml"] += [
        *trajectory_channels(device, f"{name}:BPM"),
        {
            "id": request,
            "role": "setpoint",
            "value_type": "int",
            "signal": "injection_request_command",
            "on": {"place": name},
        },
        {
            "id": busy,
            "value_type": "int",
            "signal": "injection_busy_status",
            "on": {"place": name},
        },
    ]
    record = model(tree, name)
    record["wiring"].append({"address": request, "engine": {"attribute": "inject"}})
    if marker is not None:
        record.setdefault("settings", {}).setdefault("pyat", {})["injection"] = {"marker": marker}
    measurement = tree[f"measurement/{name}.yaml"]
    measurement["kinds"] = ["trajectory"]
    measurement.setdefault("instruments", {}).update(
        {"injection_request": request, "injection_busy": busy}
    )
    measurement.update(TRAJECTORY_KEYS)
    return tree


def with_shot_wiring(tree: dict[str, Any], name: str, element: str) -> dict[str, Any]:
    """``tree`` with model ``name`` wiring the shot records :func:`with_trajectory` added.

    The three turn series read at ``element``; the arm command and the armed
    status name no element. The stem is ``<name>:BPM``, as in
    :func:`with_trajectory`.

    Args:
        tree: A tree :func:`with_trajectory` has given model ``name``.
        name: The model's name.
        element: The deck element the series read at.

    Returns:
        ``tree``, edited in place.
    """
    stem = f"{name}:BPM"
    model(tree, name)["wiring"] += [
        *(
            {
                "address": f"{stem}:TBT:{plane.upper()}",
                "element": element,
                "engine": {"axis": plane, "series": "turn"},
            }
            for plane in ("x", "y", "sum")
        ),
        {"address": f"{stem}:ARM", "engine": {"attribute": "arm"}},
        {"address": f"{stem}:ARMED", "engine": {"attribute": "armed"}},
    ]
    return tree


def write_tree(root: Path, tree: Mapping[str, Any]) -> Path:
    """Write a tree under ``root``.

    Args:
        root: The ``data/facility`` directory to create.
        tree: Path to content.

    Returns:
        ``root``.
    """
    for rel, data in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, Deck):
            data.save(path)
        elif isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    return root
