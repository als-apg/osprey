"""A render's closed-orbit models, their monitors and correctors, and their orbit response.

Membership, plane and role come from each model's wiring in the render's
simulator view (``role`` monitor or setpoint, ``plane`` x or y), never from
group names or engine-private keys. Ring order is by lattice index, which the
model's engine plug-in states through its optional ``element_index``; the
ring length through its optional ``deck_length``. Golden signals and
tolerances are read from the render's facility file.

Every view, engine and lattice import happens inside a function body, so
importing the SDK loads none of them.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.apps.sdk._facts import _render_root, read_facts

__all__ = [
    "Member",
    "ModelMembers",
    "ResponseMatrix",
    "closed_orbit_models",
    "model_members",
    "model_response_matrix",
]

#: The engine whose models solve a closed orbit unless their settings say otherwise.
_CLOSED_ORBIT_ENGINE = "pyat"

#: The ``solve`` setting of a model tracked once through a line.
_SINGLE_PASS = "single_pass"  # noqa: S105 -- a solve setting, not a secret

#: The planes a member reads or steers.
_PLANES = ("x", "y")

#: The device signal holding a BPM's golden reference, by plane.
_GOLDEN_SIGNAL = "position_{plane}_golden_readback"

#: How much of a failed child's stderr an error carries.
_STDERR_TAIL = 2000


@dataclass(frozen=True)
class Member:
    """One BPM plane or one corrector of a model, placed on the ring.

    Stable: part of the application SDK's public surface.

    Attributes:
        element: The lattice element the wiring names.
        plane: ``x`` or ``y``.
        index: The element's lattice index; ring order and "downstream" go by it.
        s_m: The element's entrance along the ring, in metres; for placing it
            on a strip only (co-located elements share one).
        unit: The unit of the BPM's position or the corrector's setpoint.
        position_addr: A BPM's position readback; ``None`` on a corrector.
        golden_addr: A BPM's golden reference, the device's
            ``position_<plane>_golden_readback``; ``None`` when the device has
            no such signal, and on a corrector.
        setpoint_addr: A corrector's setpoint; ``None`` on a BPM.
        readback_addr: A corrector's readback, the setpoint's pair; ``None``
            on a BPM.
        tolerance: A corrector setpoint's absolute tolerance, in its unit;
            ``None`` when the facility file states none or only a relative one,
            and on a BPM.
    """

    element: str
    plane: str
    index: int
    s_m: float
    unit: str | None
    position_addr: str | None = None
    golden_addr: str | None = None
    setpoint_addr: str | None = None
    readback_addr: str | None = None
    tolerance: float | None = None


@dataclass(frozen=True)
class ModelMembers:
    """A model's BPMs and correctors, each in ring order.

    Stable: part of the application SDK's public surface.

    Attributes:
        model: The model's name.
        length_m: The model's ring length, in metres.
        bpms: One member per BPM element and plane, by lattice index, ``x`` before ``y``.
        correctors: One member per corrector setpoint, by lattice index, ``x`` before ``y``.
    """

    model: str
    length_m: float
    bpms: list[Member]
    correctors: list[Member]


@dataclass(frozen=True)
class ResponseMatrix:
    """A model's solved orbit response.

    Stable: part of the application SDK's public surface.

    Attributes:
        model: The model solved.
        rows: The monitor readback addresses, one row each.
        columns: The solved corrector setpoint addresses, one column each.
        matrix: One list per row: each entry the readback change in its
            hardware unit per setpoint change in its own.
        unsolved: Each corrector asked whose step has no solve, mapped to the
            engine's text.
    """

    model: str
    rows: list[str]
    columns: list[str]
    matrix: list[list[float]]
    unsolved: dict[str, str]


def closed_orbit_models(render_root: Path | None = None) -> list[str]:
    """The served models that solve a closed orbit, in facility order.

    Stable: part of the application SDK's public surface.

    A model counts when the render serves it, its engine is ``pyat`` and its
    ``solve`` setting is not ``single_pass`` (absent means periodic, as the
    engine defaults). The ``texture`` stand-in never counts.

    Args:
        render_root: The directory holding the rendered ``config.yml``; the
            render this process runs from when omitted.
    """
    facts = read_facts(render_root)
    return [
        str(entry["name"])
        for entry in facts.get("models") or ()
        if entry.get("served")
        and entry.get("engine") == _CLOSED_ORBIT_ENGINE
        and entry.get("solve") != _SINGLE_PASS
    ]


def _deck_model(view: Any, model: str) -> Any:
    for entry in view.physics_models():
        if entry.name == model and entry.deck is not None:
            return entry
    served = ", ".join(entry.name for entry in view.physics_models() if entry.deck is not None)
    raise ValueError(
        f"model {model!r} is not a served deck model of this render; the served ones: {served}"
    )


def _engine(model: Any) -> Any:
    from osprey.simulation.model_request import engine_module

    return importlib.import_module(engine_module(str(model.engine)))


def _optional(engine: Any, name: str, model: str) -> Any:
    found = getattr(engine, name, None)
    if found is None:
        raise ValueError(
            f"the engine plug-in {engine.__name__} of model {model} states no {name}; "
            "the model's ring order needs it"
        )
    return found


def _absolute_tolerance(record: Mapping[str, Any] | None) -> float | None:
    tolerance = (record or {}).get("tolerance")
    if not isinstance(tolerance, Mapping) or "absolute" not in tolerance:
        return None
    return abs(float(tolerance["absolute"]))


def _golden(records: Any, device: str | None, plane: str) -> str | None:
    if device is None:
        return None
    signal = _GOLDEN_SIGNAL.format(plane=plane)
    for channel in records.channels_on(device):
        if channel.get("signal") == signal:
            return str(channel["id"])
    return None


def _unit(view: Any, address: str) -> str | None:
    try:
        unit = view.channel(address).unit
    except KeyError:
        return None
    return None if unit is None else str(unit)


def model_members(model: str, render_root: Path | None = None) -> ModelMembers:
    """A served model's BPMs and correctors, in ring order, with the ring's length.

    Stable: part of the application SDK's public surface.

    BPMs are the model's monitor wiring, correctors its setpoint wiring, each
    on plane ``x`` or ``y`` and naming one element. A corrector's readback is
    its setpoint's pair; its tolerance the setpoint record's absolute one.

    Args:
        model: A served model with a deck.
        render_root: The directory holding the rendered ``config.yml``; the
            render this process runs from when omitted.

    Returns:
        The model's members.

    Raises:
        ValueError: The model is not a served deck model, its engine plug-in
            is not installed or states no ``element_index`` or no
            ``deck_length``.
    """
    from osprey.facility import FACILITY_FILE
    from osprey.facility.records import FacilityRecords
    from osprey.simulation.model_request import device_of
    from osprey_connectors.simulation.view import SimulatorView

    root = _render_root() if render_root is None else Path(render_root)
    view = SimulatorView.of_render(root)
    chosen = _deck_model(view, model)
    engine = _engine(chosen)
    element_index = _optional(engine, "element_index", model)
    deck_length = _optional(engine, "deck_length", model)
    records = FacilityRecords.from_file(root / FACILITY_FILE) or FacilityRecords({})

    bpms: list[Member] = []
    correctors: list[Member] = []
    for binding in chosen.bindings:
        if binding.plane not in _PLANES or binding.element is None:
            continue
        if binding.role not in ("monitor", "setpoint"):
            continue
        element = str(binding.element)
        index = int(element_index(chosen.deck, element, model=model))
        s_m = float(engine.locate(chosen.deck, element, model=model)[0])
        unit = _unit(view, binding.address)
        if binding.role == "monitor":
            bpms.append(
                Member(
                    element=element,
                    plane=binding.plane,
                    index=index,
                    s_m=s_m,
                    unit=unit,
                    position_addr=binding.address,
                    golden_addr=_golden(records, device_of(view, binding.address), binding.plane),
                )
            )
        else:
            try:
                readback = view.channel(binding.address).pair
            except KeyError:
                readback = None
            correctors.append(
                Member(
                    element=element,
                    plane=binding.plane,
                    index=index,
                    s_m=s_m,
                    unit=unit,
                    setpoint_addr=binding.address,
                    readback_addr=readback,
                    tolerance=_absolute_tolerance(records.record("channel", binding.address)),
                )
            )

    def order(member: Member) -> tuple[int, int]:
        return member.index, _PLANES.index(member.plane)

    return ModelMembers(
        model=model,
        length_m=float(deck_length(chosen.deck, model=model)),
        bpms=sorted(bpms, key=order),
        correctors=sorted(correctors, key=order),
    )


def model_response_matrix(
    model: str,
    correctors: list[str] | None = None,
    render_root: Path | None = None,
    timeout_s: float = 120,
) -> ResponseMatrix:
    """Solve a served model's orbit response to its correctors, on the model alone.

    Stable: part of the application SDK's public surface.

    Read-only: nothing is written to the machine or to the deck. The request
    is resolved as the ``model_response_matrix`` tool resolves it (each
    corrector stepped by the model's measurement-file ``corrector_delta``,
    converted to its hardware unit) and solved by the same child script, run
    in a child interpreter that is killed at ``timeout_s``.

    Args:
        model: A served deck model.
        correctors: Corrector setpoint addresses the model wires; every one,
            horizontal then vertical, when omitted.
        render_root: The directory holding the rendered ``config.yml``; the
            render this process runs from when omitted.
        timeout_s: Seconds the solve may take.

    Returns:
        The response. ``columns`` may be empty when no corrector solves;
        ``unsolved`` then names every one.

    Raises:
        ValueError: The request is refused: by the resolver (the message
            names the argument, the model or the file), or by the engine.
        TimeoutError: The solve did not finish within ``timeout_s``.
        RuntimeError: The request has no solve, or the child ended without a
            result.
    """
    from osprey.simulation.model_request import (
        UNSOLVED_KEY,
        child_script,
        resolve_model_request,
        result_line,
    )

    root = _render_root() if render_root is None else Path(render_root)
    request = resolve_model_request(model, correctors, render_root=root)
    try:
        done = subprocess.run(
            [sys.executable, "-c", child_script(request)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"the response of {request.model} was not solved within {timeout_s} s; "
            "the solve was stopped"
        ) from None
    try:
        line = result_line(done.stdout or "")
    except ValueError:
        line = None
    if line is None:
        raise RuntimeError(
            f"the solve of {request.model} ended without a result: "
            f"{(done.stderr or '')[-_STDERR_TAIL:]}"
        )
    if line.get("cause") == "solve":
        raise RuntimeError(f"the response of {request.model} has no solve: {line['refused']}")
    if "refused" in line:
        raise ValueError(f"the response of {request.model} was refused: {line['refused']}")
    return ResponseMatrix(
        model=request.model,
        rows=[str(row) for row in line["rows"]],
        columns=[str(column) for column in line["columns"]],
        matrix=[[float(value) for value in row] for row in line["matrix"]],
        unsolved={str(key): str(text) for key, text in dict(line[UNSOLVED_KEY]).items()},
    )
