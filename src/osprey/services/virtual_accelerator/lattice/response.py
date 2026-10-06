"""One device's orbit response, measured on the model a tree serves.

:func:`orbit_response` is the oracle a facility's own exported response matrix
is checked against. It drives one actuator through the served model's write
path -- the variables, their calibrations and the single solve per write that
the serving layer wraps -- and returns what every bound monitor did about it,
in the units the export states.

**Physics over physics, through the document's own curves.** The sweep is
written in hardware units, because that is what a control system sets, and the
readings come back in hardware units, because that is what a control system
publishes. Both sides are then mapped to physics through the calibration the
binding that carries them declares: the actuator's ``calibration`` for the
setpoint, each monitor's ``calibration`` for its reading. So the quotient is
in the facility's physics units on both axes and is comparable with a matrix
measured on the machine, and neither the native unit the lattice happens to
hold nor a curve read backwards enters it.

**A bipolar sweep about the value the model is holding.** The two arms are
``held ± delta/2``, which makes the operating point the model's own retained
setpoint -- a caller sweeping about a particular point writes that point
first. The actuator's physics span is the *secant* between those two arms,
``calibration(held + delta/2) - calibration(held - delta/2)``, not the slope
of the curve at either arm or at the point between them: a sampled calibration
bends, and the span the beam actually saw is the one the two arms define.

**Nothing accumulates.** The setpoint is written back to the value it was
found at before the call returns, whether the sweep finished or a read of it
raised, so consecutive responses are independent and a caller is never left
mid-sweep. That write-back is one write. :func:`orbit_responses` sweeps many
actuators in turn and makes each one's write-back in the same write as the next
actuator's first arm, so the two share one solve and a pass over ``n``
actuators costs ``2n + 1`` solves; a write the solver refuses is made again as
the two writes a sweep per actuator would have made, so every result, and the
state the model is left in, is the one :func:`orbit_response` gives actuator by
actuator.

**The monitor bindings are passed in.** A served monitor variable carries the
facility's physics-to-hardware inverse alone, because publishing a reading is
all it does; the way back -- hardware to physics, which is what puts a reading
in the units an exported matrix is written in -- lives on the monitor's own
binding in ``va_bindings.json``. A caller holding the document that built the
model therefore hands over the monitor bindings it wants read, and one that
hands over none, or one the model does not serve, is refused rather than
quietly measuring a response with rows missing.

A physics value is the one the calibration states at the deck energy the
document was exported at, which is the energy a tree boots at while its bend
sits at nominal. A ring moved off that energy bends the same hardware setpoint
by the rigidity ratio, so its response scales by that ratio -- which is why
the energy a response file was measured at, and the energy the deck is built
for, are reported beside a comparison rather than folded into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from osprey.services.virtual_accelerator.lattice.calibration import to_physics
from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from osprey.services.virtual_accelerator.bindings import Binding, Calibration
    from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

__all__ = ["orbit_response", "orbit_responses"]

#: One actuator's response: each monitor element's ``(x, y)`` entry.
Response = dict[str, tuple[float, float]]

#: The two transverse planes, in the order a response entry pairs them.
_PLANES: tuple[str, str] = ("x", "y")


def orbit_response(
    model: PyATRingModel,
    binding: Binding,
    delta_hw: float,
    *,
    monitors: Iterable[Binding],
) -> dict[str, tuple[float, float]]:
    """Sweep one actuator and return the orbit response it produced.

    Args:
        model: The model to drive, built on the tree whose document these
            bindings come from. It is written to and left where it was found.
        binding: The actuator's binding -- the address swept, the calibration
            its hardware value is converted through, and the elements the
            write lands on.
        delta_hw: The full width of the sweep in hardware units; the arms are
            half of it either side of the setpoint the model is holding. The
            exported response file's own ``ActuatorDelta`` is what a
            comparison against that file uses.
        monitors: The monitor bindings to read, each carrying the element it
            reads, the plane it reads, and the calibration that puts its
            hardware reading in physics units.

    Returns:
        One entry per monitor element, each the element's ``(x, y)`` response
        in physics units per physics unit of the actuator. A plane no binding
        reads is ``nan``: the monitor published nothing there, which is not
        the same as having seen nothing move.

    Raises:
        ValueError: The binding is read only or carries no calibration, the
            sweep has no width, an address is not one the model serves, no
            monitor was given, one plane of one element is read twice, or the
            calibration maps both arms of the sweep onto one physics value.
        OrbitSolveError: An arm of the sweep leaves the lattice without a
            stable closed orbit. The model has restored the lattice itself,
            and the setpoint is written back before the error propagates.
    """
    (result,) = orbit_responses(model, [(binding, delta_hw)], monitors=monitors)
    if isinstance(result, OrbitSolveError):
        raise result
    return result


def orbit_responses(
    model: PyATRingModel,
    sweeps: Iterable[tuple[Binding, float]],
    *,
    monitors: Iterable[Binding],
) -> list[Response | OrbitSolveError]:
    """Sweep several actuators in turn and return each one's orbit response.

    Each result is the one :func:`orbit_response` returns for that actuator
    on its own, and the model is left as a sweep per actuator leaves it. An
    actuator whose sweep or write-back the solver refuses has that refusal as
    its result, and the actuators after it are still swept.

    Args:
        model: The model to drive, built on the tree whose document these
            bindings come from. It is written to and left where it was found.
        sweeps: Each actuator's binding and the full width of its sweep in
            hardware units, in the order they are swept.
        monitors: The monitor bindings to read, as :func:`orbit_response`
            takes them.

    Returns:
        One entry per sweep, in order: the actuator's response, or the
        ``OrbitSolveError`` the solver raised on one of its writes.

    Raises:
        ValueError: Any refusal :func:`orbit_response` makes, for any sweep,
            or one actuator swept twice -- each raised before anything is
            written.
    """
    planes = _planes_to_read(model, monitors)
    planned = _planned(model, sweeps)
    results: list[Response | OrbitSolveError] = []
    pending: list[_Owed] = []
    try:
        for plan in planned:
            current = _Owed(plan.address, plan.held)
            pending.append(current)
            arms: list[dict[str, float]] = []
            try:
                for arm in (0.5 * plan.delta, -0.5 * plan.delta):
                    _write(model, {plan.address: plan.held + arm}, pending, results)
                    arms.append(_physics_readings(model, planes))
            except OrbitSolveError as exc:
                current.result = exc
            else:
                high, low = arms
                current.result = {
                    element: (
                        _entry(monitors_by_plane.get("x"), high, low, plan.span),
                        _entry(monitors_by_plane.get("y"), high, low, plan.span),
                    )
                    for element, monitors_by_plane in planes.items()
                }
        for owed in list(pending):
            results.append(_settle(model, owed))
            pending.remove(owed)
    except BaseException:
        if pending:
            model.set({owed.address: owed.held for owed in pending})
        raise
    return results


@dataclass(frozen=True)
class _Planned:
    """One actuator's sweep, checked before anything is written."""

    address: str
    held: float
    delta: float
    span: float


@dataclass
class _Owed:
    """An actuator whose setpoint is still to be written back, and its result."""

    address: str
    held: float
    result: Response | OrbitSolveError | None = None


def _planned(model: PyATRingModel, sweeps: Iterable[tuple[Binding, float]]) -> list[_Planned]:
    """Check every sweep and read the setpoint each one is held at.

    Raises:
        ValueError: any refusal of one actuator's sweep, or an actuator named
            by two sweeps.
    """
    planned: list[_Planned] = []
    seen: set[str] = set()
    for binding, delta_hw in sweeps:
        address = _actuator_address(model, binding)
        delta = _sweep_width(delta_hw)
        if address in seen:
            raise ValueError(
                f"actuator {address!r} is swept twice: each actuator of a sweep is one column "
                "of the response"
            )
        seen.add(address)
        held = float(model.get([address])[address])
        planned.append(_Planned(address, held, delta, _actuator_span(binding, held, delta)))
    return planned


def _write(
    model: PyATRingModel,
    write: dict[str, float],
    pending: list[_Owed],
    results: list[Response | OrbitSolveError],
) -> None:
    """Write one arm, together with the write-back the previous actuator owes.

    A refused batch is made again as the two writes a sweep per actuator
    makes: the write-back alone, whose refusal is that actuator's result, then
    the arm alone, whose refusal propagates to the actuator being swept.
    """
    if len(pending) < 2:
        model.set(write)
        return
    previous = pending[0]
    try:
        model.set({previous.address: previous.held, **write})
    except Exception:
        results.append(_settle(model, previous))
        pending.remove(previous)
        model.set(write)
    else:
        results.append(_settled(previous))
        pending.remove(previous)


def _settle(model: PyATRingModel, owed: _Owed) -> Response | OrbitSolveError:
    """Write one actuator back alone; its refusal, or else its result."""
    try:
        model.set({owed.address: owed.held})
    except OrbitSolveError as exc:
        return exc
    return _settled(owed)


def _settled(owed: _Owed) -> Response | OrbitSolveError:
    """The result of an actuator whose sweep has finished.

    Raises:
        RuntimeError: the actuator is written back before its sweep finished.
    """
    if owed.result is None:
        raise RuntimeError(f"actuator {owed.address!r} is written back before it was swept")
    return owed.result


def _actuator_address(model: PyATRingModel, binding: Binding) -> str:
    """The address the sweep writes, checked against what it can be.

    Raises:
        ValueError: the binding reads rather than writes, converts through
            something other than a calibration, or names an address this
            model does not serve.
    """
    address = binding.setpoint_address
    if not binding.is_writable:
        raise ValueError(
            f"binding {address!r} is read only: an orbit response is measured by writing an "
            "actuator, and a monitor has no setpoint to sweep"
        )
    if binding.calibration is None:
        raise ValueError(
            f"binding {address!r} carries no calibration, so its hardware sweep has no physics "
            "span to divide the readings by; the energy knob converts through its energy table "
            "and is not an actuator of an orbit response"
        )
    _require_served(model, address, "actuator")
    return address


def _require_served(model: PyATRingModel, address: str, role: str) -> None:
    """Refuse an address the model has no variable for.

    A binding the served manifest never resolved reaches no variable, so
    neither writing nor reading it does anything. Naming it is the difference
    between a response with a row missing and a response a caller can trust.

    Raises:
        ValueError: the model serves no variable on this address.
    """
    if address not in model.supported_variables:
        raise ValueError(
            f"the model serves no variable on {address!r}, so it cannot be the {role} of an "
            "orbit response; the manifest the model was built on does not carry that address"
        )


def _calibration(binding: Binding) -> Calibration:
    """The conversion a binding of the sweep converts through.

    Both roles are refused before anything is written -- the actuator by
    :func:`_actuator_address`, each monitor by :func:`_planes_to_read` -- so a
    binding that reaches a conversion carries one.

    Raises:
        ValueError: the binding converts through no calibration.
    """
    if binding.calibration is None:
        raise ValueError(
            f"binding {binding.setpoint_address!r} carries no calibration, so it has no "
            "physics value an orbit response can be measured in"
        )
    return binding.calibration


def _sweep_width(delta_hw: float) -> float:
    """The sweep width, as a number the two arms can be placed around.

    Raises:
        ValueError: the width is not finite, or is zero -- either leaves the
            two arms on one point and the response undefined.
    """
    delta = float(delta_hw)
    if not math.isfinite(delta) or delta == 0.0:
        raise ValueError(
            "the sweep is half the delta either side of the held setpoint, so its width must "
            f"be finite and nonzero; got {delta_hw!r}"
        )
    return delta


def _planes_to_read(
    model: PyATRingModel, monitors: Iterable[Binding]
) -> dict[str, dict[str, Binding]]:
    """Group the monitor bindings by the element and plane each one reads.

    The grouping is what pairs an element's two planes into one entry, and it
    is where a monitor that cannot be read is refused -- before anything is
    written, so a refusal never leaves a swept setpoint behind.

    Raises:
        ValueError: a binding is not a monitor, carries no calibration, reads
            neither plane, names an address this model does not serve, reads a
            plane another binding already read, or the iterable is empty.
    """
    planes: dict[str, dict[str, Binding]] = {}
    for monitor in monitors:
        address = monitor.setpoint_address
        if monitor.kind != "monitor":
            raise ValueError(
                f"binding {address!r} is a {monitor.kind} binding, not a monitor: an orbit is "
                "read off the solve, never off a setpoint"
            )
        if monitor.calibration is None:
            raise ValueError(
                f"monitor {address!r} carries no calibration, so its hardware reading has no "
                "physics value a response can be measured in"
            )
        plane = monitor.attribute
        if plane not in _PLANES:
            raise ValueError(
                f"monitor {address!r} reads {monitor.attribute!r}, which is neither transverse "
                f"plane ({' nor '.join(_PLANES)})"
            )
        _require_served(model, address, "monitor")
        element = str(monitor.element)
        read = planes.setdefault(element, {})
        if plane in read:
            raise ValueError(
                f"element {element!r} has its {plane} plane read twice, by "
                f"{read[plane].setpoint_address!r} and {address!r}: one plane of one element is "
                "one entry of the response"
            )
        read[plane] = monitor
    if not planes:
        raise ValueError(
            "an orbit response is what the monitors did, so it needs at least one monitor "
            "binding to read; none was given"
        )
    return planes


def _actuator_span(binding: Binding, held: float, delta: float) -> float:
    """The physics distance between the two arms of the sweep.

    Raises:
        ValueError: the calibration maps both arms onto one physics value, so
            the beam is asked for two identical settings and the response has
            nothing to be measured against.
    """
    calibration = _calibration(binding)
    high = float(to_physics(calibration, held + 0.5 * delta))
    low = float(to_physics(calibration, held - 0.5 * delta))
    span = high - low
    if span == 0.0:
        raise ValueError(
            f"the calibration of {binding.setpoint_address!r} maps both arms of a sweep of "
            f"{delta} about {held} onto the physics value {high}, so the two arms ask the beam "
            "for the same thing and no response can be divided out"
        )
    return span


def _physics_readings(
    model: PyATRingModel, planes: dict[str, dict[str, Binding]]
) -> dict[str, float]:
    """Read every monitor of one arm, in physics units, keyed by address.

    One batched read, so every reading belongs to the orbit the arm's write
    solved for, and each is mapped through its own monitor's calibration.
    """
    monitors = [monitor for read in planes.values() for monitor in read.values()]
    hardware = model.get([monitor.setpoint_address for monitor in monitors])
    return {
        monitor.setpoint_address: float(
            to_physics(_calibration(monitor), hardware[monitor.setpoint_address])
        )
        for monitor in monitors
    }


def _entry(
    monitor: Binding | None,
    high: dict[str, float],
    low: dict[str, float],
    span: float,
) -> float:
    """One plane's response, or ``nan`` where no binding reads that plane."""
    if monitor is None:
        return math.nan
    address = monitor.setpoint_address
    return (high[address] - low[address]) / span
