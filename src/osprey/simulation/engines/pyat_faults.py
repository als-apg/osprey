"""What a faulty monitor reads and what a miscalibrated supply delivers.

Two formulas, ported from pySC (Python Simulated Commissioning,
https://github.com/kparasch/pySC) as straight arithmetic, so their sign and
roll conventions are pySC's:

* :func:`bpm_read` -- pySC's ``BPMSystem.capture_orbit`` chain for one
  monitor after the roll, in the unit the monitor publishes: offset,
  calibration error, polarity, noise, gain. :func:`rolled` is the roll, in the
  deck's unit.
* :func:`magnet_cal` -- pySC's ``LinearConv.transform``: the value a supply
  delivers against the one commanded.

**The faults live on the deck elements.** A monitor element carries
``readout_<field>_x`` and ``readout_<field>_y`` for ``offset``, ``gain``,
``noise`` and ``polarity``, and one ``readout_roll``. A magnet setpoint's
supply calibration sits on the first element it binds, under attributes named
for the setpoint (:func:`supply_attribute`), so setpoints that share an
element each keep their own. An attribute the element does not carry is its
identity value, so an unseeded deck reads and writes exactly.

**The reading is split from the solve.** A monitor variable returns the solved
truth; :func:`readout` turns truth -- plus whatever beam motion the caller has
added to it -- into what each monitor reports. It runs once per monitor
element on its (x, y) pair, because a roll mixes the two planes, and its noise
is a pure function of the reading's address and the time it is read at. The
roll turns the beam position before the monitor's conversion to the unit it
publishes; the rest of the chain acts on the published number.

**Element errors are keyed by the element, not by an address.** One series
supply address feeds many magnets, so a magnet's own misalignment and its own
multipole error cannot hang off the address. Every element a non-monitor
wiring record binds -- the served deck's element name -- carries ``dx``,
``dy`` and ``roll`` (:data:`MISALIGNMENT_BOUNDS`), and every element bound on
``PolynomB`` or ``PolynomA`` also carries ``dB<n>`` and ``dA<n>`` for the
polynomial index ``n`` (:data:`MULTIPOLE_INDICES`). They are fault names like
any other, ``<element>/<field>``, each with its explicit value: drawing a
random realisation is the scenario author's job, and the engine draws nothing.
They are seeded once, when the model is built, never written afterwards: the
misalignment as the element's entrance and exit transforms, the multipole as
an additive delta the element holds (:func:`multipole_attribute`) and every
write of a setpoint on the same coefficient keeps (:func:`multipole_error`).
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.simulation.engines.calibration import field as _slot
from osprey.simulation.engines.calibration import to_hardware, to_physics

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from osprey.simulation.engines.calibration import Calibration

#: The readout fields every monitor readback carries on its own axis, each at
#: its identity: a monitor that reports the true orbit position exactly, and
#: what each readout attribute reads on an element that does not carry it.
#: The key order is the order a monitor's fault variables are declared in.
READOUT_IDENTITY: dict[str, float] = {
    "offset": 0.0,
    "gain": 1.0,
    "noise": 0.0,
    "polarity": 1.0,
}

#: The supply calibration fields a magnet setpoint carries, each at its
#: identity: a supply that delivers exactly what it was commanded.
SUPPLY_IDENTITY: dict[str, float] = {"cal_factor": 1.0, "cal_offset": 0.0}

_SUPPLY_PREFIX = "supply_"

#: What separates a wired address from a field in a fault name.
FAULT_SEPARATOR = "/"

#: The roll a monitor carries once, at its identity.
ROLL = "roll"
ROLL_IDENTITY = 0.0

#: The engine attributes whose setpoints carry a calibration fault.
CALIBRATED_ATTRIBUTES: frozenset[str] = frozenset({"PolynomB", "KickAngle"})

MIN_MONITOR_GAIN = 0.1
MAX_MONITOR_GAIN = 10.0
MAX_MONITOR_ROLL_RAD = 0.1
#: A factor of -1 is a polarity flip; a magnitude beyond 5x is never a real
#: calibration error.
MAX_CALIBRATION_FACTOR = 5.0

#: Fault field -> its inclusive (min, max). A field absent here is bounded by
#: well-formedness alone.
FAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "gain": (MIN_MONITOR_GAIN, MAX_MONITOR_GAIN),
    "noise": (0.0, math.inf),
    ROLL: (-MAX_MONITOR_ROLL_RAD, MAX_MONITOR_ROLL_RAD),
    "cal_factor": (-MAX_CALIBRATION_FACTOR, MAX_CALIBRATION_FACTOR),
}

#: The misalignment fields of an element error, each at its identity.
MISALIGNMENT_IDENTITY: dict[str, float] = {"dx": 0.0, "dy": 0.0, "roll": 0.0}

#: The largest transverse offset an element error may state, in metres.
MAX_ELEMENT_OFFSET_M = 0.01
#: The largest roll an element error may state, in radians.
MAX_ELEMENT_ROLL_RAD = 0.1

#: Misalignment field -> its inclusive (min, max).
MISALIGNMENT_BOUNDS: dict[str, tuple[float, float]] = {
    "dx": (-MAX_ELEMENT_OFFSET_M, MAX_ELEMENT_OFFSET_M),
    "dy": (-MAX_ELEMENT_OFFSET_M, MAX_ELEMENT_OFFSET_M),
    ROLL: (-MAX_ELEMENT_ROLL_RAD, MAX_ELEMENT_ROLL_RAD),
}

#: Multipole error field prefix -> the polynomial it adds to.
MULTIPOLE_PREFIXES: dict[str, str] = {"dB": "PolynomB", "dA": "PolynomA"}

#: The polynomial indices an element error may name: ``dB0`` (dipole) to
#: ``dB9`` (twenty-pole), and the same for ``dA``.
MULTIPOLE_INDICES: range = range(10)

#: A multipole delta is bounded by well-formedness alone (finite); its scale
#: is the polynomial index's own.
MULTIPOLE_BOUNDS: tuple[float, float] = (-math.inf, math.inf)

#: What a slot's ``scope`` says it is keyed by.
ADDRESS_SCOPE = "address"
ELEMENT_SCOPE = "element"

_MULTIPOLE_PREFIX = "error_"

#: A polarity is a direction: it lands exactly on one of these two values.
POLARITY_OPTIONS: tuple[float, float] = (-1.0, 1.0)

#: The subkey that makes a reading's noise stream its own.
NOISE_SUBKEY = b":readout_noise"

_AXES = ("x", "y")


def rolled(x: float, y: float, roll: float) -> tuple[float, float]:
    """The beam position ``(x, y)`` as a monitor rolled by ``roll`` sees it.

    The pair is the beam position in the deck's unit. A positive roll, in
    radians about the beam axis, rotates it counterclockwise
    (``[[cos, -sin], [sin, cos]]``), so a roll of pi/2 reads a purely
    horizontal position as purely vertical.

    Args:
        x: Horizontal beam position at the monitor.
        y: Vertical beam position at the monitor.
        roll: The monitor's roll, in radians.

    Returns:
        The turned ``(x, y)``, in the unit of the input.
    """
    return (
        math.cos(roll) * x - math.sin(roll) * y,
        math.sin(roll) * x + math.cos(roll) * y,
    )


def bpm_read(
    x: float,
    y: float,
    *,
    offset_x: float,
    offset_y: float,
    gain_x: float,
    gain_y: float,
    polarity_x: float,
    polarity_y: float,
    cal_x: float,
    cal_y: float,
    noise_x: float,
    noise_y: float,
    normal_x: float,
    normal_y: float,
) -> tuple[float, float]:
    """One monitor's reading of a true (x, y) position the roll has already turned.

    Subtract the offset, apply the calibration error and the polarity, add
    noise, then apply the gain -- in that order, so the noise is added before
    the gain multiply, as pySC does.

    Every length -- the true position, the offsets, the noise widths and the
    reading -- is in the unit the monitor publishes. The formula is
    scale-free, so one unit throughout is the whole requirement. The roll
    acts before this, on the beam position (:func:`rolled`).

    Args:
        x: True horizontal position at the monitor, after the roll.
        y: True vertical position at the monitor, after the roll.
        offset_x: Horizontal offset.
        offset_y: Vertical offset.
        gain_x: Horizontal gain, multiplied in last.
        gain_y: Vertical gain, multiplied in last.
        polarity_x: Horizontal polarity, +1.0 or -1.0.
        polarity_y: Vertical polarity, +1.0 or -1.0.
        cal_x: Horizontal calibration error, applied as ``1 + cal_x``.
        cal_y: Vertical calibration error, applied as ``1 + cal_y``.
        noise_x: Standard deviation of the horizontal noise.
        noise_y: Standard deviation of the vertical noise.
        normal_x: The standard normal draw the horizontal noise scales.
        normal_y: The standard normal draw the vertical noise scales.

    Returns:
        ``(reading_x, reading_y)`` in the unit the monitor publishes.
    """
    reading_x = (x - offset_x) * (1.0 + cal_x) * polarity_x + noise_x * normal_x
    reading_y = (y - offset_y) * (1.0 + cal_y) * polarity_y + noise_y * normal_y

    reading_x *= gain_x
    reading_y *= gain_y

    return float(reading_x), float(reading_y)


def magnet_cal(setpoint: float, *, factor: float = 1.0, offset: float = 0.0) -> float:
    """The value a supply delivers when ``setpoint`` is commanded.

    Args:
        setpoint: The commanded value, in the hardware unit.
        factor: Multiplicative calibration error; ``-1.0`` is a polarity flip,
            ``1.3`` a 30% error.
        offset: Additive calibration error, in the unit of ``setpoint``.

    Returns:
        ``setpoint * factor + offset``.
    """
    return setpoint * factor + offset


def supply_attribute(setpoint: str, field: str) -> str:
    """The element attribute that holds one field of a setpoint's supply calibration.

    No pyAT pass method reads it, so a calibration on the element never moves
    the orbit by itself.

    Args:
        setpoint: The setpoint's address.
        field: ``cal_factor`` or ``cal_offset``.

    Returns:
        ``supply_<field>[<setpoint>]``.
    """
    return f"{_SUPPLY_PREFIX}{field}[{setpoint}]"


def supply_calibration(element: Any, setpoint: str) -> dict[str, float]:
    """:func:`magnet_cal`'s ``factor`` and ``offset`` for one setpoint, as ``element`` holds them.

    Args:
        element: The deck element the setpoint's calibration sits on.
        setpoint: The setpoint's address.

    Returns:
        The two keywords, each at identity where the element carries none.
    """
    return {
        keyword: float(getattr(element, supply_attribute(setpoint, field), SUPPLY_IDENTITY[field]))
        for keyword, field in (("factor", "cal_factor"), ("offset", "cal_offset"))
    }


def multipole_attribute(attribute: str, index: int) -> str:
    """The element attribute that holds a multipole error on one polynomial coefficient.

    No pyAT pass method reads it; the delta itself sits in the polynomial,
    and this attribute is what lets a setpoint write keep it.

    Args:
        attribute: ``PolynomB`` or ``PolynomA``.
        index: The coefficient's index.

    Returns:
        ``error_<attribute>[<index>]``.
    """
    return f"{_MULTIPOLE_PREFIX}{attribute}[{index}]"


def multipole_error(element: Any, attribute: str, index: int | None) -> float:
    """The multipole delta ``element`` holds on ``attribute[index]``; 0 where it holds none."""
    if index is None or attribute not in MULTIPOLE_PREFIXES.values():
        return 0.0
    return float(getattr(element, multipole_attribute(attribute, index), 0.0))


def multipole_field(field: str) -> tuple[str, int] | None:
    """``(attribute, index)`` a multipole error field names, or ``None`` for any other field.

    ``dB2`` is ``("PolynomB", 2)``; an index outside :data:`MULTIPOLE_INDICES`
    is no multipole field.
    """
    prefix, digits = field[:2], field[2:]
    attribute = MULTIPOLE_PREFIXES.get(prefix)
    if attribute is None or not digits.isdigit() or str(int(digits)) != digits:
        return None
    index = int(digits)
    return (attribute, index) if index in MULTIPOLE_INDICES else None


@dataclass(frozen=True)
class FaultSlot:
    """One fault a model declares: what it perturbs and the field.

    Attributes:
        address: The wired address, or the element name for an element error.
        field: The fault field, such as ``offset``, ``cal_factor``, ``dx`` or
            ``dB2``.
        value_range: The inclusive ``(min, max)`` a value must lie in, or
            ``None`` where only well-formedness bounds it.
        options: The exact values a value must equal one of, or ``None``
            where the field is not an enumeration.
        element: The element the address is wired to (its first slice's);
            the element itself for an element error.
        scope: ``address`` for a fault keyed by a wired address,
            ``element`` for an element error keyed by the element's name.
    """

    address: str
    field: str
    value_range: tuple[float, float] | None
    options: tuple[float, ...] | None = None
    element: str | None = None
    scope: str = ADDRESS_SCOPE


def _first_element(record: Any) -> str | None:
    slices = _slot(record, "slices")
    element = _slot(slices[0], "element") if slices else _slot(record, "element")
    return None if element is None else str(element)


def _bound_elements(record: Any) -> list[str]:
    slices = _slot(record, "slices")
    if slices:
        return [str(_slot(item, "element")) for item in slices if _slot(item, "element")]
    element = _slot(record, "element")
    return [] if element is None else [str(element)]


def fault_variables(wiring: Iterable[Any]) -> dict[str, FaultSlot]:
    """Every fault a model built from ``wiring`` declares, by fault name.

    A fault name is ``<address>/<field>`` or, for an element error,
    ``<element>/<field>``. A monitor readback (an engine
    ``axis`` and no ``attribute``) carries ``offset``, ``gain``, ``noise`` and
    ``polarity`` on its own axis, the first readback of each element and axis
    only; each monitor element carries one ``roll``, on its x-axis readback or
    else its first one. A setpoint (``direction: write``) on ``PolynomB`` or
    ``KickAngle`` carries ``cal_factor`` and ``cal_offset``.

    Every element a record that is no monitor reading binds (its ``element``
    or each of its ``slices``) carries ``dx``, ``dy`` and ``roll``; one bound
    on ``PolynomB`` or ``PolynomA`` also carries ``dB<n>`` and ``dA<n>`` for
    every ``n`` of :data:`MULTIPOLE_INDICES`. These slots are of ``element``
    scope and keyed by the element's name. An element whose name is also a
    wired address carries none, so the two keyings never share a name.
    Nothing is imported and no deck is read.

    Args:
        wiring: The model's wiring records, each with its ``direction``.

    Returns:
        Fault name -> its slot, sorted by name.
    """
    slots: dict[str, FaultSlot] = {}
    monitors: dict[str, dict[str, str]] = {}
    magnets: dict[str, bool] = {}
    addresses: set[str] = set()

    def declare(address: str, field: str, element: str) -> None:
        if field == "polarity":
            slot = FaultSlot(address, field, None, POLARITY_OPTIONS, element)
        else:
            slot = FaultSlot(address, field, FAULT_BOUNDS.get(field), element=element)
        slots[f"{address}{FAULT_SEPARATOR}{field}"] = slot

    for record in wiring:
        address = str(_slot(record, "address"))
        addresses.add(address)
        block = _slot(record, "engine")
        attribute = _slot(block, "attribute")
        axis = _slot(block, "axis")
        element = _first_element(record)
        if element is None:
            continue
        if not (attribute is None and axis in _AXES):
            polynomial = attribute in MULTIPOLE_PREFIXES.values()
            for bound in _bound_elements(record):
                magnets[bound] = magnets.get(bound, False) or polynomial
        if attribute is None and axis in _AXES:
            monitors.setdefault(element, {}).setdefault(str(axis), address)
        elif _slot(record, "direction") == "write" and attribute in CALIBRATED_ATTRIBUTES:
            for field in SUPPLY_IDENTITY:
                declare(address, field, element)
    for element, axes in monitors.items():
        for address in axes.values():
            for field in READOUT_IDENTITY:
                declare(address, field, element)
        declare(axes.get("x") or next(iter(axes.values())), ROLL, element)
    for element, polynomial in magnets.items():
        if element in addresses or element in monitors:
            continue
        for field, bounds in MISALIGNMENT_BOUNDS.items():
            slots[f"{element}{FAULT_SEPARATOR}{field}"] = FaultSlot(
                element, field, bounds, element=element, scope=ELEMENT_SCOPE
            )
        if not polynomial:
            continue
        for prefix in MULTIPOLE_PREFIXES:
            for index in MULTIPOLE_INDICES:
                field = f"{prefix}{index}"
                slots[f"{element}{FAULT_SEPARATOR}{field}"] = FaultSlot(
                    element, field, MULTIPOLE_BOUNDS, element=element, scope=ELEMENT_SCOPE
                )
    return dict(sorted(slots.items()))


@dataclass(frozen=True)
class _Plane:
    """One plane of a monitor: its address and its conversion to the published unit.

    Attributes:
        address: The reading's address.
        calibration: Published unit to the deck's unit; ``None`` when they
            are the same.
        inverse: The deck's unit back to the published one, or ``None``
            where the record states none.
    """

    address: str
    calibration: Calibration | None
    inverse: Calibration | None


_MONITORS: weakref.WeakKeyDictionary[Any, dict[str, dict[str, _Plane]]] = (
    weakref.WeakKeyDictionary()
)
""":func:`_monitors` per model. A model's variable set is fixed once it is
built, so the map is too; a rebuilt model is a new key."""


def _monitors(model: Any) -> dict[str, dict[str, _Plane]]:
    """Each monitor element's planes, axis -> plane, from the model's variables."""
    cached = _MONITORS.get(model)
    if cached is not None:
        return cached
    monitors: dict[str, dict[str, _Plane]] = {}
    for address, variable in model.supported_variables.items():
        axis = getattr(variable, "axis", None)
        element = getattr(variable, "element_name", None)
        if not getattr(variable, "read_only", False) or axis not in _AXES or element is None:
            continue
        monitors.setdefault(str(element), {})[axis] = _Plane(
            address,
            getattr(variable, "calibration", None),
            getattr(variable, "inverse", None),
        )
    _MONITORS[model] = monitors
    return monitors


def _readout_faults(element: Any) -> dict[str, float]:
    """:func:`bpm_read`'s fault keywords and the roll as ``element`` holds them, identity where absent."""
    faults = {
        f"{field}_{axis}": float(getattr(element, f"readout_{field}_{axis}", identity))
        for field, identity in READOUT_IDENTITY.items()
        for axis in _AXES
    }
    faults[ROLL] = float(getattr(element, f"readout_{ROLL}", ROLL_IDENTITY))
    return faults


def _normal(address: str | None, width: float, t_ms: int | float) -> float:
    """The standard normal draw of ``address``'s noise at ``t_ms``.

    0 where there is no address or no noise to scale, which leaves the
    reading exact.
    """
    if address is None or width == 0.0:
        return 0.0
    import numpy as np

    from osprey_connectors.simulation.series import channel_key_bytes, keyed_normals

    key = channel_key_bytes(address) + NOISE_SUBKEY
    return float(keyed_normals(key, np.asarray([t_ms]))[0])


def readout(model: Any, values: Mapping[str, float], t_ms: int | float) -> dict[str, float]:
    """What each monitor reports for the positions in ``values``.

    Every monitor element with a reading in ``values`` is read once, on its
    (x, y) pair: the roll turns the pair in the deck's unit, through each
    plane's own conversion, and :func:`bpm_read` then applies the element's
    other readout attributes, no calibration error, and noise drawn per
    address at ``t_ms``, to the published pair. A monitor with one reading
    only reads its other plane as exactly on axis. An unrolled monitor never
    leaves its published unit.

    Args:
        model: The model the readings belong to: its ``supported_variables``
            say which addresses are a monitor's (a read-only variable with an
            ``element_name`` and an ``axis``), and its ``simulator`` holds the
            elements the faults are read from.
        values: Positions by address, in the unit each monitor publishes --
            the solved truth plus any beam motion.
        t_ms: The time of the reading, in epoch milliseconds; the noise is a
            pure function of it.

    Returns:
        One reading per address in ``values``; an address that is no monitor's
        passes through unchanged.

    Raises:
        ValueError: ``values`` holds one reading of a monitor that has two;
            a roll mixes the planes, so the other cannot be guessed.
    """
    read = dict(values)
    for element_name, planes in sorted(_monitors(model).items()):
        addresses = sorted(plane.address for plane in planes.values())
        missing = [address for address in addresses if address not in values]
        if len(missing) == len(planes):
            continue
        if missing:
            raise ValueError(
                f"monitor {element_name!r} is read on {addresses}; "
                f"values hold no reading of {missing}"
            )
        faults = _readout_faults(model.simulator.element(element_name))
        roll = faults.pop(ROLL)
        published = {axis: float(values[plane.address]) for axis, plane in planes.items()}
        if roll != ROLL_IDENTITY:
            physics = {
                axis: to_physics(plane.calibration, published[axis])
                for axis, plane in planes.items()
            }
            turned = dict(
                zip(_AXES, rolled(physics.get("x", 0.0), physics.get("y", 0.0), roll), strict=True)
            )
            published = {
                axis: to_hardware(plane.calibration, plane.inverse, turned[axis])
                for axis, plane in planes.items()
            }
        x_address = planes["x"].address if "x" in planes else None
        y_address = planes["y"].address if "y" in planes else None
        reading_x, reading_y = bpm_read(
            published.get("x", 0.0),
            published.get("y", 0.0),
            **faults,
            cal_x=0.0,
            cal_y=0.0,
            normal_x=_normal(x_address, faults["noise_x"], t_ms),
            normal_y=_normal(y_address, faults["noise_y"], t_ms),
        )
        if x_address is not None:
            read[x_address] = reading_x
        if y_address is not None:
            read[y_address] = reading_y
    return read


__all__ = [
    "ELEMENT_SCOPE",
    "MISALIGNMENT_BOUNDS",
    "MULTIPOLE_INDICES",
    "READOUT_IDENTITY",
    "SUPPLY_IDENTITY",
    "bpm_read",
    "magnet_cal",
    "multipole_attribute",
    "multipole_error",
    "multipole_field",
    "readout",
    "rolled",
    "supply_attribute",
    "supply_calibration",
]
