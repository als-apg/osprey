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
``noise`` and ``polarity``, and one ``readout_roll``. Beside them it carries
the electronics' own attributes once, for both axes
(:data:`READOUT_ATTRIBUTE_IDENTITY`): the noise at unit sum, a tone, a
signal gain and a sum scale, which a facility binds its own settings to. A magnet setpoint's
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
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.simulation.engines.calibration import Linear, to_hardware, to_physics
from osprey.simulation.engines.calibration import field as _slot

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

#: The readout attributes a monitor element carries once, for both axes, each
#: at its identity -- what an element that does not carry one reads:
#:
#: * ``readout_noise_scale`` -- the electronics' position noise at unit sum, in
#:   the deck's unit; a sample's sigma is it over the larger of the sample's
#:   sum and :data:`SUM_FLOOR`.
#: * ``readout_tone_amplitude`` -- a tone the electronics add, in the sum's
#:   unit: it raises every sum sample and spikes the positions every
#:   :data:`TONE_SPIKE_PERIOD` samples.
#: * ``readout_gain`` -- the signal gain ahead of the electronics, on the sum.
#: * ``readout_sum_scale`` -- the scale the sum is reported at.
#:
#: None of them is a per-axis fault: a position is a ratio of signals, so the
#: gain and the scale reach the sum only.
READOUT_ATTRIBUTE_IDENTITY: dict[str, float] = {
    "readout_noise_scale": 0.0,
    "readout_tone_amplitude": 0.0,
    "readout_gain": 1.0,
    "readout_sum_scale": 1.0,
}

#: The sum a monitor reads with no beam, in the sum's unit (the share of a
#: shot's charge): the floor of every sum sample, and the sum a position's
#: noise is divided by when less beam than it passed. No sample is ever read
#: noisier than ``readout_noise_scale / SUM_FLOOR``.
SUM_FLOOR = 0.03

#: The samples between two of a tone's position spikes; a spike falls on
#: every series index that is a multiple of it.
TONE_SPIKE_PERIOD = 10

#: The sigma of a tone's position spike per unit of tone amplitude, in metres.
TONE_SPIKE_M = 2.0e-3

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

#: A polarity is a direction: it lands exactly on one of these two values.
POLARITY_OPTIONS: tuple[float, float] = (-1.0, 1.0)

#: The subkey that makes a reading's noise stream its own.
NOISE_SUBKEY = b":readout_noise"

#: The subkey of a turn series' noise, which the shot count is appended to: a
#: stream per series and shot, apart from the address's per-read one.
SHOT_NOISE_SUBKEY = b":shot_noise:"

#: The subkey of a turn series' electronic noise, which the shot count is
#: appended to: a stream of its own beside the noise fault's.
ELECTRONIC_NOISE_SUBKEY = b":electronic_noise:"

#: The subkey of a turn series' tone spikes, which the shot count is appended to.
TONE_SUBKEY = b":tone:"

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


@dataclass(frozen=True)
class FaultSlot:
    """One fault a model declares: the wired address it perturbs and the field.

    A model-wide fault perturbs no one address: its ``address`` is ``None``
    and its ``field`` is the fault's whole name.

    Attributes:
        address: The wired address, or ``None`` for a model-wide fault.
        field: The fault field, such as ``offset`` or ``cal_factor``.
        value_range: The inclusive ``(min, max)`` a value must lie in, or
            ``None`` where only well-formedness bounds it.
        options: The exact values a value must equal one of, or ``None``
            where the field is not an enumeration.
        element: The element the address is wired to (its first slice's);
            ``None`` for a model-wide fault.
    """

    address: str | None
    field: str
    value_range: tuple[float, float] | None
    options: tuple[float, ...] | None = None
    element: str | None = None


#: The model-wide fault that seeds every misalignment draw: a whole number,
#: 0 when a scenario states none.
MISALIGNMENT_SEED = "misalignment_seed"
MAX_MISALIGNMENT_SEED = float(2**53)

#: The magnet classes a misalignment sigma selects, each by its word.
MISALIGNED_MAGNETS: tuple[str, ...] = ("quadrupole", "sextupole", "dipole")

#: What a misaligned magnet is moved in: ``dx`` and ``dy`` in metres, ``roll``
#: in radians about the beam axis.
MISALIGNMENT_FIELDS: tuple[str, ...] = ("dx", "dy", "roll")

#: The aperture shapes a model-wide fault states, each by its two half-axes.
APERTURE_SHAPES: tuple[str, ...] = ("elliptical", "rectangular")

#: The smallest aperture half-axis, in metres: an aperture is never closed.
MIN_APERTURE_HALF_AXIS_M = 1.0e-6


def misalignment_sigma(magnet: str, field: str) -> str:
    """The model-wide fault naming one magnet class's sigma in one field.

    Args:
        magnet: One of :data:`MISALIGNED_MAGNETS`.
        field: One of :data:`MISALIGNMENT_FIELDS`.

    Returns:
        ``<magnet>_<field>_sigma``.
    """
    return f"{magnet}_{field}_sigma"


def aperture_half_axis(shape: str, axis: str) -> str:
    """The model-wide fault naming one half-axis of one aperture shape, in metres.

    Args:
        shape: One of :data:`APERTURE_SHAPES`.
        axis: ``x`` or ``y``.

    Returns:
        ``aperture_<shape>_<axis>``.
    """
    return f"aperture_{shape}_{axis}"


#: The coordinates of a tracked particle, in the order pyAT tracks them.
LAUNCH_COORDINATES: tuple[str, ...] = ("x", "px", "y", "py", "delta", "ct")


def launch_coordinate(coordinate: str) -> str:
    """The model-wide fault naming one coordinate of the launch.

    Args:
        coordinate: One of :data:`LAUNCH_COORDINATES`.

    Returns:
        ``launch_<coordinate>``.
    """
    return f"launch_{coordinate}"


#: The model-wide faults that launch a shot, one per coordinate in pyAT's
#: order: the launch is their vector, each 0 when a scenario states none.
LAUNCH_FAULTS: tuple[str, ...] = tuple(launch_coordinate(c) for c in LAUNCH_COORDINATES)

#: The model-wide faults that shape a bunch: the horizontal and vertical
#: emittances in metre-radians, the relative energy spread and the bunch
#: length in metres.
BUNCH_SHAPE_FAULTS: tuple[str, ...] = (
    "bunch_emittance_x",
    "bunch_emittance_y",
    "bunch_sigma_delta",
    "bunch_sigma_z",
)

#: The model-wide faults that jitter a bunch's centroid from shot to shot: the
#: sigma of the horizontal and the vertical offset, in metres.
CENTROID_JITTER_FAULTS: tuple[str, ...] = ("centroid_jitter_x", "centroid_jitter_y")

#: The model-wide fault holding the charge already stored in the model.
STORED_CHARGE = "stored_charge"

#: The model-wide fault holding the charge one shot injects, in the unit of
#: :data:`STORED_CHARGE`.
BUNCH_CHARGE = "bunch_charge"

#: The model-wide fault counting the particles a shot tracks: a whole number of
#: one or more. A scenario that states it declares a bunch; without it a shot
#: tracks one particle.
BUNCH_PARTICLES = "bunch_particles"
MAX_BUNCH_PARTICLES = float(2**53)

#: The subkey of a bunch's per-shot draw, between the model's key and the shot.
BUNCH_SUBKEY = b":bunch"


def bunch_key(model: str, shot: int) -> bytes:
    """The key of one shot's bunch draw on one model.

    Every random number a bunch takes is a ``keyed_normals`` draw under this
    key, so it is a pure function of the model, the shot and the counter.

    Args:
        model: The model's id.
        shot: The count of the shot.

    Returns:
        ``channel_key_bytes(model) + b":bunch" + <shot>``.
    """
    from osprey_connectors.simulation.series import channel_key_bytes

    return channel_key_bytes(model) + BUNCH_SUBKEY + str(int(shot)).encode("ascii")


def _model_wide() -> dict[str, FaultSlot]:
    bounds: dict[str, tuple[float, float]] = {MISALIGNMENT_SEED: (0.0, MAX_MISALIGNMENT_SEED)}
    for magnet in MISALIGNED_MAGNETS:
        for field in MISALIGNMENT_FIELDS:
            bounds[misalignment_sigma(magnet, field)] = (0.0, math.inf)
    for shape in APERTURE_SHAPES:
        for axis in ("x", "y"):
            bounds[aperture_half_axis(shape, axis)] = (MIN_APERTURE_HALF_AXIS_M, math.inf)
    for name in LAUNCH_FAULTS:
        bounds[name] = (-math.inf, math.inf)
    for name in (*BUNCH_SHAPE_FAULTS, *CENTROID_JITTER_FAULTS, STORED_CHARGE, BUNCH_CHARGE):
        bounds[name] = (0.0, math.inf)
    bounds[BUNCH_PARTICLES] = (1.0, MAX_BUNCH_PARTICLES)
    return {name: FaultSlot(None, name, bounds[name]) for name in sorted(bounds)}


#: The model-wide faults whose value is a whole number.
WHOLE_FAULTS: frozenset[str] = frozenset({MISALIGNMENT_SEED, BUNCH_PARTICLES})

#: Every model-wide fault, by name: the misalignment seed, a sigma per magnet
#: class and field, the two half-axes of each aperture shape, the launch, the
#: bunch's shape, centroid jitter and particle count, and the stored and
#: injected charges. None is a variable of a built model: a model is built at
#: them and holds them for its whole lifetime.
MODEL_WIDE_FAULTS: dict[str, FaultSlot] = _model_wide()


def _first_element(record: Any) -> str | None:
    slices = _slot(record, "slices")
    element = _slot(slices[0], "element") if slices else _slot(record, "element")
    return None if element is None else str(element)


def is_series_record(record: Any) -> bool:
    """Whether a wiring record's engine block carries ``series``.

    Such a record is a buffer of one reading per tracked sample, never an
    orbit monitor, though it names a monitor element and an ``axis``.
    """
    return carries_series(_slot(record, "engine"))


def carries_series(holder: Any) -> bool:
    """Whether an engine block, or a variable built from one, carries ``series``."""
    return _slot(holder, "series") is not None


def fault_variables(wiring: Iterable[Any]) -> dict[str, FaultSlot]:
    """Every fault a model built from ``wiring`` declares, by fault name.

    A fault name is ``<address>/<field>``. A monitor readback (an engine
    ``axis`` and no ``attribute``) carries ``offset``, ``gain``, ``noise`` and
    ``polarity`` on its own axis, the first readback of each element and axis
    only; each monitor element carries one ``roll``, on its x-axis readback or
    else its first one. A setpoint (``direction: write``) on ``PolynomB`` or
    ``KickAngle`` carries ``cal_factor`` and ``cal_offset``. A record whose
    engine block carries ``series`` declares none and is no monitor readback.

    Every model declares the model-wide faults of :data:`MODEL_WIDE_FAULTS`
    too, whatever its wiring: each is named flat, with no address and no
    separator. Nothing is imported and no deck is read.

    Args:
        wiring: The model's wiring records, each with its ``direction``.

    Returns:
        Fault name -> its slot, sorted by name.
    """
    slots: dict[str, FaultSlot] = dict(MODEL_WIDE_FAULTS)
    monitors: dict[str, dict[str, str]] = {}

    def declare(address: str, field: str, element: str) -> None:
        if field == "polarity":
            slot = FaultSlot(address, field, None, POLARITY_OPTIONS, element)
        else:
            slot = FaultSlot(address, field, FAULT_BOUNDS.get(field), element=element)
        slots[f"{address}{FAULT_SEPARATOR}{field}"] = slot

    for record in wiring:
        address = str(_slot(record, "address"))
        block = _slot(record, "engine")
        attribute = _slot(block, "attribute")
        axis = _slot(block, "axis")
        element = _first_element(record)
        if element is None or is_series_record(record):
            continue
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
    """Each monitor element's planes, axis -> plane, from the model's variables.

    A variable carrying ``series`` is a buffer of tracked samples and no plane
    of a monitor.
    """
    cached = _MONITORS.get(model)
    if cached is not None:
        return cached
    monitors: dict[str, dict[str, _Plane]] = {}
    for address, variable in model.supported_variables.items():
        axis = getattr(variable, "axis", None)
        element = getattr(variable, "element_name", None)
        if carries_series(variable):
            continue
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


def samples_to_hardware(
    calibration: Calibration | None, inverse: Calibration | None, physics: Any
) -> Any:
    """Finite samples in the deck's unit, mapped to hardware units sample by sample.

    The vector form of :func:`~osprey.simulation.engines.calibration.to_hardware`:
    a linear calibration with no stated inverse is undone in one array
    operation.

    Args:
        calibration: Hardware to physics; ``None`` for no calibration.
        inverse: Physics back to hardware, or ``None`` where none is stated.
        physics: The samples, in the deck's unit.

    Returns:
        A float64 array the length of ``physics``.
    """
    import numpy as np

    if calibration is None and inverse is None:
        return np.asarray(physics, dtype=np.float64)
    if inverse is None and isinstance(calibration, Linear) and calibration.gain != 0.0:
        return (physics - calibration.offset) / calibration.gain
    return np.array(
        [to_hardware(calibration, inverse, float(value)) for value in physics], dtype=np.float64
    )


def _samples_to_published(plane: _Plane | None, physics: Any) -> Any:
    """Finite samples in the deck's unit, in the unit ``plane`` publishes."""
    if plane is None:
        return physics
    return samples_to_hardware(plane.calibration, plane.inverse, physics)


def _samples_to_physics(plane: _Plane | None, published: Any) -> Any:
    """Finite samples in the unit ``plane`` publishes, in the deck's unit."""
    import numpy as np

    if plane is None or plane.calibration is None:
        return published
    if isinstance(plane.calibration, Linear):
        return published * plane.calibration.gain + plane.calibration.offset
    return np.array(
        [to_physics(plane.calibration, float(value)) for value in published], dtype=np.float64
    )


def _readout_attributes(element: Any) -> dict[str, float]:
    """The readout attributes ``element`` holds, identity where absent."""
    return {
        name: float(getattr(element, name, identity))
        for name, identity in READOUT_ATTRIBUTE_IDENTITY.items()
    }


def _shot_draws(address: str, subkey: bytes, shot: int, indexes: Any) -> Any:
    """The standard normals of one series' ``subkey`` stream for ``shot``, at ``indexes``."""
    from osprey_connectors.simulation.series import channel_key_bytes, keyed_normals

    key = channel_key_bytes(address) + subkey + str(int(shot)).encode("ascii")
    return keyed_normals(key, indexes)


def _beam_sums(seen: Any, sums: Any) -> Any:
    """The physics sum per sample: ``sums`` as given, else 1 where seen and 0 elsewhere."""
    import numpy as np

    if sums is None:
        return seen.astype(np.float64)
    return np.asarray(sums, dtype=np.float64)


def shot_read(
    model: Any,
    monitor: str,
    axis: str,
    x: Any,
    y: Any,
    *,
    address: str,
    shot: int,
    first: int,
    sums: Any = None,
) -> Any:
    """What one monitor reports, on one axis, of the positions a shot tracked at it.

    The chain is :func:`readout`'s, one sample at a time: the roll turns the
    tracked pair in the deck's unit, and the element's other readout
    attributes then act on the sample in the unit the monitor's orbit reading
    of that axis publishes, the unit they are stated in. The result is handed
    back in the deck's unit, so the same physical fault reaches a series
    whatever unit its own record publishes. An axis with no orbit reading
    takes its attributes in the deck's unit.

    The electronics act on the turned position, in the deck's unit, before
    the faults do: a noise of sigma ``readout_noise_scale / max(sum,
    SUM_FLOOR)`` per sample, and a tone's spike every
    :data:`TONE_SPIKE_PERIOD` samples. The noise fault composes on top, from
    its own stream.

    A sample that is NaN on either tracked axis carries no beam: it reads as
    a beam on axis, at the floor's sigma, so no sample is NaN. The noise and
    the spikes are pure functions of ``address``, ``shot`` and the sample's
    index, never of a time: a series' samples are drawn once, when its shot
    is tracked. A monitor that carries no fault and no electronics reads the
    tracked axis untouched, and 0 where no beam is.

    Args:
        model: The model the monitor belongs to, as :func:`readout` takes it.
        monitor: The monitor element's name.
        axis: ``x`` or ``y``.
        x: The tracked horizontal positions, one per sample, in the deck's unit.
        y: The tracked vertical positions, sample for sample with ``x``.
        address: The series' address, which keys its noise.
        shot: The count of the shot that tracked the samples.
        first: The index, in the series, of the first sample given.
        sums: The physics sum per sample -- the share of the shot's charge
            the monitor sees -- or ``None`` for 1 where both axes are a
            number and 0 where either is NaN.

    Returns:
        The samples of ``axis`` as the monitor reports them, in the deck's
        unit: a float64 array the length of ``x``, with no NaN.
    """
    import numpy as np

    element = model.simulator.element(monitor)
    faults = _readout_faults(element)
    electronics = _readout_attributes(element)
    roll = faults.pop(ROLL)
    mine = {field: faults[f"{field}_{axis}"] for field in READOUT_IDENTITY}
    tracked_x = np.asarray(x, dtype=np.float64)
    tracked_y = np.asarray(y, dtype=np.float64)
    seen = ~(np.isnan(tracked_x) | np.isnan(tracked_y))
    pair: tuple[Any, Any] = (np.where(seen, tracked_x, 0.0), np.where(seen, tracked_y, 0.0))
    position: Any = np.array(pair[_AXES.index(axis)], dtype=np.float64)
    sigma_e = electronics["readout_noise_scale"]
    tone = electronics["readout_tone_amplitude"]
    if roll == ROLL_IDENTITY and mine == READOUT_IDENTITY and sigma_e == 0.0 and tone == 0.0:
        return position
    if roll != ROLL_IDENTITY:
        position = rolled(pair[0], pair[1], roll)[_AXES.index(axis)]
    indexes = first + np.arange(position.shape[0])
    if sigma_e != 0.0:
        sigma = sigma_e / np.maximum(_beam_sums(seen, sums), SUM_FLOOR)
        position = position + sigma * _shot_draws(address, ELECTRONIC_NOISE_SUBKEY, shot, indexes)
    if tone != 0.0:
        on = indexes % TONE_SPIKE_PERIOD == 0
        spikes = tone * TONE_SPIKE_M * _shot_draws(address, TONE_SUBKEY, shot, indexes)
        position = position + np.where(on, spikes, 0.0)
    if roll == ROLL_IDENTITY and mine == READOUT_IDENTITY:
        return np.asarray(position, dtype=np.float64)
    plane = _monitors(model).get(monitor, {}).get(axis)
    published = _samples_to_published(plane, position)
    noise: Any = 0.0
    if mine["noise"] != 0.0:
        noise = mine["noise"] * _shot_draws(address, SHOT_NOISE_SUBKEY, shot, indexes)
    # bpm_read's order: offset, polarity, noise, then the gain.
    reading = ((published - mine["offset"]) * mine["polarity"] + noise) * mine["gain"]
    return np.asarray(_samples_to_physics(plane, reading), dtype=np.float64)


def shot_sum(model: Any, monitor: str, sums: Any) -> Any:
    """What one monitor reports as the sum of a shot, sample by sample.

    The physics sum, floored at :data:`SUM_FLOOR` -- the no-beam reading --
    raised by the element's tone, then multiplied by its signal gain and its
    sum scale. No readout fault reaches the sum. At identity a sample the
    whole shot passed reads exactly 1.

    Args:
        model: The model the monitor belongs to, as :func:`readout` takes it.
        monitor: The monitor element's name.
        sums: The physics sum per sample: the share of the shot's charge the
            monitor sees, 0 with no beam.

    Returns:
        A float64 array the length of ``sums``, every sample at least the
        floor's reading.
    """
    import numpy as np

    electronics = _readout_attributes(model.simulator.element(monitor))
    floored = np.maximum(np.asarray(sums, dtype=np.float64), SUM_FLOOR)
    raised = floored + electronics["readout_tone_amplitude"]
    return raised * electronics["readout_gain"] * electronics["readout_sum_scale"]


__all__ = [
    "BUNCH_CHARGE",
    "BUNCH_PARTICLES",
    "BUNCH_SHAPE_FAULTS",
    "BUNCH_SUBKEY",
    "CENTROID_JITTER_FAULTS",
    "ELECTRONIC_NOISE_SUBKEY",
    "LAUNCH_COORDINATES",
    "LAUNCH_FAULTS",
    "MAX_BUNCH_PARTICLES",
    "READOUT_ATTRIBUTE_IDENTITY",
    "READOUT_IDENTITY",
    "STORED_CHARGE",
    "SUM_FLOOR",
    "SUPPLY_IDENTITY",
    "TONE_SPIKE_M",
    "TONE_SPIKE_PERIOD",
    "TONE_SUBKEY",
    "WHOLE_FAULTS",
    "bpm_read",
    "bunch_key",
    "carries_series",
    "is_series_record",
    "launch_coordinate",
    "magnet_cal",
    "readout",
    "rolled",
    "samples_to_hardware",
    "shot_read",
    "shot_sum",
    "supply_attribute",
    "supply_calibration",
]
