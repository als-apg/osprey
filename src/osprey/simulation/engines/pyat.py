"""The pyat engine's plug-in contract.

These functions read a deck -- a lattice file pyAT loads with
``at.load_lattice`` -- and a model's settings and wiring:

* :func:`locate` gives an element's entrance and length along the lattice;
* :func:`prepare` checks the deck against the model's ``pyat`` settings block
  and normalises that block once;
* :func:`start_values` derives each wired channel's operating point from the
  deck;
* :func:`plane` says which transverse plane a wiring record drives;
* :func:`build` builds the LUME model that serves the wiring;
* :func:`error_text` gives the one line a failed build or solve reports;
* :func:`readout` turns solved monitor positions into what each monitor
  reports.

A wiring record is read by key or by attribute: ``id``, ``address``,
``element`` or ``slices`` (each ``element``, ``weight``), the ``engine`` block
in pyAT's words (``attribute``, ``index``, ``axis``) and ``calibration``
(``curve``, ``inverse``). Every refusal is a ``FacilityBuildError`` of kind
``engine-invalid``. A refusal of an element the deck does not hold exactly
once is an :class:`ElementStop`, whose ``element`` and ``count`` let the build
name the wiring record that asked for it. pyAT, numpy and the LUME packages
are imported on first use.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from osprey.facility.errors import FacilityBuildError
from osprey.simulation.engines.calibration import NoInverse, curve_from_record, field, to_hardware
from osprey.simulation.engines.pyat_faults import readout

if TYPE_CHECKING:
    import numpy as np
    from lume.model import LUMEModel

__all__ = [
    "ElementStop",
    "Prepared",
    "build",
    "error_text",
    "locate",
    "plane",
    "prepare",
    "readout",
    "start_values",
]

#: The keys the ``pyat`` settings block may carry.
SETTINGS_KEYS: frozenset[str] = frozenset({"solve", "twiss_in", "rest_mass_gev"})

#: The ``solve`` values, the first one the default.
SOLVES: tuple[str, ...] = ("periodic", "single_pass")

#: Each ``twiss_in`` key and the lengths it may take; the first length is
#: the one ``prepare`` normalises to.
TWISS_LENGTHS: dict[str, tuple[int, ...]] = {
    "beta": (2,),
    "alpha": (2,),
    "dispersion": (4,),
    "closed_orbit": (6, 4),
}

#: The ``twiss_in`` keys pyAT cannot default.
TWISS_REQUIRED: tuple[str, ...] = ("beta", "alpha")

#: The engine-block attributes that name a property of the whole deck rather
#: than of an element, each with its reader: ``energy`` is the deck energy in
#: electron-volts, pyAT's unit.
DECK_PROPERTIES: dict[str, Callable[[Any], float]] = {
    "energy": lambda lattice: float(lattice.energy),
}

#: The engine-block attributes a record naming no element reads off the
#: solved deck, each with the optics output it reads one plane of.
OPTICS_ATTRIBUTES: dict[str, str] = {"tune": "tunes", "chromaticity": "chromaticity"}

#: The ``axis`` words an optics record may name, each with the plane it reads.
OPTICS_AXES: dict[str, int] = {"x": 0, "y": 1}

Deck = str | os.PathLike[str]

_NOTHING_ACTIVE: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class Prepared:
    """A model's ``pyat`` settings block, checked against its deck.

    Attributes:
        solve: ``periodic`` or ``single_pass``.
        twiss_in: pyAT's initial conditions, every value a float ndarray and
            ``closed_orbit`` six long (dp = 0 when four were given); ``None``
            when the block states none.
        rest_mass_gev: The particle's rest energy: ``rest_mass_gev`` when the
            block states it, else the deck's particle.
        length_m: The end of the deck's s axis, the axis ``locate`` reports
            on: one pass through the elements as written, never multiplied
            by the deck's periodicity.
    """

    solve: str
    twiss_in: dict[str, np.ndarray] | None
    rest_mass_gev: float
    length_m: float


@dataclass(frozen=True)
class _Loaded:
    lattice: Any
    indices: dict[str, tuple[int, ...]]
    s_pos: tuple[float, ...]
    length_m: float


def _load(deck: Deck) -> _Loaded:
    path = Path(deck).resolve()
    stat = path.stat()
    return _load_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _load_cached(path: str, mtime_ns: int, size: int) -> _Loaded:
    """Load a deck once per (path, modification time, size) triple."""
    del mtime_ns, size
    import at

    lattice = at.load_lattice(path)
    indices: dict[str, list[int]] = {}
    for index, element in enumerate(lattice):
        indices.setdefault(element.FamName, []).append(index)
    s_pos = lattice.get_s_pos(range(len(lattice)))
    return _Loaded(
        lattice=lattice,
        indices={name: tuple(found) for name, found in indices.items()},
        s_pos=tuple(float(s) for s in s_pos),
        length_m=float(lattice.get_s_pos(len(lattice))[0]),
    )


class ElementStop(FacilityBuildError):
    """An element the deck does not hold exactly once.

    Attributes:
        element: The element's name.
        count: How often the deck holds it: 0 when absent, above 1 when
            repeated.
    """

    def __init__(
        self, deck: Deck, element: str, count: int, record_id: str, record_kind: str
    ) -> None:
        if count == 0:
            detail = f"element {element} is not in the deck"
            remedy = "name an element the deck holds"
        else:
            detail = f"element {element} appears {count} times in the deck"
            remedy = "give the element a unique name in the deck"
        super().__init__(
            "engine-invalid",
            record_id,
            [str(deck)],
            remedy,
            record_kind=record_kind,
            detail=detail,
        )
        self.element = element
        self.count = count


def _model_id(deck: Deck, model: str | None) -> str:
    return model if model is not None else Path(deck).stem


def _stop(
    deck: Deck, record_id: str, record_kind: str, detail: str, remedy: str
) -> FacilityBuildError:
    return FacilityBuildError(
        "engine-invalid", record_id, [str(deck)], remedy, record_kind=record_kind, detail=detail
    )


def _element_index(
    loaded: _Loaded, deck: Deck, element: str, record_id: str, record_kind: str
) -> int:
    found = loaded.indices.get(element, ())
    if len(found) != 1:
        raise ElementStop(deck, element, len(found), record_id, record_kind)
    return found[0]


def locate(deck: Deck, element: str, *, model: str | None = None) -> tuple[float, float]:
    """Return an element's entrance and length along the deck.

    Args:
        deck: The lattice file.
        element: The element's name (``FamName``), unique in the deck.
        model: The model name the stop names; the deck's file stem when
            omitted.

    Returns:
        ``(s_entrance_m, length_m)``.

    Raises:
        ElementStop: ``engine-invalid`` when the element is absent from the
            deck or named more than once.
    """
    loaded = _load(deck)
    index = _element_index(loaded, deck, element, _model_id(deck, model), "model")
    return loaded.s_pos[index], float(loaded.lattice[index].Length)


def _normalise_twiss(twiss: Any, deck: Deck, model_id: str) -> dict[str, np.ndarray]:
    import numpy as np

    if not isinstance(twiss, Mapping):
        raise _stop(
            deck,
            model_id,
            "model",
            "settings key pyat.twiss_in is not a map",
            "state twiss_in as a map of beta, alpha, dispersion, closed_orbit",
        )
    unknown = sorted(set(twiss) - set(TWISS_LENGTHS))
    if unknown:
        raise _stop(
            deck,
            model_id,
            "model",
            f"settings key pyat.twiss_in carries unknown keys {', '.join(unknown)}",
            "use only beta, alpha, dispersion, closed_orbit",
        )
    missing = [key for key in TWISS_REQUIRED if key not in twiss]
    if missing:
        raise _stop(
            deck,
            model_id,
            "model",
            f"settings key pyat.twiss_in lacks {', '.join(missing)}",
            "state beta and alpha in twiss_in",
        )
    normalised: dict[str, np.ndarray] = {}
    for key, lengths in TWISS_LENGTHS.items():
        raw = twiss.get(key)
        if raw is None:
            normalised[key] = np.zeros(lengths[0])
            continue
        try:
            values = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            values = None
        if values is None or values.size not in lengths or not np.all(np.isfinite(values)):
            wanted = " or ".join(str(n) for n in sorted(lengths))
            raise _stop(
                deck,
                model_id,
                "model",
                f"settings key pyat.twiss_in.{key} must be {wanted} finite numbers",
                f"give twiss_in.{key} {wanted} numbers",
            )
        if values.size < lengths[0]:
            values = np.concatenate([values, np.zeros(lengths[0] - values.size)])
        normalised[key] = values
    return normalised


def prepare(deck: Deck, settings: Any, *, model: str | None = None) -> Prepared:
    """Check a deck against a model's settings and normalise the ``pyat`` block.

    Args:
        deck: The lattice file.
        settings: The model's settings, ``{pyat: {solve?, twiss_in?,
            rest_mass_gev?}}``, or ``None``.
        model: The model name the stop names; the deck's file stem when
            omitted.

    Returns:
        The normalised block and the deck's length.

    Raises:
        FacilityBuildError: ``engine-invalid`` for an unknown settings key or
            ``solve`` value, repeated monitor names, a periodic deck holding a
            cavity without longitudinal motion, ``single_pass`` without
            ``twiss_in``, a ``twiss_in`` value of the wrong length, or a
            ``rest_mass_gev`` that is not a finite non-negative number.
    """
    import at

    model_id = _model_id(deck, model)
    block = field(settings, "pyat")
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise _stop(
            deck,
            model_id,
            "model",
            "settings key pyat is not a map",
            "state the pyat settings as a map",
        )
    unknown = sorted(set(block) - SETTINGS_KEYS)
    if unknown:
        raise _stop(
            deck,
            model_id,
            "model",
            f"settings key pyat carries unknown keys {', '.join(unknown)}",
            "use only solve, twiss_in, rest_mass_gev",
        )
    solve = block.get("solve", SOLVES[0])
    if solve not in SOLVES:
        raise _stop(
            deck,
            model_id,
            "model",
            f"settings key pyat.solve is {solve!r}",
            "set solve to periodic or single_pass",
        )

    loaded = _load(deck)
    lattice = loaded.lattice
    monitor_names: dict[str, int] = {}
    for element in lattice:
        if isinstance(element, at.Monitor):
            monitor_names[element.FamName] = monitor_names.get(element.FamName, 0) + 1
    repeated = sorted(name for name, count in monitor_names.items() if count > 1)
    if repeated:
        raise _stop(
            deck,
            model_id,
            "model",
            f"monitor names repeat in the deck: {', '.join(repeated)}",
            "give every monitor a unique name in the deck",
        )
    if solve == "periodic":
        frozen = sorted(
            {
                element.FamName
                for element in lattice
                if isinstance(element, at.RFCavity) and not element.longt_motion
            }
        )
        if frozen:
            raise _stop(
                deck,
                model_id,
                "model",
                f"cavity {', '.join(frozen)} has no longitudinal motion in a periodic deck",
                "give the cavity a longitudinal pass method such as RFCavityPass",
            )

    twiss = block.get("twiss_in")
    if twiss is None and solve == "single_pass":
        raise _stop(
            deck,
            model_id,
            "model",
            "settings key pyat.twiss_in is required when solve is single_pass",
            "state twiss_in with beta and alpha",
        )
    twiss_in = None if twiss is None else _normalise_twiss(twiss, deck, model_id)

    rest_mass = block.get("rest_mass_gev")
    if rest_mass is None:
        rest_mass_gev = float(lattice.particle.rest_energy) / 1e9
    else:
        if (
            isinstance(rest_mass, bool)
            or not isinstance(rest_mass, int | float)
            or not math.isfinite(rest_mass)
            or rest_mass < 0
        ):
            raise _stop(
                deck,
                model_id,
                "model",
                f"settings key pyat.rest_mass_gev is {rest_mass!r}",
                "give rest_mass_gev as a finite non-negative number",
            )
        rest_mass_gev = float(rest_mass)
    return Prepared(
        solve=solve, twiss_in=twiss_in, rest_mass_gev=rest_mass_gev, length_m=loaded.length_m
    )


def _read_element(loaded: _Loaded, deck: Deck, record: Any, record_id: str) -> float | None:
    slices = field(record, "slices")
    if slices:
        first = slices[0]
        element = field(first, "element")
        weight = field(first, "weight")
        weight = 1.0 if weight is None else float(weight)
    else:
        element = field(record, "element")
        weight = 1.0
    block = field(record, "engine")
    attribute = field(block, "attribute")
    if element is None:
        reader = DECK_PROPERTIES.get(attribute) if isinstance(attribute, str) else None
        return None if reader is None else reader(loaded.lattice)
    index = _element_index(loaded, deck, element, record_id, "wiring")
    if attribute is None:
        raise _stop(
            deck,
            record_id,
            "wiring",
            "the engine block names no attribute to read a start value from",
            "name the element attribute in the engine block",
        )
    raw = getattr(loaded.lattice[index], attribute, None)
    if raw is None:
        raise _stop(
            deck,
            record_id,
            "wiring",
            f"element {element} has no attribute {attribute}",
            "name an attribute the element carries",
        )
    position = field(block, "index")
    try:
        value = float(raw[int(position)]) if position is not None else float(raw)
    except (IndexError, TypeError, ValueError):
        raise _stop(
            deck,
            record_id,
            "wiring",
            f"element {element} attribute {attribute} index {position} does not hold a number",
            "give the index of one number in the attribute",
        ) from None
    return value / weight


#: What a monitor's and a setpoint's way back to hardware units serves, as a
#: stop names it.
_SERVE_READING = "serve the reading"
_DERIVE_START = "derive the start value"


def _hardware(
    physics: float, calibration: Any, deck: Deck, record_id: str, purpose: str = _DERIVE_START
) -> float:
    curve = curve_from_record(field(calibration, "curve"))
    inverse = curve_from_record(field(calibration, "inverse"))
    try:
        return to_hardware(curve, inverse, physics)
    except NoInverse as missing:
        if missing.reason == "table":
            raise _stop(
                deck,
                record_id,
                "wiring",
                f"a table calibration has no inverse to {purpose}",
                "add calibration.inverse",
            ) from None
        raise _stop(
            deck,
            record_id,
            "wiring",
            "the linear calibration's gain is 0, so it has no inverse",
            "give the calibration a non-zero gain",
        ) from None


def start_values(
    deck: Deck,
    wiring: Iterable[Any],
    settings: Any,
    *,
    readbacks: Mapping[str, str | None] | None = None,
    value_types: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Derive each wired channel's operating point from the deck.

    A setpoint's value is its first slice's element attribute (the
    ``element`` when the record has no slices) divided by that slice's
    weight, or, for a record naming no element whose engine attribute is one
    of :data:`DECK_PROPERTIES`, that property of the deck; it is mapped to
    hardware through ``calibration.inverse`` when present, else through the
    algebraic inverse of a ``linear`` curve; with no calibration the physics
    value is the hardware value. A readback takes its paired setpoint's value
    when that setpoint is in ``wiring``, else zero. Any other record that
    names no element has no start value in the deck and is left out.

    Args:
        deck: The lattice file.
        wiring: The model's wiring records.
        settings: The model's settings; pyat's start values read none.
        readbacks: Each readback address among the records, mapped to the
            address of its paired setpoint or ``None``; every other record is
            a setpoint.
        value_types: Each wired channel's ``value_type``; an address it
            does not name is float. pyat drives float channels only.

    Returns:
        ``{address: value}`` sorted by address.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the wiring id for the
            first record, in record order, whose channel is not float (raised
            before the deck is read); for an element absent from or repeated
            in the deck (an ``ElementStop``), an unreadable attribute, or a
            setpoint's or monitor's (a record whose engine block names an
            ``axis``) table calibration without an inverse or linear gain of 0.
    """
    del settings
    readbacks = readbacks or {}
    value_types = value_types or {}
    records = list(wiring)
    for record in records:
        address = field(record, "address")
        value_type = value_types.get(address, "float")
        if value_type != "float":
            raise _stop(
                deck,
                field(record, "id") or address,
                "wiring",
                f"{address} is {value_type}; pyat drives float channels only",
                "wire a float channel, or leave the channel unwired",
            )
    loaded = _load(deck)
    values: dict[str, float] = {}
    pending: list[str] = []
    for record in records:
        address = field(record, "address")
        record_id = field(record, "id") or address
        if field(field(record, "engine"), "axis") is not None:
            _hardware(0.0, field(record, "calibration"), deck, record_id, _SERVE_READING)
        pair = readbacks.get(address)
        if address in readbacks and pair != address:
            pending.append(address)
            continue
        physics = _read_element(loaded, deck, record, record_id)
        if physics is None:
            continue
        values[address] = _hardware(physics, field(record, "calibration"), deck, record_id)
    for address in pending:
        pair = readbacks[address]
        values[address] = values.get(pair, 0.0) if pair is not None else 0.0
    return dict(sorted(values.items()))


def plane(wiring_record: Any) -> Literal["x", "y"] | None:
    """Say which transverse plane a wiring record drives.

    Args:
        wiring_record: One wiring record.

    Returns:
        ``x`` for ``KickAngle`` index 0 or ``PolynomB`` index 0, ``y`` for
        ``KickAngle`` index 1 or ``PolynomA`` index 0, else ``None``.
    """
    block = field(wiring_record, "engine")
    attribute = field(block, "attribute")
    index = field(block, "index")
    if attribute == "KickAngle":
        return {0: "x", 1: "y"}.get(index)  # type: ignore[return-value]
    if index == 0:
        return {"PolynomB": "x", "PolynomA": "y"}.get(attribute)  # type: ignore[return-value]
    return None


def _target(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """What a record drives: its element or slices, and its engine attribute and index."""
    slices = tuple(
        (field(piece, "element"), field(piece, "weight")) for piece in field(record, "slices") or ()
    )
    block = field(record, "engine")
    return (field(record, "element"), slices, field(block, "attribute"), field(block, "index"))


def _names_element(record: Mapping[str, Any]) -> bool:
    return field(record, "element") is not None or bool(field(record, "slices"))


def _optics_source(record: Mapping[str, Any], deck: Deck, planes: int) -> tuple[str, int]:
    """The optics output and plane an element-free record reads.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the record when its
            engine block names no optics attribute, both or neither of
            ``axis`` and ``index``, an unknown axis, or a plane the deck does
            not have.
    """
    block = field(record, "engine")
    attribute = field(block, "attribute")
    axis = field(block, "axis")
    index = field(block, "index")
    record_id = str(field(record, "id") or field(record, "address"))
    output = OPTICS_ATTRIBUTES.get(attribute) if isinstance(attribute, str) else None
    planes_text = ", ".join(str(plane) for plane in range(planes))
    if output is None or (axis is None) == (index is None):
        raise _stop(
            deck,
            record_id,
            "wiring",
            f"the engine block names attribute {attribute!r}, axis {axis!r} and index "
            f"{index!r}; a record naming no element reads tune or chromaticity on one of "
            f"axis or index",
            f"wire attribute tune or chromaticity with axis x or y, or with index {planes_text}",
        )
    if axis is not None:
        if axis not in OPTICS_AXES:
            raise _stop(
                deck,
                record_id,
                "wiring",
                f"the engine block names axis {axis!r}",
                "name axis x or y",
            )
        return output, OPTICS_AXES[axis]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < planes:
        raise _stop(
            deck,
            record_id,
            "wiring",
            f"the engine block names index {index!r}; the deck has tune planes {planes_text}",
            f"name index {planes_text}",
        )
    return output, index


def build(
    model: str,
    wiring: Iterable[Mapping[str, Any]],
    deck: Deck,
    settings: Any,
    active: Mapping[str, Any] = _NOTHING_ACTIVE,
) -> LUMEModel:
    """Build the LUME model that serves one model's wiring over its deck.

    Each wiring entry becomes one variable, named by its address:

    * a setpoint (``direction: write``) is written through its calibration,
      bounded by the entry's ``value_range`` when it carries one; its default
      is the active value when ``active`` names it, else the entry's
      ``default``;
    * a monitor reading (an engine ``axis`` and no ``attribute``) is read off
      the solved orbit;
    * a readback naming an element and an engine ``attribute`` reads back the
      setpoint that drives the same element or slices, attribute and index,
      or its own ``default`` when no setpoint does;
    * a record naming no element reads one plane of the deck's ``tune`` or
      ``chromaticity`` -- its ``axis`` (``x`` 0, ``y`` 1) or its ``index`` --
      unless its attribute is a deck property such as ``energy``.

    The model drives its own copy of the deck, so the deck a later call loads
    is never the one a model has written to. Every other key of ``active``
    is a fault seed, ``<address>/<field>``, written onto its element before the
    model adopts the variables. The model is then reset once -- one batch, one
    solve -- so the lattice holds every setpoint's default, and a later
    :meth:`reset` returns to the active scenario rather than to the deck.

    Args:
        model: The model's name, which a refusal names.
        wiring: The model's wiring entries, as
            :func:`~osprey.facility.views.simulator.simulator_wiring` gives
            them.
        deck: The lattice file.
        settings: The model's settings, ``{pyat: {...}}``, or ``None``.
        active: The active scenario's writes: setpoint address -> hardware
            value, and fault name -> seed.

    Returns:
        The model, at the active scenario.

    Raises:
        FacilityBuildError: ``engine-invalid`` for settings :func:`prepare`
            refuses, or for a record naming no element that reads no plane of
            the tunes or the chromaticity.
        ValueError: an entry cannot be built as a variable, a default or an
            active value lies outside its ``value_range``, or a key of
            ``active`` names no setpoint and no fault of the model.
        pydantic.ValidationError: a fault seed lies outside its bounds.
        OrbitSolveError: the deck, or the deck at the active scenario, has no
            stable solve.
    """
    from osprey.simulation.engines.pyat_model import (
        PyATLatticeModel,
        ReadbackVariable,
        tune_planes,
    )
    from osprey.simulation.engines.pyat_variables import EV_PER_GEV, variable_from_wiring

    prepare(deck, settings, model=model)
    lattice = _load(deck).lattice.deepcopy()
    deck_energy_gev = float(lattice.energy) / EV_PER_GEV
    planes = tune_planes(lattice)
    entries = list(wiring)

    writes = {str(entry["address"]) for entry in entries if entry.get("direction") == "write"}
    setpoints: dict[tuple[Any, ...], str] = {}
    for entry in entries:
        if entry.get("direction") == "write" and _names_element(entry):
            setpoints.setdefault(_target(entry), str(entry["address"]))

    channels: list[Any] = []
    for entry in entries:
        address = str(entry["address"])
        attribute = field(entry.get("engine"), "attribute")
        if not _names_element(entry) and attribute not in DECK_PROPERTIES:
            output, component = _optics_source(entry, deck, planes)
            channels.append(
                ReadbackVariable(
                    name=address, source=output, component=component, unit=entry.get("unit")
                )
            )
            continue
        if entry.get("direction") == "read" and _names_element(entry) and attribute is not None:
            source = setpoints.get(_target(entry))
            channels.append(
                ReadbackVariable(
                    name=address,
                    source=source,
                    default_value=None if source is not None else float(entry["default"]),
                    unit=entry.get("unit"),
                )
            )
            continue
        overrides: dict[str, Any] = {}
        if address in writes:
            if "value_range" in entry:
                overrides["value_range"] = tuple(entry["value_range"])
            if address in active:
                overrides["default_value"] = float(active[address])
        variable = variable_from_wiring(entry, deck_energy_gev=deck_energy_gev, **overrides)
        if variable is not None:
            channels.append(variable)

    faults = {name: float(value) for name, value in active.items() if name not in writes}
    built = PyATLatticeModel(lattice, channels, faults=faults)
    built.reset()
    return built


def error_text(exc: BaseException) -> str:
    """The one line a failed build or solve is reported with.

    Args:
        exc: What the build or the solve raised.

    Returns:
        An ``OrbitSolveError``'s own message, which names the solve that
        failed; any other exception's type name and message. Whitespace runs
        collapse to one space.
    """
    from lume_pyat.exceptions import OrbitSolveError

    text = " ".join(str(exc).split())
    if isinstance(exc, OrbitSolveError) and text:
        return text
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
