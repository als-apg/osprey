"""The pyat engine's plug-in contract.

These functions read a deck -- a lattice file pyAT loads with
``at.load_lattice`` -- and a model's settings and wiring:

* :func:`locate` gives an element's entrance and length along the lattice;
* :func:`prepare` checks the deck against the model's ``pyat`` settings block
  and normalises that block once;
* :func:`start_values` derives each wired channel's operating point from the
  deck;
* :func:`describe` says what a wiring record is: its role, the transverse
  plane it steers or reads, and how often it refreshes;
* :func:`describe_input` checks one input's engine block: a turn-by-turn
  ring takes the ``shot``, ``charge`` and ``stored_current`` signals;
* :func:`build` builds the LUME model that serves the wiring;
* :func:`take_inputs` stores a turn-by-turn ring's delivered inputs, and
  :func:`advance`, :func:`client_write`, :func:`next_deadline` and
  :func:`emitted` run its shots: a client write to an ``arm`` record arms its
  monitor, a rise of the ``shot`` input tracks once for the armed monitors,
  and their armed flags drop on the first advance after the arrays are
  drained. On any other ring they do nothing;
* :func:`response_matrix` steps wired correctors and returns the orbit
  response -- or, on a turn-by-turn ring, the first-turn trajectory
  response -- at the wired monitors, naming each corrector whose step has no
  solve;
* :func:`polynomial_kicks` copies a deck so named kick elements carry their
  kicks as dipole polynomials;
* :func:`error_text` gives the one line a failed build or solve reports;
* :func:`readout` turns solved monitor positions into what each monitor
  reports;
* :func:`fault_variables` names every fault a model built from a wiring and
  its settings declares, without building it.

A wiring record is read by key or by attribute: ``id``, ``address``,
``element`` or ``slices`` (each ``element``, ``weight``), the ``engine`` block
in pyAT's words (``attribute``, ``index``, ``axis``) and ``calibration``
(``curve``, ``inverse``). Every refusal is a ``FacilityBuildError`` of kind
``engine-invalid``; a request with no solve raises one of
:data:`SOLVE_ERRORS`. A refusal of an element the deck does not hold exactly
once is an :class:`ElementStop`, whose ``element`` and ``count`` let the build
name the wiring record that asked for it. pyAT, numpy and the LUME models are
imported on first use; lume-pyat's exceptions, which import nothing, at
import.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from lume_pyat.exceptions import AmbiguousElementError, OrbitSolveError, UnknownElementError

from osprey.facility.errors import FacilityBuildError
from osprey.simulation.engines import pyat_turn_settings as turn_settings
from osprey.simulation.engines.calibration import NoInverse, curve_from_record, field, to_hardware
from osprey.simulation.engines.pyat_faults import (
    ELEMENT_SCOPE,
    ERROR_ELEMENTS_KEY,
    FAULT_SEPARATOR,
    MULTIPOLE_PREFIXES,
    FaultSlot,
    error_elements,
    fault_variables,
    multipole_attribute,
    multipole_field,
    readout,
)

if TYPE_CHECKING:
    import numpy as np
    from lume.model import LUMEModel

__all__ = [
    "ElementStop",
    "Prepared",
    "REFUSAL_ERRORS",
    "RequestRefused",
    "ResponseMatrix",
    "SOLVE_ERRORS",
    "advance",
    "build",
    "client_write",
    "deck_length",
    "element_index",
    "error_text",
    "fault_variables",
    "describe",
    "describe_input",
    "emitted",
    "locate",
    "next_deadline",
    "polynomial_kicks",
    "prepare",
    "readout",
    "response_matrix",
    "start_values",
    "take_inputs",
]

#: The keys the ``pyat`` settings block may carry.
SETTINGS_KEYS: frozenset[str] = frozenset(
    {"solve", "twiss_in", "rest_mass_gev", ERROR_ELEMENTS_KEY, turn_settings.BLOCK_KEY}
)

#: The ``solve`` values, the first one the default.
SOLVES: tuple[str, ...] = ("periodic", "single_pass")

#: The kinds of response :func:`response_matrix` answers, the first one the default.
RESPONSE_KINDS: tuple[str, ...] = ("orbit", "trajectory")

# Stand-in until lume-base states SolveError, the base of every LUME engine's no-solve exception.
#: The exceptions with which this engine says a request has no solve.
SOLVE_ERRORS: tuple[type[Exception], ...] = (OrbitSolveError,)


class RequestRefused(ValueError):
    """A request this engine does not answer: its text names what to change."""


# Stand-in until lume-base states the base of every LUME engine's request refusal.
#: The exceptions with which this engine says a request is not one it answers.
REFUSAL_ERRORS: tuple[type[Exception], ...] = (RequestRefused,)

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

#: The plane each ``index`` 0 and 1 names, for a kick and for an optics output.
_INDEX_PLANES: dict[int, str] = {0: "x", 1: "y"}

#: The kind of device an element attribute drives, for the attributes that
#: are not a strength's polynomial coefficient.
_ATTRIBUTE_KINDS: dict[str, str] = {"KickAngle": "kick", "Frequency": "rf"}

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
        turn_by_turn: The turn-by-turn block, checked and defaulted; ``None``
            on a ring that is not a turn-by-turn ring.
    """

    solve: str
    twiss_in: dict[str, np.ndarray] | None
    rest_mass_gev: float
    length_m: float
    turn_by_turn: turn_settings.TurnSettings | None = None


@dataclass(frozen=True)
class ResponseMatrix:
    """The response of a model's wired monitors to its stepped correctors.

    The closed-orbit (or single-pass) response for the ``orbit`` kind, the
    first-turn trajectory response for the ``trajectory`` kind; rows, columns
    and units are the same for both.

    Attributes:
        rows: The wired monitor readback addresses, ordered by (element
            lattice index, ``x`` before ``y``).
        columns: The correctors whose two arms solved, in the order asked.
        matrix: One row per address of ``rows`` and one column per corrector
            of ``columns``.
        unsolved: Each corrector whose step has no solve, in the order asked,
            mapped to :func:`error_text` of the failure followed by
            ``(while stepping <address>)``.
    """

    rows: list[str]
    columns: list[str]
    matrix: np.ndarray
    unsolved: dict[str, str]


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

    ``named_by`` says where the element is named when no wiring record names
    it, and leads the stop's detail.
    """

    def __init__(
        self,
        deck: Deck,
        element: str,
        count: int,
        record_id: str,
        record_kind: str,
        *,
        named_by: str | None = None,
    ) -> None:
        if count == 0:
            detail = f"element {element} is not in the deck"
            remedy = "name an element the deck holds"
        else:
            detail = f"element {element} appears {count} times in the deck"
            remedy = "give the element a unique name in the deck"
        if named_by is not None:
            detail = f"{named_by}: {detail}"
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


def element_index(deck: Deck, element: str, *, model: str | None = None) -> int:
    """Return an element's index in the deck's lattice, its place in ring order.

    Co-located elements (a monitor and a corrector at one ``s``) differ here
    where :func:`locate` places them together.

    Args:
        deck: The lattice file.
        element: The element's name (``FamName``), unique in the deck.
        model: The model name the stop names; the deck's file stem when
            omitted.

    Returns:
        The element's index, counted from the deck's first element.

    Raises:
        ElementStop: ``engine-invalid`` when the element is absent from the
            deck or named more than once.
    """
    return _element_index(_load(deck), deck, element, _model_id(deck, model), "model")


def deck_length(deck: Deck, *, model: str | None = None) -> float:
    """Return the length of the deck's lattice, in metres: a ring's circumference.

    Args:
        deck: The lattice file.
        model: The model the deck serves; accepted for the plug-in contract's
            shape, no stop names it.
    """
    del model
    return _load(deck).length_m


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
            rest_mass_gev?, error_elements?, turn_by_turn?}}``, or ``None``.
        model: The model name the stop names; the deck's file stem when
            omitted.

    Returns:
        The normalised block and the deck's length.

    Raises:
        FacilityBuildError: ``engine-invalid`` for an unknown settings key or
            ``solve`` value, repeated monitor names, a periodic deck holding a
            cavity without longitudinal motion, ``single_pass`` without
            ``twiss_in``, ``twiss_in`` on a periodic model, a ``twiss_in``
            value of the wrong length, a
            ``rest_mass_gev`` that is not a finite non-negative number, or an
            ``error_elements`` list that is malformed, names an element
            twice, or states ``multipoles`` on an element with no polynomial;
            or a ``turn_by_turn`` block :func:`pyat_turn_settings.normalise`
            refuses.
        ElementStop: an ``error_elements`` item or the ``turn_by_turn``
            injection element names an element the deck does not hold
            exactly once.
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
            f"use only solve, twiss_in, rest_mass_gev, {ERROR_ELEMENTS_KEY}, "
            f"{turn_settings.BLOCK_KEY}",
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
    if twiss is not None and solve == "periodic":
        raise _stop(
            deck,
            model_id,
            "model",
            "settings key pyat.twiss_in is set on a periodic model",
            "remove twiss_in or set solve to single_pass",
        )
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
    _check_error_elements(loaded, deck, settings, model_id)
    return Prepared(
        solve=solve,
        twiss_in=twiss_in,
        rest_mass_gev=rest_mass_gev,
        length_m=loaded.length_m,
        turn_by_turn=_turn_block(loaded, deck, settings, model_id),
    )


def _setting_overrides(entry: Mapping[str, Any], active: Mapping[str, Any]) -> dict[str, Any]:
    """A setting's ``value_range`` and its active scenario's value as its default."""
    overrides: dict[str, Any] = {}
    if "value_range" in entry:
        overrides["value_range"] = tuple(entry["value_range"])
    address = str(entry["address"])
    if address in active:
        overrides["default_value"] = float(active[address])
    return overrides


def _turn_block(
    loaded: _Loaded, deck: Deck, settings: Any, model_id: str
) -> turn_settings.TurnSettings | None:
    """The model's turn-by-turn block, its injection element looked up in the deck."""

    def index_of(element: str) -> int:
        found = loaded.indices.get(element, ())
        if len(found) != 1:
            raise ElementStop(
                deck,
                element,
                len(found),
                model_id,
                "model",
                named_by=f"settings key pyat.{turn_settings.BLOCK_KEY}.injection_element",
            )
        return found[0]

    return turn_settings.normalise(settings, str(deck), model_id, index_of)


def _check_error_elements(loaded: _Loaded, deck: Deck, settings: Any, model_id: str) -> None:
    """Hold the settings' ``error_elements`` list to its shape and to the deck.

    Raises:
        FacilityBuildError: ``engine-invalid`` for a malformed list, an
            element named twice, or ``multipoles`` on an element that carries
            no ``PolynomB`` or ``PolynomA``.
        ElementStop: a declared element is absent from the deck or repeated
            in it.
    """
    key = f"settings key pyat.{ERROR_ELEMENTS_KEY}"
    declared, problems = error_elements(settings)
    if problems:
        raise _stop(
            deck,
            model_id,
            "model",
            problems[0],
            f"state {ERROR_ELEMENTS_KEY} as a list of {{element: <name>, multipoles: "
            "true|false}, each element once",
        )
    for element, multipoles in declared.items():
        found = loaded.indices.get(element, ())
        if len(found) != 1:
            raise ElementStop(deck, element, len(found), model_id, "model", named_by=key)
        if not multipoles:
            continue
        missing = [
            attribute
            for attribute in MULTIPOLE_PREFIXES.values()
            if not hasattr(loaded.lattice[found[0]], attribute)
        ]
        if missing:
            raise _stop(
                deck,
                model_id,
                "model",
                f"{key} states multipoles on element {element}, which carries no "
                f"{' or '.join(missing)}",
                f"drop multipoles from {element}, or name an element with polynomials",
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
    shapes: Mapping[str, Iterable[int]] | None = None,
) -> dict[str, float | list[float]]:
    """Derive each wired channel's operating point from the deck.

    A setpoint's value is its first slice's element attribute (the
    ``element`` when the record has no slices) divided by that slice's
    weight, or, for a record naming no element whose engine attribute is one
    of :data:`DECK_PROPERTIES`, that property of the deck; it is mapped to
    hardware through ``calibration.inverse`` when present, else through the
    algebraic inverse of a ``linear`` curve; with no calibration the physics
    value is the hardware value. A turn-by-turn record (its engine block names
    a ``signal``) starts at its signal's value: a turn array as
    ``trigger_sample + turns`` zeros, ``rf_switch`` at 1 (on), every other
    signal at 0. A readback takes its paired setpoint's value
    when that setpoint is in ``wiring``, else zero; a waveform readback wired
    to a whole optics output starts as one zero per plane. Any other record
    that names no element has no start value in the deck and is left out.

    Args:
        deck: The lattice file.
        wiring: The model's wiring records.
        settings: The model's settings; only the ``turn_by_turn`` block is
            read.
        readbacks: Each readback address among the records, mapped to the
            address of its paired setpoint or ``None``; every other record is
            a setpoint.
        value_types: Each wired channel's ``value_type``; an address it
            does not name is float. pyat drives float channels only, but for
            a readback wired to a whole optics output.
        shapes: Each wired waveform channel's ``shape``; a readback wired to
            a whole optics output is as long as the deck has planes.

    Returns:
        ``{address: value}`` sorted by address.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the wiring id for the
            first record, in record order, whose channel is a non-float
            setpoint, or a non-float readback that is not a waveform reading
            an optics output (raised before the deck is read); for an element
            absent from or repeated in the deck (an ``ElementStop``), an
            unreadable attribute, or a setpoint's or monitor's (a record whose
            engine block names an ``axis``) table calibration without an
            inverse or linear gain of 0; for a waveform readback that names an
            ``axis`` or ``index``, or whose shape is not the deck's planes;
            for a turn-by-turn record on a ring whose settings state no
            ``turn_by_turn`` block, a turn array that is not a waveform of
            shape ``[trigger_sample + turns]``, any other turn-by-turn record
            that is not float, or a monitor signal whose element is not a
            monitor (an ``ElementStop`` when the deck does not hold it once);
            naming the model, for a ``turn_by_turn`` block
            :func:`pyat_turn_settings.normalise` refuses.
    """
    readbacks = readbacks or {}
    value_types = value_types or {}
    records = list(wiring)
    turn_records = [record for record in records if turn_settings.signal_of(record) is not None]
    records = [record for record in records if turn_settings.signal_of(record) is None]
    _check_turn_types(deck, turn_records, settings, value_types)
    for record in records:
        address = field(record, "address")
        value_type = value_types.get(address, "float")
        whole = value_type == "waveform" and address in readbacks and _reads_an_output(record)
        if value_type != "float" and not whole:
            raise _stop(
                deck,
                field(record, "id") or address,
                "wiring",
                f"{address} is {value_type}; pyat drives float channels only",
                "wire a float channel, or leave the channel unwired",
            )
    from osprey.simulation.engines.pyat_model import tune_planes

    loaded = _load(deck)
    planes = tune_planes(loaded.lattice)
    whole_outputs: set[str] = set()
    for record in records:
        address = field(record, "address")
        if value_types.get(address, "float") == "waveform":
            _check_whole_output(record, (shapes or {}).get(address, ()), deck, planes)
            whole_outputs.add(address)
    values: dict[str, float | list[float]] = {}
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
        if address in whole_outputs:
            values[address] = [0.0] * planes
        else:
            values[address] = values.get(pair, 0.0) if pair is not None else 0.0
    values.update(_turn_start_values(loaded, deck, turn_records, settings, shapes or {}))
    return dict(sorted(values.items()))


def _check_turn_types(
    deck: Deck, records: list[Any], settings: Any, value_types: Mapping[str, str]
) -> None:
    """Refuse turn-by-turn records on a ring without the block, or of the wrong value type.

    Raised before the deck is read, in record order.
    """
    for record in records:
        address = field(record, "address")
        record_id = str(field(record, "id") or address)
        signal = turn_settings.signal_of(record)
        if not turn_settings.is_turn_by_turn(settings):
            raise _stop(
                deck,
                record_id,
                "wiring",
                f"{address} carries signal {signal}, but the model's settings state no "
                f"pyat.{turn_settings.BLOCK_KEY} block",
                f"state pyat.{turn_settings.BLOCK_KEY} with the injection_element, "
                "or unwire the record",
            )
        value_type = value_types.get(address, "float")
        if signal in turn_settings.ARRAY_SIGNALS and value_type != "waveform":
            raise _stop(
                deck,
                record_id,
                "wiring",
                f"{address} is {value_type}; a turn array is a waveform",
                "wire the channel as a waveform of trigger_sample + turns samples",
            )
        if signal not in turn_settings.ARRAY_SIGNALS and value_type != "float":
            raise _stop(
                deck,
                record_id,
                "wiring",
                f"{address} is {value_type}; pyat drives float channels only",
                "wire a float channel, or leave the channel unwired",
            )


def _turn_start_values(
    loaded: _Loaded,
    deck: Deck,
    records: list[Any],
    settings: Any,
    shapes: Mapping[str, Iterable[int]],
) -> dict[str, float | list[float]]:
    """Each turn-by-turn record's start value, checking its monitor and its array length."""
    import at

    if not records:
        return {}
    model_id = _model_id(deck, None)
    block = _turn_block(loaded, deck, settings, model_id)
    assert block is not None  # _check_turn_types refused every record on a ring without it
    values: dict[str, float | list[float]] = {}
    for record in records:
        address = field(record, "address")
        record_id = str(field(record, "id") or address)
        signal = str(turn_settings.signal_of(record))
        if signal in turn_settings.MONITOR_SIGNALS:
            element = field(record, "element")
            index = _element_index(loaded, deck, element, record_id, "wiring")
            if not isinstance(loaded.lattice[index], at.Monitor):
                raise _stop(
                    deck,
                    record_id,
                    "wiring",
                    f"{address} carries signal {signal}, but element {element} is not a monitor",
                    "name a monitor element of the deck",
                )
        elif signal == "stored" and field(record, "element") is not None:
            _element_index(loaded, deck, field(record, "element"), record_id, "wiring")
        if signal in turn_settings.ARRAY_SIGNALS:
            shape = list(shapes.get(address, ()))
            if shape != [block.samples]:
                raise _stop(
                    deck,
                    record_id,
                    "wiring",
                    f"{address} is a turn array of shape {shape}; it holds trigger_sample "
                    f"{block.trigger_sample} + turns {block.turns} samples",
                    f"declare the channel's shape as [{block.samples}]",
                )
        values[address] = turn_settings.start_value(signal, block)
    return values


def _reads_an_output(record: Any) -> bool:
    """Whether a record reads an optics output: no element, an optics attribute."""
    return (
        field(record, "element") is None
        and not field(record, "slices")
        and field(field(record, "engine"), "attribute") in OPTICS_ATTRIBUTES
    )


def _check_whole_output(record: Any, shape: Iterable[int], deck: Deck, planes: int) -> None:
    """Refuse a waveform record that does not read the whole of its optics output.

    A waveform names neither ``axis`` nor ``index``, and its ``shape`` is the
    deck's number of planes.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the record otherwise.
    """
    block = field(record, "engine")
    axis = field(block, "axis")
    index = field(block, "index")
    shape = list(shape)
    if axis is None and index is None and shape == [planes]:
        return
    address = field(record, "address")
    raise _stop(
        deck,
        str(field(record, "id") or address),
        "wiring",
        f"{address} is a waveform naming axis {axis!r}, index {index!r} and shape {shape}; "
        f"a waveform reads the whole of {field(block, 'attribute')} on the deck's {planes} planes",
        f"wire the waveform with no axis or index and shape [{planes}]",
    )


def describe(wiring_record: Any) -> Mapping[str, Any]:
    """Say what a wiring record is, in words that name no engine attribute.

    The description reads the record alone, never the deck:

    * ``direction: write`` is a ``setpoint``;
    * a read naming an element (or slices) with an engine ``axis`` and no
      ``attribute`` is a ``monitor``, its plane the axis;
    * a read naming an element (or slices) and an engine ``attribute`` is a
      ``readback``;
    * a read naming no element is a ``readback`` of a deck property such as
      ``energy``, else an ``output`` of the tunes or the chromaticity, its
      plane the ``axis``, else the ``index`` (0 is ``x``, 1 is ``y``); a
      whole-output waveform has no plane.

    A record whose engine block names a ``signal`` is a turn-by-turn record,
    described by :func:`pyat_turn_settings.describe_signal`: ``arm``,
    ``rf_switch``, ``pilot_tone`` and ``stored`` are setpoints refreshing
    ``pass``; ``armed``, ``turn_x``, ``turn_y``, ``turn_sum`` and
    ``turns_tracked`` are outputs refreshing ``event``; none has a plane, and
    the kind is the signal.

    A setpoint or readback has a plane only when it steers one: ``KickAngle``
    index 0 is ``x`` and index 1 is ``y``. A chromaticity output refreshes
    ``periodic``, published by the periodic solve alone; every other record
    refreshes ``pass``, with each write's solve.

    The ``kind`` is the device a setpoint or readback drives or reads back:
    ``kick`` for ``KickAngle``, ``rf`` for ``Frequency``, ``energy`` for the
    deck energy, ``strength`` for any other element attribute. A monitor's
    kind is ``monitor``; an optics output has none.

    Args:
        wiring_record: One wiring record, read by key or by attribute.

    Returns:
        ``{role, plane, refresh, kind}``: ``role`` one of ``setpoint``,
        ``readback``, ``monitor``, ``output``; ``plane`` ``x``, ``y`` or
        ``None``; ``refresh`` ``pass``, ``periodic`` or ``event``; ``kind``
        one of ``kick``, ``strength``, ``rf``, ``energy``, ``monitor``, a
        turn-by-turn signal, or ``None``.

    Raises:
        ValueError: naming the record's address, for a record none of the
            rules describes, or a turn-by-turn record wired against its
            signal's rules.
    """
    if turn_settings.signal_of(wiring_record) is not None:
        return turn_settings.describe_signal(wiring_record)
    block = field(wiring_record, "engine")
    attribute = field(block, "attribute")
    if field(wiring_record, "direction") == "write":
        return _description("setpoint", _steered_plane(block), _setting_kind(wiring_record))
    if _names_element(wiring_record):
        if _is_monitor_reading(wiring_record):
            axis = field(block, "axis")
            return _description("monitor", axis if axis in OPTICS_AXES else None, "monitor")
        if attribute is not None:
            return _description("readback", _steered_plane(block), _setting_kind(wiring_record))
    elif attribute in DECK_PROPERTIES:
        return _description("readback", None, _setting_kind(wiring_record))
    elif attribute in OPTICS_ATTRIBUTES:
        refresh = "periodic" if OPTICS_ATTRIBUTES[attribute] == "chromaticity" else "pass"
        return _description("output", _output_plane(block), None, refresh)
    raise ValueError(
        f"{field(wiring_record, 'address')} is no setpoint, monitor, readback or optics output"
    )


def describe_input(input_record: Any) -> Mapping[str, Any]:
    """Check one model input's engine block.

    A turn-by-turn ring takes inputs whose engine block is exactly ``{signal}``,
    the signal one of ``shot``, ``charge`` and ``stored_current``. Each signal
    is declared at most once, and ``shot`` and ``charge`` both once any is;
    one input alone cannot show that, so
    :func:`pyat_turn_settings.input_signals` checks the whole list.

    Args:
        input_record: ``{address, engine}``.

    Returns:
        ``{signal}``.

    Raises:
        ValueError: naming the input's address, for an engine block that is
            not a map, states no signal or an unknown one, or carries another
            key.
    """
    return turn_settings.describe_input(input_record)


def take_inputs(
    model: LUMEModel, inputs: Iterable[tuple[Any, Any]], t_s: float, *, baseline: bool
) -> None:
    """Store a turn-by-turn ring's delivered inputs; the next :func:`advance` acts on them.

    Args:
        model: A built pyat model; any other ring ignores the call.
        inputs: ``(input_record, value)`` pairs, the record's engine block
            naming the input's ``signal``.
        t_s: The epoch second of the delivery.
        baseline: The delivery is a starting point: nothing is compared
            against the one before it.
    """
    del t_s
    from osprey.simulation.engines.pyat_turn_state import TurnByTurnModel

    if not isinstance(model, TurnByTurnModel):
        return
    values = {
        str(field(field(record, "engine"), "signal")): float(value) for record, value in inputs
    }
    model.take_inputs(values, baseline=baseline)


def advance(model: LUMEModel, t_s: float) -> None:
    """Apply a turn-by-turn ring's pending drop, then track a delivered shot, by ``t_s``."""
    from osprey.simulation.engines.pyat_turn_state import TurnByTurnModel

    if isinstance(model, TurnByTurnModel):
        model.advance(t_s)


def client_write(model: LUMEModel, values: Mapping[str, Any], t_s: float) -> None:
    """Tell a turn-by-turn ring a client wrote ``values``; an ``arm`` record arms its monitor."""
    from osprey.simulation.engines.pyat_turn_state import TurnByTurnModel

    if isinstance(model, TurnByTurnModel):
        model.client_write(values, t_s)


def next_deadline(model: LUMEModel) -> float | None:
    """The shot's time while a turn-by-turn ring's handshake is pending, else ``None``."""
    from osprey.simulation.engines.pyat_turn_state import TurnByTurnModel

    return model.next_deadline() if isinstance(model, TurnByTurnModel) else None


def emitted(model: LUMEModel) -> list[str]:
    """A turn-by-turn ring's event channels changed since the last call, cleared."""
    from osprey.simulation.engines.pyat_turn_state import TurnByTurnModel

    return model.emitted() if isinstance(model, TurnByTurnModel) else []


def _description(
    role: str, plane: str | None, kind: str | None, refresh: str = "pass"
) -> Mapping[str, Any]:
    return {"engine_role": role, "plane": plane, "refresh": refresh, "kind": kind}


def _setting_kind(record: Any) -> str:
    """The device a setting record drives: a deck property by its name, else by attribute."""
    attribute = field(field(record, "engine"), "attribute")
    if not isinstance(attribute, str):
        return "strength"
    if attribute in DECK_PROPERTIES:
        return attribute
    return _ATTRIBUTE_KINDS.get(attribute, "strength")


def _steered_plane(block: Any) -> str | None:
    """The transverse plane a kick steers: ``KickAngle`` index 0 ``x``, 1 ``y``."""
    if field(block, "attribute") != "KickAngle":
        return None
    return _index_plane(field(block, "index"))


def _output_plane(block: Any) -> str | None:
    """The plane an optics output reads: its axis, else its index; none for the whole output."""
    axis = field(block, "axis")
    if axis is not None:
        return axis if axis in OPTICS_AXES else None
    return _index_plane(field(block, "index"))


def _index_plane(index: Any) -> str | None:
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    return _INDEX_PLANES.get(index)


def _target(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """What a record drives: its element or slices, and its engine attribute and index."""
    slices = tuple(
        (field(piece, "element"), field(piece, "weight")) for piece in field(record, "slices") or ()
    )
    block = field(record, "engine")
    return (field(record, "element"), slices, field(block, "attribute"), field(block, "index"))


def _names_element(record: Mapping[str, Any]) -> bool:
    return field(record, "element") is not None or bool(field(record, "slices"))


def _is_waveform(record: Mapping[str, Any]) -> bool:
    return bool(field(record, "value_type") == "waveform")


def _optics_source(record: Mapping[str, Any], deck: Deck, planes: int) -> tuple[str, int | None]:
    """The optics output an element-free record reads, and the plane it reads.

    A scalar record reads one plane, named by exactly one of ``axis`` and
    ``index``. A ``waveform`` record reads the whole output: it names neither,
    and its ``shape`` is the output's length; its plane is ``None``.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the record when its
            engine block names no optics attribute; a scalar record naming
            both or neither of ``axis`` and ``index``, an unknown axis, or a
            plane the deck does not have; a waveform record naming either, or
            whose shape is not the output's length.
    """
    block = field(record, "engine")
    attribute = field(block, "attribute")
    axis = field(block, "axis")
    index = field(block, "index")
    record_id = str(field(record, "id") or field(record, "address"))
    output = OPTICS_ATTRIBUTES.get(attribute) if isinstance(attribute, str) else None
    planes_text = ", ".join(str(plane) for plane in range(planes))
    if output is not None and _is_waveform(record):
        _check_whole_output(record, field(record, "shape") or (), deck, planes)
        return output, None
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
      or its own ``default`` when no setpoint does; a readback of a deck
      property such as ``energy`` reads back the setpoint of that property
      the same way, so no readback is ever writable;
    * a record naming no element reads one plane of the deck's ``tune`` or
      ``chromaticity`` -- its ``axis`` (``x`` 0, ``y`` 1) or its ``index`` --
      unless its attribute is a deck property such as ``energy``; a
      ``value_type: waveform`` record names neither and reads the whole
      output, read-only, its ``shape`` the deck's number of planes;
    * the ring's RF switch (``signal: rf_switch``, a setpoint on a
      turn-by-turn ring only) zeroes every cavity's voltage when written
      zero and restores the deck's otherwise
      (:class:`~osprey.simulation.engines.pyat_variables.RFSwitchVariable`);
      it starts on;
    * every other turn-by-turn record (an engine ``signal``, on a turn-by-turn
      ring only): ``arm``, ``pilot_tone`` and ``stored`` are settings that
      hold what is written; ``armed``, the turn arrays and ``turns_tracked``
      are read-only model state, read from the ring's turn state
      (:class:`~osprey.simulation.engines.pyat_turn_state.TurnState`), which
      a turn-by-turn ring is built to hold
      (:class:`~osprey.simulation.engines.pyat_turn_state.TurnByTurnModel`).

    The settings' ``solve`` picks the solve: ``periodic`` solves the closed
    orbit; ``single_pass`` tracks one pass through the line from
    ``twiss_in`` and serves no tunes or chromaticity. A turn-by-turn ring
    (one whose settings state a ``turn_by_turn`` block) is tolerant: a
    lattice with no closed orbit, as with the RF off, is a state it holds and
    serves -- its orbit, optics and chromaticity read NaN and its settings
    stay writable -- where any other ring refuses the write that leads there
    and rolls it back.

    The model drives its own copy of the deck, so the deck a later call loads
    is never the one a model has written to. Every other key of ``active``
    is a fault seed, ``<address>/<field>``, written onto its element before the
    model adopts the variables, or an element error, ``<element>/<field>``,
    on an element the wiring binds or the settings' ``error_elements`` declare
    (:func:`~osprey.simulation.engines.pyat_faults.fault_variables`): a
    misalignment (``dx``, ``dy``, ``roll``) becomes the element's entrance and
    exit transforms, a multipole error (``dB<n>``, ``dA<n>``) is added to the
    element's polynomial coefficient and held there across every later write
    of a setpoint on the same coefficient. The model is then reset once -- one batch, one
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
            refuses, an ``error_elements`` element the deck does not hold
            exactly once among them, or for a record naming no element that
            reads no plane of the tunes or the chromaticity.
        ValueError: an entry cannot be built as a variable, a default or an
            active value lies outside its ``value_range``, a key of
            ``active`` names no setpoint and no fault of the model, an
            element error lies outside its bounds or is not finite, an
            ``error_elements`` element is a wired monitor's or is named like
            a wired address
            (:class:`~osprey.simulation.engines.pyat_faults.ErrorElementsError`),
            a ``single_pass`` model wires a tune or
            chromaticity record, or a ring that is not a turn-by-turn ring
            wires an RF switch or another turn-by-turn record, or a turn
            record's element is not a monitor.
        pydantic.ValidationError: a fault seed lies outside its bounds.
        UnknownElementError: an element error names an element the deck does
            not hold exactly once, or a multipole error one with no such
            polynomial.
        OrbitSolveError: the deck, or the deck at the active scenario, has no
            stable solve on a ring that is not a turn-by-turn ring, or loses the single-pass particle; with element
            errors seeded, the message says so and names them.
    """
    from osprey.simulation.engines.pyat_model import (
        ArrayReadbackVariable,
        PyATLatticeModel,
        ReadbackVariable,
        tune_planes,
    )
    from osprey.simulation.engines.pyat_turn_state import TurnByTurnModel
    from osprey.simulation.engines.pyat_variables import (
        EV_PER_GEV,
        RF_SWITCH_SIGNAL,
        RFSwitchVariable,
        turn_variable_from_wiring,
        variable_from_wiring,
    )

    prepared = prepare(deck, settings, model=model)
    lattice = _load(deck).lattice.deepcopy()
    deck_energy_gev = float(lattice.energy) / EV_PER_GEV
    planes = tune_planes(lattice)
    entries = list(wiring)

    writes = {str(entry["address"]) for entry in entries if entry.get("direction") == "write"}
    setpoints: dict[tuple[Any, ...], str] = {}
    for entry in entries:
        if entry.get("direction") == "write":
            setpoints.setdefault(_target(entry), str(entry["address"]))

    channels: list[Any] = []
    for entry in entries:
        address = str(entry["address"])
        if turn_settings.signal_of(entry) == RF_SWITCH_SIGNAL:
            if prepared.turn_by_turn is None:
                raise ValueError(
                    f"model {model}: {address} is an RF switch, which only a turn-by-turn "
                    "ring serves; state a turn_by_turn block in its pyat settings"
                )
            channels.append(
                RFSwitchVariable.from_wiring(
                    entry, lattice=lattice, **_setting_overrides(entry, active)
                )
            )
            continue
        if turn_settings.signal_of(entry) is not None:
            if prepared.turn_by_turn is None:
                raise ValueError(
                    f"model {model}: {address} is a turn-by-turn record, which only a "
                    "turn-by-turn ring serves; state a turn_by_turn block in its pyat settings"
                )
            overrides = _setting_overrides(entry, active) if address in writes else {}
            channels.append(
                turn_variable_from_wiring(entry, samples=prepared.turn_by_turn.samples, **overrides)
            )
            continue
        attribute = field(entry.get("engine"), "attribute")
        deck_property = not _names_element(entry) and attribute in DECK_PROPERTIES
        if not _names_element(entry) and not deck_property:
            output, component = _optics_source(entry, deck, planes)
            if component is None:
                channels.append(
                    ArrayReadbackVariable(
                        name=address, source=output, shape=(planes,), unit=entry.get("unit")
                    )
                )
                continue
            channels.append(
                ReadbackVariable(
                    name=address, source=output, component=component, unit=entry.get("unit")
                )
            )
            continue
        if entry.get("direction") == "read" and (
            deck_property or (_names_element(entry) and attribute is not None)
        ):
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
        overrides = _setting_overrides(entry, active) if address in writes else {}
        variable = variable_from_wiring(entry, deck_energy_gev=deck_energy_gev, **overrides)
        if variable is not None:
            channels.append(variable)

    faults = {name: float(value) for name, value in active.items() if name not in writes}
    element_slots = {
        name: slot
        for name, slot in fault_variables(entries, settings).items()
        if slot.scope == ELEMENT_SCOPE
    }
    errors = {name: faults.pop(name) for name in sorted(faults) if name in element_slots}
    misalignments, multipoles = _element_errors(errors, element_slots)
    _seed_multipoles(lattice, multipoles)
    try:
        if prepared.turn_by_turn is None:
            built = PyATLatticeModel(
                lattice,
                channels,
                faults=faults,
                element_misalignments=misalignments or None,
                solve=prepared.solve,
                twiss_in=prepared.twiss_in,
            )
        else:
            built = TurnByTurnModel(
                lattice,
                channels,
                turn_settings=prepared.turn_by_turn,
                model_id=str(model),
                faults=faults,
                element_misalignments=misalignments or None,
                solve=prepared.solve,
                twiss_in=prepared.twiss_in,
            )
        built.reset()
    except SOLVE_ERRORS as exc:
        if not errors:
            raise
        raise OrbitSolveError(_seeded_failure(model, exc, errors)) from exc
    return built


def _element_errors(
    errors: Mapping[str, float], slots: Mapping[str, FaultSlot]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[tuple[str, int], float]]]:
    """Split element errors into misalignments and multipole deltas, each held to its bounds.

    Returns:
        ``{element: {dx|dy|roll: value}}`` and
        ``{element: {(PolynomB|PolynomA, n): delta}}``.

    Raises:
        ValueError: a value is not finite or lies outside its slot's range.
    """
    misalignments: dict[str, dict[str, float]] = {}
    multipoles: dict[str, dict[tuple[str, int], float]] = {}
    for name, value in errors.items():
        slot = slots[name]
        if not math.isfinite(value):
            raise ValueError(f"element error {name} is {value!r}; write a finite value")
        low, high = slot.value_range or (-math.inf, math.inf)
        if not low <= value <= high:
            raise ValueError(
                f"element error {name} is {value:g}, outside its range [{low:g}, {high:g}]"
            )
        polynomial = multipole_field(slot.field)
        if polynomial is None:
            misalignments.setdefault(slot.address, {})[slot.field] = value
        else:
            multipoles.setdefault(slot.address, {})[polynomial] = value
    return misalignments, multipoles


def _seed_multipoles(
    lattice: Any, multipoles: Mapping[str, Mapping[tuple[str, int], float]]
) -> None:
    """Add each multipole delta to its element's coefficient and record it on the element.

    A coefficient past the element's polynomials pads both with zeros and
    raises ``MaxOrder`` to reach it. Every element is resolved before any is
    written.

    Raises:
        UnknownElementError: an element is absent from the deck or carries
            no polynomial to add to.
        AmbiguousElementError: the deck repeats the element's name.
    """
    if not multipoles:
        return
    import numpy as np

    found: dict[str, list[Any]] = {}
    for element in lattice:
        if element.FamName in multipoles:
            found.setdefault(element.FamName, []).append(element)
    for name in multipoles:
        held = found.get(name, [])
        if not held:
            raise UnknownElementError(f"the lattice has no element {name!r} for element errors")
        if len(held) > 1:
            raise AmbiguousElementError(
                f"{len(held)} lattice elements are named {name!r}; an element error names one"
            )
        missing = [
            attribute
            for attribute in MULTIPOLE_PREFIXES.values()
            if not hasattr(held[0], attribute)
        ]
        if missing:
            raise UnknownElementError(
                f"element {name!r} carries no {' or '.join(missing)} for a multipole error"
            )
    for name, deltas in multipoles.items():
        element = found[name][0]
        reach = max(index for _, index in deltas) + 1
        order = int(getattr(element, "MaxOrder", 0))
        length = max(reach, order + 1)
        for attribute in MULTIPOLE_PREFIXES.values():
            coefficients = np.asarray(getattr(element, attribute), dtype=float)
            if coefficients.size < length:
                coefficients = np.concatenate([coefficients, np.zeros(length - coefficients.size)])
            setattr(element, attribute, coefficients)
        element.MaxOrder = max(order, reach - 1)
        for (attribute, index), delta in sorted(deltas.items()):
            getattr(element, attribute)[index] += delta
            setattr(element, multipole_attribute(attribute, index), float(delta))


def _seeded_failure(model: str, exc: BaseException, errors: Mapping[str, float]) -> str:
    """The solve failure of a model built with element errors, naming them."""
    elements = sorted({name.split(FAULT_SEPARATOR, 1)[0] for name in errors})
    shown = ", ".join(elements[:5]) + (
        f" and {len(elements) - 5} more" if len(elements) > 5 else ""
    )
    detail = " ".join(str(exc).split()) or type(exc).__name__
    return (
        f"model {model} has no solve with the element errors seeded on {len(elements)} "
        f"element(s) ({shown}): {detail}"
    )


def _is_monitor_reading(record: Any) -> bool:
    block = field(record, "engine")
    return field(block, "attribute") is None and field(block, "axis") is not None


def response_matrix(
    deck: Deck,
    wiring: Iterable[Any],
    settings: Any,
    correctors: list[str],
    step: Mapping[str, float],
    *,
    kind: str = RESPONSE_KINDS[0],
) -> ResponseMatrix:
    """Compute the response of the wired monitors to the named correctors.

    Each corrector is stepped by central difference, ``held ± step/2`` in its
    hardware unit about the value the deck holds, each arm converted to the
    deck's attribute through the corrector's calibration and solved on a copy
    of the deck: the closed orbit for a ``periodic`` model, one tracked pass
    from ``twiss_in`` for a ``single_pass`` one. Each reading is mapped to its
    readback's hardware unit through the way back its calibration states, so
    an entry is hardware readback per hardware setpoint. The deck a later call
    loads is never written to. A corrector whose step has no solve has no
    column; it is named in ``unsolved`` and the other correctors are still
    stepped.

    The ``trajectory`` kind answers on a turn-by-turn ring only. Each arm
    tracks one particle, without jitter, from the declared launch offsets for
    one turn from the injection element, through the declared apertures; a
    monitor upstream of the injection reads the end of that turn. A corrector
    whose kicked particle is lost before a monitor is named in ``unsolved``
    with where it was lost. When the particle is lost with no corrector
    stepped, every corrector is named in ``unsolved`` with that loss and the
    matrix has no column; nothing raises.

    Args:
        deck: The lattice file.
        wiring: The model's wiring records.
        settings: The model's settings, ``{pyat: {...}}``, or ``None``.
        correctors: The setpoint addresses to step, one column each, in this
            order.
        step: Each corrector's full step width in its hardware unit.
        kind: ``orbit`` (the default) or ``trajectory``.

    Returns:
        The :class:`ResponseMatrix`: a monitor no record wires has no row, a
        corrector whose step has no solve has no column and is named in
        ``unsolved``.

    Raises:
        FacilityBuildError: ``engine-invalid`` for settings :func:`prepare`
            refuses, or a monitor element the deck does not hold exactly once.
        RequestRefused: a corrector is not a wired setpoint naming an
            element, is named twice, or has no finite non-zero step; the kind
            is not one of :data:`RESPONSE_KINDS`; or the kind is
            ``trajectory`` and the ring is not a turn-by-turn ring.
        ValueError: a monitor record cannot be built as a reading, a defect
            of the wiring rather than of the request.
    """
    import numpy as np
    from lume_pyat.simulator import PyATSimulator

    from osprey.simulation.engines.pyat_single_pass import SinglePassSimulator
    from osprey.simulation.engines.pyat_variables import (
        EV_PER_GEV,
        CalibratedSetpoint,
        MonitorVariable,
        variable_from_wiring,
    )

    if kind not in RESPONSE_KINDS:
        raise RequestRefused(
            f"response kind {kind!r} is not one this engine answers; "
            f"ask for {' or '.join(RESPONSE_KINDS)}"
        )
    prepared = prepare(deck, settings)
    trajectory = kind == RESPONSE_KINDS[1]
    turns = prepared.turn_by_turn
    if trajectory and turns is None:
        raise RequestRefused(
            "the trajectory response is answered on a turn-by-turn ring only; "
            f"this model's settings state no pyat.{turn_settings.BLOCK_KEY} block"
        )
    loaded = _load(deck)
    lattice = loaded.lattice.deepcopy()
    deck_energy_gev = float(lattice.energy) / EV_PER_GEV
    records = list(wiring)

    readings: list[tuple[tuple[int, str], MonitorVariable]] = []
    for record in records:
        if not _is_monitor_reading(record):
            continue
        reading = MonitorVariable.from_wiring(record)
        record_id = str(field(record, "id") or reading.name)
        index = _element_index(loaded, deck, reading.element_name, record_id, "wiring")
        readings.append(((index, reading.axis), reading))
    readings.sort(key=lambda item: item[0])
    monitors = [reading for _, reading in readings]
    monitor_places = [place for (place, _), _ in readings]

    setpoints = {
        str(field(record, "address")): record
        for record in records
        if field(record, "direction") == "write" and _names_element(record)
    }
    columns: list[tuple[CalibratedSetpoint, float]] = []
    for address in correctors:
        record = setpoints.get(address)
        variable = (
            None
            if record is None
            else variable_from_wiring(record, deck_energy_gev=deck_energy_gev)
        )
        if not isinstance(variable, CalibratedSetpoint):
            raise RequestRefused(f"corrector {address} is not a wired setpoint on an element")
        if any(variable.name == other.name for other, _ in columns):
            raise RequestRefused(f"corrector {address} is named twice")
        width = step.get(address)
        if width is None or not math.isfinite(float(width)) or float(width) == 0.0:
            raise RequestRefused(
                f"corrector {address} has step {width!r}; give a finite non-zero step"
            )
        columns.append((variable, float(width)))

    if trajectory and turns is not None:
        if turns.aperture is not None:
            from osprey.simulation.engines.pyat_turns import write_apertures

            write_apertures(lattice, **turns.aperture.keywords)
        simulator = PyATSimulator(lattice)

        def measure() -> np.ndarray:
            return _first_turn(lattice, turns, monitors, monitor_places)

        try:
            measure()
        except SOLVE_ERRORS as exc:
            text = error_text(exc)
            return ResponseMatrix(
                rows=[reading.name for reading in monitors],
                columns=[],
                matrix=np.empty((len(monitors), 0), dtype=np.float64),
                unsolved={
                    variable.name: f"{text} (with no corrector stepped, before stepping "
                    f"{variable.name})"
                    for variable, _ in columns
                },
            )
    else:
        simulator = (
            PyATSimulator(lattice)
            if prepared.solve == SOLVES[0]
            else SinglePassSimulator(lattice, twiss_in=prepared.twiss_in)  # type: ignore[arg-type]
        )

        def measure() -> np.ndarray:
            simulator.solve()
            return np.array([reading._get(simulator) for reading in monitors])

    matrix = np.empty((len(monitors), len(columns)), dtype=np.float64)
    solved: list[int] = []
    unsolved: dict[str, str] = {}
    for column, (variable, width) in enumerate(columns):
        elements = [simulator.element(binding.element_name) for binding in variable.bindings]
        held_fields = [
            np.array(getattr(element, binding.attribute), copy=True)
            for binding, element in zip(variable.bindings, elements, strict=True)
        ]
        held = variable._get(simulator)
        arms: list[np.ndarray] = []
        try:
            for value in (held + 0.5 * width, held - 0.5 * width):
                variable._set(simulator, value)
                arms.append(measure())
        except SOLVE_ERRORS as exc:
            unsolved[variable.name] = f"{error_text(exc)} (while stepping {variable.name})"
            continue
        finally:
            for binding, element, saved in zip(
                variable.bindings, elements, held_fields, strict=True
            ):
                setattr(element, binding.attribute, saved if saved.ndim else float(saved))
        matrix[:, column] = (arms[0] - arms[1]) / width
        solved.append(column)
    return ResponseMatrix(
        rows=[reading.name for reading in monitors],
        columns=[columns[column][0].name for column in solved],
        matrix=matrix[:, solved],
        unsolved=unsolved,
    )


def _first_turn(
    lattice: Any,
    turns: turn_settings.TurnSettings,
    monitors: list[Any],
    places: list[int],
) -> np.ndarray:
    """Each monitor's first-turn reading of one particle, in its hardware unit.

    Raises:
        OrbitSolveError: the particle is lost before a monitor; its text says
            where.
    """
    import at
    import numpy as np

    from osprey.simulation.engines.pyat_turns import single_particle

    elements = sorted(set(places))
    row = {place: position for position, place in enumerate(elements)}
    try:
        reading = single_particle(lattice, turns.injection_index, elements, turns.launch)
    except at.AtError as exc:
        raise OrbitSolveError(
            f"first-turn tracking from {turns.injection_element} failed: {error_text(exc)}"
        ) from exc
    alive = reading.alive[:, 0]
    if np.any(alive < 1.0):
        raise OrbitSolveError(_loss_text(lattice, turns, elements, alive))
    return np.array(
        [
            to_hardware(
                monitor.calibration,
                monitor.inverse,
                float((reading.x if monitor.axis == "x" else reading.y)[row[place], 0]),
            )
            for monitor, place in zip(monitors, places, strict=True)
        ]
    )


def _loss_text(
    lattice: Any, turns: turn_settings.TurnSettings, elements: list[int], alive: Any
) -> str:
    """Where the first-turn particle was lost, and the first monitor it did not reach."""
    import at
    import numpy as np

    start = turns.injection_index
    length = len(lattice)
    reached = sorted(
        ((place - start) % length, place)
        for place, survived in zip(elements, alive, strict=True)
        if survived < 1.0
    )
    missed = lattice[reached[0][1]].FamName
    where = ""
    try:
        launch = np.asarray(turns.launch, dtype=np.float64).reshape(6, 1)
        _, _, data = at.lattice_track(lattice.rotate(start), launch, nturns=1, losses=True)
        loss = data["loss_map"]
        if bool(loss.islost[0]):
            index = (int(loss.elem[0]) + start) % length
            where = f" at element {index} ({lattice[index].FamName})"
    except at.AtError:
        where = ""
    return (
        f"particle launched at {turns.injection_element} lost on the first turn{where}, "
        f"before monitor {missed}"
    )


#: The pass method of an element that applies its ``KickAngle`` and no polynomial.
_KICK_PASS = "CorrectorPass"  # noqa: S105 -- names a tracking pass method, holds no secret

#: The attributes a kick element's polynomial replacement does not carry over.
_KICK_ONLY_ATTRIBUTES = frozenset(
    {"FamName", "Length", "PassMethod", "KickAngle", "PolynomA", "PolynomB", "MaxOrder"}
)


# Stand-in until pyAML reads a dipole kick held in KickAngle; polynomial_kicks and its EnginePlugin row go with it.
def polynomial_kicks(deck: Deck, elements: Iterable[str]) -> str:
    """Copy a deck so each named element carries its kick as its dipole polynomials.

    A reader that takes a corrector's strength as ``PolynomB[0]`` (horizontal,
    the kick's negative) and ``PolynomA[0]`` (vertical) times the element's
    length, or as the terms themselves for a zero-length element, reads
    nothing off an element that kicks by ``KickAngle``. The convention is the
    one pyAT's passes apply: a zero-length element's dipole terms are its
    integrated kick. In the copy each named element holding a ``KickAngle``
    carries the same integrated kick as polynomials, so stepping them moves
    the orbit the deck's kick moves:

    * an element that applies its kick alone becomes a multipole of the same
      length with ``PolynomB = [-KickAngle[0] / L]`` and
      ``PolynomA = [KickAngle[1] / L]``; one of zero length becomes a thin
      multipole with ``PolynomB = [-KickAngle[0]]`` and
      ``PolynomA = [KickAngle[1]]``;
    * an element whose pass applies its polynomials and adds its kick to them
      (``sin(KickAngle)``, over its length when it has one) has the kick
      folded into ``PolynomB[0]`` and ``PolynomA[0]`` and its ``KickAngle``
      zeroed.

    Every element keeps the deck's length and place. An element the deck does
    not hold, or one holding no ``KickAngle``, is copied as it is. The deck's
    file is never written to.

    Args:
        deck: The lattice file.
        elements: The element names whose kick to carry as polynomials.

    Returns:
        The copy as pyAT's JSON, two-space indented, without pyAT's version
        key, ending in a newline.
    """
    import at
    import numpy as np

    lattice = _load(deck).lattice.deepcopy()
    wanted = set(elements)
    for index in range(len(lattice)):
        element = lattice[index]
        name = str(element.FamName)
        kick = getattr(element, "KickAngle", None)
        if name not in wanted or kick is None:
            continue
        horizontal, vertical = (float(value) for value in np.ravel(kick)[:2])
        length = float(element.Length)
        if element.PassMethod != _KICK_PASS:
            _fold_kick(element, horizontal, vertical, length or 1.0)
        elif length == 0.0:
            kept = {
                key: value
                for key, value in vars(element).items()
                if key not in _KICK_ONLY_ATTRIBUTES
            }
            lattice[index] = at.ThinMultipole(name, [vertical], [-horizontal], **kept)
        else:
            lattice[index] = _polynomial_kick(element, horizontal, vertical, length)
    return _json_text(lattice)


def _polynomial_kick(element: Any, horizontal: float, vertical: float, length: float) -> Any:
    """A multipole of ``length`` applying ``element``'s kick, its other attributes kept."""
    import at

    kept = {key: value for key, value in vars(element).items() if key not in _KICK_ONLY_ATTRIBUTES}
    return at.Multipole(
        element.FamName, length, [vertical / length], [-horizontal / length], **kept
    )


def _fold_kick(element: Any, horizontal: float, vertical: float, length: float) -> None:
    """Move a multipole's ``KickAngle`` into its dipole polynomial terms, as its pass adds it.

    ``length`` is the element's length, or 1 for a zero-length element, whose
    terms are the integrated kick its thin pass applies.
    """
    import numpy as np

    for name, term in (
        ("PolynomB", -math.sin(horizontal) / length),
        ("PolynomA", math.sin(vertical) / length),
    ):
        polynomial = np.array(getattr(element, name, np.zeros(1)), dtype=float, copy=True)
        if polynomial.size == 0:
            polynomial = np.zeros(1)
        polynomial[0] += term
        setattr(element, name, polynomial)
    element.KickAngle = np.zeros(2)


def _json_text(lattice: Any) -> str:
    """``lattice`` as pyAT's JSON without its version key, so equal decks give equal text."""
    import contextlib
    import io
    import json

    import at

    rendered = io.StringIO()
    with contextlib.redirect_stdout(rendered):
        at.save_json(lattice)
    document = json.loads(rendered.getvalue())
    document.pop("at_version", None)
    return json.dumps(document, indent=2) + "\n"


def error_text(exc: BaseException) -> str:
    """The one line a failed build or solve is reported with.

    Args:
        exc: What the build or the solve raised.

    Returns:
        An ``OrbitSolveError``'s own message, which names the solve that
        failed; any other exception's type name and message. Whitespace runs
        collapse to one space.
    """
    text = " ".join(str(exc).split())
    if isinstance(exc, OrbitSolveError) and text:
        return text
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
