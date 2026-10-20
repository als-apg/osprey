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
* :func:`build` builds the LUME model that serves the wiring;
* :func:`response_matrix` steps wired correctors and returns the orbit
  response at the wired monitors, naming each corrector whose step has no
  solve;
* :func:`polynomial_kicks` copies a deck so named kick elements carry their
  kicks as dipole polynomials;
* :func:`error_text` gives the one line a failed build or solve reports;
* :func:`condition` says whether a built model is degraded, and which of its
  values that voids;
* :func:`readout` turns solved monitor positions into what each monitor
  reports;
* :func:`fault_variables` names every fault a model built from a wiring
  declares, without building it;
* :func:`scenario_stops` says what a scenario asks of a model that its deck
  cannot give.

A wiring record is read by key or by attribute: ``id``, ``address``,
``element`` or ``slices`` (each ``element``, ``weight``), the ``engine`` block
in pyAT's words (``attribute``, ``index``, ``axis``, ``series``) and ``calibration``
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
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from lume_pyat.exceptions import OrbitSolveError

from osprey.facility.errors import FacilityBuildError
from osprey.simulation.engines.calibration import NoInverse, curve_from_record, field, to_hardware
from osprey.simulation.engines.pyat_faults import (
    APERTURE_SHAPES,
    LAUNCH_COORDINATES,
    LAUNCH_FAULTS,
    MISALIGNED_MAGNETS,
    MISALIGNMENT_FIELDS,
    MISALIGNMENT_SEED,
    MODEL_WIDE_FAULTS,
    READOUT_ATTRIBUTE_IDENTITY,
    WHOLE_FAULTS,
    aperture_half_axis,
    fault_variables,
    is_series_record,
    misalignment_sigma,
    readout,
)

if TYPE_CHECKING:
    import numpy as np
    from lume.model import LUMEModel

__all__ = [
    "CONDITIONS",
    "GLOBAL_ATTRIBUTES",
    "GLOBAL_KIND",
    "STORED_CURRENT_ATTRIBUTE",
    "TURNS_TRACKED_ATTRIBUTE",
    "TURNS_TRACKED_DEFAULT",
    "ElementStop",
    "Injection",
    "Prepared",
    "ResponseMatrix",
    "SOLVE_ERRORS",
    "build",
    "condition",
    "error_text",
    "fault_variables",
    "describe",
    "locate",
    "polynomial_kicks",
    "prepare",
    "readout",
    "response_matrix",
    "scenario_stops",
    "start_values",
]

#: The keys the ``pyat`` settings block may carry.
SETTINGS_KEYS: frozenset[str] = frozenset({"solve", "twiss_in", "rest_mass_gev", "injection"})

#: The coordinates of a launch vector, in the order pyAT tracks them. A
#: scenario states a periodic model's launch through the model-wide faults
#: ``launch_<coordinate>``; the ``injection`` block states none.
LAUNCH_KEYS: tuple[str, ...] = LAUNCH_COORDINATES

#: The keys the ``injection`` block may carry. ``trigger_sample`` is never
#: authored with the model: the build writes it from the measurement file's
#: ``trajectory`` declaration, the one place the trigger sample is stated.
INJECTION_KEYS: frozenset[str] = frozenset({"marker", "trigger_sample"})

#: The samples a turn series holds before the launch when the block states no
#: trigger sample: a model whose measurement file declares no trajectory.
PRE_TRIGGER_SAMPLES: int = 10

#: The ``solve`` values, the first one the default.
SOLVES: tuple[str, ...] = ("periodic", "single_pass")

# Stand-in until lume-base states SolveError, the base of every LUME engine's no-solve exception.
#: The exceptions with which this engine says a request has no solve.
SOLVE_ERRORS: tuple[type[Exception], ...] = (OrbitSolveError,)

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
_ATTRIBUTE_KINDS: dict[str, str] = {"KickAngle": "kick", "Frequency": "rf", "Voltage": "rf"}

#: The engine-block attributes a write naming no element commands the model
#: with: ``arm`` readies the turn series, ``inject`` requests a launch, and
#: ``clear_stored_charge`` empties the charge the model has stored.
COMMAND_ATTRIBUTES: frozenset[str] = frozenset({"arm", "inject", "clear_stored_charge"})

#: The engine-block attribute a read naming no element reads the armed flag by.
ARMED_ATTRIBUTE: str = "armed"

#: The engine-block attribute a read naming no element reads the model's
#: stored charge by, served as its current. Its engine block also carries the
#: decay data of a stored beam with no RF: ``surviving_fraction`` per shot and
#: ``charge_floor``.
STORED_CURRENT_ATTRIBUTE: str = "stored_current"

#: The engine-block attribute of the model's turns tracked: the turns after
#: the trigger sample a shot tracks and its turn series read beam on.
TURNS_TRACKED_ATTRIBUTE: str = "turns_tracked"

#: The engine-block attributes a record naming no element sets or reads one
#: model-wide engine value by: a global setting, held by the model and written
#: to no element.
GLOBAL_ATTRIBUTES: frozenset[str] = frozenset({TURNS_TRACKED_ATTRIBUTE})

#: The kind a global setting and its readback are described with.
GLOBAL_KIND: str = "global"

#: The turns tracked of a model that wires no setting for it, or whose setting
#: states no default, up to the samples its turn series hold after the trigger.
TURNS_TRACKED_DEFAULT: int = 100

#: The key a command's description carries, ``True``; no other description
#: carries it.
COMMAND_MARK: str = "command"

#: The role of a record a shot fills: a turn series, or the armed flag.
TURN_SERIES: str = "turn_series"

#: What a periodic model's lattice lacks while its solve finds no closed orbit.
NO_CLOSED_ORBIT: str = "no closed orbit"

#: Every text :func:`condition` can answer.
CONDITIONS: tuple[str, ...] = (NO_CLOSED_ORBIT,)

Deck = str | os.PathLike[str]

_NOTHING_ACTIVE: Mapping[str, Any] = MappingProxyType({})

# Each built model's closed-orbit names, held for as long as the model lives.
_CLOSED_ORBIT_NAMES: weakref.WeakKeyDictionary[Any, frozenset[str]] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class Injection:
    """Where a model launches a tracked particle, and from what.

    Attributes:
        marker: The deck element the particle is launched at; ``None`` on a
            single-pass model, which launches at the start of the deck.
        index: That element's index in the deck.
        launch: The launch vector, six floats in pyAT's order ``x px y py
            delta ct``: zeros as a periodic model's settings give it -- the
            identity launch its response matrix linearises about -- and the
            scenario's ``launch_<coordinate>`` faults on a built model.
        trigger_sample: The samples a turn series holds before the
            launch.
    """

    marker: str | None
    index: int
    launch: np.ndarray
    trigger_sample: int


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
        injection: The launch of a tracked particle: the ``injection`` block
            of a periodic model, ``None`` when it states none; on a
            single-pass model the launch from ``twiss_in.closed_orbit`` at
            the start of the deck.
    """

    solve: str
    twiss_in: dict[str, np.ndarray] | None
    rest_mass_gev: float
    length_m: float
    injection: Injection | None = None


@dataclass(frozen=True)
class ResponseMatrix:
    """The response of a model's wired readings to its stepped correctors.

    For the ``orbit`` kind the rows are the wired monitor readback addresses,
    ordered by (element lattice index, ``x`` before ``y``), and the columns
    keep the order asked. For the ``trajectory`` kind the rows are the
    turn-series record addresses, every ``x`` record in the order the launched
    particle meets its monitor and then every ``y`` record; the columns are
    the horizontal correctors in the order asked, then the vertical ones.

    Attributes:
        rows: The reading addresses, one row each.
        columns: The correctors whose two arms solved.
        matrix: One row per address of ``rows`` and one column per corrector
            of ``columns``, hardware reading per hardware setpoint.
        unsolved: Each corrector whose step has no solve, in column order,
            mapped to :func:`error_text` of the failure followed by
            ``(while stepping <address>)``.
        index: For the ``trajectory`` kind, the lattice index of every row
            and of every corrector asked, solved or not: the element's
            position as the launched particle meets it, the deck rotated to
            the launch marker for a ``periodic`` model and as written for a
            ``single_pass`` one. ``None`` for the ``orbit`` kind.
    """

    rows: list[str]
    columns: list[str]
    matrix: np.ndarray
    unsolved: dict[str, str]
    index: dict[str, int] | None = None


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


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def _normalise_injection(injection: Any, loaded: _Loaded, deck: Deck, model_id: str) -> Injection:
    import numpy as np

    if not isinstance(injection, Mapping):
        raise _stop(
            deck,
            model_id,
            "model",
            "settings key pyat.injection is not a map",
            "state injection as a map holding the marker",
        )
    unknown = sorted(str(key) for key in set(injection) - INJECTION_KEYS)
    if unknown:
        raise _stop(
            deck,
            model_id,
            "model",
            f"settings key pyat.injection carries unknown keys {', '.join(unknown)}",
            "use only marker (the build adds trigger_sample from the measurement "
            "file's trajectory declaration); a scenario states the launch "
            "as the model-wide faults launch_x, launch_px, launch_y, launch_py, "
            "launch_delta, launch_ct",
        )
    if "marker" not in injection:
        raise _stop(
            deck,
            model_id,
            "model",
            "settings key pyat.injection lacks marker",
            "state marker as the name of a deck element",
        )
    marker = injection["marker"]
    if not isinstance(marker, str) or not marker:
        raise _stop(
            deck,
            model_id,
            "model",
            f"settings key pyat.injection.marker is {marker!r}",
            "state marker as the name of a deck element",
        )
    index = _element_index(loaded, deck, marker, model_id, "model")
    launch = np.zeros(len(LAUNCH_KEYS))
    samples = injection.get("trigger_sample", PRE_TRIGGER_SAMPLES)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise _stop(
            deck,
            model_id,
            "model",
            f"settings key pyat.injection.trigger_sample is {samples!r}",
            "declare trajectory.trigger_sample as a whole number of samples, zero or more",
        )
    return Injection(marker=marker, index=index, launch=launch, trigger_sample=samples)


def prepare(deck: Deck, settings: Any, *, model: str | None = None) -> Prepared:
    """Check a deck against a model's settings and normalise the ``pyat`` block.

    Args:
        deck: The lattice file.
        settings: The model's settings, ``{pyat: {solve?, twiss_in?,
            rest_mass_gev?, injection?}}``, or ``None``. ``injection`` is
            ``{marker, trigger_sample?}``; the build writes ``trigger_sample``
            from the measurement file's ``trajectory`` declaration.
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
            ``rest_mass_gev`` that is not a finite non-negative number,
            ``injection`` on a single-pass model, or an ``injection`` block
            with an unknown key, a ``marker`` the deck does not hold exactly
            once (an :class:`ElementStop`), a coordinate that is not a finite
            number or a ``trigger_sample`` that is not a count.
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
            "use only solve, twiss_in, rest_mass_gev, injection",
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

    stated = block.get("injection")
    injection: Injection | None = None
    if solve == "single_pass":
        if stated is not None:
            raise _stop(
                deck,
                model_id,
                "model",
                "settings key pyat.injection is set on a single_pass model",
                "remove injection or set solve to periodic",
            )
        # A single_pass model without twiss_in stopped above.
        assert twiss_in is not None
        injection = Injection(
            marker=None,
            index=0,
            launch=twiss_in["closed_orbit"].copy(),
            trigger_sample=PRE_TRIGGER_SAMPLES,
        )
    elif stated is not None:
        injection = _normalise_injection(stated, loaded, deck, model_id)
    return Prepared(
        solve=solve,
        twiss_in=twiss_in,
        rest_mass_gev=rest_mass_gev,
        length_m=loaded.length_m,
        injection=injection,
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
    raw: Any = getattr(loaded.lattice[index], attribute, READOUT_ATTRIBUTE_IDENTITY.get(attribute))
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


def _labelled_readout(record: Any, deck: Deck, address: str) -> bool:
    """Whether a record is a labelled readout setting whose table maps its label indices.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the record when it is
            marked ``discrete`` but its table is refused.
    """
    from osprey.simulation.engines.pyat_variables import (
        is_readout_setting_record,
        readout_setting_levels,
    )

    if not is_readout_setting_record(record):
        return False
    try:
        readout_setting_levels(record)
    except ValueError as exc:
        raise _stop(
            deck,
            field(record, "id") or address,
            "wiring",
            str(exc),
            "give the record a calibration table from each label index to its value",
        ) from exc
    return True


def _readout_setting_default(record: Any, deck: Deck) -> int:
    """The label index a labelled readout setting boots at: its stated default, else 0.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the record when its
            stated default is not one of its label indices.
    """
    from osprey.simulation.engines.pyat_variables import readout_setting_from_wiring

    stated = field(record, "default")
    try:
        built = readout_setting_from_wiring(record, 0 if stated is None else stated)
    except ValueError as exc:
        raise _stop(
            deck,
            field(record, "id") or field(record, "address"),
            "wiring",
            str(exc),
            "state a default that is one of the record's label indices",
        ) from exc
    return int(built.default_value)


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
    value is the hardware value. The RF-on switch (a cavity's ``Voltage``)
    starts on, at its on index. A labelled readout setting (a readout
    attribute on an element, its engine block marked ``discrete``) starts at
    the label index its record states as ``default``, else at 0, the first
    label. A readback takes its paired setpoint's value
    when that setpoint is in ``wiring``, else zero; a waveform readback wired
    to a whole optics output starts as one zero per plane. Any other record
    that names no element has no start value in the deck and is left out.

    The records a shot fills start at rest. A turn series (an engine block
    carrying ``series``) starts as one zero per sample of its ``shape``, the
    armed flag (``attribute: armed`` on no element) and a command (one of
    :data:`COMMAND_ATTRIBUTES` on no element) as the whole number 0. Every
    turn series of a model holds the same number of samples, at least one
    more than the model's pre-trigger samples. The turns tracked (one of
    :data:`GLOBAL_ATTRIBUTES` on no element) starts at the default its record
    states, else at :data:`TURNS_TRACKED_DEFAULT` up to the samples the
    series hold after the trigger; its readback takes its paired setting's
    value like any readback, or that default when it has no pair.

    Args:
        deck: The lattice file.
        wiring: The model's wiring records.
        settings: The model's settings, read only for a model wiring a turn
            series: its pre-trigger samples bound the series' length.
        readbacks: Each readback address among the records, mapped to the
            address of its paired setpoint or ``None``; every other record is
            a setpoint.
        value_types: Each wired channel's ``value_type``; an address it
            does not name is float. pyat drives float channels only, but for
            a waveform readback wired to a whole optics output, a waveform
            turn series, an int armed flag, command or global setting, and a
            bool or enum RF-on switch, labelled readout setting, or readback
            of either.
        shapes: Each wired waveform channel's ``shape``; a readback wired to
            a whole optics output is as long as the deck has planes.

    Returns:
        ``{address: value}`` sorted by address.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the wiring id for the
            first record, in record order, whose channel is a turn series
            that is not a waveform, a non-float setpoint that is not an int
            command, or a non-float readback that is neither a waveform
            reading an optics output nor the int armed flag, or a bool or
            enum labelled readout setting whose table grid is not its label
            indices (raised before the deck is read);
            for a labelled readout setting whose stated default is no label
            index; for an element
            absent from or repeated in the deck (an ``ElementStop``), an
            unreadable attribute, or a setpoint's or monitor's (a record whose
            engine block names an ``axis``) table calibration without an
            inverse or linear gain of 0; for a waveform readback that names an
            ``axis`` or ``index``, or whose shape is not the deck's planes;
            for a turn series whose shape is not one length of at least the
            pre-trigger samples and one, or not the length of the model's
            other turn series, or whose model states no injection; and
            whatever :func:`prepare` raises for the settings of a model
            wiring a turn series.
    """
    from osprey.simulation.engines.pyat_variables import (
        RF_ON_INDEX,
        is_readout_setting_record,
        is_rf_switch_record,
    )
    from osprey_connectors.simulation.values import is_waveform

    readbacks = readbacks or {}
    value_types = value_types or {}
    records = list(wiring)
    for record in records:
        address = field(record, "address")
        value_type = value_types.get(address, "float")
        if is_series_record(record):
            if not is_waveform(value_type):
                raise _stop(
                    deck,
                    field(record, "id") or address,
                    "wiring",
                    f"{address} is {value_type}; a turn series is a waveform",
                    "wire a waveform channel, or leave the channel unwired",
                )
            continue
        labelled = value_type in ("bool", "enum")
        if is_waveform(value_type):
            admitted = address in readbacks and _reads_an_output(record)
        else:
            admitted = (
                value_type == "float"
                or (value_type == "int" and (_holds_shot_state(record) or _is_global(record)))
                or (
                    labelled
                    and (is_rf_switch_record(record) or _labelled_readout(record, deck, address))
                )
            )
        if not admitted:
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
        if is_waveform(value_types.get(address)) and not is_series_record(record):
            _check_whole_output(record, (shapes or {}).get(address, ()), deck, planes)
            whole_outputs.add(address)
    samples = _series_length(records, shapes or {}, deck, settings)
    bound = (
        _turns_bound(samples, prepare(deck, settings).injection)
        if samples and any(_is_global(record) for record in records)
        else None
    )
    values: dict[str, float | list[float]] = {}
    pending: list[str] = []
    for record in records:
        address = field(record, "address")
        record_id = field(record, "id") or address
        if field(field(record, "engine"), "axis") is not None:
            _hardware(0.0, field(record, "calibration"), deck, record_id, _SERVE_READING)
        if is_series_record(record):
            values[address] = [0.0] * samples
            continue
        if _holds_shot_state(record):
            values[address] = 0
            continue
        pair = readbacks.get(address)
        if _is_global(record) and (address not in readbacks or pair is None):
            turns = turns_tracked_default(field(record, "default"), bound)
            values[address] = int(turns) if value_types.get(address) == "int" else float(turns)
            continue
        if address in readbacks and pair != address:
            pending.append(address)
            continue
        if is_rf_switch_record(record):
            values[address] = RF_ON_INDEX
            continue
        if is_readout_setting_record(record):
            values[address] = _readout_setting_default(record, deck)
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
    return dict(sorted(values.items()))


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


def _series_length(
    records: Iterable[Any], shapes: Mapping[str, Iterable[int]], deck: Deck, settings: Any
) -> int:
    """The samples every turn series of a model holds; zero for a model wiring none.

    Each series record's ``shape`` is one length, the same on every series
    record of the model, and long enough to hold the launch after the model's
    pre-trigger samples. The settings are read only for a model wiring a
    series.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the first series record,
            in record order, that breaks the rule, or the first one of a model
            that launches nothing; whatever :func:`prepare` raises for the
            settings.
    """
    series = [record for record in records if is_series_record(record)]
    if not series:
        return 0
    injection = prepare(deck, settings).injection
    samples: int | None = None
    for record in series:
        address = field(record, "address")
        record_id = str(field(record, "id") or address)
        if injection is None:
            raise _stop(
                deck,
                record_id,
                "wiring",
                f"{address} is a turn series on a model whose settings state no injection",
                "state settings.pyat.injection, or leave the series unwired",
            )
        least = injection.trigger_sample + 1
        shape = list(shapes.get(address, ()))
        length = shape[0] if len(shape) == 1 else None
        if isinstance(length, bool) or not isinstance(length, int) or length < least:
            raise _stop(
                deck,
                record_id,
                "wiring",
                f"{address} is a turn series of shape {shape}; a turn series holds the "
                f"{injection.trigger_sample} samples before the launch and at least one after",
                f"give the channel a shape of one length, {least} or more",
            )
        if samples is None:
            samples = length
        elif length != samples:
            raise _stop(
                deck,
                record_id,
                "wiring",
                f"{address} is a turn series of {length} samples; "
                f"the model's other turn series hold {samples}",
                "give every turn series of the model the same shape",
            )
    return samples or 0


def describe(wiring_record: Any) -> Mapping[str, Any]:
    """Say what a wiring record is, in words that name no engine attribute.

    The description reads the record alone, never the deck:

    * ``direction: write`` is a ``setpoint``; one naming no element whose
      engine ``attribute`` is one of :data:`COMMAND_ATTRIBUTES` is a command,
      and its description alone carries :data:`COMMAND_MARK`, ``True``;
    * a read naming an element (or slices) whose engine block carries
      ``series`` is a ``turn_series``, its plane the axis when that is ``x``
      or ``y``;
    * a read naming an element (or slices) with an engine ``axis`` and no
      ``attribute`` is a ``monitor``, its plane the axis;
    * a read naming an element (or slices) and an engine ``attribute`` is a
      ``readback``;
    * a read naming no element is a ``readback`` of a deck property such as
      ``energy``, else an ``output`` of the tunes or the chromaticity, its
      plane the ``axis``, else the ``index`` (0 is ``x``, 1 is ``y``); a
      whole-output waveform has no plane;
    * a read naming no element whose engine ``attribute`` is ``armed`` is a
      ``turn_series`` with no plane;
    * a read naming no element whose engine ``attribute`` is
      :data:`STORED_CURRENT_ATTRIBUTE` is an ``output`` with no plane and no
      kind, refreshed ``periodic``: a shot, a clearing and a lost closed
      orbit each move it, so the periodic pass publishes it;
    * a record naming no element whose engine ``attribute`` is one of
      :data:`GLOBAL_ATTRIBUTES` is a global setting: a ``setpoint`` when
      written, else its ``readback``, of kind :data:`GLOBAL_KIND` and no
      plane.

    A setpoint or readback has a plane only when it steers one: ``KickAngle``
    index 0 is ``x`` and index 1 is ``y``. A chromaticity output refreshes
    ``periodic``, published by the periodic solve alone; every other record
    refreshes ``pass``, with each write's solve.

    The ``kind`` is the device a setpoint or readback drives or reads back:
    ``kick`` for ``KickAngle``, ``rf`` for ``Frequency`` and the RF-on
    switch's ``Voltage``, ``energy`` for the deck energy, ``global`` for a
    global setting, ``strength`` for any other element attribute. A monitor's
    kind is ``monitor``; an optics output, a turn series and a command have
    none.

    Args:
        wiring_record: One wiring record, read by key or by attribute.

    Returns:
        ``{role, plane, refresh, kind}``: ``role`` one of ``setpoint``,
        ``readback``, ``monitor``, ``output``, ``turn_series``; ``plane``
        ``x``, ``y`` or ``None``; ``refresh`` ``pass`` or ``periodic``;
        ``kind`` one of ``kick``, ``strength``, ``rf``, ``energy``,
        ``global``, ``monitor``, or ``None``. A command's description carries
        :data:`COMMAND_MARK` as a fifth key.

    Raises:
        ValueError: naming the record's address, for a record none of the
            rules describes.
    """
    block = field(wiring_record, "engine")
    attribute = field(block, "attribute")
    if field(wiring_record, "direction") == "write":
        if _is_command(wiring_record):
            return {**_description("setpoint", None, None), COMMAND_MARK: True}
        return _description("setpoint", _steered_plane(block), _setting_kind(wiring_record))
    if _names_element(wiring_record):
        if is_series_record(wiring_record):
            axis = field(block, "axis")
            return _description(TURN_SERIES, axis if axis in OPTICS_AXES else None, None)
        if _is_monitor_reading(wiring_record):
            axis = field(block, "axis")
            return _description("monitor", axis if axis in OPTICS_AXES else None, "monitor")
        if attribute is not None:
            return _description("readback", _steered_plane(block), _setting_kind(wiring_record))
    elif _is_stored_current(wiring_record):
        return _description("output", None, None, "periodic")
    elif attribute in DECK_PROPERTIES or _is_global(wiring_record):
        return _description("readback", None, _setting_kind(wiring_record))
    elif attribute in OPTICS_ATTRIBUTES:
        refresh = "periodic" if OPTICS_ATTRIBUTES[attribute] == "chromaticity" else "pass"
        return _description("output", _output_plane(block), None, refresh)
    elif attribute == ARMED_ATTRIBUTE:
        return _description(TURN_SERIES, None, None)
    raise ValueError(
        f"{field(wiring_record, 'address')} is no setpoint, monitor, readback or optics output"
    )


def _is_command(record: Any) -> bool:
    """Whether a record's engine block is a command's: no element, a command attribute."""
    attribute = field(field(record, "engine"), "attribute")
    return (
        not _names_element(record)
        and isinstance(attribute, str)
        and attribute in COMMAND_ATTRIBUTES
    )


def _is_stored_current(record: Any) -> bool:
    """Whether a record reads the stored charge: a read naming no element, its attribute."""
    return (
        field(record, "direction") != "write"
        and not _names_element(record)
        and field(field(record, "engine"), "attribute") == STORED_CURRENT_ATTRIBUTE
    )


def _is_global(record: Any) -> bool:
    """Whether a record sets or reads a global setting: no element, a global attribute."""
    attribute = field(field(record, "engine"), "attribute")
    return (
        not _names_element(record) and isinstance(attribute, str) and attribute in GLOBAL_ATTRIBUTES
    )


def turns_tracked_default(stated: Any, bound: int | None) -> float:
    """The turns tracked a model starts at.

    The default a setting's record states when it states one, else
    :data:`TURNS_TRACKED_DEFAULT` up to ``bound``, the samples the model's
    turn series hold after the trigger (``None`` for no bound).
    """
    if stated is not None:
        return float(stated)
    if bound is None:
        return float(TURNS_TRACKED_DEFAULT)
    return float(min(TURNS_TRACKED_DEFAULT, bound))


def _turns_bound(samples: int, injection: Injection | None) -> int | None:
    """The most turns a model's series hold after the trigger; ``None`` with no series."""
    if not samples or injection is None:
        return None
    return samples - injection.trigger_sample


def _holds_shot_state(record: Any) -> bool:
    """Whether a record is a command or the armed flag: a whole number, zero at rest."""
    return _is_command(record) or (
        not _names_element(record)
        and field(field(record, "engine"), "attribute") == ARMED_ATTRIBUTE
    )


def _description(
    role: str, plane: str | None, kind: str | None, refresh: str = "pass"
) -> Mapping[str, Any]:
    return {"role": role, "plane": plane, "refresh": refresh, "kind": kind}


def _setting_kind(record: Any) -> str:
    """The device a setting record drives: a deck property by its name, a global
    setting as :data:`GLOBAL_KIND`, else by attribute."""
    attribute = field(field(record, "engine"), "attribute")
    if not isinstance(attribute, str):
        return "strength"
    if _is_global(record):
        return GLOBAL_KIND
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
    from osprey_connectors.simulation.values import is_waveform

    return is_waveform(field(record, "value_type"))


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
    * a command (a write naming no element, its attribute one of
      :data:`COMMAND_ATTRIBUTES`) writes nothing to the deck: ``arm`` written
      1 sets the model's one armed flag, and an ``inject`` request written
      above the last value the model saw is one shot;
    * a read naming no element whose attribute is ``armed`` reads that flag;
    * a labelled readout setting (a readout attribute on an element, its
      engine block marked ``discrete``) holds the index of one of its labels
      and writes the value its calibration table states for that index onto
      the element; its default is the active value when ``active`` names it,
      else the entry's ``default``, else 0. A read of the same record answers
      the index whose table value the element carries;
    * a read naming no element whose attribute is ``stored_current`` reads the
      charge the model has stored, as a current: the charge times the deck's
      revolution frequency, through the entry's calibration. Its engine block
      states the share of the stored charge a shot with every cavity off
      leaves (``surviving_fraction``, 0 when unstated) and the charge at or
      below which it is gone (``charge_floor``, 0 when unstated);
    * a record naming no element whose attribute is ``turns_tracked`` is the
      model's one turns tracked, a global setting: the turns after the
      trigger sample a shot tracks, whole turns from 1 to the samples the
      series hold after the trigger, :data:`TURNS_TRACKED_DEFAULT` up to that
      bound unless the record states a default; a read of it is its readback;
    * a turn series (an element and an engine ``series``) holds one sample of
      its monitor per sample, as long as its ``shape``: zero until an armed
      shot fills it, through the entry's calibration; the turns tracked
      after the trigger read beam, and every other sample the no-beam
      reading.

    The settings' ``solve`` picks the solve: ``periodic`` solves the closed
    orbit; ``single_pass`` tracks one pass through the line from
    ``twiss_in`` and serves no tunes or chromaticity. A ``periodic`` model
    whose settings state an ``injection`` is built whether or not its deck
    has a closed orbit: without one its monitors and optics read NaN and its
    ``no_closed_orbit`` is true.

    The model drives its own copy of the deck, so the deck a later call loads
    is never the one a model has written to. A key of ``active`` that is a
    model-wide fault is the scenario's physics, and never a variable of the
    model:

    * a ``<magnet>_<field>_sigma`` misaligns every quadrupole, sextupole or
      dipole of the copy, each by its sigma times one standard normal draw
      keyed on the ``misalignment_seed``, the magnet's ``FamName`` and the
      field (``dx``, ``dy`` in metres, ``roll`` in radians), never truncated;
      a magnet's draw is its own whatever the deck's other elements are;
    * ``aperture_elliptical_x`` and ``_y``, and ``aperture_rectangular_x`` and
      ``_y``, are the half-axes in metres of an aperture centred on the axis,
      written on every element of the copy that has a length; the two
      half-axes of a shape are stated together;
    * ``launch_x``, ``launch_px``, ``launch_y``, ``launch_py``,
      ``launch_delta`` and ``launch_ct`` are the vector a shot is launched
      from at the injection marker, each 0 when unstated; they apply to a
      periodic model whose settings state an ``injection`` and stop on any
      other. The response matrix is linearised about the identity launch
      whatever they are;
    * the bunch's shape, centroid jitter, particle count and the charges are
      held for the shot path; a scenario that states ``bunch_particles``
      declares a bunch, and without it a shot tracks one particle. A bunch
      injects ``bunch_charge`` (1 when unstated) and the model holds
      ``stored_charge`` (0 when unstated) at rest, which a bunch's shots
      accumulate.

    Every other key of ``active`` is a fault seed, ``<address>/<field>``,
    written onto its element before the model adopts the variables. The
    model is then reset once -- one batch, one
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
            value, fault name -> seed, and model-wide fault name -> value.

    Returns:
        The model, at the active scenario.

    Raises:
        FacilityBuildError: ``engine-invalid`` for settings :func:`prepare`
            refuses, or for a record naming no element that reads no plane of
            the tunes or the chromaticity.
        FacilityBuildError: ``engine-invalid`` for a turn series
            :func:`start_values` would refuse for its shape or for a model
            with no injection; for a turns-tracked record on a model with no
            turn series, or whose ``value_range`` reaches outside 1 to the
            samples the series hold after the trigger.
        ValueError: an entry cannot be built as a variable, a default or an
            active value lies outside its ``value_range``, a key of
            ``active`` names no setpoint and no fault of the model, a
            model-wide fault is not a finite number inside its range, the
            misalignment seed is not whole, one half-axis of an aperture is
            stated without the other, a
            ``single_pass`` model wires a tune or chromaticity record, or a
            turn series names an axis other than ``x``, ``y`` or ``sum``.
        pydantic.ValidationError: a fault seed lies outside its bounds.
        OrbitSolveError: the deck, or the deck at the active scenario, has no
            stable solve and the settings state no ``injection``, or loses
            the single-pass particle.
    """
    from osprey.simulation.engines.pyat_model import (
        ArmedVariable,
        ArrayReadbackVariable,
        CommandVariable,
        PyATLatticeModel,
        ReadbackVariable,
        SeriesVariable,
        StoredCurrentVariable,
        tune_planes,
    )
    from osprey.simulation.engines.pyat_variables import (
        EV_PER_GEV,
        is_readout_setting_record,
        is_rf_switch_record,
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

    samples = _series_length(
        entries,
        {str(entry["address"]): entry.get("shape") or () for entry in entries},
        deck,
        settings,
    )

    bound = _turns_bound(samples, prepared.injection)

    channels: list[Any] = []
    for entry in entries:
        address = str(entry["address"])
        attribute = field(entry.get("engine"), "attribute")
        deck_property = not _names_element(entry) and attribute in DECK_PROPERTIES
        if _is_global(entry):
            channels.append(_global_variable(entry, deck, bound, setpoints, active))
            continue
        if _is_command(entry):
            limits: dict[str, Any] = (
                {"value_range": tuple(entry["value_range"])} if "value_range" in entry else {}
            )
            channels.append(
                CommandVariable(
                    name=address,
                    command=attribute,
                    default_value=float(entry.get("default") or 0),
                    unit=entry.get("unit"),
                    **limits,
                )
            )
            continue
        if _holds_shot_state(entry) and entry.get("direction") == "read":
            channels.append(ArmedVariable(name=address, unit=entry.get("unit")))
            continue
        if _is_stored_current(entry):
            channels.append(StoredCurrentVariable.from_wiring(entry))
            continue
        if _names_element(entry) and is_series_record(entry):
            channels.append(SeriesVariable.from_wiring(entry, samples))
            continue
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
        if entry.get("direction") == "read" and is_rf_switch_record(entry):
            channels.append(_rf_switch(entry, lattice, deck, readback=True))
            continue
        if is_readout_setting_record(entry):
            default = active.get(address, entry.get("default")) if address in writes else None
            if address in writes and default is None:
                default = 0
            channels.append(_readout_setting(entry, deck, default))
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
        if address in writes and is_rf_switch_record(entry):
            channels.append(_rf_switch(entry, lattice, deck))
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

    physics = _physics(active, writes)
    faults = {
        name: float(value)
        for name, value in active.items()
        if name not in writes and name not in physics
    }
    _write_apertures(lattice, physics)
    built = PyATLatticeModel(
        lattice,
        channels,
        faults=faults,
        element_misalignments=_misalignments(lattice, physics),
        solve=prepared.solve,
        twiss_in=prepared.twiss_in,
        injection=_launched(model, prepared, physics),
        name=model,
        bunch=physics,
    )
    built.reset()
    return built


def _readout_setting(entry: Mapping[str, Any], deck: Deck, default: Any) -> Any:
    """A labelled readout setting booting at index ``default``; its reading for ``None``.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the record when its
            table is not over its label indices, or ``default`` is no index.
    """
    from osprey.simulation.engines.pyat_variables import (
        readout_setting_from_wiring,
        readout_setting_readback_from_wiring,
    )

    try:
        if default is None:
            return readout_setting_readback_from_wiring(entry)
        return readout_setting_from_wiring(entry, default)
    except ValueError as exc:
        raise _stop(
            deck,
            str(entry.get("id") or entry["address"]),
            "wiring",
            str(exc),
            "give the record a calibration table from each label index to its value, "
            "and boot it at one of those indices",
        ) from exc


def _rf_switch(
    entry: Mapping[str, Any], lattice: Any, deck: Deck, *, readback: bool = False
) -> Any:
    """The RF-on switch of a cavity-voltage record, or with ``readback`` its reading.

    Each bound cavity is weighted by the voltage it carries in ``lattice``,
    read here, before any scenario touches it. A scenario never writes the
    switch: it boots on whatever ``active`` names. The reading follows the
    cavity itself rather than a held setpoint.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the record when its
            labels are not two, or a bound cavity carries no voltage, or zero.
    """
    from osprey.simulation.engines.pyat_variables import (
        rf_switch_from_wiring,
        rf_switch_readback_from_wiring,
    )

    voltages = {
        str(element.FamName): float(element.Voltage)
        for element in lattice
        if hasattr(element, "Voltage")
    }
    try:
        build = rf_switch_readback_from_wiring if readback else rf_switch_from_wiring
        return build(entry, voltages)
    except ValueError as exc:
        raise _stop(
            deck,
            str(entry.get("id") or entry["address"]),
            "wiring",
            str(exc),
            "bind the RF-on switch to a cavity carrying its voltage in the deck, "
            "with two labels, off then on",
        ) from exc


def _global_variable(
    entry: Mapping[str, Any],
    deck: Deck,
    bound: int | None,
    setpoints: Mapping[tuple[Any, ...], str],
    active: Mapping[str, Any],
) -> Any:
    """The variable of a global setting's record: the setting, or its readback.

    The turns tracked ranges over whole turns from 1 to ``bound``, or over the
    record's ``value_range`` inside that. Its default is the active value
    when ``active`` names it, else the one :func:`turns_tracked_default`
    gives. A readback follows the setting that drives the same attribute, or
    reads that default when none does.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the record on a model
            whose turns are not tracked (no turn series), or for a
            ``value_range`` reaching outside 1 to ``bound``.
    """
    from osprey.simulation.engines.pyat_model import ReadbackVariable, TurnsTrackedVariable

    address = str(entry["address"])
    record_id = str(entry.get("id") or address)
    if bound is None:
        raise _stop(
            deck,
            record_id,
            "wiring",
            f"{address} is the turns tracked of a model with no turn series",
            "wire the model's turn series, or leave the record unwired",
        )
    default = turns_tracked_default(entry.get("default"), bound)
    if entry.get("direction") != "write":
        source = setpoints.get(_target(entry))
        return ReadbackVariable(
            name=address,
            source=source,
            default_value=None if source is not None else default,
            unit=entry.get("unit"),
        )
    value_range: tuple[float, float] = (1, bound)
    if "value_range" in entry:
        low, high = entry["value_range"]
        if low < 1 or high > bound:
            raise _stop(
                deck,
                record_id,
                "wiring",
                f"{address} ranges over [{low}, {high}]; the model's turn series hold "
                f"{bound} turns after the trigger sample",
                f"give a value_range inside [1, {bound}]",
            )
        value_range = (low, high)
    return TurnsTrackedVariable(
        name=address,
        default_value=float(active.get(address, default)),
        value_range=value_range,
        unit=entry.get("unit"),
    )


def _launched(model: str, prepared: Prepared, physics: Mapping[str, float]) -> Injection | None:
    """The model's injection, launched from the scenario's ``launch_<coordinate>`` faults.

    Raises:
        ValueError: when the scenario states a launch coordinate on a model
            without an injection marker -- a single-pass model, or a periodic
            one whose settings state no ``injection``.
    """
    stated = [name for name in LAUNCH_FAULTS if name in physics]
    injection = prepared.injection
    if not stated:
        return injection
    if injection is None or injection.marker is None:
        raise ValueError(
            f"model-wide fault {stated[0]} is set on model {model}, which launches at no "
            "injection marker; state the launch on a periodic model whose settings "
            "state pyat.injection"
        )
    import numpy as np

    launch = np.array([physics.get(name, 0.0) for name in LAUNCH_FAULTS], dtype=np.float64)
    return replace(injection, launch=launch)


def _physics(active: Mapping[str, Any], writes: Iterable[str]) -> dict[str, float]:
    """The model-wide faults among a model's active writes, each a checked float.

    A key that is also a setpoint address of the model is that setpoint's
    write, and no fault.

    Raises:
        ValueError: naming the fault whose value is not a finite number
            inside its slot's range, or the misalignment seed that is not
            whole.
    """
    taken = set(writes)
    physics: dict[str, float] = {}
    for name, slot in MODEL_WIDE_FAULTS.items():
        if name not in active or name in taken:
            continue
        raw = active[name]
        value = float(raw) if _number(raw) else math.nan
        low, high = slot.value_range or (-math.inf, math.inf)
        if not low <= value <= high:
            raise ValueError(
                f"model-wide fault {name} is {raw!r}; give a number in [{low:g}, {high:g}]"
            )
        if name in WHOLE_FAULTS and value != int(value):
            raise ValueError(f"model-wide fault {name} is {raw!r}; give a whole number")
        physics[name] = value
    return physics


def _magnet_classes() -> dict[str, type]:
    """Each word of :data:`MISALIGNED_MAGNETS` and the pyAT element class it selects."""
    import at

    classes = {"quadrupole": at.Quadrupole, "sextupole": at.Sextupole, "dipole": at.Dipole}
    return {magnet: classes[magnet] for magnet in MISALIGNED_MAGNETS}


def _misalignments(
    elements: Iterable[Any], physics: Mapping[str, float]
) -> dict[str, dict[str, float]] | None:
    """Each selected magnet's misalignment: ``FamName`` -> ``{dx, dy, roll}``.

    A magnet is selected when its class has a non-zero sigma in any field.
    Each value is its sigma times the standard normal draw addressed by the
    misalignment seed, the magnet's ``FamName`` and the field, so it depends
    on no other element and on no order; a field with no sigma is 0. Magnets
    that share a ``FamName`` share one entry.

    Args:
        elements: The deck's elements.
        physics: The model-wide faults, as :func:`_physics` gives them.

    Returns:
        The misalignments; ``None`` when no sigma is non-zero.
    """
    sigmas = {
        magnet: {
            field: physics.get(misalignment_sigma(magnet, field), 0.0)
            for field in MISALIGNMENT_FIELDS
        }
        for magnet in MISALIGNED_MAGNETS
    }
    selected = {magnet: widths for magnet, widths in sigmas.items() if any(widths.values())}
    if not selected:
        return None
    import numpy as np

    from osprey_connectors.simulation.series import channel_key_bytes, keyed_normals

    seed = np.asarray([int(physics.get(MISALIGNMENT_SEED, 0.0))])
    classes = _magnet_classes()
    drawn: dict[str, dict[str, float]] = {}
    for element in elements:
        widths = next(
            (selected[magnet] for magnet in selected if isinstance(element, classes[magnet])),
            None,
        )
        name = str(element.FamName)
        if widths is None or name in drawn:
            continue
        key = channel_key_bytes(name)
        drawn[name] = {
            field: (
                width * float(keyed_normals(key + f":misalignment_{field}".encode(), seed)[0])
                if width
                else 0.0
            )
            for field, width in widths.items()
        }
    return drawn


#: How many repeated magnet names one scenario stop lists before it counts the rest.
_REPEATS_SHOWN = 3


def scenario_stops(deck: Deck, faults: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """What one scenario asks of a model that its deck cannot give.

    A misalignment is drawn per magnet name, so two magnets of one name would
    take one error: a non-zero ``<magnet>_<field>_sigma`` stops when the deck
    holds a magnet of that class under a name it holds more than once. The
    deck is read only when ``faults`` states such a sigma.

    Args:
        deck: The lattice file, one :func:`prepare` accepts.
        faults: The faults one scenario names on the model that are not its
            channels, name -> value.

    Returns:
        One ``(fault name, detail, remedy)`` per magnet class that stops,
        under the first of its sigmas by name; ``detail`` reads on after the
        fault's name. Empty when nothing stops.
    """
    stated: dict[str, str] = {}
    for magnet in MISALIGNED_MAGNETS:
        for name in sorted(misalignment_sigma(magnet, moved) for moved in MISALIGNMENT_FIELDS):
            raw = faults.get(name)
            if _number(raw) and raw != 0:
                stated.setdefault(magnet, name)
    if not stated:
        return []
    loaded = _load(deck)
    classes = _magnet_classes()
    stops: list[tuple[str, str, str]] = []
    for magnet, name in stated.items():
        repeated = sorted(
            (fam_name, len(found))
            for fam_name, found in loaded.indices.items()
            if len(found) > 1
            and any(isinstance(loaded.lattice[index], classes[magnet]) for index in found)
        )
        if not repeated:
            continue
        listing = ", ".join(f"{n} ({count} times)" for n, count in repeated[:_REPEATS_SHOWN])
        if len(repeated) > _REPEATS_SHOWN:
            listing += f" and {len(repeated) - _REPEATS_SHOWN} more"
        stops.append(
            (
                name,
                f"draws one misalignment per {magnet} name, and the deck repeats {listing}",
                f"give every {magnet} a unique name in the deck, or state no {magnet} sigma",
            )
        )
    return stops


def _write_apertures(lattice: Any, physics: Mapping[str, float]) -> None:
    """Write the stated apertures on every element of ``lattice`` that has a length.

    An elliptical aperture is pyAT's ``EApertures``, its two half-axes; a
    rectangular one is ``RApertures``, the four edges of a rectangle centred
    on the axis.

    Raises:
        ValueError: naming the half-axis a shape states without its other.
    """
    import numpy as np

    apertures: dict[str, Any] = {}
    for shape in APERTURE_SHAPES:
        names = [aperture_half_axis(shape, axis) for axis in ("x", "y")]
        stated = [name for name in names if name in physics]
        if not stated:
            continue
        if len(stated) != len(names):
            (missing,) = set(names) - set(stated)
            raise ValueError(
                f"model-wide fault {stated[0]} is stated without {missing}; "
                f"state both half-axes of the {shape} aperture"
            )
        x, y = (physics[name] for name in names)
        if shape == "elliptical":
            apertures["EApertures"] = (x, y)
        else:
            apertures["RApertures"] = (-x, x, -y, y)
    if not apertures:
        return
    for element in lattice:
        if float(getattr(element, "Length", 0.0)) > 0.0:
            for attribute, values in apertures.items():
                setattr(element, attribute, np.array(values, dtype=float))


def _is_monitor_reading(record: Any) -> bool:
    """Whether a record reads an orbit position: an ``axis``, no ``attribute``, no ``series``."""
    if is_series_record(record):
        return False
    block = field(record, "engine")
    return field(block, "attribute") is None and field(block, "axis") is not None


#: The responses :func:`response_matrix` computes, the first one the default.
RESPONSE_KINDS: tuple[str, ...] = ("orbit", "trajectory")


def response_matrix(
    deck: Deck,
    wiring: Iterable[Any],
    settings: Any,
    correctors: list[str],
    step: Mapping[str, float],
    *,
    kind: str = "orbit",
) -> ResponseMatrix:
    """Compute the response of the wired readings to the named correctors.

    Each corrector is stepped by central difference, ``held ± step/2`` in its
    hardware unit about the value the deck holds, each arm converted to the
    deck's attribute through the corrector's calibration and solved on a copy
    of the deck. Each reading is mapped to its record's hardware unit through
    the way back its calibration states, so an entry is hardware reading per
    hardware setpoint. The deck a later call loads is never written to. A
    corrector whose step has no solve has no column; it is named in
    ``unsolved`` and the other correctors are still stepped.

    ``kind="orbit"`` reads the orbit records off the closed orbit of a
    ``periodic`` model, or off one tracked pass from ``twiss_in`` for a
    ``single_pass`` one.

    ``kind="trajectory"`` reads the turn-series records, each through its own
    calibration, off the first turn of one launched particle: on a
    ``periodic`` model the copy is rotated to the ``injection`` marker and the
    particle tracked once round from the launch vector, radiation and cavities
    as the deck has them; on a ``single_pass`` model it is the tracked pass
    from ``twiss_in``, unrotated. A series record of an axis other than ``x``
    or ``y`` has no row. A step that loses the launched particle has no
    solve.

    Args:
        deck: The lattice file.
        wiring: The model's wiring records.
        settings: The model's settings, ``{pyat: {...}}``, or ``None``.
        correctors: The setpoint addresses to step, one column each: in this
            order for ``orbit``; for ``trajectory`` the ones steering ``x``
            in this order, then the ones steering ``y``.
        step: Each corrector's full step width in its hardware unit.
        kind: ``orbit`` or ``trajectory``.

    Returns:
        The :class:`ResponseMatrix`: a monitor no record wires has no row, a
        corrector whose step has no solve has no column and is named in
        ``unsolved``; for ``trajectory`` it carries the ``index`` of every
        row and every corrector asked.

    Raises:
        FacilityBuildError: ``engine-invalid`` for settings :func:`prepare`
            refuses, a monitor or corrector element the deck does not hold
            exactly once, or ``trajectory`` on a ``periodic`` model whose
            settings state no ``injection``.
        ValueError: ``kind`` is neither word; a corrector is not a wired
            setpoint naming an element, is named twice, has no finite non-zero
            step, or for ``trajectory`` steers neither plane; a monitor record
            cannot be built as a reading.
    """
    import numpy as np
    from lume_pyat.simulator import PyATSimulator

    from osprey.simulation.engines.pyat_single_pass import SinglePassSimulator
    from osprey.simulation.engines.pyat_variables import EV_PER_GEV, MonitorVariable

    if kind not in RESPONSE_KINDS:
        raise ValueError(f"response kind is {kind!r}; use {' or '.join(RESPONSE_KINDS)}")
    trajectory = kind == RESPONSE_KINDS[1]
    prepared = prepare(deck, settings)
    loaded = _load(deck)
    periodic = prepared.solve == SOLVES[0]
    injection = prepared.injection
    if trajectory and injection is None:
        raise _stop(
            deck,
            _model_id(deck, None),
            "model",
            "a trajectory response needs settings key pyat.injection, which the model does not "
            "state",
            "state injection with the marker the particle is launched at",
        )
    # A particle launched at the marker meets the deck from there on: the
    # tracked copy starts at it, and every index is counted from it.
    start = injection.index if trajectory and periodic and injection is not None else 0
    lattice = loaded.lattice.deepcopy()
    if start:
        lattice = lattice.rotate(start)
    size = len(lattice)
    deck_energy_gev = float(lattice.energy) / EV_PER_GEV
    records = list(wiring)

    def position(element: str, record_id: str) -> int:
        return (_element_index(loaded, deck, element, record_id, "wiring") - start) % size

    readings: list[tuple[tuple[Any, ...], int, MonitorVariable]] = []
    for record in records:
        if trajectory:
            if not is_series_record(record) or field(field(record, "engine"), "axis") not in (
                "x",
                "y",
            ):
                continue
        elif not _is_monitor_reading(record):
            continue
        reading = MonitorVariable.from_wiring(record)
        record_id = str(field(record, "id") or reading.name)
        index = position(reading.element_name, record_id)
        order = (reading.axis, index) if trajectory else (index, reading.axis)
        readings.append((order, index, reading))
    readings.sort(key=lambda item: item[0])
    monitors = [reading for _, _, reading in readings]

    columns = _stepped_correctors(records, correctors, step, deck_energy_gev)
    if trajectory:
        steered = {
            variable.name: _steered_plane(field(record, "engine"))
            for variable, _, record in columns
        }
        for variable, _, _ in columns:
            if steered[variable.name] is None:
                raise ValueError(f"corrector {variable.name} steers neither plane")
        # A stable sort: within a plane the correctors keep the order named.
        columns.sort(key=lambda column: steered[column[0].name] != "x")

    simulator = (
        PyATSimulator(lattice)
        if periodic
        else SinglePassSimulator(lattice, twiss_in=prepared.twiss_in)  # type: ignore[arg-type]
    )
    if trajectory and periodic and injection is not None:
        read = _first_turn_reader(lattice, injection.launch, monitors)
    else:

        def read() -> np.ndarray:
            simulator.solve()
            return np.array([reading._get(simulator) for reading in monitors])

    matrix = np.empty((len(monitors), len(columns)), dtype=np.float64)
    solved: list[int] = []
    unsolved: dict[str, str] = {}
    for column, (variable, width, _) in enumerate(columns):
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
                arms.append(read())
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
    places: dict[str, int] | None = None
    if trajectory:
        places = {reading.name: place for _, place, reading in readings}
        for variable, _, record in columns:
            places[variable.name] = position(
                variable.bindings[0].element_name, str(field(record, "id") or variable.name)
            )
    return ResponseMatrix(
        rows=[reading.name for reading in monitors],
        columns=[columns[column][0].name for column in solved],
        matrix=matrix[:, solved],
        unsolved=unsolved,
        index=places,
    )


def _stepped_correctors(
    records: list[Any], correctors: list[str], step: Mapping[str, float], deck_energy_gev: float
) -> list[tuple[Any, float, Any]]:
    """Each named corrector's setpoint variable, full step width and wiring record."""
    from osprey.simulation.engines.pyat_variables import CalibratedSetpoint, variable_from_wiring

    setpoints = {
        str(field(record, "address")): record
        for record in records
        if field(record, "direction") == "write" and _names_element(record)
    }
    columns: list[tuple[Any, float, Any]] = []
    for address in correctors:
        record = setpoints.get(address)
        variable = (
            None
            if record is None
            else variable_from_wiring(record, deck_energy_gev=deck_energy_gev)
        )
        if not isinstance(variable, CalibratedSetpoint):
            raise ValueError(f"corrector {address} is not a wired setpoint on an element")
        if any(variable.name == other.name for other, _, _ in columns):
            raise ValueError(f"corrector {address} is named twice")
        width = step.get(address)
        if width is None or not math.isfinite(float(width)) or float(width) == 0.0:
            raise ValueError(f"corrector {address} has step {width!r}; give a finite non-zero step")
        columns.append((variable, float(width), record))
    return columns


def _first_turn_reader(
    lattice: Any, launch: np.ndarray, monitors: list[Any]
) -> Callable[[], np.ndarray]:
    """A reader of each monitor reading on the first turn of a particle launched at the start.

    Args:
        lattice: The lattice to track, as it stands at each call.
        launch: The launch vector, six floats in pyAT's order.
        monitors: The readings, each a monitor element and an axis.

    Returns:
        A function that tracks one turn and answers the readings in hardware
        units, in the order given; it raises ``OrbitSolveError`` saying where
        the particle is lost.
    """
    import at
    import numpy as np

    from osprey.simulation.engines.pyat_single_pass import loss_text

    names = [element.FamName for element in lattice]
    places = sorted({names.index(reading.element_name) for reading in monitors})
    refpts = np.array(places, dtype=np.uint32)
    slots = [(places.index(names.index(reading.element_name)), reading) for reading in monitors]
    coordinates = {"x": 0, "y": 2}

    def read() -> np.ndarray:
        r_in = np.array(launch, dtype=np.float64).reshape(6, 1)
        try:
            r_out, _, data = at.lattice_track(lattice, r_in, nturns=1, refpts=refpts, losses=True)
        except (at.AtError, ValueError) as exc:
            raise OrbitSolveError(f"first-turn track raised {type(exc).__name__}: {exc}") from exc
        lost = loss_text(lattice, data["loss_map"])
        if lost is not None:
            raise OrbitSolveError(lost)
        return np.array(
            [
                to_hardware(
                    reading.calibration,
                    reading.inverse,
                    float(r_out[coordinates[reading.axis], 0, slot, 0]),
                )
                for slot, reading in slots
            ]
        )

    return read


#: The pass method of an element that applies its ``KickAngle`` and no polynomial.
_KICK_PASS = "CorrectorPass"

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


def condition(model: Any) -> tuple[str, frozenset[str]] | None:
    """The degraded state a built model is in, and the values that state voids.

    A degraded model is built and serving, and some of what it serves has no
    meaning. This engine knows one such state, ``no closed orbit``: a
    ``periodic`` model whose settings state an ``injection`` keeps a write
    that leaves its lattice without a closed orbit, and everything read off
    a closed orbit is then NaN.

    The answer is one read of the model's flag. The names depend only on what
    the model declares, so they are worked out the first time a model is
    asked about and are the same set on every later answer; a caller may ask
    once per address.

    Args:
        model: A model :func:`build` returned.

    Returns:
        ``None`` for a model in no such state. Otherwise the text, one of
        :data:`CONDITIONS`, and the names of the model's variables it voids:
        every orbit reading, every readback of the tunes or the chromaticity,
        and the optics arrays (tunes, chromaticity, beta and orbit at the
        monitors). A setpoint, a readback of a setpoint, a fault, a turn
        series, the armed flag, the stored current and a command are never
        among them: each holds what was written, tracked or stored, closed
        orbit or none.
    """
    if not getattr(model, "no_closed_orbit", False):
        return None
    names = _CLOSED_ORBIT_NAMES.get(model)
    if names is None:
        names = _CLOSED_ORBIT_NAMES[model] = _closed_orbit_names(model)
    return NO_CLOSED_ORBIT, names


def _closed_orbit_names(model: Any) -> frozenset[str]:
    """The names of a model's variables whose values are read off its closed orbit."""
    from osprey.simulation.engines.pyat_model import (
        OPTICS_NAMES,
        PLANE_OPTICS,
        ArrayReadbackVariable,
        ReadbackVariable,
    )
    from osprey.simulation.engines.pyat_variables import MonitorVariable

    names: set[str] = set()
    for name, variable in model.supported_variables.items():
        if isinstance(variable, (MonitorVariable, ArrayReadbackVariable)):
            names.add(name)
        elif isinstance(variable, ReadbackVariable):
            if variable.source in PLANE_OPTICS:
                names.add(name)
        elif name in OPTICS_NAMES:
            names.add(name)
    return frozenset(names)
