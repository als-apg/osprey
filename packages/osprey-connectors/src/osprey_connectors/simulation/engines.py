"""The engine plug-in contract: the entry-point group, the protocol, calibration.

Every simulation engine registers a module under :data:`ENGINE_GROUP`, and
this module is the one place the group's name is written. A build reaches an
engine only through its entry point, never by importing the module by name,
and :class:`EnginePlugin` states the contract the module implements: the five
functions every engine provides and the optional names it may add
(:data:`OPTIONAL_NAMES`, among them the :data:`TIME_CONTRACT`).

An engine that cannot answer a request because the facility file it was built
from is wrong raises :class:`FacilityDefectError`; the caller reports the
request as refused by a facility-file defect, not as an engine failure.

Wiring calibrations are the curve between a hardware value and a physics
value. A curve maps a channel's hardware value to the engine's physics value;
its inverse maps back. Either direction is a straight line (:class:`Linear`)
or a sampled table (:class:`Table`). A wiring record states its curves as
``{linear: {gain, offset}}`` or ``{table: {grid, values}}``, read by attribute
or by key; :func:`curve_from_record` turns one into these dataclasses,
:func:`evaluate` applies it and :func:`shifted` moves it along its hardware
axis.

The way back from a physics value to hardware units is one rule,
:func:`to_hardware`: the record's ``inverse`` when it states one, else the
algebraic inverse of a ``linear`` curve. A ``table`` without an ``inverse``,
and a linear gain of 0, have no way back (:class:`NoInverse`).

A physics value stated at the deck energy is worth :func:`energy_factor` of
itself at another energy when it scales with the beam rigidity
(``energy_scaling: brho``).

The module is pure stdlib at import time: it loads neither numpy nor an
engine library, so a reader that only needs the group's name or the
calibration math pays for nothing else.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from lume.model import LUMEModel

#: A value the engine contract leaves to the engine: the deck, settings and
#: wiring an engine reads, and the facility records it is handed. The package
#: names its open values ``osprey_connectors._contract_types.ContractAny``, but
#: this module stays stdlib-only, so its one open value is named here.
_EngineAny: TypeAlias = Any  # type: ignore[explicit-any]

__all__ = [
    "ENGINE_GROUP",
    "OPTIONAL_NAMES",
    "REST_MASS_GEV",
    "RIGIDITY_PER_MOMENTUM",
    "TIME_CONTRACT",
    "Calibration",
    "EnginePlugin",
    "FacilityDefectError",
    "Linear",
    "NoInverse",
    "Table",
    "brho",
    "check_inverse",
    "curve_from_record",
    "energy_factor",
    "evaluate",
    "field",
    "shifted",
    "to_hardware",
    "to_physics",
]

#: The entry-point group every engine plug-in registers under. It is named
#: after this module, which owns it; every reader and every entry-point table
#: takes the name from here.
ENGINE_GROUP = "osprey_connectors.simulation.engines"

#: The optional functions by which time and client writes reach a model.
TIME_CONTRACT = ("advance", "next_deadline", "emitted", "client_write")

#: Every name a plug-in may state beyond :class:`EnginePlugin`'s, each looked
#: up on the module by name; ``deckless_start_values`` and the ``*_ERRORS``
#: tuples are values, the rest are functions.
OPTIONAL_NAMES = (
    *TIME_CONTRACT,
    "deck_length",
    "deckless_start_values",
    "describe_input",
    "element_index",
    "error_text",
    "fault_variables",
    "polynomial_kicks",
    "readout",
    "response_matrix",
    "scenario_stops",
    "take_inputs",
    "REFUSAL_ERRORS",
    "SOLVE_ERRORS",
)


class FacilityDefectError(Exception):
    """A request refused because the facility file it was built from is wrong.

    An engine raises it, or a subclass of it, when a request names something
    the facility file states wrongly or not at all, so the request has no
    answer until the file is fixed. The caller reports the refusal as a
    facility-file defect with this exception's text, never as a failure of the
    engine. The text names the defect and where in the file it sits.
    """


@runtime_checkable
class EnginePlugin(Protocol):
    """The functions an engine plug-in module states.

    A facility build calls the first four without building a model; the
    virtual accelerator calls ``build`` once per served model.
    ``isinstance(module, EnginePlugin)`` checks each is present, not its
    signature.

    A plug-in may state these names too, each looked up on the module by name:

    ======================================  ============================================  ==========================================
    name                                    what it gives                                 a module without it
    ======================================  ============================================  ==========================================
    ``advance(model, t_s)``                 applies every step due by epoch second        the model is never advanced
                                            ``t_s``; the first call after a build or a
                                            reset anchors the model's time
    ``client_write(model, values, t_s)``    tells the engine a client wrote these         a client write starts nothing
                                            setpoints, already forwarded; the one call
                                            that starts what a client write triggers
    ``deck_length(deck, *, model=None)``    the length of the deck's lattice in metres    the SDK's ``model_members`` raises
                                            (a ring's circumference)
    ``deckless_start_values``               ``True``: ``start_values`` may be called      ``start_values`` is not called without a
                                            with no deck                                  deck
    ``describe_input(input_record)``        checks one model input's ``{address,          every input's ``engine`` block is
                                            engine}`` record at facility build;           accepted as given
                                            raising ``ValueError`` refuses it and the
                                            build stops naming the input
    ``element_index(deck, element,``        the element's lattice index: ring order,      the SDK's ``model_members`` raises
    ``*, model=None)``                      which tells co-located elements apart
    ``emitted(model)``                      the addresses of ``event`` channels fired     the model emits nothing
                                            since the last call, cleared as returned
    ``error_text(exc)``                     the one line a failed build reports           the exception's own text
    ``fault_variables(wiring, settings)``   every fault a model built from the wiring     the model declares no faults
                                            and its settings declares, by address or by
                                            element, without a deck load. Raising
                                            ``ValueError`` refuses the model: the build
                                            stops ``engine-invalid`` with its text
    ``next_deadline(model)``                the epoch second of the model's next due      the model has no deadline
                                            step, or ``None``
    ``polynomial_kicks(deck, elements)``    a deck copy carrying named kicks as dipole    the pyAML view stops ``engine-invalid``
                                            polynomials; stand-in until pyAML reads a     when design correctors need it
                                            dipole kick held in ``KickAngle``
    ``readout(model, values, t_ms)``        what each monitor reads of ``values``, its    a monitor reads its beam truth as given
                                            beam truth at epoch millisecond ``t_ms``
    ``response_matrix(...)``                called with ``(deck, wiring, settings,        ``model_response_matrix`` ends without a
                                            correctors, step)`` and the keyword           result; the response check stops
                                            ``kind``, ``"orbit"`` or ``"trajectory"``:
                                            the response of the wired monitors,
                                            ``rows``, ``columns``, ``matrix`` and the
                                            ``unsolved`` correctors with their texts.
                                            ``row_positions`` and ``column_positions``
                                            place each row and column by lattice element
                                            index, counted for ``"trajectory"`` from the
                                            injection element; an answer without them,
                                            or with ``None``, states no positions and
                                            callers get none. For ``"trajectory"`` the
                                            rows are the monitors' turn-buffer records.
                                            One that takes no ``kind`` is called without
                                            it for ``"orbit"``, and any other kind is
                                            refused naming the engine. Raising one of
                                            ``REFUSAL_ERRORS`` refuses the request with
                                            its text; a ``FacilityDefectError``
                                            refuses it as a facility-file defect; a
                                            return with no column is refused with the
                                            ``unsolved`` map
    ``scenario_stops(wiring, faults)``      ``(fault, detail, remedy)`` per refusal of a  no scenario stops
                                            scenario's faults on the model, channel-
                                            keyed ones included, without a deck load;
                                            each stops the build ``engine-invalid``
                                            with the plug-in's remedy
    ``take_inputs(model, inputs, t_s,``     stores ``inputs``: one ``(input_record,       the build refuses a model that declares
    ``*, baseline)``                        value)`` pair per declared input, the         inputs
                                            record its ``{address, engine}`` and the
                                            value the producer's current plain value;
                                            ``baseline=True`` marks a starting point
    ``REFUSAL_ERRORS``                      the exceptions meaning "the request is not    no exception is a request refusal
                                            one this engine answers"; stand-in until
                                            lume-base states it
    ``SOLVE_ERRORS``                        the exceptions meaning "this request has no   every exception is a failure, never a
                                            solve"; stand-in until lume-base states       refusal
                                            ``SolveError``
    ======================================  ============================================  ==========================================

    The four names of :data:`TIME_CONTRACT` make an engine time-driven: a lazy
    state machine with no thread of its own. The same calls and clock readings
    always yield the same values. A model's ``set`` only stores values and
    never starts time-driven behaviour, since restore, rollback, journal replay
    and scenario writes all reach it; what a client write triggers lives behind
    ``client_write`` alone. ``build`` receives no clock, so the caller calls
    ``advance`` right after a build, a rebuild or a reset. The caller advances
    every built child on every pass, whether or not it has a step due. A model
    that declares inputs gets ``take_inputs`` immediately before its
    ``advance``, with the values its producers hold on that same pass.
    ``take_inputs`` only stores, like ``set``; what a new value starts happens
    in the ``advance`` that follows. ``baseline=True`` means the delivery is a
    starting point: compare nothing against what was held before. The
    ``engine`` block travelling with every delivery is how a model tells its
    inputs apart; ``build`` is unchanged. :data:`OPTIONAL_NAMES` lists every
    name of this table.
    """

    def locate(
        self, deck: _EngineAny, element: str, *, model: str | None = None
    ) -> tuple[float, float]:
        """An element's entrance and length along the deck."""
        ...

    def prepare(
        self, deck: _EngineAny, settings: _EngineAny, *, model: str | None = None
    ) -> _EngineAny:
        """The model's settings, checked against the deck and normalised once."""
        ...

    def start_values(
        self,
        deck: _EngineAny,
        wiring: _EngineAny,
        settings: _EngineAny,
        *,
        readbacks: Mapping[str, str | None] | None = None,
        value_types: Mapping[str, str] | None = None,
        shapes: Mapping[str, _EngineAny] | None = None,
    ) -> Mapping[str, _EngineAny]:
        """Each wired channel's operating point, derived from the deck."""
        ...

    def describe(self, wiring_record: _EngineAny) -> Mapping[str, _EngineAny]:
        """What a wiring record is: its engine_role, plane and refresh, and any marks."""
        ...

    def build(
        self,
        model: str,
        wiring: _EngineAny,
        deck: _EngineAny,
        settings: _EngineAny,
        active: Mapping[str, _EngineAny] = ...,
    ) -> LUMEModel:
        """The LUME model that serves one model's wiring over its deck."""
        ...


# -- Calibration -------------------------------------------------------------

#: Electron rest mass in GeV. A deck's energy is kinetic, so the rest mass is
#: what turns it into a momentum.
REST_MASS_GEV: float = 0.51099906e-3

#: Tesla-metres of rigidity per GeV/c of momentum: 1e9 divided by the speed of
#: light in metres per second.
RIGIDITY_PER_MOMENTUM: float = 10.0 / 2.99792458


@dataclass(frozen=True)
class Linear:
    """A straight-line conversion ``y = gain * x + offset``."""

    gain: float
    offset: float


@dataclass(frozen=True)
class Table:
    """A sampled conversion: piecewise linear through ``(grid, values)``.

    ``grid`` is strictly monotonic, so the curve is well defined everywhere;
    beyond either end the consumer extrapolates along the last segment.
    """

    grid: tuple[float, ...]
    values: tuple[float, ...]


#: Either shape a conversion may take.
Calibration = Linear | Table


def field(record: _EngineAny, name: str) -> _EngineAny:
    """Return one field of a record read by key or by attribute.

    Args:
        record: A mapping, an object with attributes, or ``None``.
        name: The field name.

    Returns:
        The field's value, or ``None`` when the record does not carry it.
    """
    if record is None:
        return None
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def curve_from_record(curve: _EngineAny) -> Calibration | None:
    """Turn a facility-file curve into a :data:`Calibration`.

    Args:
        curve: ``{linear: {gain, offset}}`` or ``{table: {grid, values}}``,
            as a mapping or an object with those attributes; ``None`` for no
            curve.

    Returns:
        The curve as :class:`Linear` or :class:`Table`, or ``None`` when
        ``curve`` is ``None`` or states neither shape.
    """
    linear = field(curve, "linear")
    if linear is not None:
        return Linear(gain=float(field(linear, "gain")), offset=float(field(linear, "offset")))
    table = field(curve, "table")
    if table is not None:
        return Table(
            grid=tuple(float(x) for x in field(table, "grid")),
            values=tuple(float(y) for y in field(table, "values")),
        )
    return None


def evaluate(curve: Calibration, x: float) -> float:
    """Apply a curve to one value.

    A table interpolates linearly between its samples and extrapolates along
    its first or last segment beyond the grid.

    Args:
        curve: The conversion to apply.
        x: The value on the curve's input side.

    Returns:
        The value on the curve's output side.
    """
    if isinstance(curve, Linear):
        return curve.gain * x + curve.offset
    grid, values = curve.grid, curve.values
    if grid[0] > grid[-1]:
        grid, values = grid[::-1], values[::-1]
    segment = len(grid) - 2
    for index in range(1, len(grid) - 1):
        if x < grid[index]:
            segment = index - 1
            break
    x0, x1 = grid[segment], grid[segment + 1]
    y0, y1 = values[segment], values[segment + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def shifted(curve: Calibration, by: float, *, on_values: bool = False) -> Calibration:
    """Move a curve along its hardware axis.

    A forward curve (hardware in) answers at ``x + by`` what ``curve``
    answers at ``x``. A way back (hardware out, ``on_values``) answers
    ``by`` more than ``curve`` at every input.

    Args:
        curve: The conversion to move.
        by: How far, in hardware units.
        on_values: Whether the hardware axis is the curve's output.

    Returns:
        The moved curve; ``curve`` itself when ``by`` is 0.
    """
    if by == 0.0:
        return curve
    if isinstance(curve, Linear):
        if on_values:
            return Linear(gain=curve.gain, offset=curve.offset + by)
        return Linear(gain=curve.gain, offset=curve.offset - curve.gain * by)
    if on_values:
        return Table(grid=curve.grid, values=tuple(value + by for value in curve.values))
    return Table(grid=tuple(point + by for point in curve.grid), values=curve.values)


class NoInverse(ValueError):
    """A calibration with no way back from a physics value to hardware units.

    Attributes:
        reason: ``table`` for a sampled curve stating no ``inverse``,
            ``zero-gain`` for a linear curve of gain 0.
    """

    def __init__(self, reason: Literal["table", "zero-gain"]) -> None:
        if reason == "table":
            message = "a table calibration has no inverse"
        else:
            message = "the linear calibration's gain is 0, so it has no inverse"
        super().__init__(message)
        self.reason = reason


def check_inverse(curve: Calibration | None, inverse: Calibration | None) -> None:
    """Refuse a calibration :func:`to_hardware` cannot map back.

    Args:
        curve: Hardware to physics; ``None`` for no calibration.
        inverse: Physics to hardware as the record states it, or ``None``.

    Raises:
        NoInverse: ``inverse`` is ``None`` and ``curve`` is a table or a
            linear curve of gain 0.
    """
    if inverse is not None or curve is None:
        return
    if isinstance(curve, Table):
        raise NoInverse("table")
    if curve.gain == 0.0:
        raise NoInverse("zero-gain")


def to_physics(curve: Calibration | None, hardware: float) -> float:
    """Map a hardware value to the engine's physics value.

    Args:
        curve: Hardware to physics; ``None`` for no calibration, under which
            the hardware value is the physics value.
        hardware: The value in the channel's unit.

    Returns:
        The physics value.
    """
    if curve is None:
        return float(hardware)
    return evaluate(curve, hardware)


def to_hardware(curve: Calibration | None, inverse: Calibration | None, physics: float) -> float:
    """Map a physics value back to hardware units.

    ``inverse`` when present, else the algebraic inverse of a linear
    ``curve``; with neither, the physics value is the hardware value.

    Args:
        curve: Hardware to physics; ``None`` for no calibration.
        inverse: Physics to hardware as the record states it, or ``None``.
        physics: The engine's value.

    Returns:
        The value in the channel's unit.

    Raises:
        NoInverse: ``inverse`` is ``None`` and ``curve`` is a table or a
            linear curve of gain 0.
    """
    check_inverse(curve, inverse)
    if inverse is not None:
        return evaluate(inverse, physics)
    if isinstance(curve, Linear):
        return (physics - curve.offset) / curve.gain
    return float(physics)


def brho(energy_gev: float) -> float:
    """Return the beam rigidity in tesla-metres at a kinetic energy.

    The rest mass enters the momentum twice, so the massless form ``E / c``
    is not used: at a few GeV it is high by parts in ten thousand.

    Args:
        energy_gev: The kinetic energy in GeV.

    Returns:
        The rigidity in tesla-metres.
    """
    total = float(energy_gev) + REST_MASS_GEV
    return RIGIDITY_PER_MOMENTUM * math.sqrt(total**2 - REST_MASS_GEV**2)


def energy_factor(energy_gev: float, deck_energy_gev: float) -> float:
    """Return what a rigidity-scaled physics value is worth at ``energy_gev``.

    A calibration states its physics value at the deck energy. The same
    hardware value bends a stiffer beam less, in the ratio of the two
    rigidities, and the factor is one at the deck energy.

    Args:
        energy_gev: The energy the beam is at, in GeV.
        deck_energy_gev: The energy the calibration was stated at, in GeV.

    Returns:
        ``brho(deck_energy_gev) / brho(energy_gev)``.
    """
    return brho(deck_energy_gev) / brho(energy_gev)
