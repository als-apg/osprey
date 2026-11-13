"""The injection engine plug-in: a shot-based injection chain with no deck.

The plug-in wraps :class:`~osprey.simulation.engines.injection_model.InjectionModel`
for the ``osprey.simulation.engines`` group. A model using it names no deck;
its wiring records and its settings are the whole chain. Each record's engine
block names a ``signal``:

* ``request`` -- the setpoint whose client write starts one shot;
* ``parameter`` -- a setpoint of the chain, its ``quantity`` one of ``charge``
  (the source charge per shot), ``efficiency`` (the efficiency of the stage at
  index ``stage``) or ``current`` (the stored current), its optional ``start``
  the value it starts at;
* ``counter``, ``charge`` and ``efficiency`` -- readings a shot changes at
  ``offset_s`` seconds after its request, the last two showing the stage at
  index ``stage``;
* ``current`` -- the stored current, raised by the shot at ``offset_s``;
* ``current_fast`` -- the stored current again, read at the fast period; a
  ``current`` or ``current_fast`` may state a ``scale``, the factor from the
  stored current's unit to its own (a second monitor reading in another unit);
* ``busy`` -- the free-running cycle signal.

The settings sit under the key ``injection``: ``cycle: {period_s, width_s}``,
the ordered list of ``stages`` and ``current_per_charge``. Every address,
offset, period and coefficient is facility data; this module names none.

The module states:

* :func:`locate` and :func:`prepare` -- refusals: there is no deck to read;
* :func:`describe` -- a record's engine_role and refresh from its ``signal``;
* :func:`start_values` -- each setpoint's start value, with no deck;
* :func:`build` -- the :class:`InjectionModel` serving one model's wiring;
* :func:`advance`, :func:`client_write`, :func:`next_deadline` and
  :func:`emitted` -- the time contract, each delegating to the model;
* :data:`deckless_start_values` -- ``True``.

Per-record defects raise :class:`ValueError` from :func:`describe`;
defects across records or in the settings raise
:class:`~osprey.facility.errors.FacilityBuildError` naming the model from
:func:`start_values`, and again from :func:`build`. The model module imports
LUME, so it is imported inside :func:`build` alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeGuard

from osprey.facility.errors import FacilityBuildError
from osprey.simulation.engines.calibration import field

if TYPE_CHECKING:
    from osprey.simulation.engines.injection_model import InjectionModel

__all__ = [
    "advance",
    "build",
    "client_write",
    "deckless_start_values",
    "describe",
    "emitted",
    "locate",
    "next_deadline",
    "prepare",
    "start_values",
]

#: ``start_values`` may be called with no deck.
deckless_start_values = True

ENGINE = "injection"

REQUEST = "request"
PARAMETER = "parameter"
COUNTER = "counter"
CHARGE = "charge"
EFFICIENCY = "efficiency"
CURRENT = "current"
CURRENT_FAST = "current_fast"
BUSY = "busy"

#: The quantities a ``parameter`` sets.
QUANTITIES = (CHARGE, EFFICIENCY, CURRENT)

#: The keys each signal's engine block may carry beside ``signal``.
_KEYS: dict[str, frozenset[str]] = {
    REQUEST: frozenset(),
    PARAMETER: frozenset({"quantity", "stage", "start"}),
    COUNTER: frozenset({"offset_s"}),
    CHARGE: frozenset({"offset_s", "stage"}),
    EFFICIENCY: frozenset({"offset_s", "stage"}),
    CURRENT: frozenset({"offset_s", "scale"}),
    CURRENT_FAST: frozenset({"scale"}),
    BUSY: frozenset(),
}
_STEPS = (COUNTER, CHARGE, EFFICIENCY, CURRENT)
_STAGED_STEPS = (CHARGE, EFFICIENCY)
_SETPOINTS = (REQUEST, PARAMETER)
_REFRESH = {CURRENT: "pass", CURRENT_FAST: "fast", REQUEST: "pass", PARAMETER: "pass"}
_EVENT = "event"

_SETTINGS_KEYS = frozenset({"cycle", "stages", "current_per_charge"})
_CYCLE_KEYS = frozenset({"period_s", "width_s"})
_SOURCES = ("models.yaml",)
_ZERO = 0.0
_NOTHING_ACTIVE: Mapping[str, Any] = {}


@dataclass(frozen=True)
class _Entry:
    """One wiring record, read."""

    address: str
    signal: str
    offset_s: float | None = None
    stage: int | None = None
    quantity: str | None = None
    start: float | None = None
    scale: float | None = None
    default: Any = None


@dataclass(frozen=True)
class _Chain:
    """The model's wiring and settings, resolved into the model's arguments."""

    request: str | None
    source_charge: str
    efficiencies: tuple[str, ...]
    stored_current: str
    steps: tuple[_Entry, ...]
    currents: tuple[str, ...]
    scales: Mapping[str, float]
    busy: tuple[str, ...]
    period_s: float
    width_s: float
    current_per_charge: float


# -- locate and prepare --------------------------------------------------------------


def locate(deck: Any, element: str, *, model: str | None = None) -> tuple[float, float]:  # noqa: ARG001
    """Refuse: the injection engine reads no deck, so it locates no element.

    Raises:
        FacilityBuildError: Always, ``engine-invalid`` on the model.
    """
    raise _model_stop(
        model or ENGINE,
        f"the injection engine reads no deck, so it locates no element {element}",
        "wire the chain by engine blocks naming a signal, not by element",
    )


def prepare(deck: Any, settings: Any, *, model: str | None = None) -> Any:  # noqa: ARG001
    """Refuse: the injection engine reads no deck, so it has none to prepare.

    A deckless model is never prepared; this is reached only by a model that
    names a deck.

    Raises:
        FacilityBuildError: Always, ``engine-invalid`` on the model.
    """
    raise _model_stop(
        model or ENGINE,
        "the injection engine reads no deck, so it prepares none",
        "drop the model's deck",
    )


# -- describe ------------------------------------------------------------------------


def describe(wiring_record: Any) -> Mapping[str, Any]:
    """Say what a wiring record is, from the ``signal`` of its engine block.

    ``request`` and ``parameter`` are ``setpoint`` records refreshed by
    ``pass``; every other signal is an ``output``: ``counter``, ``charge``,
    ``efficiency`` and ``busy`` refresh by ``event``, ``current`` by
    ``pass`` and ``current_fast`` by ``fast``. No record has a plane. The
    ``request`` carries the ``command`` mark: a write to it asks for one shot.

    Args:
        wiring_record: One wiring record, read by key or by attribute.

    Returns:
        ``{role, plane, refresh, kind}``, the ``kind`` the signal, or the
        quantity of a ``parameter``; the ``request`` also carries ``marks``.

    Raises:
        ValueError: Naming the record's address, for an engine block that
            names no known signal, a step without an ``offset_s`` or with a
            negative one, a ``charge`` or ``efficiency`` without a ``stage``,
            a ``parameter`` without a known ``quantity``, a non-numeric
            ``start`` or ``scale``, or a key its signal does not take.
    """
    entry = _read(wiring_record)
    role = "setpoint" if entry.signal in _SETPOINTS else "output"
    description = {
        "engine_role": role,
        "plane": None,
        "refresh": _REFRESH.get(entry.signal, _EVENT),
        "kind": entry.quantity or entry.signal,
    }
    return {**description, "marks": ["command"]} if entry.signal == REQUEST else description


# -- start values --------------------------------------------------------------------


def start_values(
    deck: Any,
    wiring: Any,
    settings: Any,
    *,
    readbacks: Mapping[str, str | None] | None = None,  # noqa: ARG001
    value_types: Mapping[str, str] | None = None,  # noqa: ARG001
    shapes: Mapping[str, Any] | None = None,  # noqa: ARG001
) -> Mapping[str, Any]:
    """Each setpoint's start value: a ``parameter``'s ``start`` (zero when absent).

    The request starts at zero. Readings have no start value here. The
    keyword arguments are the contract's and are not read.

    Args:
        deck: ``None``; the engine reads no deck.
        wiring: The model's wiring records, each read by key or by attribute.
        settings: The model's settings, the chain under ``injection``.

    Returns:
        ``{address: value}`` for every ``request`` and ``parameter`` record.

    Raises:
        FacilityBuildError: ``engine-invalid``: on a record, for a defect
            :func:`describe` refuses; on the model, for a deck, more than one
            request or current step, a missing or malformed ``cycle``,
            ``stages`` or ``current_per_charge``, a stage out of range, or a
            parameter missing or doubled.
    """
    records = list(wiring or ())
    name = _model_name(records)
    entries = []
    for record in records:
        try:
            entries.append(_read(record))
        except ValueError as error:
            raise FacilityBuildError(
                "engine-invalid",
                str(field(record, "id") or name),
                _SOURCES,
                "give the record an engine block the injection engine describes",
                record_kind="wiring",
                detail=str(error),
            ) from None
    _chain(name, entries, deck, settings)
    return {
        entry.address: entry.start if entry.start is not None else _ZERO
        for entry in entries
        if entry.signal in _SETPOINTS
    }


# -- build ---------------------------------------------------------------------------


def build(
    model: str,
    wiring: Iterable[Any],
    deck: Any,
    settings: Any,
    active: Mapping[str, Any] = _NOTHING_ACTIVE,
) -> InjectionModel:
    """Build the injection chain serving one model's wiring.

    Every record becomes one variable named by its address; setpoints are
    writable, readings read-only. A setpoint starts at ``active[address]``
    when ``active`` names it, else at the record's ``default``, else at its
    engine block's ``start``, else at zero; a reading starts at its record's
    ``default``, else at zero, and a current reading always shows the stored
    current. The model's time is anchored by the caller's first
    :func:`advance`.

    Args:
        model: The model's name.
        wiring: The model's wiring records, each read by key or by attribute.
        deck: ``None``; the engine reads no deck.
        settings: The model's settings, the chain under ``injection``.
        active: Start values by setpoint address.

    Returns:
        The :class:`InjectionModel`, at its start values.

    Raises:
        ValueError: A record :func:`describe` refuses, or a key of
            ``active`` that names no setpoint of the wiring.
        FacilityBuildError: Each defect :func:`start_values` refuses on the
            model.
    """
    from osprey.simulation.engines.injection_model import InjectionModel, Step

    records = list(wiring)
    entries = [_read(record) for record in records]
    chain = _chain(model, entries, deck, settings)
    setpoints = {entry.address for entry in entries if entry.signal in _SETPOINTS}
    stray = sorted(str(key) for key in active if key not in setpoints)
    if stray:
        raise ValueError(f"a key of active names no setpoint of {model}: {', '.join(stray)}")
    start: dict[str, float] = {}
    for entry in entries:
        if entry.address in active:
            start[entry.address] = float(active[entry.address])
        elif _is_number(entry.default):
            start[entry.address] = float(entry.default)
        elif entry.start is not None:
            start[entry.address] = entry.start
    steps = [
        Step(entry.address, entry.signal, float(entry.offset_s or _ZERO), entry.stage)
        for entry in chain.steps
    ]
    return InjectionModel(
        request=chain.request,
        source_charge=chain.source_charge,
        efficiencies=chain.efficiencies,
        stored_current=chain.stored_current,
        steps=steps,
        currents=chain.currents,
        scales=chain.scales,
        busy=chain.busy,
        period_s=chain.period_s,
        width_s=chain.width_s,
        current_per_charge=chain.current_per_charge,
        start=start,
    )


# -- the time contract ---------------------------------------------------------------


def advance(model: InjectionModel, t_s: float) -> None:
    """Apply every cycle edge and step due by epoch second ``t_s``."""
    model.advance(t_s)


def client_write(model: InjectionModel, values: Mapping[str, Any], t_s: float) -> None:
    """Tell the chain a client wrote ``values`` at ``t_s``; a request starts a shot."""
    model.client_write(values, t_s)


def next_deadline(model: InjectionModel) -> float | None:
    """The epoch second of the chain's next edge or step, or ``None``."""
    return model.next_deadline()


def emitted(model: InjectionModel) -> list[str]:
    """The event channels changed since the last call, cleared."""
    return model.emitted()


# -- reading records -----------------------------------------------------------------


def _read(record: Any) -> _Entry:
    """One record's engine block, checked on its own.

    Raises:
        ValueError: Naming the address, for each defect :func:`describe` lists.
    """
    address = str(field(record, "address"))
    block = field(record, "engine")
    if not isinstance(block, Mapping):
        raise ValueError(f"{address}: names no engine block with a signal")
    signal = block.get("signal")
    if signal not in _KEYS:
        known = ", ".join(_KEYS)
        raise ValueError(f"{address}: signal {signal!r} is not one of {known}")
    extra = sorted(str(key) for key in block if key != "signal" and key not in _KEYS[signal])
    if extra:
        raise ValueError(f"{address}: a {signal} takes no {', '.join(extra)}")
    offset = block.get("offset_s")
    stage = block.get("stage")
    quantity = block.get("quantity")
    start = block.get("start")
    if signal in _STEPS:
        if not _is_number(offset):
            raise ValueError(f"{address}: a {signal} step needs a numeric offset_s")
        if offset < 0:
            raise ValueError(f"{address}: a step offset_s cannot be negative ({offset})")
    staged = signal in _STAGED_STEPS or (signal == PARAMETER and quantity == EFFICIENCY)
    if signal == PARAMETER and quantity not in QUANTITIES:
        raise ValueError(
            f"{address}: a parameter needs a quantity of {', '.join(QUANTITIES)}, not {quantity!r}"
        )
    if staged and not (isinstance(stage, int) and not isinstance(stage, bool) and stage >= 0):
        raise ValueError(f"{address}: a {quantity or signal} needs a stage index, not {stage!r}")
    if not staged and stage is not None:
        raise ValueError(f"{address}: a {quantity or signal} takes no stage")
    if start is not None and not _is_number(start):
        raise ValueError(f"{address}: start must be a number, not {start!r}")
    scale = block.get("scale")
    if scale is not None and not _is_number(scale):
        raise ValueError(f"{address}: scale must be a number, not {scale!r}")
    return _Entry(
        address=address,
        signal=signal,
        offset_s=float(offset) if offset is not None else None,
        stage=stage,
        quantity=quantity,
        start=float(start) if start is not None else None,
        scale=float(scale) if scale is not None else None,
        default=field(record, "default"),
    )


def _chain(model: str, entries: Sequence[_Entry], deck: Any, settings: Any) -> _Chain:
    """The model's chain, checked across records and against its settings.

    Raises:
        FacilityBuildError: ``engine-invalid`` on the model, for each defect
            :func:`start_values` lists on the model.
    """
    if deck is not None:
        raise _model_stop(
            model, f"the injection engine reads no deck, not {deck}", "drop the model's deck"
        )
    period, width, stages, per_charge = _settings(model, settings)

    def stop(detail: str, remedy: str) -> FacilityBuildError:
        return _model_stop(model, detail, remedy)

    requests = [e.address for e in entries if e.signal == REQUEST]
    if len(requests) > 1:
        raise stop(
            f"more than one request: {', '.join(requests)}", "wire one request record at most"
        )
    currents = [e.address for e in entries if e.signal == CURRENT]
    if len(currents) > 1:
        raise stop(
            f"more than one current step: {', '.join(currents)}",
            "wire one current record; read the stored current again as current_fast",
        )
    for entry in entries:
        if entry.stage is not None and entry.stage >= stages:
            raise stop(
                f"{entry.address} names stage {entry.stage}; the settings state {stages}",
                "name a stage index into the settings' stages",
            )
    parameters = [e for e in entries if e.signal == PARAMETER]

    def one(quantity: str, stage: int | None = None) -> str:
        found = [p.address for p in parameters if p.quantity == quantity and p.stage == stage]
        what = f"{quantity} parameter" + (f" for stage {stage}" if stage is not None else "")
        if len(found) != 1:
            said = ", ".join(found) if found else "none"
            raise stop(f"needs one {what}, has {said}", f"wire exactly one {what}")
        return found[0]

    return _Chain(
        request=requests[0] if requests else None,
        source_charge=one(CHARGE),
        efficiencies=tuple(one(EFFICIENCY, stage) for stage in range(stages)),
        stored_current=one(CURRENT),
        steps=tuple(e for e in entries if e.signal in _STEPS),
        currents=tuple(e.address for e in entries if e.signal in (CURRENT, CURRENT_FAST)),
        scales={e.address: e.scale for e in entries if e.scale is not None},
        busy=tuple(e.address for e in entries if e.signal == BUSY),
        period_s=period,
        width_s=width,
        current_per_charge=per_charge,
    )


def _settings(model: str, settings: Any) -> tuple[float, float, int, float]:
    """``(period_s, width_s, number of stages, current_per_charge)`` from the settings."""

    def stop(detail: str) -> FacilityBuildError:
        return _model_stop(
            model,
            detail,
            "state settings injection: {cycle: {period_s, width_s}, stages, current_per_charge}",
        )

    block = field(settings, ENGINE)
    if block is None:
        raise stop(f"the settings carry no key {ENGINE}")
    if not isinstance(block, Mapping):
        raise stop(f"settings key {ENGINE} is not a map")
    unknown = sorted(str(key) for key in block if key not in _SETTINGS_KEYS)
    if unknown:
        raise stop(f"settings key {ENGINE} carries unknown keys {', '.join(unknown)}")
    cycle = block.get("cycle")
    if not isinstance(cycle, Mapping):
        raise stop(f"settings key {ENGINE} states no cycle map")
    unknown = sorted(str(key) for key in cycle if key not in _CYCLE_KEYS)
    if unknown:
        raise stop(f"the cycle carries unknown keys {', '.join(unknown)}")
    period, width = cycle.get("period_s"), cycle.get("width_s")
    if not (_is_number(period) and period > 0):
        raise stop(f"the cycle needs a positive period_s, not {period!r}")
    if not (_is_number(width) and 0 < width < period):
        raise stop(f"the cycle needs a width_s inside its period {period}, not {width!r}")
    stages = block.get("stages")
    if isinstance(stages, str | bytes) or not isinstance(stages, Sequence) or not stages:
        raise stop(f"settings key {ENGINE} needs a non-empty list of stages, not {stages!r}")
    per_charge = block.get("current_per_charge")
    if not _is_number(per_charge):
        raise stop(f"settings key {ENGINE} needs a numeric current_per_charge, not {per_charge!r}")
    return float(period), float(width), len(stages), float(per_charge)


def _model_name(records: Sequence[Any]) -> str:
    """The model a wiring belongs to, from its first record's ``<model>/<address>`` id."""
    for record in records:
        record_id, address = field(record, "id"), field(record, "address")
        if isinstance(record_id, str) and isinstance(address, str):
            suffix = f"/{address}"
            if record_id.endswith(suffix) and len(record_id) > len(suffix):
                return record_id[: -len(suffix)]
    return ENGINE


def _model_stop(model: str, detail: str, remedy: str) -> FacilityBuildError:
    return FacilityBuildError(
        "engine-invalid", model, _SOURCES, remedy, record_kind="model", detail=detail
    )


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)
