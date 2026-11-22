"""The pyat engine's turn-by-turn vocabulary: its settings block, signals and inputs.

A pyat model whose ``pyat`` settings carry the :data:`BLOCK_KEY` block is a
*turn-by-turn ring*; :func:`is_turn_by_turn` is the one test of that, and a
ring without the block is the ring it always was. The block states, every key
but ``injection_element`` defaulted:

* ``injection_element`` -- the deck element a shot is launched at;
* ``turns`` and ``trigger_sample`` -- a shot tracks ``turns`` turns, and every
  turn array holds ``trigger_sample + turns`` samples, injected turn 1 at index
  ``trigger_sample``;
* ``particles`` -- the particles of one tracked bunch;
* ``launch`` -- the launch offsets, a map of ``x px y py delta ct``;
* ``emittance_x``, ``emittance_y``, ``sigma_delta``, ``sigma_z`` -- the
  bunch's size;
* ``twiss`` -- the optics the bunch arrives with, a map of ``beta``, ``alpha``
  and ``dispersion``; a key left out is the ring's own at the injection;
* ``mismatch`` -- a factor on those beta functions, a map of ``x`` and ``y``;
* ``jitter`` -- the shot-to-shot centroid jitter, a map of ``x px y py delta
  ct``;
* ``aperture`` -- the half-axes ``x`` and ``y`` written on every element with a
  length, and its ``shape`` (``elliptical``, the default, or
  ``rectangular``); none when unstated;
* ``sum_per_charge``, ``sum_per_stored``, ``sum_floor``, ``position_noise``,
  ``sum_noise``, ``floor_noise``, ``tone_period``, ``tone_amplitude`` -- the
  monitor readout, in :class:`~osprey.simulation.engines.pyat_turn_readout.TurnReadout`'s
  words.

A wiring record's engine block names one of :data:`SIGNALS` by ``signal``;
an input's engine block names one of :data:`INPUT_SIGNALS`. The
:data:`BEAM_SIGNALS` are the injected beam's settings a client or a scenario
moves: ``offset_<coordinate>`` is added to the launch, and ``mismatch_x`` and
``mismatch_y`` replace the block's factors. This module reads
plain data and imports neither pyAT nor, until a block is normalised, numpy:
the caller looks the injection element up in its deck.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.facility.errors import FacilityBuildError
from osprey_connectors.simulation.engines import field

if TYPE_CHECKING:
    from osprey.simulation.engines.pyat_turn_readout import TurnReadout

__all__ = [
    "APERTURE_SHAPES",
    "ARRAY_SIGNALS",
    "Aperture",
    "BEAM_SIGNALS",
    "BLOCK_KEY",
    "INPUT_SIGNALS",
    "LAUNCH_KEYS",
    "MISMATCH_SIGNALS",
    "MONITOR_SIGNALS",
    "OFFSET_SIGNALS",
    "RING_SIGNALS",
    "SETTING_SIGNALS",
    "SIGNALS",
    "InjectedTwiss",
    "TurnSettings",
    "describe_input",
    "describe_signal",
    "input_signals",
    "is_turn_by_turn",
    "normalise",
    "signal_of",
    "start_value",
]

#: The key of the turn-by-turn block in the ``pyat`` settings.
BLOCK_KEY = "turn_by_turn"

#: The launch offsets' keys, in the order pyAT tracks them.
LAUNCH_KEYS: tuple[str, ...] = ("x", "px", "y", "py", "delta", "ct")

#: The aperture shapes, the first one the default; each is ``write_apertures``' keyword.
APERTURE_SHAPES: tuple[str, ...] = ("elliptical", "rectangular")

DEFAULT_TURNS = 250
DEFAULT_TRIGGER_SAMPLE = 10
DEFAULT_PARTICLES = 16
#: The centroid jitter's sigma per launch coordinate, in :data:`LAUNCH_KEYS`' order.
DEFAULT_JITTER: tuple[float, float, float, float, float, float] = (5e-5, 0.0, 3e-5, 0.0, 0.0, 0.0)
#: The planes a mismatch factor is stated for.
PLANES: tuple[str, ...] = ("x", "y")
#: The ``twiss`` keys, each with the numbers it holds.
TWISS_KEYS: dict[str, int] = {"beta": 2, "alpha": 2, "dispersion": 4}
DEFAULT_TONE_PERIOD = 10

#: The readout keys with their defaults; ``tone_period`` is a whole number.
_READOUT_DEFAULTS: dict[str, float] = {
    "sum_per_charge": 1.0,
    "sum_per_stored": 1.0,
    "sum_floor": 0.0,
    "position_noise": 1e-5,
    "sum_noise": 0.0,
    "floor_noise": 1e-3,
    "tone_amplitude": 1.0,
}

#: The bunch-size keys, each defaulting to 0.
_SIZE_KEYS: tuple[str, ...] = ("emittance_x", "emittance_y", "sigma_delta", "sigma_z")

#: Every key the block may carry.
KEYS: frozenset[str] = frozenset(
    {
        "injection_element",
        "turns",
        "trigger_sample",
        "particles",
        "launch",
        "twiss",
        "mismatch",
        "jitter",
        "aperture",
        "tone_period",
        *_SIZE_KEYS,
        *_READOUT_DEFAULTS,
    }
)

#: The signals a monitor element's records carry.
MONITOR_SIGNALS: frozenset[str] = frozenset({"arm", "armed", "turn_x", "turn_y", "turn_sum"})
#: The array signals: ``trigger_sample + turns`` samples per monitor.
ARRAY_SIGNALS: frozenset[str] = frozenset({"turn_x", "turn_y", "turn_sum"})
#: The static launch offset's signals, in :data:`LAUNCH_KEYS`' order: each
#: is added to its launch coordinate.
OFFSET_SIGNALS: tuple[str, ...] = tuple(f"offset_{key}" for key in LAUNCH_KEYS)
#: The mismatch factors' signals, in :data:`PLANES`' order.
MISMATCH_SIGNALS: tuple[str, ...] = tuple(f"mismatch_{plane}" for plane in PLANES)
#: The injected beam's settings: every shot reads the values they hold.
BEAM_SIGNALS: frozenset[str] = frozenset({*OFFSET_SIGNALS, *MISMATCH_SIGNALS})
#: The signals of the whole ring, whose records name no element.
RING_SIGNALS: frozenset[str] = frozenset(
    {"turns_tracked", "rf_switch", "pilot_tone", *BEAM_SIGNALS}
)
#: The signals written by a client: each a setpoint.
SETTING_SIGNALS: frozenset[str] = frozenset(
    {"arm", "rf_switch", "pilot_tone", "stored", *BEAM_SIGNALS}
)

#: Every wiring signal with its start value; an array starts as zeros.
SIGNALS: dict[str, float] = {
    "arm": 0.0,
    "armed": 0.0,
    "turn_x": 0.0,
    "turn_y": 0.0,
    "turn_sum": 0.0,
    "turns_tracked": 0.0,
    "rf_switch": 1.0,
    "pilot_tone": 0.0,
    "stored": 0.0,
    **dict.fromkeys(OFFSET_SIGNALS, 0.0),
    **dict.fromkeys(MISMATCH_SIGNALS, 1.0),
}

#: The signals an input's engine block may name.
INPUT_SIGNALS: tuple[str, ...] = ("shot", "charge", "stored_current")
#: The input signals required once any input is declared.
REQUIRED_INPUTS: tuple[str, ...] = ("charge", "shot")


@dataclass(frozen=True)
class Aperture:
    """The aperture written on every element with a length: its shape and half-axes, metres."""

    shape: str
    x: float
    y: float

    @property
    def keywords(self) -> dict[str, tuple[float, float]]:
        """The aperture as :func:`~osprey.simulation.engines.pyat_turns.write_apertures`'s keyword."""
        return {self.shape: (self.x, self.y)}


@dataclass(frozen=True)
class InjectedTwiss:
    """The optics the injected bunch arrives with; ``None`` is the ring's own at the injection.

    Attributes:
        beta: ``(beta_x, beta_y)`` in metres.
        alpha: ``(alpha_x, alpha_y)``.
        dispersion: ``(eta_x, eta_px, eta_y, eta_py)``.
    """

    beta: tuple[float, float] | None = None
    alpha: tuple[float, float] | None = None
    dispersion: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class TurnSettings:
    """A turn-by-turn ring's settings block, checked and defaulted.

    Attributes:
        injection_element: The deck element a shot is launched at.
        injection_index: That element's index in the deck.
        turns: The turns one shot tracks.
        trigger_sample: The samples a turn array holds before turn 1.
        particles: The particles of a tracked bunch.
        launch: Six launch offsets in pyAT's order ``x px y py delta ct``.
        emittance_x: Horizontal emittance, metre-radians.
        emittance_y: Vertical emittance, metre-radians.
        sigma_delta: Relative energy spread.
        sigma_z: Bunch length, metres.
        jitter: The centroid jitter's sigma per launch coordinate, in the
            launch's order and units.
        aperture: The aperture, or ``None`` for none.
        readout: The monitor readout.
        twiss: The optics the bunch arrives with.
        mismatch: The factor on the arriving beta function in ``x`` and
            ``y``; 1 is the stated optics.
    """

    injection_element: str
    injection_index: int
    turns: int
    trigger_sample: int
    particles: int
    launch: tuple[float, float, float, float, float, float]
    emittance_x: float
    emittance_y: float
    sigma_delta: float
    sigma_z: float
    jitter: tuple[float, float, float, float, float, float]
    aperture: Aperture | None
    readout: TurnReadout
    twiss: InjectedTwiss = InjectedTwiss()
    mismatch: tuple[float, float] = (1.0, 1.0)

    @property
    def samples(self) -> int:
        """The samples of one turn array: ``trigger_sample + turns``."""
        return self.trigger_sample + self.turns


def _block(settings: Any) -> Any:
    return field(field(settings, "pyat"), BLOCK_KEY)


def is_turn_by_turn(settings: Any) -> bool:
    """Whether a model's settings make its ring a turn-by-turn ring.

    Args:
        settings: The model's settings, as authored.

    Returns:
        ``True`` when the ``pyat`` block states the :data:`BLOCK_KEY` block.
    """
    return _block(settings) is not None


def normalise(
    settings: Any,
    deck: str,
    model_id: str,
    index_of: Callable[[str], int],
) -> TurnSettings | None:
    """Check and default a model's turn-by-turn block.

    Args:
        settings: The model's settings.
        deck: The lattice file, which every stop names as its source.
        model_id: The model the stops name.
        index_of: Returns the injection element's deck index, raising when
            the deck does not hold it exactly once.

    Returns:
        The block, or ``None`` on a ring that is not a turn-by-turn ring.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the model for a block
            that is not a map, carries an unknown key, lacks or misstates
            ``injection_element``, or states a value of the wrong type or
            range; whatever ``index_of`` raises.
    """
    from osprey.simulation.engines.pyat_turn_readout import TurnReadout

    block = _block(settings)
    if block is None:
        return None
    where = f"settings key pyat.{BLOCK_KEY}"

    def stop(detail: str, remedy: str) -> FacilityBuildError:
        return FacilityBuildError(
            "engine-invalid", model_id, [str(deck)], remedy, record_kind="model", detail=detail
        )

    if not isinstance(block, Mapping):
        raise stop(f"{where} is not a map", f"state {BLOCK_KEY} as a map")
    _known(block, KEYS, where, stop)
    if "injection_element" not in block:
        raise stop(f"{where} lacks injection_element", "name the deck element a shot starts at")
    element = block["injection_element"]
    if not isinstance(element, str) or not element:
        raise stop(
            f"{where}.injection_element is {element!r}", "name the deck element a shot starts at"
        )

    def whole(key: str, default: int, least: int) -> int:
        value = block.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            raise stop(f"{where}.{key} is {value!r}", f"give {key} as a whole number >= {least}")
        return int(value)

    def number(holder: Mapping[str, Any], key: str, default: float, name: str, *, signed=False):
        value = holder.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or (not signed and value < 0)
        ):
            kind = "a finite number" if signed else "a finite non-negative number"
            raise stop(f"{where}.{name} is {value!r}", f"give {name} as {kind}")
        return float(value)

    def sub_map(key: str, keys: Iterable[str]) -> Mapping[str, Any]:
        value = block.get(key, {})
        if not isinstance(value, Mapping):
            raise stop(f"{where}.{key} is not a map", f"state {key} as a map of {', '.join(keys)}")
        _known(value, frozenset(keys), f"{where}.{key}", stop)
        return value

    turns = whole("turns", DEFAULT_TURNS, 1)
    trigger_sample = whole("trigger_sample", DEFAULT_TRIGGER_SAMPLE, 0)
    particles = whole("particles", DEFAULT_PARTICLES, 1)
    tone_period = whole("tone_period", DEFAULT_TONE_PERIOD, 1)
    launch_map = sub_map("launch", LAUNCH_KEYS)
    launch = tuple(
        number(launch_map, key, 0.0, f"launch.{key}", signed=True) for key in LAUNCH_KEYS
    )
    jitter_map = sub_map("jitter", LAUNCH_KEYS)
    jitter = tuple(
        number(jitter_map, key, default, f"jitter.{key}")
        for key, default in zip(LAUNCH_KEYS, DEFAULT_JITTER, strict=True)
    )
    mismatch_map = sub_map("mismatch", PLANES)
    mismatch = tuple(number(mismatch_map, plane, 1.0, f"mismatch.{plane}") for plane in PLANES)
    if 0.0 in mismatch:
        raise stop(
            f"{where}.mismatch is {dict(mismatch_map)!r}",
            "give each mismatch factor as a number above 0; 1 is the stated optics",
        )
    twiss_map = sub_map("twiss", TWISS_KEYS)
    twiss: dict[str, Any] = {}
    for key, length in TWISS_KEYS.items():
        if key not in twiss_map:
            continue
        stated = twiss_map[key]
        if not isinstance(stated, list | tuple) or len(stated) != length:
            raise stop(f"{where}.twiss.{key} is {stated!r}", f"give {key} as {length} numbers")
        numbers = tuple(
            number({key: value}, key, 0.0, f"twiss.{key}", signed=key != "beta") for value in stated
        )
        if key == "beta" and 0.0 in numbers:
            raise stop(
                f"{where}.twiss.beta is {stated!r}", "give each beta function as metres above 0"
            )
        twiss[key] = numbers
    aperture = None
    if "aperture" in block:
        stated = sub_map("aperture", ("x", "y", "shape"))
        for axis in ("x", "y"):
            if axis not in stated:
                raise stop(f"{where}.aperture lacks {axis}", "state both half-axes x and y")
        half_axes = []
        for axis in ("x", "y"):
            value = number(stated, axis, 0.0, f"aperture.{axis}")
            if value == 0:
                raise stop(
                    f"{where}.aperture.{axis} is {stated[axis]!r}",
                    f"give aperture.{axis} as a positive half-axis in metres",
                )
            half_axes.append(value)
        shape = stated.get("shape", APERTURE_SHAPES[0])
        if shape not in APERTURE_SHAPES:
            raise stop(
                f"{where}.aperture.shape is {shape!r}",
                f"set aperture.shape to {' or '.join(APERTURE_SHAPES)}",
            )
        aperture = Aperture(shape, half_axes[0], half_axes[1])
    sizes = {key: number(block, key, 0.0, key) for key in _SIZE_KEYS}
    readout = {key: number(block, key, default, key) for key, default in _READOUT_DEFAULTS.items()}
    return TurnSettings(
        injection_element=element,
        injection_index=index_of(element),
        turns=turns,
        trigger_sample=trigger_sample,
        particles=particles,
        launch=launch,
        jitter=jitter,
        twiss=InjectedTwiss(**twiss),
        mismatch=mismatch,
        aperture=aperture,
        readout=TurnReadout(tone_period=tone_period, **readout),
        **sizes,
    )


def _known(
    block: Mapping[str, Any],
    keys: frozenset[str],
    where: str,
    stop: Callable[[str, str], FacilityBuildError],
) -> None:
    unknown = sorted(str(key) for key in set(block) - keys)
    if unknown:
        raise stop(
            f"{where} carries unknown keys {', '.join(unknown)}",
            f"use only {', '.join(sorted(keys))}",
        )


def signal_of(record: Any) -> str | None:
    """The turn-by-turn signal a wiring record's engine block names, or ``None``."""
    signal = field(field(record, "engine"), "signal")
    return None if signal is None else str(signal)


def describe_signal(record: Any) -> Mapping[str, Any]:
    """Describe a wiring record whose engine block names a ``signal``.

    A setting signal (``arm``, ``rf_switch``, ``pilot_tone``, ``stored`` and
    the injected beam's :data:`BEAM_SIGNALS`) is a ``setpoint`` refreshing
    ``pass``; the rest (``armed``, the turn arrays,
    ``turns_tracked``) are ``output`` refreshing ``event``. No signal has a
    plane, so none is taken for an orbit monitor or corrector. The ``kind`` is
    the signal. ``arm`` carries the ``command`` mark: a write to it starts one
    acquisition and holds no value.

    Raises:
        ValueError: naming the address for an unknown signal; any other key
            beside ``signal``; a setting not wired ``direction: write``, or
            another signal that is; a monitor signal naming no single
            ``element``; a ring signal naming an element.
    """
    address = field(record, "address")
    signal = signal_of(record)
    if signal not in SIGNALS:
        raise ValueError(
            f"{address} names unknown signal {signal!r}; use one of {', '.join(SIGNALS)}"
        )
    block = field(record, "engine")
    extra = (
        sorted(str(key) for key in block if key != "signal") if isinstance(block, Mapping) else []
    )
    if extra:
        raise ValueError(f"{address} names signal {signal} beside {', '.join(extra)}")
    write = field(record, "direction") == "write"
    if signal in SETTING_SIGNALS and not write:
        raise ValueError(f"{address}: signal {signal} is a setting; wire it with direction write")
    if signal not in SETTING_SIGNALS and write:
        raise ValueError(f"{address}: signal {signal} is read only; wire it as a read")
    element = field(record, "element")
    if signal in MONITOR_SIGNALS and (element is None or field(record, "slices")):
        raise ValueError(f"{address}: signal {signal} names no monitor element")
    if signal in RING_SIGNALS and (element is not None or field(record, "slices")):
        raise ValueError(f"{address}: signal {signal} is the ring's and names an element")
    if signal in SETTING_SIGNALS:
        setting = {"engine_role": "setpoint", "plane": None, "refresh": "pass", "kind": signal}
        return {**setting, "marks": ["command"]} if signal == "arm" else setting
    return {"engine_role": "output", "plane": None, "refresh": "event", "kind": signal}


def start_value(signal: str, block: TurnSettings) -> float | list[float]:
    """A signal's start value.

    Zeros of ``block.samples`` for an array, the block's own factor for a
    mismatch, else the signal's scalar.
    """
    if signal in ARRAY_SIGNALS:
        return [0.0] * block.samples
    if signal in MISMATCH_SIGNALS:
        return block.mismatch[MISMATCH_SIGNALS.index(signal)]
    return SIGNALS[signal]


def describe_input(input_record: Any) -> Mapping[str, Any]:
    """Check one input's engine block: exactly a ``signal`` among :data:`INPUT_SIGNALS`.

    Returns:
        ``{signal}``.

    Raises:
        ValueError: naming the input's address otherwise.
    """
    address = field(input_record, "address")
    block = field(input_record, "engine")
    if not isinstance(block, Mapping):
        raise ValueError(f"input {address}: the engine block is not a map; state its signal")
    if "signal" not in block:
        raise ValueError(
            f"input {address}: the engine block states no signal; "
            f"use one of {', '.join(INPUT_SIGNALS)}"
        )
    extra = sorted(str(key) for key in block if key != "signal")
    if extra:
        raise ValueError(
            f"input {address}: the engine block carries unknown keys {', '.join(extra)}"
        )
    signal = block["signal"]
    if signal not in INPUT_SIGNALS:
        raise ValueError(
            f"input {address} names unknown signal {signal!r}; "
            f"use one of {', '.join(INPUT_SIGNALS)}"
        )
    return {"signal": signal}


def input_signals(inputs: Iterable[Any]) -> dict[str, str]:
    """Check a model's whole input list and map each signal to its address.

    Each signal is declared at most once, and ``shot`` and ``charge`` both
    once any input is declared. :func:`describe_input` checks each input
    alone; this is the check across them.

    Raises:
        ValueError: for a malformed input, a signal declared twice, or a
            non-empty list lacking ``shot`` or ``charge``.
    """
    found: dict[str, str] = {}
    for record in inputs:
        signal = describe_input(record)["signal"]
        address = str(field(record, "address"))
        if signal in found:
            raise ValueError(f"signal {signal} is declared twice: {found[signal]}, {address}")
        found[signal] = address
    missing = sorted(set(REQUIRED_INPUTS) - set(found))
    if found and missing:
        raise ValueError(
            f"inputs {', '.join(sorted(found))} lack {', '.join(missing)}; "
            "a turn-by-turn ring takes shot and charge together"
        )
    return dict(sorted(found.items()))
