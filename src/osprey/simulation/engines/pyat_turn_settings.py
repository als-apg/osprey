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
* ``aperture`` -- the ring's aperture, written on every element with a length:
  its ``shape`` (``elliptical``, the default, or ``rectangular``) and its
  half-axes ``x`` and ``y``, either of which a rectangle may state as a
  ``[low, high]`` pair of limits instead; none when unstated;
* ``apertures`` -- the apertures of single elements, a list read in order.
  Each item states an aperture as ``aperture`` does and selects its elements
  by exactly one of ``class`` (a pyAT element class; its elements with a
  length), ``element`` (a deck element name; every element so named) and ``s``
  (a ``[from, to]`` range in metres from the deck's first element; the
  elements with a length whose midpoint lies in it). An element takes the
  last item that selects it, and the ring's aperture where none does;
* ``pickup`` -- the button pickup of every monitor of the ring, in
  :mod:`~osprey.simulation.engines.pyat_pickup`'s words: a ``round`` pipe's
  radius, button angles and button width, or a ``table`` of button responses,
  with the buttons each plane adds and subtracts. None when unstated: a
  monitor then reads the beam position as it is;
* ``pickups`` -- the pickups of single monitors, a list read in order. Each
  item states a pickup as ``pickup`` does and names its monitors' deck
  elements under ``elements``. A monitor takes the last item that names it,
  and the ring's pickup where none does. A table's path is relative to the
  directory of the model's deck, and a served view carries the table beside
  its copy of the deck (:func:`served_settings`);
* ``sum_per_charge``, ``sum_per_stored``, ``sum_floor``, ``position_noise``,
  ``sum_noise``, ``floor_noise``, ``tone_period``, ``tone_amplitude`` -- the
  monitor readout, in :class:`~osprey.simulation.engines.pyat_turn_readout.TurnReadout`'s
  words.

A wiring record's engine block names one of :data:`SIGNALS` by ``signal``.
Such a record is served as the engine holds it -- a turn array's positions in
metres, its sum in the block's sum units -- so it states no ``calibration``;
an input's engine block names one of :data:`INPUT_SIGNALS`. The
:data:`BEAM_SIGNALS` are the injected beam's settings a client or a scenario
moves: ``offset_<coordinate>`` is added to the launch, and ``mismatch_x`` and
``mismatch_y`` replace the block's factors. A :data:`PULSED_KICK` record is
the setting of a pulsed magnet: its engine block states the ``plane`` (``x`` or
``y``) beside the signal, and may state a ``scale``, the radians one unit of
the record is worth (1 by default), and a ``pulse``, the share of the kick the
bunch meets on each turn from the first (``[1.0]`` by default, one turn). A
shot is kicked at the record's element by what the record holds; the closed
orbit is not. This module reads
plain data and imports neither pyAT nor, until a block is normalised, numpy:
the caller looks the injection element and the ``apertures`` selectors up in
its deck.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey_connectors.simulation.engines import field
from osprey_facility.errors import FacilityBuildError

if TYPE_CHECKING:
    from osprey.simulation.engines.pyat_pickup import Pickup
    from osprey.simulation.engines.pyat_turn_readout import TurnReadout

__all__ = [
    "APERTURE_SELECTORS",
    "APERTURE_SHAPES",
    "ARRAY_SIGNALS",
    "Aperture",
    "BEAM_SIGNALS",
    "BLOCK_KEY",
    "ELEMENTS_KEY",
    "ElementAperture",
    "INPUT_SIGNALS",
    "LAUNCH_KEYS",
    "MISMATCH_SIGNALS",
    "MONITOR_SIGNALS",
    "OFFSET_SIGNALS",
    "PICKUPS_KEY",
    "PICKUP_KEY",
    "PULSED_KICK",
    "RING_SIGNALS",
    "SETTING_SIGNALS",
    "SHOT_SIGNALS",
    "SIGNALS",
    "InjectedTwiss",
    "TurnSettings",
    "describe_input",
    "describe_signal",
    "input_signals",
    "is_turn_by_turn",
    "kick_words",
    "normalise",
    "served_settings",
    "signal_of",
    "start_value",
    "stated_buttons",
    "stated_counts",
]

#: The key of the turn-by-turn block in the ``pyat`` settings.
BLOCK_KEY = "turn_by_turn"

#: The turn-by-turn block's key for the pickup of every monitor of the ring.
PICKUP_KEY = "pickup"
#: Its key for the pickups of single monitors, a list read in order.
PICKUPS_KEY = "pickups"
#: The key of a list item that names the item's monitor elements.
ELEMENTS_KEY = "elements"

#: The launch offsets' keys, in the order pyAT tracks them.
LAUNCH_KEYS: tuple[str, ...] = ("x", "px", "y", "py", "delta", "ct")

#: The aperture shapes, the first one the default; each is ``write_apertures``' keyword.
APERTURE_SHAPES: tuple[str, ...] = ("elliptical", "rectangular")

#: The keys an ``apertures`` item selects its elements by; it states exactly one.
APERTURE_SELECTORS: tuple[str, ...] = ("class", "element", "s")

#: The keys that state an aperture, on the ring's and on an item alike.
_APERTURE_KEYS: tuple[str, ...] = ("x", "y", "shape")

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
        "apertures",
        PICKUP_KEY,
        PICKUPS_KEY,
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
#: The signal of a pulsed magnet's setting: a kick on a shot's first turns.
PULSED_KICK = "pulsed_kick"
#: The words a :data:`PULSED_KICK` record's engine block may state beside the signal.
KICK_WORDS: tuple[str, ...] = ("plane", "scale", "pulse")
#: The ring's read-outs of its last shot: the turns it was tracked for, the
#: turns the beam survived, the monitors the beam reached on the first turn,
#: and how the shot ended.
SHOT_SIGNALS: tuple[str, ...] = (
    "turns_tracked",
    "turns_survived",
    "monitors_reached",
    "shot_status",
)
#: The signals of the whole ring, whose records name no element.
RING_SIGNALS: frozenset[str] = frozenset({*SHOT_SIGNALS, "rf_switch", "pilot_tone", *BEAM_SIGNALS})
#: The signals written by a client: each a setpoint.
SETTING_SIGNALS: frozenset[str] = frozenset(
    {"arm", "rf_switch", "pilot_tone", "stored", PULSED_KICK, *BEAM_SIGNALS}
)

#: Every wiring signal with its start value; an array starts as zeros.
SIGNALS: dict[str, float] = {
    "arm": 0.0,
    "armed": 0.0,
    "turn_x": 0.0,
    "turn_y": 0.0,
    "turn_sum": 0.0,
    **dict.fromkeys(SHOT_SIGNALS, 0.0),
    "rf_switch": 1.0,
    "pilot_tone": 0.0,
    "stored": 0.0,
    **dict.fromkeys(OFFSET_SIGNALS, 0.0),
    **dict.fromkeys(MISMATCH_SIGNALS, 1.0),
    PULSED_KICK: 0.0,
}

#: The signals an input's engine block may name.
INPUT_SIGNALS: tuple[str, ...] = ("shot", "charge", "stored_current")
#: The input signals required once any input is declared.
REQUIRED_INPUTS: tuple[str, ...] = ("charge", "shot")


#: One plane of an aperture, metres: a half-axis, or a rectangle's ``(low, high)`` limits.
Extent = float | tuple[float, float]


@dataclass(frozen=True)
class Aperture:
    """An aperture: its shape and, per plane, its half-axis or a rectangle's limits, metres."""

    shape: str
    x: Extent
    y: Extent

    @property
    def keywords(self) -> dict[str, tuple[Extent, Extent]]:
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
class ElementAperture:
    """One ``apertures`` item: an aperture and the deck indexes of the elements it selects."""

    aperture: Aperture
    indexes: tuple[int, ...]


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
        aperture: The ring's aperture, or ``None`` for none.
        apertures: The apertures of single elements, in the order stated.
        readout: The monitor readout.
        twiss: The optics the bunch arrives with.
        mismatch: The factor on the arriving beta function in ``x`` and
            ``y``; 1 is the stated optics.
        pickup: The button pickup of every monitor, or ``None`` for none.
        pickups: The pickups of single monitors, each with its deck
            element's name, in the order stated.
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
    apertures: tuple[ElementAperture, ...]
    readout: TurnReadout
    twiss: InjectedTwiss = InjectedTwiss()
    mismatch: tuple[float, float] = (1.0, 1.0)
    pickup: Pickup | None = None
    pickups: tuple[tuple[str, Pickup], ...] = ()

    def pickup_of(self, monitor: str) -> Pickup | None:
        """The pickup of the monitor at deck element ``monitor``.

        Returns:
            The pickup of the last ``pickups`` item that names it, else the
            ring's, else ``None``.
        """
        for element, pickup in reversed(self.pickups):
            if element == monitor:
                return pickup
        return self.pickup

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


def served_settings(settings: Any, carry: Callable[[str], str]) -> Any:
    """A model's settings as a served view holds them: each table under the name it is carried by.

    A served view holds a copy of the deck and not the directory the deck
    came from, so a file the settings name beside the deck has to travel with
    it. Every ``table`` a pickup statement names is handed to ``carry``, and
    the statement in the answer names what ``carry`` returns.

    Args:
        settings: The model's settings, as authored. They are not changed.
        carry: Takes a table's path as stated, relative to the directory of
            the model's deck, and returns the path the view serves the table
            under, relative to the directory of the view's copy of the deck.

    Returns:
        ``settings`` itself when no pickup names a table, else a copy naming
        the carried tables. A statement that is not a ``table`` with a path
        is left as it is; :func:`normalise` is what refuses a malformed one.
    """
    block = _block(settings)
    if not isinstance(block, Mapping):
        return settings

    def carried(stated: Any) -> Any:
        if (
            isinstance(stated, Mapping)
            and stated.get("shape") == "table"
            and isinstance(stated.get("table"), str)
            and stated["table"]
        ):
            return {**stated, "table": carry(stated["table"])}
        return stated

    changed: dict[str, Any] = {}
    if PICKUP_KEY in block:
        changed[PICKUP_KEY] = carried(block[PICKUP_KEY])
    listed = block.get(PICKUPS_KEY)
    if isinstance(listed, list | tuple):
        changed[PICKUPS_KEY] = [carried(item) for item in listed]
    if all(changed[key] == block[key] for key in changed):
        return settings
    return {**settings, "pyat": {**settings["pyat"], BLOCK_KEY: {**block, **changed}}}


def stated_buttons(stated: Any) -> int | None:
    """How many buttons a pickup statement declares, read without checking the rest.

    A ``round`` pickup has one button per angle of its ``buttons`` list; a
    ``table`` states the count as ``buttons``. ``None`` for a statement that
    declares none, which :func:`~osprey.simulation.engines.pyat_pickup.parse` refuses.
    """
    if not isinstance(stated, Mapping):
        return None
    buttons = stated.get("buttons")
    if stated.get("shape") == "table":
        count = buttons if isinstance(buttons, int) and not isinstance(buttons, bool) else None
    else:
        count = len(buttons) if isinstance(buttons, list | tuple) else None
    return count if count is not None and count >= 2 else None


def stated_counts(settings: Any) -> tuple[int | None, dict[str, int]]:
    """The button counts a turn-by-turn block declares, read without checking the rest.

    Args:
        settings: The model's settings as authored, or anything else.

    Returns:
        ``(ring, by_element)``: the count of the block's :data:`PICKUP_KEY`
        statement, ``None`` where it states none, and the count per monitor
        element its :data:`PICKUPS_KEY` items name, a later item replacing an
        earlier one.
    """
    block = _block(settings)
    if not isinstance(block, Mapping):
        return None, {}
    by_element: dict[str, int] = {}
    listed = block.get(PICKUPS_KEY)
    for item in listed if isinstance(listed, list | tuple) else ():
        count = stated_buttons(item)
        elements = item.get(ELEMENTS_KEY) if isinstance(item, Mapping) else None
        if count is None or not isinstance(elements, list | tuple):
            continue
        by_element.update({element: count for element in elements if isinstance(element, str)})
    return stated_buttons(block.get(PICKUP_KEY)), by_element


def normalise(
    settings: Any,
    deck: str,
    model_id: str,
    index_of: Callable[[str], int],
    select: Callable[[str, Any], tuple[int, ...]],
) -> TurnSettings | None:
    """Check and default a model's turn-by-turn block.

    Args:
        settings: The model's settings.
        deck: The lattice file, which every stop names as its source.
        model_id: The model the stops name.
        index_of: Returns the injection element's deck index, raising when
            the deck does not hold it exactly once.
        select: Returns the deck indexes an ``apertures`` item selects, given
            its selector key and value, raising ``ValueError`` with the reason
            when the deck cannot answer it.

    Returns:
        The block, or ``None`` on a ring that is not a turn-by-turn ring.

    Raises:
        FacilityBuildError: ``engine-invalid`` naming the model for a block
            that is not a map, carries an unknown key, lacks or misstates
            ``injection_element``, states a value of the wrong type or range,
            states an ``apertures`` item that selects no element, or states a
            pickup that
            :func:`~osprey.simulation.engines.pyat_pickup.parse` refuses or
            that names an element the deck does not hold; whatever
            ``index_of`` raises.
    """
    from osprey.simulation.engines import pyat_pickup
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

    def extent(stated: Mapping[str, Any], axis: str, shape: str, name: str) -> Extent:
        value = stated[axis]
        if isinstance(value, list | tuple):
            if shape != "rectangular":
                raise stop(
                    f"{where}.{name}.{axis} states limits on an ellipse",
                    f"give {name}.{axis} as a half-axis, or set shape to rectangular",
                )
            limits = [
                float(limit)
                for limit in value
                if not isinstance(limit, bool)
                and isinstance(limit, int | float)
                and math.isfinite(limit)
            ]
            if len(value) != 2 or len(limits) != 2 or not limits[0] < limits[1]:
                raise stop(
                    f"{where}.{name}.{axis} is {value!r}",
                    f"give {name}.{axis} as a [low, high] pair of limits in metres, low < high",
                )
            return (limits[0], limits[1])
        half_axis = float(number(stated, axis, 0.0, f"{name}.{axis}"))
        if half_axis == 0:
            raise stop(
                f"{where}.{name}.{axis} is {value!r}",
                f"give {name}.{axis} as a positive half-axis in metres",
            )
        return half_axis

    def stated_aperture(stated: Mapping[str, Any], name: str) -> Aperture:
        for axis in ("x", "y"):
            if axis not in stated:
                raise stop(f"{where}.{name} lacks {axis}", "state both x and y")
        shape = stated.get("shape", APERTURE_SHAPES[0])
        if shape not in APERTURE_SHAPES:
            raise stop(
                f"{where}.{name}.shape is {shape!r}",
                f"set {name}.shape to {' or '.join(APERTURE_SHAPES)}",
            )
        return Aperture(shape, extent(stated, "x", shape, name), extent(stated, "y", shape, name))

    def element_aperture(item: Any, name: str) -> ElementAperture:
        if not isinstance(item, Mapping):
            raise stop(
                f"{where}.{name} is not a map",
                f"state each item as a map of one of {', '.join(APERTURE_SELECTORS)} "
                f"with {', '.join(_APERTURE_KEYS)}",
            )
        _known(item, frozenset((*APERTURE_SELECTORS, *_APERTURE_KEYS)), f"{where}.{name}", stop)
        named = [key for key in APERTURE_SELECTORS if key in item]
        if len(named) != 1:
            stated = " and ".join(named) if named else f"none of {', '.join(APERTURE_SELECTORS)}"
            raise stop(
                f"{where}.{name} names {stated}",
                f"select the item's elements by exactly one of {', '.join(APERTURE_SELECTORS)}",
            )
        selector = named[0]
        value = item[selector]
        if selector == "s":
            ends = (
                [
                    end
                    for end in value
                    if not isinstance(end, bool)
                    and isinstance(end, int | float)
                    and math.isfinite(end)
                ]
                if isinstance(value, list | tuple)
                else []
            )
            if len(ends) != 2 or len(value) != 2 or not 0 <= ends[0] < ends[1]:
                raise stop(
                    f"{where}.{name}.s is {value!r}",
                    f"give {name}.s as a [from, to] pair of metres along the deck, 0 <= from < to",
                )
            value = (float(ends[0]), float(ends[1]))
        elif not isinstance(value, str) or not value:
            raise stop(
                f"{where}.{name}.{selector} is {value!r}", f"give {name}.{selector} as a name"
            )
        aperture = stated_aperture(item, name)
        try:
            indexes = select(selector, value)
        except ValueError as error:
            raise stop(
                f"{where}.{name}: {error}",
                "select elements the deck holds, or drop the item",
            ) from error
        return ElementAperture(aperture, indexes)

    aperture = None
    if "aperture" in block:
        aperture = stated_aperture(sub_map("aperture", _APERTURE_KEYS), "aperture")
    items = block.get("apertures", [])
    if not isinstance(items, list | tuple):
        raise stop(f"{where}.apertures is not a list", "state apertures as a list of items")
    apertures = tuple(
        element_aperture(item, f"apertures[{position}]") for position, item in enumerate(items)
    )
    directory = Path(deck).resolve().parent
    pickup = None
    if PICKUP_KEY in block:
        pickup = pyat_pickup.parse(block[PICKUP_KEY], f"{where}.{PICKUP_KEY}", directory, stop)
    listed = block.get(PICKUPS_KEY, [])
    if not isinstance(listed, list | tuple):
        raise stop(f"{where}.pickups is not a list", "state pickups as a list of items")
    pickups: list[tuple[str, Pickup]] = []
    for position, item in enumerate(listed):
        name = f"{where}.{PICKUPS_KEY}[{position}]"
        monitors = item.get(ELEMENTS_KEY) if isinstance(item, Mapping) else None
        if (
            not isinstance(monitors, list | tuple)
            or not monitors
            or not all(isinstance(monitor, str) and monitor for monitor in monitors)
        ):
            raise stop(
                f"{name}.elements is {monitors!r}",
                "state each item as a pickup with elements, the list of its monitors' "
                "deck element names",
            )
        for monitor in monitors:
            try:
                select("element", monitor)
            except ValueError as error:
                raise stop(
                    f"{name}: {error}", "name monitors the deck holds, or drop the item"
                ) from error
        stated = pyat_pickup.parse(item, name, directory, stop, extra_keys=(ELEMENTS_KEY,))
        pickups.extend((monitor, stated) for monitor in monitors)
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
        apertures=apertures,
        readout=TurnReadout(tone_period=tone_period, **readout),
        pickup=pickup,
        pickups=tuple(pickups),
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


def kick_words(record: Any) -> tuple[int, float, tuple[float, ...]]:
    """A :data:`PULSED_KICK` record's plane, scale and pulse, checked and defaulted.

    Returns:
        The plane's index in :data:`PLANES`, the radians per unit of the
        record, and the share of the kick per turn.

    Raises:
        ValueError: naming the address for a plane that is missing or is not
            ``x`` or ``y``, a scale that is not a finite number, or a pulse
            that is not a list of one or more finite numbers.
    """
    address = field(record, "address")
    block = field(record, "engine")
    plane = field(block, "plane")
    if plane is None:
        raise ValueError(f"{address}: signal {PULSED_KICK} states no plane; use x or y")
    if plane not in PLANES:
        raise ValueError(f"{address}: signal {PULSED_KICK} states plane {plane!r}; use x or y")
    scale = field(block, "scale")
    scale = 1.0 if scale is None else scale
    if not _is_number(scale):
        raise ValueError(f"{address}: scale {scale!r} is not a number")
    pulse = field(block, "pulse")
    pulse = [1.0] if pulse is None else pulse
    if not isinstance(pulse, (list, tuple)) or not pulse or not all(map(_is_number, pulse)):
        raise ValueError(f"{address}: pulse {pulse!r} is not a list of numbers")
    return PLANES.index(plane), float(scale), tuple(float(share) for share in pulse)


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a finite number and no boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def describe_signal(record: Any) -> Mapping[str, Any]:
    """Describe a wiring record whose engine block names a ``signal``.

    A setting signal (``arm``, ``rf_switch``, ``pilot_tone``, ``stored``,
    :data:`PULSED_KICK` and the injected beam's :data:`BEAM_SIGNALS`) is a
    ``setpoint`` refreshing
    ``pass``; the rest (``armed``, the turn arrays and the shot read-outs
    :data:`SHOT_SIGNALS`) are ``output`` refreshing ``event``. No signal has a
    plane, so none is taken for an orbit monitor or corrector. The ``kind`` is
    the signal. ``arm`` carries the ``command`` mark: a write to it starts one
    acquisition and holds no value.

    Raises:
        ValueError: naming the address for an unknown signal; any other key
            beside ``signal``, but a pulsed kick's :data:`KICK_WORDS`; a
            ``calibration``, which no turn-by-turn record is read or written
            through; a setting not wired ``direction: write``, or another signal that
            is; a monitor signal or a pulsed kick naming no single
            ``element``; a ring signal naming an element; whatever
            :func:`kick_words` refuses.
    """
    address = field(record, "address")
    signal = signal_of(record)
    if signal not in SIGNALS:
        raise ValueError(
            f"{address} names unknown signal {signal!r}; use one of {', '.join(SIGNALS)}"
        )
    block = field(record, "engine")
    allowed = {"signal", *KICK_WORDS} if signal == PULSED_KICK else {"signal"}
    extra = (
        sorted(str(key) for key in block if key not in allowed)
        if isinstance(block, Mapping)
        else []
    )
    if extra:
        raise ValueError(f"{address} names signal {signal} beside {', '.join(extra)}")
    if field(record, "calibration") is not None:
        hint = "; a monitor's own sum scale is its fault `gain`" if signal == "turn_sum" else ""
        raise ValueError(
            f"{address}: signal {signal} states a calibration, but a turn-by-turn record is "
            f"served as the engine holds it, through no curve; remove the calibration{hint}"
        )
    write = field(record, "direction") == "write"
    if signal in SETTING_SIGNALS and not write:
        raise ValueError(f"{address}: signal {signal} is a setting; wire it with direction write")
    if signal not in SETTING_SIGNALS and write:
        raise ValueError(f"{address}: signal {signal} is read only; wire it as a read")
    element = field(record, "element")
    if signal in MONITOR_SIGNALS and (element is None or field(record, "slices")):
        raise ValueError(f"{address}: signal {signal} names no monitor element")
    if signal == PULSED_KICK:
        if element is None or field(record, "slices"):
            raise ValueError(f"{address}: signal {signal} names no element")
        kick_words(record)
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
