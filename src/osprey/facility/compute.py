"""Stage S6: positions, spans, places, ordinals and the checks that need a deck.

The stage runs after S5 and after the wiring slots are filled, and writes the
computed device slots in place on the document:

* ``model``, ``s``, ``length``: for a wired device, from the elements its
  wiring names, located in its model's deck by the engine's ``locate``. A
  periodic model takes the shortest cyclic arc over the elements (s = the
  arc's first entrance, length = (exit of its last - s) mod the deck's
  length); a ``single_pass`` model takes the lowest entrance and the highest
  exit. A source may state them for an unwired device; a wired device's
  stated slot must agree with the computed one (``place-conflict``), and an
  agreeing stated value stands as stated. On a periodic deck of length L a
  stated s is read modulo L (L + d is the same point as d), and the device is
  placed, compared and numbered at that point, and the build warns once per
  such device (``place-wrapped``), naming the stated s, L and the point; on a
  single-pass deck a stated s outside [0, L] stops (``place-conflict``).
* ``place``: when no source states one, the deepest span of the device's model
  that contains its s, computed or stated; ``provenance.place_from`` says
  where the place came from.
* ``ordinalInPlace``, ``ordinalInModel``: per (class, place) and per
  (class, model), in s order with ties by id, a periodic model's s read
  modulo its length; a device with an ``s`` and no ``model`` is numbered in
  its place only.
* ``groups``: the groups naming the device as a member.

It also stops on what only a deck can show: a span whose markers do not
resolve or that overlaps another at its level (``span-invalid``), an imported
place that contradicts its span, a stated position that contradicts the deck
or lies outside a single-pass deck (``place-conflict``), a device or address
wired by two models or a wired element repeated in its deck
(``wiring-conflict``), a declared ``texture`` model or a channel on a status
address (``model-conflict``), a nominal outside its limits band
(``seed-invalid``), an unseeded setpoint whose limits band excludes 0
(``seed-missing``), and a scenario fault keyed by a name that is neither a
channel nor a variable of the model's engine (``engine-invalid``); a wired
element missing from its deck is ``engine-invalid`` naming the wiring record.
``texture`` is then listed last among the models.

A span is half open, ``[from_marker, to_marker)``; with no ``to_marker`` it
runs to the end of the deck, and on a periodic model it may wrap past the end.
A span places only devices whose s came from the span's model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from osprey.facility import TEXTURE
from osprey.facility.errors import FacilityBuildError, FacilityBuildWarning
from osprey.facility.provenance import add_defaults, set_place_from
from osprey.facility.scenarios import (
    check_scenario_attachments,
    check_scenario_engines,
    check_scenario_events,
)
from osprey.facility.sources import AUTHORED
from osprey.facility.validate import Validated, need, paired_nominal_error, stating_files
from osprey.facility.wiring import element_stop

__all__ = [
    "TEXTURE",
    "check_compute",
    "compute_ordinals",
    "resolve_groups",
]

_MODELS_FILE = "models.yaml"
_LIMITS_FILE = "limits.yaml"
_PERIODIC = "periodic"


@dataclass(frozen=True)
class _Deck:
    """One deck-bearing model, prepared by its engine."""

    name: str
    path: Path
    engine: Any
    periodic: bool
    length_m: float


@dataclass(frozen=True)
class _Span:
    place: str
    model: str
    start: float
    end: float
    wraps: bool

    @property
    def depth(self) -> int:
        return self.place.count("/") + 1

    def pieces(self, length_m: float) -> list[tuple[float, float]]:
        """The span as half-open intervals inside [0, length_m)."""
        if self.wraps:
            return [(self.start, length_m), (0.0, self.end)]
        return [(self.start, self.end)]

    def contains(self, s: float) -> bool:
        if self.wraps:
            return s >= self.start or s < self.end
        return self.start <= s < self.end


@dataclass
class _Position:
    model: str
    s: float
    length: float


@dataclass
class _Run:
    """The document being computed and the stops and warnings found so far."""

    document: dict[str, Any]
    facility_dir: Path
    fixes: list[Mapping[str, Any]]
    errors: list[FacilityBuildError] = field(default_factory=list)
    warnings: list[FacilityBuildWarning] = field(default_factory=list)

    def stop(
        self,
        kind: str,
        record_kind: str,
        record_id: str,
        files: Sequence[str],
        detail: str,
        remedy: str,
    ) -> None:
        self.errors.append(
            FacilityBuildError(
                kind, record_id, files, remedy, record_kind=record_kind, detail=detail
            )
        )

    def wrapped(self, device: Mapping[str, Any], deck: _Deck, stated: float) -> None:
        """Warn that a periodic deck places a device's stated s modulo its length."""
        point = stated % deck.length_m
        self.warnings.append(
            FacilityBuildWarning(
                "place-wrapped",
                "device",
                device["id"],
                f"layer {_stating_layers(device, 's')} states s {stated:.9g} in periodic model "
                f"{deck.name}, outside its deck of length {deck.length_m:.9g}; the device is "
                f"placed at s {point:.9g}",
                f"state s {point:.9g}, or drop it",
            )
        )


def check_compute(validated: Validated) -> list[FacilityBuildError]:
    """Compute positions, places, ordinals and groups, and run the deck checks (S6).

    Every computed slot is written in place on ``validated.document``; the
    stage runs after the wiring slots are filled, so each wired record
    carries its ``default``. A stated s a periodic deck wraps is appended to
    ``validated.warnings`` (``place-wrapped``), one per device.

    Args:
        validated: What the earlier stages produced.

    Returns:
        Every stop of the stage; empty when the document is complete.

    Raises:
        RuntimeError: The stage ran before S2 produced the document.
    """
    document: dict[str, Any] = need(validated.document)
    raw_fixes = validated.sources.fixes if validated.sources is not None else None
    entries = raw_fixes.get("fixes") if isinstance(raw_fixes, dict) else None
    run = _Run(
        document,
        validated.facility_dir,
        [e for e in entries or [] if isinstance(e, dict)],
    )
    try:
        return _compute(run, validated)
    finally:
        validated.warnings.extend(run.warnings)


def _compute(run: _Run, validated: Validated) -> list[FacilityBuildError]:
    document = run.document
    _model_conflicts(run)
    twice = _addresses_wired_twice(run)
    decks = _prepare_decks(run)
    computed = _positions(run, decks, twice)
    positions = {**_stated_positions(run, decks, computed), **computed}
    spans = _spans(run, decks)
    if run.errors:
        return run.errors
    _agree(run, computed, decks)
    _places(run, positions, spans)
    _nominal_band(run)
    run.errors.extend(check_scenario_engines(document))
    run.errors.extend(check_scenario_attachments(document, validated.facility_dir))
    run.errors.extend(check_scenario_events(document))
    if run.errors:
        return run.errors
    _write_positions(document, computed)
    _write_ordinals(document, {d.name: d.length_m for d in decks.values() if d.periodic})
    _write_groups(document)
    _list_texture(document)
    return []


# --- models --------------------------------------------------------------------------


def _model_conflicts(run: _Run) -> None:
    models = run.document.get("models", [])
    for model in models:
        if str(model["name"]).casefold() == TEXTURE:
            run.stop(
                "model-conflict",
                "model",
                model["name"],
                stating_files(model, None, _MODELS_FILE),
                f"a source declares model {model['name']}, which the build provides",
                f"remove model {model['name']}; every channel no model wires is {TEXTURE}'s",
            )
    code = run.document["identity"]["code"]
    status = {
        f"{code}:SIM:{model['name']}:STATUS": model["name"]
        for model in models
        if str(model["name"]).casefold() != TEXTURE
    }
    for channel in run.document.get("channels", []):
        name = status.get(channel["id"])
        if name is not None:
            run.stop(
                "model-conflict",
                "channel",
                channel["id"],
                stating_files(channel, None, _MODELS_FILE),
                f"the address is model {name}'s status channel",
                "rename the channel; the simulator serves the status address itself",
            )


def _addresses_wired_twice(run: _Run) -> set[str]:
    """Stop on each address more than one wiring record names; return those addresses."""
    wired: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in run.document.get("models", []):
        for record in model.get("wiring", []):
            wired[record["address"]].append(record)
    for address, records in sorted(wired.items()):
        if len(records) < 2:
            continue
        names = sorted(str(record["id"]).partition("/")[0] for record in records)
        run.stop(
            "wiring-conflict",
            "channel",
            address,
            sorted({f for record in records for f in stating_files(record, None, _MODELS_FILE)}),
            f"models {', '.join(names)} each wire the address",
            "wire the address in one model",
        )
    return {address for address, records in wired.items() if len(records) > 1}


def _prepare_decks(run: _Run) -> dict[str, _Deck]:
    from osprey.simulation.engines import ENTRY_POINT_GROUP

    engines = metadata.entry_points(group=ENTRY_POINT_GROUP)
    decks: dict[str, _Deck] = {}
    for model in run.document.get("models", []):
        if "deck" not in model:
            continue
        engine = engines[model["engine"]].load()
        path = run.facility_dir / model["deck"]
        try:
            prepared = engine.prepare(path, model.get("settings"), model=model["name"])
        except FacilityBuildError as stop:
            run.errors.append(stop)
            continue
        decks[model["name"]] = _Deck(
            model["name"], path, engine, prepared.solve == _PERIODIC, float(prepared.length_m)
        )
    return decks


# --- positions -----------------------------------------------------------------------


def _positions(run: _Run, decks: Mapping[str, _Deck], twice: set[str]) -> dict[str, _Position]:
    """Each wired device's model, s and length.

    A record whose address is wired twice places no device: its address has
    already stopped the build with its own line.
    """
    channels = {c["id"]: c for c in run.document.get("channels", [])}
    located: dict[tuple[str, str], tuple[float, float] | None] = {}
    elements: dict[str, dict[str, dict[str, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for model in run.document.get("models", []):
        deck = decks.get(model["name"])
        if deck is None:
            continue
        for record in model.get("wiring", []):
            if record["address"] in twice:
                continue
            for element, device in _wired_elements(record, channels.get(record["address"])):
                key = (deck.name, element)
                if key not in located:
                    located[key] = _locate(run, deck, record, element)
                where = located[key]
                if where is not None and device is not None:
                    elements[device][deck.name][element] = where
    positions: dict[str, _Position] = {}
    devices = {d["id"]: d for d in run.document.get("devices", [])}
    for device_id, by_model in sorted(elements.items()):
        if len(by_model) > 1:
            run.stop(
                "wiring-conflict",
                "device",
                device_id,
                stating_files(devices.get(device_id, {}), None, _MODELS_FILE),
                f"models {', '.join(sorted(by_model))} each wire an element of the device",
                "wire the device's elements in one model",
            )
            continue
        ((name, found),) = by_model.items()
        deck = decks[name]
        pieces = [found[element] for element in sorted(found, key=lambda e: (found[e], e))]
        s, length = _arc(pieces, deck.length_m) if deck.periodic else _stretch(pieces)
        positions[device_id] = _Position(name, s, length)
    return positions


def _wired_elements(
    record: Mapping[str, Any], channel: Mapping[str, Any] | None
) -> Iterator[tuple[str, str | None]]:
    """Each element a wiring record names, with the device it belongs to."""
    on = channel.get("on") if channel is not None else None
    own = on.get("device") if isinstance(on, dict) else None
    if "element" in record:
        yield str(record["element"]), own
    for piece in record.get("slices") or []:
        yield str(piece["element"]), piece.get("device", own)


def _locate(
    run: _Run, deck: _Deck, record: Mapping[str, Any], element: str
) -> tuple[float, float] | None:
    try:
        entrance, length = deck.engine.locate(deck.path, element, model=deck.name)
    except FacilityBuildError as stop:
        run.errors.append(element_stop(stop, record, deck.name) or stop)
        return None
    return float(entrance), float(length)


def _stretch(elements: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """The lowest entrance and the distance to the highest exit."""
    s = min(entrance for entrance, _ in elements)
    return s, max(entrance + length for entrance, length in elements) - s


def _arc(elements: Sequence[tuple[float, float]], length_m: float) -> tuple[float, float]:
    """The shortest cyclic arc over elements sorted by entrance.

    The arc starts at the element after the largest gap between one element's
    exit and the next one's entrance, the gap from the last back to the first
    wrapping through the end of the deck.
    """
    count = len(elements)
    gaps = [elements[i + 1][0] - (elements[i][0] + elements[i][1]) for i in range(count - 1)]
    wrap = elements[0][0] + length_m - (elements[-1][0] + elements[-1][1])
    first = 0
    widest = wrap
    for i, gap in enumerate(gaps):
        if gap > widest:
            first, widest = i + 1, gap
    s = elements[first][0]
    if first == 0:
        return s, (elements[-1][0] + elements[-1][1]) - s
    last = elements[first - 1]
    return s, (last[0] + last[1] - s) % length_m


#: How far a stated position may lie from the computed one: a deck and an
#: export of the same lattice agree to float rounding.
_POSITION_TOLERANCE_M = 1e-6


def _outside(s: float, deck: _Deck) -> bool:
    """Whether s lies outside [0, L] of the deck, beyond the position tolerance."""
    return not -_POSITION_TOLERANCE_M <= s <= deck.length_m + _POSITION_TOLERANCE_M


def _stating_layers(device: Mapping[str, Any], slot: str) -> str:
    """The layers whose sources state ``slot`` of a device, joined."""
    return ", ".join(
        sorted(
            str(source["layer"])
            for source in device.get("provenance", {}).get("sources", [])
            if slot in source.get("fields", ())
        )
    )


def _stated_positions(
    run: _Run, decks: Mapping[str, _Deck], computed: Mapping[str, _Position]
) -> dict[str, _Position]:
    """Each unwired device that states ``s`` in a deck-bearing ``model``.

    A periodic deck's s is cyclic, so a stated s is placed modulo the deck's
    length, with a ``place-wrapped`` warning when it lay outside the deck; on a
    single-pass deck an s outside the deck stops.
    """
    positions: dict[str, _Position] = {}
    for device in run.document.get("devices", []):
        deck = decks.get(str(device.get("model")))
        if device["id"] in computed or "s" not in device or deck is None:
            continue
        s = float(device["s"])
        if deck.periodic:
            if _outside(s, deck):
                run.wrapped(device, deck, s)
            s %= deck.length_m
        elif _outside(s, deck):
            run.stop(
                "place-conflict",
                "device",
                device["id"],
                stating_files(device, "s", _MODELS_FILE),
                f"layer {_stating_layers(device, 's')} states s {s:g} in model {deck.name}, "
                f"outside its deck of length {deck.length_m:g}",
                "state an s inside the deck, or drop it",
            )
            continue
        positions[device["id"]] = _Position(deck.name, s, float(device.get("length", 0.0)))
    return positions


def _agree(run: _Run, computed: Mapping[str, _Position], decks: Mapping[str, _Deck]) -> None:
    """Stop on a wired device whose stated ``model``, ``s`` or ``length`` is not the deck's.

    An agreeing s a periodic deck wraps warns (``place-wrapped``).
    """
    for device in run.document.get("devices", []):
        position = computed.get(device["id"])
        if position is None:
            continue
        deck = decks[position.model]
        for slot in ("model", "s", "length"):
            if slot not in device:
                continue
            stated = device[slot]
            if slot == "model":
                if str(stated) == position.model:
                    continue
                said = f"model {stated}"
                found = f"wires the device in model {position.model}"
            else:
                value = float(getattr(position, slot))
                delta = abs(float(stated) - value)
                if slot == "s" and deck.periodic:
                    delta %= deck.length_m
                    delta = min(delta, deck.length_m - delta)
                if delta <= _POSITION_TOLERANCE_M:
                    if slot == "s" and deck.periodic and _outside(float(stated), deck):
                        run.wrapped(device, deck, float(stated))
                    continue
                said = f"{slot} {float(stated):g} in model {position.model}"
                found = f"puts the device at {slot} {value:g}"
            run.stop(
                "place-conflict",
                "device",
                device["id"],
                stating_files(device, slot, _MODELS_FILE),
                f"layer {_stating_layers(device, slot)} states {said}, but the deck {found}",
                f"drop `{slot}` from the layer, or add a fix `set` of the deck's value",
            )


# --- spans ---------------------------------------------------------------------------


def _spans(run: _Run, decks: Mapping[str, _Deck]) -> list[_Span]:
    spans: list[_Span] = []
    for place in run.document.get("places", []):
        span = place.get("span")
        if not isinstance(span, dict):
            continue
        files = stating_files(place, "span", _MODELS_FILE)
        name = str(span["model"])
        deck = decks.get(name)
        if deck is None:
            if not any(m["name"] == name and "deck" in m for m in run.document["models"]):
                run.stop(
                    "span-invalid",
                    "place",
                    place["id"],
                    files,
                    f"`span.model` {name} has no deck to place the span in",
                    "name a model with a deck, or place the devices by hand",
                )
            continue
        start = _marker(run, deck, place["id"], files, "from_marker", span["from_marker"])
        if "to_marker" in span:
            end = _marker(run, deck, place["id"], files, "to_marker", span["to_marker"])
        else:
            end = deck.length_m
        if start is None or end is None:
            continue
        wraps = end <= start
        if wraps and (not deck.periodic or end == start):
            run.stop(
                "span-invalid",
                "place",
                place["id"],
                files,
                f"`span.to_marker` {span.get('to_marker')} is not after `span.from_marker` "
                f"{span['from_marker']} in model {name}'s deck",
                "name a to_marker after the from_marker",
            )
            continue
        spans.append(_Span(place["id"], name, start, end, wraps))
    _overlaps(run, spans, decks)
    return spans


def _marker(
    run: _Run, deck: _Deck, place_id: str, files: list[str], slot: str, marker: Any
) -> float | None:
    try:
        entrance, _length = deck.engine.locate(deck.path, str(marker), model=deck.name)
    except FacilityBuildError as stop:
        run.stop(
            "span-invalid",
            "place",
            place_id,
            files,
            f"`span.{slot}`: {stop.detail} of model {deck.name}",
            "name a marker that appears once in the deck",
        )
        return None
    return float(entrance)


def _overlaps(run: _Run, spans: Sequence[_Span], decks: Mapping[str, _Deck]) -> None:
    ordered = sorted(spans, key=lambda span: span.place)
    reported: set[str] = set()
    for i, one in enumerate(ordered):
        for other in ordered[i + 1 :]:
            if one.model != other.model or one.depth != other.depth:
                continue
            length_m = decks[one.model].length_m
            if other.place in reported or not _intersect(
                one.pieces(length_m), other.pieces(length_m)
            ):
                continue
            reported.add(other.place)
            run.stop(
                "span-invalid",
                "place",
                other.place,
                stating_files(_place(run, other.place), "span", _MODELS_FILE),
                f"span overlaps place {one.place}'s span in model {one.model}",
                "make the spans of one level disjoint",
            )


def _intersect(ours: Iterable[tuple[float, float]], theirs: Iterable[tuple[float, float]]) -> bool:
    pieces = list(theirs)
    return any(a < d and c < b for a, b in ours for c, d in pieces if a < b and c < d)


def _place(run: _Run, place_id: str) -> Mapping[str, Any]:
    return next(p for p in run.document["places"] if p["id"] == place_id)


# --- places --------------------------------------------------------------------------


def _places(run: _Run, positions: Mapping[str, _Position], spans: Sequence[_Span]) -> None:
    fixed = {
        str(entry.get("id"))
        for entry in run.fixes
        if entry.get("kind") == "device"
        and (
            (entry.get("op") == "set" and "place" in (entry.get("fields") or {}))
            or (entry.get("op") == "add" and "place" in (entry.get("record") or {}))
        )
    }
    for device in run.document.get("devices", []):
        position = positions.get(device["id"])
        span = _deepest(spans, position) if position is not None else None
        stated = device.get("place")
        origin = _origin(device, fixed)
        if stated is None:
            if span is not None:
                device["place"] = span.place
                _place_from(device, "span")
            continue
        if span is not None and (span.place == stated or span.place.startswith(f"{stated}/")):
            device["place"] = span.place
            _place_from(device, origin if span.place == stated and origin else "span")
            continue
        if origin is not None:
            _place_from(device, origin)
            continue
        if span is not None and position is not None:
            run.stop(
                "place-conflict",
                "device",
                device["id"],
                stating_files(device, "place", _MODELS_FILE),
                f"layer {_stating_layers(device, 'place')} states place {stated}, but the span of place "
                f"{span.place} holds the device at s {position.s:g} in model {position.model}",
                f"drop `place` from the layer, or add a fix `set` of place {stated}",
            )


def _deepest(spans: Iterable[_Span], position: _Position) -> _Span | None:
    holding = [s for s in spans if s.model == position.model and s.contains(position.s)]
    return max(holding, key=lambda span: (span.depth, span.place), default=None)


def _origin(device: Mapping[str, Any], fixed: set[str]) -> str | None:
    """``fix`` or ``authored`` for a place that is itself the decision, else ``None``."""
    if device["id"] in fixed:
        return "fix"
    for source in device.get("provenance", {}).get("sources", []):
        if source.get("layer") == AUTHORED and "place" in source.get("fields", ()):
            return "authored"
    return None


def _place_from(device: dict[str, Any], place_from: str) -> None:
    device["provenance"] = set_place_from(device.get("provenance", {}), place_from)


# --- nominal band --------------------------------------------------------------------


def _nominal_band(run: _Run) -> None:
    from osprey_connectors.simulation.values import coerce, zero

    channels = {c["id"]: c for c in run.document.get("channels", [])}
    defaults: dict[str, tuple[Any, dict[str, Any]]] = {}
    for model in run.document.get("models", []):
        for record in model.get("wiring", []):
            if "default" in record:
                defaults[record["address"]] = (record["default"], record)
    setpoint_of = {
        str(c["pair"]): c["id"]
        for c in channels.values()
        if c.get("role") == "setpoint" and c.get("pair", c["id"]) != c["id"]
    }

    def seed(address: str) -> dict[str, Any]:
        value = channels[address].get("simulation")
        return value if isinstance(value, dict) else {}

    def nominal(address: str) -> tuple[Any, list[str]]:
        channel = channels[address]
        if address in defaults:
            value, record = defaults[address]
            return value, stating_files(record, None, _MODELS_FILE)
        own = seed(address)
        if "nominal" in own:
            value_type = channel.get("value_type")
            coerced = coerce(
                own["nominal"], value_type, channel.get("options"), channel.get("shape")
            )
            return coerced, stating_files(channel, "simulation", _MODELS_FILE)
        setpoint = setpoint_of.get(address)
        if setpoint is not None:
            return nominal(setpoint)
        value = zero(channel.get("value_type"), channel.get("options"), channel.get("shape"))
        return value, stating_files(channel, None, _MODELS_FILE)

    for readback, setpoint in sorted(setpoint_of.items()):
        own = seed(readback)
        if readback in defaults or setpoint not in defaults or "nominal" not in own:
            continue
        mine, _stating = nominal(readback)
        theirs = defaults[setpoint][0]
        if mine != theirs:
            run.errors.append(
                paired_nominal_error(
                    readback,
                    own["nominal"],
                    setpoint,
                    theirs,
                    stating_files(channels[readback], "simulation", _MODELS_FILE),
                    derived="pair" in channels[setpoint].get("provenance", {}).get("defaults", []),
                )
            )

    limits = run.document.get("limits")
    records = limits.get("records") if isinstance(limits, dict) else None
    for limit in sorted(records or [], key=lambda r: str(r.get("address"))):
        address = str(limit["address"])
        low, high = limit.get("min_value"), limit.get("max_value")
        if low is None or high is None or "linear" in seed(address):
            continue
        value, files = nominal(address)
        if (
            channels[address].get("role") == "setpoint"
            and not low <= 0 <= high
            and address not in defaults
            and "nominal" not in seed(address)
        ):
            run.stop(
                "seed-missing",
                "channel",
                address,
                sorted({*files, _LIMITS_FILE}),
                f"limits band [{low:g}, {high:g}] excludes 0 and the channel has no seed",
                "add a nominal for it in data/facility/seeds.yaml",
            )
            continue
        if value < low:
            side, bound = "`min_value`", low
        elif value > high:
            side, bound = "`max_value`", high
        else:
            continue
        run.stop(
            "seed-invalid",
            "channel",
            address,
            sorted({*files, _LIMITS_FILE}),
            f"nominal {value:g} lies {'below' if value < low else 'above'} {side} {bound:g}",
            "move the operating point inside [min_value, max_value], or widen the limits record",
        )


# --- writes --------------------------------------------------------------------------


def _write_positions(document: dict[str, Any], positions: Mapping[str, _Position]) -> None:
    """Write each wired device's computed slots that no source states."""
    for device in document.get("devices", []):
        position = positions.get(device["id"])
        if position is None:
            continue
        values = {"model": position.model, "s": position.s, "length": position.length}
        unstated = {slot: value for slot, value in values.items() if slot not in device}
        device.update(unstated)
        device["provenance"] = add_defaults(device.get("provenance", {}), unstated)


def _write_ordinals(document: dict[str, Any], periodic: Mapping[str, float]) -> None:
    devices = document.get("devices", [])
    for device_id, ordinals in compute_ordinals(devices, periodic=periodic).items():
        device = next(d for d in document["devices"] if d["id"] == device_id)
        device.update(ordinals)
        device["provenance"] = add_defaults(device.get("provenance", {}), ordinals)


def _write_groups(document: dict[str, Any]) -> None:
    membership = resolve_groups(document.get("groups", []))
    for device in document.get("devices", []):
        groups = membership.get(device["id"])
        if groups:
            device["groups"] = groups
            device["provenance"] = add_defaults(device.get("provenance", {}), ("groups",))


def _list_texture(document: dict[str, Any]) -> None:
    models = [m for m in document.get("models", []) if m["name"] != TEXTURE]
    models.sort(key=lambda model: model["name"])
    document["models"] = [*models, {"name": TEXTURE, "engine": TEXTURE}]


# --- lookups derived from the records ------------------------------------------------


def compute_ordinals(
    devices: Iterable[Mapping[str, Any]], *, periodic: Mapping[str, float] | None = None
) -> dict[str, dict[str, int]]:
    """Number each positioned device among its class, per place and per model.

    Only devices with a ``class`` and an ``s`` are numbered, from 1 in s order
    with ties broken by id. ``ordinalInPlace`` counts within (class, place)
    and is given only to a device with a place; ``ordinalInModel`` counts
    within (class, model) and is given only to a device with a model. No
    ordinal compares positions of two models. On a periodic model the s is
    read modulo the deck's length, so a stated s past the end is numbered
    where it is placed.

    Args:
        devices: The device records, with ``model`` and ``s`` computed or stated.
        periodic: The deck length of each periodic model, by model name.

    Returns:
        ``{device id: {ordinalInPlace?, ordinalInModel}}`` for each numbered
        device.
    """
    by_place: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    by_model: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for device in devices:
        if "s" not in device or "class" not in device:
            continue
        s = float(device["s"])
        length_m = (periodic or {}).get(str(device.get("model")))
        if length_m:
            s %= length_m
        key = (s, str(device["id"]))
        if "model" in device:
            by_model[(str(device["class"]), str(device["model"]))].append(key)
        if "place" in device:
            by_place[(str(device["class"]), str(device["place"]))].append(key)
    ordinals: dict[str, dict[str, int]] = defaultdict(dict)
    for slot, table in (("ordinalInPlace", by_place), ("ordinalInModel", by_model)):
        for members in table.values():
            for ordinal, (_s, device_id) in enumerate(sorted(members), start=1):
                ordinals[device_id][slot] = ordinal
    return {device_id: dict(sorted(slots.items())) for device_id, slots in sorted(ordinals.items())}


def resolve_groups(groups: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Invert group membership: each device's groups, sorted.

    Members are device ids; no name is resolved.

    Args:
        groups: The group records.

    Returns:
        ``{device id: [group id, ...]}`` for each device some group names.
    """
    membership: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        for member in group.get("members") or []:
            membership[str(member)].add(str(group["id"]))
    return {device: sorted(ids) for device, ids in sorted(membership.items())}
