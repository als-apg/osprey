"""Facility questions the e2e lanes ask of a built render.

A lane asks the built facility for a kind of channel -- a seeded reading, a
bounded writable setpoint, a corrector a limits record bounds -- and never
names an address. Every answer is read from the simulator view the build
rendered: each channel's role, ``writable``, ``value_range`` (set only when its
limits record states both ends), its seed, and the bindings of the served
models. A facility change therefore moves a lane's channel with it instead of
leaving the lane holding a stale address.

A selection is scoped by a stated reference -- a model, a limits record, a
role -- and address or document order only breaks ties inside that scope.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey_connectors.simulation.view import SimulatorView


@cache
def _view_of_resolved(path: Path) -> SimulatorView:
    """The memoized open behind :func:`repo_view`, keyed on a resolved view directory."""
    from osprey_connectors.simulation.view import SimulatorView

    return SimulatorView.open(path)


def repo_view(repo: Path) -> SimulatorView:
    """The simulator view the build rendered into ``repo``.

    The one authority on which channels the accelerator model drives and what
    each of them does to it: every binding of a served model names the
    address, its role and the transverse plane it steers or reads. A lane
    choosing devices with a plane or a role in mind reads them from here.

    Opened once per view directory per process, keyed on the resolved path: a
    lane that rebuilds a repo already read here must build it under a fresh
    path to be served the new view.

    Lives HERE rather than in the product because choosing a physics-appropriate
    subset of a facility's channels is a harness concern: the staged view
    holds every channel and says which way each points, and which of them a
    given plan can do physics with is the lane's own question.
    """
    from osprey_connectors.simulation.view import SimulatorView

    path = SimulatorView.path_for_project(repo)
    if not path.is_dir():
        raise AssertionError(f"the build rendered no simulator view at {path}")
    return _view_of_resolved(path.resolve())


def corrector_addresses(
    view: SimulatorView, *, plane: str | None = None, model: str | None = None
) -> frozenset[str]:
    """Every corrector setpoint the view's bindings serve, optionally of one
    plane and of one model.

    A corrector is a setpoint binding that steers a transverse plane. Grouping
    by what a binding DOES -- steer a plane, read one -- rather than by the
    family name its address spells is what keeps a lane's device choice the
    same question on every facility.
    """
    return frozenset(
        binding.address
        for binding in view.bindings(role="setpoint", model=model)
        if binding.plane is not None and (plane is None or binding.plane == plane)
    )


def monitor_addresses(
    view: SimulatorView, *, plane: str | None = None, model: str | None = None
) -> frozenset[str]:
    """Every monitor reading the view's bindings serve, optionally of one plane
    and of one model."""
    return frozenset(
        binding.address for binding in view.bindings(role="monitor", plane=plane, model=model)
    )


def monitors_in_plane(
    view: SimulatorView, plane: str, count: int, *, model: str
) -> tuple[str, ...]:
    """The first ``count`` monitor readings of ``model`` in ``plane``, in address order.

    A lane that names a run of beam-position readings asks for them here, so
    the run moves with the facility instead of spelling device numbers.
    """
    found = sorted(monitor_addresses(view, plane=plane, model=model))
    if len(found) < count:
        raise AssertionError(
            f"the view serves {len(found)} monitor(s) of model {model!r} in plane "
            f"{plane!r}; the lane needs {count}"
        )
    return tuple(found[:count])


def claimed_addresses(view: SimulatorView) -> frozenset[str]:
    """Every corrector setpoint and monitor reading the view's bindings serve.

    The single spelling of the channels a corrector-sweeping plan can do
    physics with, so a lane narrowing the staged devices and a lane selecting
    them read the rule from here.
    """
    return corrector_addresses(view) | monitor_addresses(view)


def bounded_corrector_of(view: SimulatorView) -> tuple[str, str, str] | None:
    """The first corrector of ``view``, in address order, that a limits record
    bounds at both ends and that names its own readback, as
    ``(setpoint, readback, model)``; ``None`` when no corrector qualifies.

    ``model`` is the model whose binding wires the setpoint: a lane that
    drives correctors drives that model's, and reads their response on that
    model's monitors.
    """
    for address in sorted(corrector_addresses(view)):
        channel = view.channel(address)
        readback = channel.pair
        if channel.value_range is not None and readback is not None and readback != address:
            binding = view.binding(address)
            assert binding is not None, f"corrector {address} has no served binding"
            return address, readback, binding.model
    return None


def lane_model(view: SimulatorView) -> str:
    """The model every lane works on: the model of :func:`bounded_corrector_of`.

    The facility's own limits choose it -- the model whose corrector a limits
    record bounds at both ends -- so a lane's devices and readings come from
    the part of the machine a write can be sent to.
    """
    found = bounded_corrector_of(view)
    if found is None:
        raise AssertionError(
            "no corrector of the served view has a limits record with both min_value and max_value"
        )
    return found[2]


def lane_monitors(repo: Path, count: int) -> tuple[str, ...]:
    """The first ``count`` horizontal monitors of the lane's model, from the built view."""
    view = repo_view(repo)
    return monitors_in_plane(view, "x", count, model=lane_model(view))


@dataclass(frozen=True)
class LaneSelection:
    """The channels of the lane model a lane reads and writes, as one selection.

    Every field is read from the served view: the model is :func:`lane_model`,
    the setpoint its bounded corrector, and the rest are that model's own
    channels or the facility's limits records.

    Attributes:
        model: The lane model.
        setpoint: The model's corrector a limits record bounds at both ends.
        plane: The transverse plane, ``"x"`` or ``"y"``, the setpoint's served
            binding steers.
        readback: That corrector's readback, the setpoint's ``pair``.
        band: The setpoint's ``value_range``, ``(min, max)``.
        status_address: The model's status address.
        locked_setpoint: The first setpoint, by address, a limits record locks.
        monitor_x: The model's first horizontal monitor reading, by address.
        monitor_y: The model's first vertical monitor reading, by address.
    """

    model: str
    setpoint: str
    plane: str
    readback: str
    band: tuple[float, float]
    status_address: str
    locked_setpoint: str
    monitor_x: str
    monitor_y: str


def lane_selection(view: SimulatorView) -> LaneSelection:
    """The :class:`LaneSelection` of ``view``.

    The locked setpoint is a setpoint the view serves ``writable: false``. In a
    view rendered under the optional limits mode only a limits record locks a
    setpoint, while every readback is unwritable whatever the records say, so
    the role is what tells the two apart.
    """
    found = bounded_corrector_of(view)
    if found is None:
        raise AssertionError(
            "no corrector of the served view has a limits record with both min_value and max_value"
        )
    setpoint, readback, model = found
    plane = next(
        binding.plane
        for binding in view.bindings(role="setpoint", model=model)
        if binding.address == setpoint
    )
    if plane is None:
        raise AssertionError(f"the lane corrector {setpoint} steers no transverse plane")
    value_range = view.channel(setpoint).value_range
    assert value_range is not None
    low, high = value_range
    locked = [
        address
        for address in view.channels()
        if view.channel(address).role == "setpoint" and not view.channel(address).writable
    ]
    if not locked:
        raise AssertionError("no setpoint of the served view is locked by a limits record")
    return LaneSelection(
        model=model,
        setpoint=setpoint,
        plane=plane,
        readback=readback,
        band=(float(low), float(high)),
        status_address=view.status_addresses()[model],
        locked_setpoint=locked[0],
        monitor_x=monitors_in_plane(view, "x", 1, model=model)[0],
        monitor_y=monitors_in_plane(view, "y", 1, model=model)[0],
    )


def second_models(view: SimulatorView, document: dict[str, Any]) -> tuple[str, ...]:
    """The served models a lane tests beside the lane model, sorted by name.

    A second model is served, is not :func:`lane_model`, measures ``orm`` and
    not ``dispersion`` (its ``measurement.kinds`` in the built ``document``),
    and seeds none of the addresses it wires: its readings are its solved
    lattice alone.
    """
    lane = lane_model(view)
    served = set(view.served())
    found = []
    for entry in document["models"]:
        name = str(entry["name"])
        kinds = (entry.get("measurement") or {}).get("kinds") or ()
        if name not in served or name == lane or "orm" not in kinds or "dispersion" in kinds:
            continue
        if any(view.seed(binding.address) is not None for binding in view.bindings(model=name)):
            continue
        found.append(name)
    return tuple(sorted(found))


def second_model(view: SimulatorView, document: dict[str, Any]) -> str:
    """The first of :func:`second_models` by name."""
    found = second_models(view, document)
    if not found:
        raise AssertionError(
            "no served model other than the lane model measures orm without dispersion "
            "and seeds none of its addresses"
        )
    return found[0]


def bounded_corrector(repo: Path) -> tuple[str, str, str]:
    """:func:`bounded_corrector_of` the view the build rendered into ``repo``.

    The device a lane names when its request depends on a corrector having a
    range: the limits table holds only the records the facility tree authors,
    so a corrector without one has no range to sweep across.
    """
    found = bounded_corrector_of(repo_view(repo))
    if found is None:
        raise AssertionError(
            "no corrector has a limits record with both min_value and max_value "
            f"in the limits table of {repo / 'data' / 'facility' / 'limits.yaml'}"
        )
    return found


def seeded_readback(view: SimulatorView) -> str:
    """The first channel, in address order, that is not a setpoint, has a seed
    and is wired by no served model.

    Reading it returns a number near the seed's nominal: no model moves it.
    """
    for address in view.channels():
        if (
            view.channel(address).role != "setpoint"
            and view.seed(address) is not None
            and view.binding(address) is None
        ):
            return address
    raise AssertionError("the view has no seeded channel that no served model wires")


def bounded_setpoint(view: SimulatorView) -> tuple[str, float]:
    """The first writable setpoint, in address order, that a limits record
    bounds at both ends, with the midpoint of that band.

    The midpoint is a value the limits hook passes on, derived from the band
    rather than typed.
    """
    for address in view.channels():
        channel = view.channel(address)
        if channel.role == "setpoint" and channel.writable and channel.value_range is not None:
            low, high = channel.value_range
            return address, (float(low) + float(high)) / 2
    raise AssertionError("the view has no writable setpoint with a band at both ends")


def facility_view(tree: Path, root: Path) -> SimulatorView:
    """The simulator view a facility ``tree`` renders, rendered into ``root``.

    For a lane that must name a channel in a file it writes before ``osprey
    build`` runs: it renders the repo's own ``data/facility/`` the way the
    build does and asks the view, and checks the build agrees afterwards.
    ``root`` is created when missing, as the build's render root exists before
    the facility file is written into it.
    """
    from osprey.facility.build import build_facility
    from osprey.facility.render import render_facility_outputs
    from osprey_connectors.simulation.view import SimulatorView

    root.mkdir(parents=True, exist_ok=True)
    render_facility_outputs(root, build_facility(tree, project_name=tree.name), {}, tree)
    return SimulatorView.of_render(root)


def repo_facility(repo: Path) -> dict[str, Any]:
    """The facility file the build rendered into ``repo``."""
    from osprey.facility import FACILITY_FILE
    from osprey_connectors.workspace import rendered_config_path

    path = rendered_config_path(Path(repo)).parent / FACILITY_FILE
    if not path.is_file():
        raise AssertionError(f"the build rendered no facility file at {path}")
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


@dataclass(frozen=True)
class ProbeDevice:
    """One device of the facility file and the facts a graph lane grades against.

    Every field is read from the facility file, which is upstream of the graph
    the lane's agent queries: a rubric built from these figures cannot be
    satisfied by a graph that is wrong.
    """

    device_id: str
    name: str
    label: str
    klass: str
    class_label: str
    model: str
    place: str
    section: str
    top_place: str
    top_place_name: str
    s: float
    ordinal_in_place: int
    ordinal_in_model: int
    bindings: tuple[str, ...]
    setpoint: str
    uri: str
    previous: tuple[str, float, str] | None
    next: tuple[str, float, str] | None
    class_count_in_model: int
    class_count_in_facility: int
    class_per_place: int
    class_span: tuple[float, float]
    class_spacing: tuple[float, float]


def class_label_of(klass: str) -> str:
    """A class name in words: ``BeamPositionMonitor`` -> ``beam position monitor``."""
    return " ".join(word.lower() for word in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+", klass))


def _channels_on(document: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Each device id's channel records, by the device their ``on`` names."""
    on: dict[str, list[dict[str, Any]]] = {}
    for channel in document["channels"]:
        device = (channel.get("on") or {}).get("device")
        if device is not None:
            on.setdefault(device, []).append(channel)
    return on


def _qualifying(document: dict[str, Any], model: str) -> list[dict[str, Any]]:
    """The devices of ``model`` a graph lane can probe, in document order.

    A device qualifies when it is placed and positioned, no other device of
    its model shares its position, and at least three channels sit on it, of
    which exactly one is a setpoint.
    """
    on = _channels_on(document)
    placed = [
        device
        for device in document["devices"]
        if device.get("model") == model and "s" in device and "place" in device
    ]
    positions = Counter(device["s"] for device in placed)
    return [
        device
        for device in placed
        if positions[device["s"]] == 1
        and len(on.get(device["id"], ())) >= 3
        and sum(c.get("role") == "setpoint" for c in on[device["id"]]) == 1
    ]


def graph_probe_device(document: dict[str, Any], *, model: str) -> ProbeDevice:
    """The device of ``model`` a graph lane asks about, with its facts.

    Among the qualifying devices of ``model`` (see :func:`_qualifying`), the
    class with the fewest devices in that model is taken -- ties broken by
    class name -- and of that class the qualifying device first by ``s``.
    """
    from osprey.facility.views.graph_iri import iri

    qualifying = _qualifying(document, model)
    if not qualifying:
        raise AssertionError(
            f"no device of model {model!r} in the facility file is positioned, "
            "unshared in position, and carries at least three channels with exactly "
            "one setpoint"
        )
    in_model = sorted(
        (d for d in document["devices"] if d.get("model") == model and "s" in d),
        key=lambda d: d["s"],
    )
    per_class = Counter(d["class"] for d in in_model)
    klass = min({d["class"] for d in qualifying}, key=lambda c: (per_class[c], c))
    device = min((d for d in qualifying if d["class"] == klass), key=lambda d: d["s"])

    channels = _channels_on(document)[device["id"]]
    setpoint = next(c["id"] for c in channels if c.get("role") == "setpoint")
    top_place = device["place"].split("/")[0]
    places = {place["id"]: place for place in document.get("places", ())}
    top_names = places.get(top_place, {}).get("names") or [top_place]

    def neighbour(candidates: list[dict[str, Any]]) -> tuple[str, float, str] | None:
        if not candidates:
            return None
        other = candidates[0]
        return (other["names"][0], other["s"], other["class"])

    below = [d for d in reversed(in_model) if d["s"] < device["s"]]
    above = [d for d in in_model if d["s"] > device["s"]]
    same_class = [d["s"] for d in in_model if d["class"] == klass]
    gaps = [b - a for a, b in zip(same_class, same_class[1:], strict=False)]
    per_place = Counter(d["place"] for d in in_model if d["class"] == klass)

    return ProbeDevice(
        device_id=device["id"],
        name=device["names"][0],
        label=device["label"],
        klass=klass,
        class_label=class_label_of(klass),
        model=model,
        place=device["place"],
        section=device["place"].split("/")[-1],
        top_place=top_place,
        top_place_name=top_names[0],
        s=device["s"],
        ordinal_in_place=device["ordinalInPlace"],
        ordinal_in_model=device["ordinalInModel"],
        bindings=tuple(sorted(c["id"] for c in channels)),
        setpoint=setpoint,
        uri=iri(document["identity"]["code"], "device", device["id"]),
        previous=neighbour(below),
        next=neighbour(above),
        class_count_in_model=per_class[klass],
        class_count_in_facility=sum(d.get("class") == klass for d in document["devices"]),
        class_per_place=Counter(per_place.values()).most_common(1)[0][0],
        class_span=(same_class[0], same_class[-1]),
        class_spacing=(min(gaps), max(gaps)) if gaps else (0.0, 0.0),
    )
