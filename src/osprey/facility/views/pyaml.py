"""The pyAML view: one pyAML configuration per measured model.

Written to ``<render>/data/pyaml/<model>/`` for every model the render serves
that names a deck and has a ``measurement/<model>.yaml``::

    configuration.yaml   the pyAML accelerator: its design simulator, the
                         ``live`` OSPREY control system, one device per magnet,
                         BPM and instrument the measurement file's groups and
                         instruments name, one array per group role, named
                         after its group, or after the role when the role
                         names several (one group named as both ``hcor`` and
                         ``vcor`` is two arrays, one per plane; a group role
                         only a registered kind declares has no array), and each
                         measurement kind's tool section, written by the kind's
                         declaration (:mod:`osprey.facility.kinds`)
    lattice.json         the design simulator's lattice, referenced as
                         ``${path:lattice.json}``: the model's deck with each
                         corrector the view drives carrying its kick as the
                         dipole polynomials pyAML reads, as the model's engine
                         writes it
    <kind files>         each file a kind's declaration adds to the view
                         (``view_files``): the built-in ``trm`` and ``crm`` add
                         ``trm.json`` and ``crm.json``, the tune and
                         chromaticity response matrices of the design optics,
                         for a periodic model whose group has a finite column
                         for every member; the correction tool that loads one
                         is added only when the matrix can drive a two-plane
                         correction (rank two)

A model solved ``single_pass`` has no design simulator (``simulators: []``):
pyAML's design simulator solves a periodic orbit. Every other model the render
serves names a note on stderr saying why it has no view. A setpoint pyAML cannot
convert is left out of the view and named in a note.

Every name the configuration holds comes from :mod:`pyaml_cs_osprey.names`, so
the view, the measurement tools and a reader of the configuration cannot spell
one two ways. Nothing in a view depends on where the render sits: references
between its files are relative, and equal sources and configs give equal bytes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.facility import TEXTURE
from osprey.facility.kinds import SHIPPED_SORTS
from osprey.facility.records import FacilityRecords
from osprey.facility.views import ViewInputs, report_note
from osprey.facility.views.facts import CONFIGURATION_FILE, PYAML_DIR

if TYPE_CHECKING:
    from osprey.facility.kinds import ViewContext
    from pyaml_cs_osprey.names import ViewNames

__all__ = [
    "CONFIGURATION_FILE",
    "GROUP_ROLES",
    "LATTICE_FILE",
    "PYAML_DIR",
    "ROLE_PLANES",
    "kind_records",
    "measured_models",
    "measurement_groups",
    "pyaml_view_wanted",
    "tune_references",
    "view_names",
    "write_pyaml_view",
]

#: The file a model record is named in when no source states it.
_MODELS_FILE = "models.yaml"

#: The deck's copy beside the configuration.
LATTICE_FILE = "lattice.json"

#: The group roles the view writes an array for, in the order it writes them.
GROUP_ROLES: tuple[str, ...] = tuple(
    role for role, sort in SHIPPED_SORTS.items() if sort != "instrument"
)

#: Each magnet role and the pyAML magnet its setpoints are.
MAGNET_TYPES: dict[str, str] = {
    "hcor": "pyaml.magnet.hcorrector",
    "vcor": "pyaml.magnet.vcorrector",
    "quad": "pyaml.magnet.quadrupole",
    "sext": "pyaml.magnet.sextupole",
}

#: Each magnet role and the unit of the integrated strength pyAML steps it in.
STRENGTH_UNITS: dict[str, str] = {
    "hcor": "rad",
    "vcor": "rad",
    "quad": "1/m",
    "sext": "1/m**2",
}

ACCELERATOR_TYPE = "pyaml.accelerator"
SIMULATOR_TYPE = "pyaml.lattice.simulator"
CONTROL_SYSTEM_TYPE = "pyaml_cs_osprey.controlsystem"
BPM_TYPE = "pyaml.bpm.bpm"
BPM_ARRAY_TYPE = "pyaml.arrays.bpm"
MAGNET_ARRAY_TYPE = "pyaml.arrays.magnet"
LINEAR_MODEL_TYPE = "pyaml.magnet.linear_model"
INLINE_CURVE_TYPE = "pyaml.magnet.inline_curve"
RF_PLANT_TYPE = "pyaml.rf.rf_plant"
TUNE_MONITOR_TYPE = "pyaml.diagnostics.tune_monitor"
RESPONSE_MATRIX_DATA_TYPE = "pyaml.tuning_tools.response_matrix_data"

#: pyAML exposes these as ``accelerator.design`` and ``accelerator.live``.
SIMULATOR_NAME = "design"
CONTROL_SYSTEM_NAME = "live"

#: The keys of a tool section that name an array, each with the group role it is.
ARRAY_KEYS: dict[str, str] = {
    "bpm_array_name": "bpm",
    "hcorr_array_name": "hcor",
    "vcorr_array_name": "vcor",
    "quad_array_name": "quad",
    "sextu_array_name": "sext",
}

#: The speed of light in m/s, exact; pyAML's rigidity is the energy in eV over it.
_SPEED_OF_LIGHT = 299792458.0

#: The solve whose model has no design simulator.
_SINGLE_PASS = "single_pass"  # noqa: S105 -- names a lattice mode, holds no secret

#: Wide enough that no generated string or reference is folded across lines.
_YAML_WIDTH = 4096

#: A device whose record states no ``s`` sorts after every placed device.
_UNPLACED = math.inf


@dataclass(frozen=True)
class _Magnet:
    """One magnet: its setpoint address, role and wiring entry."""

    address: str
    role: str
    entry: Mapping[str, Any]


@dataclass(frozen=True)
class _Bpm:
    """One BPM: its device id, monitored element and readbacks by plane."""

    device: str
    element: str
    planes: Mapping[str, str]


# --- which models -------------------------------------------------------------


def measured_models(inputs: ViewInputs) -> tuple[list[str], dict[str, str]]:
    """The models a render writes a pyAML view for, and why each other one has none.

    Args:
        inputs: The render's view inputs.

    Returns:
        ``(written, omitted)``: the model names, sorted, and each served model
        that has no view with the reason, sorted by name. ``texture`` is never
        named.
    """
    written: list[str] = []
    omitted: dict[str, str] = {}
    for model in sorted(inputs.records.of_kind("model"), key=lambda record: str(record["name"])):
        name = str(model["name"])
        if name == TEXTURE:
            continue
        if name not in inputs.served:
            continue
        if not model.get("deck"):
            omitted[name] = "the model names no deck"
        elif not model.get("measurement"):
            omitted[name] = f"no measurement/{name}.yaml"
        else:
            written.append(name)
    return written, omitted


def pyaml_view_wanted(inputs: ViewInputs) -> tuple[bool, str]:
    """Whether a render carries the pyAML view, and what decided it.

    Args:
        inputs: The render's view inputs.

    Returns:
        ``(True, ...)`` when one served model names a deck and has a
        measurement file; otherwise ``False`` with the reason, which names each
        served deck model as its note would.
    """
    written, omitted = measured_models(inputs)
    if written:
        return True, "a served deck model has a measurement file"
    if not omitted:
        return False, "no served model names a deck"
    return False, "; ".join(_omitted_note(name, reason) for name, reason in omitted.items())


def _omitted_note(model: str, reason: str) -> str:
    return f"pyAML view omitted for {model}: {reason}"


# --- groups -------------------------------------------------------------------


def _device_order(records: FacilityRecords, members: Iterable[str]) -> list[str]:
    """``members`` in ``(s, id)`` order; a device with no ``s`` after every placed one."""

    def key(device: str) -> tuple[float, str]:
        s = (records.record("device", device) or {}).get("s")
        return (float(s) if isinstance(s, int | float) else _UNPLACED, device)

    return sorted(dict.fromkeys(str(member) for member in members), key=key)


def _wiring(records: FacilityRecords, model: str) -> dict[str, Mapping[str, Any]]:
    """The wiring records a model states, by address."""
    wiring = records.require("model", model).get("wiring", ())
    return {str(entry["address"]): entry for entry in wiring}


#: The plane each corrector role steers; a role not named here takes every plane.
ROLE_PLANES: dict[str, str] = {"hcor": "x", "vcor": "y"}


def measurement_groups(
    records: FacilityRecords, model: str, sorts: Mapping[str, str] | None = None
) -> dict[str, list[str]]:
    """The addresses each group role of a model's measurement file stands for.

    ``bpm`` is the readbacks the model wires as monitors, in both planes;
    ``hcor`` the setpoints the model wires whose plane its engine describes as
    ``x``, ``vcor`` those it describes as ``y``; ``quad`` and ``sext`` every
    setpoint the model wires; a role a registered kind declares the monitors
    or the setpoints the model wires, in every plane, as the kind sorts it.
    Each is taken over the member devices of every
    group the role names: the channels on a member device and the channels
    naming one as an endpoint, visiting the
    devices in ``(s, id)`` order, an address listed once at its first device and
    the addresses within a device sorted. A combined corrector, one device
    steered in both planes, keeps each setpoint in the role of its own plane, so
    ``hcor`` and ``vcor`` share no address.

    Args:
        records: The facility file's records.
        model: The model's name.
        sorts: Each member role's sort (:func:`osprey.facility.kinds.member_sorts`);
            the shipped roles' when omitted, and a role of no known sort is
            left out.

    Returns:
        Role to addresses, for each group role the measurement file names: the
        shipped ones in :data:`GROUP_ROLES` order, then the others by name.

    Raises:
        KeyError: The facility file has no model ``model``.
    """
    from osprey.facility.sources import group_families

    record = records.require("model", model)
    groups_named = (record.get("measurement") or {}).get("groups") or {}
    known = SHIPPED_SORTS if sorts is None else sorts
    wiring = _wiring(records, model)
    found: dict[str, list[str]] = {}
    roles = [*GROUP_ROLES, *sorted(set(groups_named) - set(GROUP_ROLES))]
    for role in roles:
        if groups_named.get(role) is None or known.get(role) in (None, "instrument"):
            continue
        wanted = "monitor" if known[role] == "monitors" else "setpoint"
        plane = ROLE_PLANES.get(role)
        members = [
            member
            for group_id in group_families(groups_named.get(role))
            for member in (records.record("group", group_id) or {}).get("members") or []
        ]
        addresses: dict[str, None] = {}
        for device in _device_order(records, members):
            for address in sorted({str(c["id"]) for c in records.channels_on(device)}):
                entry = wiring.get(address)
                if entry is None or entry.get("role") != wanted:
                    continue
                if plane is not None and entry.get("plane") != plane:
                    continue
                addresses.setdefault(address, None)
        found[role] = list(addresses)
    return found


# --- the configuration ----------------------------------------------------------


def _yaml_text(body: Mapping[str, Any]) -> str:
    """``body`` as block-style YAML in insertion order, no line folded."""
    from osprey_connectors import yaml_loader

    return yaml_loader.safe_dump(
        dict(body),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=_YAML_WIDTH,
    )


def _matrix_text(body: Mapping[str, Any]) -> str:
    import json

    return json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _path_reference(target: str) -> str:
    return f"${{path:{target}}}"


def _reference(address: str, unit: str | None, *, readback: str | None = None) -> str:
    """A pyaml_cs_osprey channel reference, with ``[unit]`` when one is given."""
    text = f"({readback}, {address})" if readback and readback != address else address
    return f"{text}[{unit}]" if unit else text


def _slice_elements(entry: Mapping[str, Any]) -> list[tuple[str, float]]:
    """``(element, weight)`` of every slice of a wiring entry, a bare element weighing 1."""
    if entry.get("element") is not None:
        return [(str(entry["element"]), 1.0)]
    pieces = []
    for piece in entry.get("slices") or []:
        weight = piece.get("weight")
        pieces.append((str(piece["element"]), 1.0 if weight is None else float(weight)))
    return pieces


def _lattice_names(elements: Sequence[str]) -> str:
    """pyAML's ``list(a,b)`` selector of the deck elements a device drives."""
    return f"list({','.join(dict.fromkeys(elements))})"


@dataclass(frozen=True)
class _Deck:
    """The facts a view reads off the model's deck."""

    energy_ev: float
    lengths: Mapping[str, float]
    lattice: Any


def _load_deck(path: Path) -> _Deck:
    import at

    lattice = at.load_lattice(str(path))
    lengths: dict[str, float] = {}
    for element in lattice:
        lengths.setdefault(str(element.FamName), float(element.Length))
    return _Deck(energy_ev=float(lattice.energy), lengths=lengths, lattice=lattice)


def _mixes_thin_and_thick(magnet: _Magnet, deck: _Deck) -> bool:
    """Whether the magnet's slices mix zero-length and thick elements of the deck."""
    lengths = [deck.lengths.get(element, 0.0) for element, _ in _slice_elements(magnet.entry)]
    # Stand-in until pyAML round-trips or refuses a magnet whose slices mix zero-length and thick elements; this refusal goes with it.
    return 0.0 in lengths and any(lengths)


def _magnet_model(
    magnet: _Magnet, deck: _Deck, limits: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """The pyAML model one setpoint is converted by, or ``None`` when pyAML has none.

    pyAML's strength is the integrated field over the rigidity. The deck holds
    ``curve(hardware) * weight`` on each slice, so the magnet's integrated
    strength is ``curve(hardware) * scale`` with ``scale`` the sum of the slice
    weights times their lengths for a multipole and of the weights for a kick
    or for a magnet whose every slice is zero-length, which pyAML reads at
    unit length:
    a linear curve becomes a ``linear_model`` of factor ``gain * scale * brho``,
    a table an ``inline_curve`` of ``(hardware, value * scale * brho)`` rows
    extended one knot span past each end and past the address's limits band.
    A curve pyAML cannot represent (one per slice, a zero gain or scale, a table
    that is not strictly monotone) has no model. Nor has a magnet whose slices
    mix zero-length and thick elements: pyAML writes one polynomial into every
    slice, which a zero-length slice reads as integrated and a thick one per
    metre.
    """
    from osprey.simulation.engines.calibration import Linear, Table, curve_from_record

    entry = magnet.entry
    calibration = entry.get("calibration") or {}
    pieces = _slice_elements(entry)
    if any((piece.get("curve") is not None) for piece in entry.get("slices") or []):
        return None
    if _mixes_thin_and_thick(magnet, deck):
        return None
    lengths = [deck.lengths.get(element, 0.0) for element, _ in pieces]
    unit = magnet.role in ("hcor", "vcor") or not any(lengths)
    scale = sum(
        weight * (1.0 if unit else length)
        for (_, weight), length in zip(pieces, lengths, strict=True)
    )
    if scale == 0.0:
        return None
    brho = deck.energy_ev / _SPEED_OF_LIGHT
    model: dict[str, Any] = {
        "type": LINEAR_MODEL_TYPE,
        "unit": STRENGTH_UNITS[magnet.role],
        "hardware_unit": str(entry.get("unit") or ""),
        "powerconverter": _reference(magnet.address, None, readback=_pair(entry)),
    }
    curve = curve_from_record(calibration.get("curve"))
    if curve is None:
        curve = Linear(gain=1.0, offset=0.0)
    if isinstance(curve, Linear):
        if curve.gain == 0.0:
            return None
        model["calibration_factor"] = float(curve.gain * scale * brho)
        if curve.offset:
            model["calibration_offset"] = float(-curve.offset / curve.gain)
        return model
    if isinstance(curve, Table):
        knots = _table_knots(curve, limits.get(magnet.address) or {})
        if knots is None:
            return None
        model["curve"] = {
            "type": INLINE_CURVE_TYPE,
            "mat": [[hardware, value * scale * brho] for hardware, value in knots],
        }
        return model
    return None


def _pair(entry: Mapping[str, Any]) -> str | None:
    return entry.get("readback")


def _table_knots(table: Any, limit: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    """A table's knots in ascending hardware value, extended past both ends.

    The rows are reversed only when the grid falls, so each stays paired with its
    value. Each end gains one knot on its end segment, one knot span past the end
    or past the limits band's bound there, whichever is further out, so a
    strength past the table is extrapolated as the simulator does, never
    clamped.
    """
    grid = [float(point) for point in table.grid]
    values = [float(value) for value in table.values]
    if len(grid) < 2:
        return None
    if grid[-1] < grid[0]:
        grid.reverse()
        values.reverse()
    steps = [b - a for a, b in zip(values, values[1:], strict=False)]
    if not (all(step > 0 for step in steps) or all(step < 0 for step in steps)):
        return None
    knots = list(zip(grid, values, strict=True))
    low_span = grid[1] - grid[0]
    low = min(grid[0], _bound(limit, "min_value", grid[0])) - low_span
    knots.insert(0, (low, values[0] + (values[1] - values[0]) / low_span * (low - grid[0])))
    high_span = grid[-1] - grid[-2]
    high = max(grid[-1], _bound(limit, "max_value", grid[-1])) + high_span
    knots.append((high, values[-1] + (values[-1] - values[-2]) / high_span * (high - grid[-1])))
    return knots


def _bound(limit: Mapping[str, Any], key: str, default: float) -> float:
    value = limit.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return default
    return float(value)


def _magnets(
    records: FacilityRecords, model: str, groups: Mapping[str, list[str]]
) -> list[_Magnet]:
    wiring = _wiring(records, model)
    magnets: list[_Magnet] = []
    for role in MAGNET_TYPES:
        for address in groups.get(role, ()):
            if any(magnet.address == address for magnet in magnets):
                continue
            entry = dict(wiring[address])
            entry["readback"] = (records.record("channel", address) or {}).get("pair")
            magnets.append(_Magnet(address, role, entry))
    return magnets


def _bpms(records: FacilityRecords, model: str, addresses: Sequence[str]) -> list[_Bpm]:
    """The BPM devices the ``bpm`` readbacks are on, in the readbacks' order."""
    wiring = _wiring(records, model)
    bpms: dict[str, dict[str, Any]] = {}
    for address in addresses:
        entry = wiring[address]
        device = str((records.require("channel", address).get("on") or {}).get("device"))
        held = bpms.setdefault(device, {"element": _slice_elements(entry)[0][0], "planes": {}})
        held["planes"].setdefault(str(entry.get("plane")), address)
    return [
        _Bpm(device, str(held["element"]), dict(held["planes"])) for device, held in bpms.items()
    ]


def _unit(records: FacilityRecords, address: str) -> str | None:
    """The unit the channel at ``address`` states, if any."""
    unit = (records.record("channel", address) or {}).get("unit")
    return str(unit) if unit else None


def tune_references(records: FacilityRecords, model: str, address: str) -> tuple[str, str] | None:
    """The horizontal and vertical tune references the ``tune`` instrument stands for.

    A waveform reading the whole tune output is read by element: ``@0`` and
    ``@1``. A scalar reading one plane is paired with the model's reading on the
    other plane whose engine block differs from its own in one key alone, the
    one that names the plane.
    """
    entries = records.require("model", model).get("wiring", ())
    entry = next((entry for entry in entries if str(entry["address"]) == address), None)
    if entry is None:
        return None
    plane = entry.get("plane")
    if plane is None:
        return f"{address}@0", f"{address}@1"
    block = entry.get("engine") or {}
    other = {"x": "y", "y": "x"}.get(str(plane))
    partners = [
        str(candidate["address"])
        for candidate in entries
        if candidate.get("role") == entry.get("role")
        and candidate.get("plane") == other
        and _differ_in_one_key(candidate.get("engine") or {}, block)
    ]
    if len(partners) != 1:
        return None
    return (address, partners[0]) if plane == "x" else (partners[0], address)


def _differ_in_one_key(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """True when two engine blocks name the same keys and differ in the value of one."""
    if set(first) != set(second):
        return False
    differing = [key for key in first if first[key] != second[key]]
    return len(differing) == 1


@dataclass(frozen=True)
class _Optics:
    """The design optics a periodic model's view reads."""

    alphac: float | None
    harmonic_number: int | None
    beta: Any
    dispersion: Any
    positions: Mapping[str, int]


def _design_optics(deck: _Deck) -> _Optics | None:
    """The 4D design optics of the deck, computed on a copy; ``None`` when it has none.

    A deck with no stable closed solution has no optics. A deck that bends
    nothing has no momentum compaction, and its ``alphac`` is ``None``.
    """
    import at
    import numpy as np

    lattice = deck.lattice.disable_6d(copy=True)
    try:
        _, beta, _, dispersion, _, _, _ = at.avlinopt(lattice, refpts=np.arange(len(lattice)))
    except at.AtError:
        return None
    try:
        alphac: float | None = float(at.get_mcf(lattice))
    except (at.AtError, IndexError):
        alphac = None
    try:
        harmonic: int | None = int(lattice.harmonic_number)
    except (AttributeError, ValueError):
        harmonic = None
    positions: dict[str, int] = {}
    repeated: set[str] = set()
    for position, element in enumerate(lattice):
        name = str(element.FamName)
        if name in positions:
            repeated.add(name)
        positions[name] = position
    return _Optics(
        alphac=alphac,
        harmonic_number=harmonic,
        beta=np.asarray(beta, dtype=float),
        dispersion=np.asarray(dispersion, dtype=float),
        positions={name: index for name, index in positions.items() if name not in repeated},
    )


def _slice_average(magnet: _Magnet, deck: _Deck, optics: _Optics, quantity: Any) -> float | None:
    """``sum(q_i * w_i * L_i) / sum(w_i * L_i)`` over the magnet's slices.

    A magnet whose every slice is zero-length weighs each slice at unit length,
    as its pyAML magnet model scales it.
    """
    pieces = _slice_elements(magnet.entry)
    lengths = [deck.lengths.get(element, 0.0) for element, _ in pieces]
    unit = not any(lengths)
    weighted = total = 0.0
    for (element, weight), length in zip(pieces, lengths, strict=True):
        position = optics.positions.get(element)
        if position is None:
            return None
        share = weight * (1.0 if unit else length)
        weighted += float(quantity(position)) * share
        total += share
    return None if total == 0.0 else weighted / total


def _response_matrix(
    columns: Sequence[tuple[str, float, float]], monitor: str
) -> dict[str, Any] | None:
    """A two-row response matrix, or ``None`` when a column is missing or not finite."""
    import numpy as np

    if not columns:
        return None
    rows = [
        [float(f"{x:.9g}") for _, x, _ in columns],
        [float(f"{y:.9g}") for _, _, y in columns],
    ]
    if not np.all(np.isfinite(np.asarray(rows, dtype=float))):
        return None
    return {
        "type": RESPONSE_MATRIX_DATA_TYPE,
        "matrix": rows,
        "observable_names": [f"{monitor}.x", f"{monitor}.y"],
        "variable_names": [name for name, _, _ in columns],
    }


def _drives_a_correction(matrix: Mapping[str, Any]) -> bool:
    """Whether the matrix can drive a two-plane correction: rank two."""
    import numpy as np

    return int(np.linalg.matrix_rank(np.asarray(matrix["matrix"], dtype=float))) >= 2


def _tuning_matrix(
    role: str,
    addresses: Sequence[str],
    magnets: Sequence[_Magnet],
    names: ViewNames,
    deck: _Deck,
    optics: _Optics,
    monitor: str,
) -> dict[str, Any] | None:
    """The tune (``quad``) or chromaticity (``sext``) response of the design optics.

    A tune column is ``(+beta_x, -beta_y) / 4pi``, a chromaticity column
    ``(+beta_x * eta_x, -beta_y * eta_x) / 2pi``, each averaged over the
    magnet's slices by weight times length (unit length when every slice is
    zero-length): the response per unit of the integrated strength pyAML steps.
    """
    beta = optics.beta
    eta = optics.dispersion[:, 0]
    by_address = {magnet.address: magnet for magnet in magnets}
    columns: list[tuple[str, float, float]] = []
    for address in addresses:
        magnet = by_address[address]
        if role == "quad":
            scale = 1.0 / (4.0 * math.pi)
            x = _slice_average(magnet, deck, optics, lambda p: beta[p, 0])
            y = _slice_average(magnet, deck, optics, lambda p: beta[p, 1])
        else:
            scale = 1.0 / (2.0 * math.pi)
            x = _slice_average(magnet, deck, optics, lambda p: beta[p, 0] * eta[p])
            y = _slice_average(magnet, deck, optics, lambda p: beta[p, 1] * eta[p])
        if x is None or y is None:
            return None
        columns.append((names.magnet_name(magnet.address), scale * x, -scale * y))
    return _response_matrix(columns, monitor)


def _names(
    magnets: Sequence[_Magnet], bpms: Sequence[_Bpm], groups_named: Mapping[str, Any], rf: Any
) -> ViewNames:
    from osprey.facility.sources import group_families
    from pyaml_cs_osprey.names import ViewNames

    return ViewNames.build(
        magnets=[magnet.address for magnet in magnets],
        bpms=[bpm.device for bpm in bpms],
        groups={
            role: group_families(groups_named[role]) for role in GROUP_ROLES if role in groups_named
        },
        rf=str(rf) if rf is not None else None,
    )


def view_names(records: FacilityRecords, model: str) -> ViewNames:
    """The names a model's pyAML view gives its magnets, BPMs, arrays and RF plant.

    Args:
        records: The facility file's records.
        model: A model with a measurement file.

    Returns:
        The view's names, built from the facility file as the view writer
        builds them.

    Raises:
        KeyError: The facility file has no model ``model``.
        ValueError: Two addresses or ids of the view would share a name.
    """
    record = records.require("model", model)
    measurement = record.get("measurement") or {}
    groups = measurement_groups(records, model)
    return _names(
        _magnets(records, model, groups),
        _bpms(records, model, groups.get("bpm", ())),
        measurement.get("groups") or {},
        (measurement.get("instruments") or {}).get("rf"),
    )


def _measurement_value(measurement: Mapping[str, Any], key: str) -> float | int:
    value = measurement[key]
    return int(value) if key in ("n_step", "n_avg_meas", "fit_order") else float(value)


def _configuration(
    inputs: ViewInputs, model: str
) -> tuple[dict[str, Any], dict[str, str], list[str], list[str]]:
    """One model's pyAML configuration, the extra files it references and what it leaves out.

    A setpoint whose calibration pyAML cannot represent has no magnet and is in
    no array.

    Returns:
        ``(configuration, files, unmodelled, mixed)``: the configuration as a
        mapping, each file beside it by name with its text, the setpoints left
        out for want of a pyAML magnet model and those left out because their
        slices mix zero-length and thick elements, each in group order.
    """
    from pyaml_cs_osprey.names import (
        RF_PLANT_NAME,
        TUNE_MONITOR_NAME,
    )

    records = inputs.records
    record = records.require("model", model)
    measurement = record.get("measurement") or {}
    solve = ((record.get("settings") or {}).get(str(record["engine"])) or {}).get("solve")
    periodic = solve != _SINGLE_PASS
    deck = _load_deck(inputs.facility_dir / str(record["deck"]))
    groups = measurement_groups(records, model)
    groups_named = measurement.get("groups") or {}
    instruments = measurement.get("instruments") or {}
    kinds = list(measurement.get("kinds") or [])
    limit_rows = (records.document.get("limits") or {}).get("records") or []
    limits = {str(row["address"]): row for row in limit_rows}

    magnets = _magnets(records, model, groups)
    bpms = _bpms(records, model, groups.get("bpm", ()))
    rf = instruments.get("rf")
    names = _names(magnets, bpms, groups_named, rf)
    models = {magnet.address: _magnet_model(magnet, deck, limits) for magnet in magnets}
    mixed = [
        magnet.address
        for magnet in magnets
        if models[magnet.address] is None and _mixes_thin_and_thick(magnet, deck)
    ]
    unmodelled = [
        address for address, found in models.items() if found is None and address not in mixed
    ]
    files: dict[str, str] = {}
    if periodic:
        files[LATTICE_FILE] = _design_lattice(inputs, record, magnets)
    left_out = {*unmodelled, *mixed}
    magnets = [magnet for magnet in magnets if magnet.address not in left_out]
    groups = {
        role: [address for address in addresses if address not in left_out]
        for role, addresses in groups.items()
    }

    configuration: dict[str, Any] = {
        "type": ACCELERATOR_TYPE,
        "facility": str(records.document["identity"]["code"]),
        "machine": model,
        "energy": deck.energy_ev,
    }
    optics = _design_optics(deck) if periodic else None
    if optics is not None:
        if optics.alphac is not None:
            configuration["alphac"] = optics.alphac
        if optics.harmonic_number is not None:
            configuration["harmonic_number"] = optics.harmonic_number
    configuration["simulators"] = (
        [
            {
                "type": SIMULATOR_TYPE,
                "name": SIMULATOR_NAME,
                "lattice": _path_reference(LATTICE_FILE),
            }
        ]
        if periodic
        else []
    )
    configuration["controls"] = [{"type": CONTROL_SYSTEM_TYPE, "name": CONTROL_SYSTEM_NAME}]

    arrays: list[dict[str, Any]] = []
    for role in GROUP_ROLES:
        if role not in groups_named:
            continue
        if role == "bpm":
            members = [names.bpm_name(bpm.device) for bpm in bpms]
            array_type = BPM_ARRAY_TYPE
        else:
            members = [names.magnet_name(address) for address in groups.get(role, ())]
            array_type = MAGNET_ARRAY_TYPE
        arrays.append(
            {
                "type": array_type,
                "name": names.array_name(role),
                "elements": members,
            }
        )
    configuration["arrays"] = arrays

    devices: list[dict[str, Any]] = []
    for magnet in magnets:
        section: dict[str, Any] = {
            "type": MAGNET_TYPES[magnet.role],
            "name": names.magnet_name(magnet.address),
            "lattice_names": _lattice_names([e for e, _ in _slice_elements(magnet.entry)]),
        }
        section["model"] = models[magnet.address]
        devices.append(section)
    for bpm in bpms:
        section = {
            "type": BPM_TYPE,
            "name": names.bpm_name(bpm.device),
            "lattice_names": _lattice_names([bpm.element]),
        }
        for plane in ("x", "y"):
            if plane in bpm.planes:
                address = bpm.planes[plane]
                section[f"{plane}_pos"] = _reference(address, _unit(records, address))
        devices.append(section)
    if rf is not None:
        devices.append(
            {
                "type": RF_PLANT_TYPE,
                "name": names.rf_plant_name(str(rf)),
                "masterclock": _reference(str(rf), _unit(records, str(rf))),
            }
        )
    tune = instruments.get("tune")
    tunes = tune_references(records, model, str(tune)) if tune is not None else None
    if tunes is not None:
        monitor: dict[str, Any] = {
            "type": TUNE_MONITOR_TYPE,
            "name": TUNE_MONITOR_NAME,
            "tune_h": tunes[0],
            "tune_v": tunes[1],
        }
        if rf is not None:
            monitor["rf_plant_name"] = RF_PLANT_NAME
        devices.append(monitor)

    present = {"tune_monitor"} if tunes is not None else set()
    if rf is not None:
        present.add("rf_plant")
    context = _view_context(
        model,
        kinds,
        names,
        groups_named,
        measurement,
        frozenset(present),
        optics,
        lambda role, monitor: (
            _tuning_matrix(role, groups.get(role, ()), magnets, names, deck, optics, monitor)
            if optics is not None
            else None
        ),
    )
    tools, extras, kind_files = _kind_sections(kinds, context, measurement)
    devices.extend(tools)
    devices.extend(extras)
    files.update(kind_files)
    configuration["devices"] = devices
    return configuration, files, unmodelled, mixed


def _view_context(
    model: str,
    kinds: Sequence[str],
    names: ViewNames,
    groups_named: Mapping[str, Any],
    measurement: Mapping[str, Any],
    present: frozenset[str],
    optics: _Optics | None,
    tuning: Callable[[str, str], dict[str, Any] | None],
) -> ViewContext:
    """What the model's kinds' view writers are handed; each matrix is computed once."""
    from osprey.facility.kinds import ViewContext
    from pyaml_cs_osprey.names import (
        CHROMATICITY_MONITOR_NAME,
        RF_PLANT_NAME,
        TUNE_MONITOR_NAME,
    )

    computed: dict[tuple[str, str], dict[str, Any] | None] = {}

    def tuning_matrix(role: str, monitor: str) -> dict[str, Any] | None:
        if (role, monitor) not in computed:
            computed[role, monitor] = tuning(role, monitor)
        return computed[role, monitor]

    return ViewContext(
        model=model,
        kinds=tuple(kinds),
        arrays={
            str(role): names.array_name(str(role)) for role in groups_named if role in GROUP_ROLES
        },
        measurement=measurement,
        devices={
            "rf_plant": RF_PLANT_NAME,
            "tune_monitor": TUNE_MONITOR_NAME,
            "chromaticity_monitor": CHROMATICITY_MONITOR_NAME,
        },
        present=present,
        alphac=optics.alphac if optics is not None else None,
        tuning_matrix=tuning_matrix,
        matrix_drives_a_correction=_drives_a_correction,
    )


def _kind_sections(
    kinds: Sequence[str], context: ViewContext, measurement: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Each listed kind's tool section, the devices its writer adds after every tool, its files.

    A kind with no tool writes no tool section. Each tool section is ``type``
    and ``name``, the writer's keys, then the kind's ``tool_keys`` the
    measurement file states.
    """
    from osprey.facility.kinds import registered

    declared = registered()
    tools: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    for kind in kinds:
        declaration = declared[kind]
        if declaration.tool is not None and declaration.view_section is not None:
            written = declaration.view_section(context)
            section: dict[str, Any] = {
                "type": declaration.tool.type,
                "name": declaration.tool.name,
                **written.fields,
            }
            for key in declaration.tool_keys:
                if key in measurement:
                    section[key] = _measurement_value(measurement, key)
            tools.append(section)
            extras.extend(dict(device) for device in written.devices)
        for name, writer in declaration.view_files.items():
            text = writer(context)
            if text is not None:
                files[name] = text
    return tools, extras, files


def kind_records(
    records: FacilityRecords, model: str, configuration: Mapping[str, Any], files: Sequence[str]
) -> dict[str, Any]:
    """A model's kind block, as the facts document records it beside its pyAML view.

    Read from the view's configuration as written and the model's measurement
    file: ``kinds`` holds each kind the file allows, ``names`` the view's name
    for each role that has one (a group role's array, the RF plant, the tune
    monitor), ``instruments`` each instrument's address and ``rf`` the setpoint
    each of the view's RF plants steps.

    Args:
        records: The facility file's records.
        model: A model with a pyAML view.
        configuration: The view's configuration, parsed.
        files: The names of the view's other files.

    Returns:
        ``{kinds, names, instruments, rf}``. Each kind is ``{members: {role:
        {requirement, addresses}}, stepped_roles, tool_keys, tool, single_pass,
        matrix, takes_by, corrector_roles, view_files, measure_plan,
        description}``: a group role's addresses are those its array's members
        stand for, a magnet's setpoint and both planes of a BPM, or, for a role
        the view writes no array for, those :func:`measurement_groups` resolves;
        an instrument's its one channel.
    """
    from osprey.facility.kinds import member_sorts, registered

    arrays = {str(a["name"]): list(a["elements"]) for a in configuration.get("arrays") or []}
    devices = {str(d["name"]): d for d in configuration.get("devices") or []}
    measurement = records.require("model", model).get("measurement") or {}
    instruments = {
        str(role): str(address)
        for role, address in sorted((measurement.get("instruments") or {}).items())
        if address is not None
    }
    names = view_names(records, model)
    array_of = {
        str(role): names.array_name(str(role))
        for role in measurement.get("groups") or {}
        if role in GROUP_ROLES and names.array_name(str(role)) in arrays
    }
    group_addresses = {
        role: [
            address for member in arrays[array] for address in _member_addresses(devices[member])
        ]
        for role, array in array_of.items()
    }
    declared = registered()
    listed = [declared[str(kind)] for kind in measurement.get("kinds") or []]
    group_addresses |= {
        role: addresses
        for role, addresses in measurement_groups(records, model, member_sorts(listed)).items()
        if role not in GROUP_ROLES
    }
    plants = [device for device in devices.values() if device.get("type") == RF_PLANT_TYPE]
    rf = sorted(_parse_reference(d["masterclock"]) for d in plants if "masterclock" in d)
    role_names = dict(array_of)
    if plants and "rf" in instruments:
        role_names["rf"] = str(plants[-1]["name"])
    monitors = [d for d in devices.values() if d.get("type") == TUNE_MONITOR_TYPE]
    if monitors and "tune" in instruments:
        role_names["tune"] = str(monitors[-1]["name"])
    kinds: dict[str, Any] = {}
    for kind in measurement.get("kinds") or []:
        declaration = declared[str(kind)]
        members = {
            role: {
                "requirement": requirement,
                "addresses": list(group_addresses[role])
                if role in group_addresses
                else ([instruments[role]] if role in instruments else []),
            }
            for role, requirement in declaration.members.items()
        }
        kinds[str(kind)] = {
            "members": members,
            "stepped_roles": list(declaration.stepped_roles),
            "tool_keys": list(declaration.tool_keys),
            "tool": (
                {"type": declaration.tool.type, "name": declaration.tool.name}
                if declaration.tool is not None
                else None
            ),
            "single_pass": declaration.single_pass,
            "matrix": declaration.matrix,
            "takes_by": declaration.takes_by,
            "corrector_roles": list(declaration.corrector_roles),
            "view_files": [name for name in declaration.view_files if name in files],
            "measure_plan": dict(declaration.measure_plan),
            "description": declaration.description,
        }
    return {"kinds": kinds, "names": role_names, "instruments": instruments, "rf": rf}


def _parse_reference(reference: str) -> str:
    from pyaml_cs_osprey.catalog import parse_reference

    return str(parse_reference(reference).address)


def _member_addresses(device: Mapping[str, Any]) -> list[str]:
    """The addresses an array member stands for: a magnet's setpoint, a BPM's readbacks."""
    if "model" in device:
        return [_parse_reference(device["model"]["powerconverter"])]
    return sorted(_parse_reference(device[key]) for key in ("x_pos", "y_pos") if key in device)


def _design_lattice(
    inputs: ViewInputs, record: Mapping[str, Any], magnets: Sequence[_Magnet]
) -> str:
    """The design simulator's lattice text.

    The model's engine copies the deck with the elements the view's correctors
    drive carrying their kicks as polynomials (``polynomial_kicks``). For a
    view driving no corrector, an engine without that function gives the deck
    as it is.

    Raises:
        FacilityBuildError: ``engine-invalid``: the view drives a corrector and
            the model's engine states no ``polynomial_kicks``.
    """
    from importlib import metadata

    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.validate import stating_files
    from osprey.simulation.engines import ENTRY_POINT_GROUP

    deck = inputs.facility_dir / str(record["deck"])
    elements = sorted(
        {
            element
            for magnet in magnets
            if magnet.role in ROLE_PLANES
            for element, _ in _slice_elements(magnet.entry)
        }
    )
    engines = metadata.entry_points(group=ENTRY_POINT_GROUP)
    engine = str(record["engine"])
    model = str(record["name"])
    convert = (
        getattr(engines[engine].load(), "polynomial_kicks", None)
        if engine in engines.names
        else None
    )
    if convert is None and not elements:
        return deck.read_bytes().decode("utf-8")
    if convert is None:
        raise FacilityBuildError(
            "engine-invalid",
            model,
            stating_files(record, None, fallback=_MODELS_FILE),
            "add polynomial_kicks() to the engine plug-in",
            record_kind="model",
            detail=(
                f"engine {engine} states no polynomial_kicks(), so model {model}'s design "
                "correctors would carry no kick"
            ),
        )
    return str(convert(deck, elements))


def _shown(addresses: Sequence[str]) -> str:
    """The first three of ``addresses``, with an ellipsis when there are more."""
    return ", ".join(addresses[:3]) + (", …" if len(addresses) > 3 else "")


def write_pyaml_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write every measured model's pyAML view into ``root``.

    Each served model without a view is named in a note on stderr, once per build.

    Args:
        root: The render's ``data/pyaml`` directory.
        inputs: The render's view inputs.

    Returns:
        The files written, in the order they were written.
    """
    written_models, omitted = measured_models(inputs)
    for model, reason in omitted.items():
        report_note(inputs, _omitted_note(model, reason))
    written: list[Path] = []
    for model in written_models:
        configuration, files, unmodelled, mixed = _configuration(inputs, model)
        if unmodelled:
            report_note(
                inputs,
                f"view pyaml: {model} leaves out {len(unmodelled)} "
                f"setpoint{'' if len(unmodelled) == 1 else 's'} pyAML has no magnet model "
                f"for: {_shown(unmodelled)}",
            )
        if mixed:
            report_note(
                inputs,
                f"view pyaml: {model} leaves out {len(mixed)} "
                f"setpoint{'' if len(mixed) == 1 else 's'} whose slices mix zero-length and "
                f"thick elements, which pyAML drives with the wrong strength: {_shown(mixed)}",
            )
        directory = root / model
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / CONFIGURATION_FILE
        target.write_text(_yaml_text(configuration), encoding="utf-8")
        written.append(target)
        for name, text in sorted(files.items()):
            path = directory / name
            path.write_text(text, encoding="utf-8")
            written.append(path)
    return written
