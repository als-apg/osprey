"""The pyAML view: one pyAML configuration per measured model.

Written to ``<render>/data/pyaml/<model>/`` for every model the render serves
that names a deck and has a ``measurement/<model>.yaml``::

    configuration.yaml   the pyAML accelerator: its design simulator, the
                         ``live`` OSPREY control system, one device per magnet,
                         BPM and instrument the measurement file's groups and
                         instruments name, one array per group and one tool per
                         measurement kind
    lattice.json         the design simulator's lattice, referenced as
                         ``${path:lattice.json}``: the model's deck with each
                         corrector the view drives carrying its kick as the
                         dipole polynomials pyAML reads, as the model's engine
                         writes it
    trm.json, crm.json   the tune and chromaticity response matrices of the
                         design optics, each with the correction tool that
                         loads it, for a periodic model whose measurement file
                         allows ``trm`` or ``crm``

A model solved ``single_pass`` has no design simulator (``simulators: []``):
pyAML's design simulator solves a periodic orbit. Every other model the render
serves names a note on stderr saying why it has no view. A setpoint pyAML cannot
convert, or a corrector whose element the engine cannot give its kick as
polynomials, is left out of the view and named in a note.

Every name the configuration holds comes from :mod:`pyaml_cs_osprey.names`, so
the view, the measurement tools and a reader of the configuration cannot spell
one two ways. Nothing in a view depends on where the render sits: references
between its files are relative, and equal sources and configs give equal bytes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.facility import TEXTURE
from osprey.facility.views import ViewInputs, report_note

if TYPE_CHECKING:
    from pyaml_cs_osprey.names import ViewNames

__all__ = [
    "CONFIGURATION_FILE",
    "CRM_FILE",
    "GROUP_ROLES",
    "KIND_MEMBERS",
    "LATTICE_FILE",
    "PYAML_DIR",
    "ROLE_PLANES",
    "TOOL_KEYS",
    "TRM_FILE",
    "measured_models",
    "measurement_groups",
    "pyaml_view_wanted",
    "view_names",
    "write_pyaml_view",
]

#: The view's directory under the render's ``data/``.
PYAML_DIR = "pyaml"

#: The configuration file of one model's view.
CONFIGURATION_FILE = "configuration.yaml"

#: The deck's copy beside the configuration.
LATTICE_FILE = "lattice.json"

#: The response matrices the correction tools load.
TRM_FILE = "trm.json"
CRM_FILE = "crm.json"

#: The measurement file's group roles, in the order the arrays are written.
GROUP_ROLES: tuple[str, ...] = ("bpm", "hcor", "vcor", "quad", "sext")

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

#: Each measurement kind's pyAML tool: its ``type:`` path and its name.
KIND_TOOLS: dict[str, tuple[str, str]] = {
    "orm": ("pyaml.tuning_tools.orbit_response_matrix", "DEFAULT_ORBIT_RESPONSE_MATRIX"),
    "dispersion": ("pyaml.tuning_tools.dispersion", "DEFAULT_DISPERSION"),
    "trm": ("pyaml.tuning_tools.tune_response_matrix", "DEFAULT_TUNE_RESPONSE_MATRIX"),
    "crm": (
        "pyaml.tuning_tools.chromaticity_response_matrix",
        "DEFAULT_CHROMATICITY_RESPONSE_MATRIX",
    ),
    "chromaticity_monitor": ("pyaml.tuning_tools.chromaticity_monitor", ""),
}

#: The correction tools a periodic model's design optics give a matrix to.
TUNE_CORRECTION = ("pyaml.tuning_tools.tune", "DEFAULT_TUNE_CORRECTION")
CHROMATICITY_CORRECTION = ("pyaml.tuning_tools.chromaticity", "DEFAULT_CHROMATICITY_CORRECTION")

#: The keys of a tool section that name an array, each with the group role it is.
ARRAY_KEYS: dict[str, str] = {
    "bpm_array_name": "bpm",
    "hcorr_array_name": "hcor",
    "vcorr_array_name": "vcor",
    "quad_array_name": "quad",
    "sextu_array_name": "sext",
}

#: The groups and instruments each measurement kind needs: a name in
#: :data:`GROUP_ROLES` is a group role, any other an instrument.
KIND_MEMBERS: dict[str, tuple[str, ...]] = {
    "orm": ("bpm", "hcor", "vcor"),
    "dispersion": ("bpm", "hcor", "vcor", "rf"),
    "trm": ("quad", "tune"),
    "crm": ("sext", "tune", "rf"),
    "chromaticity_monitor": ("tune", "rf"),
}

#: The step and settle keys each measurement tool takes from the measurement
#: file, pyAML's own spellings.
TOOL_KEYS: dict[str, tuple[str, ...]] = {
    "orm": ("corrector_delta",),
    "dispersion": ("frequency_delta",),
    "trm": ("quad_delta",),
    "crm": ("sextu_delta",),
    "chromaticity_monitor": (
        "n_step",
        "n_avg_meas",
        "fit_order",
        "sleep_between_step",
        "sleep_between_meas",
    ),
}

#: Upper bound of the chromaticity monitor's relative momentum step.
E_DELTA_CAP = 0.01
#: The orbit shift ``e_delta * alphac`` the momentum step aims for.
E_DELTA_ORBIT = 5.0e-6

#: The speed of light in m/s, exact; pyAML's rigidity is the energy in eV over it.
_SPEED_OF_LIGHT = 299792458.0

#: The solve whose model has no design simulator.
_SINGLE_PASS = "single_pass"

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
    for model in sorted(inputs.doc.get("models", []), key=lambda record: str(record["name"])):
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


def _model_record(doc: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    for record in doc.get("models", []):
        if record["name"] == model:
            found: Mapping[str, Any] = record
            return found
    raise KeyError(f"the facility file has no model {model!r}")


def _device_order(doc: Mapping[str, Any], members: Iterable[str]) -> list[str]:
    """``members`` in ``(s, id)`` order; a device with no ``s`` after every placed one."""
    placed = {str(device["id"]): device.get("s") for device in doc.get("devices", [])}

    def key(device: str) -> tuple[float, str]:
        s = placed.get(device)
        return (float(s) if isinstance(s, int | float) else _UNPLACED, device)

    return sorted(dict.fromkeys(str(member) for member in members), key=key)


def _channels_of(doc: Mapping[str, Any]) -> dict[str, list[str]]:
    """Each device id to the addresses of the channels on it or naming it an endpoint."""
    held: dict[str, list[str]] = {}
    for channel in doc.get("channels", []):
        devices = [(channel.get("on") or {}).get("device"), *(channel.get("endpoint_of") or [])]
        for device in dict.fromkeys(devices):
            if device is not None:
                held.setdefault(str(device), []).append(str(channel["id"]))
    return held


#: The plane each corrector role steers; a role not named here takes every plane.
ROLE_PLANES: dict[str, str] = {"hcor": "x", "vcor": "y"}


def measurement_groups(doc: Mapping[str, Any], model: str) -> dict[str, list[str]]:
    """The addresses each group role of a model's measurement file stands for.

    ``bpm`` is the readbacks the model wires as monitors, in both planes;
    ``hcor`` the setpoints the model wires whose plane its engine describes as
    ``x``, ``vcor`` those it describes as ``y``; ``quad`` and ``sext`` every
    setpoint the model wires. Each is taken over the channels on a member device
    of the role's group and the channels naming one as an endpoint, visiting the
    devices in ``(s, id)`` order, an address listed once at its first device and
    the addresses within a device sorted. A combined corrector, one device
    steered in both planes, keeps each setpoint in the role of its own plane, so
    ``hcor`` and ``vcor`` share no address.

    Args:
        doc: The facility file.
        model: The model's name.

    Returns:
        Role to addresses, for each role the measurement file names, in
        :data:`GROUP_ROLES` order.

    Raises:
        KeyError: The facility file has no model ``model``.
    """
    from osprey.facility.views.simulator import simulator_wiring

    record = _model_record(doc, model)
    groups_named = (record.get("measurement") or {}).get("groups") or {}
    groups = {str(group["id"]): group for group in doc.get("groups", [])}
    wiring = {str(entry["address"]): entry for entry in simulator_wiring(dict(doc), model)}
    on_device = _channels_of(doc)
    found: dict[str, list[str]] = {}
    for role in GROUP_ROLES:
        group_id = groups_named.get(role)
        if group_id is None:
            continue
        wanted = "monitor" if role == "bpm" else "setpoint"
        plane = ROLE_PLANES.get(role)
        members = (groups.get(str(group_id)) or {}).get("members") or []
        addresses: dict[str, None] = {}
        for device in _device_order(doc, members):
            for address in sorted(set(on_device.get(device, ()))):
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
    import yaml

    return yaml.safe_dump(
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


def _magnet_model(
    magnet: _Magnet, deck: _Deck, limits: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """The pyAML model one setpoint is converted by, or ``None`` when pyAML has none.

    pyAML's strength is the integrated field over the rigidity. The deck holds
    ``curve(hardware) * weight`` on each slice, so the magnet's integrated
    strength is ``curve(hardware) * scale`` with ``scale`` the sum of the slice
    weights times their lengths for a multipole and of the weights for a kick:
    a linear curve becomes a ``linear_model`` of factor ``gain * scale * brho``,
    a table an ``inline_curve`` of ``(hardware, value * scale * brho)`` rows
    extended one knot span past each end and past the address's limits band.
    A curve pyAML cannot represent (one per slice, a zero gain or scale, a table
    that is not strictly monotone) has no model.
    """
    from osprey.simulation.engines.calibration import Linear, Table, curve_from_record

    entry = magnet.entry
    calibration = entry.get("calibration") or {}
    pieces = _slice_elements(entry)
    if any((piece.get("curve") is not None) for piece in entry.get("slices") or []):
        return None
    kick = magnet.role in ("hcor", "vcor")
    scale = sum(
        weight * (1.0 if kick else deck.lengths.get(element, 0.0)) for element, weight in pieces
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


def _magnets(doc: Mapping[str, Any], model: str, groups: Mapping[str, list[str]]) -> list[_Magnet]:
    from osprey.facility.views.simulator import simulator_wiring

    wiring = {str(entry["address"]): entry for entry in simulator_wiring(dict(doc), model)}
    pairs = {
        str(channel["id"]): channel.get("pair")
        for channel in doc.get("channels", [])
        if channel.get("pair") is not None
    }
    magnets: list[_Magnet] = []
    for role in MAGNET_TYPES:
        for address in groups.get(role, ()):
            if any(magnet.address == address for magnet in magnets):
                continue
            entry = dict(wiring[address])
            entry["readback"] = pairs.get(address)
            magnets.append(_Magnet(address, role, entry))
    return magnets


def _bpms(doc: Mapping[str, Any], model: str, addresses: Sequence[str]) -> list[_Bpm]:
    """The BPM devices the ``bpm`` readbacks are on, in the readbacks' order."""
    from osprey.facility.views.simulator import simulator_wiring

    wiring = {str(entry["address"]): entry for entry in simulator_wiring(dict(doc), model)}
    device_of = {
        str(channel["id"]): str((channel.get("on") or {}).get("device"))
        for channel in doc.get("channels", [])
    }
    bpms: dict[str, dict[str, Any]] = {}
    for address in addresses:
        entry = wiring[address]
        device = device_of[address]
        held = bpms.setdefault(device, {"element": _slice_elements(entry)[0][0], "planes": {}})
        held["planes"].setdefault(str(entry.get("plane")), address)
    return [
        _Bpm(device, str(held["element"]), dict(held["planes"])) for device, held in bpms.items()
    ]


def _units(doc: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(channel["id"]): str(channel["unit"])
        for channel in doc.get("channels", [])
        if channel.get("unit")
    }


def _tune_references(doc: Mapping[str, Any], model: str, address: str) -> tuple[str, str] | None:
    """The horizontal and vertical tune references the ``tune`` instrument stands for.

    A waveform reading the whole tune output is read by element: ``@0`` and
    ``@1``. A scalar reading one plane is paired with the model's reading on the
    other plane whose engine block differs from its own in one key alone, the
    one that names the plane.
    """
    from osprey.facility.views.simulator import simulator_wiring

    entries = simulator_wiring(dict(doc), model)
    wired = {str(entry["address"]): entry for entry in entries}
    entry = wired.get(address)
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
    """``sum(q_i * w_i * L_i) / sum(w_i * L_i)`` over the magnet's slices."""
    weighted = total = 0.0
    for element, weight in _slice_elements(magnet.entry):
        position = optics.positions.get(element)
        if position is None:
            return None
        share = weight * deck.lengths.get(element, 0.0)
        weighted += float(quantity(position)) * share
        total += share
    return None if total == 0.0 else weighted / total


def _response_matrix(
    columns: Sequence[tuple[str, float, float]], monitor: str
) -> dict[str, Any] | None:
    """A two-row response matrix, or ``None`` when it cannot drive a correction."""
    import numpy as np

    if not columns:
        return None
    rows = [
        [float(f"{x:.9g}") for _, x, _ in columns],
        [float(f"{y:.9g}") for _, _, y in columns],
    ]
    matrix = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(matrix)) or int(np.linalg.matrix_rank(matrix)) < 2:
        return None
    return {
        "type": RESPONSE_MATRIX_DATA_TYPE,
        "matrix": rows,
        "observable_names": [f"{monitor}.x", f"{monitor}.y"],
        "variable_names": [name for name, _, _ in columns],
    }


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
    magnet's slices by weight times length: the response per unit of the
    integrated strength pyAML steps.
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
    from pyaml_cs_osprey.names import ViewNames

    return ViewNames.build(
        magnets=[magnet.address for magnet in magnets],
        bpms=[bpm.device for bpm in bpms],
        groups=[str(groups_named[role]) for role in GROUP_ROLES if role in groups_named],
        rf=str(rf) if rf is not None else None,
    )


def view_names(doc: Mapping[str, Any], model: str) -> ViewNames:
    """The names a model's pyAML view gives its magnets, BPMs, arrays and RF plant.

    Args:
        doc: The facility file.
        model: A model with a measurement file.

    Returns:
        The view's names, built from the facility file as the view writer
        builds them.

    Raises:
        KeyError: The facility file has no model ``model``.
        ValueError: Two addresses or ids of the view would share a name.
    """
    record = _model_record(doc, model)
    measurement = record.get("measurement") or {}
    groups = measurement_groups(doc, model)
    return _names(
        _magnets(doc, model, groups),
        _bpms(doc, model, groups.get("bpm", ())),
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
        ``(configuration, files, unmodelled, unkicked)``: the configuration as
        a mapping, each file beside it by name with its text, the setpoints
        left out for want of a pyAML magnet model and the corrector setpoints
        left out for want of an element that carries their kick as
        polynomials, each in group order.
    """
    from pyaml_cs_osprey.names import (
        CHROMATICITY_MONITOR_NAME,
        RF_PLANT_NAME,
        TUNE_MONITOR_NAME,
    )

    doc = inputs.doc
    record = _model_record(doc, model)
    measurement = record.get("measurement") or {}
    solve = ((record.get("settings") or {}).get(str(record["engine"])) or {}).get("solve")
    periodic = solve != _SINGLE_PASS
    deck = _load_deck(inputs.facility_dir / str(record["deck"]))
    groups = measurement_groups(doc, model)
    groups_named = measurement.get("groups") or {}
    instruments = measurement.get("instruments") or {}
    kinds = list(measurement.get("kinds") or [])
    units = _units(doc)
    limits = {str(row["address"]): row for row in (doc.get("limits") or {}).get("records") or []}

    magnets = _magnets(doc, model, groups)
    bpms = _bpms(doc, model, groups.get("bpm", ()))
    rf = instruments.get("rf")
    names = _names(magnets, bpms, groups_named, rf)
    models = {magnet.address: _magnet_model(magnet, deck, limits) for magnet in magnets}
    unmodelled = [address for address, found in models.items() if found is None]
    files: dict[str, str] = {}
    unkicked: list[str] = []
    if periodic:
        files[LATTICE_FILE], refused = _design_lattice(inputs, record, magnets)
        unkicked = [
            magnet.address
            for magnet in magnets
            if magnet.role in ROLE_PLANES
            and models[magnet.address] is not None
            and any(element in refused for element, _ in _slice_elements(magnet.entry))
        ]
    left_out = set(unmodelled) | set(unkicked)
    magnets = [magnet for magnet in magnets if magnet.address not in left_out]
    groups = {
        role: [address for address in addresses if address not in left_out]
        for role, addresses in groups.items()
    }

    configuration: dict[str, Any] = {
        "type": ACCELERATOR_TYPE,
        "facility": str(doc["identity"]["code"]),
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
                "name": names.array_name(str(groups_named[role])),
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
                section[f"{plane}_pos"] = _reference(address, units.get(address))
        devices.append(section)
    if rf is not None:
        devices.append(
            {
                "type": RF_PLANT_TYPE,
                "name": names.rf_plant_name(str(rf)),
                "masterclock": _reference(str(rf), units.get(str(rf))),
            }
        )
    tune = instruments.get("tune")
    tunes = _tune_references(doc, model, str(tune)) if tune is not None else None
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

    array_of = {role: names.array_name(str(group)) for role, group in groups_named.items()}
    for kind in kinds:
        tool_type, tool_name = KIND_TOOLS[kind]
        section = {"type": tool_type}
        if kind == "orm":
            section |= {
                "name": tool_name,
                "bpm_array_name": array_of["bpm"],
                "hcorr_array_name": array_of["hcor"],
                "vcorr_array_name": array_of["vcor"],
            }
        elif kind == "dispersion":
            section |= {
                "name": tool_name,
                "bpm_array_name": array_of["bpm"],
                "rf_plant_name": RF_PLANT_NAME,
            }
        elif kind == "trm":
            if tunes is None:
                continue
            section |= {
                "name": tool_name,
                "quad_array_name": array_of["quad"],
                "betatron_tune_name": TUNE_MONITOR_NAME,
            }
        elif kind == "crm":
            if tunes is None or rf is None:
                continue
            section |= {
                "name": tool_name,
                "sextu_array_name": array_of["sext"],
                "chromaticity_name": CHROMATICITY_MONITOR_NAME,
            }
        else:
            if tunes is None or rf is None:
                continue
            section |= {
                "name": CHROMATICITY_MONITOR_NAME,
                "betatron_tune_name": TUNE_MONITOR_NAME,
                "rf_plant_name": RF_PLANT_NAME,
            }
            alphac = abs(optics.alphac) if optics is not None and optics.alphac else 0.0
            e_delta = (
                min(E_DELTA_CAP, E_DELTA_ORBIT / alphac)
                if math.isfinite(alphac) and alphac > 0
                else E_DELTA_CAP
            )
            section |= {"e_delta": e_delta, "max_e_delta": 2.0 * e_delta}
        for key in TOOL_KEYS[kind]:
            if key in measurement:
                section[key] = _measurement_value(measurement, key)
        devices.append(section)
    if "crm" in kinds and "chromaticity_monitor" not in kinds and tunes is not None and rf:
        devices.append(
            {
                "type": KIND_TOOLS["chromaticity_monitor"][0],
                "name": CHROMATICITY_MONITOR_NAME,
                "betatron_tune_name": TUNE_MONITOR_NAME,
                "rf_plant_name": RF_PLANT_NAME,
            }
        )

    if optics is not None:
        if "trm" in kinds and tunes is not None:
            trm = _tuning_matrix(
                "quad", groups.get("quad", ()), magnets, names, deck, optics, TUNE_MONITOR_NAME
            )
            if trm is not None:
                files[TRM_FILE] = _matrix_text(trm)
                tool_type, tool_name = TUNE_CORRECTION
                devices.append(
                    {
                        "type": tool_type,
                        "name": tool_name,
                        "quad_array_name": array_of["quad"],
                        "betatron_tune_name": TUNE_MONITOR_NAME,
                        "response_matrix": _path_reference(TRM_FILE),
                    }
                )
        if "crm" in kinds and tunes is not None and rf is not None:
            crm = _tuning_matrix(
                "sext",
                groups.get("sext", ()),
                magnets,
                names,
                deck,
                optics,
                CHROMATICITY_MONITOR_NAME,
            )
            if crm is not None:
                files[CRM_FILE] = _matrix_text(crm)
                tool_type, tool_name = CHROMATICITY_CORRECTION
                devices.append(
                    {
                        "type": tool_type,
                        "name": tool_name,
                        "sextu_array_name": array_of["sext"],
                        "chromaticity_monitor_name": CHROMATICITY_MONITOR_NAME,
                        "response_matrix": _path_reference(CRM_FILE),
                    }
                )
    configuration["devices"] = devices
    return configuration, files, unmodelled, unkicked


def _design_lattice(
    inputs: ViewInputs, record: Mapping[str, Any], magnets: Sequence[_Magnet]
) -> tuple[str, frozenset[str]]:
    """The design simulator's lattice text and the corrector elements it could not convert.

    The model's engine copies the deck with the elements the view's correctors
    drive carrying their kicks as polynomials (``polynomial_kicks``); an engine
    without that function gives the deck as it is.
    """
    from importlib import metadata

    from osprey.simulation.engines import ENTRY_POINT_GROUP

    deck = inputs.facility_dir / str(record["deck"])
    engines = metadata.entry_points(group=ENTRY_POINT_GROUP)
    engine = str(record["engine"])
    convert = (
        getattr(engines[engine].load(), "polynomial_kicks", None)
        if engine in engines.names
        else None
    )
    if convert is None:
        return deck.read_bytes().decode("utf-8"), frozenset()
    elements = sorted(
        {
            element
            for magnet in magnets
            if magnet.role in ROLE_PLANES
            for element, _ in _slice_elements(magnet.entry)
        }
    )
    copy = convert(deck, elements)
    return str(copy.text), frozenset(copy.refused)


def write_pyaml_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write every measured model's pyAML view into ``root``.

    Each served model without a view is named in a note on stderr, once per build.

    Args:
        root: The render's ``data/pyaml`` directory.
        inputs: The render's view inputs.

    Returns:
        The files written, sorted.
    """
    written_models, omitted = measured_models(inputs)
    for model, reason in omitted.items():
        report_note(inputs, _omitted_note(model, reason))
    written: list[Path] = []
    for model in written_models:
        configuration, files, unmodelled, unkicked = _configuration(inputs, model)
        if unkicked:
            shown = ", ".join(unkicked[:3]) + (", …" if len(unkicked) > 3 else "")
            report_note(
                inputs,
                f"view pyaml: {model} leaves out {len(unkicked)} "
                f"setpoint{'' if len(unkicked) == 1 else 's'} whose element has no length "
                f"to carry a kick: {shown}",
            )
        if unmodelled:
            shown = ", ".join(unmodelled[:3]) + (", …" if len(unmodelled) > 3 else "")
            report_note(
                inputs,
                f"view pyaml: {model} leaves out {len(unmodelled)} "
                f"setpoint{'' if len(unmodelled) == 1 else 's'} pyAML has no magnet model "
                f"for: {shown}",
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
    return sorted(written)
