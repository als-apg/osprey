"""The sandbox script the ``pyaml_measure`` tool runs.

:func:`build_measure_script` renders one self-contained Python script for a
resolved measurement request. The script runs in the executor sandbox
(``_execution_dir`` names the run's execution folder) and:

1. reads the render's facts record of the model's pyAML view
   (``<render>/data/facility_facts.json``), copies the configuration and every
   file the record lists into the execution folder, and refuses, before any
   setpoint moves, unless the configuration's digest equals the record's and,
   when the call was approved, the approved digest, and every listed file
   equals its recorded digest; the copy is what is loaded;
2. loads the copy with pyAML and sizes the step from the view's own tool
   settings, so that no stepped setpoint's largest single write exceeds half
   its ``max_step`` (limits.yaml, converted to the hardware unit pyAML works
   in); a group stepped as one knob is first sized from the view's
   ``trm.json``/``crm.json`` so the tune or chromaticity moves no more than
   under the strongest single-magnet step, and never by more than one
   magnet's step; a strength, ``max_step`` or write that is not a finite
   number, or a ``max_step`` not above zero, refuses the measurement;
3. runs the measurement through ``pyaml_cs_osprey.run_tool.run_tool``, called
   once at the top level of the script, never inside a guarded run the script
   opens itself: bipolar orbit responses, dispersion and magnet-by-magnet tune
   and chromaticity responses run pyAML's own tools, every other stepping runs
   ``pyaml_cs_osprey.measure.StepMeasurement.measure``; a completed
   measurement is saved as JSON, every array as a list;
4. prints one ``OSPREY_PYAML_MEASURE <json>`` line. The guarded run prints its
   restore report on stderr.

Result line fields (JSON object):

* ``kind``, ``model``, ``sha256`` (the loaded configuration's digest),
  ``completed``, ``modulation``; ``by`` for ``trm`` and ``crm``;
* ``orm``, ``dispersion``, ``trm``, ``crm``: ``deltas`` (the step of each
  variable: integrated strength for a magnet or a group, Hz for the RF) and
  ``delta_limited_by`` (the setpoint whose ``max_step`` bounded a step, or
  ``null``); a group adds ``delta_basis``, one of :data:`DELTA_BASES`;
* ``crm``, ``chromaticity_monitor``: ``e_delta_used``, ``rf_span_hz`` and
  ``rf_limited_by``; ``dispersion``: ``rf_span_hz`` and ``rf_limited_by``;
* a completed matrix kind: ``shape``, ``observables`` (``[name, plane]``),
  ``variables`` and ``data_file``; a completed ``chromaticity_monitor``:
  ``chromaticity`` and ``data_file``;
* a stopped run: ``remainder``, the variables not measured, or the kind;
* a refusal: ``{"kind", "refused"}`` and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "DELTA_BASES",
    "RESULT_TAG",
    "MeasureScript",
    "build_measure_script",
]

#: The tag of the script's one result line.
RESULT_TAG = "OSPREY_PYAML_MEASURE"

#: What sized a group step: the tune or chromaticity excursion of the strongest
#: single-magnet step, one magnet's step, or a member's ``max_step``.
DELTA_BASES = ("tune excursion", "chromaticity excursion", "magnet step", "max_step")


@dataclass(frozen=True)
class MeasureScript:
    """What the script is rendered from: a resolved request and the server's lookups.

    Attributes:
        kind: The measurement kind.
        model: The model measured.
        modulation: ``bipolar`` or ``unipolar``.
        by: ``magnet`` or ``group`` for ``trm`` and ``crm``, else ``None``.
        corrector_names: The pyAML magnet name of each corrector given, in the
            order given; ``None`` for every corrector of the view.
        render_root: The directory holding the rendered ``config.yml``.
        view_path: The configuration's path under the render, as the facts
            record states it.
        approved_view_sha256: The view digest the approval prompt showed, or
            ``None`` when nobody was asked.
        settle: Seconds waited after each setpoint change
            (``sleep_between_step``, as the measurement file states it).
        n_avg_meas: Readings averaged at each setting.
        sleep_between_meas: Seconds between two averaged readings.
    """

    kind: str
    model: str
    modulation: str
    by: str | None
    corrector_names: Sequence[str] | None
    render_root: str
    view_path: str
    approved_view_sha256: str | None
    settle: float
    n_avg_meas: int
    sleep_between_meas: float


def _literal(value: object) -> str:
    """``value`` as a JSON literal that is also a Python literal of the same value.

    ``ensure_ascii`` keeps every string escape to ``\\" \\\\ \\n \\uXXXX``, all of
    which Python reads identically; ``None``, ``true`` and ``false`` are the JSON
    values Python spells differently, and none of them is embedded inside a
    container.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    return json.dumps(value, ensure_ascii=True)


def build_measure_script(script: MeasureScript) -> str:
    """Render the measurement script.

    Args:
        script: The request and the server's lookups.

    Returns:
        Python source for the executor sandbox; every value is embedded as a
        JSON literal ahead of a fixed body.
    """
    names: Mapping[str, object] = {
        "KIND": script.kind,
        "MODEL": script.model,
        "MODULATION": script.modulation,
        "BY": script.by,
        "CORRECTORS": None if script.corrector_names is None else list(script.corrector_names),
        "RENDER_ROOT": script.render_root,
        "VIEW_PATH": script.view_path,
        "APPROVED_VIEW": script.approved_view_sha256,
        "SETTLE": float(script.settle),
        "N_AVG_MEAS": int(script.n_avg_meas),
        "SLEEP_BETWEEN_MEAS": float(script.sleep_between_meas),
        "RESULT_TAG": RESULT_TAG,
        "DELTA_BASES": list(DELTA_BASES),
    }
    header = "# pyaml_measure: generated by osprey; every value below is a JSON literal.\n"
    header += "".join(f"{name} = {_literal(value)}\n" for name, value in names.items())
    return header + _BODY


_BODY = r'''
import hashlib
import json
import math
import shutil
from pathlib import Path, PurePosixPath

import numpy as np

import osprey.runtime
from pyaml.accelerator import Accelerator
from pyaml_cs_osprey import units
from pyaml_cs_osprey.catalog import parse_reference
from pyaml_cs_osprey.measure import StepMeasurement
from pyaml_cs_osprey.run_tool import run_tool

FACTS_FILE = "data/facility_facts.json"
VIEW_ROOT = "data/pyaml"
GROUP_MATRIX = {"trm": "trm.json", "crm": "crm.json"}
EXCURSION = {"trm": DELTA_BASES[0], "crm": DELTA_BASES[1]}
MAGNET_STEP = DELTA_BASES[2]
MAX_STEP = DELTA_BASES[3]
TIE_ULPS = 4
BISECTIONS = 200


class MeasureRefused(Exception):
    """The measurement was refused before any setpoint moved."""


def refuse(message):
    raise MeasureRefused(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def execution_root():
    folder = globals().get("_execution_dir")
    if folder is None:
        refuse("no execution folder to copy the pyAML view into")
    root = Path(folder).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_name(name):
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and str(path) not in ("", ".")


def copy_view(root):
    """Copy the recorded view into ``root`` after checking every digest; the copy and its digest."""
    render = Path(RENDER_ROOT)
    try:
        facts = json.loads((render / FACTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        refuse(f"{FACTS_FILE} cannot be read: {exc}")
    record = (facts.get("measurement_models") or {}).get(MODEL)
    if not isinstance(record, dict) or record.get("path") != VIEW_PATH:
        refuse(f"{FACTS_FILE} records no pyAML view of {MODEL} at {VIEW_PATH}")
    source = render / VIEW_PATH
    if not source.resolve().is_relative_to((render / VIEW_ROOT).resolve()):
        refuse(f"the pyAML view of {MODEL} does not resolve under {VIEW_ROOT}/")
    target = root / "pyaml_view" / MODEL
    target.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    digest = sha256(raw)
    if digest != record.get("sha256"):
        refuse(f"the pyAML view of {MODEL} differs from the one the build recorded")
    if APPROVED_VIEW is not None and digest != APPROVED_VIEW:
        refuse(f"the pyAML view of {MODEL} differs from the one the approval prompt showed")
    copy = target / source.name
    copy.write_bytes(raw)
    for name, expected in sorted((record.get("files") or {}).items()):
        if not safe_name(name):
            refuse(f"{FACTS_FILE} lists {name!r} outside the pyAML view of {MODEL}")
        data = (source.parent / name).read_bytes()
        if sha256(data) != expected:
            refuse(f"{name} of the pyAML view of {MODEL} differs from the one the build recorded")
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(data)
    return copy, digest


def setpoint_of(magnet):
    """(model, setpoint reference) of a magnet driven by one power converter."""
    model = magnet.model
    devices = list(model.get_device_names())
    if len(devices) != 1:
        refuse(f"cannot tell which power converter of {magnet.get_name()} its strength moves")
    return model, parse_reference(devices[0])


def finite(value, what):
    """``value`` as a float, refusing unless it is a finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        refuse(f"{what} is not a number: {value!r}")
    if not math.isfinite(number):
        refuse(f"{what} is not a finite number: {number!r}")
    return number


def positive_bound(value, what):
    """Half a ``max_step``, refusing unless the max_step is a finite number above zero."""
    bound = finite(value, f"the max_step of {what}")
    if bound <= 0.0:
        refuse(f"the max_step of {what} is not above zero: {bound!r}")
    return 0.5 * bound


def max_step(reference, owner):
    """The setpoint's max_step in the unit pyAML works in, or None."""
    limits = osprey.runtime.channel_limits(reference.address)
    try:
        return units.max_step_si(getattr(limits, "max_step", None), reference)
    except units.UnitError as exc:
        refuse(f"{owner}: {exc}")


def largest_write(model, strength, delta, address):
    """The hardware move of the largest single write a step of ``delta`` takes.

    Every hardware value and the move itself are finite, or the step is refused.
    """
    def hardware(value):
        native = float(np.asarray(model.compute_hardware_values(np.array([value])))[0])
        return finite(native, f"the hardware value of {address}")

    if MODULATION == "bipolar":
        move = abs(hardware(strength + 0.5 * delta) - hardware(strength - 0.5 * delta))
    else:
        move = abs(hardware(strength + delta) - hardware(strength))
    return finite(move, f"the largest write of {address}")


def fit_member(magnet, delta):
    """The step, no larger than ``delta``, whose largest write is within half max_step.

    Returns ``(step, address)``: ``address`` names the setpoint when its
    max_step shrank the step, else ``None``. The strength, the bound and every
    write compared are finite numbers, or the step is refused.
    """
    model, reference = setpoint_of(magnet)
    name = magnet.get_name()
    address = reference.address
    limit = max_step(reference, name)
    if limit is None:
        return delta, None
    bound = positive_bound(limit, address)
    strength = finite(magnet.strength.get(), f"the strength of {address}")
    if largest_write(model, strength, delta, address) <= bound:
        return delta, None
    low, high = 0.0, abs(delta)
    for _ in range(BISECTIONS):
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if largest_write(model, strength, math.copysign(middle, delta), address) > bound:
            high = middle
        else:
            low = middle
    if low <= 0.0:
        refuse(f"no step of {reference.address} stays within half its max_step")
    return math.copysign(low, delta), reference.address


def fit_common(delta, magnets):
    """One step for every magnet: the smallest of their fits, and the setpoint that bound it."""
    fitted, limited_by = float(delta), None
    for magnet in magnets:
        step, address = fit_member(magnet, fitted)
        if address is not None:
            fitted, limited_by = step, address
    return fitted, limited_by


def group_step(kind, delta, array, copy_dir):
    """The group step by the excursion rule, the one-magnet clamp and half max_step.

    Returns ``(step, basis, limited_by)``.
    """
    name = GROUP_MATRIX[kind]
    path = copy_dir / name
    if not path.is_file():
        refuse(f"{kind} by group needs {name}, which the pyAML view of {MODEL} does not have")
    matrix = json.loads(path.read_text(encoding="utf-8"))
    rows = np.asarray(matrix["matrix"], dtype=float)
    columns = {str(n): rows[:, j] for j, n in enumerate(matrix["variable_names"])}
    members = list(array)
    missing = [m.get_name() for m in members if m.get_name() not in columns]
    if missing:
        refuse(f"{name} has no column for {', '.join(missing)}")
    strongest = max(float(np.linalg.norm(column)) for column in columns.values())
    if strongest == 0.0:
        refuse(f"{name} gives no single magnet of {MODEL} a response")
    total = float(np.linalg.norm(sum(columns[m.get_name()] for m in members)))
    magnet_step = abs(float(delta))
    excursion = magnet_step * strongest / total if total > 0.0 else math.inf
    if excursion <= magnet_step + TIE_ULPS * math.ulp(magnet_step):
        step, basis = min(excursion, magnet_step), EXCURSION[kind]
    else:
        step, basis = magnet_step, MAGNET_STEP
    step = math.copysign(step, float(delta))
    fitted, limited_by = fit_common(step, members)
    if limited_by is not None:
        basis = MAX_STEP
    return fitted, basis, limited_by


def rf_reference(rf):
    if getattr(rf, "masterclock", None) is None:
        refuse("the RF plant has no master-clock channel to step")
    return parse_reference(rf.masterclock)


def fit_rf_span(span_hz, reference):
    """The RF span, in Hz, whose largest write is within half max_step, and the bound's address."""
    limits = osprey.runtime.channel_limits(reference.address)
    limit = getattr(limits, "max_step", None)
    if limit is None:
        return span_hz, None
    bound = positive_bound(limit, reference.address)
    try:
        native = units.to_native(span_hz, reference)
    except units.UnitError as exc:
        refuse(f"{reference.address}: {exc}")
    write = abs(finite(native, f"the RF write of {reference.address}"))
    if write <= bound:
        return span_hz, None
    return span_hz * bound / write, reference.address


def fit_monitor(monitor):
    """Fit the chromaticity monitor's momentum step to half the RF max_step."""
    alphac = getattr(monitor, "_alphac", None)
    if alphac is None:
        refuse("no momentum compaction factor (alphac) to size the RF step")
    steps = int(getattr(monitor, "n_step", 0) or 0)
    if steps < 2:
        refuse("the chromaticity monitor needs n_step of at least 2")
    reference = rf_reference(monitor.rf_plant)
    f0 = float(monitor.rf_plant.frequency.get())
    e_delta = abs(float(monitor.e_delta))
    half = abs(float(alphac) * f0 * e_delta)
    widest = half * max(1.0, 2.0 / (steps - 1))
    fitted, limited_by = fit_rf_span(widest, reference)
    if limited_by is not None:
        e_delta *= fitted / widest
    monitor.e_delta = e_delta
    if abs(float(monitor.max_e_delta)) < e_delta:
        monitor.max_e_delta = 2.0 * e_delta
    return {
        "e_delta_used": e_delta,
        "rf_span_hz": 2.0 * abs(float(alphac) * f0 * e_delta),
        "rf_limited_by": limited_by,
    }


class Recorder:
    """Names each variable a run finished, so a stopped run reports what is left."""

    def __init__(self):
        self.done = []

    def __call__(self, action, data):
        if isinstance(data, dict) and "idx" in data:
            name = data.get("magnet", data.get("group"))
            if name is not None and ("dtune" in data or "dchroma" in data or "group" in data):
                if action.name == "RESTORE" and name not in self.done:
                    self.done.append(name)
        return True


#: pyAML's plane names, spelled as the result's planes.
PLANES = {"h": "x", "v": "y", "x": "x", "y": "y"}


def observables(measured, holder, bpm_array):
    names = list(measured.get("observable_names") or [])
    planes = measured.get("observable_planes")
    if planes:
        return [[str(n), PLANES.get(str(p).lower(), str(p).lower())] for n, p in zip(names, planes)]
    if names:
        return [[n.rsplit(".", 1)[0], n.rsplit(".", 1)[1]] for n in names]
    bpms = list(holder.bpms.get(bpm_array).names())
    return [[n, "x"] for n in bpms] + [[n, "y"] for n in bpms]


def plan(holder, copy_dir):
    """The method to run, its arguments, the variables and the result's step fields."""
    fields = {}
    if KIND == "orm":
        tool = holder.orm
        members = {m.get_name(): m for m in [*tool.hcorrectors, *tool.vcorrectors]}
        names = list(members) if CORRECTORS is None else list(CORRECTORS)
        unknown = [name for name in names if name not in members]
        if unknown:
            refuse(f"not a corrector of the pyAML view of {MODEL}: {', '.join(unknown)}")
        delta = float(tool.corrector_delta)
        if MODULATION == "bipolar":
            step, limited_by = fit_common(delta, [members[n] for n in names])
            tool.corrector_delta = step
            deltas = {name: step for name in names}
            call = (tool.measure, {
                "corrector_names": names,
                "sleep_between_step": SETTLE,
                "n_avg_meas": N_AVG_MEAS,
                "sleep_between_meas": SLEEP_BETWEEN_MEAS,
            })
        else:
            deltas, limited_by = {}, None
            for name in names:
                step, address = fit_member(members[name], delta)
                deltas[name] = step
                limited_by = address if address is not None else limited_by
            measurement = StepMeasurement.orm(
                holder,
                bpm_array=tool.bpm_array_name,
                correctors=names,
                deltas=[deltas[n] for n in names],
                set_wait_time=SETTLE,
                n_avg_meas=N_AVG_MEAS,
            )
            call = (measurement.measure, {})
            tool = measurement
        fields.update(deltas=deltas, delta_limited_by=limited_by)
        return tool, call, names, fields, tool_bpm_array(holder)
    if KIND == "dispersion":
        tool = holder.dispersion
        reference = rf_reference(holder.rf.get(tool.rf_plant_name))
        span, limited_by = fit_rf_span(float(tool.frequency_delta), reference)
        fields.update(
            deltas={tool.rf_plant_name: span},
            delta_limited_by=limited_by,
            rf_span_hz=span,
            rf_limited_by=limited_by,
        )
        if MODULATION == "bipolar":
            tool.frequency_delta = span
            call = (tool.measure, {"set_waiting_time": SETTLE})
        else:
            tool = StepMeasurement.dispersion(
                holder,
                bpm_array=tool.bpm_array_name,
                rf_plant=tool.rf_plant_name,
                delta=span,
                set_wait_time=SETTLE,
                n_avg_meas=N_AVG_MEAS,
            )
            call = (tool.measure, {})
        return tool, call, list(fields["deltas"]), fields, tool_bpm_array(holder)
    if KIND in ("trm", "crm"):
        tool = holder.trm if KIND == "trm" else holder.crm
        array_name = tool.quad_array_name if KIND == "trm" else tool.sextu_array_name
        array = holder.magnets.get(array_name)
        delta = float(tool.quad_delta if KIND == "trm" else tool.sextu_delta)
        if KIND == "crm":
            fields.update(fit_monitor(tool.chromaticity_monitor))
        if BY == "group":
            step, basis, limited_by = group_step(KIND, delta, array, copy_dir)
            fields.update(deltas={array_name: step}, delta_limited_by=limited_by, delta_basis=basis)
            if KIND == "trm":
                measurement = StepMeasurement.trm(
                    holder,
                    groups=[array_name],
                    delta=step,
                    tune_monitor=tool.betatron_tune_name,
                    set_wait_time=SETTLE,
                    n_avg_meas=N_AVG_MEAS,
                    sleep_between_meas=SLEEP_BETWEEN_MEAS,
                )
            else:
                measurement = StepMeasurement.crm(
                    holder,
                    groups=[array_name],
                    delta=step,
                    chromaticity_monitor=tool.chromaticity_name,
                    set_wait_time=SETTLE,
                )
            return measurement, (measurement.measure, {}), [array_name], fields, None
        magnets = list(array)
        step, limited_by = fit_common(delta, magnets)
        names = [m.get_name() for m in magnets]
        fields.update(deltas={name: step for name in names}, delta_limited_by=limited_by)
        key = "quad_delta" if KIND == "trm" else "sextu_delta"
        arguments = {
            "sleep_between_step": SETTLE,
            "n_avg_meas": N_AVG_MEAS,
            "sleep_between_meas": SLEEP_BETWEEN_MEAS,
        }
        if MODULATION == "bipolar":
            arguments.update({key: 0.5 * step, "n_step": 2})
        else:
            arguments.update({key: -step, "n_step": 1})
        return tool, (tool.measure, arguments), names, fields, None
    monitor = holder.get_chromaticity_monitor("CHROMATICITY_MONITOR")
    fields.update(fit_monitor(monitor))
    return monitor, (monitor.measure, {}), [KIND], fields, None


def plain(value):
    """A numpy value as the JSON value it holds."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not a measurement value")


def save(tool, path):
    """Write the tool's latest measurement to ``path`` as JSON, arrays as lists."""
    path.write_text(json.dumps(tool.latest_measurement, indent=4, default=plain), encoding="utf-8")


def tool_bpm_array(holder):
    return holder.orm.bpm_array_name if KIND == "orm" else holder.dispersion.bpm_array_name


def main():
    root = execution_root()
    try:
        copy, digest = copy_view(root)
        holder = Accelerator.load(str(copy)).live
        tool, (method, arguments), variables, fields, bpm_array = plan(holder, copy.parent)
    except MeasureRefused as refusal:
        print(RESULT_TAG, json.dumps({"kind": KIND, "refused": str(refusal)}), flush=True)
        return
    result = {"kind": KIND, "model": MODEL, "sha256": digest, "modulation": MODULATION}
    if KIND in ("trm", "crm"):
        result["by"] = BY
    result.update(fields)

    recorder = Recorder()
    report = run_tool(method, callback=recorder, **arguments)

    result["completed"] = not report.aborted
    if report.aborted:
        if KIND in ("trm", "crm"):
            result["remainder"] = [name for name in variables if name not in recorder.done]
        else:
            result["remainder"] = KIND
        print(RESULT_TAG, json.dumps(result), flush=True)
        return
    data_file = root / f"pyaml_measure_{KIND}.json"
    save(tool, data_file)
    result["data_file"] = str(data_file)
    measured = tool.latest_measurement
    if KIND == "chromaticity_monitor":
        result["chromaticity"] = [float(x) for x in tool.chromaticity.get()]
    elif KIND == "dispersion" and "matrix" not in measured:
        planes = ("frequency_response_x", "frequency_response_y")
        response = np.concatenate([np.ravel(measured[plane]) for plane in planes])
        result["shape"] = [int(response.size), 1]
        result["observables"] = observables({}, holder, bpm_array)
        result["variables"] = list(fields["deltas"])
    else:
        matrix = np.asarray(measured["matrix"], dtype=float)
        result["shape"] = list(matrix.shape)
        result["observables"] = observables(measured, holder, bpm_array)
        result["variables"] = list(measured.get("variable_names") or variables)
    print(RESULT_TAG, json.dumps(result), flush=True)


main()
'''
