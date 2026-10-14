"""Read what the ``pyaml_measure`` sandbox script printed, and label it.

The script ends with one ``OSPREY_PYAML_MEASURE <json>`` line on stdout, also
when the measurement stopped, failed or was interrupted.
:func:`parse_measure_result` reads that line and returns the tool's result for
one kind:

* every result carries ``kind``, ``model``, ``completed`` and ``sha256``, the
  digest of the pyAML view the script loaded, the ``modulation`` (and ``by``)
  the script ran, and ``moved``: ``[address, before]`` for every setpoint the
  call moved, in first-write order, with the value it held before the call (a
  non-finite value spelled as Python spells it);
* the steps: ``deltas`` by variable and ``delta_limited_by``, the setpoint
  whose ``max_step`` bounded a step; a group adds ``delta_basis``; an RF step
  adds ``rf_span_hz`` and ``rf_limited_by``, the chromaticity monitor
  ``e_delta_used``;
* a completed ``orm``, ``dispersion``, ``trm`` or ``crm``: ``shape``,
  ``observables`` (``[name, plane]`` per row, the plane read as ``x`` or
  ``y``), ``variables`` (one name per column), the steps used and
  ``data_file``;
* a completed ``chromaticity_monitor``: ``chromaticity`` ``[xi_x, xi_y]`` and
  ``data_file``;
* a stopped run: the steps used, ``remainder``, the variables not measured
  (the kind itself when no column is partial), and ``error``, the first line
  of what stopped it, when an exception did;
* a refusal: ``{"kind", "refused"}``.

:func:`label_result` then names every row and column by the address and device
id it stands for, through the view's one name mapping
(:mod:`pyaml_cs_osprey.names`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "RESULT_TAG",
    "MeasureLabels",
    "MeasureResultError",
    "label_result",
    "measure_labels",
    "parse_measure_result",
]

#: The tag of the script's one result line.
RESULT_TAG = "OSPREY_PYAML_MEASURE"

#: The kinds whose result is a response matrix.
_MATRIX_KINDS = ("orm", "dispersion", "trm", "crm")

#: The planes a row may be read in.
_PLANES = ("x", "y")

#: Each plane spelling pyAML labels a row with, matched without case, and the
#: plane it is read in.
_PLANE_SPELLINGS = {"h": "x", "x": "x", "v": "y", "y": "y"}

#: How a row is labelled, said when a label is refused.
_LABELLED = "observables are labelled <name>.<plane> with plane H, V, x or y"

#: The request values the script echoes: the stepping it ran and, for a group,
#: what sized its step.
_ECHOED = ("modulation", "by", "delta_basis")

#: The setpoints that bounded a step, each an address or null.
_NAMED = ("delta_limited_by", "rf_limited_by")

#: The RF step the chromaticity monitor or the dispersion took.
_RF_NUMBERS = ("e_delta_used", "rf_span_hz")


class MeasureResultError(ValueError):
    """The script's output does not hold a well-formed result for the kind."""


def _tagged(text: str, tag: str) -> list[str]:
    """The payload of every line starting with ``tag``, in order."""
    prefix = tag + " "
    return [line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)]


def _last_object(text: str, tag: str) -> dict[str, Any] | None:
    """The last ``tag`` line's JSON object, or ``None`` when no such line was printed."""
    payloads = _tagged(text, tag)
    if not payloads:
        return None
    try:
        value = json.loads(payloads[-1])
    except ValueError as exc:
        raise MeasureResultError(f"the {tag} line is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise MeasureResultError(f"the {tag} line is not a JSON object")
    return value


def _require(line: Mapping[str, Any], name: str) -> Any:
    if name not in line:
        raise MeasureResultError(f"the {RESULT_TAG} line has no {name!r} field")
    return line[name]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise MeasureResultError(f"{name!r} is not a finite number: {value!r}")
    return float(value)


def _string(line: Mapping[str, Any], name: str) -> str:
    value = _require(line, name)
    if not isinstance(value, str):
        raise MeasureResultError(f"{name!r} is not a string: {value!r}")
    return value


def _names(line: Mapping[str, Any], name: str) -> list[str]:
    value = _require(line, name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise MeasureResultError(f"{name!r} is not a list of names: {value!r}")
    return list(value)


def _observables(line: Mapping[str, Any]) -> list[list[str]]:
    """Each row's ``[name, plane]``, the plane read as ``x`` or ``y``."""
    value = _require(line, "observables")
    if not isinstance(value, list) or not all(
        isinstance(item, list) and len(item) == 2 and all(isinstance(part, str) for part in item)
        for item in value
    ):
        raise MeasureResultError(f"'observables' is not a list of [name, plane]: {value!r}")
    rows = []
    for name, plane in value:
        if not name:
            raise MeasureResultError(f"observable with plane {plane!r} has no name; {_LABELLED}")
        if not plane:
            raise MeasureResultError(f"observable {name!r} has no plane; {_LABELLED}")
        read = _PLANE_SPELLINGS.get(plane.lower())
        if read is None:
            raise MeasureResultError(f"observable {name!r} has plane {plane!r}; {_LABELLED}")
        rows.append([name, read])
    return rows


def _shape(line: Mapping[str, Any]) -> list[int]:
    value = _require(line, "shape")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in value)
    ):
        raise MeasureResultError(f"'shape' is not two non-negative integers: {value!r}")
    return list(value)


def _moved(line: Mapping[str, Any]) -> list[list[Any]]:
    """Each moved setpoint's ``[address, before]``, the value a number or a spelling."""
    value = _require(line, "moved")
    if not isinstance(value, list) or not all(
        isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) for item in value
    ):
        raise MeasureResultError(f"'moved' is not a list of [address, value]: {value!r}")
    return [
        [address, before if isinstance(before, str) else _number(before, f"moved.{address}")]
        for address, before in value
    ]


def _steps(line: Mapping[str, Any]) -> dict[str, Any]:
    """The steps the script took, as it reported them: ``deltas`` by variable."""
    value = _require(line, "deltas")
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise MeasureResultError(f"'deltas' is not a mapping of names to steps: {value!r}")
    return {key: _number(step, f"deltas.{key}") for key, step in value.items()}


def parse_measure_result(stdout: str | bytes, kind: str) -> dict[str, Any]:
    """The ``pyaml_measure`` result for ``kind`` from the script's output.

    Args:
        stdout: Everything the script printed to stdout.
        kind: The kind the script was built for.

    Returns:
        The result described in the module docstring.

    Raises:
        MeasureResultError: No result line was printed (the script was killed
            or failed before it could print one), it is not a JSON object, it
            names another kind, or a field the kind needs is missing or
            mistyped.
    """
    out = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout or ""

    line = _last_object(out, RESULT_TAG)
    if line is None:
        raise MeasureResultError(f"the script printed no {RESULT_TAG} line")
    if line.get("kind") != kind:
        raise MeasureResultError(
            f"the {RESULT_TAG} line is for kind {line.get('kind')!r}, not {kind!r}"
        )
    if "refused" in line:
        return {"kind": kind, "refused": str(line["refused"])}

    completed = _require(line, "completed")
    if not isinstance(completed, bool):
        raise MeasureResultError(f"'completed' is not a boolean: {completed!r}")
    result: dict[str, Any] = {
        "kind": kind,
        "model": _string(line, "model"),
        "sha256": _string(line, "sha256"),
        "completed": completed,
        "moved": _moved(line),
    }
    for name in _ECHOED:
        if name in line:
            result[name] = _string(line, name)
    if kind in _MATRIX_KINDS:
        result["deltas"] = _steps(line)
    for name in _NAMED:
        if name in line:
            value = line[name]
            if value is not None and not isinstance(value, str):
                raise MeasureResultError(f"{name!r} is not a name or null: {value!r}")
            result[name] = value
    for name in _RF_NUMBERS:
        if name in line:
            result[name] = _number(line[name], name)

    if completed:
        if kind in _MATRIX_KINDS:
            result["shape"] = _shape(line)
            result["observables"] = _observables(line)
            result["variables"] = _names(line, "variables")
            if result["shape"] != [len(result["observables"]), len(result["variables"])]:
                raise MeasureResultError(
                    f"'shape' {result['shape']} does not match {len(result['observables'])} "
                    f"observables and {len(result['variables'])} variables"
                )
        else:
            value = _require(line, "chromaticity")
            if not isinstance(value, list) or len(value) != 2:
                raise MeasureResultError(f"'chromaticity' is not [xi_x, xi_y]: {value!r}")
            result["chromaticity"] = [_number(item, "chromaticity") for item in value]
        result["data_file"] = _string(line, "data_file")
        return result

    if "error" in line:
        result["error"] = _string(line, "error")
    remainder = line.get("remainder")
    result["remainder"] = _names(line, "remainder") if isinstance(remainder, list) else kind
    return result


@dataclass(frozen=True)
class MeasureLabels:
    """What each pyAML name of one view stands for.

    Attributes:
        bpms: BPM name to ``{device, x, y}``, each plane's readback address.
        magnets: Magnet name to ``{address, device}``.
        arrays: Array name to its group id.
        rf: The RF plant's name and ``{address, device}``, or ``None``.
        tune: The tune monitor's name and ``{x, y}`` readback addresses, or
            ``None``.
        devices: Channel address to the device id it belongs to, ``None``
            for a channel of no device.
    """

    bpms: Mapping[str, Mapping[str, str | None]] = field(default_factory=dict)
    magnets: Mapping[str, Mapping[str, str | None]] = field(default_factory=dict)
    arrays: Mapping[str, str] = field(default_factory=dict)
    rf: tuple[str, Mapping[str, str | None]] | None = None
    tune: tuple[str, Mapping[str, str | None]] | None = None
    devices: Mapping[str, str | None] = field(default_factory=dict)


def measure_labels(render_root: Path, model: str, config_path: Path) -> MeasureLabels:
    """The labels of a model's pyAML view, read from the render.

    Args:
        render_root: The directory holding the rendered ``config.yml``.
        model: The model the view is for.
        config_path: The view's configuration.

    Returns:
        The labels of every BPM, magnet, array, RF plant and tune monitor the
        view holds.
    """
    import yaml

    from osprey.facility import FACILITY_FILE
    from osprey.facility.views.pyaml import view_names
    from pyaml_cs_osprey.catalog import parse_reference
    from pyaml_cs_osprey.names import RF_PLANT_NAME, TUNE_MONITOR_NAME

    doc = json.loads((render_root / FACILITY_FILE).read_text(encoding="utf-8"))
    names = view_names(doc, model)
    device_of = {
        str(channel["id"]): (channel.get("on") or {}).get("device")
        for channel in doc.get("channels", [])
    }
    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    devices = {str(device.get("name")): device for device in configuration.get("devices") or []}

    bpms: dict[str, dict[str, str | None]] = {}
    for name, device_id in sorted(_inverse(names.bpms).items()):
        entry: dict[str, str | None] = {"device": device_id}
        section = devices.get(name) or {}
        for plane in _PLANES:
            reference = section.get(f"{plane}_pos")
            entry[plane] = parse_reference(reference).address if reference else None
        bpms[name] = entry
    magnets = {
        name: {"address": address, "device": device_of.get(address)}
        for name, address in sorted(_inverse(names.magnets).items())
    }
    arrays = {name: group for (_role, group), name in sorted(names.arrays.items())}
    rf = None
    if names.rf is not None:
        rf = (RF_PLANT_NAME, {"address": names.rf, "device": device_of.get(names.rf)})
    tune = None
    monitor = devices.get(TUNE_MONITOR_NAME)
    if monitor is not None:
        tune = (
            TUNE_MONITOR_NAME,
            {
                "x": parse_reference(monitor["tune_h"]).address,
                "y": parse_reference(monitor["tune_v"]).address,
            },
        )
    return MeasureLabels(
        bpms=bpms, magnets=magnets, arrays=arrays, rf=rf, tune=tune, devices=device_of
    )


def _inverse(table: Mapping[str, str]) -> dict[str, str]:
    return {name: key for key, name in table.items()}


def _row(labels: MeasureLabels, name: str, plane: str) -> dict[str, Any]:
    bpm = labels.bpms.get(name)
    if bpm is not None:
        return {"name": name, "plane": plane, "address": bpm.get(plane), "device": bpm["device"]}
    if labels.tune is not None and name == labels.tune[0]:
        return {"name": name, "plane": plane, "address": labels.tune[1][plane], "device": None}
    return {"name": name, "plane": plane, "address": None, "device": None}


def _column(labels: MeasureLabels, name: str) -> dict[str, Any]:
    magnet = labels.magnets.get(name)
    if magnet is not None:
        return {"name": name, "address": magnet["address"], "device": magnet["device"]}
    if name in labels.arrays:
        return {"name": name, "group": labels.arrays[name]}
    if labels.rf is not None and name == labels.rf[0]:
        return {"name": name, **labels.rf[1]}
    return {"name": name, "address": None, "device": None}


def label_result(result: Mapping[str, Any], labels: MeasureLabels) -> dict[str, Any]:
    """``result`` with each row, column and step named by address and device id.

    A row is ``{name, plane, address, device}``: a BPM's readback in that plane
    and its device, a tune row's readback, or no address for a chromaticity
    row. A column is ``{name, address, device}`` for a magnet or the RF plant,
    ``{name, group}`` for a group stepped as one knob. ``deltas`` and
    ``remainder`` are keyed and listed by the same address, or group id, where
    one exists. Each ``moved`` entry is ``{address, device, before}``.

    Args:
        result: A result :func:`parse_measure_result` returned.
        labels: The labels of the view the script loaded.

    Returns:
        A new result; a refusal is returned unchanged.
    """
    labelled = dict(result)
    if "observables" in result:
        labelled["rows"] = [_row(labels, name, plane) for name, plane in result["observables"]]
        del labelled["observables"]
    if "variables" in result:
        labelled["columns"] = [_column(labels, name) for name in result["variables"]]
        del labelled["variables"]
    if "deltas" in result:
        labelled["deltas"] = {_key(labels, name): step for name, step in result["deltas"].items()}
    if "moved" in result:
        labelled["moved"] = [
            {"address": address, "device": labels.devices.get(address), "before": before}
            for address, before in result["moved"]
        ]
    if isinstance(result.get("remainder"), list):
        labelled["remainder"] = [_key(labels, name) for name in result["remainder"]]
    return labelled


def _key(labels: MeasureLabels, name: str) -> str:
    """The address or group id a variable name stands for, else the name."""
    column = _column(labels, name)
    return str(column.get("address") or column.get("group") or name)
