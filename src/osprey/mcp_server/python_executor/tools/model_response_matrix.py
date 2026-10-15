"""MCP tool: model_response_matrix — a served model's response to its correctors, read-only.

``model_response_matrix(model=None, correctors=None, kind="orbit")`` solves the
response of a served model's wired readings to its corrector setpoints on the
model's own deck, through its engine plug-in's ``response_matrix()``. Nothing
is written to the machine, to the deck or to the render.

``kind`` names the response: ``orbit`` is the monitor readbacks' orbit
response; ``trajectory`` is the first-turn response of the turn-by-turn
readings, the content of the pyAML view's ``trajectory_rm.json`` and
``trajectory_rm.index.json`` solved for the correctors named: each row and
column carries its plane and its lattice index, and the matrix holds the
files' precision.

Everything is read from the render the server runs against (the directory of
the rendered ``config.yml``): the simulator view's ``decks/<model>.json``, the
model's wiring and settings and its served status, and the step from the
model's measurement file in ``facility.json``. The step is the measurement
file's ``corrector_delta``, the kick-angle change ``pyaml_measure`` ``orm``
applies, converted per corrector to the hardware width the plug-in takes
through that corrector's wiring calibration, so the physics span of every
column is exactly ``corrector_delta``.

The engine solves in a child process launched by the executor's sandbox runner,
so a solve holding the interpreter neither blocks the server nor outlives the
executor's timeout: the child is killed at the timeout and the call is refused
naming the model.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from osprey.mcp_server.errors import make_error
from osprey.simulation.response import DEFAULT_KIND, INDEX_KEYS, RESULT_TAG

logger = logging.getLogger("osprey.mcp_server.tools.model_response_matrix")

__all__ = [
    "KINDS",
    "RESULT_TAG",
    "STEP_KEY",
    "TOOL_NAME",
    "ModelRequest",
    "model_response_matrix",
    "register",
    "resolve_model_request",
]

#: The tool's name as the agent knows it (``mcp__python__model_response_matrix``).
TOOL_NAME = "model_response_matrix"

#: The measurement-file key holding the kick-angle change, in radians.
STEP_KEY = "corrector_delta"

#: The responses the tool solves; the first is the one solved when none is named.
KINDS = (DEFAULT_KIND, "trajectory")

#: The solve reads the machine's model only.
_EXECUTION_MODE = "readonly"

#: The planes a corrector setpoint steers, in column order when none are named.
_PLANES = ("x", "y")

#: The unit statement every result carries.
_UNIT = "each entry is a readback change in its hardware unit per setpoint change in its own"


@dataclass(frozen=True)
class ModelRequest:
    """A resolved ``model_response_matrix`` call.

    Attributes:
        render_root: The directory holding the rendered ``config.yml``.
        model: The served model solved.
        engine_module: The module of the model's engine plug-in.
        correctors: The corrector setpoint addresses, one column each, in order.
        corrector_delta: The kick-angle change of every column, in radians.
        steps: Each corrector's full step width in its hardware unit.
        devices: Each corrector's device id, ``None`` when it is on none.
        view: The render's simulator view the request was resolved against.
        kind: The response solved, one of :data:`KINDS`.
    """

    render_root: Path
    model: str
    engine_module: str
    correctors: tuple[str, ...]
    corrector_delta: float
    steps: Mapping[str, float]
    devices: Mapping[str, str | None]
    view: Any = field(default=None, repr=False, compare=False)
    kind: str = DEFAULT_KIND


def _refuse(message: str, suggestion: str) -> NoReturn:
    make_error("validation_error", message, [suggestion])


def _engine_module(engine: str) -> str:
    """The module the engine's plug-in entry point names, without importing it."""
    from importlib import metadata

    from osprey.simulation.engines import ENTRY_POINT_GROUP

    found = metadata.entry_points(group=ENTRY_POINT_GROUP)
    if engine not in found.names:
        _refuse(
            f"no simulation engine plug-in is installed for engine {engine!r}",
            "Install the engine's plug-in and rebuild.",
        )
    return str(found[engine].module)


def _device(view: Any, address: str) -> str | None:
    try:
        on = view.channel(address).on
    except KeyError:
        return None
    if on is None:
        return None
    device = on.get("device")
    return str(device) if device is not None else None


def _model(view: Any, model: object) -> Any:
    candidates = [entry for entry in view.physics_models() if entry.deck is not None]
    listed = ", ".join(entry.name for entry in candidates)
    if not candidates:
        _refuse("this render serves no model with a deck", "Serve a deck model and rebuild.")
    if model is None:
        if len(candidates) > 1:
            _refuse(
                f"model is omitted and this render serves more than one deck model: {listed}",
                f"Pass model as one of: {listed}.",
            )
        return candidates[0]
    for entry in candidates:
        if entry.name == model:
            return entry
    _refuse(
        f"model {model!r} is not a served deck model of this render; the served ones: {listed}",
        f"Pass model as one of: {listed}.",
    )


def _corrector_delta(render_root: Path, model: str) -> float:
    from osprey.facility import FACILITY_FILE

    document = json.loads((render_root / FACILITY_FILE).read_text(encoding="utf-8"))
    record: dict[str, Any] = next(
        (entry for entry in document.get("models") or [] if entry.get("name") == model), {}
    )
    measurement = record.get("measurement") or {}
    value = measurement.get(STEP_KEY)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value == 0
    ):
        _refuse(
            f"measurement/{model}.yaml states no {STEP_KEY}, the kick-angle step the response "
            "is solved at",
            f"Add {STEP_KEY} (radians) to measurement/{model}.yaml and rebuild.",
        )
    return float(value)


def _correctors(model: Any, correctors: object) -> dict[str, Any]:
    """The named correctors' wiring records, in column order."""
    wired = {
        binding.address: binding
        for binding in model.bindings
        if binding.role == "setpoint" and binding.plane in _PLANES
    }
    if correctors is None:
        chosen = [
            binding.address
            for plane in _PLANES
            for binding in model.bindings
            if binding.address in wired and binding.plane == plane
        ]
        if not chosen:
            _refuse(
                f"{model.name} wires no corrector setpoint",
                "Wire a corrector setpoint in the model and rebuild.",
            )
        return {address: wired[address].record for address in chosen}
    if isinstance(correctors, str | bytes) or not isinstance(correctors, Sequence):
        _refuse(
            f"correctors must be a list of setpoint addresses, got {correctors!r}",
            "Pass correctors as a list of corrector setpoint addresses.",
        )
    if not correctors:
        _refuse(
            "correctors is empty",
            "Name the corrector setpoints to step, or omit correctors to step every one.",
        )
    chosen_records: dict[str, Any] = {}
    for address in correctors:
        if not isinstance(address, str) or address not in wired:
            _refuse(
                f"{address} is no corrector setpoint {model.name} wires",
                f"Use the corrector setpoint addresses {model.name} wires.",
            )
        if address in chosen_records:
            _refuse(
                f"corrector setpoint {address} appears more than once in correctors",
                "Name each corrector once.",
            )
        chosen_records[address] = wired[address].record
    return chosen_records


def _hardware_width(address: str, record: Mapping[str, Any], delta: float) -> float:
    """The hardware step whose physics span about the held value is ``delta``."""
    from osprey.facility.views.pyaml import hardware_width

    try:
        return hardware_width(record, delta)
    except ValueError as exc:
        _refuse(
            f"corrector {address}: its calibration gives no hardware step for "
            f"{STEP_KEY} {delta}: {exc}",
            f"State an inverse for the corrector's calibration, check it and {STEP_KEY}, "
            "and rebuild.",
        )


def _kind(kind: object) -> str:
    listed = ", ".join(KINDS)
    if not isinstance(kind, str) or kind not in KINDS:
        _refuse(f"kind must be one of {listed}, got {kind!r}", f"Pass kind as one of: {listed}.")
    return kind


def resolve_model_request(
    model: object, correctors: object, *, render_root: Path, kind: object = DEFAULT_KIND
) -> ModelRequest:
    """Resolve a ``model_response_matrix`` call against the render it runs in.

    Args:
        model: A served deck model, or ``None`` when the render serves exactly one.
        correctors: Corrector setpoint addresses the model wires, or ``None`` for
            every one, horizontal then vertical, in wiring order.
        render_root: The directory holding the rendered ``config.yml``.
        kind: The response to solve, one of :data:`KINDS`.

    Returns:
        The resolved request.

    Raises:
        fastmcp.exceptions.ToolError: A ``validation_error`` envelope naming
            the argument, the model or the file at fault.
    """
    from osprey_connectors.simulation.view import NoSimulatorView, SimulatorView, ViewSchemaError

    kind = _kind(kind)
    try:
        view = SimulatorView.of_render(render_root)
    except (NoSimulatorView, ViewSchemaError) as exc:
        _refuse(str(exc), "Run `osprey build`; the simulator view is written by the build.")
    chosen = _model(view, model)
    delta = _corrector_delta(render_root, chosen.name)
    records = _correctors(chosen, correctors)
    steps = {
        address: _hardware_width(address, record, delta) for address, record in records.items()
    }
    return ModelRequest(
        render_root=Path(render_root),
        model=chosen.name,
        engine_module=_engine_module(str(chosen.engine)),
        correctors=tuple(records),
        corrector_delta=delta,
        steps=steps,
        devices={address: _device(view, address) for address in records},
        view=view,
        kind=kind,
    )


def _child_payload(request: ModelRequest) -> dict[str, Any]:
    """What the child solves, as ``osprey.simulation.response.solve_and_print`` reads it."""
    return {
        "render_root": str(request.render_root),
        "model": request.model,
        "engine_module": request.engine_module,
        "correctors": list(request.correctors),
        "steps": dict(request.steps),
        "kind": request.kind,
    }


def _child_script(request: ModelRequest) -> str:
    """The script the sandbox runs: one call to the child's solve."""
    payload = _child_payload(request)
    return (
        "from osprey.simulation.response import solve_and_print\n"
        f"solve_and_print({json.dumps(payload, ensure_ascii=True)})\n"
    )


def _result_line(stdout: str, kind: str = DEFAULT_KIND) -> dict[str, Any] | None:
    """The child's last result line, or ``None`` when it printed none of ``kind``'s shape."""
    prefix = RESULT_TAG + " "
    lines = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]
    if not lines:
        return None
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        return None
    if "error" in value:
        return value
    if "rows" not in value or "matrix" not in value:
        return None
    if kind == DEFAULT_KIND:
        return value
    parts: list[list[Any]] = []
    for key in ("rows", "columns", "matrix", *INDEX_KEYS):
        part = value.get(key)
        if not isinstance(part, list):
            return None
        parts.append(part)
    rows, columns, matrix, row_index, *column_index = parts
    if len(row_index) != len(rows) or sum(map(len, column_index)) != len(columns):
        return None
    if len(matrix) != len(rows):
        return None
    return value


def _label(request: ModelRequest, line: Mapping[str, Any]) -> dict[str, Any]:
    """The result the agent reads: a line's rows and columns named by address and device.

    A line of a kind other than the default names its own columns, and each
    of its rows and columns also gets its plane word and its lattice index.

    Raises:
        ValueError: The line's columns are not the correctors the request stepped.
    """
    view = request.view
    if request.kind != DEFAULT_KIND and sorted(line["columns"]) != sorted(request.correctors):
        raise ValueError("the columns solved are not the correctors asked for")
    labelled: dict[str, Any] = {
        "model": request.model,
        "corrector_delta": request.corrector_delta,
        "unit": _UNIT,
        "rows": [{"address": row, "device": _device(view, row)} for row in line["rows"]],
        "columns": [
            {
                "address": address,
                "device": request.devices.get(address),
                "step": request.steps[address],
            }
            for address in (request.correctors if request.kind == DEFAULT_KIND else line["columns"])
        ],
        "matrix": line["matrix"],
    }
    if request.kind == DEFAULT_KIND:
        return labelled
    from osprey.facility.views.pyaml import PLANE_WORDS

    row_index, horizontal, vertical = (line[key] for key in INDEX_KEYS)
    planes = {binding.address: str(binding.plane) for binding in view.model(request.model).bindings}
    for row, index in zip(labelled["rows"], row_index, strict=True):
        row["plane"] = PLANE_WORDS.get(planes.get(row["address"], ""))
        row["index"] = index
    steered = [*(PLANE_WORDS["x"] for _ in horizontal), *(PLANE_WORDS["y"] for _ in vertical)]
    for column, plane, index in zip(
        labelled["columns"], steered, [*horizontal, *vertical], strict=True
    ):
        column["plane"] = plane
        column["index"] = index
    return {"kind": request.kind, **labelled}


async def model_response_matrix(
    model: str | None = None, correctors: list[str] | None = None, kind: str = DEFAULT_KIND
) -> str:
    """Solve a served model's response to its correctors, on the model alone.

    Read-only: nothing is written to the machine, to the deck or to any file.
    Each corrector is stepped about the value the deck holds by the model's
    measurement-file ``corrector_delta`` (radians), converted to the
    corrector's hardware unit, and every reading of the kind is solved on both
    sides.

    Args:
        model: The served model; omit when the render serves exactly one deck
            model.
        correctors: Corrector setpoint addresses; omit for every corrector the
            model wires, horizontal then vertical.
        kind: ``orbit`` (the default) for the orbit response of the monitor
            readbacks; ``trajectory`` for the first-turn response of the
            turn-by-turn readings, the response a first-turn steering
            correction is computed from.

    Returns:
        JSON: ``model``, ``corrector_delta``, ``unit``, ``rows`` (``{address,
        device}`` per reading), ``columns`` (``{address, device, step}`` per
        corrector, ``step`` in its hardware unit) and ``matrix`` (one list per
        row). For ``trajectory`` the result also names its ``kind``, the
        columns are the horizontal correctors then the vertical ones, and each
        row and column also carries ``plane`` (``H`` or ``V``) and ``index``,
        its element's position along the lattice from where the beam starts.
    """
    from osprey.mcp_server.python_executor import executor
    from osprey.mcp_server.python_executor.tools._pyaml_measure_args import TIMEOUT_KEY
    from osprey.utils.config_paths import resolve_config_dir

    config_dir = resolve_config_dir()
    if config_dir is None:
        _refuse("no rendered config.yml to read the model from", "Run `osprey build`.")
    request = resolve_model_request(model, correctors, render_root=Path(config_dir), kind=kind)

    result = await executor.execute_code(
        code=_child_script(request),
        execution_mode=_EXECUTION_MODE,
        description=f"{TOOL_NAME} {request.model}",
        tool=TOOL_NAME,
    )
    if result.failure_kind == executor.FAILURE_KIND_TIMEOUT:
        make_error(
            "execution_error",
            f"the response of {request.model} was not solved within the executor's timeout; "
            "the solve was stopped",
            [f"Name fewer correctors, or raise {TIMEOUT_KEY}."],
            details={"model": request.model, "kind": executor.FAILURE_KIND_TIMEOUT},
        )
    labelled: dict[str, Any] | None = None
    try:
        line = _result_line(result.stdout or "", request.kind)
        if line is not None and "error" not in line:
            labelled = _label(request, line)
    except ValueError:
        line = None
    if line is None:
        make_error(
            "execution_error",
            f"the solve of {request.model} ended without a result",
            ["Read stderr for the reason."],
            details={"model": request.model, "stderr": (result.stderr or "")[-2000:]},
        )
    if "error" in line:
        make_error(
            "execution_error",
            f"the response of {request.model} has no solve: {line['error']}",
            [f"Name other correctors, or check {STEP_KEY} in measurement/{request.model}.yaml."],
            details={"model": request.model},
        )
    return json.dumps(labelled)


def register(server: Any) -> None:
    """Add ``model_response_matrix`` to ``server``.

    Args:
        server: The FastMCP server to register on; nothing else is touched.
    """
    server.tool(model_response_matrix, name="model_response_matrix")
