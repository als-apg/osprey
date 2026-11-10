"""MCP tool: model_response_matrix — a served model's orbit response, read-only.

``model_response_matrix(model=None, correctors=None)`` solves the orbit
response of a served model's wired monitors to its corrector setpoints on the
model's own deck, through its engine plug-in's ``response_matrix()``. Nothing
is written to the machine or to the deck.

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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

from osprey.mcp_server.errors import MISSING_KEY, make_error, refuse_nothing_answered
from osprey.simulation import model_request as _request
from osprey.simulation.model_request import (
    RESULT_TAG,
    STEP_KEY,
    ModelRequest,
    ModelRequestRefused,
    child_payload,
    child_script,
    result_line,
)

logger = logging.getLogger("osprey.mcp_server.tools.model_response_matrix")

__all__ = [
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

#: The solve reads the machine's model only.
_EXECUTION_MODE = "readonly"

#: The unit statement every result carries.
_UNIT = "each entry is a readback change in its hardware unit per setpoint change in its own"

#: The child's payload and script, shared with the application SDK.
_child_payload = child_payload
_child_script = child_script
_result_line = result_line
_device = _request.device_of


def _refuse(message: str, suggestion: str) -> NoReturn:
    make_error("validation_error", message, [suggestion])


def resolve_model_request(model: object, correctors: object, *, render_root: Path) -> ModelRequest:
    """Resolve a ``model_response_matrix`` call against the render it runs in.

    Args:
        model: A served deck model, or ``None`` when the render serves exactly one.
        correctors: Corrector setpoint addresses the model wires, or ``None`` for
            every one, horizontal then vertical, in wiring order.
        render_root: The directory holding the rendered ``config.yml``.

    Returns:
        The resolved request.

    Raises:
        fastmcp.exceptions.ToolError: A ``validation_error`` envelope naming
            the argument, the model or the file at fault.
    """
    try:
        return _request.resolve_model_request(model, correctors, render_root=render_root)
    except ModelRequestRefused as refused:
        _refuse(refused.message, refused.suggestion)


def _label(request: ModelRequest, line: Mapping[str, Any]) -> dict[str, Any]:
    view = request.view
    return {
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
            for address in line["columns"]
        ],
        "matrix": line["matrix"],
        MISSING_KEY: line[MISSING_KEY],
    }


async def model_response_matrix(
    model: str | None = None, correctors: list[str] | None = None
) -> str:
    """Solve a served model's orbit response to its correctors, on the model alone.

    Read-only: nothing is written to the machine or to the deck. Each corrector
    is stepped about the value the deck holds by the model's measurement-file
    ``corrector_delta`` (radians), converted to the corrector's hardware unit,
    and every wired monitor readback is solved on both sides.

    Args:
        model: The served model; omit when the render serves exactly one deck
            model.
        correctors: Corrector setpoint addresses; omit for every corrector the
            model wires, horizontal then vertical.

    Returns:
        JSON: ``model``, ``corrector_delta``, ``unit``, ``rows`` (``{address,
        device}`` per monitor readback), ``columns`` (``{address, device,
        step}`` per solved corrector, ``step`` in its hardware unit),
        ``matrix`` (one list per row) and ``missing`` (each corrector whose
        step has no solve, mapped to the engine's text). When no corrector
        solves, the call is refused as ``nothing_answered`` with
        ``details.missing`` holding the same map (docs: Reference → Contracts →
        Partial Answers).
        When the engine refuses the request, the call is refused with the
        engine's text; when the facility file is defective, it is refused
        with the rebuild remedy; when the whole request has no solve, it
        fails with the engine's text. Any other engine error is a failure
        carrying its stderr.
    """
    from osprey.mcp_server.python_executor import executor
    from osprey.mcp_server.python_executor.tools._pyaml_measure_args import TIMEOUT_KEY
    from osprey.utils.config_paths import resolve_config_dir

    config_dir = resolve_config_dir()
    if config_dir is None:
        _refuse("no rendered config.yml to read the model from", "Run `osprey build`.")
    request = resolve_model_request(model, correctors, render_root=Path(config_dir))

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
    try:
        line = _result_line(result.stdout or "")
    except ValueError:
        line = None
    if line is None:
        make_error(
            "execution_error",
            f"the solve of {request.model} ended without a result",
            ["Read stderr for the reason."],
            details={"model": request.model, "stderr": (result.stderr or "")[-2000:]},
        )
    if line.get("cause") == "solve":
        make_error(
            "execution_error",
            f"the response of {request.model} has no solve: {line['refused']}",
            [],
            details={"model": request.model},
        )
    if "refused" in line:
        remedy = ["Fix the facility file and run `osprey build`."]
        make_error(
            "validation_error",
            f"the response of {request.model} was refused: {line['refused']}",
            remedy if line["cause"] == "facility" else [],
            details={"model": request.model},
        )
    if not line["columns"]:
        refuse_nothing_answered(
            TOOL_NAME, "correctors", line[MISSING_KEY], details={"model": request.model}
        )
    return json.dumps(_label(request, line))


def register(server: Any) -> None:
    """Add ``model_response_matrix`` to ``server``.

    Args:
        server: The FastMCP server to register on; nothing else is touched.
    """
    server.tool(model_response_matrix, name="model_response_matrix")
