"""A served model's response to its correctors, solved on the model alone.

Every view, engine and lattice import happens inside a function body, so
importing the SDK loads none of them.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from osprey.apps.sdk._facts import _render_root

__all__ = ["ResponseMatrix", "model_response_matrix"]

#: How much of a failed child's stderr an error carries.
_STDERR_TAIL = 2000


@dataclass(frozen=True)
class ResponseMatrix:
    """A model's solved response, of the kind asked for.

    Stable: part of the application SDK's public surface.

    A position is a lattice element index. For the ``orbit`` kind it is
    counted from the deck's first element. For
    the ``trajectory`` kind it is counted from the injection element, in the
    order the bunch meets the elements on its first turn: of two positions the
    lower one is upstream.

    Attributes:
        model: The model solved.
        rows: One address per row, ordered by (position, ``x`` before ``y``).
            For the ``orbit`` kind the monitor readbacks. For the
            ``trajectory`` kind the monitors' turn-buffer records (the wiring
            signals ``turn_x`` and ``turn_y``), the arrays a shot is read
            from; a monitor wiring neither has no row.
        columns: The solved corrector setpoint addresses, one column each, in
            the order asked.
        matrix: One list per row: each entry the readback change in its
            hardware unit per setpoint change in its own. A turn-buffer row
            is in metres, as the arrays are.
        unsolved: Each corrector asked whose step has no solve, mapped to the
            engine's text.
        row_positions: Each row's position, aligned with ``rows``; ``None``
            when the model's engine states no positions.
        column_positions: Each column's position, aligned with ``columns``;
            ``None`` when the model's engine states no positions.
    """

    model: str
    rows: list[str]
    columns: list[str]
    matrix: list[list[float]]
    unsolved: dict[str, str]
    row_positions: list[int] | None = None
    column_positions: list[int] | None = None


def model_response_matrix(
    model: str,
    correctors: list[str] | None = None,
    render_root: Path | None = None,
    timeout_s: float = 120,
    *,
    kind: str = "orbit",
) -> ResponseMatrix:
    """Solve a served model's response to its correctors, on the model alone.

    Stable: part of the application SDK's public surface.

    Read-only: nothing is written to the machine or to the deck. The request
    is resolved as the ``model_response_matrix`` tool resolves it (each
    corrector stepped by the model's measurement-file ``corrector_delta``,
    converted to its hardware unit) and solved by the same child script, run
    in a child interpreter that is killed at ``timeout_s``.

    Args:
        model: A served deck model.
        correctors: Corrector setpoint addresses the model wires; every one,
            horizontal then vertical, when omitted.
        render_root: The directory holding the rendered ``config.yml``; the
            render this process runs from when omitted.
        timeout_s: Seconds the solve may take.
        kind: ``"orbit"``, the orbit readbacks' response, or ``"trajectory"``,
            the first-turn response of the monitors' turn-buffer records. Any
            other value is refused before a child starts; an engine that
            answers only the orbit response refuses ``"trajectory"``, and so
            does a ring wiring no turn-buffer record.

    Returns:
        The response, every row and column with its position along the ring.
        ``columns`` may be empty when no corrector solves; ``unsolved`` then
        names every one.

    Raises:
        ValueError: The request is refused: by the resolver (the message
            names the kind, the argument, the model or the file), or by the
            engine.
        TimeoutError: The solve did not finish within ``timeout_s``.
        RuntimeError: The request has no solve, or the child ended without a
            result.
    """
    from osprey.simulation.model_request import (
        UNSOLVED_KEY,
        child_script,
        line_positions,
        resolve_model_request,
        result_line,
    )

    root = _render_root() if render_root is None else Path(render_root)
    request = resolve_model_request(model, correctors, render_root=root, kind=kind)
    try:
        done = subprocess.run(
            [sys.executable, "-c", child_script(request)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"the response of {request.model} was not solved within {timeout_s} s; "
            "the solve was stopped"
        ) from None
    try:
        line = result_line(done.stdout or "")
    except ValueError:
        line = None
    if line is None:
        raise RuntimeError(
            f"the solve of {request.model} ended without a result: "
            f"{(done.stderr or '')[-_STDERR_TAIL:]}"
        )
    if line.get("cause") == "solve":
        raise RuntimeError(f"the response of {request.model} has no solve: {line['refused']}")
    if "refused" in line:
        raise ValueError(f"the response of {request.model} was refused: {line['refused']}")
    row_positions, column_positions = line_positions(line)
    return ResponseMatrix(
        model=request.model,
        rows=[str(row) for row in line["rows"]],
        columns=[str(column) for column in line["columns"]],
        matrix=[[float(value) for value in row] for row in line["matrix"]],
        unsolved={str(key): str(text) for key, text in dict(line[UNSOLVED_KEY]).items()},
        row_positions=row_positions,
        column_positions=column_positions,
    )
