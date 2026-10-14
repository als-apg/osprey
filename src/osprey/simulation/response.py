"""A served model's response to its correctors, solved in the process that imports this module.

The ``model_response_matrix`` tool runs :func:`solve_and_print` in a sandbox
child, so the engine's solve can be stopped at the executor's timeout without
taking the server with it. The module imports neither the MCP server nor its
dependencies: the child loads the simulator view, the model's engine plug-in
and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

__all__ = ["DEFAULT_KIND", "INDEX_KEYS", "RESULT_TAG", "solve_and_print"]

#: The tag of the child's one result line.
RESULT_TAG = "OSPREY_MODEL_RESPONSE_MATRIX"

#: The response solved when a payload names no ``kind``: the engine is called
#: without the word, and the line holds ``rows`` and ``matrix`` alone.
DEFAULT_KIND = "orbit"

#: The lattice-index lists a line of any other kind carries beside ``rows``,
#: ``columns`` and ``matrix``: one entry per row, then one per column, the
#: horizontal columns before the vertical ones.
INDEX_KEYS = ("bpm_index", "hcor_index", "vcor_index")

#: The significant digits a matrix entry of a line carrying indices is printed
#: with, the precision the built response files hold.
_DIGITS = 9


def _thaw(value: Any) -> Any:
    """A plain JSON copy of a value the view froze."""
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw(item) for item in value]
    return value


def _refusal(exc: Exception, engine: Any) -> bool:
    """Whether *exc* is *engine* saying the request has no solve, or a request the engine refuses."""
    from osprey.facility.errors import FacilityBuildError

    return isinstance(exc, (*getattr(engine, "SOLVE_ERRORS", ()), FacilityBuildError, ValueError))


def solve_and_print(payload: Mapping[str, Any]) -> None:
    """Solve one request and print its :data:`RESULT_TAG` line.

    Prints ``{"rows", "matrix"}`` for a solve of :data:`DEFAULT_KIND`; for any
    other kind ``{"rows", "columns", "matrix"}`` and the three
    :data:`INDEX_KEYS`, the columns as the engine ordered them. Prints
    ``{"error"}`` with the engine's own text when the engine refuses or a step
    has no solve. Any other failure raises, and no result line is printed.

    Args:
        payload: ``render_root``, ``model``, ``engine_module``, ``correctors``
            and ``steps`` (each corrector's full step width in its hardware
            unit), and optionally ``kind``, the word the engine's
            ``response_matrix()`` takes for the response.
    """
    import importlib

    from osprey_connectors.simulation.view import SimulatorView

    view = SimulatorView.of_render(payload["render_root"])
    model = view.model(payload["model"])
    engine = importlib.import_module(payload["engine_module"])
    kind = str(payload.get("kind") or DEFAULT_KIND)
    words = {} if kind == DEFAULT_KIND else {"kind": kind}
    try:
        answer = engine.response_matrix(
            model.deck,
            [_thaw(binding.record) for binding in model.bindings],
            _thaw(model.settings),
            list(payload["correctors"]),
            dict(payload["steps"]),
            **words,
        )
    except Exception as exc:
        if not _refusal(exc, engine):
            raise
        print(RESULT_TAG, json.dumps({"error": str(exc)}), flush=True)
        return
    line: dict[str, Any]
    if kind == DEFAULT_KIND:
        rows, matrix = answer
        line = {
            "rows": [str(row) for row in rows],
            "matrix": [[float(value) for value in row] for row in matrix],
        }
    else:
        rows, columns, matrix, *indices = answer
        line = {
            "rows": [str(row) for row in rows],
            "columns": [str(column) for column in columns],
            "matrix": [[float(f"{float(value):.{_DIGITS}g}") for value in row] for row in matrix],
        }
        for key, index in zip(INDEX_KEYS, indices, strict=True):
            line[key] = [int(entry) for entry in index]
    print(RESULT_TAG, json.dumps(line), flush=True)
