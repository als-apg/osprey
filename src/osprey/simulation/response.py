"""A served model's orbit response, solved in the process that imports this module.

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

__all__ = ["CAUSES", "RESULT_TAG", "solve_and_print"]

#: The tag of the child's one result line.
RESULT_TAG = "OSPREY_MODEL_RESPONSE_MATRIX"

#: Why the engine answered no matrix: the request is one it does not answer,
#: the facility file is defective, or the whole request has no solve.
CAUSES: tuple[str, ...] = ("request", "facility", "solve")


def _thaw(value: Any) -> Any:
    """A plain JSON copy of a value the view froze."""
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw(item) for item in value]
    return value


def _cause(exc: Exception, engine: Any) -> str | None:
    """The :data:`CAUSES` entry *exc* states for *engine*, or ``None`` for a failure."""
    from osprey.facility.errors import FacilityBuildError

    if isinstance(exc, getattr(engine, "REFUSAL_ERRORS", ())):
        return "request"
    if isinstance(exc, FacilityBuildError):
        return "facility"
    if isinstance(exc, getattr(engine, "SOLVE_ERRORS", ())):
        return "solve"
    return None


def solve_and_print(payload: Mapping[str, Any]) -> None:
    """Solve one request and print its :data:`RESULT_TAG` line.

    Prints ``{"refused", "cause"}`` with the engine's own text when the
    engine answers no matrix: ``cause`` is ``request`` for one of the
    engine's ``REFUSAL_ERRORS``, ``facility`` for a ``FacilityBuildError``,
    ``solve`` for one of its ``SOLVE_ERRORS``. Otherwise prints ``{"rows",
    "columns", "matrix", "missing"}``: ``columns`` the solved correctors,
    possibly none, ``matrix`` one column each, ``missing`` each corrector
    whose step has no solve mapped to the engine's text; a line with no
    column carries the map of every corrector asked, and the tool answers it
    as ``nothing_answered``. Any other exception raises, and no result line
    is printed.

    Args:
        payload: ``render_root``, ``model``, ``engine_module``, ``correctors``
            and ``steps`` (each corrector's full step width in its hardware
            unit).
    """
    import importlib

    from osprey_connectors.simulation.view import SimulatorView

    view = SimulatorView.of_render(payload["render_root"])
    model = view.model(payload["model"])
    engine = importlib.import_module(payload["engine_module"])
    try:
        result = engine.response_matrix(
            model.deck,
            [_thaw(binding.record) for binding in model.bindings],
            _thaw(model.settings),
            list(payload["correctors"]),
            dict(payload["steps"]),
        )
    except Exception as exc:
        cause = _cause(exc, engine)
        if cause is None:
            raise
        print(RESULT_TAG, json.dumps({"refused": str(exc), "cause": cause}), flush=True)  # noqa: T201 -- stdout carries the result line the parent reads
        return
    line = {
        "rows": [str(row) for row in result.rows],
        "columns": [str(column) for column in result.columns],
        "matrix": [[float(value) for value in row] for row in result.matrix],
        "missing": {str(address): str(text) for address, text in result.unsolved.items()},
    }
    print(RESULT_TAG, json.dumps(line), flush=True)  # noqa: T201 -- stdout carries the result line the parent reads
