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

__all__ = ["DEFAULT_KIND", "RESULT_TAG", "solve_and_print"]

#: The tag of the child's one result line.
RESULT_TAG = "OSPREY_MODEL_RESPONSE_MATRIX"

#: The response solved when a payload names no ``kind``: the engine is called
#: without the word.
DEFAULT_KIND = "orbit"

#: The significant digits a matrix entry of a line carrying an ``index`` is
#: printed with, the precision the built response files hold.
_DIGITS = 9


def _thaw(value: Any) -> Any:
    """A plain JSON copy of a value the view froze."""
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw(item) for item in value]
    return value


def _entry(value: Any, rounded: bool) -> float:
    """A matrix entry as printed: at :data:`_DIGITS` significant digits when ``rounded``."""
    return float(f"{float(value):.{_DIGITS}g}") if rounded else float(value)


def _refusal(exc: Exception, engine: Any) -> bool:
    """Whether *exc* is *engine* saying the request has no solve, or a request the engine refuses."""
    from osprey.facility.errors import FacilityBuildError

    return isinstance(exc, (*getattr(engine, "SOLVE_ERRORS", ()), FacilityBuildError, ValueError))


def solve_and_print(payload: Mapping[str, Any]) -> None:
    """Solve one request and print its :data:`RESULT_TAG` line.

    Prints ``{"refused"}`` with the engine's own text when the engine refuses
    the request. Otherwise prints ``{"rows", "columns", "matrix",
    "unsolved"}``: ``columns`` the solved correctors, possibly none,
    ``matrix`` one column each, ``unsolved`` each corrector whose step has no
    solve mapped to the engine's text; the tool decides that a line with no
    column is a refusal. A response that carries an ``index`` adds it, each
    row and corrector address mapped to its lattice index, and its entries
    hold the built files' precision. Any other failure raises, and no result
    line is printed.

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
        result = engine.response_matrix(
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
        print(RESULT_TAG, json.dumps({"refused": str(exc)}), flush=True)
        return
    index = getattr(result, "index", None)
    line: dict[str, Any] = {
        "rows": [str(row) for row in result.rows],
        "columns": [str(column) for column in result.columns],
        "matrix": [[_entry(value, index is not None) for value in row] for row in result.matrix],
        "unsolved": {str(address): str(text) for address, text in result.unsolved.items()},
    }
    if index is not None:
        line["index"] = {str(address): int(place) for address, place in index.items()}
    print(RESULT_TAG, json.dumps(line), flush=True)
