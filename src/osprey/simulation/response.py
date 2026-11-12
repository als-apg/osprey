"""A served model's response, solved in the process that imports this module.

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

__all__ = [
    "CAUSES",
    "COLUMN_POSITIONS_KEY",
    "DEFAULT_KIND",
    "RESPONSE_KINDS",
    "RESULT_TAG",
    "ROW_POSITIONS_KEY",
    "solve_and_print",
]

#: The tag of the child's one result line.
RESULT_TAG = "OSPREY_MODEL_RESPONSE_MATRIX"

#: Why the engine answered no matrix: the request is one it does not answer,
#: the facility file is defective, or the whole request has no solve.
CAUSES: tuple[str, ...] = ("request", "facility", "solve")

#: The responses a request may ask for: the orbit readbacks' response, or the
#: first-turn response of the monitors' turn-buffer records.
RESPONSE_KINDS: tuple[str, ...] = ("orbit", "trajectory")

#: The result line's keys, and the engine answer's attributes, holding each
#: row's and each column's position along the lattice.
ROW_POSITIONS_KEY = "row_positions"
COLUMN_POSITIONS_KEY = "column_positions"

#: The kind asked for when none is named; an engine whose ``response_matrix``
#: takes no ``kind`` answers this one only.
DEFAULT_KIND = RESPONSE_KINDS[0]


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


def _takes_kind(function: Any) -> bool:
    """Whether *function* accepts a ``kind`` keyword argument."""
    import inspect

    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "kind"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        for parameter in parameters
    )


def _positions(result: Any, name: str, count: int, engine_module: str) -> list[int] | None:
    """The engine's *name* positions as whole numbers, ``None`` when it states none.

    Raises:
        ValueError: The engine states positions that do not line up with its
            rows or columns.
    """
    stated = getattr(result, name, None)
    if stated is None:
        return None
    positions = [int(position) for position in stated]
    if len(positions) != count:
        raise ValueError(
            f"the engine plug-in {engine_module} states {len(positions)} {name} for {count} of them"
        )
    return positions


def solve_and_print(payload: Mapping[str, Any]) -> None:
    """Solve one request and print its :data:`RESULT_TAG` line.

    Prints ``{"refused", "cause"}`` with the engine's own text when the
    engine answers no matrix: ``cause`` is ``request`` for one of the
    engine's ``REFUSAL_ERRORS``, ``facility`` for a ``FacilityBuildError``,
    ``solve`` for one of its ``SOLVE_ERRORS``. Otherwise prints ``{"rows",
    "columns", "matrix", "missing", "row_positions", "column_positions"}``:
    ``columns`` the solved correctors, possibly none, ``matrix`` one column
    each, ``missing`` each corrector whose step has no solve mapped to the
    engine's text; a line with no column carries the map of every corrector
    asked, and the tool answers it as ``nothing_answered``. The two position
    lists are the engine's, one whole number per row and per column, each
    ``null`` when the engine's answer states none. Any other exception
    raises, and no result line is printed; an engine whose positions do not
    line up with its rows or columns raises so.

    The engine's ``response_matrix`` is called with ``kind`` when it takes
    one. One that takes none is called as it always was for the
    :data:`DEFAULT_KIND`, and any other kind is refused as a ``request``
    naming the engine.

    Args:
        payload: ``render_root``, ``model``, ``engine_module``, ``correctors``,
            ``steps`` (each corrector's full step width in its hardware
            unit) and ``kind``, one of :data:`RESPONSE_KINDS`;
            :data:`DEFAULT_KIND` when absent.
    """
    import importlib

    from osprey_connectors.simulation.view import SimulatorView

    view = SimulatorView.of_render(payload["render_root"])
    model = view.model(payload["model"])
    engine = importlib.import_module(payload["engine_module"])
    kind = str(payload.get("kind", DEFAULT_KIND))
    takes_kind = _takes_kind(engine.response_matrix)
    if not takes_kind and kind != DEFAULT_KIND:
        text = (
            f"the engine plug-in {payload['engine_module']} answers only the {DEFAULT_KIND} "
            f"response: its response_matrix takes no kind, so the {kind} response cannot "
            "be asked of it"
        )
        print(RESULT_TAG, json.dumps({"refused": text, "cause": "request"}), flush=True)  # noqa: T201 -- stdout carries the result line the parent reads
        return
    arguments = (
        model.deck,
        [_thaw(binding.record) for binding in model.bindings],
        _thaw(model.settings),
        list(payload["correctors"]),
        dict(payload["steps"]),
    )
    try:
        if takes_kind:
            result = engine.response_matrix(*arguments, kind=kind)
        else:
            result = engine.response_matrix(*arguments)
    except Exception as exc:
        cause = _cause(exc, engine)
        if cause is None:
            raise
        print(RESULT_TAG, json.dumps({"refused": str(exc), "cause": cause}), flush=True)  # noqa: T201 -- stdout carries the result line the parent reads
        return
    rows = [str(row) for row in result.rows]
    columns = [str(column) for column in result.columns]
    module = str(payload["engine_module"])
    line = {
        "rows": rows,
        "columns": columns,
        "matrix": [[float(value) for value in row] for row in result.matrix],
        "missing": {str(address): str(text) for address, text in result.unsolved.items()},
        ROW_POSITIONS_KEY: _positions(result, ROW_POSITIONS_KEY, len(rows), module),
        COLUMN_POSITIONS_KEY: _positions(result, COLUMN_POSITIONS_KEY, len(columns), module),
    }
    print(RESULT_TAG, json.dumps(line), flush=True)  # noqa: T201 -- stdout carries the result line the parent reads
