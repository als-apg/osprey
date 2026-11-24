"""The tool-error check every agentic run shares.

A run fails on any tool result marked as an error. The only way to excuse one
is a named predicate that reads a single trace and is pinned by its own unit
test, so the check cannot be widened by a pattern no test holds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def assert_no_tool_errors(result: Any, exempt: Callable[[Any], bool] | None = None) -> None:
    """Assert no tool trace of *result* has ``is_error`` set.

    ``exempt`` is a predicate over one trace; a trace it accepts is not counted.
    """
    errors = [
        t for t in result.tool_traces if t.is_error and not (exempt is not None and exempt(t))
    ]
    assert len(errors) == 0, f"{len(errors)} tool error(s): " + "; ".join(
        f"{t.name}: {t.result[:200] if t.result else 'N/A'}" for t in errors
    )
