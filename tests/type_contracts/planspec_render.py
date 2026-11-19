# mypy: warn-unused-ignores
"""`PlanSpec.render` is typed against the plan's own params and a `RowWindow`.

``PlanSpec.render`` is ``Callable[[RowWindow, SchemaT], Figure] | None``, the same
``SchemaT`` as the plan's ``schema``. A render that takes ``dict[str, Any]`` instead
of its PARAMS, or the old row list instead of a window, is therefore a type error
while the plan is being written rather than a failure at the first poll tick of a
live run. Nothing at runtime observes the annotation, so this file proves it to the
type check instead, which runs over this directory because ``[tool.mypy] files``
names it:

- ``assert_type`` pins the well-typed spec to exactly ``PlanSpec[DemoParams]``: the
  contract is satisfiable, and ``SchemaT`` is inferred from the plan rather than
  widened to ``Any``.
- Each mis-typed render carries ``# type: ignore[arg-type]``, and that ignore is
  the assertion. The ``# mypy: warn-unused-ignores`` line at the top of this file
  is its own declaration that its ignores are assertions: it turns the flag on
  for this file whatever the repository's config says, so an ignore that no
  longer silences an error is itself an error. If the field stops refusing
  either render, or ``PlanSpec`` resolves to ``Any``, the type check fails on
  that line. Under ``Any`` the ``assert_type`` above passes too, so these ignores
  are the only guard against it.

A case's ignore never lists ``unused-ignore``: that code switches the case off
for good, because the ignore may then go unused without an error. If a mypy
upgrade reports the ``arg-type`` error on a different line, the ignore moves to
that line. `tests/scripts/test_mypy_gate.py` holds what the file relies on: the
directory is in ``[tool.mypy] files``, and every file here carries the inline
flag, which mypy applies over the config file's settings.

This is not a test module: its name does not match ``python_files``, so pytest
does not collect it when walking the directory. Named directly on the command
line it imports cleanly, runs nothing but the dataclass constructors, and holds
no tests. A new static contract, a property only the type checker can see,
joins this directory as a file of its own written the same way.
"""

from __future__ import annotations

from typing import Any, assert_type

from pydantic import BaseModel

from osprey.services.bluesky_bridge.figure import Figure, RowWindow
from osprey.services.bluesky_bridge.plan_types import PlanSpec


class DemoParams(BaseModel):
    amplitude: float = 1.0


def build_plan(_devices: dict[str, Any], _params: DemoParams) -> Any:
    return None


def render(_window: RowWindow, _params: DemoParams) -> Figure:
    return Figure(partial=False, source="live")


def render_by_dict(_window: RowWindow, _params: dict[str, Any]) -> Figure:
    """A render annotated against a plain dict instead of the plan's PARAMS."""
    return Figure(partial=False, source="live")


def render_by_rows(_rows: list[dict[str, Any]], _params: DemoParams) -> Figure:
    """A render still written against the row list `RowWindow` replaced."""
    return Figure(partial=False, source="live")


assert_type(
    PlanSpec(name="demo", plan=build_plan, schema=DemoParams, render=render),
    PlanSpec[DemoParams],
)

PlanSpec(
    name="by_dict",
    plan=build_plan,
    schema=DemoParams,
    render=render_by_dict,  # type: ignore[arg-type]
)
PlanSpec(
    name="by_rows",
    plan=build_plan,
    schema=DemoParams,
    render=render_by_rows,  # type: ignore[arg-type]
)
