"""Every shipped render, bound, states the unit the run declared for the columns it draws.

One relation, checked over every render the bridge ships (the default figure
and every built-in plan's ``render``), each routed through `bind_units` as the
figure route routes it: every axis names the columns the tables below say it
draws, and states the one unit those columns declare, none when any of them
declares none or two of them disagree. A render that types a unit by hand is
refused. Each render is drawn over three windows:

- distinct: every column declares its own unit, so a single-column axis states
  exactly that column's unit and an axis over several columns states none;
- shared: the columns of one role share one unit, so the multi-column axes
  state it too;
- none: the run declares no units, so no axis of any panel states one.

The per-render tables below say which columns each panel's axes draw. A
built-in plan without a table fails `test_every_shipped_plan_has_a_case`.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import pytest

pytest.importorskip("bluesky")

from osprey.services.bluesky_bridge import plans_core
from osprey.services.bluesky_bridge.figure import (
    REASON_NO_RENDER,
    Columns,
    Figure,
    HeatmapMark,
    Panel,
    RowWindow,
    bind_units,
    default_figure,
    rows_from_columnar,
)
from osprey.services.bluesky_bridge.plans_core import grid_scan, orm
from osprey.services.bluesky_bridge.plans_core import orbit_bump_sweep as bump
from tests.services.bluesky_bridge import test_bump_render, test_grid_render, test_orm_render


class Ratio(NamedTuple):
    """An axis whose unit is one role's unit per another's (a response slope)."""

    over: tuple[str, ...]
    per: tuple[str, ...]


Axis = tuple[str, ...] | Ratio


class Binding(NamedTuple):
    """The columns one panel's axes draw; an empty axis draws no column."""

    x: Axis = ()
    y: Axis = ()
    value: Axis = ()


class Case(NamedTuple):
    """One render over its rows: the columns by role, and each panel's binding.

    ``draw`` returns the render's own figure, unbound, and the window it drew.
    """

    roles: Mapping[str, Sequence[str]]
    draw: Callable[[Mapping[str, str]], tuple[Figure, RowWindow]]
    bindings: Mapping[str, Binding]


def _declared(units: Mapping[str, str], columns: Axis) -> str | None:
    """The unit an axis over *columns* states under *units*: the one they share."""
    if isinstance(columns, Ratio):
        over, per = _declared(units, columns.over), _declared(units, columns.per)
        return None if over is None or per is None else f"{over} / {per}"
    stated = {units.get(column) for column in columns}
    if len(stated) != 1:
        return None
    (unit,) = stated
    return unit


# =========================================================================
# The renders, each with the columns its panels draw
# =========================================================================


def _default_row_index() -> Case:
    columns = ["bpm1", "bpm2"]
    values = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def draw(units: Mapping[str, str]) -> tuple[Figure, RowWindow]:
        window = rows_from_columnar(columns, values, len(values), units=units)
        return default_figure(window, reason=REASON_NO_RENDER, partial=False, source="live"), window

    return Case(
        roles={"readable": columns},
        draw=draw,
        bindings={name: Binding(y=(name,)) for name in columns},
    )


def _default_movable_x() -> Case:
    columns = ["motor1", "det1", "det2"]
    values = [[0.0, 1.0, 4.0], [1.0, 2.0, 5.0], [2.0, 3.0, 6.0]]

    def draw(units: Mapping[str, str]) -> tuple[Figure, RowWindow]:
        window = rows_from_columnar(columns, values, len(values), units=units)
        figure = default_figure(
            window,
            reason=REASON_NO_RENDER,
            partial=False,
            source="live",
            movable=["motor1"],
            readable=["det1", "det2"],
        )
        return figure, window

    return Case(
        roles={"movable": ["motor1"], "readable": ["det1", "det2"]},
        draw=draw,
        bindings={name: Binding(x=("motor1",), y=(name,)) for name in ("det1", "det2")},
    )


def _complete_window(rows: list[dict[str, Any]], units: Mapping[str, str]) -> RowWindow:
    return RowWindow(
        rows=rows,
        columns=sorted({name for row in rows for name in row}),
        rows_complete=True,
        total_seen=len(rows),
        units=units,
    )


def _drawing(
    render: Callable[[RowWindow, Any], Figure], rows: list[dict[str, Any]], params: Any
) -> Callable[[Mapping[str, str]], tuple[Figure, RowWindow]]:
    def draw(units: Mapping[str, str]) -> tuple[Figure, RowWindow]:
        window = _complete_window(rows, units)
        return render(window, params), window

    return draw


def _grid(axes: list[grid_scan.GridAxis]) -> Case:
    readbacks = ["det1", "det2"]
    params = grid_scan.PARAMS(readbacks=readbacks, axes=axes)
    rows = test_grid_render._rows(axes, {name: name for name in readbacks})
    setpoints = [axis.setpoint for axis in axes]
    outer, inner = setpoints[0], setpoints[-1]

    if len(axes) == 1:
        bindings = {f"{name} vs {inner}": Binding(x=(inner,), y=(name,)) for name in readbacks}
    elif len(axes) == 2:
        bindings = {
            f"{name} over the {outer} × {inner} grid": Binding(
                x=(inner,), y=(outer,), value=(name,)
            )
            for name in readbacks
        }
    else:
        bindings = {f"{name} vs {inner}": Binding(x=(inner,), y=(name,)) for name in readbacks}

    return Case(
        roles={**{name: [name] for name in setpoints}, "readable": readbacks},
        draw=_drawing(grid_scan.render, rows, params),
        bindings=bindings,
    )


def _orbit_bump_sweep() -> Case:
    params = test_bump_render._params(monitors=["mon1", "mon2"])
    rows = test_bump_render._rows(params)
    bpms = tuple(test_bump_render._bpm_names(params))
    constrained = tuple(
        [target.readback for target in params.targets] + list(params.closure_readbacks)
    )
    return Case(
        roles={
            "bpm": bpms,
            "corrector": list(params.correctors),
            "monitor": list(params.monitors),
        },
        draw=_drawing(bump.render, rows, params),
        bindings={
            "Orbit shift across BPMs": Binding(y=bpms),
            "Band residual at constrained BPMs": Binding(y=constrained),
            "Corrector offsets": Binding(y=tuple(params.correctors)),
            "Monitor response": Binding(y=tuple(params.monitors)),
        },
    )


def _orm() -> Case:
    params = test_orm_render._params()
    correctors, bpms = tuple(params.correctors), tuple(params.readbacks)
    rows = test_orm_render._rows(
        list(correctors),
        list(bpms),
        test_orm_render._truth(len(bpms), len(correctors)),
        test_orm_render._currents(params.span, params.num, sweep="symmetric"),
    )
    response = Ratio(over=bpms, per=correctors)
    return Case(
        roles={"corrector": correctors, "bpm": bpms},
        draw=_drawing(orm.render, rows, params),
        bindings={
            "Response matrix": Binding(value=response),
            "Response by BPM": Binding(y=response),
            "Corrector anomaly score": Binding(),
            "BPM anomaly score": Binding(),
            "Singular values": Binding(),
            **{f"{name} sweep": Binding(x=(name,), y=bpms) for name in correctors},
        },
    )


PLAN_CASES: dict[str, list[Callable[[], Case]]] = {
    "grid_scan": [
        lambda: _grid([test_grid_render._axis("motor1", 0.0, 4.0, 5)]),
        lambda: _grid(
            [
                test_grid_render._axis("motor1", 0.0, 2.0, 3),
                test_grid_render._axis("motor2", 0.0, 3.0, 4),
            ]
        ),
        lambda: _grid(
            [
                test_grid_render._axis("motor1", 0.0, 1.0, 2),
                test_grid_render._axis("motor2", 0.0, 2.0, 3),
                test_grid_render._axis("motor3", 0.0, 3.0, 4),
            ]
        ),
    ],
    "orbit_bump_sweep": [_orbit_bump_sweep],
    "orm": [_orm],
}
"""Every built-in plan with a ``render``, keyed by its ``plans_core`` module name."""

CASES: dict[str, Callable[[], Case]] = {
    "default_figure-row-index": _default_row_index,
    "default_figure-movable-x": _default_movable_x,
    **{
        f"{plan}-{index}": build
        for plan, builds in PLAN_CASES.items()
        for index, build in enumerate(builds)
    },
}


def _shipped_plans_with_a_render() -> set[str]:
    found: set[str] = set()
    for module_info in pkgutil.iter_modules(plans_core.__path__):
        module = importlib.import_module(f"{plans_core.__name__}.{module_info.name}")
        if callable(getattr(module, "render", None)):
            found.add(module_info.name)
    return found


def test_every_shipped_plan_has_a_case() -> None:
    assert _shipped_plans_with_a_render() == set(PLAN_CASES)


# =========================================================================
# The relation, over three windows
# =========================================================================


def _columns(case: Case) -> list[str]:
    return [column for columns in case.roles.values() for column in columns]


def _distinct(case: Case) -> dict[str, str]:
    return {column: f"u:{column}" for column in _columns(case)}


def _shared(case: Case) -> dict[str, str]:
    return {column: f"u:{role}" for role, columns in case.roles.items() for column in columns}


def _axis(columns: Columns | None) -> Axis:
    if columns is None:
        return ()
    if columns.per:
        return Ratio(over=tuple(columns.names), per=tuple(columns.per))
    return tuple(columns.names)


def _named(panel: Panel) -> Binding:
    value = panel.mark.value_columns if isinstance(panel.mark, HeatmapMark) else None
    return Binding(x=_axis(panel.x_columns), y=_axis(panel.y_columns), value=_axis(value))


def _stated(panel: Panel) -> Binding:
    value = panel.mark.value_units if isinstance(panel.mark, HeatmapMark) else None
    return Binding(x=panel.x_units, y=panel.y_units, value=value)


def _expected(binding: Binding, units: Mapping[str, str]) -> Binding:
    return Binding(*(_declared(units, columns) for columns in binding))


def _drawn(case: Case, units: Mapping[str, str]) -> list[Panel]:
    figure = bind_units(*case.draw(units))
    assert figure.panels, "the case draws no panel, so it proves nothing"
    assert {panel.title for panel in figure.panels} == set(case.bindings)
    return figure.panels


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_axis_names_the_columns_it_draws(name: str) -> None:
    case = CASES[name]()
    for panel in _drawn(case, _distinct(case)):
        assert _named(panel) == case.bindings[panel.title], panel.title


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_render_that_types_a_unit_is_refused(name: str) -> None:
    case = CASES[name]()
    figure, window = case.draw(_distinct(case))
    figure.panels[0].y_units = "typed"
    with pytest.raises(ValueError, match="y_units"):
        bind_units(figure, window)


@pytest.mark.parametrize("window", ["distinct", "shared"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_each_axis_states_the_unit_its_columns_declare(name: str, window: str) -> None:
    case = CASES[name]()
    units = _distinct(case) if window == "distinct" else _shared(case)

    for panel in _drawn(case, units):
        assert _stated(panel) == _expected(case.bindings[panel.title], units), panel.title


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_run_that_declares_no_units_states_none(name: str) -> None:
    for panel in _drawn(CASES[name](), {}):
        assert _stated(panel) == Binding(None, None, None), panel.title


def test_a_single_column_axis_states_its_own_column_unit() -> None:
    """The distinct window does reach a stated unit: no case passes on None alone."""
    for name in sorted(CASES):
        case = CASES[name]()
        units = _distinct(case)
        stated = [
            unit for panel in _drawn(case, units) for unit in _stated(panel) if unit is not None
        ]
        single = [
            columns
            for binding in case.bindings.values()
            for columns in binding
            if not isinstance(columns, Ratio) and len(columns) == 1
        ]
        assert bool(stated) == bool(single), name


def test_every_bump_panel_states_its_role_unit_when_the_role_shares_one() -> None:
    case = _orbit_bump_sweep()
    for panel in _drawn(case, _shared(case)):
        assert panel.y_units is not None, panel.title
