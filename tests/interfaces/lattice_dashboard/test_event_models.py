"""Every live-update event is a declared model, joined in one union on ``type``.

An event survives its own wire form: dumped to JSON and read back through
``DashboardEvent`` it is the same class with the same fields. A
``figure_error`` always carries its reason, and a ``type`` outside the union
is refused.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from osprey.interfaces.lattice_dashboard.schema import (
    BaselineClearedEvent,
    BaselineSetEvent,
    DashboardEvent,
    FigureErrorEvent,
    FigureReadyEvent,
    FigureStatusEvent,
    SettingsResponse,
    SettingsUpdatedEvent,
    StateUpdatedEvent,
)

_ADAPTER: TypeAdapter[DashboardEvent] = TypeAdapter(DashboardEvent)

_EVENTS: list[BaseModel] = [
    StateUpdatedEvent(type="state_updated"),
    FigureStatusEvent(type="figure_status", name="optics", status="computing"),
    FigureReadyEvent(type="figure_ready", name="optics"),
    FigureErrorEvent(type="figure_error", name="lma", error="Worker exited with code 1"),
    SettingsUpdatedEvent(
        type="settings_updated", settings=SettingsResponse({"da": {"nturns": 512}})
    ),
    BaselineSetEvent(type="baseline_set", summary={"energy_gev": 2.0}),
    BaselineClearedEvent(type="baseline_cleared"),
]


@pytest.mark.parametrize("event", _EVENTS, ids=lambda e: type(e).__name__)
def test_event_round_trips_through_the_union(event: BaseModel) -> None:
    back = _ADAPTER.validate_json(event.model_dump_json())

    assert type(back) is type(event)
    assert back == event


def test_figure_error_without_a_reason_is_refused() -> None:
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python({"type": "figure_error", "name": "lma"})


def test_figure_error_with_a_null_reason_is_refused() -> None:
    with pytest.raises(ValidationError):
        FigureErrorEvent(type="figure_error", name="lma", error=None)  # type: ignore[arg-type]


def test_an_unknown_type_is_refused() -> None:
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python({"type": "ping"})
