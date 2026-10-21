"""The lattice dashboard's HTTP responses.

One model per response body, so the page, the agent tools and the pinned
OpenAPI document read one contract. A figure route answers 200 only for a
figure of the inputs on screen (``FigureResponse`` / ``DataResponse``), 404
with ``FigureNotCurrent`` while it is not, and 409 with ``FigureUnavailable``
for a figure the selected model cannot draw.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, RootModel

#: A figure's status for the inputs on screen.
FigureStatus = Literal["ready", "computing", "failed", "stale", "not_computed"]

#: Every status but ``ready``: the figure route answers 404 with it.
NotCurrentStatus = Literal["computing", "failed", "stale", "not_computed"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


class CapabilitiesModel(BaseModel):
    """The figures the selected model draws."""

    figures: list[str]
    fast_figures: list[str]
    verify: bool


class SelectionModel(BaseModel):
    """The model the dashboard draws."""

    model: str | None
    deck_sha256: str | None
    status: Literal["loading", "ready", "failed", "none"]
    error: str | None
    capabilities: CapabilitiesModel


class FigureStatusModel(BaseModel):
    """One figure's status for the inputs on screen."""

    status: FigureStatus
    key: str | None
    updated: str | None
    error: str | None


class FamilyModel(BaseModel):
    """One magnet family of the selected deck."""

    type: Literal["quadrupole", "sextupole"]
    param: str
    value: float
    count: int
    range: list[float]


class BaselineResponse(BaseModel):
    """The comparison baseline: the overrides and summary it was set at."""

    summary: dict[str, Any]
    overrides: dict[str, float]
    set_at: str


class SettingsResponse(RootModel[dict[str, dict[str, float | int | None]]]):
    """The computation settings, by group."""


class SettingsResetResponse(BaseModel):
    status: Literal["ok"]
    settings: SettingsResponse


class StatusResponse(BaseModel):
    status: Literal["ok"]


class StateResponse(BaseModel):
    """The dashboard state for the inputs on screen."""

    selection: SelectionModel
    families: dict[str, FamilyModel]
    overrides: dict[str, float]
    summary: dict[str, Any]
    figures: dict[str, FigureStatusModel]
    baseline: BaselineResponse | None
    settings: SettingsResponse
    notice: str | None


class ModelEntry(BaseModel):
    """One model the build offers."""

    name: str
    served: bool
    solve: str | None
    selected: bool
    error: str | None


class LaunchResponse(BaseModel):
    """The figures a refresh launched."""

    status: Literal["ok"]
    launched: list[str]


class FigureResponse(BaseModel):
    """A figure of the inputs on screen, as a Plotly figure."""

    status: Literal["ready"]
    key: str
    figure: dict[str, Any]


class DataResponse(BaseModel):
    """A figure of the inputs on screen, as the raw data it is drawn from."""

    status: Literal["ready"]
    key: str
    data: dict[str, Any]


class FigureNotCurrent(BaseModel):
    """Why there is no figure for the inputs on screen (404)."""

    status: NotCurrentStatus
    key: str | None
    error: str | None


class FigureUnavailable(BaseModel):
    """A figure the selected model cannot draw (409)."""

    status: Literal["unavailable"]
    reason: str
