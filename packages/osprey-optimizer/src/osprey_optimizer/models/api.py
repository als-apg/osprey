"""Request and response models of the optimizer service's HTTP API.

Shared by the API server and its clients.
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic import ConfigDict as PydanticConfigDict

from .config import ControlTarget, VariableOverride
from .enums import EnvironmentType, JobType
from .protocols import EnvironmentProtocol, ObservableMetadata, PVReading, VariableMetadata
from .state import OptimizationState
from .types import BestPoint, ConfigDict, Direction


class EnvironmentInfo(BaseModel):
    """Information about an available environment."""

    name: str
    display_name: str
    description: str | None = None
    valid: bool = True
    error: str | None = None
    source: str = "plugin"


class EnvironmentDetails(BaseModel):
    """Detailed information about an environment, including its variables."""

    name: str
    environment_type: EnvironmentType
    variables: dict[str, VariableMetadata]
    observables: dict[str, ObservableMetadata]
    default_objective: str
    available_objectives: list[str]

    @classmethod
    def from_environment(cls, name: str, env: "EnvironmentProtocol") -> "EnvironmentDetails":
        """Project a live environment onto this response model.

        The single projection: the API's details endpoint and the CLI's local
        rendering both go through it, so adding a field here cannot leave one
        of them behind.
        """
        return cls(
            name=name,
            environment_type=env.environment_type,
            variables=env.variables,
            observables=env.observables,
            default_objective=env.default_objective,
            available_objectives=env.available_objectives,
        )


class EnvironmentStatusResponse(BaseModel):
    """Response model for environment connectivity status."""

    name: str
    connected: bool | None
    message: str | None = None


class SetVariablesRequest(BaseModel):
    """Request body for PUT /environments/{name}/variables."""

    model_config = PydanticConfigDict(extra="forbid")

    variables: dict[str, float]
    override_unhealthy: bool = False


class EvaluateRequest(BaseModel):
    """Request body for POST /environments/{name}/evaluate."""

    model_config = PydanticConfigDict(extra="forbid")

    variables: dict[str, float]
    observables: list[str] = Field(min_length=1)
    override_unhealthy: bool = False


class EvaluateResponse(BaseModel):
    """Response from POST /environments/{name}/evaluate."""

    # Variable readbacks and observables cross the wire as native PVReadings,
    # carrying value/connected/timestamp; consumers coerce to float as needed.
    variable_readbacks: dict[str, PVReading]
    observables: dict[str, PVReading]


class ResolveBoundsRequest(BaseModel):
    """Request body for POST /environments/{name}/resolve-bounds."""

    model_config = PydanticConfigDict(extra="forbid")

    overrides: dict[str, VariableOverride] = Field(default_factory=dict)
    current_values: dict[str, float]


class ResolvedBoundsModel(BaseModel):
    """Bounds resolved for a single variable, after clamping to the safety range.

    ``requested_min`` / ``requested_max`` are the pre-clamp values; consumers
    that emit clamp notifications use them to report what the operator asked
    for vs. what was actually applied.
    """

    min: float
    max: float
    requested_min: float
    requested_max: float
    was_clamped: bool


class ResolveBoundsResponse(BaseModel):
    """Response from POST /environments/{name}/resolve-bounds.

    Holds only resolved ``bounds``; the caller already supplied the current
    values in the request body.
    """

    bounds: dict[str, ResolvedBoundsModel]


class InitialBoundsResponse(BaseModel):
    """Response from GET /environments/{name}/initial-bounds.

    Bundles resolved ``bounds`` with the env's ``current_values`` (live
    readbacks) so callers get both in a single PV walk.
    """

    bounds: dict[str, ResolvedBoundsModel]
    current_values: dict[str, PVReading] = Field(default_factory=dict)


class AnnotatedValue(BaseModel):
    """A value annotated with metadata for human/LLM readability."""

    value: float
    description: str = ""
    units: str = ""
    delta_vs_initial: float | None = None


class PointDelta(BaseModel):
    """A single optimization point with deltas vs initial state and target."""

    values: dict[str, float]
    delta_vs_initial: dict[str, float] | None = None
    delta_vs_target: dict[str, float] | None = None
    annotated_values: dict[str, AnnotatedValue] | None = None


class ResultSummary(BaseModel):
    """Summary of the top optimization results with deltas."""

    best_points: list[PointDelta]
    initial_point: dict[str, float] | None = None
    target_value: dict[str, float] | None = None

    objective_name: str | None = None
    objective_description: str | None = None
    direction: Direction | None = None
    total_iterations: int | None = None
    best_objective_value: float | None = None
    initial_objective_value: float | None = None
    target_value_scalar: float | None = None
    target_units: str | None = None
    target_achievement_pct: float | None = None
    improved_over_initial: bool | None = None
    tied_count: int | None = None
    tied_note: str | None = None
    text_summary: str | None = None
    model_file: str | None = None


class ResultInfo(BaseModel):
    """Summary of a historical optimization run for the results listing."""

    run_id: str
    environment: str
    timestamp: str
    objective_name: str | None = None
    has_model: bool
    job_id: str | None = None
    """Control-plane job that produced this run (None for legacy runs
    whose directory predates metadata.json)."""


class ModelFileInfo(BaseModel):
    """Information about a saved surrogate model file."""

    label: str
    path: str
    variables: list[str] | None = None
    objective_name: str | None = None


class HistoricalRunDetails(BaseModel):
    """Detailed data for a historical run."""

    timestamp: str
    config: ConfigDict
    best_point: BestPoint | None
    initial_point: BestPoint | None
    data_records: list[dict[str, Any]] | None


class JobResponse(OptimizationState):
    """Unified response for any job.

    Extends the shared ``OptimizationState`` (job_id, status, error_message,
    results_path, logs, data_version, environment_name, objective_name,
    objective_direction, variable_names, data, snapshots) with the job-level
    fields. Optimization-only fields are null/defaults for the basic
    (no-``detail``) read; ``result_summary`` is populated for
    ``detail=summary`` and ``detail=full``; the comprehensive state fields are
    populated only for ``detail=full``.
    """

    job_type: JobType
    created_at: str | None = None
    result_summary: ResultSummary | None = None


class CurrentJobResponse(BaseModel):
    """The server-side 'current job' pointer (id or null)."""

    job_id: str | None = None


class CurrentJobRequest(BaseModel):
    """Body for setting the current job pointer."""

    job_id: str


class RestoreResponse(BaseModel):
    """Response for ``POST /results/{run_id}/restore``."""

    environment: str
    control_target: ControlTarget
    restored: dict[str, float]
    count: int


class RestorePreviewResponse(BaseModel):
    """Response for ``GET /results/{run_id}/restore-preview`` (no write performed)."""

    environment: str
    control_target: ControlTarget
    values: dict[str, float]
    count: int
    preview_token: str
