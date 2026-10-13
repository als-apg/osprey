from pydantic import BaseModel, Field

from .enums import JobStatus
from .types import Direction


class OptimizationState(BaseModel):
    """Comprehensive optimization state.

    Used for real-time state synchronization between backend and frontend.
    """

    job_id: str
    status: JobStatus
    error_message: str | None = None
    results_path: str | None = None
    logs: str = ""
    data_version: int = 0
    environment_name: str | None = None
    objective_name: str | None = None
    objective_direction: Direction | None = None
    variable_names: list[str] | None = None

    # Resolved initial samples + iterations, set by the runner. Progress-bar denominator,
    # so it counts initial shots and not just iterations.
    total_evaluations: int | None = None

    # Data is stored as records (list of dicts) to avoid Pandas/Polars dependencies in models
    # and to be JSON-serializable for API responses.
    data: list[dict[str, float | str | int | None]] | None = None

    snapshots: dict[str, dict[str, float]] = Field(default_factory=dict)

    # Safety-clamped, grid-snapped VOCS bounds the run actually used. Normalizes the live
    # figure to a fixed [0, 1] band instead of the shifting observed column min/max.
    variable_bounds: dict[str, list[float]] | None = None
