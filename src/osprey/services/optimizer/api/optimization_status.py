"""Build the unified ``JobResponse`` for optimization jobs (summary/full).

Lives outside the router so ``jobs.py`` can produce the rich optimization
response without importing a router module (avoids a circular import).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from osprey.services.optimizer.api.job_status import load_job_meta
from osprey.services.optimizer.api.job_tracker import JobTracker
from osprey.services.optimizer.api.summary import SummaryContext, compute_result_summary
from osprey.services.optimizer.models.api import JobResponse, ResultSummary
from osprey.services.optimizer.models.enums import JobStatus, JobType
from osprey.services.optimizer.optimization.state import (
    OptimizationStateReader,
    get_optimization_reader,
)
from osprey.services.optimizer.utils.json_safe import drop_non_finite

logger = logging.getLogger(__name__)


def _metadata_section(
    metadata: dict[str, Any] | None, key: str, job_id: str
) -> dict[str, dict[str, object]] | None:
    """Return ``metadata[key]`` as name -> field-dict, or None.

    The reader only type-checks the outer metadata container; a corrupt section
    (or a corrupt per-name entry) would otherwise reach ``SummaryContext`` and
    500 the status read (#85). Such a section is dropped with a warning.
    """
    section = metadata.get(key) if metadata else None
    if section is None:
        return None
    if not isinstance(section, dict) or not all(isinstance(v, dict) for v in section.values()):
        logger.warning(
            "Ignoring malformed %r metadata for job %s; omitting it from result_summary",
            key,
            job_id,
        )
        return None
    return section


def _compute_summary(
    job_id: str,
    state: OptimizationStateReader,
    opt_status: JobStatus,
    last_results: Path | None,
) -> ResultSummary | None:
    """Compute the result summary, or None when state is insufficient."""
    if opt_status == JobStatus.IDLE:
        return None
    data_df = state.get_data()
    objective_name = state.get_objective_name()
    variable_names = state.get_variable_names()
    if data_df is None or not objective_name or not variable_names:
        return None
    direction = state.get_objective_direction()
    if direction is None:
        logger.warning(
            "Objective direction not set in state for job %s; omitting result_summary",
            job_id,
        )
        return None

    snapshots = state.get_available_snapshots()
    metadata = state.get_metadata()
    model_file: str | None = None
    if last_results:
        model_path = last_results / "model.joblib"
        if model_path.exists():
            model_file = "model.joblib"

    return compute_result_summary(
        data=data_df,
        context=SummaryContext(
            objective_name=objective_name,
            variable_names=variable_names,
            direction=direction,
            initial_point=snapshots.get("initial"),
            target_value=snapshots.get("target"),
            variable_metadata=_metadata_section(metadata, "variables", job_id),
            observable_metadata=_metadata_section(metadata, "observables", job_id),
            model_file=model_file,
        ),
    )


def build_optimization_response(
    job_id: str,
    detail: Literal["summary", "full"],
    tracker: JobTracker,
) -> JobResponse:
    """Build the unified ``JobResponse`` for an optimization job.

    ``state`` is the read-only Reader Protocol (mypy rejects setter calls here,
    enforcing the read-only guarantee). Marks the job completed on terminal
    status. ``detail=full`` is a superset: comprehensive state plus
    ``result_summary``.
    """
    created_at = load_job_meta(tracker, job_id).created_at
    created_at_iso = datetime.fromtimestamp(created_at, tz=UTC).isoformat()

    state = get_optimization_reader(job_id)
    opt_status = state.get_status()
    error_msg = state.get_error_message()
    last_results = state.get_last_results_path()

    if opt_status.is_terminal:
        tracker.mark_completed(job_id)

    results_path = None
    if last_results:
        results_path = f"{last_results.parent.name}/{last_results.name}"

    result_summary = _compute_summary(job_id, state, opt_status, last_results)

    response = JobResponse(
        job_id=job_id,
        job_type=JobType.OPTIMIZATION,
        status=opt_status,
        created_at=created_at_iso,
        error_message=error_msg,
        results_path=results_path,
        logs=state.get_logs(),
        result_summary=result_summary,
    )

    if detail == "full":
        data_df = state.get_data()
        response.data_version = state.get_data_version()
        response.environment_name = state.get_environment_name()
        response.objective_name = state.get_objective_name()
        response.objective_direction = state.get_objective_direction()
        response.variable_names = state.get_variable_names()
        response.variable_bounds = state.get_variable_bounds()
        response.total_evaluations = state.get_total_evaluations()
        # Failed objective/observable reads surface as NaN; scrub to None so the
        # Starlette renderer (json.dumps(allow_nan=False)) does not 500 the response.
        response.data = data_df.fill_nan(None).to_dicts() if data_df is not None else None
        # Snapshots are sparse reference points; drop non-finite entries (e.g. an
        # unreadable initial objective surfacing as NaN) so the response stays
        # JSON-serializable.
        response.snapshots = {
            name: drop_non_finite(values)
            for name, values in state.get_available_snapshots().items()
        }

    return response
