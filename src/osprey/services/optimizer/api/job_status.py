"""Job metadata lookup + the canonical ``JobResponse`` builder.

Lives outside the jobs router so the job lifecycle modules can
build the same response shape without importing a router (which would be a
circular import). ``read_job_status`` is THE single source of every lifecycle
endpoint's response body.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import HTTPException, status
from pydantic import ValidationError

from osprey.services.optimizer.api.job_tracker import JobMeta, JobTracker, create_job_state
from osprey.services.optimizer.models.api import JobResponse

logger = logging.getLogger(__name__)


def load_job_meta(tracker: JobTracker, job_id: str) -> JobMeta:
    """Fetch a job's metadata, raising 404 if missing (Redis expiry), 500 if corrupt.

    The single read seam for job metadata: every caller that needs a job's type
    or creation time goes through here, so the HTTP mapping of a vanished or
    unparsable record is defined in exactly one place.
    """
    try:
        job_data = tracker.get_job(job_id)
    except ValidationError as e:
        logger.exception("Corrupt job metadata for job %s", job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Corrupt job metadata for job {job_id}: {e}",
        ) from e
    if job_data is None:
        logger.warning("Job %s metadata missing (possible Redis expiry)", job_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return job_data


def read_job_status(tracker: JobTracker, job_id: str) -> JobResponse:
    """Build the canonical ``JobResponse`` from meta + live state.

    404 if meta is missing, 500 if corrupt. The single response builder shared
    by every lifecycle endpoint (create/start/pause/resume/cancel) and the
    ``GET /jobs`` reads.
    """
    meta = load_job_meta(tracker, job_id)
    reader = create_job_state(meta.job_type, job_id)
    created_at_iso = datetime.fromtimestamp(meta.created_at, tz=UTC).isoformat()
    return JobResponse(
        job_id=job_id,
        job_type=meta.job_type,
        status=reader.get_status(),
        logs=reader.get_logs(),
        error_message=reader.get_error_message(),
        created_at=created_at_iso,
    )
