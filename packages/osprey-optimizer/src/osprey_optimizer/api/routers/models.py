"""Models API router.

Provides endpoints for listing and uploading surrogate model files.
"""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, status

from osprey_optimizer.models.api import ModelFileInfo
from osprey_optimizer.results import save_uploaded_model, scan_model_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

# Cap uploaded model files so a large upload can't be read unbounded into memory.
MAX_MODEL_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@router.get("", response_model=list[ModelFileInfo])
def list_models() -> list[ModelFileInfo]:
    """List available saved surrogate model files."""
    try:
        return scan_model_files()
    except Exception as e:
        logger.exception("Failed to list models")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to list models: {e}",
        ) from e


@router.post("", response_model=ModelFileInfo)
async def upload_model(file: UploadFile) -> ModelFileInfo:
    """Upload a surrogate model file (.joblib)."""
    filename = file.filename or ""
    if not filename.endswith(".joblib"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Only .joblib files are accepted.",
        )
    # Read one byte past the limit so we can detect (and reject) oversize files
    # without pulling an unbounded amount into memory.
    content = await file.read(MAX_MODEL_UPLOAD_BYTES + 1)
    if len(content) > MAX_MODEL_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Model file exceeds the {MAX_MODEL_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    try:
        return save_uploaded_model(filename, content)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Failed to save uploaded model")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded model: {e}",
        ) from e
