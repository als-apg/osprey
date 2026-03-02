"""Claude memory gallery routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.interfaces.web_terminal.claude_memory_service import (
    ClaudeMemoryService,
    MemoryFileExistsError,
    MemoryFileNotFoundError,
    MemoryValidationError,
)

router = APIRouter()


class MemoryFileRequest(BaseModel):
    content: str
    filename: str | None = None


def _memory_service(request: Request) -> ClaudeMemoryService:
    """Construct a ClaudeMemoryService from the request's project dir."""
    return ClaudeMemoryService(request.app.state.project_cwd)


@router.get("/api/claude-memory")
async def list_memory_files(request: Request):
    """List all memory files with metadata."""
    service = _memory_service(request)
    files = service.list_files()
    return {"files": files, "count": len(files)}


@router.get("/api/claude-memory/{filename}")
async def get_memory_file(filename: str, request: Request):
    """Read a single memory file."""
    service = _memory_service(request)
    try:
        return service.read_file(filename)
    except MemoryFileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except MemoryValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/api/claude-memory")
async def create_memory_file(body: MemoryFileRequest, request: Request):
    """Create a new memory file."""
    if not body.filename:
        raise HTTPException(status_code=422, detail="filename is required")
    service = _memory_service(request)
    try:
        return service.create_file(body.filename, body.content)
    except MemoryFileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except MemoryValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.put("/api/claude-memory/{filename}")
async def update_memory_file(filename: str, body: MemoryFileRequest, request: Request):
    """Update an existing memory file."""
    service = _memory_service(request)
    try:
        return service.update_file(filename, body.content)
    except MemoryFileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except MemoryValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.delete("/api/claude-memory/{filename}")
async def delete_memory_file(filename: str, request: Request):
    """Delete a memory file."""
    service = _memory_service(request)
    try:
        return service.delete_file(filename)
    except MemoryFileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except MemoryValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
