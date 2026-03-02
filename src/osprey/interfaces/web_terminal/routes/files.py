"""Workspace file tree, content, and SSE event routes."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

_UUID_RE = re.compile(r"^[a-f0-9-]{36}$")


def _resolve_workspace(request: Request) -> Path:
    """Resolve workspace dir, optionally scoped to a session.

    Reads ``?session_id=`` query param. Returns the session-scoped
    subdirectory if valid, otherwise the base workspace dir.
    """
    workspace_base: Path = request.app.state.workspace_dir
    session_id = request.query_params.get("session_id")
    if session_id and _UUID_RE.match(session_id):
        return workspace_base / "sessions" / session_id
    return workspace_base


@router.get("/api/files/tree")
async def file_tree(request: Request):
    """Return the workspace directory tree as JSON."""
    workspace_dir: Path = _resolve_workspace(request)

    if not workspace_dir.exists():
        return {"name": workspace_dir.name, "type": "directory", "children": []}

    def build_tree(directory: Path, depth: int = 0) -> dict:
        node = {
            "name": directory.name,
            "path": str(directory.relative_to(workspace_dir)),
            "type": "directory",
        }
        if depth > 10:
            return node

        children = []
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            entries = []

        for entry in entries:
            # Skip hidden/ignored
            if entry.name.startswith(".") or entry.name in (
                "__pycache__",
                "_notebook_cache",
                "node_modules",
            ):
                continue

            if entry.is_dir():
                children.append(build_tree(entry, depth + 1))
            else:
                children.append(
                    {
                        "name": entry.name,
                        "path": str(entry.relative_to(workspace_dir)),
                        "type": "file",
                        "size": entry.stat().st_size,
                    }
                )
        node["children"] = children
        return node

    return build_tree(workspace_dir)


@router.get("/api/files/content/{filepath:path}")
async def file_content(filepath: str, request: Request):
    """Return file content with path traversal protection."""
    workspace_dir: Path = _resolve_workspace(request)
    resolved = (workspace_dir / filepath).resolve()

    if not resolved.is_relative_to(workspace_dir.resolve()):
        raise HTTPException(status_code=403, detail="Path traversal blocked")

    if not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not resolved.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    # Limit file size to 1MB for preview
    size = resolved.stat().st_size
    if size > 1_048_576:
        raise HTTPException(status_code=413, detail="File too large for preview (>1MB)")

    # Detect binary files
    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=415, detail="Binary file — preview not supported"
        ) from None

    return {
        "path": filepath,
        "content": content,
        "size": size,
        "extension": resolved.suffix,
    }


@router.get("/api/files/events")
async def file_events(request: Request):
    """SSE endpoint for real-time file change events."""
    broadcaster = request.app.state.broadcaster
    q = broadcaster.subscribe()

    async def stream():
        try:
            while True:
                data = await q.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            broadcaster.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream")
