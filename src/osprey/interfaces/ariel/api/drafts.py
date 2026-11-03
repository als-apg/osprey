"""ARIEL Draft Entry API — filesystem-based draft store.

Allows the agent (via MCP tool) to pre-fill the web form so a human
can review, edit, and submit manually. Drafts are stored as JSON files
in _agent_data/drafts/ and expire after 1 hour.

A draft is ``<store>/<id>.json``; the artifact gallery may add a
``<store>/<id>-metadata.json`` sidecar beside it. Every route that takes an id
checks it against ``DRAFT_ID_PATTERN`` before touching the store, so a sidecar
id or a traversal id is refused with 422.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from osprey.interfaces.ariel.api.attachment_response import attachment_response
from osprey_connectors.logger import get_logger
from osprey_connectors.workspace import resolve_shared_data_root

logger = get_logger("ariel.drafts")


# Drafts directory is resolved lazily so OSPREY_CONFIG changes (via
# site_config) are respected at access time, not import time. We expose it
# as a helper function rather than a module-level __getattr__ because PEP-562
# __getattr__ only fires on external `module.NAME` lookups — bare-name
# references inside this module would NameError. The matching `entry.py`
# pattern uses `_get_drafts_dir`; tests monkeypatch `_drafts_dir` to redirect.
def _drafts_dir() -> Path:
    return resolve_shared_data_root() / "drafts"


draft_router = APIRouter(prefix="/api")


class DraftCreateRequest(BaseModel):
    """Request body for creating a draft."""

    subject: str
    details: str
    author: str | None = None
    logbook: str | None = None
    shift: str | None = None
    tags: list[str] | None = None


class DraftCreateResponse(BaseModel):
    """Response after creating a draft."""

    draft_id: str
    message: str


class DraftResponse(BaseModel):
    """Full draft data returned to the frontend."""

    draft_id: str
    subject: str
    details: str
    author: str | None = None
    logbook: str | None = None
    shift: str | None = None
    tags: list[str] | None = None
    attachment_paths: list[str] | None = None
    metadata: dict | None = None
    fields: dict[str, Any] | None = None


class DraftSummary(BaseModel):
    """One row of the Drafts mailbox."""

    draft_id: str
    subject: str | None = None
    author: str | None = None
    logbook: str | None = None
    created_at: str
    staged_by: str | None = None
    attachment_count: int = 0


class DraftListResponse(BaseModel):
    """Every unexpired draft, newest first."""

    drafts: list[DraftSummary]


DRAFT_TTL_SECONDS = 3600  # 1 hour

#: The only draft id shape the routes accept (C-DRAFT-ID).
DRAFT_ID_PATTERN = re.compile(r"draft-[0-9a-f]{12}")

SIDECAR_SUFFIX = "-metadata"


def is_draft_id(value: str) -> bool:
    """Return whether ``value`` is a well-formed draft id."""
    return DRAFT_ID_PATTERN.fullmatch(value) is not None


def _require_draft_id(draft_id: str) -> str:
    """Refuse a malformed id with 422 before any filesystem access."""
    if not is_draft_id(draft_id):
        raise HTTPException(status_code=422, detail="Invalid draft id")
    return draft_id


def _ensure_drafts_dir() -> Path:
    """Ensure the drafts directory exists and return its path."""
    drafts_dir = _drafts_dir()
    drafts_dir.mkdir(parents=True, exist_ok=True)
    return drafts_dir


def _draft_paths(drafts_dir: Path, draft_id: str) -> tuple[Path, Path]:
    """The draft JSON and its sidecar, both built from a validated id."""
    return drafts_dir / f"{draft_id}.json", drafts_dir / f"{draft_id}{SIDECAR_SUFFIX}.json"


def _is_expired(path: Path, now: float) -> bool:
    try:
        return now - path.stat().st_mtime > DRAFT_TTL_SECONDS
    except FileNotFoundError:
        return False


def _cleanup_expired_drafts() -> None:
    """Delete drafts older than DRAFT_TTL_SECONDS together with their sidecars.

    A sidecar follows its draft; a sidecar with no draft expires on its own
    mtime. Files whose name does not follow the draft-id contract are left alone.
    """
    drafts_dir = _drafts_dir()
    if not drafts_dir.exists():
        return
    now = time.time()
    for path in drafts_dir.glob("draft-*.json"):
        stem = path.stem
        if is_draft_id(stem):
            if _is_expired(path, now):
                for target in _draft_paths(drafts_dir, stem):
                    target.unlink(missing_ok=True)
        elif stem.endswith(SIDECAR_SUFFIX) and is_draft_id(stem[: -len(SIDECAR_SUFFIX)]):
            draft_path, _ = _draft_paths(drafts_dir, stem[: -len(SIDECAR_SUFFIX)])
            if not draft_path.exists() and _is_expired(path, now):
                path.unlink(missing_ok=True)


def _summary(path: Path) -> DraftSummary | None:
    """One list row for a draft file, or None (logged) when it is unusable."""
    try:
        mtime = path.stat().st_mtime
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Skipping unreadable draft %s: %s", path.name, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Skipping draft %s: not a JSON object", path.name)
        return None
    if data.get("draft_id") != path.stem:
        logger.warning("Skipping draft %s: draft_id does not match the file name", path.name)
        return None
    metadata = data.get("metadata")
    session = metadata.get("session_metadata") if isinstance(metadata, dict) else None
    staged_by = session.get("created_via") if isinstance(session, dict) else None
    attachments = data.get("attachment_paths")

    def text(key: str) -> str | None:
        value = data.get(key)
        return value if isinstance(value, str) else None

    return DraftSummary(
        draft_id=path.stem,
        subject=text("subject"),
        author=text("author"),
        logbook=text("logbook"),
        created_at=datetime.fromtimestamp(mtime, tz=UTC).isoformat(),
        staged_by=staged_by if isinstance(staged_by, str) else None,
        attachment_count=len(attachments) if isinstance(attachments, list) else 0,
    )


def generate_draft_id() -> str:
    """Generate a unique draft ID."""
    return f"draft-{uuid.uuid4().hex[:12]}"


def write_draft(draft_id: str, data: dict[str, Any]) -> Path:
    """Write a draft JSON file to the drafts directory."""
    drafts_dir = _ensure_drafts_dir()
    filepath = drafts_dir / f"{draft_id}.json"
    filepath.write_text(json.dumps(data, indent=2))
    return filepath


def read_draft(draft_id: str) -> dict[str, Any] | None:
    """Read a draft JSON file. Returns None if not found."""
    filepath = _drafts_dir() / f"{draft_id}.json"
    if not filepath.exists():
        return None
    return cast(dict[str, Any], json.loads(filepath.read_text()))


@draft_router.post("/drafts", response_model=DraftCreateResponse)
async def create_draft(req: DraftCreateRequest) -> DraftCreateResponse:
    """Create a new draft entry (pre-fill data for the web form)."""
    draft_id = generate_draft_id()
    data = {
        "draft_id": draft_id,
        "subject": req.subject,
        "details": req.details,
        "author": req.author,
        "logbook": req.logbook,
        "shift": req.shift,
        "tags": req.tags or [],
    }
    write_draft(draft_id, data)
    return DraftCreateResponse(
        draft_id=draft_id,
        message=f"Draft {draft_id} created",
    )


@draft_router.get("/drafts", response_model=DraftListResponse)
async def list_drafts() -> DraftListResponse:
    """List every unexpired draft, newest first. Triggers cleanup of expired drafts."""
    _cleanup_expired_drafts()
    drafts_dir = _drafts_dir()
    if not drafts_dir.exists():
        return DraftListResponse(drafts=[])
    rows = [
        row
        for path in drafts_dir.glob("draft-*.json")
        if is_draft_id(path.stem) and (row := _summary(path)) is not None
    ]
    rows.sort(key=lambda row: row.created_at, reverse=True)
    return DraftListResponse(drafts=rows)


@draft_router.get("/drafts/{draft_id}", response_model=DraftResponse)
async def get_draft(draft_id: str) -> DraftResponse:
    """Retrieve a draft by ID. Triggers cleanup of expired drafts."""
    _require_draft_id(draft_id)
    _cleanup_expired_drafts()

    data = read_draft(draft_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")

    return DraftResponse(**data)


@draft_router.get("/drafts/{draft_id}/attachments/{filename}")
async def get_draft_attachment(draft_id: str, filename: str) -> Response:
    """Serve a staged attachment file from a draft.

    Only files listed in the draft's ``attachment_paths`` are served
    (prevents path-traversal attacks).
    """
    _require_draft_id(draft_id)
    data = read_draft(draft_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")

    attachment_paths = data.get("attachment_paths") or []
    # Find the path whose basename matches the requested filename
    matched_path: str | None = None
    for p in attachment_paths:
        if Path(p).name == filename:
            matched_path = p
            break

    if matched_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Attachment '{filename}' not found in draft {draft_id}",
        )

    file_path = Path(matched_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Attachment file not found on disk")

    return attachment_response(file_path.read_bytes(), filename)


@draft_router.delete("/drafts/{draft_id}", status_code=204)
async def delete_draft(draft_id: str) -> Response:
    """Discard a draft: remove exactly its JSON file and its sidecar.

    Both paths are built from the validated id; nothing named inside the
    draft (such as its staged attachments) is ever removed.
    """
    _require_draft_id(draft_id)
    draft_path, sidecar_path = _draft_paths(_drafts_dir(), draft_id)
    try:
        draft_path.unlink()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found") from None
    sidecar_path.unlink(missing_ok=True)
    return Response(status_code=204)
