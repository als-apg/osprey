"""Workspace file tree, content, and SSE event routes."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

logger = logging.getLogger(__name__)

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


#: The server-side stores the file routes never show, each named by the
#: ``app.state`` attribute its directory is published under and by what a
#: warning calls it. Both are the server's own state, written through their own
#: routes: the feedback store holds session context users submitted privately,
#: and the bar-items store holds each operator's saved bar layout.
_CONCEALED_STORES: tuple[tuple[str, str], ...] = (
    ("feedback_dir", "feedback store"),
    ("bar_items_dir", "bar-items store"),
)


def _concealed_stores(request: Request, workspace_root: Path) -> tuple[Path, ...]:
    """Resolve every concealed store that lies inside the served tree.

    Returns the *resolved* path of each store in :data:`_CONCEALED_STORES` that
    sits inside ``workspace_root`` (itself already resolved). A store being
    unset, unresolvable, or sited outside the served tree (as it is when
    ``web_terminal.watch_dir`` points elsewhere) means there is nothing of it to
    conceal here, and the routes behave for it exactly as they would without
    it. An empty tuple conceals nothing.

    Identity is the resolved path, never the directory name, so a directory a
    user legitimately named ``feedback`` or ``bar_items`` elsewhere in the
    workspace stays browsable.
    """
    stores: list[Path] = []
    for attribute, label in _CONCEALED_STORES:
        store_dir = getattr(request.app.state, attribute, None)
        if store_dir is None:
            continue
        try:
            resolved = Path(store_dir).resolve()
        except (OSError, TypeError, ValueError):
            # ``TypeError`` is the realistic one: a non-path value on app.state.
            # Non-strict ``resolve()`` swallows ENOENT and ELOOP, so ``OSError`` is
            # belt-and-braces. Fail open — there is no path to compare against — but
            # never quietly: this is a privacy control, and a silent skip looks
            # exactly like success.
            logger.warning(
                "Could not resolve the %s path %r; it will NOT be concealed from the file browser",
                label,
                store_dir,
                exc_info=True,
            )
            continue
        # A lexical test is sound for siting: both paths come from ``app.state``,
        # derived from one config read, so they share their spelling. It also holds
        # before the store directory has been created, which ``samefile`` cannot.
        if resolved.is_relative_to(workspace_root):
            stores.append(resolved)
    return tuple(stores)


def _is_within_concealed_store(path: Path, stores: tuple[Path, ...], workspace_root: Path) -> bool:
    """Whether ``path`` *is* one of ``stores`` or lives underneath one.

    Compares filesystem identity while walking ``path``'s ancestors up to
    ``workspace_root``, because comparing path parts is not sound here:
    :meth:`Path.resolve` follows symlinks but does **not** canonicalize case, so
    on a case-insensitive filesystem (APFS, NTFS) a request for a store spelled
    in another case resolves to itself, compares unequal to the store, and
    would otherwise be served.

    :meth:`Path.samefile` raises ``OSError`` when a path does not exist — the
    ordinary case for the leaf of a miss — so a failing comparison is skipped
    and the walk continues to its parents.
    """
    for candidate in (path, *path.parents):
        for store in stores:
            try:
                if candidate.samefile(store):
                    return True
            except OSError:
                pass
        # Stop at the served root, and at the filesystem root for a path that
        # resolved outside it — neither has an ancestor worth stat-ing.
        if candidate == workspace_root or candidate == candidate.parent:
            break
    return False


@router.get("/api/files/tree")
async def file_tree(request: Request):
    """Return the workspace directory tree as JSON.

    A concealed store is omitted when it lies inside the served tree, as is any
    symlink leading into one — see :func:`_concealed_stores`.
    """
    workspace_dir: Path = _resolve_workspace(request)
    workspace_root = workspace_dir.resolve()
    stores = _concealed_stores(request, workspace_root)

    def conceals(entry: Path) -> bool:
        return bool(stores) and _is_within_concealed_store(entry.resolve(), stores, workspace_root)

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
                if conceals(entry):
                    continue
                children.append(build_tree(entry, depth + 1))
            else:
                # A symlink is the case worth paying for: a concealed store's
                # directory is pruned above, so its documents are never walked
                # into. A hardlink to a document in one would still be listed —
                # no path-based predicate can see that, and making one costs
                # write access to the workspace, which grants more than this
                # leaks.
                if entry.is_symlink() and conceals(entry):
                    continue
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
    """Return file content with path traversal protection.

    Anything under a concealed store answers 404 — byte-for-byte what a path
    that was never there returns, so a probe cannot confirm the store exists.
    """
    workspace_dir: Path = _resolve_workspace(request)
    workspace_root = workspace_dir.resolve()
    resolved = (workspace_dir / filepath).resolve()

    if not resolved.is_relative_to(workspace_root):
        raise HTTPException(status_code=403, detail="Path traversal blocked")

    stores = _concealed_stores(request, workspace_root)
    if stores and _is_within_concealed_store(resolved, stores, workspace_root):
        raise HTTPException(status_code=404, detail="File not found")

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
        raise HTTPException(status_code=415, detail="Binary file — preview not supported") from None

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
