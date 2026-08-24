"""Scaffold gallery routes.

The two refusals that most of these routes share — a claim the profile will
not take, and an ownership store that will not take the write — are translated
into 409s by app-level handlers (see
:func:`~osprey.interfaces.web_terminal.app.register_scaffold_conflict_handlers`),
so they are deliberately not caught here. What each route does catch is the
translation that is specific to it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.interfaces.web_terminal.routes.agent_activity import record_activity
from osprey.interfaces.web_terminal.scaffold_gallery_service import (
    ProtectedArtifactError,
    ScaffoldGalleryService,
)

router = APIRouter()

#: Length of the activity ``detail`` string these routes publish. The frame is
#: broadcast verbatim to every connected browser, and a channel phrase plus a
#: path is well under this — the cap exists so a pathological artifact name
#: cannot inflate the stream.
_MAX_ACTIVITY_DETAIL = 1024


class ScaffoldOverrideRequest(BaseModel):
    content: str


def _scaffold_service(request: Request) -> ScaffoldGalleryService:
    """Construct a ScaffoldGalleryService from the request's project dir."""
    return ScaffoldGalleryService(Path(request.app.state.project_cwd))


def _require_scaffold_writes(request: Request) -> None:
    """Refuse with 403 when this deployment has closed the gallery's write surface.

    ``web.scaffold_gallery.write_enabled: false`` is a TIER boundary, not a
    cosmetic one. What these routes author is ``.claude/rules``, ``.claude/
    skills`` and ``.claude/agents`` content, which the agent loads at PROJECT
    scope — instruction it obeys on every turn. A tier that may not re-render
    the agent may not author what the agent reads, so it may not reach these
    routes at all. Hiding the gallery's buttons is the other half and cannot be
    the only half: a client-side gate is undone by typing the URL.

    Called FIRST in every write and delete handler, ahead of the service
    construction and therefore ahead of anything that touches disk or the
    ownership store. This gate and the protected set answer different
    questions — "may this deployment's operators author at all" versus "may
    this particular artifact be written by anyone through the gallery" — and a
    closed gallery never reports a protected-set refusal, because it never gets
    as far as having an artifact to judge.

    Read routes are deliberately not gated. Seeing what the agent is running is
    not authoring it, and a tier that cannot edit still has to be able to look.

    Args:
        request: Incoming request carrying ``app.state``.

    Raises:
        HTTPException: 403 when gallery writes are disabled for this deployment.
    """
    if not getattr(request.app.state, "scaffold_write_enabled", True):
        raise HTTPException(
            status_code=403,
            detail=(
                "scaffold writes are disabled for this deployment "
                "(web.scaffold_gallery.write_enabled: false)"
            ),
        )


def _refuse_protected(
    request: Request, tool: str, name: str, exc: ProtectedArtifactError
) -> HTTPException:
    """Publish a protected-set refusal into the agent-activity history, and build its 403.

    The service raises the refusal and audits it durably, but it is
    constructed per request from a path and holds no ``Request`` — so the one
    place that can put the refusal in front of the operator watching the
    session is the handler. The frame joins the history that
    ``GET /api/agent-activity/recent`` serves, which is what a browser reads
    after connecting, and nothing here broadcasts.

    Returns the 403 rather than raising it so every handler spells the refusal
    as one ``raise _refuse_protected(...) from e``: recording the refusal and
    answering it cannot come apart, and no site can grow a 403 the operator
    never sees.

    Args:
        request: Incoming request, carrying ``app.state``.
        tool: The gallery operation that was refused, named as the agent knows it.
        name: Canonical artifact name the operation asked for.
        exc: The refusal, carrying the channel that owns the target.

    Returns:
        The 403 the caller raises.
    """
    detail = f"refused: {name} — {exc.channel}"[:_MAX_ACTIVITY_DETAIL]
    record_activity(request, tool, {"kind": "artifact", "detail": detail})
    return HTTPException(status_code=403, detail=str(exc))


@router.get("/api/scaffold")
async def list_scaffold(request: Request):
    """List all build artifacts with status and summary counts."""
    service = _scaffold_service(request)
    artifacts = service.list_artifacts()
    framework_count = sum(1 for a in artifacts if a["status"] == "framework")
    user_owned_count = sum(1 for a in artifacts if a["status"] == "user-owned")
    return {
        "artifacts": artifacts,
        "summary": {
            "total": len(artifacts),
            "framework": framework_count,
            "user_owned": user_owned_count,
        },
    }


class UntrackedRegisterRequest(BaseModel):
    name: str


@router.get("/api/scaffold/untracked")
async def list_untracked_scaffold(request: Request):
    """Detect files active in Claude Code but not managed by OSPREY."""
    service = _scaffold_service(request)
    untracked = service.scan_untracked()
    return {"untracked": untracked, "count": len(untracked)}


@router.post("/api/scaffold/untracked/register")
async def register_untracked_scaffold(body: UntrackedRegisterRequest, request: Request):
    """Register an untracked file by adding it to config.yml."""
    _require_scaffold_writes(request)
    service = _scaffold_service(request)
    try:
        return service.register_untracked(body.name)
    except ProtectedArtifactError as e:
        # The name arrives in the request body rather than from the untracked
        # listing, which already drops reserved paths — so this refusal is
        # reachable only by a caller that went around the UI, and it has to be
        # a named 403 rather than the 500 an uncaught PermissionError becomes.
        raise _refuse_protected(request, "register_untracked", body.name, e) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.delete("/api/scaffold/untracked/{name:path}")
async def delete_untracked_scaffold(name: str, request: Request):
    """Delete an untracked file from disk."""
    _require_scaffold_writes(request)
    service = _scaffold_service(request)
    try:
        return service.delete_untracked(name)
    except ProtectedArtifactError as e:
        # 403 rather than 400: the request is well-formed and the file may well
        # exist — the answer is that this writer is not the one allowed to
        # remove it, and the detail names the channel that is.
        raise _refuse_protected(request, "delete_untracked", name, e) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class CreateArtifactRequest(BaseModel):
    category: str
    name: str
    content: str = ""


@router.post("/api/scaffold/create")
async def create_artifact(body: CreateArtifactRequest, request: Request):
    """Create a new custom artifact."""
    _require_scaffold_writes(request)
    service = _scaffold_service(request)
    try:
        return service.create_artifact(body.category, body.name, body.content)
    except ProtectedArtifactError as e:
        # A create aimed at a reserved subtree — ``rules``, ``skills``, an
        # ``osprey_`` hook — is refused by the service before anything is
        # written; without this clause the operator would see that refusal as a
        # 500 with the channel stripped out of it.
        raise _refuse_protected(
            request, "create_artifact", f"{body.category}/{body.name}", e
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.get("/api/scaffold/{name:path}/framework")
async def get_scaffold_framework(name: str, request: Request):
    """Get the framework-rendered content for an artifact."""
    service = _scaffold_service(request)
    try:
        content = service.get_framework_content(name)
        return {"name": name, "content": content, "source": "framework"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/api/scaffold/{name:path}/diff")
async def get_scaffold_diff(name: str, request: Request):
    """Get unified diff between framework and override."""
    service = _scaffold_service(request)
    try:
        return service.compute_diff(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/api/scaffold/{name:path}/claim")
async def claim_scaffold(name: str, request: Request):
    """Claim an artifact for editing (scaffold an override from the framework template)."""
    _require_scaffold_writes(request)
    service = _scaffold_service(request)
    try:
        return service.scaffold_override(name)
    except ProtectedArtifactError as e:
        # Distinct from the 409 a claim the profile will not take becomes: that
        # one says "this artifact is generated, claim something else", while
        # this says the subtree has an owner and it is not this writer. Only
        # the exactly-reserved paths reach the 409 handler at all, so without
        # this clause a claim on a reserved SUBTREE would be a 500.
        raise _refuse_protected(request, "claim", name, e) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.put("/api/scaffold/{name:path}/override")
async def save_scaffold_override(name: str, body: ScaffoldOverrideRequest, request: Request):
    """Save content to an existing override file.

    The service's payload is returned as it stands, and it carries
    ``applies_on_restart``: true when the project tree would not take the write
    and the body is held on the claude-config volume alone. A save that only
    lands at the next container start is still a save, so it is a 200 — but the
    browser has to be told, or the operator reads "saved" and assumes the agent
    is already running their text.
    """
    _require_scaffold_writes(request)
    service = _scaffold_service(request)
    try:
        return service.save_override(name, body.content)
    except ProtectedArtifactError as e:
        # 403, as on the delete path: the artifact exists and the operator may
        # well own it — what they may not do is be the writer of it, and the
        # detail names the channel that is.
        raise _refuse_protected(request, "save_override", name, e) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        # A directory-shaped artifact has no body to save; that is a bad
        # request, not a server fault.
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/api/scaffold/{name:path}/override")
async def delete_scaffold_override(name: str, request: Request):
    """Remove an override, restoring framework management."""
    _require_scaffold_writes(request)
    delete_file = request.query_params.get("delete_file", "false").lower() == "true"
    service = _scaffold_service(request)
    try:
        outcome = service.unoverride(name, delete_file=delete_file)
        if outcome.get("status") == "still-supplied-by-profile":
            # Nothing was released, so this must not read as success. A 409
            # carries the reason into the gallery's error banner, which is the
            # only place the operator would otherwise have seen "done".
            raise HTTPException(status_code=409, detail=outcome["message"])
        return outcome
    except ProtectedArtifactError as e:
        # ``?delete_file=true`` on a reserved artifact: the same 403 the
        # untracked-delete route gives, for the same reason — the file exists
        # and the operator may even own it, but removing it is another
        # channel's call.
        raise _refuse_protected(request, "unoverride", name, e) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/api/scaffold/{name:path}")
async def get_scaffold(name: str, request: Request):
    """Get artifact content (auto-resolves framework vs override)."""
    service = _scaffold_service(request)
    try:
        result = service.get_content(name)
        result["name"] = name
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
