"""Scaffold gallery routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.interfaces.web_terminal.scaffold_gallery_service import ScaffoldGalleryService

router = APIRouter()


class ScaffoldOverrideRequest(BaseModel):
    content: str


def _scaffold_service(request: Request) -> ScaffoldGalleryService:
    """Construct a ScaffoldGalleryService from the request's project dir."""
    return ScaffoldGalleryService(Path(request.app.state.project_cwd))


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
    service = _scaffold_service(request)
    try:
        return service.register_untracked(body.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.delete("/api/scaffold/untracked/{name:path}")
async def delete_untracked_scaffold(name: str, request: Request):
    """Delete an untracked file from disk."""
    service = _scaffold_service(request)
    try:
        return service.delete_untracked(name)
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
    service = _scaffold_service(request)
    try:
        return service.create_artifact(body.category, body.name, body.content)
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
    service = _scaffold_service(request)
    try:
        return service.scaffold_override(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.put("/api/scaffold/{name:path}/override")
async def save_scaffold_override(name: str, body: ScaffoldOverrideRequest, request: Request):
    """Save content to an existing override file."""
    service = _scaffold_service(request)
    try:
        return service.save_override(name, body.content)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/api/scaffold/{name:path}/override")
async def delete_scaffold_override(name: str, request: Request):
    """Remove an override, restoring framework management."""
    delete_file = request.query_params.get("delete_file", "false").lower() == "true"
    service = _scaffold_service(request)
    try:
        return service.unoverride(name, delete_file=delete_file)
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
