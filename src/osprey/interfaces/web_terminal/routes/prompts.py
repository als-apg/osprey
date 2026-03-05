"""Prompt gallery routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.interfaces.web_terminal.prompt_gallery_service import PromptGalleryService

router = APIRouter()


class PromptOverrideRequest(BaseModel):
    content: str


def _prompt_service(request: Request) -> PromptGalleryService:
    """Construct a PromptGalleryService from the request's project dir."""
    return PromptGalleryService(Path(request.app.state.project_cwd))


@router.get("/api/prompts")
async def list_prompts(request: Request):
    """List all prompt artifacts with status and summary counts."""
    service = _prompt_service(request)
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


@router.get("/api/prompts/untracked")
async def list_untracked_prompts(request: Request):
    """Detect files active in Claude Code but not managed by OSPREY."""
    service = _prompt_service(request)
    untracked = service.scan_untracked()
    return {"untracked": untracked, "count": len(untracked)}


@router.post("/api/prompts/untracked/register")
async def register_untracked_prompt(body: UntrackedRegisterRequest, request: Request):
    """Register an untracked file by adding it to config.yml."""
    service = _prompt_service(request)
    try:
        return service.register_untracked(body.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.delete("/api/prompts/untracked/{name:path}")
async def delete_untracked_prompt(name: str, request: Request):
    """Delete an untracked file from disk."""
    service = _prompt_service(request)
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


@router.post("/api/prompts/create")
async def create_artifact(body: CreateArtifactRequest, request: Request):
    """Create a new custom artifact."""
    service = _prompt_service(request)
    try:
        return service.create_artifact(body.category, body.name, body.content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.get("/api/prompts/{name:path}/framework")
async def get_prompt_framework(name: str, request: Request):
    """Get the framework-rendered content for an artifact."""
    service = _prompt_service(request)
    try:
        content = service.get_framework_content(name)
        return {"name": name, "content": content, "source": "framework"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/api/prompts/{name:path}/diff")
async def get_prompt_diff(name: str, request: Request):
    """Get unified diff between framework and override."""
    service = _prompt_service(request)
    try:
        return service.compute_diff(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/api/prompts/{name:path}/scaffold")
async def scaffold_prompt(name: str, request: Request):
    """Scaffold an override from the framework template."""
    service = _prompt_service(request)
    try:
        return service.scaffold_override(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.put("/api/prompts/{name:path}/override")
async def save_prompt_override(name: str, body: PromptOverrideRequest, request: Request):
    """Save content to an existing override file."""
    service = _prompt_service(request)
    try:
        return service.save_override(name, body.content)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/api/prompts/{name:path}/override")
async def delete_prompt_override(name: str, request: Request):
    """Remove an override, restoring framework management."""
    delete_file = request.query_params.get("delete_file", "false").lower() == "true"
    service = _prompt_service(request)
    try:
        return service.unoverride(name, delete_file=delete_file)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/api/prompts/{name:path}")
async def get_prompt(name: str, request: Request):
    """Get artifact content (auto-resolves framework vs override)."""
    service = _prompt_service(request)
    try:
        result = service.get_content(name)
        result["name"] = name
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
