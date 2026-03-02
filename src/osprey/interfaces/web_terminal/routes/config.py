"""Configuration and Claude setup routes."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.interfaces.web_terminal.claude_code_files import ClaudeCodeFileService

logger = logging.getLogger(__name__)

router = APIRouter()

# Config sections relevant to the agent (exclude internal/infra sections)
_AGENT_CONFIG_SECTIONS = [
    "control_system",
    "approval",
    "channel_finder",
    "ariel",
    "python_execution",
    "artifact_server",
    "workspace",
    "screen_capture",
]


@router.get("/api/config")
async def get_config(request: Request):
    """Return agent-relevant config sections as structured JSON + raw YAML."""
    config_path: Path | None = request.app.state.config_path
    if not config_path or not config_path.exists():
        raise HTTPException(status_code=404, detail="No config.yml found")

    raw = config_path.read_text(encoding="utf-8")
    try:
        full_config = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Invalid YAML: {e}") from e

    # Extract only agent-relevant sections
    sections = {}
    for key in _AGENT_CONFIG_SECTIONS:
        if key in full_config:
            sections[key] = full_config[key]

    return {
        "sections": sections,
        "raw": raw,
        "path": str(config_path),
    }


class ConfigUpdate(BaseModel):
    raw: str


@router.put("/api/config")
async def put_config(body: ConfigUpdate, request: Request):
    """Validate YAML, back up config.yml, and write updated config."""
    config_path: Path | None = request.app.state.config_path
    if not config_path:
        raise HTTPException(status_code=404, detail="No config.yml found")

    # Validate YAML parses cleanly
    try:
        yaml.safe_load(body.raw)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {e}") from e

    # Backup existing config
    if config_path.exists():
        backup_path = config_path.with_suffix(".yml.bak")
        shutil.copy2(config_path, backup_path)
        logger.info("Config backed up to %s", backup_path)

    # Write new config (fsync to ensure data is on disk before restart)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(body.raw)
        f.flush()
        os.fsync(f.fileno())
    logger.info("Config updated at %s", config_path)

    return {"status": "ok", "requires_restart": True}


class ConfigPatch(BaseModel):
    updates: dict[str, object]


@router.patch("/api/config")
async def patch_config(body: ConfigPatch, request: Request):
    """Apply structured field updates to config.yml, preserving comments.

    Accepts dot-notation keys (e.g. ``"control_system.writes_enabled": true``).
    Uses ruamel.yaml round-trip mode so comments, ordering, and formatting
    in the YAML file are retained.
    """
    from osprey.utils.yaml_config import config_update_fields

    config_path: Path | None = request.app.state.config_path
    if not config_path or not config_path.exists():
        raise HTTPException(status_code=404, detail="No config.yml found")

    if not body.updates:
        raise HTTPException(status_code=422, detail="No updates provided")

    # Backup before mutation
    backup_path = config_path.with_suffix(".yml.bak")
    shutil.copy2(config_path, backup_path)

    try:
        config_update_fields(config_path, body.updates)
    except Exception as e:
        # Restore backup on failure
        shutil.copy2(backup_path, config_path)
        logger.error("Config patch failed, restored backup: %s", e)
        raise HTTPException(status_code=500, detail=f"Config update failed: {e}") from e

    logger.info("Config patched (%d fields) at %s", len(body.updates), config_path)
    return {"status": "ok", "requires_restart": True, "fields_updated": len(body.updates)}


# ---- Claude Setup Endpoints ---- #


class ClaudeSetupSaveRequest(BaseModel):
    path: str
    content: str


@router.get("/api/claude-setup")
async def get_claude_setup(request: Request):
    """Read all Claude Code integration files from the project directory."""
    service = ClaudeCodeFileService(Path(request.app.state.project_cwd))
    return {"files": service.list_files()}


@router.put("/api/claude-setup")
async def save_claude_setup(request: Request, body: ClaudeSetupSaveRequest):
    """Save an existing Claude Code file."""
    service = ClaudeCodeFileService(Path(request.app.state.project_cwd))
    try:
        return service.write_file(body.path, body.content)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/api/claude-setup")
async def create_claude_setup(request: Request, body: ClaudeSetupSaveRequest):
    """Create a new Claude Code file in an allowed .claude/ subdirectory."""
    service = ClaudeCodeFileService(Path(request.app.state.project_cwd))
    try:
        return service.create_file(body.path, body.content)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
