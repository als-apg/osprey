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
    "claude_code",
    "channel_finder",
    "ariel",
    "python_execution",
    "artifact_server",
    "workspace",
    "screen_capture",
    "hooks",
]


def _regen_if_drift(request: Request) -> list[str]:
    """Re-render Claude Code artifacts if config.yml drifted from them.

    Called after a successful config write so safety-critical fields (e.g. the
    writes_enabled kill-switch baked into settings.json's permissions.deny) take
    effect on the next terminal restart, which respawns the agent and re-reads
    the on-disk ``.claude/`` artifacts. Fails open: a regen error must never undo
    a config write that already succeeded. ``regen_if_drift`` no-ops when the
    project has no rendered ``.claude/`` to re-sync.
    """
    project_cwd = getattr(request.app.state, "project_cwd", None)
    config_path: Path | None = request.app.state.config_path
    project_dir = (
        Path(project_cwd) if project_cwd else (config_path.parent if config_path else None)
    )
    if project_dir is None:
        return []
    try:
        from osprey.cli.templates.manager import TemplateManager

        return TemplateManager().regen_if_drift(project_dir)
    except Exception:  # noqa: BLE001 — config write already succeeded; never raise here
        logger.warning("Claude Code artifact regen after config write failed", exc_info=True)
        return []


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

    regenerated = _regen_if_drift(request)
    return {"status": "ok", "requires_restart": True, "regenerated": regenerated}


class ConfigPatch(BaseModel):
    updates: dict[str, object]


@router.patch("/api/config")
async def patch_config(body: ConfigPatch, request: Request):
    """Apply structured field updates to config.yml, preserving comments.

    Accepts dot-notation keys (e.g. ``"control_system.writes_enabled": true``).
    Uses ruamel.yaml round-trip mode so comments, ordering, and formatting
    in the YAML file are retained.
    """
    from osprey.utils.config_writer import config_update_fields

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
    regenerated = _regen_if_drift(request)
    return {
        "status": "ok",
        "requires_restart": True,
        "fields_updated": len(body.updates),
        "regenerated": regenerated,
    }


# ---- Hook Debug Endpoints ---- #


@router.get("/api/hooks/debug-status")
async def get_hook_debug_status(request: Request):
    """Return current hooks.debug state from config.yml."""
    config_path: Path | None = request.app.state.config_path
    if not config_path or not config_path.exists():
        return {"enabled": False}

    try:
        full_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        enabled = bool(full_config.get("hooks", {}).get("debug", False))
        return {"enabled": enabled}
    except Exception:
        return {"enabled": False}


@router.get("/api/hooks/debug-log")
async def get_hook_debug_log(request: Request, limit: int = 50):
    """Return recent hook debug log entries from hook_debug.jsonl."""
    import json

    project_cwd = getattr(request.app.state, "project_cwd", None)
    if not project_cwd:
        return {"entries": []}

    log_path = Path(project_cwd) / ".claude" / "hooks" / "hook_debug.jsonl"
    if not log_path.exists():
        return {"entries": []}

    try:
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        # Take the last N entries (most recent)
        recent = lines[-limit:] if len(lines) > limit else lines
        entries = []
        for line in reversed(recent):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {"entries": entries}
    except Exception:
        logger.warning("Failed to read hook debug log", exc_info=True)
        return {"entries": []}


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
