"""Channel Finder Database REST API.

Exposes database operations via REST endpoints, adapting for each pipeline
type (hierarchical, middle_layer, in_context). Calls database instances
directly via app.state, avoiding MCP server dependencies.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pipeline_type(request: Request) -> str:
    """Get the active pipeline type from app state."""
    return getattr(request.app.state, "pipeline_type", "in_context")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    """Request body for channel validation."""

    channels: list[str]


class AddNodeRequest(BaseModel):
    """Request body for adding a hierarchical node."""

    level: str
    parent_selections: dict[str, str] = {}
    name: str
    description: str = ""


class EditNodeRequest(BaseModel):
    """Request body for editing a hierarchical node (name and/or description)."""

    level: str
    selections: dict[str, str] = {}
    old_name: str
    new_name: str | None = None
    description: str | None = None


class DeleteNodeRequest(BaseModel):
    """Request body for deleting a hierarchical node."""

    level: str
    selections: dict[str, str] = {}
    name: str


class EditExpansionRequest(BaseModel):
    """Request body for editing an instance-level expansion config."""

    level: str
    selections: dict[str, str] = {}
    pattern: str | None = None
    range_start: int | None = None
    range_end: int | None = None


class AddFamilyRequest(BaseModel):
    """Request body for adding a middle-layer family."""

    system: str
    family: str
    description: str = ""


class DeleteFamilyRequest(BaseModel):
    """Request body for deleting a middle-layer family."""

    system: str
    family: str


class AddMLChannelRequest(BaseModel):
    """Request body for adding a middle-layer channel."""

    system: str
    family: str
    field: str
    channel_name: str
    subfield: str | None = None


class DeleteMLChannelRequest(BaseModel):
    """Request body for deleting a middle-layer channel."""

    system: str
    family: str
    field: str
    channel_name: str
    subfield: str | None = None


class AddICChannelRequest(BaseModel):
    """Request body for adding an in-context channel."""

    channel_name: str
    address: str = ""
    description: str = ""


class UpdateICChannelRequest(BaseModel):
    """Request body for updating an in-context channel."""

    description: str | None = None
    address: str | None = None


# ---------------------------------------------------------------------------
# Common endpoints (all pipelines)
# ---------------------------------------------------------------------------


@router.get("/info")
async def get_info(request: Request):
    """Return pipeline type and pipeline-specific metadata."""
    pt = _pipeline_type(request)
    available = getattr(request.app.state, "available_pipelines", [pt])
    info: dict = {"pipeline_type": pt, "available_pipelines": available}

    try:
        info["db_path"] = _get_db_path(request)
    except Exception:
        info["db_path"] = None

    try:
        db = _get_database(request)
        if pt == "hierarchical":
            info["metadata"] = {
                "hierarchy_levels": db.hierarchy_levels,
                "hierarchy_config": db.hierarchy_config,
                "naming_pattern": db.naming_pattern,
                "facility_name": _get_facility_name(request),
            }
        elif pt == "middle_layer":
            systems = db.list_systems()
            info["metadata"] = {"system_count": len(systems)}
        else:  # in_context
            stats = db.get_statistics()
            chunks = db.chunk_database(50)
            stats["total_chunks_at_50"] = len(chunks)
            stats["facility_name"] = _get_facility_name(request)
            info["metadata"] = stats

    except Exception as exc:
        logger.exception("Failed to get pipeline info")
        info["metadata"] = {"error": str(exc)}

    return info


class SwitchPipelineRequest(BaseModel):
    """Request body for switching the active pipeline type."""

    pipeline_type: str


@router.put("/pipeline")
async def switch_pipeline(request: Request, body: SwitchPipelineRequest):
    """Switch the active pipeline type at runtime (dev mode).

    Only allows switching to pipelines that were successfully initialized.
    """
    available = getattr(request.app.state, "available_pipelines", [])
    if body.pipeline_type not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline '{body.pipeline_type}' not available. Available: {available}",
        )
    request.app.state.pipeline_type = body.pipeline_type
    logger.info("Switched active pipeline to %s", body.pipeline_type)
    return {"pipeline_type": body.pipeline_type}


@router.get("/statistics")
async def get_statistics(request: Request):
    """Return database statistics for the active pipeline."""
    pt = _pipeline_type(request)

    try:
        db = _get_database(request)
        if pt == "in_context":
            stats = db.get_statistics()
            chunks = db.chunk_database(50)
            stats["total_chunks_at_50"] = len(chunks)
            stats["facility_name"] = _get_facility_name(request)
            return stats
        else:
            return db.get_statistics()

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get statistics")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/validate")
async def validate_channels(request: Request, body: ValidateRequest):
    """Validate channel names against the database."""
    pt = _pipeline_type(request)

    try:
        db = _get_database(request)
        if pt == "in_context":
            validation_results = db.validate_channels(body.channels)
            valid = db.get_valid_channels(validation_results)
            invalid = db.get_invalid_channels(validation_results)
            return {
                "total": len(body.channels),
                "valid_count": len(valid),
                "invalid_count": len(invalid),
                "valid_channels": valid,
                "invalid_channels": invalid,
                "results": validation_results,
            }
        else:  # hierarchical or middle_layer
            results = []
            valid_count = 0
            for ch in body.channels:
                is_valid = db.validate_channel(ch)
                results.append({"channel": ch, "valid": is_valid})
                valid_count += is_valid
            return {
                "results": results,
                "valid_count": valid_count,
                "invalid_count": len(body.channels) - valid_count,
                "total": len(body.channels),
            }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to validate channels")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Hierarchical pipeline endpoints
# ---------------------------------------------------------------------------


@router.get("/explore/options")
async def explore_options(request: Request, level: str, selections: str | None = None):
    """Get available options at a hierarchy level.

    Args:
        request: FastAPI request.
        level: Hierarchy level name (e.g., "system", "device").
        selections: JSON-encoded dict of previous selections.
    """
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        parsed_selections = json.loads(selections) if selections else None
        options = db.get_options_at_level(level, parsed_selections or {})
        return {"level": level, "options": options, "total": len(options)}

    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid selections JSON: {exc}") from exc
    except Exception as exc:
        logger.exception("Failed to get explore options")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/build")
async def explore_build(request: Request, selections: str):
    """Build channel addresses from hierarchy selections.

    Args:
        request: FastAPI request.
        selections: JSON-encoded dict of hierarchy selections.
    """
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        parsed_selections = json.loads(selections)
        channels = db.build_channels_from_selections(parsed_selections)
        valid = [ch for ch in channels if db.validate_channel(ch)]
        invalid = [ch for ch in channels if not db.validate_channel(ch)]
        return {
            "channels": channels,
            "total": len(channels),
            "valid": valid,
            "invalid": invalid,
            "valid_count": len(valid),
            "invalid_count": len(invalid),
        }

    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid selections JSON: {exc}") from exc
    except Exception as exc:
        logger.exception("Failed to build channels")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/hierarchy-info")
async def explore_hierarchy_info(request: Request):
    """Get hierarchy structure information."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        return {
            "hierarchy_levels": db.hierarchy_levels,
            "hierarchy_config": db.hierarchy_config,
            "naming_pattern": db.naming_pattern,
            "facility_name": _get_facility_name(request),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get hierarchy info")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Middle layer pipeline endpoints
# ---------------------------------------------------------------------------


@router.get("/explore/systems")
async def explore_systems(request: Request):
    """List all systems in the channel database."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        systems = db.list_systems()
        return {"systems": systems, "total": len(systems)}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list systems")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/families")
async def explore_families(request: Request, system: str):
    """List device families in a system.

    Args:
        request: FastAPI request.
        system: System name (e.g., "SR" for Storage Ring).
    """
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        families = db.list_families(system)
        return {"families": families, "total": len(families)}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list families")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/fields")
async def explore_fields(
    request: Request,
    system: str,
    family: str,
    field: str | None = None,
):
    """Inspect fields of a device family.

    Args:
        request: FastAPI request.
        system: System name.
        family: Family name.
        field: Optional specific field to inspect.
    """
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        fields = db.inspect_fields(system, family, field)
        return {"fields": fields}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to inspect fields")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/channels")
async def explore_channels(
    request: Request,
    system: str,
    family: str,
    field: str,
    subfield: str | None = None,
    sectors: str | None = None,
    devices: str | None = None,
):
    """Get channel names for a system/family/field path.

    Args:
        request: FastAPI request.
        system: System name.
        family: Family name.
        field: Field name (e.g., "Monitor", "Setpoint").
        subfield: Optional subfield name.
        sectors: Optional JSON-encoded list of sector numbers.
        devices: Optional JSON-encoded list of device numbers.
    """
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        parsed_sectors = json.loads(sectors) if sectors else None
        parsed_devices = json.loads(devices) if devices else None
        channels = db.list_channel_names(
            system, family, field, subfield, parsed_sectors, parsed_devices
        )
        return {"channels": channels, "total": len(channels)}

    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid JSON in sectors or devices parameter: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to list channels")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/explore/device-info")
async def explore_device_info(request: Request, system: str, family: str):
    """Get device arrangement info for a middle-layer family."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")
    try:
        db = _get_database(request)
        return db.get_device_info(system, family)
    except Exception as exc:
        logger.exception("Failed to get device info")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# In-context pipeline endpoints
# ---------------------------------------------------------------------------


@router.get("/channels")
async def get_channels(
    request: Request,
    chunk_idx: int | None = None,
    chunk_size: int = 50,
):
    """Get channels from the in-context database.

    Args:
        request: FastAPI request.
        chunk_idx: Optional chunk index (0-based). If omitted, returns all.
        chunk_size: Number of channels per chunk (default 50).
    """
    if _pipeline_type(request) != "in_context":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    try:
        db = _get_database(request)
        if chunk_idx is not None:
            chunks = db.chunk_database(chunk_size)
            if chunk_idx < 0 or chunk_idx >= len(chunks):
                raise HTTPException(
                    status_code=422,
                    detail=f"chunk_idx {chunk_idx} out of range (0-{len(chunks) - 1})",
                )
            chunk = chunks[chunk_idx]
            formatted = db.format_chunk_for_prompt(chunk)
            return {
                "chunk_idx": chunk_idx,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "channels": chunk,
                "formatted": formatted,
            }
        else:
            channels = db.get_all_channels()
            return {"channels": channels, "total": len(channels)}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get channels")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def _get_database(request: Request):
    """Get the database instance for the active pipeline type."""
    pt = _pipeline_type(request)
    databases = getattr(request.app.state, "databases", {})
    db = databases.get(pt)
    if db is None:
        raise HTTPException(status_code=503, detail=f"Database not available for pipeline '{pt}'")
    return db


def _get_db_path(request: Request) -> str:
    """Get the database file path for the active pipeline type."""
    return _get_database(request).db_path


def _get_facility_name(request: Request) -> str:
    """Get the facility name for the active pipeline type."""
    pt = _pipeline_type(request)
    facility_names = getattr(request.app.state, "facility_names", {})
    return facility_names.get(pt, "")


# ---------------------------------------------------------------------------
# Hierarchical CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/tree/node")
async def add_tree_node(request: Request, body: AddNodeRequest):
    """Add a new node at a hierarchy level."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.add_node(
            level=body.level,
            parent_selections=body.parent_selections,
            name=body.name,
            description=body.description,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to add tree node")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/tree/node")
async def edit_tree_node(request: Request, body: EditNodeRequest):
    """Edit a node's name and/or description at a hierarchy level."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.edit_node(
            level=body.level,
            selections=body.selections,
            old_name=body.old_name,
            new_name=body.new_name,
            description=body.description,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to edit tree node")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/tree/node")
async def delete_tree_node(request: Request, body: DeleteNodeRequest):
    """Delete a node (and all descendants) at a hierarchy level."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.delete_node(
            level=body.level,
            selections=body.selections,
            name=body.name,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete tree node")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/tree/impact")
async def tree_impact(request: Request, body: DeleteNodeRequest):
    """Preview the impact of deleting a hierarchy node."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        impact = db.count_descendants(
            level=body.level,
            selections=body.selections,
            name=body.name,
        )
        return {
            "affected_channels": impact.get("channels", 0),
            "breakdown": {k: v for k, v in impact.items() if k != "channels"},
        }
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to compute tree impact")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/tree/expansion")
async def get_tree_expansion(request: Request, level: str, selections: str | None = None):
    """Get the current expansion config for an instance-type level."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        parsed_selections = json.loads(selections) if selections else {}
        db = _get_database(request)
        return db.get_expansion(
            level=level,
            selections=parsed_selections,
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid selections JSON: {exc}") from exc
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to get expansion config")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/tree/expansion")
async def edit_tree_expansion(request: Request, body: EditExpansionRequest):
    """Edit the expansion config for an instance-type level."""
    if _pipeline_type(request) != "hierarchical":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.edit_expansion(
            level=body.level,
            selections=body.selections,
            pattern=body.pattern,
            range_start=body.range_start,
            range_end=body.range_end,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to edit expansion config")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Middle Layer CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/structure/family")
async def add_family(request: Request, body: AddFamilyRequest):
    """Add a new family to a system."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.add_family(
            system=body.system,
            family=body.family,
            description=body.description,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to add family")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/structure/family")
async def delete_family(request: Request, body: DeleteFamilyRequest):
    """Delete a family and all its channels."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.delete_family(
            system=body.system,
            family=body.family,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete family")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/structure/channel")
async def add_ml_channel(request: Request, body: AddMLChannelRequest):
    """Add a channel to a family's field."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.add_channel(
            system=body.system,
            family=body.family,
            field=body.field,
            channel_name=body.channel_name,
            subfield=body.subfield,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to add ML channel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/structure/channel")
async def delete_ml_channel(request: Request, body: DeleteMLChannelRequest):
    """Delete a channel from a family's field."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.delete_channel(
            system=body.system,
            family=body.family,
            field=body.field,
            channel_name=body.channel_name,
            subfield=body.subfield,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete ML channel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/structure/impact")
async def structure_impact(request: Request, body: DeleteFamilyRequest):
    """Preview the impact of deleting a middle-layer family."""
    if _pipeline_type(request) != "middle_layer":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        count = db.count_family_channels(
            system=body.system,
            family=body.family,
        )
        return {"affected_channels": count}
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to compute structure impact")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# In-Context CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/channels")
async def create_channel(request: Request, body: AddICChannelRequest):
    """Add a new channel to the in-context database."""
    if _pipeline_type(request) != "in_context":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.add_channel(
            channel=body.channel_name,
            address=body.address,
            description=body.description,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create channel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/channels/{channel_id:path}")
async def update_channel(channel_id: str, request: Request, body: UpdateICChannelRequest):
    """Update an in-context channel's description and/or address.

    Args:
        channel_id: Channel name (uses :path converter for colon-separated PV names).
        request: FastAPI request.
        body: Fields to update.
    """
    if _pipeline_type(request) != "in_context":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.update_channel(
            channel=channel_id,
            new_description=body.description,
            new_address=body.address,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to update channel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/channels/{channel_id:path}")
async def delete_channel(channel_id: str, request: Request):
    """Delete a channel from the in-context database.

    Args:
        channel_id: Channel name (uses :path converter for colon-separated PV names).
        request: FastAPI request.
    """
    if _pipeline_type(request) != "in_context":
        raise HTTPException(status_code=404, detail="Not available for this pipeline type")

    from osprey.services.channel_finder.core.base_database import DatabaseWriteError

    try:
        db = _get_database(request)
        return db.delete_channel(
            channel=channel_id,
        )
    except DatabaseWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete channel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
