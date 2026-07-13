"""Authoring metadata for scan plans: the ``PlanMetadata`` model and its parser.

A plan module (built-in, preset, or facility-supplied) declares a module-level
``PLAN_METADATA`` dict describing itself to operators and to the agent's
discovery surface (`GET /plans`). This module defines the required shape of
that dict and a fail-closed parser: a malformed or incomplete block is a typed
`PlanMetadataError` naming the offending field(s), never a silently
default-filled object. Provenance (trust tier) is deliberately *not* part of
this model -- it is assigned by the loader based on which layer a file came
from, not self-declared by the plan author.

Kept pydantic-only (no bluesky/ophyd/tiled imports) so it can be imported from
``plan_types.py`` without breaking that module's bluesky-free boundary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError


class PlanMetadataError(ValueError):
    """Raised when a plan module's ``PLAN_METADATA`` is missing, malformed, or invalid.

    A single exception type for every failure mode (absent attribute, wrong
    container type, missing/mistyped field) so callers only need to catch one
    thing; the message names the offending field(s) and the module/file the
    metadata came from.
    """


class PlanMetadata(BaseModel):
    """Authoring-declared metadata for one scan plan, surfaced via `GET /plans`.

    Every field is required: a plan lacking one is an authoring error to be
    rejected at load time, not a gap to paper over with a default. JSON-
    serializable via `model_dump()` for direct inclusion in `PlanSpec.to_dict()`.
    """

    name: str
    description: str
    category: str
    required_devices: list[str]
    writes: bool


def parse_plan_metadata_dict(raw: dict[str, Any], *, source: str) -> PlanMetadata:
    """Validate ``raw`` (a plan module's ``PLAN_METADATA`` dict) into a `PlanMetadata`.

    Wraps pydantic's `ValidationError` into `PlanMetadataError`, naming the
    failing field(s) and ``source`` (a path/module label for operator
    debugging), so every caller sees one exception type regardless of whether
    a field is missing or wrong-typed.
    """
    try:
        return PlanMetadata.model_validate(raw)
    except ValidationError as exc:
        fields = ", ".join(".".join(str(part) for part in error["loc"]) for error in exc.errors())
        raise PlanMetadataError(
            f"{source}: PLAN_METADATA is invalid for field(s): {fields}"
        ) from exc


def parse_plan_metadata(module: Any, *, source: str | None = None) -> PlanMetadata:
    """Read and validate ``PLAN_METADATA`` from an already-imported plan ``module``.

    Raises `PlanMetadataError` before ever reaching pydantic validation if the
    module has no ``PLAN_METADATA`` attribute or that attribute isn't a dict.
    ``source`` defaults to the module's ``__name__`` for the error message;
    pass it explicitly when the module was imported under a synthetic name
    (e.g. a path-hash `sys.modules` key) that wouldn't mean anything to an
    operator.
    """
    label = source if source is not None else getattr(module, "__name__", repr(module))
    raw = getattr(module, "PLAN_METADATA", None)
    if raw is None:
        raise PlanMetadataError(f"{label}: module has no PLAN_METADATA attribute")
    if not isinstance(raw, dict):
        raise PlanMetadataError(f"{label}: PLAN_METADATA must be a dict, got {type(raw).__name__}")
    return parse_plan_metadata_dict(raw, source=label)
