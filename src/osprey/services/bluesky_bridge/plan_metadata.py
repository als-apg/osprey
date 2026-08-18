"""Authoring metadata for plans: the ``PlanMetadata`` model and its parser.

A plan module (built-in, preset, or facility-supplied) declares a module-level
``PLAN_METADATA`` dict describing itself to operators and to the agent's
discovery surface (`GET /plans`). This module defines the required shape of
that dict and a fail-closed parser: a malformed, incomplete, or over-declared
block is a typed `PlanMetadataError` naming the offending field(s) or unknown
key(s), never a silently default-filled or silently trimmed object. Provenance (trust tier) is deliberately *not* part of
this model -- it is assigned by the loader based on which layer a file came
from, not self-declared by the plan author.

Kept pydantic-only (no bluesky/ophyd/tiled imports) so it can be imported from
``plan_types.py`` without breaking that module's bluesky-free boundary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class PlanMetadataError(ValueError):
    """Raised when a plan module's ``PLAN_METADATA`` is missing, malformed, or invalid.

    A single exception type for every failure mode (absent attribute, wrong
    container type, missing/mistyped field, unknown key) so callers only need
    to catch one thing; the message names the offending field(s) or key(s) and
    the module/file the metadata came from.
    """


class PlanMetadata(BaseModel):
    """Authoring-declared metadata for one plan, surfaced via `GET /plans`.

    Exactly three fields, all required: a plan lacking one is an authoring
    error to be rejected at load time, not a gap to paper over with a default.
    Unknown keys are rejected too (``extra="forbid"``) -- a plan that declares
    anything outside this contract, including keys from an earlier shape, is
    quarantined loudly rather than parsed clean with the surplus dropped.
    Which channels a plan touches is *not* declared here: it is read off the
    role-typed fields of the plan's parameter model. JSON-serializable via
    `model_dump()` for direct inclusion in `PlanSpec.to_dict()`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    writes: bool
    """Whether the plan moves anything, as declared by its author.

    True means the plan changes machine state through its movable fields;
    false means it only reads. The load gate cross-checks this declaration
    against the plan's role-typed fields, so a mismatch is caught at load
    time rather than at execution.
    """


def parse_plan_metadata_dict(raw: dict[str, Any], *, source: str) -> PlanMetadata:
    """Validate ``raw`` (a plan module's ``PLAN_METADATA`` dict) into a `PlanMetadata`.

    Wraps pydantic's `ValidationError` into `PlanMetadataError`, naming the
    failing field(s) and ``source`` (a path/module label for operator
    debugging), so every caller sees one exception type regardless of whether
    a field is missing, wrong-typed, or outside the contract. Unknown keys are
    reported in their own clause, so a plan carrying a key this model no longer
    accepts says exactly which key to delete.
    """
    try:
        return PlanMetadata.model_validate(raw)
    except ValidationError as exc:
        invalid: list[str] = []
        unknown: list[str] = []
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            bucket = unknown if error["type"] == "extra_forbidden" else invalid
            bucket.append(field)
        message = f"{source}: PLAN_METADATA is invalid"
        if invalid:
            message += f" for field(s): {', '.join(invalid)}"
        if unknown:
            message += f"; unknown key(s): {', '.join(unknown)}"
        raise PlanMetadataError(message) from exc


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
