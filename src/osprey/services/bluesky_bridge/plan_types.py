"""The facility plan-injection contract's shared type: ``PlanSpec``.

Kept free of bluesky/ophyd/tiled imports (pydantic is a core bridge dependency,
pulled in transitively via FastAPI, so it's fine here) so both sides of the
injection seam stay on the right side of the import-clean boundary:

- ``plan_loader.py`` (task 2.4) loads a facility module exposing
  ``PLANS: dict[str, PlanSpec]`` from a config-pointed path *without* itself
  importing bluesky — only the loaded module needs it.
- ``plans.py`` (this task, 2.3) builds the v1 built-in ``PlanSpec`` set by
  wrapping ``bluesky.plans`` callables; it imports bluesky, but doing so here
  in the shared type would force that import onto the loader too.

A plan's ``plan`` callable is intentionally opaque: ``(devices, params) ->
Any``, where ``devices`` is whatever ``get_devices()`` returned and ``params``
is a validated instance of ``schema``. Neither this module nor callers need to
know if the callable returns a real bluesky plan generator or, in a test
double, something else entirely.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

from .plan_metadata import PlanMetadata

SchemaT = TypeVar("SchemaT", bound=BaseModel)

Provenance = Literal["shipped", "preset", "facility", "session", "unreviewed"]
"""Trust/origin tier, in ascending ephemerality order (``shipped`` is the trust
floor; ``session``/``unreviewed`` is agent-authored and least trusted).
Assigned by the loader based on which layer a plan file came from — never
self-declared in a plan's own ``PLAN_METADATA``.
"""


@dataclass
class PlanSpec(Generic[SchemaT]):
    """One registered plan: its name, parameter schema, and implementation.

    Generic over its own ``schema`` type so each concrete plan's ``plan``
    callable can be typed against its own pydantic model (e.g. ``CountParams``)
    rather than the common ``BaseModel`` supertype — callers that don't care
    about a specific plan's schema can still hold these as ``PlanSpec[Any]``.
    """

    name: str
    plan: Callable[[dict[str, Any], SchemaT], Any]
    schema: type[SchemaT]
    description: str = ""
    metadata: PlanMetadata | None = None
    provenance: Provenance = "shipped"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for `GET /plans`: name, description, schema, metadata, provenance."""
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.schema.model_json_schema(),
            "metadata": self.metadata.model_dump() if self.metadata is not None else None,
            "provenance": self.provenance,
        }
