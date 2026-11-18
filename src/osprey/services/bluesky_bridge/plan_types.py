"""The facility plan-injection contract's shared type: ``PlanSpec``.

Kept free of bluesky/ophyd/tiled imports (pydantic is a core bridge dependency,
pulled in transitively via FastAPI, so it's fine here) so both sides of the
injection seam stay on the right side of the import-clean boundary:

- ``plan_loader.py`` loads a facility module exposing
  ``PLANS: dict[str, PlanSpec]`` from a config-pointed path *without* itself
  importing bluesky — only the loaded module needs it.
- the shipped plan files under ``plans_core/`` (e.g. ``orm.py``, ``grid_scan.py``)
  build their ``PlanSpec`` by wrapping ``bluesky.plans`` callables; they import
  bluesky, but doing so here in the shared type would force that import onto the
  loader too.

A plan's ``plan`` callable is intentionally opaque: ``(devices, params) ->
Any``, where ``devices`` is whatever ``get_devices()`` returned and ``params``
is a validated instance of ``schema``. Neither this module nor callers need to
know if the callable returns a real bluesky plan generator or, in a test
double, something else entirely.

A plan's optional ``render`` callable is the opposite — fully typed, because
it is the seam the figure route serves to the panel and the agent. ``figure.py``
is pydantic-only, so importing `Figure` here costs the loader nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, NoDefault
from pydantic_core import core_schema

from .figure import Figure, RowWindow
from .plan_fields import ChannelRole
from .plan_metadata import PlanMetadata

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

Provenance = Literal["shipped", "preset", "facility", "session", "unreviewed"]
"""Trust/origin tier, in ascending ephemerality order (``shipped`` is the trust
floor; ``session``/``unreviewed`` is agent-authored and least trusted).
Assigned by the loader based on which layer a plan file came from — never
self-declared in a plan's own ``PLAN_METADATA``.
"""


def _schema_generator(plan_name: str) -> type[GenerateJsonSchema]:
    """A JSON-schema generator that states a factory default like a written one.

    pydantic writes ``default`` only for a written default; a field whose
    default comes from a zero-argument ``default_factory`` gets the factory's
    value here, at every nesting level, evaluated each time the schema is
    generated. A factory that takes the validated data has no value of its own
    and is left to the base generator. A factory that raises yields no default
    and one warning naming *plan_name* and the field.
    """

    class _FactoryDefaults(GenerateJsonSchema):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._field_names: dict[int, str] = {}

        def model_fields_schema(self, schema: core_schema.ModelFieldsSchema) -> Any:
            for name, field in schema["fields"].items():
                self._field_names[id(field["schema"])] = name
            return super().model_fields_schema(schema)

        def get_default_value(self, schema: core_schema.WithDefaultSchema) -> Any:
            factory = schema.get("default_factory")
            if factory is None or schema.get("default_factory_takes_data"):
                return super().get_default_value(schema)
            try:
                return factory()  # type: ignore[call-arg]
            except Exception as exc:
                name = self._field_names.get(id(schema))
                logger.warning(
                    "plan %r: the default factory of %s raised, so the catalog "
                    "states no default for it: %s",
                    plan_name,
                    f"field {name!r}" if name else "a field",
                    exc,
                )
                return NoDefault

    return _FactoryDefaults


@dataclass
class PlanSpec(Generic[SchemaT]):
    """One registered plan: its name, parameter schema, and implementation.

    Generic over its own ``schema`` type so each concrete plan's ``plan``
    callable can be typed against its own pydantic model (e.g. ``CountParams``)
    rather than the common ``BaseModel`` supertype — callers that don't care
    about a specific plan's schema can still hold these as ``PlanSpec[Any]``.

    ``render`` is the plan's own view of a run: ``(window, params) ->
    Figure``, where ``window`` is a `RowWindow` — the run's data rows as plain
    dicts *plus* how much of the run they are — and ``params`` is a validated
    ``schema`` instance — the *same* ``SchemaT``, so a plan whose ``render``
    takes anything other than its own PARAMS is a type error rather than a
    runtime surprise at the first poll tick. The window rather than a bare
    row list because a render that reads rows positionally cannot tell a whole
    run from a truncated one by looking at the rows, and guessing (comparing
    the row count against a duplicated copy of the buffer's cap) is a guess
    that goes stale silently. ``None`` (the default) means the plan has no view
    of its own and the bridge's default figure stands in for it. The loader
    only ever populates this for operator-supplied tiers; see
    ``plan_loader._resolve_render``.

    ``roles`` is what the plan's ``schema`` declared about its channel fields:
    ``(field_path, role)`` pairs exactly as ``plan_fields.channel_roles``
    returns them — depth-first in declaration order, paths spelled
    ``"correctors"`` / ``"axes[].setpoint"``. The loader introspects the schema
    once at load time and stores the result here, so request-time consumers
    (the enqueue pre-check, the pre-flight preview, the default figure,
    analysis) read the declaration off the spec instead of re-walking the model
    on every call. An empty tuple (the default, so a hand-built ``PlanSpec``
    stays valid) means the schema declares no channel roles at all.
    """

    name: str
    plan: Callable[[dict[str, Any], SchemaT], Any]
    schema: type[SchemaT]
    description: str = ""
    metadata: PlanMetadata | None = None
    provenance: Provenance = "shipped"
    render: Callable[[RowWindow, SchemaT], Figure] | None = None
    roles: tuple[tuple[str, ChannelRole], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for `GET /plans`: name, description, schema, metadata, provenance.

        The served schema states every default the plan has: a written one, and
        the value a zero-argument ``default_factory`` makes when the catalog is
        read.
        """
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.schema.model_json_schema(schema_generator=_schema_generator(self.name)),
            "metadata": self.metadata.model_dump() if self.metadata is not None else None,
            "provenance": self.provenance,
        }
