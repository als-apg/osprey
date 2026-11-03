"""The per-entry state of each enhancement module, in Python and in SQL.

One decision table, two forms that must agree:

* :func:`enhancement_view` — the state of every enabled module for one entry,
  read from its stored ``enhancement_status`` JSONB.
* :func:`enhancement_state_sql` — the ``failed`` and ``stale`` rows of the same
  table as a SQL predicate with positional parameters, for filters and counts.

The table, first matching row wins:

1. An image module and the entry has no viewable picture → ``n/a``.
2. No key → ``pending``.
3. A marker module whose current marker is not null and whose stored marker
   differs from it → ``stale``, whatever the status.
4. Status ``failed``, or ``gave_up``, or a text module with attempts ≥
   :data:`MAX_ENHANCEMENT_ATTEMPTS` → ``failed``.
5. Status ``complete`` → ``complete``.
6. Anything else → ``pending``.

Supporting pieces:

* :func:`has_images_sql` / :func:`has_images` — the one notion of "has a
  picture": a viewable ``attachment_files`` row (never the attachments JSONB).
* :func:`current_markers` / :func:`image_embedding_marker` — the completion
  markers the image modules are currently written under.
* :func:`enabled_modules` and :class:`EnhancementContext` — the enabled modules
  in registry execution order, and the context filters and counts take.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from osprey.imaging.formats import viewable_sql

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository

__all__ = [
    "EnhancementContext",
    "EnhancementState",
    "IMAGE_MODULES",
    "STATES",
    "SQL_STATES",
    "current_markers",
    "enabled_modules",
    "enhancement_context",
    "enhancement_state",
    "enhancement_state_sql",
    "enhancement_view",
    "has_images",
    "has_images_sql",
    "image_embedding_marker",
    "store_enhancement_context",
]

EnhancementState = Literal["n/a", "pending", "stale", "failed", "complete"]

#: Every state :func:`enhancement_view` can report.
STATES: tuple[str, ...] = ("n/a", "pending", "stale", "failed", "complete")

#: The states :func:`enhancement_state_sql` (and an enhancement filter) can select.
SQL_STATES: tuple[str, ...] = ("failed", "stale")

#: The modules whose input is the entry's pictures. They are the marker modules:
#: their status records the marker (caption model id, image table) it was written
#: under, and they are ``n/a`` on an entry with no viewable picture. Equal to
#: ``repository.IMAGE_STATUS_KEYS``, kept here so this module imports no repository.
IMAGE_MODULES: tuple[str, ...] = ("image_caption", "image_embedding")

#: The alias of ``attachment_files`` inside :func:`has_images_sql`.
_FILES_ALIAS = "f"


def _max_attempts() -> int:
    # Lazy: the repository will import has_images_sql from here.
    from osprey.services.ariel_search.database.repository import MAX_ENHANCEMENT_ATTEMPTS

    return MAX_ENHANCEMENT_ATTEMPTS


def _check_alias(alias: str) -> str:
    if not (isinstance(alias, str) and alias.isascii() and alias.isidentifier()):
        raise ValueError(f"not a SQL identifier: {alias!r}")
    return alias


# ---------------------------------------------------------------------------
# Markers and enabled modules
# ---------------------------------------------------------------------------


def image_embedding_marker(config: ARIELConfig) -> tuple[str | None, bool]:
    """The completion marker of an enabled ``image_embedding`` block.

    Returns:
        ``(table, False)`` when the block resolves, ``(None, True)`` when it is
        enabled but misconfigured, and ``(None, False)`` when it is off.
    """
    from osprey.services.ariel_search.database.migrations import image_embedding_target
    from osprey.services.ariel_search.exceptions import ModuleConfigError

    if not config.is_enhancement_module_enabled("image_embedding"):
        return None, False
    try:
        target = image_embedding_target(
            config.get_enhancement_module_config("image_embedding") or {}
        )
    except ModuleConfigError:
        return None, True
    return target.table, False


def current_markers(config: ARIELConfig) -> dict[str, str | None]:
    """The marker each image module is currently written under.

    ``image_caption`` is the caption model id (its ``enabled`` flag ignored, as
    :func:`~osprey.services.ariel_search.attachments.compose.caption_model_id`
    does); ``image_embedding`` is the image table, None when the module is off
    or misconfigured.
    """
    from osprey.services.ariel_search.attachments.compose import caption_model_id

    return {
        "image_caption": caption_model_id(config),
        "image_embedding": image_embedding_marker(config)[0],
    }


def enabled_modules(config: ARIELConfig) -> list[str]:
    """The enabled enhancement modules, in registry execution order."""
    from osprey.registry import get_registry

    return [
        name
        for name in get_registry().list_ariel_enhancement_modules()
        if config.is_enhancement_module_enabled(name)
    ]


@dataclass(frozen=True)
class EnhancementContext:
    """What a module's state depends on beyond the entry itself.

    Attributes:
        markers: The current marker per module (:func:`current_markers`).
        enabled: The enabled modules, in registry execution order.
        has_copy_state: Whether ``attachment_files.copy_status`` exists; without
            it no entry has a viewable picture (:func:`has_images_sql`).
    """

    markers: Mapping[str, str | None] = field(default_factory=dict)
    enabled: Sequence[str] = ()
    has_copy_state: bool = True

    def view(self, entry: Mapping[str, Any]) -> dict[str, EnhancementState]:
        """The state of every enabled module for ``entry`` (:func:`enhancement_view`)."""
        return enhancement_view(
            entry.get("enhancement_status"),
            has_images=has_images(entry),
            enabled=self.enabled,
            markers=self.markers,
        )

    def state_sql(self, module: str, state: str, alias: str) -> tuple[str, list[Any]]:
        """``module`` being in ``state`` over ``alias`` (:func:`enhancement_state_sql`)."""
        return enhancement_state_sql(
            module,
            state,
            alias,
            marker=self.markers.get(module),
            has_copy_state=self.has_copy_state,
        )


def enhancement_context(config: ARIELConfig, *, has_copy_state: bool = True) -> EnhancementContext:
    """The :class:`EnhancementContext` of ``config``, resolved once."""
    return EnhancementContext(
        markers=current_markers(config),
        enabled=tuple(enabled_modules(config)),
        has_copy_state=has_copy_state,
    )


async def store_enhancement_context(
    config: ARIELConfig, repository: ARIELRepository
) -> EnhancementContext:
    """The :class:`EnhancementContext` of ``config`` under ``repository``'s schema facts."""
    facts = await repository.schema_facts()
    return enhancement_context(config, has_copy_state=facts.has_copy_state)


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------


def has_images_sql(alias: str, *, has_copy_state: bool = True) -> str:
    """Whether the entry ``alias`` has a viewable picture, as a SQL fragment.

    ``EXISTS`` over ``attachment_files f`` filtered by
    :func:`~osprey.imaging.formats.viewable_sql`; the constant ``FALSE`` when the
    store has no ``copy_status`` column. Binds no parameters.

    Raises:
        ValueError: If ``alias`` is not a plain SQL identifier, or is ``f``.
    """
    _check_alias(alias)
    if alias == _FILES_ALIAS:
        raise ValueError(f"alias {alias!r} collides with the attachment_files alias")
    if not has_copy_state:
        return "FALSE"
    return (
        f"EXISTS (SELECT 1 FROM attachment_files {_FILES_ALIAS}"
        f" WHERE {_FILES_ALIAS}.entry_id = {alias}.entry_id"
        f" AND {viewable_sql(_FILES_ALIAS)})"
    )


def has_images(entry: Mapping[str, Any]) -> bool:
    """The ``has_images`` boolean a read selected with :func:`has_images_sql`."""
    return entry.get("has_images") is True


# ---------------------------------------------------------------------------
# The decision table in Python
# ---------------------------------------------------------------------------


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "t", "yes", "y", "on", "1")
    return False


def _attempts(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_status(status_json: Any) -> Mapping[str, Any]:
    if isinstance(status_json, (str, bytes)):
        try:
            status_json = json.loads(status_json)
        except ValueError:
            return {}
    return status_json if isinstance(status_json, Mapping) else {}


def enhancement_state(
    module: str,
    status: Mapping[str, Any],
    *,
    has_images: bool,
    marker: str | None,
) -> EnhancementState:
    """The state of one module for one entry (the decision table above).

    Args:
        module: The module name.
        status: The entry's parsed ``enhancement_status`` object.
        has_images: Whether the entry has a viewable picture.
        marker: The module's current marker (ignored for text modules).
    """
    image = module in IMAGE_MODULES
    if image and not has_images:
        return "n/a"
    if module not in status:
        return "pending"
    stored = status.get(module)
    obj: Mapping[str, Any] = stored if isinstance(stored, Mapping) else {}
    if image and marker is not None and obj.get("marker") != marker:
        return "stale"
    state = obj.get("status")
    if (
        state == "failed"
        or _truthy(obj.get("gave_up"))
        or (not image and _attempts(obj.get("attempts")) >= _max_attempts())
    ):
        return "failed"
    if state == "complete":
        return "complete"
    return "pending"


def enhancement_view(
    status_json: Any,
    *,
    has_images: bool,
    enabled: Sequence[str],
    markers: Mapping[str, str | None],
) -> dict[str, EnhancementState]:
    """The state of every enabled module for one entry.

    Args:
        status_json: The stored ``enhancement_status`` (a mapping, a JSON string,
            or None).
        has_images: Whether the entry has a viewable picture (:func:`has_images`).
        enabled: The enabled modules, in registry execution order; a disabled
            module is absent from the result.
        markers: The current marker per module (:func:`current_markers`).

    Returns:
        ``{module: state}`` in the order of ``enabled``.
    """
    status = _parse_status(status_json)
    return {
        module: enhancement_state(module, status, has_images=has_images, marker=markers.get(module))
        for module in enabled
    }


# ---------------------------------------------------------------------------
# The failed and stale rows in SQL
# ---------------------------------------------------------------------------


def enhancement_state_sql(
    module: str,
    state: str,
    alias: str,
    *,
    marker: str | None = None,
    has_copy_state: bool = True,
) -> tuple[str, list[Any]]:
    """The predicate "``module`` is in ``state``" over ``enhanced_entries <alias>``.

    Implements the ``stale`` and ``failed`` rows of the decision table, with the
    rows above each one negated, so the SQL selects exactly the entries
    :func:`enhancement_view` reports in that state. For image modules both
    predicates AND :func:`has_images_sql`, so an entry with no viewable picture
    is never ``failed`` or ``stale`` (rule 1).

    Args:
        module: The module name (bound as a parameter).
        state: ``failed`` or ``stale``.
        alias: The alias of ``enhanced_entries`` in the enclosing query.
        marker: The module's current marker (:func:`current_markers`). A text
            module, or an image module without one, is never ``stale``.
        has_copy_state: ``SchemaFacts.has_copy_state``.

    Returns:
        ``(sql, params)``: a fragment with positional ``%s`` placeholders and its
        parameters, in order.

    Raises:
        ValueError: On an unknown state or an alias that is not a plain identifier.
    """
    if state not in SQL_STATES:
        raise ValueError(f"unknown enhancement state {state!r}; expected one of {SQL_STATES}")
    _check_alias(alias)
    image = module in IMAGE_MODULES
    obj = f"{alias}.enhancement_status->%s"

    clauses: list[str] = [f"({alias}.enhancement_status ? %s)"]
    params: list[Any] = [module]

    marker_applies = image and marker is not None
    differs = f"({obj}->>'marker' IS DISTINCT FROM %s)"

    if state == "stale":
        if not marker_applies:
            return "FALSE", []
        clauses.append(differs)
        params += [module, marker]
    else:
        if marker_applies:
            clauses.append(f"NOT {differs}")
            params += [module, marker]
        failed = [
            f"{obj}->>'status' = 'failed'",
            f"COALESCE(({obj}->>'gave_up')::boolean, false)",
        ]
        params += [module, module]
        if not image:
            failed.append(f"COALESCE(({obj}->>'attempts')::int, 0) >= %s")
            params += [module, _max_attempts()]
        clauses.append("(" + " OR ".join(failed) + ")")

    if image:
        clauses.append(has_images_sql(alias, has_copy_state=has_copy_state))
    return "(" + " AND ".join(clauses) + ")", params
