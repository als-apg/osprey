"""The per-entry state of each enhancement module, in Python and in SQL.

One decision table, two forms that must agree:

* :func:`enhancement_view` — the state of every enabled module for one entry,
  read from its stored ``enhancement_status`` JSONB.
* :func:`enhancement_state_case_sql` — the same table as one SQL ``CASE``
  yielding the state, and :func:`enhancement_state_sql` — "is in this state" as
  a predicate over it, for filters and counts. Both take positional parameters.

The table, first matching row wins:

1. An image module and the entry has no viewable picture → ``n/a``.
2. No key → ``pending``.
3. Status ``failed``, or ``gave_up``, or a text module with attempts ≥
   :data:`MAX_ENHANCEMENT_ATTEMPTS` → ``failed``. Failed wins over stale.
4. The module's current marker is not null and its stored marker is "marker
   stale" (:func:`marker_stale`) → ``stale``:

   * a text module: the stored marker is present and differs. A completion
     written before markers existed is not stale (upgrade-safe);
   * an image module: the stored marker differs, a missing one included;
   * ``text_embedding``: its "marker" is the build key. The current one is the
     running key (:func:`text_embedding_build`), the stored one is
     ``COALESCE(input_key, 'r1')`` of the entry's row in the current text table
     (absent when it has none). Without a verdict — no ``input_key`` column, an
     unusable vocabulary, no text table — it is never stale.

5. Status ``complete`` → ``complete``.
6. Anything else → ``pending``.

Supporting pieces:

* :func:`has_images_sql` / :func:`has_images` — the one notion of "has a
  picture": a viewable ``attachment_files`` row (never the attachments JSONB).
* :func:`current_markers` — the marker every marker module is currently
  written under: the caption model, the explicit ``semantic_processor`` model
  (:func:`semantic_processor_marker`), the image table
  (:func:`image_embedding_marker`) and the mirror format
  (:func:`qmd_export_marker`). Each is None on any configuration error.
* :func:`marker_stale_sql` — rule 4 as the one SQL fragment the CASE and the
  repository's owed and count queries share.
* :func:`enabled_modules` and :class:`EnhancementContext` — the enabled modules
  in registry execution order, and the context filters and counts take; its
  markers are resolved once, by :func:`enhancement_context`.
* :func:`text_embedding_build` — the running build key of a text-embedding
  table, the one computation ``osprey ariel status`` and the context share.
* :func:`rerun_command` — the shell command that re-runs one module for one
  entry, and :func:`reembed_command`, the table rebuild ``status`` prints.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from osprey.imaging.formats import viewable_sql
from osprey_connectors.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.vocabulary.entry import EntryExpander

__all__ = [
    "EnhancementContext",
    "EnhancementState",
    "IMAGE_MODULES",
    "MARKER_MODULES",
    "STATES",
    "SQL_STATES",
    "TEXT_EMBEDDING",
    "TextEmbeddingBuild",
    "build_key_stale_sql",
    "current_text_table",
    "current_markers",
    "enabled_modules",
    "enhancement_context",
    "enhancement_state",
    "enhancement_state_case_sql",
    "enhancement_state_sql",
    "enhancement_view",
    "filter_needs_build_keys",
    "has_images",
    "has_images_sql",
    "image_embedding_marker",
    "marker_stale",
    "marker_stale_sql",
    "parse_status",
    "qmd_export_marker",
    "reembed_command",
    "rerun_command",
    "semantic_processor_marker",
    "status_expander",
    "store_enhancement_context",
    "text_embedding_build",
    "with_build_keys",
]

logger = get_logger("ariel")

EnhancementState = Literal["n/a", "pending", "stale", "failed", "complete"]

#: Every state :func:`enhancement_view` can report.
STATES: tuple[str, ...] = ("n/a", "pending", "stale", "failed", "complete")

#: The states an enhancement filter (``<module>:<state>``) can select.
#: :func:`enhancement_state_sql` itself takes any of :data:`STATES`.
SQL_STATES: tuple[str, ...] = ("failed", "stale")

#: The modules whose input is the entry's pictures. They are ``n/a`` on an entry
#: with no viewable picture, and a missing stored marker is stale for them.
#: Equal to ``repository.IMAGE_STATUS_KEYS``, kept here so this module imports no
#: repository.
IMAGE_MODULES: tuple[str, ...] = ("image_caption", "image_embedding")

#: The modules whose status records the marker it was written under (caption
#: model, summary model, image table, mirror format), in :func:`current_markers`'
#: order. ``text_embedding`` is not one: its staleness is the build key's.
MARKER_MODULES: tuple[str, ...] = (
    "image_caption",
    "semantic_processor",
    "image_embedding",
    "qmd_export",
)

#: The module whose staleness is its rows' build key, not a stored marker.
TEXT_EMBEDDING = "text_embedding"

#: The build key a text-embedding row with a NULL ``input_key`` was written under.
LEGACY_BUILD_KEY = "r1"

#: The alias of the text-embedding table inside :func:`build_key_stale_sql`.
_TEXT_ALIAS = "t"

#: The ``semantic_processor`` marker when the module names no ``model.model_id``
#: and runs on the deployment's main model: a main-model change never re-summarises.
DEFAULT_MODEL_MARKER = "default"

#: The expander part of the ``qmd_export`` marker when entry expansion is off.
NO_EXPANDER_KEY = "none"

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


def semantic_processor_marker(config: ARIELConfig) -> str | None:
    """The completion marker of an enabled ``semantic_processor`` block.

    The module's explicit ``model.model_id``, stripped, or
    :data:`DEFAULT_MODEL_MARKER` when it names none (it then runs on the main
    model). A prompt change is not part of it.

    Returns:
        The marker; None when the module is off or its block is broken (no
        ``provider``, a ``model`` that is not a mapping, a non-string
        ``model_id``), or reading the configuration raised.
    """
    try:
        if not config.is_enhancement_module_enabled("semantic_processor"):
            return None
        block = config.get_enhancement_module_config("semantic_processor")
        if not isinstance(block, Mapping) or not block.get("provider"):
            return None
        model = block.get("model")
        if model is None:
            model = {}
        if not isinstance(model, Mapping):
            return None
        model_id = model.get("model_id")
        if model_id is None:
            return DEFAULT_MODEL_MARKER
        if not isinstance(model_id, str):
            return None
        return model_id.strip() or DEFAULT_MODEL_MARKER
    except Exception as error:
        logger.debug("semantic_processor marker unavailable: %s", error)
        return None


def qmd_export_marker(config: ARIELConfig) -> str | None:
    """The completion marker of an enabled ``qmd_export`` block.

    ``"<QMD_FORMAT_VERSION>:<mirror_path>:<expander key>"``. ``mirror_path`` is
    the configured string, never the resolved root: ariel-sync, tiered terminals
    and the host mount one mirror at different paths. The expander key is
    :data:`NO_EXPANDER_KEY` when entry expansion is off.

    Returns:
        The marker; None when the module is off, names no ``mirror_path``, entry
        expansion is on and the vocabulary is unusable, or reading the
        configuration raised.
    """
    try:
        if not config.is_enhancement_module_enabled("qmd_export"):
            return None
        block = config.get_enhancement_module_config("qmd_export")
        raw = block.get("mirror_path") if isinstance(block, Mapping) else None
        if not isinstance(raw, str) or not raw.strip():
            return None
        expander = config.entry_expander()
        key = NO_EXPANDER_KEY if expander is None else expander.key
        from osprey.services.ariel_search.enhancement.qmd_export.exporter import (
            QMD_FORMAT_VERSION,
        )
    except Exception as error:
        logger.debug("qmd_export marker unavailable: %s", error)
        return None
    return f"{QMD_FORMAT_VERSION}:{raw}:{key}"


def current_markers(config: ARIELConfig) -> dict[str, str | None]:
    """The marker each of :data:`MARKER_MODULES` is currently written under.

    ``image_caption`` is the caption model id (its ``enabled`` flag ignored, as
    :func:`~osprey.services.ariel_search.attachments.compose.caption_model_id`
    does); ``semantic_processor`` is :func:`semantic_processor_marker`;
    ``image_embedding`` is the image table; ``qmd_export`` is
    :func:`qmd_export_marker`. Each is None when its module is off or
    misconfigured. Resolve it once per :class:`EnhancementContext`
    (:func:`enhancement_context`).
    """
    from osprey.services.ariel_search.attachments.compose import caption_model_id

    return {
        "image_caption": caption_model_id(config),
        "semantic_processor": semantic_processor_marker(config),
        "image_embedding": image_embedding_marker(config)[0],
        "qmd_export": qmd_export_marker(config),
    }


def enabled_modules(config: ARIELConfig) -> list[str]:
    """The enabled enhancement modules, in registry execution order."""
    from osprey.registry import get_registry

    return [
        name
        for name in get_registry().list_ariel_enhancement_modules()
        if config.is_enhancement_module_enabled(name)
    ]


# ---------------------------------------------------------------------------
# The text-embedding build key
# ---------------------------------------------------------------------------


def status_expander(config: ARIELConfig) -> tuple[EntryExpander | None, bool]:
    """The entry expander, resolved without raising.

    Returns:
        ``(expander, unusable)``: the expander or None, and whether entry
        expansion is configured on but the vocabulary cannot back it.
    """
    from osprey.services.ariel_search.exceptions import ModuleConfigError

    try:
        return config.entry_expander(), False
    except ModuleConfigError:
        return None, True


def current_text_table(config: ARIELConfig) -> str | None:
    """The current text-embedding table: the search model's, else the first model's.

    Returns:
        The table name; None when no search model is set and ``text_embedding``
        names no model.
    """
    from osprey.services.ariel_search.database.migrations import model_to_table_name

    try:
        search_model = config.get_search_model()
        if search_model:
            return model_to_table_name(search_model)
        module = config.enhancement_modules.get(TEXT_EMBEDDING)
        models = (module.models if module else None) or []
        return model_to_table_name(models[0].name) if models else None
    except Exception as error:
        logger.debug("current text table unavailable: %s", error)
        return None


@dataclass(frozen=True)
class TextEmbeddingBuild:
    """How the rows of one text-embedding table are built now.

    Attributes:
        table: The ``text_embeddings_*`` table.
        model: The configured model ``{name, dimension}`` of ``table``, or None.
        running_key: The build key a row written now carries; None when no
            verdict is possible from the configuration.
        unavailable: Why no plain verdict is available: ``disabled``,
            ``unconfigured``, ``vocabulary`` (no running key), or
            ``vocabulary_disabled`` (a running key without an expander); None
            for a plain verdict.
    """

    table: str | None
    model: Mapping[str, Any] | None
    running_key: str | None
    unavailable: str | None


def text_embedding_build(
    config: ARIELConfig,
    table: str | None = None,
    *,
    expander: tuple[EntryExpander | None, bool] | None = None,
) -> TextEmbeddingBuild:
    """The running build key of ``table`` under ``config``.

    The one running-key computation: ``osprey ariel status`` reports a table's
    stale rows against it and :func:`enhancement_context` reports each entry's.
    The first applicable reason wins: the module is off (``disabled``), no
    configured model maps to ``table`` (``unconfigured``), entry expansion is on
    but the vocabulary cannot back it (``vocabulary``). Otherwise the running key
    is :func:`running_input_key` of the expander; with entry expansion on while
    the vocabulary is disabled it carries no expander (``vocabulary_disabled``).
    Whether ``table`` carries ``input_key`` is the caller's to probe.

    Args:
        config: The parsed ARIEL config.
        table: The ``text_embeddings_*`` table; None for :func:`current_text_table`.
        expander: ``(expander, unusable)`` already resolved by
            :func:`status_expander`; resolved here when None.
    """
    from osprey.services.ariel_search.database.migrations import model_to_table_name
    from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
        running_input_key,
    )

    if table is None:
        table = current_text_table(config)
    module = config.enhancement_modules.get(TEXT_EMBEDDING)
    match = next(
        (
            m
            for m in (module.models if module else None) or []
            if table is not None and model_to_table_name(m.name) == table
        ),
        None,
    )
    model = {"name": match.name, "dimension": match.dimension} if match else None

    def unavailable(reason: str) -> TextEmbeddingBuild:
        return TextEmbeddingBuild(table, model, None, reason)

    if not config.is_enhancement_module_enabled(TEXT_EMBEDDING):
        return unavailable("disabled")
    if match is None:
        return unavailable("unconfigured")
    resolved, unusable = expander if expander is not None else status_expander(config)
    if unusable:
        return unavailable("vocabulary")
    reason = None
    if config.vocabulary.expand_entries and not config.vocabulary.enabled:
        reason = "vocabulary_disabled"
    return TextEmbeddingBuild(table, model, running_input_key(resolved), reason)


# ---------------------------------------------------------------------------
# The context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnhancementContext:
    """What a module's state depends on beyond the entry itself.

    Attributes:
        markers: The current marker per module (:func:`current_markers`).
        enabled: The enabled modules, in registry execution order.
        has_copy_state: Whether ``attachment_files.copy_status`` exists; without
            it no entry has a viewable picture (:func:`has_images_sql`).
        text_model: The model (``name``, ``dimension``) of the current text
            embedding table, which :func:`rerun_command` names for a
            ``text_embedding`` rebuild; None when not resolved.
        text_table: The current text-embedding table, set only when a
            ``text_embedding`` verdict is possible.
        text_key: The running build key of ``text_table``; None when no verdict
            is possible, and ``text_embedding`` is then never stale.
        build_keys: ``COALESCE(input_key, 'r1')`` per entry id of the rows of
            ``text_table`` for the entries this context was read for
            (:func:`store_enhancement_context` ``entry_ids``); an entry absent
            from it has no row.
    """

    markers: Mapping[str, str | None] = field(default_factory=dict)
    enabled: Sequence[str] = ()
    has_copy_state: bool = True
    text_model: Mapping[str, Any] | None = None
    text_table: str | None = None
    text_key: str | None = None
    build_keys: Mapping[str, str] = field(default_factory=dict)

    def view(self, entry: Mapping[str, Any]) -> dict[str, EnhancementState]:
        """The state of every enabled module for ``entry`` (:func:`enhancement_view`)."""
        entry_id = entry.get("entry_id")
        return enhancement_view(
            entry.get("enhancement_status"),
            has_images=has_images(entry),
            enabled=self.enabled,
            markers=self.markers,
            text_key=self.text_key,
            build_key=None if entry_id is None else self.build_keys.get(str(entry_id)),
        )

    def marker(self, module: str) -> str | None:
        """``module``'s current marker; for ``text_embedding`` the running build key."""
        return self.text_key if module == TEXT_EMBEDDING else self.markers.get(module)

    def state_sql(self, module: str, state: str, alias: str) -> tuple[str, list[Any]]:
        """``module`` being in ``state`` over ``alias`` (:func:`enhancement_state_sql`)."""
        return enhancement_state_sql(
            module,
            state,
            alias,
            marker=self.marker(module),
            has_copy_state=self.has_copy_state,
            text_table=self.text_table,
        )

    def case_sql(self, module: str, alias: str) -> tuple[str, list[Any]]:
        """``module``'s state over ``alias`` (:func:`enhancement_state_case_sql`)."""
        return enhancement_state_case_sql(
            module,
            alias,
            marker=self.marker(module),
            has_copy_state=self.has_copy_state,
            text_table=self.text_table,
        )


def enhancement_context(
    config: ARIELConfig, *, has_copy_state: bool, input_key_column: bool
) -> EnhancementContext:
    """The :class:`EnhancementContext` of ``config``, resolved once.

    The markers and the running build key are read here and only here, so one
    context reads the configuration once however many entries and predicates
    it serves. Both schema facts are required: a site that guessed one would
    silently report a picture or a stale embedding the store cannot have.

    Args:
        config: The parsed ARIEL config.
        has_copy_state: ``SchemaFacts.has_copy_state``.
        input_key_column: Whether :func:`current_text_table` carries an
            ``input_key`` column; without it ``text_embedding`` is never stale.
    """
    build = text_embedding_build(config)
    verdict = input_key_column and build.running_key is not None
    return EnhancementContext(
        markers=current_markers(config),
        enabled=tuple(enabled_modules(config)),
        has_copy_state=has_copy_state,
        text_model=build.model,
        text_table=build.table if verdict else None,
        text_key=build.running_key if verdict else None,
    )


async def store_enhancement_context(
    config: ARIELConfig,
    repository: ARIELRepository,
    *,
    entry_ids: Sequence[str] | None = None,
    text_embedding: bool = True,
) -> EnhancementContext:
    """The :class:`EnhancementContext` of ``config`` under ``repository``'s schema.

    Args:
        config: The parsed ARIEL config.
        repository: Supplies the schema facts, the cached ``input_key`` probe
            of the current text table and the build keys.
        entry_ids: The entries the context's :meth:`~EnhancementContext.view`
            serves (a page, or one entry read); their build keys are read in one
            batched query. None for a context that serves SQL only.
        text_embedding: False when the context is provably never asked about
            ``text_embedding`` (:func:`filter_needs_build_keys`); it then probes
            nothing and gives ``text_embedding`` no verdict.
    """
    facts = await repository.schema_facts()
    table = current_text_table(config)
    column = False
    if (
        text_embedding
        and table is not None
        and config.is_enhancement_module_enabled(TEXT_EMBEDDING)
    ):
        column = await repository.text_input_key_column(table) is True
    context = enhancement_context(
        config, has_copy_state=facts.has_copy_state, input_key_column=column
    )
    if entry_ids:
        context = await with_build_keys(context, repository, entry_ids)
    return context


def filter_needs_build_keys(entry_filter: Any) -> bool:
    """Whether ``entry_filter`` (an ``EntryFilter``) selects on a ``text_embedding`` state."""
    table = getattr(entry_filter, "table", {}) or {}
    for name, values in getattr(entry_filter, "dims", ()):
        dim = table.get(name)
        if getattr(dim, "kind", None) != "enhancement":
            continue
        if any(value.partition(":")[0] == TEXT_EMBEDDING for value in values):
            return True
    return False


async def with_build_keys(
    context: EnhancementContext, repository: ARIELRepository, entry_ids: Sequence[str]
) -> EnhancementContext:
    """``context`` serving the entries ``entry_ids``: their build keys, read in one query.

    ``context`` itself when it has no ``text_embedding`` verdict or there are no
    entries; no read then happens.
    """
    if context.text_table is None or not entry_ids:
        return context
    keys = await repository.get_text_build_keys(context.text_table, list(entry_ids))
    return replace(context, build_keys=keys)


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


def parse_status(status_json: Any) -> Mapping[str, Any]:
    """A stored ``enhancement_status`` (a mapping, JSON text or None); ``{}`` when unreadable."""
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
    build_key: str | None = None,
) -> EnhancementState:
    """The state of one module for one entry (the decision table above).

    Args:
        module: The module name.
        status: The entry's parsed ``enhancement_status`` object.
        has_images: Whether the entry has a viewable picture.
        marker: The module's current marker (:func:`current_markers`), for
            ``text_embedding`` the running build key; None means the module is
            never stale.
        build_key: ``text_embedding`` only: ``COALESCE(input_key, 'r1')`` of the
            entry's row in the current text table, None when it has none.
    """
    image = module in IMAGE_MODULES
    if image and not has_images:
        return "n/a"
    if module not in status:
        return "pending"
    stored = status.get(module)
    obj: Mapping[str, Any] = stored if isinstance(stored, Mapping) else {}
    state = obj.get("status")
    if (
        state == "failed"
        or _truthy(obj.get("gave_up"))
        or (not image and _attempts(obj.get("attempts")) >= _max_attempts())
    ):
        return "failed"
    stored_marker = build_key if module == TEXT_EMBEDDING else obj.get("marker")
    if marker_stale(module, stored_marker, marker):
        return "stale"
    if state == "complete":
        return "complete"
    return "pending"


def marker_stale(module: str, stored: Any, marker: str | None) -> bool:
    """Whether ``module``'s stored marker is stale against the current one (rule 4).

    Args:
        module: The module name.
        stored: The marker its status records (None when absent); for
            ``text_embedding`` the build key of its row (None without one).
        marker: The module's current marker; None is never stale.
    """
    if marker is None:
        return False
    if module in IMAGE_MODULES:
        return bool(stored != marker)
    return stored is not None and bool(stored != marker)


def enhancement_view(
    status_json: Any,
    *,
    has_images: bool,
    enabled: Sequence[str],
    markers: Mapping[str, str | None],
    text_key: str | None = None,
    build_key: str | None = None,
) -> dict[str, EnhancementState]:
    """The state of every enabled module for one entry.

    Args:
        status_json: The stored ``enhancement_status`` (a mapping, a JSON string,
            or None).
        has_images: Whether the entry has a viewable picture (:func:`has_images`).
        enabled: The enabled modules, in registry execution order; a disabled
            module is absent from the result.
        markers: The current marker per module (:func:`current_markers`).
        text_key: The running build key; None when no ``text_embedding``
            verdict is possible.
        build_key: The entry's stored build key (:func:`enhancement_state`).

    Returns:
        ``{module: state}`` in the order of ``enabled``.
    """
    status = parse_status(status_json)
    return {
        module: enhancement_state(
            module,
            status,
            has_images=has_images,
            marker=text_key if module == TEXT_EMBEDDING else markers.get(module),
            build_key=build_key,
        )
        for module in enabled
    }


# ---------------------------------------------------------------------------
# The decision table in SQL
# ---------------------------------------------------------------------------


def marker_stale_sql(module: str, alias: str, *, marker: str | None) -> tuple[str, list[Any]]:
    """Rule 4, "marker stale", over ``enhanced_entries <alias>`` (:func:`marker_stale`).

    The one fragment the five-state CASE, the owed-work query and the counts
    share. It never yields NULL, so it may be negated. It does not check that
    the module's key is present.

    ``text_embedding`` stores no marker: its staleness is the build key's
    (:func:`build_key_stale_sql`), which re-embedding by ``reembed`` clears, so
    this fragment, and with it the owed-work query, is ``FALSE`` for it.

    Args:
        module: The module name (bound as a parameter).
        alias: The alias of ``enhanced_entries`` in the enclosing query.
        marker: The module's current marker; None gives the constant ``FALSE``.

    Returns:
        ``(sql, params)`` with positional ``%s`` placeholders.

    Raises:
        ValueError: If ``alias`` is not a plain SQL identifier.
    """
    _check_alias(alias)
    if marker is None or module == TEXT_EMBEDDING:
        return "FALSE", []
    stored = f"{alias}.enhancement_status->%s->>'marker'"
    if module in IMAGE_MODULES:
        return f"({stored} IS DISTINCT FROM %s)", [module, marker]
    return f"({stored} IS NOT NULL AND {stored} <> %s)", [module, module, marker]


def _quoted_table(table: str) -> str:
    return '"' + table.replace('"', '""') + '"'


def build_key_stale_sql(alias: str, table: str, running_key: str) -> tuple[str, list[Any]]:
    """Rule 4 for ``text_embedding`` over ``enhanced_entries <alias>``.

    The entry has a row in ``table`` whose ``COALESCE(input_key, 'r1')`` differs
    from ``running_key``: :func:`marker_stale`'s text rule with the row's build
    key as the stored marker, and the entries
    :func:`~osprey.services.ariel_search.vocabulary.entry.classify_build_keys`
    counts stale. Never NULL.

    Args:
        alias: The alias of ``enhanced_entries`` in the enclosing query.
        table: The text-embedding table (a column-probed table name).
        running_key: The running build key (:func:`text_embedding_build`).

    Returns:
        ``(sql, params)`` with positional ``%s`` placeholders.

    Raises:
        ValueError: If ``alias`` is not a plain SQL identifier, or is ``t``.
    """
    _check_alias(alias)
    if alias == _TEXT_ALIAS:
        raise ValueError(f"alias {alias!r} collides with the text-embedding alias")
    t = _TEXT_ALIAS
    return (
        f"EXISTS (SELECT 1 FROM {_quoted_table(table)} {t}"
        f" WHERE {t}.entry_id = {alias}.entry_id"
        f" AND COALESCE({t}.input_key, '{LEGACY_BUILD_KEY}') <> %s)",
        [running_key],
    )


def enhancement_state_case_sql(
    module: str,
    alias: str,
    *,
    marker: str | None = None,
    has_copy_state: bool = True,
    text_table: str | None = None,
) -> tuple[str, list[Any]]:
    """The state of ``module`` over ``enhanced_entries <alias>``, as one SQL CASE.

    The decision table's rows as ``WHEN`` arms in order, yielding the same
    state :func:`enhancement_state` gives for the row: ``n/a``, ``pending``,
    ``stale``, ``failed`` or ``complete``. Image modules start with the
    :func:`has_images_sql` arm; the ``stale`` arm is left out when ``marker`` is
    None.

    Args:
        module: The module name (bound as a parameter).
        alias: The alias of ``enhanced_entries`` in the enclosing query.
        marker: The module's current marker (:func:`current_markers`); for
            ``text_embedding`` the running build key.
        has_copy_state: ``SchemaFacts.has_copy_state``.
        text_table: ``text_embedding`` only: the current text table; its
            ``stale`` arm needs both it and ``marker``.

    Returns:
        ``(sql, params)``: a parenthesised ``CASE`` expression with positional
        ``%s`` placeholders and its parameters, in order.

    Raises:
        ValueError: If ``alias`` is not a plain SQL identifier.
    """
    _check_alias(alias)
    image = module in IMAGE_MODULES
    obj = f"{alias}.enhancement_status->%s"
    arms: list[str] = []
    params: list[Any] = []

    if image:
        arms.append(f"WHEN NOT {has_images_sql(alias, has_copy_state=has_copy_state)} THEN 'n/a'")
    arms.append(f"WHEN NOT COALESCE({alias}.enhancement_status ? %s, false) THEN 'pending'")
    params.append(module)

    failed = [
        f"{obj}->>'status' = 'failed'",
        f"COALESCE(({obj}->>'gave_up')::boolean, false)",
    ]
    params += [module, module]
    if not image:
        failed.append(f"COALESCE(({obj}->>'attempts')::int, 0) >= %s")
        params += [module, _max_attempts()]
    arms.append(f"WHEN {' OR '.join(failed)} THEN 'failed'")

    if module == TEXT_EMBEDDING:
        stale_sql, stale_params = (
            build_key_stale_sql(alias, text_table, marker)
            if text_table is not None and marker is not None
            else ("FALSE", [])
        )
    else:
        stale_sql, stale_params = marker_stale_sql(module, alias, marker=marker)
    if stale_params:
        arms.append(f"WHEN {stale_sql} THEN 'stale'")
        params += stale_params

    arms.append(f"WHEN {obj}->>'status' = 'complete' THEN 'complete'")
    params.append(module)
    return "(CASE " + " ".join(arms) + " ELSE 'pending' END)", params


def enhancement_state_sql(
    module: str,
    state: str,
    alias: str,
    *,
    marker: str | None = None,
    has_copy_state: bool = True,
    text_table: str | None = None,
) -> tuple[str, list[Any]]:
    """The predicate "``module`` is in ``state``" over ``enhanced_entries <alias>``.

    :func:`enhancement_state_case_sql` compared with ``state``, so it selects
    exactly the entries :func:`enhancement_view` reports in that state. A
    ``stale`` predicate without a current marker is the constant ``FALSE``,
    which callers (the problem facet) may drop.

    Args:
        module: The module name (bound as a parameter).
        state: One of :data:`STATES`.
        alias: The alias of ``enhanced_entries`` in the enclosing query.
        marker: The module's current marker (:func:`current_markers`).
        has_copy_state: ``SchemaFacts.has_copy_state``.
        text_table: ``text_embedding`` only (:func:`enhancement_state_case_sql`).

    Returns:
        ``(sql, params)``: a fragment with positional ``%s`` placeholders and its
        parameters, in order.

    Raises:
        ValueError: On an unknown state or an alias that is not a plain identifier.
    """
    if state not in STATES:
        raise ValueError(f"unknown enhancement state {state!r}; expected one of {STATES}")
    _check_alias(alias)
    if state == "stale" and (marker is None or (module == TEXT_EMBEDDING and text_table is None)):
        return "FALSE", []
    case, params = enhancement_state_case_sql(
        module, alias, marker=marker, has_copy_state=has_copy_state, text_table=text_table
    )
    return f"({case} = %s)", [*params, state]


# ---------------------------------------------------------------------------
# The re-run command
# ---------------------------------------------------------------------------


def reembed_command(name: str, dimension: Any) -> str:
    """The command that rebuilds the text-embedding table of model ``name``.

    The one producer of the ``reembed`` line: ``osprey ariel status`` prints it
    and :func:`rerun_command` returns it.
    """
    return f"osprey ariel reembed --model {name} --dimension {dimension} --force"


def rerun_command(
    module: str, entry_id: str, state: str, context: EnhancementContext
) -> str | None:
    """The shell command that re-runs ``module`` for one entry in ``state``.

    Args:
        module: The module name.
        entry_id: The entry, shell-quoted into the command.
        state: The module's state for the entry (:func:`enhancement_state`).
        context: The context the state was read under.

    Returns:
        ``osprey ariel enhance --module M --entry ID`` for a failed or stale
        module; for ``text_embedding`` stale (a build-key mismatch, which only a
        table rebuild clears) the :func:`reembed_command` of the current text
        table, None when the context names none; None in any other state.
    """
    if state not in ("failed", "stale"):
        return None
    if module == "text_embedding" and state == "stale":
        model = context.text_model
        if not model or not model.get("name"):
            return None
        return reembed_command(str(model["name"]), model.get("dimension"))
    return f"osprey ariel enhance --module {module} --entry {shlex.quote(entry_id)}"
