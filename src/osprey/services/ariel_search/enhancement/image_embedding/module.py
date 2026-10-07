"""ARIEL ``image_embedding`` module: one vector per viewable picture.

The module runs only in the catch-up (``runs_inline = False``), which drives it
through :meth:`ImageEmbeddingModule.run_entry`. Each entry is a three-phase
write:

1. read the entry's pictures with no lock held and pick the viewable ones that
   have no row in the target table;
2. per picture, one ``execute_image_embedding`` call on a daemon thread
   (:func:`~osprey.services.ariel_search.enhancement._offload.run_blocking`)
   with no connection held;
3. per result, one guarded statement: ``INSERT … SELECT … WHERE EXISTS`` the
   picture is still viewable, so a picture deleted or re-rendered during the
   call is never written.

The model, vector width and table come from
:func:`~osprey.services.ariel_search.database.migrations.image_embedding_target`;
the table name is also the completion marker, so another model or width makes
every picture owed again. The adapter truncates and renormalises to the
configured width; the module never truncates again.

A deterministic failure (a picture the server rejects, a degenerate vector) is
stored as a row with a NULL ``embedding`` and a ``skip_reason`` once the gate
allows it. The module keeps no breaker of its own: the catch-up driver's pass
breakers cover it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from osprey.imaging.formats import viewable_sql
from osprey.models.providers.health import HealthResult
from osprey.services.ariel_search.database.migrations import (
    ImageEmbeddingTarget,
    image_embedding_target,
)
from osprey.services.ariel_search.database.vector_literal import vector_literal
from osprey.services.ariel_search.enhancement import availability
from osprey.services.ariel_search.enhancement._offload import run_blocking
from osprey.services.ariel_search.enhancement.base import (
    BaseEnhancementModule,
    ImageEntryOutcome,
    PictureGate,
)
from osprey.services.ariel_search.enhancement.image_driver import viewable_in_list_order
from osprey.services.ariel_search.enhancement.image_embedding.migration import (
    ImageEmbeddingMigration,
)
from osprey.services.ariel_search.enhancement.provider_resolver import (
    ResolvedProvider,
    provider_given,
    resolve_provider,
    resolve_reachable_base_url,
)
from osprey.services.ariel_search.enhancement.vision_errors import (
    error_signature,
    failed_call_outcome,
)
from osprey.services.ariel_search.exceptions import ModuleConfigError
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = get_logger("ariel")

#: The configuration block of the module.
IMAGE_EMBEDDING_KEY = "ariel.enhancement_modules.image_embedding"

#: Seconds one picture embedding call may take; a timeout is transient.
DEFAULT_TIMEOUT_SECONDS = 120

#: Seconds the health check gives the server (the probe and ``/v1/models``).
HEALTH_PROBE_TIMEOUT_S = 5

#: The key that fixes ``no_reader``.
HYBRID_ENABLED_KEY = "ariel.search_modules.hybrid.enabled"

#: The availability-tracker key the ``no_reader`` transition is recorded under.
#: Kept apart from the module's own key, which every clean catch-up pass resets,
#: so the WARNING is logged once per transition, not once per pass.
READER_TRACKER_KEY = "image_embedding.reader"

#: ``skip_reason`` of a picture whose vector has zero norm or a non-finite value.
SKIP_DEGENERATE_VECTOR = "degenerate_vector"


def _timeout_seconds(config: Mapping[str, Any]) -> float:
    """Read ``timeout_seconds``, refusing anything but a positive number, naming its key."""
    value = config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if value is None:
        value = DEFAULT_TIMEOUT_SECONDS
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        key = f"{IMAGE_EMBEDDING_KEY}.timeout_seconds"
        raise ModuleConfigError(f"{key} must be a number > 0 (got {value!r})", key=key)
    return float(value)


def hybrid_search_enabled() -> bool | None:
    """``ariel.search_modules.hybrid.enabled`` of the loaded configuration.

    ``configure()`` receives only the module's own block, so the search-module
    switch is read from the loaded configuration. None when no configuration is
    loaded (tests, bare library use): unknown never reads as ``no_reader``.
    """
    try:
        from osprey.utils.config import get_config_value

        modules = get_config_value("ariel.search_modules", None)
    except Exception:  # no config.yml (tests, bare library use)
        return None
    if not isinstance(modules, Mapping):
        return None
    hybrid = modules.get("hybrid")
    if not isinstance(hybrid, Mapping):
        return False
    return bool(hybrid.get("enabled", False))


def _skip_reason(exc: BaseException) -> str:
    """The ``skip_reason`` a deterministic failure is stored with."""
    from osprey.models.providers.base import DegenerateVectorError

    for item in (exc, exc.__cause__, exc.__context__):
        if isinstance(item, DegenerateVectorError):
            return SKIP_DEGENERATE_VECTOR
    return error_signature(exc)


class ImageEmbeddingModule(BaseEnhancementModule):
    """Embed every viewable picture of an entry into the configured image table."""

    runs_inline = False

    def __init__(self) -> None:
        """Initialize the module unconfigured."""
        self._resolved: ResolvedProvider | None = None
        self._target: ImageEmbeddingTarget | None = None
        self._reachable_base_url: str | None = None
        #: Seconds one call may take; read by the catch-up to size its backstop.
        self.timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @property
    def name(self) -> str:
        """Return module identifier."""
        return "image_embedding"

    @property
    def migration(self) -> type[ImageEmbeddingMigration]:
        """Return the migration class; the runner builds it with the configured target."""
        return ImageEmbeddingMigration

    # -- configuration -----------------------------------------------------

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the module from ``ariel.enhancement_modules.image_embedding``.

        Does no network I/O and logs nothing: the catch-up configures the
        module on every pass.

        Args:
            config: The module's config dict (``provider``, ``provider_key``,
                ``model``, ``dimensions``, ``timeout_seconds``).

        Raises:
            ModuleConfigError: When no provider is set (``ariel.embedding.provider``
                is never a stand-in), the provider is unknown or serves no image
                embeddings, its adapter refuses the base URL, the model is
                missing, the dimensions are out of range, or ``timeout_seconds``
                is malformed. Each names the key that fixes it.
        """
        provider = config.get("provider")
        if not provider_given(provider):
            raise ModuleConfigError(
                f"{IMAGE_EMBEDDING_KEY}.provider is required",
                key=f"{IMAGE_EMBEDDING_KEY}.provider",
            )
        provider_key = config.get("provider_key") or f"{IMAGE_EMBEDDING_KEY}.provider"
        resolved = resolve_provider(
            provider, provider_key=provider_key, default=None, serves="image_embeddings"
        )
        target = image_embedding_target(config)
        timeout = _timeout_seconds(config)

        self._resolved = resolved
        self._target = target
        self._reachable_base_url = None
        self.timeout_seconds = timeout

    @property
    def target(self) -> ImageEmbeddingTarget | None:
        """The configured model, width and table; None before :meth:`configure`."""
        return self._target

    def required_relations(self) -> list[str]:
        """The image table the module writes; the pre-pass check looks it up."""
        return [self._target.table] if self._target is not None else []

    def completion_marker(self) -> str | None:
        """The image table name: entries are complete per table."""
        return self._target.table if self._target is not None else None

    # -- health ------------------------------------------------------------

    def health_reason(self) -> str | None:
        """The reason ``status`` reports over the health check's own.

        ``no_reader`` while the hybrid search module, the one reader of the
        vectors, is off; the transition is recorded with the availability
        tracker, which logs one WARNING when it appears and one INFO when it
        clears. Otherwise ``model`` when this process's last pass ended
        because the server rejected every picture (a server serving the alias
        but started without its picture projector, which ``/v1/models`` cannot
        show); else None.
        """
        if hybrid_search_enabled() is False:
            availability.report_unavailable(
                READER_TRACKER_KEY,
                "no_reader",
                "the hybrid search module is disabled, so no search reads the picture vectors",
                HYBRID_ENABLED_KEY,
            )
            return "no_reader"
        if availability.current_reason(READER_TRACKER_KEY) is not None:
            availability.report_available(READER_TRACKER_KEY)
        if availability.current_reason(self.name) == "model":
            return "model"
        return None

    async def health_check(self) -> HealthResult:
        """Whether the image embedding server answers and serves the configured model.

        Resolves the reachable base URL (re-checked once per call) and asks the
        adapter's ``check_embedding_health`` for its verdict, both on a daemon
        thread; the calls of the pass then go to the URL resolved here.

        Returns:
            The adapter's verdict unchanged (``unreachable``, ``auth`` or
            ``model`` set by the adapter); ``config`` when unconfigured.
        """
        self.health_reason()
        if self._resolved is None or self._target is None:
            return HealthResult(False, "image_embedding is not configured", "config")
        return await run_blocking(self._probe_health, key="image_embedding")

    def _probe_health(self) -> HealthResult:
        """The blocking half of :meth:`health_check`."""
        resolved = self._resolved
        target = self._target
        assert resolved is not None and target is not None
        try:
            base_url = self._base_url(refresh=True)
            return resolved.instance.check_embedding_health(
                resolved.api_key, base_url, model_id=target.model, timeout=HEALTH_PROBE_TIMEOUT_S
            )
        except Exception as exc:
            return HealthResult(
                False,
                f"{type(exc).__name__}: {exc}",
                availability.unavailable_reason(exc) or "unreachable",
            )

    def _base_url(self, *, refresh: bool = False) -> str | None:
        """The URL calls go to: the reachable one, resolved once and kept (``refresh`` re-checks)."""
        resolved = self._resolved
        if resolved is None or resolved.base_url is None:
            return None
        if self._reachable_base_url is None or refresh:
            self._reachable_base_url = resolve_reachable_base_url(
                resolved.cls, resolved.base_url, refresh=refresh
            )
        return self._reachable_base_url

    # -- the pass ----------------------------------------------------------

    async def enhance(self, entry: EnhancedLogbookEntry, conn: AsyncConnection) -> None:
        """Not used: the module runs in the catch-up through :meth:`run_entry`."""
        raise NotImplementedError("image_embedding runs in the catch-up; use run_entry()")

    async def run_entry(
        self,
        entry: EnhancedLogbookEntry,
        repository: ARIELRepository,
        *,
        gate: PictureGate,
    ) -> ImageEntryOutcome:
        """Embed the entry's viewable pictures that have no row in the image table.

        Args:
            entry: The entry whose pictures to embed.
            repository: Repository of the database the module works on.
            gate: Per-picture admission control of the pass.

        Returns:
            ``done`` when the entry was marked complete; ``partial`` when
            pictures are left; ``transient_error`` or ``unavailable`` from a
            failed call; ``module_error`` when the module is not configured.
        """
        target = self._target
        if target is None or self._resolved is None:
            return ImageEntryOutcome.module_error("image_embedding is not configured")
        entry_id = entry["entry_id"]
        work = await self._todo(entry, repository, target)
        if work is None:
            return ImageEntryOutcome.unavailable("config")

        for attachment_id in work:
            if not gate.may_start_picture():
                return ImageEntryOutcome.partial()
            rendition = await repository.get_rendition(attachment_id)
            if rendition is None:  # deleted or re-rendered since the read
                continue
            try:
                vectors = await run_blocking(self._call, rendition, key="image_embedding")
                vector = vectors[0]
            except Exception as exc:
                outcome = failed_call_outcome(exc, gate)
                if outcome is not None:
                    return outcome
                await self._store(repository, target, attachment_id, None, _skip_reason(exc))
                continue
            if await self._store(repository, target, attachment_id, vector, None):
                gate.succeeded()

        if await repository.mark_image_module_complete(entry_id, self.name, target.table):
            return ImageEntryOutcome.done()
        return ImageEntryOutcome.partial()

    async def _todo(
        self,
        entry: EnhancedLogbookEntry,
        repository: ARIELRepository,
        target: ImageEmbeddingTarget,
    ) -> list[str] | None:
        """The viewable pictures with no row in the table, in attachment list order.

        Read with no lock. Returns None when the store has no copy state.
        """
        viewable = await viewable_in_list_order(entry, repository)
        if viewable is None:
            return None
        if not viewable:
            return []
        async with repository.pool.connection() as conn:
            cursor = await conn.execute(
                f"SELECT attachment_id FROM {target.table} WHERE attachment_id = ANY(%(ids)s)",
                {"ids": viewable},
            )
            done = {row[0] for row in await cursor.fetchall()}
        return [a for a in viewable if a not in done]

    def _call(self, rendition: Mapping[str, Any]) -> list[list[float]]:
        """One picture embedding call for one rendition (runs on a daemon thread)."""
        resolved = self._resolved
        target = self._target
        assert resolved is not None and target is not None
        mime = rendition.get("rendition_mime") or rendition.get("mime_type") or "image/png"
        return resolved.instance.execute_image_embedding(
            [(bytes(rendition["rendition_bytes"]), str(mime))],
            model_id=target.model,
            api_key=resolved.api_key,
            base_url=self._base_url(),
            dimensions=target.dims,
            timeout=self.timeout_seconds,
        )

    async def _store(
        self,
        repository: ARIELRepository,
        target: ImageEmbeddingTarget,
        attachment_id: str,
        vector: list[float] | None,
        skip_reason: str | None,
    ) -> bool:
        """Write one picture's row, only while the picture is still viewable.

        One guarded statement: a picture deleted or re-rendered during the call
        matches no row (or violates the foreign key) and nothing is written.

        Returns:
            True when the row was written.
        """
        from psycopg import errors

        literal = vector_literal(vector) if vector is not None else None
        try:
            async with repository.pool.connection() as conn:
                cursor = await conn.execute(
                    f"INSERT INTO {target.table}"
                    " (attachment_id, embedding, skip_reason, model_ref)"
                    " SELECT %(id)s, %(vector)s::vector, %(skip)s, %(model)s"
                    " WHERE EXISTS (SELECT 1 FROM attachment_files f"
                    f" WHERE f.attachment_id = %(id)s AND {viewable_sql('f')})"
                    " ON CONFLICT (attachment_id) DO UPDATE SET"
                    " embedding = EXCLUDED.embedding, skip_reason = EXCLUDED.skip_reason,"
                    " model_ref = EXCLUDED.model_ref, created_at = NOW()",
                    {
                        "id": attachment_id,
                        "vector": literal,
                        "skip": skip_reason,
                        "model": target.model,
                    },
                )
        except errors.ForeignKeyViolation:
            return False
        return bool(cursor.rowcount)
