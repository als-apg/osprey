"""ARIEL enhancement module base class.

This module provides the abstract base class for enhancement modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from osprey.models.providers.health import HealthResult

__all__ = [
    "UNAVAILABLE_REASONS",
    "BaseEnhancementModule",
    "HealthResult",
    "ImageEntryOutcome",
    "ImageOutcomeKind",
    "PictureGate",
    "as_health_result",
]

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from osprey.services.ariel_search.database.migrations import BaseMigration
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry


class PictureGate(Protocol):
    """Per-picture admission control the catch-up hands to a picture module.

    A ``runs_inline=False`` module consults the gate around every picture of an
    entry inside :meth:`BaseEnhancementModule.run_entry`.
    """

    def may_start_picture(self) -> bool:
        """Return whether the module may start the next picture.

        Checked before each picture. False means stop now; the entry then
        returns ``partial`` unless every picture was already handled.
        """
        ...

    def succeeded(self) -> None:
        """Report one picture whose call succeeded and whose result was stored."""
        ...

    def deterministic(self, signature: str) -> bool:
        """Ask whether a deterministic per-picture failure may be written.

        Args:
            signature: Stable signature of the failure (same input, same failure).

        Returns:
            True to write the failure for that picture; False to write nothing
            for it, in which case the entry returns ``partial``.
        """
        ...


ImageOutcomeKind = Literal["done", "partial", "transient_error", "unavailable", "module_error"]

UNAVAILABLE_REASONS: frozenset[str] = frozenset({"unreachable", "auth", "model", "config"})


@dataclass(frozen=True)
class ImageEntryOutcome:
    """Result of one :meth:`BaseEnhancementModule.run_entry` call.

    The driver of the catch-up maps each kind to exactly one action:

    ==================== =======================================================
    kind                 driver action
    ==================== =======================================================
    ``done``             No write; the module marked the entry complete itself.
    ``partial``          No write; the entry stays eligible for a later pass.
    ``transient_error``  ``mark_enhancement_failed(entry, module, reason,
                         marker=current)``, subject to the attempts rule.
    ``unavailable``      Reason ``unreachable``: re-run the shared pre-pass
                         check under 5 s; unhealthy ends the pass with no write
                         and the entry uncharged, healthy is handled as
                         ``transient_error``. Reason ``auth``, ``model`` or
                         ``config``: the call is authoritative, so no write,
                         the entry is uncharged and the pass ends.
    ``module_error``     The module's own refusal: no write, the pass ends.
    ==================== =======================================================

    Every ``unavailable`` and ``module_error`` reason is reported to the
    per-process availability tracker. The pass breakers apply after this
    table, so a ``transient_error`` converted from ``unavailable('unreachable')``
    counts like any other.

    Attributes:
        kind: Outcome kind.
        reason: Reason text; required for ``transient_error``, ``unavailable``
            and ``module_error``, absent for ``done`` and ``partial``.
    """

    kind: ImageOutcomeKind
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind in ("done", "partial"):
            if self.reason is not None:
                raise ValueError(f"{self.kind} carries no reason")
        elif self.kind in ("transient_error", "unavailable", "module_error"):
            if not self.reason:
                raise ValueError(f"{self.kind} requires a reason")
            if self.kind == "unavailable" and self.reason not in UNAVAILABLE_REASONS:
                raise ValueError(
                    f"unavailable reason must be one of {sorted(UNAVAILABLE_REASONS)}, "
                    f"got {self.reason!r}"
                )
        else:
            raise ValueError(f"unknown outcome kind {self.kind!r}")

    @classmethod
    def done(cls) -> "ImageEntryOutcome":
        """Every picture handled and the entry marked complete by the module."""
        return cls("done")

    @classmethod
    def partial(cls) -> "ImageEntryOutcome":
        """Some pictures left for a later pass."""
        return cls("partial")

    @classmethod
    def transient_error(cls, reason: str) -> "ImageEntryOutcome":
        """A failure worth retrying on a later pass."""
        return cls("transient_error", reason)

    @classmethod
    def unavailable(cls, reason: str) -> "ImageEntryOutcome":
        """The model service is unavailable (``unreachable|auth|model|config``)."""
        return cls("unavailable", reason)

    @classmethod
    def module_error(cls, reason: str) -> "ImageEntryOutcome":
        """The module refused to proceed."""
        return cls("module_error", reason)


def as_health_result(result: "HealthResult | tuple[bool, str]") -> HealthResult:
    """Return a module's health answer as a :class:`HealthResult`.

    A ``HealthResult`` passes through unchanged; a plain ``(healthy, message)``
    pair, the older module contract, becomes ``HealthResult(healthy, message,
    None)``.

    Args:
        result: What a module's ``health_check()`` returned.

    Returns:
        The verdict as a ``HealthResult``.
    """
    if isinstance(result, HealthResult):
        return result
    return HealthResult(bool(result[0]), str(result[1]), None)


class BaseEnhancementModule(ABC):
    """Abstract base class for enhancement modules.

    Enhancement modules enrich logbook entries during ingestion.
    They run sequentially as a pipeline, each adding data to the entry.

    Attributes:
        runs_inline: Whether the module runs inside the ingest pipeline. A module
            that sets it to False runs only in the catch-up, never during a poll.
    """

    runs_inline: bool = True

    @property
    @abstractmethod
    def name(self) -> str:
        """Return module identifier.

        Returns:
            Module name (e.g., 'text_embedding', 'semantic_processor')
        """

    @property
    def migration(self) -> "type[BaseMigration] | None":
        """Return migration class for this module.

        Override in subclasses that need database migrations.

        Returns:
            Migration class or None if no migration needed
        """
        return None

    def configure(self, config: dict[str, Any]) -> None:  # noqa: B027
        """Configure the module with settings from config.yml.

        Called by create_enhancers_from_config() after instantiation.
        Override in subclasses that accept configuration.

        Args:
            config: Module-specific configuration dict
        """

    async def run_entry(
        self,
        entry: "EnhancedLogbookEntry",
        repository: "ARIELRepository",
        *,
        gate: PictureGate,
    ) -> ImageEntryOutcome:
        """Process one entry's pictures for a ``runs_inline=False`` module.

        The module owns its three-phase write (claim, call, store) and its
        guarded completion mark; the catch-up driver only applies the outcome
        table documented on :class:`ImageEntryOutcome`. Consult ``gate``
        before each picture, after each stored success, and before writing a
        deterministic per-picture failure.

        Args:
            entry: The entry whose pictures to process.
            repository: Repository for the module's own reads and writes.
            gate: Per-picture admission control.

        Returns:
            The entry's outcome.

        Raises:
            NotImplementedError: For inline modules, which run through
                :meth:`enhance` instead.
        """
        raise NotImplementedError(f"{self.name} runs inline; use enhance()")

    @abstractmethod
    async def enhance(
        self,
        entry: "EnhancedLogbookEntry",
        conn: "AsyncConnection",
    ) -> None:
        """Enhance an entry and store results.

        Args:
            entry: The entry to enhance
            conn: Database connection from pool

        The module should:
        1. Extract relevant data from entry
        2. Process using configured model/algorithm
        3. Store results to appropriate table/column

        A ``runs_inline=False`` module raises NotImplementedError here; the
        catch-up drives it through :meth:`run_entry`.
        """

    def required_relations(self) -> list[str]:
        """Return the database relations the module needs before it can run.

        The catch-up's pre-pass check looks each one up and skips the module
        when one is missing, before calling :meth:`health_check`.

        Returns:
            Relation names; empty by default.
        """
        return []

    def completion_marker(self) -> str | None:
        """Return the marker a ``runs_inline=False`` module completes entries under.

        A change of marker (another caption model, another vector table) makes
        every entry incomplete again for the module.

        Returns:
            The current marker, or None for a module that has none.
        """
        return None

    async def health_check(self) -> HealthResult:
        """Check whether the module can do its work now.

        The default answers ``HealthResult(None, "no health check", None)`` so
        a module with no check of its own reads as unchecked, never as healthy.
        An override classifies its own failures: ``reason`` is set whenever
        ``reachable`` is False. Blocking work belongs on
        :func:`~osprey.services.ariel_search.enhancement._offload.run_blocking`,
        so a caller's timeout can always return.

        Returns:
            The verdict.
        """
        return HealthResult(None, "no health check", None)

    def health_reason(self) -> str | None:
        """Return a reason that overrides the verdict of :meth:`health_check`.

        ``osprey ariel status`` reads it after the check: a non-None value
        (for example ``no_reader``) becomes the module's reported reason,
        while ``reachable`` still carries the check's own answer.

        Returns:
            The overriding reason, or None (the default) to keep the check's.
        """
        return None
