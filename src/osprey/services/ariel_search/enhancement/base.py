"""ARIEL enhancement module base class.

This module provides the abstract base class for enhancement modules.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from osprey.services.ariel_search.database.migration import BaseMigration
    from osprey.services.ariel_search.models import EnhancedLogbookEntry


class BaseEnhancementModule(ABC):
    """Abstract base class for enhancement modules.

    Enhancement modules enrich logbook entries during ingestion.
    They run sequentially as a pipeline, each adding data to the entry.
    """

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
        """

    async def health_check(self) -> tuple[bool, str]:
        """Check if module is ready.

        Returns:
            Tuple of (healthy, message)
        """
        return (True, "OK")
