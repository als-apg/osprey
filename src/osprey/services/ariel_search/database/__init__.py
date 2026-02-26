"""ARIEL database layer.

This module provides database connectivity, migrations, and repository
for the ARIEL search service.

Note: Database functionality requires psycopg[pool] to be installed.
Functions that require the database will raise ImportError if not available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.core_migration import CoreMigration
from osprey.services.ariel_search.database.migration import (
    BaseMigration,
    model_to_table_name,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search.database.connection import (
        create_connection_pool as create_connection_pool,
    )
    from osprey.services.ariel_search.database.migrate import (
        KNOWN_MIGRATIONS as KNOWN_MIGRATIONS,
    )
    from osprey.services.ariel_search.database.migrate import (
        MigrationRunner as MigrationRunner,
    )
    from osprey.services.ariel_search.database.migrate import (
        run_migrations as run_migrations,
    )
    from osprey.services.ariel_search.database.repository import (
        ARIELRepository as ARIELRepository,
    )
    from osprey.services.ariel_search.database.repository import (
        requires_module as requires_module,
    )

__all__ = [
    "create_connection_pool",
    "BaseMigration",
    "CoreMigration",
    "KNOWN_MIGRATIONS",
    "MigrationRunner",
    "model_to_table_name",
    "run_migrations",
    "ARIELRepository",
    "requires_module",
]


def __getattr__(name: str):
    """Lazy load database-dependent modules."""
    if name == "create_connection_pool":
        from osprey.services.ariel_search.database.connection import (
            create_connection_pool,
        )

        return create_connection_pool

    if name in ("KNOWN_MIGRATIONS", "MigrationRunner", "run_migrations"):
        from osprey.services.ariel_search.database.migrate import (
            KNOWN_MIGRATIONS,
            MigrationRunner,
            run_migrations,
        )

        return {
            "KNOWN_MIGRATIONS": KNOWN_MIGRATIONS,
            "MigrationRunner": MigrationRunner,
            "run_migrations": run_migrations,
        }[name]

    if name in ("ARIELRepository", "requires_module"):
        from osprey.services.ariel_search.database.repository import (
            ARIELRepository,
            requires_module,
        )

        return {"ARIELRepository": ARIELRepository, "requires_module": requires_module}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
