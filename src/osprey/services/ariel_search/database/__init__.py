"""ARIEL database layer.

This module provides database connectivity, migrations, and repository
for the ARIEL search service.

Note: Database functionality requires psycopg[pool] to be installed.
Functions that require the database will raise ImportError if not available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.database.core_migration import CoreMigration
from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    model_to_table_name,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search.database.connection import (
        create_connection_pool as create_connection_pool,
    )
    from osprey.services.ariel_search.database.migrations import (
        KNOWN_MIGRATIONS as KNOWN_MIGRATIONS,
    )
    from osprey.services.ariel_search.database.migrations import (
        MigrationRunner as MigrationRunner,
    )
    from osprey.services.ariel_search.database.migrations import (
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

#: Public name -> the submodule of this package that defines it. Entries are
#: resolved on first attribute access, never at import.
_LAZY_EXPORTS: dict[str, str] = {
    "create_connection_pool": ".connection",
    "KNOWN_MIGRATIONS": ".migrations",
    "MigrationRunner": ".migrations",
    "run_migrations": ".migrations",
    "ARIELRepository": ".repository",
    "requires_module": ".repository",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
