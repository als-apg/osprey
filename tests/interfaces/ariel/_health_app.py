"""The real ARIEL panel app over a stand-in store, for the ``/health`` tests.

The app is the one ``create_app`` builds — its real lifespan, its real
``ARIELSearchService`` and ``ARIELConfig``, and the sign-in gate
``configure_interface_app`` installs. Only the database is stood in, by a
repository answering the two status queries the open page makes.

The config names a store whose DSN carries a password and a host. Neither may
appear on the open page, so the tests that read it assert their absence.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import yaml

from osprey.services.ariel_search.exceptions import DatabaseQueryError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime
    from pathlib import Path

    from fastapi import FastAPI

STORE_PASSWORD = "s3cret-pw"
STORE_HOST = "db.internal.example"
STORE_DSN = f"postgresql://ariel:{STORE_PASSWORD}@{STORE_HOST}:5432/ariel"

#: The ``ariel`` section every app here is built from: one search module and one
#: enhancement module enabled, so the module facts are not empty.
ARIEL_SECTION: dict[str, Any] = {
    "database": {"uri": STORE_DSN},
    "search_modules": {"keyword": {"enabled": True}},
    "enhancement_modules": {
        "text_embedding": {
            "enabled": True,
            "provider": "ollama",
            "models": [{"name": "nomic-embed-text", "dimension": 768}],
        }
    },
}


class StoreDouble:
    """Answers the two status queries ``/health`` makes, or fails them.

    A failure is raised the way ``ARIELRepository`` raises one, with the
    driver's text — which names the store's host — in the message.
    """

    def __init__(
        self,
        *,
        entry_count: int = 48291,
        last_ingestion: datetime | None = None,
        failing: bool = False,
    ) -> None:
        self.entry_count = entry_count
        self.last_ingestion = last_ingestion
        self.failing = failing

    def _refuse(self, what: str) -> None:
        if self.failing:
            raise DatabaseQueryError(
                f'Failed to {what}: connection to server at "{STORE_HOST}", port 5432 '
                f'failed: password authentication failed for user "ariel"',
            )

    async def count_entries(self, **_filters: Any) -> int:
        self._refuse("count entries")
        return self.entry_count

    async def get_last_ingestion(self) -> datetime | None:
        self._refuse("get last ingestion")
        return self.last_ingestion


class _PoolDouble:
    """The service closes its pool on shutdown; nothing else touches it."""

    async def close(self) -> None:
        return None


@asynccontextmanager
async def ariel_app(
    tmp_path: Path, store: StoreDouble | None, *, section: dict[str, Any] | None = None
) -> AsyncIterator[FastAPI]:
    """Build the real panel app over ``store`` and run its lifespan.

    ``store=None`` makes the service construction fail, which is the panel
    running without its search service. ``section`` replaces
    :data:`ARIEL_SECTION` as the config's ``ariel`` block.
    """
    from osprey.interfaces.ariel import create_app
    from osprey.services.ariel_search.service import ARIELSearchService

    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"ariel": ARIEL_SECTION if section is None else section}))

    async def create_service(config: Any) -> ARIELSearchService:
        if store is None:
            raise RuntimeError(f'connection to server at "{STORE_HOST}" refused')
        return ARIELSearchService(config=config, pool=_PoolDouble(), repository=store)

    with patch("osprey.services.ariel_search.create_ariel_service", create_service):
        app = create_app(config_file)
        async with app.router.lifespan_context(app):
            yield app
