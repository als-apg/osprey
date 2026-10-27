"""Plain helpers the ARIEL integration modules share: scratch-store setup and render fakes."""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository


#: The rendition digest ``fake_prepare`` stamps on every picture it renders.
FAKE_RENDITION_SHA = "cd" * 32


@contextlib.asynccontextmanager
async def migrated_repository(config: ARIELConfig) -> AsyncIterator[ARIELRepository]:
    """An ``ARIELRepository`` on *config*'s database, brought to today's schema."""
    from osprey.services.ariel_search.database import ARIELRepository
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    async with await create_connection_pool(config.database) as pool:
        await run_migrations(pool, config)
        yield ARIELRepository(pool, config)


async def migrate(config_dict: dict[str, Any]) -> None:
    """Bring the scratch database to today's schema, as ``osprey ariel migrate`` does."""
    from osprey.services.ariel_search.config import ARIELConfig

    async with migrated_repository(ARIELConfig.from_dict(json.loads(json.dumps(config_dict)))):
        pass


def fake_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render every sniffed picture instantly; non-images answer from the sniff."""
    from osprey.imaging.formats import sniff
    from osprey.services.ariel_search.attachments import prepare as prepare_mod

    async def _prepare(data, **_kwargs):
        sniffed = sniff(data)
        if not sniffed.is_image:
            return prepare_mod.PreparedPicture(sniffed.mime, sniffed.skip_reason)
        return prepare_mod.PreparedPicture(
            mime_type=sniffed.mime,
            skip_reason=None,
            rendition_bytes=b"r",
            rendition_mime="image/png",
            rendition_w=1,
            rendition_h=1,
            rendition_sha256=FAKE_RENDITION_SHA,
        )

    monkeypatch.setattr(prepare_mod, "prepare_picture", _prepare)


def fake_prepare_as_own_rendition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every fetched picture is its own rendition; no render worker starts."""
    from osprey.services.ariel_search.attachments import prepare as prepare_mod

    async def _prepare(data, **_kwargs):
        return prepare_mod.PreparedPicture(
            mime_type="image/png",
            skip_reason=None,
            rendition_bytes=bytes(data),
            rendition_mime="image/png",
            rendition_w=1,
            rendition_h=1,
            rendition_sha256=hashlib.sha256(data).hexdigest(),
        )

    monkeypatch.setattr(prepare_mod, "prepare_picture", _prepare)
