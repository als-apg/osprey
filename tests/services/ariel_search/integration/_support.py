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
    from osprey.services.qmd import QMDSearchResult


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


def is_ollama_available() -> bool:
    """Check if Ollama is available for tests."""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


class FakeQMDClient:
    """qmd faked at its client boundary: one hit, the entry's mirror file."""

    is_configured = True
    base_url = "http://127.0.0.1:8180"

    def __init__(self, mirror_file: str, title: str, text: str) -> None:
        self.mirror_file = mirror_file
        self.title = title
        self.text = text
        self.queries = 0

    def is_available(self) -> bool:
        return True

    def query(self, collection: str | None, _text: str, **kwargs: Any) -> list[QMDSearchResult]:
        from osprey.services.qmd import QMDSearchResult

        self.queries += 1
        hits = [
            QMDSearchResult(
                docid="#000000",
                file=self.mirror_file,
                collection=collection or "ariel",
                title=self.title,
                score=0.9,
                line=1,
                snippet=f"1: {self.text}",
            )
        ]
        return hits[: kwargs.get("limit", len(hits))]


def insert_picture(uri: str, attachment_id: str, entry_id: str, *, copy_status: str) -> None:
    """One PNG attachment row; a copied one carries a rendition, so it is viewable."""
    import psycopg

    viewable = copy_status == "copied"
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO attachment_files (
                attachment_id, entry_id, filename, mime_type, source_url,
                copy_status, rendition_sha256
            ) VALUES (%s, %s, 'f.png', 'image/png', 'https://h.example/f.png', %s, %s)
            """,
            (attachment_id, entry_id, copy_status, "ab" * 32 if viewable else None),
        )


def enhancement_status(uri: str, entry_id: str) -> dict[str, Any]:
    """The entry's ``enhancement_status`` as stored."""
    import psycopg

    with psycopg.connect(uri) as conn:
        row = conn.execute(
            "SELECT enhancement_status FROM enhanced_entries WHERE entry_id = %s", (entry_id,)
        ).fetchone()
    assert row is not None
    return row[0]
