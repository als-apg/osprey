"""Integration test for entry expansion through the real PostgreSQL schema.

One entry is ingested with ``vocabulary.expand_entries`` on: the embedder sees
the vocabulary terms inserted into the entry's text and stores the expander's
build key beside the vector. Flipping ``expand_entries`` off then makes
``get_status`` report those rows stale for the ``expanded`` cause.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from osprey.models.provider_registry import ProviderRegistry
from osprey.services.ariel_search.cli_operations import get_status, run_migrate
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.connection import create_connection_pool
from osprey.services.ariel_search.database.migrations import model_to_table_name
from osprey.services.ariel_search.database.repository import ARIELRepository
from osprey.services.ariel_search.enhancement.factory import create_enhancers_from_config
from osprey.services.ariel_search.enhancement.text_embedding.embedder import running_input_key
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.ingestion.ingest import ingest_one
from tests.services.ariel_search.fake_providers import make_fake_embedding_provider

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

MODEL = "fake-embed"
DIMENSION = 4
ENTRY_ID = "entry-expansion-bpm"
RAW_TEXT = "BPM readings drifted after the ramp and were rezeroed."
EXPANDED = "BPM beam position monitor readings drifted"


class _StubAdapter(FacilityAdapter):
    """A source whose single entry the test hands to ``ingest_one`` directly."""

    @property
    def source_system_name(self) -> str:
        return "Stub"

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        for entry in ():  # never yields
            yield entry


def _config_dict(database_uri: str, vocabulary_path: Path, *, expand: bool) -> dict[str, Any]:
    """The raw ``ariel`` section: text embedding on, semantic search off."""
    return {
        "database": {"uri": database_uri},
        "search_modules": {"keyword": {"enabled": True}, "semantic": {"enabled": False}},
        "enhancement_modules": {
            "semantic_processor": {"enabled": False},
            "text_embedding": {
                "enabled": True,
                "provider": {"name": "fake", "base_url": "http://fake-embeddings.invalid:11434"},
                "models": [{"name": MODEL, "dimension": DIMENSION}],
            },
        },
        "vocabulary": {"enabled": True, "path": str(vocabulary_path), "expand_entries": expand},
    }


def _entry() -> dict[str, Any]:
    return {
        "entry_id": ENTRY_ID,
        "source_system": "test",
        "timestamp": datetime.now(UTC),
        "author": "tester",
        "raw_text": RAW_TEXT,
        "attachments": [],
        "metadata": {},
        "enhancement_status": {},
    }


@pytest.fixture
def provider_cls(monkeypatch):
    """Route every provider lookup to one recording fake embedding provider."""
    fake = make_fake_embedding_provider(dimension=DIMENSION)

    def get_provider(_self, _name):
        return fake

    monkeypatch.setattr(ProviderRegistry, "get_provider", get_provider)
    monkeypatch.setattr("osprey.models.config.get_provider_config", lambda name: {"api_key": "k"})
    return fake


async def _has_pgvector(config: ARIELConfig) -> bool:
    pool = await create_connection_pool(config.database)
    try:
        async with pool.connection() as conn:
            result = await conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            return await result.fetchone() is not None
    finally:
        await pool.close()


async def test_expanded_entry_is_embedded_keyed_and_reported_stale_when_expansion_turns_off(
    scratch_database: str, vocabulary_file: Path, provider_cls
):
    config_dict = _config_dict(scratch_database, vocabulary_file, expand=True)
    config = ARIELConfig.from_dict(config_dict)
    if not await _has_pgvector(config):
        pytest.skip("pgvector is not installed on the test server")

    assert await run_migrate(copy.deepcopy(config_dict)) == []

    expander = config.entry_expander()
    assert expander is not None
    table = model_to_table_name(MODEL)

    pool = await create_connection_pool(config.database)
    try:
        repository = ARIELRepository(pool, config)
        enhancers = create_enhancers_from_config(config)
        assert [e.name for e in enhancers] == ["text_embedding"]
        await ingest_one(_entry(), _StubAdapter(config), repository, enhancers, config, None)

        async with pool.connection() as conn:
            result = await conn.execute(f"SELECT entry_id, input_key FROM {table}")
            rows = await result.fetchall()
    finally:
        await pool.close()

    embed_calls = [call for call in provider_cls.calls if "texts" in call]
    assert embed_calls, "the embedder never reached the provider"
    assert EXPANDED in embed_calls[0]["texts"][0]

    assert rows == [(ENTRY_ID, running_input_key(expander))]
    assert expander.key in rows[0][1]

    off = copy.deepcopy(config_dict)
    off["vocabulary"]["expand_entries"] = False
    status = await get_status(off, config_dir=None)

    assert status["status"] != "error", status
    (entry,) = [t for t in status["embedding_tables"] if t["table"] == table]
    assert entry["stale"] == {"expanded": len(rows)}
    assert entry["model"] == {"name": MODEL, "dimension": DIMENSION}
