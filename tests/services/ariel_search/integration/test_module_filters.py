"""Every built-in search tier confines its SQL to the dimension filter (C-FILTER).

Each test calls a module directly against seeded rows on a fresh database and
asserts two things: no row outside the filter comes back, and the service's
post-filter (``EntryFilter.matches``) would remove nothing from what the module
returned, so a broken SQL filter cannot hide behind it (C-MODOUT, H13).

Rows come in pairs that differ only in author -- ``Alice`` inside the filter,
``Bob`` outside -- so a tier that ignores the filter returns Bob too.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import ARIELRepository
from osprey.services.ariel_search.enhancement.status import EnhancementContext, enhancement_context
from osprey.services.ariel_search.filters import EntryFilter
from osprey.services.ariel_search.search.base import ModuleOutput
from osprey.services.ariel_search.search.keyword import keyword_search
from tests.services.ariel_search.integration._support import insert_picture, migrated_repository
from tests.services.ariel_search.test_filters import FILTER_DIMS

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(120)]

MODEL = "nomic-embed-text"
DIM = 768

#: (entry_id, author, raw_text). Each text appears once per author.
ROWS: tuple[tuple[str, str, str], ...] = (
    ("strict-a", "Alice", "beam dump injection kicker misfire"),
    ("strict-b", "Bob", "beam dump injection kicker misfire"),
    ("partial-a", "Alice", "septum water flow alarm"),
    ("partial-b", "Bob", "septum water flow alarm"),
    ("fuzzy-a", "Alice", "quadrupole"),
    ("fuzzy-b", "Bob", "quadrupole"),
)

ALICE = {"author": ["alice"]}


@pytest.fixture
def _registry() -> Iterator[None]:
    """A registry listing the enhancement modules ``FILTER_DIMS`` names."""
    fake = MagicMock()
    fake.list_ariel_enhancement_modules.return_value = ["text_embedding", "image_caption"]
    with patch("osprey.registry.get_registry", return_value=fake):
        yield


@pytest.fixture
def config(scratch_database: str) -> ARIELConfig:
    return ARIELConfig.from_dict(
        {
            "database": {"uri": scratch_database},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": MODEL},
            },
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "models": [{"name": MODEL, "dimension": DIM}],
                },
                "image_caption": {"enabled": True, "model": {"model_id": "m2"}},
            },
        }
    )


def _seed(uri: str, rows: tuple[tuple[str, str, str], ...] = ROWS, **status: Any) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        for entry_id, author, text in rows:
            conn.execute(
                """
                INSERT INTO enhanced_entries (
                    entry_id, source_system, timestamp, author, raw_text, metadata,
                    enhancement_status
                ) VALUES (%s, 'als', '2026-01-05T12:00:00Z', %s, %s, '{}'::jsonb, %s::jsonb)
                """,
                (entry_id, author, text, json.dumps(status.get(entry_id, {}))),
            )


@pytest.fixture
async def repo(
    config: ARIELConfig, scratch_database: str, _registry: None
) -> AsyncIterator[ARIELRepository]:
    async with migrated_repository(config) as repository:
        _seed(scratch_database)
        yield repository


async def _ctx(repo: ARIELRepository, config: ARIELConfig) -> EnhancementContext:
    return enhancement_context(config, has_copy_state=(await repo.schema_facts()).has_copy_state)


async def _assert_confined(
    out: Any, flt: EntryFilter, repo: ARIELRepository, config: ARIELConfig, *, expect: set[str]
) -> ModuleOutput:
    assert isinstance(out, ModuleOutput)
    assert out.filter_applied is True
    ids = {entry["entry_id"] for entry, *_ in out.entries}
    assert ids == expect
    ctx = await _ctx(repo, config)
    removed = [entry["entry_id"] for entry, *_ in out.entries if not flt.matches(entry, ctx)]
    assert removed == []
    return out


async def test_keyword_strict_tier_is_confined(repo: ARIELRepository, config: ARIELConfig) -> None:
    flt = EntryFilter.parse(ALICE, FILTER_DIMS)

    out = await keyword_search("kicker misfire", repo, config, entry_filter=flt, offset=0)

    out = await _assert_confined(out, flt, repo, config, expect={"strict-a"})
    assert out.tier == "strict"


async def test_keyword_partial_tier_is_confined(repo: ARIELRepository, config: ARIELConfig) -> None:
    flt = EntryFilter.parse(ALICE, FILTER_DIMS)

    out = await keyword_search("septum water flow magnet", repo, config, entry_filter=flt, offset=0)

    out = await _assert_confined(out, flt, repo, config, expect={"partial-a"})
    assert out.tier == "partial"


async def test_keyword_fuzzy_tier_is_confined(repo: ARIELRepository, config: ARIELConfig) -> None:
    flt = EntryFilter.parse(ALICE, FILTER_DIMS)

    out = await keyword_search("quadrupolle", repo, config, entry_filter=flt, offset=0)

    out = await _assert_confined(out, flt, repo, config, expect={"fuzzy-a"})
    assert out.tier == "fuzzy"


async def test_legacy_author_still_leaves_fuzzy_unfiltered(
    repo: ARIELRepository, config: ARIELConfig
) -> None:
    """The MCP search path: a legacy author on a query that falls to fuzzy.

    Fuzzy never applied the legacy free-text author, and still does not, so
    the result is the same as without it.
    """
    without = await keyword_search("quadrupolle", repo, config)
    with_author = await keyword_search("quadrupolle", repo, config, author="Alice")

    ids = [entry["entry_id"] for entry, *_ in with_author]
    assert ids == [entry["entry_id"] for entry, *_ in without]
    assert {"fuzzy-a", "fuzzy-b"} <= set(ids)


async def test_keyword_without_a_filter_is_unconfined(
    repo: ARIELRepository, config: ARIELConfig
) -> None:
    out = await keyword_search("kicker misfire", repo, config, offset=0)

    assert {entry["entry_id"] for entry, *_ in out.entries} == {"strict-a", "strict-b"}
    assert out.filter_applied is False


async def test_image_caption_stale_through_keyword_removes_nothing_after(
    repo: ARIELRepository, config: ARIELConfig, scratch_database: str
) -> None:
    """``matches`` sees the ``has_images`` the statement selected, as ``to_sql`` does."""
    stale = {"image_caption": {"status": "complete", "marker": "m1"}}
    rows = (
        ("pic-stale", "Carol", "cryo plant trip report"),
        ("nopic-stale", "Carol", "cryo plant trip report"),
        ("pic-current", "Carol", "cryo plant trip report"),
    )
    _seed(
        scratch_database,
        rows,
        **{
            "pic-stale": stale,
            "nopic-stale": stale,
            "pic-current": {"image_caption": {"status": "complete", "marker": "m2"}},
        },
    )
    insert_picture(scratch_database, "a-pic-stale", "pic-stale", copy_status="copied")
    insert_picture(scratch_database, "a-pic-current", "pic-current", copy_status="copied")
    flt = EntryFilter.parse({"enhancement": ["image_caption:stale"]}, FILTER_DIMS)

    out = await keyword_search("cryo plant trip", repo, config, entry_filter=flt, offset=0)

    out = await _assert_confined(out, flt, repo, config, expect={"pic-stale"})
    ((entry, *_),) = out.entries
    assert entry["has_images"] is True


async def test_semantic_is_confined(
    repo: ARIELRepository, config: ARIELConfig, scratch_database: str, search_calls: Any
) -> None:
    from osprey.services.ariel_search.database.migrations import model_to_table_name
    from osprey.services.ariel_search.search.semantic import semantic_search
    from tests.services.ariel_search.fake_providers import make_fake_embedding_provider

    with psycopg.connect(scratch_database, autocommit=True) as conn:
        table = conn.execute(
            "SELECT to_regclass(%s) IS NOT NULL", (model_to_table_name(MODEL),)
        ).fetchone()
    if not table or not table[0]:
        pytest.skip("this server has no pgvector; the embedding table was not created")

    vector = [1.0] + [0.0] * (DIM - 1)
    for entry_id, *_ in ROWS:
        await repo.store_text_embedding(entry_id, vector, MODEL)

    embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
    embedder.execute_embedding = MagicMock(return_value=[vector])
    flt = EntryFilter.parse(ALICE, FILTER_DIMS)

    out = await semantic_search(
        "beam",
        repo,
        config,
        embedder,
        provider_config={"base_url": "http://localhost:11434", "api_key": None},
        max_results=50,
        entry_filter=flt,
        offset=0,
        search_calls=search_calls,
    )

    await _assert_confined(out, flt, repo, config, expect={"strict-a", "partial-a", "fuzzy-a"})
