"""Search totals and paging against real PostgreSQL (C-MODOUT).

The module's ``total`` must equal ``COUNT(*)`` over the same filter and tier,
capped at 1000, and rows that tie on rank or similarity must still page
deterministically: the ``e.entry_id DESC`` tiebreak keeps page 1 and page 2
disjoint and, together, covering every match.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import ARIELRepository
from osprey.services.ariel_search.filters import EntryFilter
from osprey.services.ariel_search.search.base import ModuleOutput
from osprey.services.ariel_search.search.keyword import keyword_search
from tests.services.ariel_search.integration._support import migrated_repository
from tests.services.ariel_search.test_filters import FILTER_DIMS

pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(120)]

MODEL = "nomic-embed-text"
DIM = 768
PAGE = 10

#: Identical text, so every row ties on ts_rank and on trigram similarity.
TIED = "orbit feedback trip"
#: 24 tied rows, half by Alice: each author holds PAGE + 2 of them.
TIED_ROWS = [(f"tied-{i:02d}", "Alice" if i % 2 else "Bob", TIED) for i in range(24)]
FUZZY_ROWS = [(f"fuzzy-{i:02d}", "Alice", "quadrupole") for i in range(PAGE + 2)]


@pytest.fixture
def _registry() -> Iterator[None]:
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
                "text_embedding": {"enabled": True, "models": [{"name": MODEL, "dimension": DIM}]},
            },
        }
    )


def _seed(uri: str, rows: list[tuple[str, str, str]]) -> None:
    with psycopg.connect(uri, autocommit=True) as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, author, raw_text, metadata
            ) VALUES (%s, 'als', '2026-01-05T12:00:00Z', %s, %s, '{}'::jsonb)
            """,
            rows,
        )


def _count(uri: str, where: str, params: tuple[Any, ...] = ()) -> int:
    with psycopg.connect(uri, autocommit=True) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM enhanced_entries e WHERE {where}", params)
        return int(row.fetchone()[0])


@pytest.fixture
async def repo(
    config: ARIELConfig, scratch_database: str, _registry: None
) -> AsyncIterator[ARIELRepository]:
    async with migrated_repository(config) as repository:
        _seed(scratch_database, TIED_ROWS + FUZZY_ROWS)
        yield repository


async def _pages(fetch: Any, n: int) -> list[ModuleOutput]:
    pages = []
    for page in range(n):
        out = await fetch(page * PAGE)
        assert isinstance(out, ModuleOutput)
        pages.append(out)
    return pages


def _ids(out: ModuleOutput) -> list[str]:
    return [row[0]["entry_id"] for row in out.entries]


async def test_strict_total_matches_count_and_tied_pages_cover_every_row(
    repo: ARIELRepository, config: ARIELConfig, scratch_database: str
) -> None:
    async def fetch(offset: int) -> Any:
        return await keyword_search("orbit feedback", repo, config, max_results=PAGE, offset=offset)

    pages = await _pages(fetch, 3)

    expected = _count(scratch_database, "raw_text = %s", (TIED,))
    assert expected == 24
    assert [(p.tier, p.total, p.total_exact) for p in pages] == [("strict", 24, True)] * 3
    ids = [_ids(p) for p in pages]
    assert not set(ids[0]) & set(ids[1])
    assert sorted(i for page in ids for i in page) == sorted(e for e, *_ in TIED_ROWS)
    assert [p.has_more for p in pages] == [True, True, False]
    # The tiebreak is entry_id DESC.
    assert ids[0] == sorted((e for e, *_ in TIED_ROWS), reverse=True)[:PAGE]


async def test_filtered_total_matches_count_under_the_filter(
    repo: ARIELRepository, config: ARIELConfig, scratch_database: str
) -> None:
    flt = EntryFilter.parse({"author": ["alice"]}, FILTER_DIMS)

    async def fetch(offset: int) -> Any:
        return await keyword_search(
            "orbit feedback", repo, config, max_results=PAGE, offset=offset, entry_filter=flt
        )

    page1, page2 = await _pages(fetch, 2)

    expected = _count(scratch_database, "raw_text = %s AND author = 'Alice'", (TIED,))
    assert expected == PAGE + 2
    assert (page1.total, page1.total_exact, page1.has_more) == (expected, True, True)
    assert (page2.total, page2.has_more) == (expected, False)
    assert not set(_ids(page1)) & set(_ids(page2))
    assert len(_ids(page1)) + len(_ids(page2)) == expected


async def test_fuzzy_total_matches_count_and_is_inexact(
    repo: ARIELRepository, config: ARIELConfig, scratch_database: str
) -> None:
    async def fetch(offset: int) -> Any:
        return await keyword_search("quadrupolx", repo, config, max_results=PAGE, offset=offset)

    page1, page2 = await _pages(fetch, 2)

    expected = _count(scratch_database, "raw_text = 'quadrupole'")
    assert (page1.tier, page2.tier) == ("fuzzy", "fuzzy")
    assert (page1.total, page1.total_exact, page1.has_more) == (expected, False, True)
    assert not set(_ids(page1)) & set(_ids(page2))
    assert sorted(_ids(page1) + _ids(page2)) == sorted(e for e, *_ in FUZZY_ROWS)


async def test_total_is_capped_at_1000(
    repo: ARIELRepository, config: ARIELConfig, scratch_database: str
) -> None:
    _seed(scratch_database, [(f"cap-{i:04d}", "Carol", "septum magnet") for i in range(1003)])

    out = await keyword_search("septum magnet", repo, config, max_results=PAGE, offset=0)

    assert _count(scratch_database, "raw_text = 'septum magnet'") == 1003
    assert isinstance(out, ModuleOutput)
    assert (out.total, out.total_exact, out.has_more) == (1000, False, True)


async def test_semantic_total_matches_count_and_tied_pages_cover_every_row(
    repo: ARIELRepository, scratch_database: str
) -> None:
    from osprey.services.ariel_search.database.migrations import model_to_table_name

    with psycopg.connect(scratch_database, autocommit=True) as conn:
        table = conn.execute(
            "SELECT to_regclass(%s) IS NOT NULL", (model_to_table_name(MODEL),)
        ).fetchone()
    if not table or not table[0]:
        pytest.skip("this server has no pgvector; the embedding table was not created")

    vector = [1.0] + [0.0] * (DIM - 1)
    for entry_id, *_ in TIED_ROWS:
        await repo.store_text_embedding(entry_id, vector, MODEL)

    pages = [
        await repo.semantic_search(
            query_embedding=vector, model_name=MODEL, max_results=PAGE, offset=offset, count=True
        )
        for offset in (0, PAGE, 2 * PAGE)
    ]

    assert [(p.total, p.total_exact) for p in pages] == [(len(TIED_ROWS), True)] * 3
    ids = [[entry["entry_id"] for entry, _sim in p] for p in pages]
    assert not set(ids[0]) & set(ids[1])
    assert sorted(i for page in ids for i in page) == sorted(e for e, *_ in TIED_ROWS)
