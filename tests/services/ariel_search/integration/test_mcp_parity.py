"""MCP ``browse`` and ``keyword_search`` keep their pre-filter meaning (C-FILTER).

The dimension filter and keyset paging rewrote the SQL under both tools, but
neither tool takes the filter: every parameter they had before still selects
the same entries. Each case pins the entries one parameter selects on a fixed
corpus, under the legacy semantics: browse's ``author`` and ``source_system``
are exact and case-sensitive and the list is newest first, keyword search's
``author`` is a case-blind partial match, and a bare ``end_date`` covers that
whole day. "J. Smith", "J. Smithson" and "j. smith" are all present, so an
exact match, a partial match and a case-blind match each select a different set.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import psycopg
import pytest

from tests.mcp_server.conftest import get_tool_fn

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.usefixtures("mcp_service"),
    pytest.mark.xdist_group("docker"),
    pytest.mark.timeout(120),
]

#: (entry_id, day of March 2026, author, source_system, raw_text); 18:00 UTC is
#: mid-morning in the facility zone, so no entry sits near a day boundary.
ROWS: tuple[tuple[str, int, str, str, str], ...] = (
    ("p-1", 1, "J. Smith", "als", "kicker misfire at injection"),
    ("p-2", 2, "J. Smithson", "als", "kicker misfire septum"),
    ("p-3", 3, "j. smith", "linac", "kicker timing drift"),
    ("p-4", 4, "Operator", "als", "vacuum burst sector three"),
    ("p-5", 5, "J. Smith", "linac", "kicker misfire again"),
)


def _seed(uri: str) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        for entry_id, day, author, source, text in ROWS:
            conn.execute(
                """
                INSERT INTO enhanced_entries (
                    entry_id, source_system, timestamp, author, raw_text, metadata
                ) VALUES (%s, %s, %s, %s, %s, '{}'::jsonb)
                """,
                (entry_id, source, datetime(2026, 3, day, 18, tzinfo=UTC), author, text),
            )


@pytest.fixture
async def mcp_service(scratch_database: str, monkeypatch, tmp_path) -> AsyncIterator[Any]:
    """The MCP context over a migrated, seeded scratch store with keyword search on."""
    from osprey.mcp_server.ariel.server_context import initialize_ariel_context
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations
    from osprey.services.ariel_search.service import create_ariel_service

    raw = {"database": {"uri": scratch_database}, "search_modules": {"keyword": {"enabled": True}}}
    config = ARIELConfig.from_dict(raw)
    async with await create_connection_pool(config.database) as pool:
        await run_migrations(pool, config)
    _seed(scratch_database)

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(json.dumps({"ariel": raw}))
    initialize_ariel_context()
    service = await create_ariel_service(config)
    try:
        with patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=service),
        ):
            yield service
    finally:
        await service.aclose()


async def _browse(**kwargs: Any) -> dict[str, Any]:
    from osprey.mcp_server.ariel.tools.browse import browse

    return json.loads(await get_tool_fn(browse)(**kwargs))


async def _keyword(**kwargs: Any) -> dict[str, Any]:
    from osprey.mcp_server.ariel.tools.keyword_search import keyword_search

    return json.loads(await get_tool_fn(keyword_search)(**kwargs))


def _ids(document: dict[str, Any]) -> list[str]:
    entries = document.get("entries")
    assert isinstance(entries, list), document
    return [e["entry_id"] for e in entries]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, ["p-5", "p-4", "p-3", "p-2", "p-1"], id="no-arguments"),
        pytest.param({"page_size": 2}, ["p-5", "p-4"], id="page_size"),
        pytest.param({"start_date": "2026-03-03"}, ["p-5", "p-4", "p-3"], id="start_date"),
        pytest.param({"end_date": "2026-03-03"}, ["p-3", "p-2", "p-1"], id="end_date-whole-day"),
        pytest.param({"author": "J. Smith"}, ["p-5", "p-1"], id="author-exact-case-sensitive"),
        pytest.param({"author": "j. smith"}, ["p-3"], id="author-lower-case"),
        pytest.param({"author": "Smith"}, [], id="author-no-partial"),
        pytest.param({"source_system": "linac"}, ["p-5", "p-3"], id="source_system"),
        pytest.param(
            {"author": "J. Smith", "source_system": "als", "end_date": "2026-03-04"},
            ["p-1"],
            id="combined",
        ),
    ],
)
async def test_browse_selects_the_same_entries(kwargs: dict[str, Any], expected: list[str]) -> None:
    document = await _browse(**kwargs)

    assert _ids(document) == expected
    if "page_size" not in kwargs:
        assert document["total_count"] == len(expected)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, {"p-1", "p-2", "p-5"}, id="query-only"),
        pytest.param({"author": "smith"}, {"p-1", "p-2", "p-5"}, id="author-partial-case-blind"),
        pytest.param({"author": "Smithson"}, {"p-2"}, id="author-longer-name"),
        pytest.param({"source_system": "linac"}, {"p-5"}, id="source_system"),
        pytest.param({"start_date": "2026-03-02"}, {"p-2", "p-5"}, id="start_date"),
        pytest.param({"end_date": "2026-03-02"}, {"p-1", "p-2"}, id="end_date-whole-day"),
        pytest.param({"exclude_entry_ids": ["p-1"]}, {"p-2", "p-5"}, id="exclude_entry_ids"),
        pytest.param({"expand_query": False}, {"p-1", "p-2", "p-5"}, id="expand_query"),
    ],
)
async def test_keyword_search_selects_the_same_entries(
    kwargs: dict[str, Any], expected: set[str]
) -> None:
    document = await _keyword(query="kicker misfire", **kwargs)

    assert set(_ids(document)) == expected


async def test_keyword_search_max_results_caps_the_page() -> None:
    ids = _ids(await _keyword(query="kicker misfire", max_results=1))

    assert len(ids) == 1
    assert ids[0] in {"p-1", "p-2", "p-5"}
