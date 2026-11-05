"""Keyset paging of the entry list on real Postgres (C-PAGE).

``search_by_time_range`` orders ``timestamp DESC, entry_id DESC`` and, given an
``after_cursor``, continues strictly after that entry. Entries that share one
timestamp straddle a page boundary here: each must be listed exactly once, and
the page that empties the list yields no cursor. The legacy ``author`` kwarg
stays a literal, case-sensitive match next to the filter object's case-blind one
(C-FILTER H2), on both the list and its count.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from osprey.services.ariel_search.filters import (
    BUILTIN_DIMENSIONS,
    EntryFilter,
    FilterValidationError,
    encode_cursor,
)
from tests.services.ariel_search.integration._support import insert_picture

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(120)]

TIE = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)

#: (entry_id, timestamp, author): four entries share ``TIE``, so a page of three
#: always cuts through the tie.
ROWS: list[tuple[str, datetime, str]] = [
    ("n-newest", TIE + timedelta(hours=1), "Operator"),
    ("t-a", TIE, "Operator"),
    ("t-b", TIE, "J. Smith"),
    ("t-c", TIE, "Operator"),
    ("t-d", TIE, "J. Smith"),
    ("o-older", TIE - timedelta(hours=1), "Operator"),
    ("o-oldest", TIE - timedelta(days=1), "J. Smith"),
]

#: The listing order: timestamp DESC, then entry_id DESC inside the tie.
EXPECTED_ORDER = ["n-newest", "t-d", "t-c", "t-b", "t-a", "o-older", "o-oldest"]


def _seed(uri: str) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        for entry_id, ts, author in ROWS:
            conn.execute(
                """
                INSERT INTO enhanced_entries (
                    entry_id, source_system, timestamp, author, raw_text, metadata
                ) VALUES (%s, 'als_logbook', %s, %s, 'text', '{}'::jsonb)
                """,
                (entry_id, ts, author),
            )


async def _pages(repo: Any, page_size: int, **kwargs: Any) -> list[tuple[list[str], str | None]]:
    """Walk the list a page at a time the way the route does: one surplus row decides."""
    pages: list[tuple[list[str], str | None]] = []
    cursor: str | None = None
    while True:
        rows = await repo.search_by_time_range(limit=page_size + 1, after_cursor=cursor, **kwargs)
        page = rows[:page_size]
        cursor = (
            encode_cursor(page[-1]["timestamp"], page[-1]["entry_id"])
            if len(rows) > page_size
            else None
        )
        pages.append(([e["entry_id"] for e in page], cursor))
        if cursor is None:
            return pages
        assert len(pages) < 20, "paging never terminated"


async def test_entries_sharing_a_timestamp_are_each_listed_once(
    scratch_repo: Any, scratch_database: str
) -> None:
    _seed(scratch_database)

    pages = await _pages(scratch_repo, page_size=3)

    listed = [entry_id for ids, _ in pages for entry_id in ids]
    assert listed == EXPECTED_ORDER
    assert [ids for ids, _ in pages] == [EXPECTED_ORDER[0:3], EXPECTED_ORDER[3:6], ["o-oldest"]]
    assert pages[-1][1] is None
    assert all(cursor is not None for _, cursor in pages[:-1])


async def test_a_page_ending_exactly_at_the_end_yields_no_cursor(
    scratch_repo: Any, scratch_database: str
) -> None:
    _seed(scratch_database)

    pages = await _pages(scratch_repo, page_size=len(ROWS))

    assert pages == [(EXPECTED_ORDER, None)]


async def test_keyset_paging_composes_with_the_filter(
    scratch_repo: Any, scratch_database: str
) -> None:
    _seed(scratch_database)
    flt = EntryFilter.parse({"author": ["j. smith"]}, BUILTIN_DIMENSIONS)

    pages = await _pages(scratch_repo, page_size=1, entry_filter=flt)

    assert [ids for ids, _ in pages] == [["t-d"], ["t-b"], ["o-oldest"]]
    assert pages[-1][1] is None


async def test_legacy_author_is_case_sensitive_the_filter_is_not(
    scratch_repo: Any, scratch_database: str
) -> None:
    """Legacy ``author='j. smith'`` misses 'J. Smith'; filter ``author=['j. smith']`` finds it."""
    _seed(scratch_database)
    flt = EntryFilter.parse({"author": ["j. smith"]}, BUILTIN_DIMENSIONS)
    smiths = ["t-d", "t-b", "o-oldest"]

    assert await scratch_repo.search_by_time_range(author="j. smith") == []
    assert await scratch_repo.count_entries(author="j. smith") == 0

    legacy_exact = await scratch_repo.search_by_time_range(author="J. Smith")
    assert [e["entry_id"] for e in legacy_exact] == smiths

    filtered = await scratch_repo.search_by_time_range(entry_filter=flt)
    assert [e["entry_id"] for e in filtered] == smiths
    assert await scratch_repo.count_entries(entry_filter=flt) == 3

    # both: the legacy literal ANDed with the filter
    assert await scratch_repo.search_by_time_range(author="j. smith", entry_filter=flt) == []
    assert await scratch_repo.count_entries(author="J. Smith", entry_filter=flt) == 3


async def test_the_list_carries_has_images(scratch_repo: Any, scratch_database: str) -> None:
    _seed(scratch_database)
    insert_picture(scratch_database, "a-t-b", "t-b", copy_status="copied")

    rows = await scratch_repo.search_by_time_range()

    flags = {e["entry_id"]: e.get("has_images") for e in rows}
    assert flags["t-b"] is True
    assert {k for k, v in flags.items() if v is not True} == set(EXPECTED_ORDER) - {"t-b"}


async def test_a_bad_cursor_is_a_cursor_validation_error(scratch_repo: Any) -> None:
    with pytest.raises(FilterValidationError) as excinfo:
        await scratch_repo.search_by_time_range(after_cursor="not-a-cursor")

    assert excinfo.value.field == "cursor"
