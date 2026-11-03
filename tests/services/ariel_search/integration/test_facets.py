"""Facet counts equal the lengths of the lists they select (C-FACETS).

Each test seeds a fresh database, asks ``ARIELSearchService.facets`` for the
counts of one list, then for every counted value runs the list with that value
selected and asserts the count is the list's length. The rows include a stale
picture entry, a stale entry with no picture (never stale, rule 1), a failed
text embedding, mixed-case authors, source systems and ``metadata.tags``, and
no entry-field declarations, so the built-in ``tags`` dimension is the only
multi-valued one.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import psycopg
import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.filters import EntryFilter, filter_dimensions
from osprey.services.ariel_search.service import ARIELSearchService
from tests.services.ariel_search.integration._support import insert_picture, migrated_repository

pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(120)]

MODEL = "nomic-embed-text"
DIM = 768
ZONE = ZoneInfo("America/Los_Angeles")

#: A local day well inside the histogram's 365-day window.
BASE_DAY = datetime.now(ZONE).date() - timedelta(days=10)
#: The day of the late-evening entry, with no other entry on it.
LATE_DAY = datetime.now(ZONE).date() - timedelta(days=5)

STALE = {"image_caption": {"status": "complete", "marker": "m1"}}

#: (entry_id, author, source_system, tags or None, raw_text, enhancement_status)
ROWS: tuple[tuple[str, str, str, list[str] | None, str, dict[str, Any]], ...] = (
    ("f-1", "Alice", "als", ["Beam", "RF"], "kicker misfire beam dump", {}),
    ("f-2", "alice", "als", ["beam"], "kicker misfire septum", {}),
    ("f-3", "Bob", "ALS", ["Vacuum"], "kicker misfire vacuum", {}),
    ("f-4", "Bob", "linac", None, "unrelated cryo note", {}),
    ("f-stale", "Carol", "als", ["RF"], "kicker misfire stale picture", STALE),
    ("f-nopic", "Carol", "linac", "not-a-list", "kicker misfire no picture", STALE),
    (
        "f-failed",
        "Dave",
        "als",
        ["beam"],
        "kicker misfire failed",
        {"text_embedding": {"status": "failed"}},
    ),
    ("f-bpm-long", "Erin", "als", ["Diag"], "beam position monitor drift", {}),
    ("f-bpm-short", "erin", "linac", ["diag"], "bpm readback noise", {}),
)


def _registry() -> MagicMock:
    """A registry with the built-in search modules and the filtered enhancement modules."""
    from osprey.services.ariel_search.search import keyword, semantic

    fake = MagicMock()
    fake.list_ariel_enhancement_modules.return_value = ["text_embedding", "image_caption"]
    hybrid = SimpleNamespace(
        get_tool_descriptor=lambda: dataclasses.replace(
            keyword.get_tool_descriptor(), name="hybrid_search", search_mode="hybrid"
        )
    )
    modules = {"keyword": keyword, "semantic": semantic, "hybrid": hybrid}
    fake.list_ariel_search_modules.return_value = list(modules)
    fake.get_ariel_search_module.side_effect = modules.get
    return fake


@pytest.fixture(autouse=True)
def _environment() -> Iterator[None]:
    """The fake registry, and the facility zone both the histogram and the time filter use."""
    with (
        patch("osprey.registry.get_registry", return_value=_registry()),
        patch("osprey_connectors.config.get_facility_timezone", return_value=ZONE),
    ):
        yield


@pytest.fixture
def config(scratch_database: str, vocabulary_file: Path) -> ARIELConfig:
    return ARIELConfig.from_dict(
        {
            "database": {"uri": scratch_database},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": MODEL},
                # Served by the keyword module under the hybrid name (see _registry).
                "hybrid": {"enabled": True},
            },
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "models": [{"name": MODEL, "dimension": DIM}],
                },
                "image_caption": {"enabled": True, "model": {"model_id": "m2"}},
            },
            "vocabulary": {"enabled": True, "path": str(vocabulary_file)},
        }
    )


def _local(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=ZONE)


def _seed(uri: str) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        for entry_id, author, source, tags, text, status in ROWS:
            metadata = {} if tags is None else {"tags": tags}
            conn.execute(
                """
                INSERT INTO enhanced_entries (
                    entry_id, source_system, timestamp, author, raw_text, metadata,
                    enhancement_status
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                """,
                (
                    entry_id,
                    source,
                    _local(BASE_DAY, 12),
                    author,
                    text,
                    json.dumps(metadata),
                    json.dumps(status),
                ),
            )
    insert_picture(uri, "a-stale", "f-stale", copy_status="copied")


@pytest.fixture
async def service(config: ARIELConfig, scratch_database: str) -> AsyncIterator[ARIELSearchService]:
    async with migrated_repository(config) as repo:
        _seed(scratch_database)
        svc = ARIELSearchService(config=config, pool=repo.pool, repository=repo)
        try:
            yield svc
        finally:
            await svc.aclose()


def _parse(config: ARIELConfig, raw: dict[str, Any]) -> EntryFilter:
    return EntryFilter.parse(raw, filter_dimensions(config))


async def _list_length(
    service: ARIELSearchService,
    query: str,
    mode: str | None,
    flt: EntryFilter,
    advanced: dict[str, Any],
) -> int:
    if not query:
        return len(await service.repository.search_by_time_range(limit=1000, entry_filter=flt))
    result = await service.search(
        query, mode=mode, max_results=100, entry_filter=flt, advanced_params=dict(advanced)
    )
    assert len(result.entries) < 100
    return len(result.entries)


async def _assert_counts_are_list_lengths(
    service: ARIELSearchService,
    query: str,
    mode: str | None,
    base: dict[str, Any],
    advanced: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every counted value's count is the length of the list selecting it."""
    advanced = advanced or {}
    facets = await service.facets(query, mode, _parse(service.config, base), dict(advanced))
    assert facets["exact"] is True
    checked = 0
    for dim in facets["dimensions"]:
        for item in dim["values"]:
            selected = _parse(service.config, {**base, dim["name"]: [item["value"]]})
            length = await _list_length(service, query, mode, selected, advanced)
            assert item["count"] == length, (dim["name"], item)
            checked += 1
    assert checked > 0
    return facets


def _values(facets: dict[str, Any], name: str) -> dict[str, int]:
    """One dimension's counts, keyed lower-case: the representative's case is collation's."""
    (dim,) = [d for d in facets["dimensions"] if d["name"] == name]
    return {item["value"].lower(): item["count"] for item in dim["values"]}


async def test_browse_facet_counts_equal_list_lengths(service: ARIELSearchService) -> None:
    facets = await _assert_counts_are_list_lengths(service, "", None, {})

    assert facets["tier"] is None
    assert _values(facets, "author") == {"alice": 2, "bob": 2, "carol": 2, "dave": 1, "erin": 2}
    assert _values(facets, "source_system") == {"als": 6, "linac": 3}
    # Only the picture entry is stale; the stale marker without a picture is not.
    assert _values(facets, "enhancement") == {
        "image_caption:stale": 1,
        "text_embedding:failed": 1,
    }


async def test_mixed_case_tags_without_declarations_count_like_the_list(
    service: ARIELSearchService,
) -> None:
    facets = await _assert_counts_are_list_lengths(service, "", None, {})

    assert _values(facets, "tags") == {"beam": 3, "diag": 2, "rf": 2, "vacuum": 1}


async def test_each_dimension_ignores_its_own_selection(service: ARIELSearchService) -> None:
    base = {"author": ["alice"], "source_system": ["als"]}

    facets = await _assert_counts_are_list_lengths(service, "", None, base)

    # Authors are counted under source_system only, tags under both.
    assert _values(facets, "author") == {"alice": 2, "bob": 1, "carol": 1, "dave": 1, "erin": 1}
    assert _values(facets, "tags") == {"beam": 2, "rf": 1}


async def test_keyword_strict_counts_equal_list_lengths(service: ARIELSearchService) -> None:
    facets = await _assert_counts_are_list_lengths(
        service, "kicker misfire", "keyword", {"author": ["alice"]}
    )

    assert facets["tier"] == "strict"
    assert _values(facets, "author") == {"alice": 2, "bob": 1, "carol": 2, "dave": 1}


async def test_keyword_with_vocabulary_expansion_counts_equal_list_lengths(
    service: ARIELSearchService,
) -> None:
    advanced = {"expand_query": True}

    facets = await _assert_counts_are_list_lengths(service, "bpm", "keyword", {}, advanced)

    assert _values(facets, "author") == {"erin": 2}


async def test_a_keyword_tier_unlike_the_lists_is_inexact(service: ARIELSearchService) -> None:
    """The matched set is strict, the list under ``author=bob`` falls to partial."""
    flt = _parse(service.config, {"author": ["bob"]})

    facets = await service.facets("kicker misfire septum", "keyword", flt, {})

    assert facets["tier"] == "strict"
    assert facets["exact"] is False


async def test_hybrid_is_never_exact(service: ARIELSearchService) -> None:
    facets = await service.facets("kicker misfire", "hybrid", _parse(service.config, {}), {})

    assert facets["exact"] is False
    assert sum(_values(facets, "author").values()) == 6


async def test_late_evening_entry_counts_on_its_facility_day(
    service: ARIELSearchService, scratch_database: str
) -> None:
    late = _local(LATE_DAY, 23, 30)
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO enhanced_entries (entry_id, source_system, timestamp, author, raw_text)"
            " VALUES ('f-late', 'als', %s, 'Frank', 'late shift handover')",
            (late,),
        )
    day = LATE_DAY.isoformat()
    # The same instant is already the next day in UTC.
    assert late.astimezone(ZoneInfo("UTC")).date() != LATE_DAY

    facets = await service.facets("", None, _parse(service.config, {"time": {"start": day}}), {})

    histogram = {bucket["day"]: bucket["count"] for bucket in facets["histogram"]}
    assert histogram[day] == 1
    assert histogram[BASE_DAY.isoformat()] == len(ROWS)
    listed = await service.repository.search_by_time_range(
        limit=100, entry_filter=_parse(service.config, {"time": {"start": day, "end": day}})
    )
    assert [entry["entry_id"] for entry in listed] == ["f-late"]


async def test_semantic_counts_equal_list_lengths_under_threshold_and_expansion(
    service: ARIELSearchService, scratch_database: str
) -> None:
    from osprey.services.ariel_search.database.migrations import model_to_table_name
    from tests.services.ariel_search.fake_providers import make_fake_embedding_provider

    with psycopg.connect(scratch_database, autocommit=True) as conn:
        table = conn.execute(
            "SELECT to_regclass(%s) IS NOT NULL", (model_to_table_name(MODEL),)
        ).fetchone()
    if not table or not table[0]:
        pytest.skip("this server has no pgvector; the embedding table was not created")

    def unit(x: float, y: float) -> list[float]:
        return [x, y] + [0.0] * (DIM - 2)

    near = {"f-1", "f-2", "f-3", "f-bpm-long"}
    middle = {"f-stale", "f-failed"}
    for entry_id, *_ in ROWS:
        vector = unit(1.0, 0.0) if entry_id in near else unit(0.6, 0.8)
        if entry_id not in near | middle:
            vector = unit(0.0, 1.0)
        await service.repository.store_text_embedding(entry_id, vector, MODEL)

    embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
    embedder.execute_embedding = MagicMock(return_value=[unit(1.0, 0.0)])
    service._embedder = embedder
    advanced = {
        "provider_config": {"base_url": "http://localhost:11434", "api_key": None},
        "similarity_threshold": 0.8,
        "expand_query": True,
    }

    facets = await _assert_counts_are_list_lengths(
        service, "bpm drift", "semantic", {"source_system": ["als"]}, advanced
    )

    assert facets["tier"] is None
    # The default threshold would also take the two 0.6-similarity entries.
    assert _values(facets, "author") == {"alice": 2, "bob": 1, "erin": 1}


async def test_an_author_facet_never_lists_a_longer_name(
    service: ARIELSearchService, scratch_database: str
) -> None:
    """ "J. Smith" counts and lists only J. Smith, never J. Smithson; Mine for
    "J. Smith" sends the same author filter and lists the same entries."""
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        for entry_id, author in (("s-1", "J. Smith"), ("s-2", "J. Smith"), ("s-3", "J. Smithson")):
            conn.execute(
                "INSERT INTO enhanced_entries (entry_id, source_system, timestamp, author,"
                " raw_text, metadata) VALUES (%s, 'als', %s, %s, 'kicker misfire', '{}'::jsonb)",
                (entry_id, _local(BASE_DAY, 13), author),
            )

    facets = await _assert_counts_are_list_lengths(service, "", None, {})
    assert _values(facets, "author")["j. smith"] == 2
    assert _values(facets, "author")["j. smithson"] == 1

    # The facet click and the Mine mailbox both send ``author: [<name>]``.
    mine = _parse(service.config, {"author": ["J. Smith"]})
    listed = await service.repository.search_by_time_range(limit=1000, entry_filter=mine)
    assert sorted(e["entry_id"] for e in listed) == ["s-1", "s-2"]
    searched = await service.search("kicker misfire", mode="keyword", entry_filter=mine)
    assert sorted(e["entry_id"] for e in searched.entries) == ["s-1", "s-2"]
    # Counts within the Mine mailbox: every other dimension sees only J. Smith.
    within = await service.facets("", None, mine, {})
    assert _values(within, "source_system") == {"als": 2}
