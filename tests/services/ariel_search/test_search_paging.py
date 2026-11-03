"""Paging and totals of the built-in search modules and the service (C-MODOUT).

Each module is driven over a repository double that pages and counts the way
the SQL does, so these tests pin the module and service logic: pages never
overlap, totals are exact or flagged as specified, and ``has_more`` turns false
on the last page. The SQL itself is pinned against real PostgreSQL in
``integration/test_search_totals.py``.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import SearchRows
from osprey.services.ariel_search.search import fusion
from osprey.services.ariel_search.search.base import ModuleOutput, SearchToolDescriptor
from osprey.services.ariel_search.search.fusion import ImageHit
from osprey.services.ariel_search.search.keyword import keyword_search
from osprey.services.ariel_search.search.qmd import hybrid_search
from osprey.services.ariel_search.search.semantic import semantic_search
from osprey.services.ariel_search.service import ARIELSearchService
from tests.services.ariel_search.fake_providers import make_fake_embedding_provider
from tests.services.ariel_search.repo_fakes import attach_fake_fts
from tests.services.ariel_search.test_hybrid_search import (
    FakeLane,
    LaneRepository,
    StubClient,
    StubRepository,
    lane_config,
    make_config,
    make_entry,
    make_hit,
)

pytestmark = pytest.mark.asyncio

PAGE = 20


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


def _paged(rows: list[Any], kwargs: dict[str, Any]) -> list[Any]:
    """Page `rows` (already in SQL order) as ``LIMIT/OFFSET`` and attach the count."""
    offset = kwargs.get("offset", 0)
    page = SearchRows(rows[offset : offset + kwargs["max_results"]])
    if kwargs.get("count"):
        page.total, page.total_exact = min(len(rows), 1000), len(rows) <= 1000
    return page


def _keyword_repo(strict: list[Any], fuzzy: list[Any] | None = None) -> MagicMock:
    repo = MagicMock()
    attach_fake_fts(repo, has_v2=False, has_copy_state=False)
    repo.keyword_search = AsyncMock(side_effect=lambda **kw: _paged(strict, kw))
    repo.fuzzy_search = AsyncMock(side_effect=lambda **kw: _paged(fuzzy or [], kw))
    return repo


def _rows(n: int) -> list[tuple[dict[str, Any], float, list[str]]]:
    """``n`` rows of equal score, in the SQL tiebreak order (``entry_id DESC``)."""
    return [(make_entry(f"E{i:03d}"), 0.5, []) for i in reversed(range(n))]


def _ids(result: Any) -> list[str]:
    entries = result.entries if isinstance(result, ModuleOutput) else result
    return [row[0]["entry_id"] for row in entries]


@pytest.fixture(autouse=True)
def _hermetic_local_server(monkeypatch):
    """The semantic module probes no real port."""
    from osprey.models.providers import _local_server

    _local_server.reset_cache()
    monkeypatch.setattr(_local_server, "probe", lambda url, path, timeout: False)
    yield
    _local_server.reset_cache()


# --------------------------------------------------------------------------
# Keyword
# --------------------------------------------------------------------------


class TestKeywordPaging:
    async def _page(self, repo: MagicMock, offset: int) -> ModuleOutput:
        out = await keyword_search(
            "beam",
            repo,
            ARIELConfig.from_dict({"database": {"uri": "postgresql://x/y"}}),
            max_results=PAGE,
            offset=offset,
        )
        assert isinstance(out, ModuleOutput)
        return out

    async def test_pages_are_disjoint_and_cover_every_row(self) -> None:
        rows = _rows(45)
        repo = _keyword_repo(rows)

        pages = [await self._page(repo, offset) for offset in (0, 20, 40)]

        ids = [_ids(page) for page in pages]
        assert not set(ids[0]) & set(ids[1])
        assert not set(ids[1]) & set(ids[2])
        assert [i for page in ids for i in page] == [r[0]["entry_id"] for r in rows]
        assert [(p.total, p.total_exact, p.has_more) for p in pages] == [
            (45, True, True),
            (45, True, True),
            (45, True, False),
        ]
        assert {p.tier for p in pages} == {"strict"}

    async def test_fuzzy_total_is_inexact(self) -> None:
        repo = _keyword_repo([], fuzzy=_rows(3))

        out = await self._page(repo, 0)

        assert out.tier == "fuzzy"
        assert (out.total, out.total_exact, out.has_more) == (3, False, False)

    async def test_a_later_page_never_falls_back_to_fuzzy(self) -> None:
        repo = _keyword_repo(_rows(5), fuzzy=_rows(30))

        out = await self._page(repo, 20)

        assert out.tier == "strict"
        assert _ids(out) == []
        assert (out.total, out.has_more) == (5, False)
        repo.fuzzy_search.assert_not_awaited()

    async def test_capped_total_is_inexact_and_keeps_paging(self) -> None:
        repo = _keyword_repo(_rows(1005))

        out = await self._page(repo, 980)

        assert (out.total, out.total_exact, out.has_more) == (1000, False, True)


# --------------------------------------------------------------------------
# Semantic
# --------------------------------------------------------------------------


class TestSemanticPaging:
    async def test_pages_are_disjoint_with_exact_totals(self, search_calls) -> None:
        rows = [(entry, score) for entry, score, _h in _rows(25)]
        repo = MagicMock()
        repo.semantic_search = AsyncMock(side_effect=lambda **kw: _paged(rows, kw))
        repo.get_embedding_tables = AsyncMock(return_value=[])
        embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
        embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"semantic": {"enabled": True, "model": "test-model"}},
            }
        )

        pages = [
            await semantic_search(
                "beam",
                repo,
                config,
                embedder,
                max_results=PAGE,
                offset=offset,
                search_calls=search_calls,
            )
            for offset in (0, 20)
        ]

        assert not set(_ids(pages[0])) & set(_ids(pages[1]))
        assert len(_ids(pages[0])) + len(_ids(pages[1])) == 25
        assert [(p.total, p.total_exact, p.has_more) for p in pages] == [
            (25, True, True),
            (25, True, False),
        ]
        assert repo.semantic_search.call_args.kwargs["count"] is True


# --------------------------------------------------------------------------
# Hybrid
# --------------------------------------------------------------------------


class RerankingClient(StubClient):
    """A qmd stub whose reranker reverses the base order."""

    def query(self, collection: str | None, text: str, **kwargs: Any) -> list[Any]:
        hits = super().query(collection, text, **kwargs)
        return list(reversed(hits)) if kwargs.get("rerank") else hits


TEXT_IDS = [f"T{i:02d}" for i in range(50)]


def _text_fixture(client_cls: type[StubClient] = StubClient):
    client = client_cls([make_hit(e, score=1.0 - i / 100) for i, e in enumerate(TEXT_IDS)])
    return client, StubRepository([make_entry(e) for e in TEXT_IDS])


class TestHybridPaging:
    async def test_rerank_is_confined_to_page_one(self, search_calls) -> None:
        client, repo = _text_fixture(RerankingClient)
        config = make_config({"rerank": True})

        page1 = await hybrid_search(
            "beam",
            repo,
            config,
            client=client,
            max_results=PAGE,
            offset=0,
            search_calls=search_calls,
        )
        page2 = await hybrid_search(
            "beam",
            repo,
            config,
            client=client,
            max_results=PAGE,
            offset=20,
            search_calls=search_calls,
        )

        assert set(_ids(page1)) | set(_ids(page2)) == set(TEXT_IDS[:40])
        assert not set(_ids(page1)) & set(_ids(page2))
        # Page 1 holds the base page, in the reranked order.
        assert _ids(page1) == list(reversed(TEXT_IDS[:20]))
        assert _ids(page2) == TEXT_IDS[20:40]
        # Page 1 fetches the base and the reranked list; page 2 only the base.
        assert [call["rerank"] for call in client.calls] == [False, True, False]
        assert (page1.total_exact, page2.total_exact) == (False, False)

    async def test_page_one_over_many_text_matches_has_more(self, search_calls) -> None:
        client, repo = _text_fixture()

        page1 = await hybrid_search(
            "beam",
            repo,
            make_config({"rerank": False}),
            client=client,
            max_results=PAGE,
            offset=0,
            search_calls=search_calls,
        )

        assert len(_ids(page1)) == PAGE
        assert page1.has_more is True
        assert page1.total == PAGE + 1
        assert client.calls[0]["limit"] == PAGE + 1

    async def test_last_page_has_no_more(self, search_calls) -> None:
        client, repo = _text_fixture()

        page = await hybrid_search(
            "beam",
            repo,
            make_config({"rerank": False}),
            client=client,
            max_results=PAGE,
            offset=40,
            search_calls=search_calls,
        )

        assert _ids(page) == TEXT_IDS[40:]
        assert (page.total, page.total_exact, page.has_more) == (50, False, False)

    async def test_fused_pages_cover_the_fused_list(self, monkeypatch, search_calls) -> None:
        images = {
            "T00": ImageHit("att-0", 0.90),
            "I00": ImageHit("att-i0", 0.89),
            "I01": ImageHit("att-i1", 0.88),
        }
        monkeypatch.setattr(
            "osprey.services.ariel_search.search.image_lane.search_images", FakeLane(images)
        )
        client = StubClient([make_hit(e, score=1.0 - i / 100) for i, e in enumerate(TEXT_IDS)])
        repo = LaneRepository([make_entry(e) for e in [*TEXT_IDS, "I00", "I01"]])
        config = lane_config()
        config.search_modules["hybrid"].settings["rerank"] = False

        pages = [
            await hybrid_search(
                "beam",
                repo,
                config,
                client=client,
                max_results=PAGE,
                offset=offset,
                include_images=True,
                search_calls=search_calls,
            )
            for offset in (0, 20)
        ]

        window = 2 * PAGE + 1
        fused = fusion.fuse_lanes(
            [(e, 1.0 - i / 100) for i, e in enumerate(TEXT_IDS[:window])],
            images,
            cap=math.ceil(window / 3),
        )
        expected = [hit.entry_id for hit in fused][: 2 * PAGE]
        assert {"I00", "I01"} <= set(expected)
        assert _ids(pages[0]) + _ids(pages[1]) == expected
        assert pages[0].has_more is True


# --------------------------------------------------------------------------
# Service fallback for modules that cannot count
# --------------------------------------------------------------------------


class _Input(BaseModel):
    query: str


def _service_with(execute: Any, *, offset: bool) -> tuple[ARIELSearchService, MagicMock]:
    descriptor = SearchToolDescriptor(
        name="third_search",
        description="a third-party module",
        search_mode="third",
        args_schema=_Input,
        execute=execute,
        format_result=lambda *a, **k: {},
        accepts_offset=offset,
    )
    module = MagicMock()
    module.get_tool_descriptor.return_value = descriptor
    registry = MagicMock()
    registry.list_ariel_search_modules.return_value = ["third"]
    registry.get_ariel_search_module.side_effect = lambda n: module if n == "third" else None
    registry.list_ariel_enhancement_modules.return_value = []
    config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"third": {"enabled": True}},
        }
    )
    pool = MagicMock()
    pool.close = AsyncMock()
    repository = MagicMock()
    repository.health_check = AsyncMock(return_value=(True, "OK"))
    repository.validate_search_model_table = AsyncMock()
    attach_fake_fts(repository, has_v2=False, has_copy_state=False)
    return ARIELSearchService(config=config, pool=pool, repository=repository), registry


class TestServiceTotals:
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [(0, (20, 21, False, True)), (20, (5, 25, False, False))],
    )
    async def test_unflagged_module_gets_a_derived_inexact_total(self, offset, expected) -> None:
        rows = _rows(25)

        async def execute(*_a: Any, **kw: Any) -> Any:
            return rows[: kw["max_results"]]

        service, registry = _service_with(execute, offset=False)
        with patch("osprey.registry.get_registry", return_value=registry):
            result = await service.search("beam", mode="third", max_results=PAGE, offset=offset)

        assert (len(result.entries), result.total, result.total_exact, result.has_more) == expected

    async def test_offset_module_without_a_count_gets_offset_plus_page(self) -> None:
        async def execute(*_a: Any, **kw: Any) -> Any:
            return _rows(PAGE)

        service, registry = _service_with(execute, offset=True)
        with patch("osprey.registry.get_registry", return_value=registry):
            result = await service.search("beam", mode="third", max_results=PAGE, offset=40)

        assert (result.total, result.total_exact, result.has_more) == (61, False, True)

    async def test_counted_module_total_passes_through(self) -> None:
        async def execute(*_a: Any, **kw: Any) -> Any:
            return ModuleOutput(entries=_rows(5), total=25, total_exact=True, has_more=False)

        service, registry = _service_with(execute, offset=True)
        with patch("osprey.registry.get_registry", return_value=registry):
            result = await service.search("beam", mode="third", max_results=PAGE, offset=20)

        assert (result.total, result.total_exact, result.has_more) == (25, True, False)
