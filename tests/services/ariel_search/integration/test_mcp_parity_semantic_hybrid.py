"""REST ``/api/search`` and the MCP search tools return the same entries (semantic, hybrid).

Both surfaces end in one ``ARIELSearchService.search`` call; each translates its
own parameters (dates, author, source system, thresholds, hybrid overrides) into
that call. Every case here sends one query with one set of filters through the
REST route and through the MCP tool against one seeded database and one service,
and asserts the same entry ids come back in the same order -- and that the order
equals what a direct ``service.search`` returns, so a surface that departs from
the shared call is the one that fails.

Semantic search embeds with the fake embedding provider; hybrid search reads a
stubbed qmd client whose ranking is fixed, so both run without a model server or
a sidecar. Hybrid runs under both strategies: ``qmd`` (sidecar only, filters
applied after ranking) and ``rrf`` (qmd's lexical lane fused with the semantic
lane in PostgreSQL).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import psycopg
import pytest

from tests.interfaces.ariel._routes_app import routes_app
from tests.mcp_server.conftest import get_tool_fn
from tests.services.ariel_search.conftest import skip_or_fail
from tests.services.ariel_search.fake_providers import make_fake_embedding_provider

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.xdist_group("docker"),
    pytest.mark.timeout(120),
]

MODEL = "nomic-embed-text"
DIM = 4
QUERY = "kicker misfire at injection"
QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]

#: (entry_id, day of March 2026, author, source_system, raw_text, embedding).
#: Cosine similarity to QUERY_VECTOR falls strictly from p-1 to p-6, so the
#: semantic ranking is total; p-6 sits below the default 0.5 threshold.
ROWS: tuple[tuple[str, int, str, str, str, list[float]], ...] = (
    ("p-1", 1, "J. Smith", "als", "kicker misfire at injection", [1.0, 0.0, 0.0, 0.0]),
    ("p-2", 2, "J. Smithson", "als", "kicker misfire septum", [0.95, 0.31, 0.0, 0.0]),
    ("p-3", 3, "j. smith", "linac", "kicker timing drift", [0.8, 0.6, 0.0, 0.0]),
    ("p-4", 4, "Operator", "als", "vacuum burst sector three", [0.7, 0.71, 0.0, 0.0]),
    ("p-5", 5, "J. Smith", "linac", "kicker misfire again", [0.6, 0.8, 0.0, 0.0]),
    ("p-6", 6, "Operator", "als", "rf cavity trip", [0.1, 0.99, 0.0, 0.0]),
)

#: The fixed qmd ranking: deliberately not the semantic order, so the rrf fusion
#: reorders it and the qmd strategy keeps it.
QMD_RANKING = ("p-5", "p-3", "p-1", "p-6", "p-2", "p-4")


class RankedQMDClient:
    """qmd faked at its client boundary: a fixed ranking over every seeded entry."""

    is_configured = True
    base_url = "http://127.0.0.1:8180"

    def __init__(self, ranking: tuple[str, ...]) -> None:
        self.ranking = ranking
        self.queries = 0

    def is_available(self) -> bool:
        return True

    def query(self, collection: str | None, _text: str, **kwargs: Any) -> list[Any]:
        from osprey.services.ariel_search.enhancement.qmd_export.writer import encode_entry_id
        from osprey.services.qmd import QMDSearchResult

        self.queries += 1
        texts = {row[0]: row[4] for row in ROWS}
        hits = [
            QMDSearchResult(
                docid=f"#{rank:06d}",
                file=f"2026/03/{encode_entry_id(entry_id)}.md",
                collection=collection or "ariel",
                title=f"Entry {entry_id}",
                score=1.0 - rank / 10,
                line=1,
                snippet=f"1: {texts[entry_id]}",
            )
            for rank, entry_id in enumerate(self.ranking)
        ]
        return hits[: kwargs.get("limit", len(hits))]


def _seed(uri: str) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        for entry_id, day, author, source, text, _vector in ROWS:
            conn.execute(
                """
                INSERT INTO enhanced_entries (
                    entry_id, source_system, timestamp, author, raw_text, metadata
                ) VALUES (%s, %s, %s, %s, %s, '{}'::jsonb)
                """,
                (entry_id, source, datetime(2026, 3, day, 18, tzinfo=UTC), author, text),
            )


def _raw(uri: str, strategy: str) -> dict[str, Any]:
    return {
        "database": {"uri": uri},
        "search_modules": {
            "keyword": {"enabled": True},
            "semantic": {
                "enabled": True,
                "model": MODEL,
                "provider": {"name": "fake", "base_url": "http://fake-embeddings.invalid:11434"},
            },
            "hybrid": {"enabled": True, "settings": {"strategy": strategy}},
        },
        "enhancement_modules": {
            "text_embedding": {
                "enabled": True,
                "models": [{"name": MODEL, "dimension": DIM}],
            },
        },
    }


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Fresh MCP, config and qmd state; every provider lookup is the fake embedder."""
    from osprey.mcp_server.ariel.server_context import reset_ariel_context
    from osprey.models.provider_registry import ProviderRegistry
    from osprey.services.ariel_search.search import qmd as qmd_module
    from osprey_connectors.workspace import reset_config_cache

    reset_ariel_context()
    reset_config_cache()
    qmd_module._reset_client_cache()
    fake = make_fake_embedding_provider(dimension=DIM, vector=QUERY_VECTOR)
    monkeypatch.setattr(ProviderRegistry, "get_provider", lambda _self, _name: fake)
    monkeypatch.setattr(
        "osprey.models.config.get_provider_config", lambda name, *a, **k: {"api_key": "k"}
    )
    qmd_client = RankedQMDClient(QMD_RANKING)
    monkeypatch.setattr(qmd_module, "_resolve_client", lambda client: (qmd_client, True))
    yield
    reset_ariel_context()
    reset_config_cache()
    qmd_module._reset_client_cache()


@pytest.fixture
def strategy(request) -> str:
    """The hybrid strategy; semantic search reads none, so it runs under the default."""
    return getattr(request, "param", "qmd")


#: Hybrid cases run once per strategy.
STRATEGIES = pytest.mark.parametrize("strategy", ["qmd", "rrf"], indirect=True)


@pytest.fixture
async def service(
    scratch_database: str, strategy: str, monkeypatch, tmp_path
) -> AsyncIterator[Any]:
    """One service over a migrated, seeded, embedded store, shared by MCP and REST."""
    from osprey.mcp_server.ariel.server_context import initialize_ariel_context
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import (
        model_to_table_name,
        run_migrations,
    )
    from osprey.services.ariel_search.service import create_ariel_service

    raw = _raw(scratch_database, strategy)
    config = ARIELConfig.from_dict(raw)
    async with await create_connection_pool(config.database) as pool:
        await run_migrations(pool, config)
    _seed(scratch_database)
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        table = conn.execute(
            "SELECT to_regclass(%s) IS NOT NULL", (model_to_table_name(MODEL),)
        ).fetchone()
    if not table or not table[0]:
        skip_or_fail("this server has no pgvector; the embedding table was not created")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(json.dumps({"ariel": raw}))
    initialize_ariel_context()
    svc = await create_ariel_service(config)
    try:
        for entry_id, *_rest, vector in ROWS:
            await svc.repository.store_text_embedding(entry_id, vector, MODEL)
        with patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=svc),
        ):
            yield svc
    finally:
        await svc.aclose()


# --- the two surfaces ----------------------------------------------------------------


async def _mcp(mode: str, **kwargs: Any) -> list[str]:
    if mode == "semantic":
        from osprey.mcp_server.ariel.tools.semantic_search import semantic_search as tool
    else:
        from osprey.mcp_server.ariel.tools.hybrid_search import hybrid_search as tool
    document = json.loads(await get_tool_fn(tool)(query=QUERY, **kwargs))
    entries = document.get("entries")
    assert isinstance(entries, list), document
    return [e["entry_id"] for e in entries]


async def _rest(service: Any, mode: str, body: dict[str, Any]) -> list[str]:
    app = routes_app(service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ariel.test") as client:
        response = await client.post("/api/search", json={"query": QUERY, "mode": mode, **body})
    assert response.status_code == 200, response.text
    return [e["entry_id"] for e in response.json()["entries"]]


async def _direct(service: Any, mode: str, max_results: int, **search: Any) -> list[str]:
    result = await service.search(QUERY, max_results=max_results, mode=mode, **search)
    return [e["entry_id"] for e in result.entries[:max_results]]


def _day(day: int, *, end: bool = False) -> datetime:
    from osprey.services.ariel_search.models import parse_time_bound

    return parse_time_bound(f"2026-03-{day:02d}", end=end)


#: (id, MCP kwargs, REST body, direct service.search kwargs). Each row states one
#: request the same way to all three; max_results is in every row.
Case = tuple[str, dict[str, Any], dict[str, Any], Callable[[], dict[str, Any]]]

COMMON: tuple[Case, ...] = (
    ("query-only", {"max_results": 10}, {"max_results": 10}, lambda: {}),
    ("max_results", {"max_results": 2}, {"max_results": 2}, lambda: {}),
    (
        "author",
        {"max_results": 10, "author": "smith"},
        {"max_results": 10, "author": "smith"},
        lambda: {"advanced_params": {"author": "smith"}},
    ),
    (
        "source_system",
        {"max_results": 10, "source_system": "linac"},
        {"max_results": 10, "source_system": "linac"},
        lambda: {"advanced_params": {"source_system": "linac"}},
    ),
    (
        "start_date",
        {"max_results": 10, "start_date": "2026-03-03"},
        {"max_results": 10, "start_date": "2026-03-03"},
        lambda: {"time_range": (_day(3), None)},
    ),
    (
        "end_date-whole-day",
        {"max_results": 10, "end_date": "2026-03-03"},
        {"max_results": 10, "end_date": "2026-03-03"},
        lambda: {"time_range": (None, _day(3, end=True))},
    ),
    (
        "combined",
        {"max_results": 10, "author": "smith", "source_system": "als", "end_date": "2026-03-04"},
        {"max_results": 10, "author": "smith", "source_system": "als", "end_date": "2026-03-04"},
        lambda: {
            "time_range": (None, _day(4, end=True)),
            "advanced_params": {"author": "smith", "source_system": "als"},
        },
    ),
    (
        "expand_query-off",
        {"max_results": 10, "expand_query": False},
        {"max_results": 10, "advanced_params": {"expand_query": False}},
        lambda: {"advanced_params": {"expand_query": False}},
    ),
)

SEMANTIC_ONLY: tuple[Case, ...] = (
    (
        "similarity_threshold",
        {"max_results": 10, "similarity_threshold": 0.9},
        {"max_results": 10, "advanced_params": {"similarity_threshold": 0.9}},
        lambda: {"advanced_params": {"similarity_threshold": 0.9}},
    ),
)

HYBRID_ONLY: tuple[Case, ...] = (
    (
        "rerank-off",
        {"max_results": 10, "rerank": False},
        {"max_results": 10, "advanced_params": {"rerank": False}},
        lambda: {"advanced_params": {"rerank": False}},
    ),
    (
        "include_images-off",
        {"max_results": 10, "include_images": False},
        {"max_results": 10, "advanced_params": {"include_images": False}},
        lambda: {"advanced_params": {"include_images": False}},
    ),
)


def _params(cases: tuple[Case, ...]) -> list[Any]:
    return [pytest.param(mcp, rest, direct, id=case_id) for case_id, mcp, rest, direct in cases]


async def _assert_parity(
    service: Any,
    mode: str,
    mcp_kwargs: dict[str, Any],
    rest_body: dict[str, Any],
    direct: Callable[[], dict[str, Any]],
) -> list[str]:
    via_mcp = await _mcp(mode, **mcp_kwargs)
    via_rest = await _rest(service, mode, json.loads(json.dumps(rest_body)))
    via_service = await _direct(service, mode, mcp_kwargs["max_results"], **direct())

    assert via_mcp == via_service, f"MCP {mode} departs from service.search"
    assert via_rest == via_service, f"REST {mode} departs from service.search"
    return via_service


@pytest.mark.parametrize(("mcp_kwargs", "rest_body", "direct"), _params(COMMON + SEMANTIC_ONLY))
async def test_semantic_search_parity(
    service: Any,
    mcp_kwargs: dict[str, Any],
    rest_body: dict[str, Any],
    direct: Callable[[], dict[str, Any]],
) -> None:
    await _assert_parity(service, "semantic", mcp_kwargs, rest_body, direct)


@STRATEGIES
@pytest.mark.parametrize(("mcp_kwargs", "rest_body", "direct"), _params(COMMON + HYBRID_ONLY))
async def test_hybrid_search_parity(
    service: Any,
    mcp_kwargs: dict[str, Any],
    rest_body: dict[str, Any],
    direct: Callable[[], dict[str, Any]],
) -> None:
    await _assert_parity(service, "hybrid", mcp_kwargs, rest_body, direct)


@pytest.mark.usefixtures("service")
async def test_semantic_ranking_is_the_seeded_similarity_order() -> None:
    """The corpus actually ranks: a parity of empty lists would prove nothing."""
    assert await _mcp("semantic", max_results=10) == ["p-1", "p-2", "p-3", "p-4", "p-5"]


@STRATEGIES
@pytest.mark.usefixtures("service")
async def test_hybrid_ranking_covers_every_qmd_hit(strategy: str) -> None:
    """Both strategies return every qmd hit, so the parity cases compare real rankings.

    The qmd strategy keeps the sidecar's order; rrf fuses it with the semantic lane.
    """
    ids = await _mcp("hybrid", max_results=10)
    assert set(ids) == set(QMD_RANKING)
    if strategy == "qmd":
        assert ids == list(QMD_RANKING)
    else:
        assert ids != list(QMD_RANKING)
