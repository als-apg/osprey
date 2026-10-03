"""The picture lane of hybrid search: bounds, breaker, reasons and the vector SQL."""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.models.providers import _local_server
from osprey.models.providers.health import HealthResult
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.migrations import image_table_name
from osprey.services.ariel_search.enhancement import availability
from osprey.services.ariel_search.enhancement.provider_resolver import ResolvedProvider
from osprey.services.ariel_search.search import image_lane
from osprey.services.ariel_search.search.fusion import ImageHit
from tests.services.ariel_search.conftest import _FakePool
from tests.services.ariel_search.llama_stub import MODEL

DIMS = 1024
TABLE = image_table_name(MODEL, DIMS)
BOUND_S = image_lane.IMAGE_QUERY_TIMEOUT_S + 1.5


#: Query-embed p95 under bulk picture embedding on native amd64, with the
#: documented llama-server command carrying ``--image-max-tokens 256``.
MEASURED_QUERY_P95_S = 0.94


def test_query_timeout_is_twice_the_measured_p95_and_never_under_five():
    assert image_lane.IMAGE_QUERY_TIMEOUT_S == max(5.0, 2 * MEASURED_QUERY_P95_S)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _refused_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def _ariel_config(url: str = "http://127.0.0.1:1", **block: Any) -> ARIELConfig:
    image_embedding: dict[str, Any] = {
        "enabled": True,
        "provider": {"name": "llama-cpp", "base_url": url},
        "model": MODEL,
        "dimensions": DIMS,
    }
    image_embedding.update(block)
    image_embedding = {k: v for k, v in image_embedding.items() if v is not None}
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "enhancement_modules": {"image_embedding": image_embedding},
        }
    )


def _repo(pool: Any) -> SimpleNamespace:
    return SimpleNamespace(pool=pool)


def _pool(rows: Any = None) -> _FakePool:
    return _FakePool(rows_for={f"FROM {TABLE}": rows if rows is not None else []})


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(image_lane, "_now", fake)
    return fake


class _FakeImageProvider:
    """An image-embedding adapter double: records calls, answers a unit vector."""

    resolves_fallback_outside_calls = False

    def __init__(self) -> None:
        self.embed_calls: list[dict[str, Any]] = []
        self.health_calls: list[tuple[Any, ...]] = []
        self.health = HealthResult(True, "ok", None)
        self.error: Exception | None = None
        self.release = threading.Event()
        self.sleep_s = 0.0

    def check_embedding_health(self, api_key, base_url, model_id=None, timeout=10.0):
        self.health_calls.append((api_key, base_url, model_id, timeout))
        return self.health

    def execute_image_embedding(
        self,
        inputs,
        model_id,
        api_key=None,  # noqa: ARG002 - adapter contract
        base_url=None,
        dimensions=None,
        timeout=600.0,
    ):
        self.embed_calls.append(
            {
                "inputs": list(inputs),
                "model_id": model_id,
                "base_url": base_url,
                "dimensions": dimensions,
                "timeout": timeout,
            }
        )
        if self.sleep_s:
            self.release.wait(self.sleep_s)
        if self.error is not None:
            raise self.error
        return [[1.0] + [0.0] * ((dimensions or 4) - 1)]


@pytest.fixture
def fake_provider(monkeypatch):
    provider = _FakeImageProvider()
    calls: list[Any] = []

    def _resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return ResolvedProvider(
            cls=type(provider),  # type: ignore[arg-type]
            instance=provider,  # type: ignore[arg-type]
            base_url="http://user:secret@fake-host:8080",
            api_key="key",
        )

    monkeypatch.setattr(image_lane, "resolve_provider", _resolve)
    provider.resolve_calls = calls  # type: ignore[attr-defined]
    yield provider
    provider.release.set()


def _lane_records(caplog, level: int) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name.startswith("ariel") and r.levelno == level and "image_lane" in r.getMessage()
    ]


# ---------------------------------------------------------------------------
# The vector SQL
# ---------------------------------------------------------------------------


class TestVectorSql:
    @pytest.mark.usefixtures("fake_provider")
    async def test_k_and_ef_search_follow_fetch_limit(self):
        pool = _pool([("e1", "a1", 0.8), ("e2", "a3", 0.6)])
        hits = await image_lane.search_images(
            "beam loss", _repo(pool), _ariel_config(), fetch_limit=200
        )

        assert hits == {"e1": ImageHit("a1", 0.8), "e2": ImageHit("a3", 0.6)}
        (settings_call,) = pool.matching("set_config('hnsw.ef_search'")
        assert settings_call[1] == {"ef": "600", "st": "2000"}
        assert "set_config('statement_timeout'" in settings_call[0]
        (query_call,) = pool.matching(f"FROM {TABLE}")
        assert query_call[1]["k"] == 600
        assert query_call[1]["q"].startswith("[1.0,")
        assert "%(q)s::vector" in query_call[0]
        assert "1 - (t.embedding <=> %(q)s::vector) AS similarity" in query_call[0]
        assert "DISTINCT ON (f.entry_id)" in query_call[0]
        assert "embedding IS NOT NULL" in query_call[0]
        assert pool.connection_timeouts == [image_lane.IMAGE_LANE_SQL_TIMEOUT_S]
        assert pool.conn.transactions == ["BEGIN", "COMMIT"]
        for sql in pool.sql:
            assert not ("SET " in sql and "%s" in sql)
            assert "hnsw.iterative_scan" not in sql
            assert "pg_available_extensions" not in sql

    @pytest.mark.parametrize(
        ("fetch_limit", "k", "ef"), [(200, 600, "600"), (5, 15, "40"), (500, 1000, "1000")]
    )
    @pytest.mark.usefixtures("fake_provider")
    async def test_k_is_capped_and_ef_search_floored(self, fetch_limit, k, ef):
        pool = _pool()
        assert (
            await image_lane.search_images(
                "q", _repo(pool), _ariel_config(), fetch_limit=fetch_limit
            )
            == {}
        )
        assert pool.matching(f"FROM {TABLE}")[0][1]["k"] == k
        assert pool.matching("hnsw.ef_search")[0][1]["ef"] == ef

    @pytest.mark.usefixtures("fake_provider")
    async def test_dict_rows_keep_the_nearest_picture_per_entry(self):
        pool = _pool(
            [
                {"entry_id": "e1", "attachment_id": "a1", "similarity": 0.9},
                {"entry_id": "e1", "attachment_id": "a2", "similarity": 0.5},
            ]
        )
        hits = await image_lane.search_images("q", _repo(pool), _ariel_config(), fetch_limit=10)
        assert hits == {"e1": ImageHit("a1", 0.9)}

    async def test_original_query_text_is_embedded(self, fake_provider):
        await image_lane.search_images(
            "orbit plot sector 3", _repo(_pool()), _ariel_config(), fetch_limit=10
        )
        (call,) = fake_provider.embed_calls
        assert call["inputs"] == ["orbit plot sector 3"]
        assert call["dimensions"] == DIMS
        assert call["model_id"] == MODEL

    @pytest.mark.usefixtures("clock")
    async def test_missing_table_is_a_config_failure(self, fake_provider, caplog):
        from psycopg.errors import UndefinedTable

        pool = _pool(UndefinedTable(f'relation "{TABLE}" does not exist'))
        with caplog.at_level(logging.DEBUG, logger="ariel"):
            hits = await image_lane.search_images("q", _repo(pool), _ariel_config(), fetch_limit=10)
        assert hits is None
        assert image_lane.last_unavailable_reason() == "config"
        assert len(_lane_records(caplog, logging.WARNING)) == 1
        # The breaker is open: the next query reaches neither the provider nor the database.
        calls = len(fake_provider.embed_calls)
        assert (
            await image_lane.search_images("q", _repo(pool), _ariel_config(), fetch_limit=10)
            is None
        )
        assert len(fake_provider.embed_calls) == calls

    @pytest.mark.parametrize("which", ["query_canceled", "pool_timeout"])
    @pytest.mark.usefixtures("fake_provider")
    async def test_statement_and_pool_timeouts_are_unreachable(self, which):
        from psycopg.errors import QueryCanceled
        from psycopg_pool import PoolTimeout

        if which == "query_canceled":
            pool = _pool(QueryCanceled("canceling statement due to statement timeout"))
        else:
            pool = _FakePool(error=PoolTimeout("couldn't get a connection after 2.00 sec"))
        hits = await image_lane.search_images("q", _repo(pool), _ariel_config(), fetch_limit=10)
        assert hits is None
        assert image_lane.last_unavailable_reason() == "unreachable"

    @pytest.mark.usefixtures("fake_provider")
    async def test_a_statement_past_its_timeout_gives_text_only_within_the_bound(self):
        release = asyncio.Event()

        class _SlowConn:
            def __init__(self) -> None:
                self.transactions: list[str] = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def transaction(self):
                from tests.services.ariel_search.conftest import _FakeTransaction

                return _FakeTransaction(self.transactions)

            async def execute(self, sql, params=None):  # noqa: ARG002 - psycopg shape
                if "FROM image_embeddings" in sql:
                    await asyncio.wait_for(release.wait(), 30)
                return SimpleNamespace(fetchall=_empty)

        async def _empty():
            return []

        class _SlowPool:
            def connection(self, timeout=None):  # noqa: ARG002 - psycopg_pool shape
                return _SlowConn()

        started = time.monotonic()
        hits = await image_lane.search_images(
            "q", _repo(_SlowPool()), _ariel_config(), fetch_limit=10
        )
        elapsed = time.monotonic() - started
        assert hits is None
        assert elapsed < BOUND_S
        assert image_lane.last_unavailable_reason() == "unreachable"


# ---------------------------------------------------------------------------
# Bounds of the provider step
# ---------------------------------------------------------------------------


class TestProviderBounds:
    async def test_a_sleeping_embed_call_receives_the_query_timeout(self, fake_provider):
        fake_provider.sleep_s = 30.0
        started = time.monotonic()
        hits = await image_lane.search_images("q", _repo(_pool()), _ariel_config(), fetch_limit=10)
        elapsed = time.monotonic() - started
        assert hits is None
        assert fake_provider.embed_calls[0]["timeout"] == 5
        assert elapsed < BOUND_S
        assert image_lane.last_unavailable_reason() == "unreachable"

    async def test_a_slow_resolver_probe_leaves_the_loop_free(self, llama_stub, monkeypatch):
        stub = llama_stub()
        release = threading.Event()
        monkeypatch.setattr(_local_server, "probe", lambda url, path, timeout: release.wait(10))
        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        ticker = asyncio.create_task(_ticker())
        started = time.monotonic()
        try:
            hits = await image_lane.search_images(
                "q", _repo(_pool()), _ariel_config(stub.url), fetch_limit=10
            )
        finally:
            elapsed = time.monotonic() - started
            ticker.cancel()
            release.set()
        assert hits is None
        assert elapsed < BOUND_S
        assert ticks >= elapsed / 0.05 * 0.5
        assert stub.embeddings == []


# ---------------------------------------------------------------------------
# Breaker, tracker and reason
# ---------------------------------------------------------------------------


class TestBreakerAndReason:
    async def test_repeated_reason_logs_one_warning_then_debug(self, fake_provider, caplog, clock):
        import requests

        fake_provider.error = requests.ConnectionError("refused")
        config = _ariel_config()
        with caplog.at_level(logging.DEBUG, logger="ariel"):
            for _ in range(3):
                assert (
                    await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10)
                    is None
                )
                clock.now += 31
            warnings = _lane_records(caplog, logging.WARNING)
            assert len(warnings) == 1
            assert len(_lane_records(caplog, logging.DEBUG)) == 2
            message = warnings[0].getMessage()
            assert "ConnectionError" in message
            assert "secret" not in message
            assert "fake-host:8080" in message

            fake_provider.error = None
            fake_provider.health = HealthResult(False, "serves other", "model")
            assert (
                await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10) is None
            )
            assert len(_lane_records(caplog, logging.WARNING)) == 2
            assert image_lane.last_unavailable_reason() == "model"

            clock.now += 31
            fake_provider.health = HealthResult(True, "ok", None)
            assert await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10) == {}
            assert len(_lane_records(caplog, logging.INFO)) == 1
            assert image_lane.last_unavailable_reason() is None
            assert availability.current_reason(image_lane.TRACKER_KEY) is None

    async def test_open_breaker_skips_the_provider_and_keeps_the_reason(self, fake_provider, clock):
        fake_provider.health = HealthResult(False, "down", "unreachable")
        config = _ariel_config()
        assert await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10) is None
        health_calls = len(fake_provider.health_calls)

        clock.now += 10
        assert await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10) is None
        assert len(fake_provider.health_calls) == health_calls
        assert fake_provider.embed_calls == []

        clock.now += 31
        assert image_lane.last_unavailable_reason() == "unreachable"

    async def test_unhealthy_verdict_sends_no_post(self, fake_provider):
        fake_provider.health = HealthResult(False, "serves another model", "model")
        assert (
            await image_lane.search_images("q", _repo(_pool()), _ariel_config(), fetch_limit=10)
            is None
        )
        assert fake_provider.embed_calls == []
        assert image_lane.last_unavailable_reason() == "model"

    async def test_reset_state_clears_breaker_reason_verdict_and_settings(self, fake_provider):
        fake_provider.health = HealthResult(False, "down", "unreachable")
        await image_lane.search_images("q", _repo(_pool()), _ariel_config(), fetch_limit=10)
        image_lane._reset_state()
        assert image_lane.last_unavailable_reason() is None
        assert image_lane._breaker_until is None
        assert image_lane._verdict is None
        assert image_lane._settings_cache is None


# ---------------------------------------------------------------------------
# Settings from configuration
# ---------------------------------------------------------------------------


class TestSettings:
    async def test_settings_resolve_once_per_config(self, llama_stub, monkeypatch):
        stub = llama_stub()
        real = image_lane.resolve_provider
        calls: list[Any] = []

        def _spy(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(image_lane, "resolve_provider", _spy)
        config = _ariel_config(stub.url)
        pool = _pool()
        for _ in range(2):
            assert await image_lane.search_images("q", _repo(pool), config, fetch_limit=10) == {}
        assert len(calls) == 1
        assert calls[0]["serves"] == "image_embeddings"
        settings = image_lane._settings_cache[1]  # type: ignore[index]
        assert settings.table == TABLE
        assert settings.dims == DIMS
        assert settings.model_id == MODEL
        assert len(pool.matching(f"FROM {TABLE}")) == 2

        # A different config object is resolved afresh.
        await image_lane.search_images("q", _repo(pool), _ariel_config(stub.url), fetch_limit=10)
        assert len(calls) == 2

    async def test_provider_is_an_instance_whose_health_check_gets_key_and_url(
        self, llama_stub, monkeypatch
    ):
        from osprey.models.providers.llama_cpp import LlamaCppProviderAdapter

        stub = llama_stub()
        seen: list[tuple[Any, ...]] = []
        real = LlamaCppProviderAdapter.check_embedding_health

        def _spy(self, api_key, base_url, model_id=None, timeout=10.0):
            seen.append((api_key, base_url, model_id))
            return real(self, api_key, base_url, model_id=model_id, timeout=timeout)

        monkeypatch.setattr(LlamaCppProviderAdapter, "check_embedding_health", _spy)
        config = _ariel_config(stub.url)
        assert await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10) == {}
        settings = image_lane._settings_cache[1]  # type: ignore[index]
        assert isinstance(settings.provider, LlamaCppProviderAdapter)
        assert seen == [(None, stub.url, MODEL)]

    @pytest.mark.parametrize(
        "block",
        [
            {"model": None},
            {"provider": {"name": "no-such-provider", "base_url": "http://x:1"}},
            {"provider": None},
        ],
        ids=["no-model", "unknown-provider", "no-provider"],
    )
    async def test_misconfigured_block_is_a_config_failure(self, block, caplog, clock, llama_stub):
        stub = llama_stub()
        config = _ariel_config(stub.url, **block)
        with caplog.at_level(logging.DEBUG, logger="ariel"):
            for _ in range(2):
                assert (
                    await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10)
                    is None
                )
                clock.now += 31
        assert image_lane.last_unavailable_reason() == "config"
        assert len(_lane_records(caplog, logging.WARNING)) == 1
        assert image_lane._breaker_until is not None
        assert stub.embeddings == []

    async def test_absent_block_is_a_config_failure(self):
        config = ARIELConfig.from_dict({"database": {"uri": "postgresql://localhost/test"}})
        assert await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10) is None
        assert image_lane.last_unavailable_reason() == "config"


# ---------------------------------------------------------------------------
# Against a real llama-server stand-in
# ---------------------------------------------------------------------------


class TestAgainstServer:
    async def test_server_serving_another_alias_gives_model_and_no_post(self, llama_stub):
        stub = llama_stub()
        stub.alias = "other"
        hits = await image_lane.search_images(
            "q", _repo(_pool()), _ariel_config(stub.url), fetch_limit=10
        )
        assert hits is None
        assert stub.embeddings == []
        assert image_lane.last_unavailable_reason() == "model"

    async def test_ten_queries_send_ten_posts_and_at_most_two_gets(self, llama_stub):
        from osprey.models.providers.llama_cpp import LlamaCppProviderAdapter
        from osprey.services.ariel_search.enhancement.provider_resolver import (
            resolve_reachable_base_url,
        )

        stub = llama_stub()
        # The reachable URL is already in the shared resolver cache, as it is
        # once any picture path of the process has found the server.
        assert resolve_reachable_base_url(LlamaCppProviderAdapter, stub.url) == stub.url
        stub.models_gets = 0

        config = _ariel_config(stub.url)
        for i in range(10):
            assert (
                await image_lane.search_images(f"q{i}", _repo(_pool()), config, fetch_limit=10)
                == {}
            )
        assert len(stub.embeddings) == 10
        assert stub.models_gets <= 2
        texts = [r["input"][0]["content"][0]["text"] for r in stub.embeddings]
        assert texts == [f"q{i}" for i in range(10)]

    async def test_verdict_is_reused_within_its_ttl_and_rechecked_after(self, llama_stub, clock):
        stub = llama_stub()
        config = _ariel_config(stub.url)
        assert await image_lane.search_images("warm", _repo(_pool()), config, fetch_limit=10) == {}
        gets = stub.models_gets

        for _ in range(10):
            clock.now += 1
            assert await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10) == {}
        assert stub.models_gets - gets <= 1

        posts = len(stub.embeddings)
        clock.now += image_lane.VERDICT_TTL_S + 1
        stub.alias = "other"
        assert await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10) is None
        assert image_lane.last_unavailable_reason() == "model"
        assert len(stub.embeddings) == posts

    async def test_fallback_found_after_cooldown(self, llama_stub, monkeypatch, clock):
        stub = llama_stub()
        configured = _refused_url()
        real_probe = _local_server.probe

        def _probe(url, path, timeout):
            if url.rstrip("/") == configured:
                time.sleep(3)
                return False
            return real_probe(url, path, timeout)

        monkeypatch.setattr(_local_server, "probe", _probe)
        monkeypatch.setattr(
            _local_server,
            "container_fallback_urls",
            lambda base_url, default_port: [stub.url] if base_url == configured else [],
        )
        config = _ariel_config(configured)

        assert await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10) is None
        clock.now += image_lane.IMAGE_LANE_COOLDOWN_S + 1
        hits = await image_lane.search_images("q", _repo(_pool()), config, fetch_limit=10)
        assert hits == {}
        assert image_lane.last_unavailable_reason() is None
        assert len(stub.embeddings) >= 1
