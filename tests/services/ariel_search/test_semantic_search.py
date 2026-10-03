"""Tests for the ARIEL semantic search module's settings parser.

The parser is what turns ``search_modules.semantic.settings`` from a bag of
whatever YAML happened to hold into a typed value the query path can trust. A
present-but-malformed threshold used to travel all the way to the repository
(and to the capabilities slider) as-is; these tests pin the refusal instead.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search.config import (
    ARIELConfig,
    DatabaseConfig,
    SearchModuleConfig,
)
from osprey.services.ariel_search.search.semantic import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SemanticSearchSettings,
    get_parameter_descriptors,
    semantic_provider,
    semantic_search,
)
from tests.services.ariel_search.fake_providers import make_fake_embedding_provider

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_config(settings: dict[str, Any] | None = None) -> ARIELConfig:
    """Build an ARIELConfig whose ``semantic`` module carries *settings*."""
    config = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost/ariel"))
    config.search_modules["semantic"] = SearchModuleConfig(enabled=True, settings=settings or {})
    return config


def make_wired_config(settings: dict[str, Any] | None = None) -> ARIELConfig:
    """Build a config complete enough for :func:`semantic_search` to run."""
    module: dict[str, Any] = {"enabled": True, "model": "test-model"}
    if settings is not None:
        module["settings"] = settings
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost/test"},
            "search_modules": {"semantic": module},
        }
    )


@pytest.fixture
def mock_repository():
    """A repository that records the arguments the query path hands it."""
    repo = MagicMock()
    repo.semantic_search = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_embedder():
    """An embedding provider that answers with a fixed vector."""
    embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
    embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    return embedder


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


class TestSettings:
    """``search_modules.semantic.settings`` resolution."""

    def test_defaults_when_unconfigured(self):
        settings = SemanticSearchSettings.from_ariel_config(make_config())
        assert settings.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD

    def test_defaults_when_module_absent_entirely(self):
        bare = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost/ariel"))
        parsed = SemanticSearchSettings.from_ariel_config(bare)
        assert parsed.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD

    def test_defaults_when_config_is_none(self):
        parsed = SemanticSearchSettings.from_ariel_config(None)
        assert parsed.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD

    def test_config_sets_threshold(self):
        settings = SemanticSearchSettings.from_ariel_config(
            make_config({"similarity_threshold": 0.85})
        )
        assert settings.similarity_threshold == 0.85

    @pytest.mark.parametrize("value", [0, 1])
    def test_integer_bounds_are_accepted_as_floats(self, value):
        settings = SemanticSearchSettings.from_ariel_config(
            make_config({"similarity_threshold": value})
        )
        assert settings.similarity_threshold == float(value)
        assert isinstance(settings.similarity_threshold, float)

    @pytest.mark.parametrize("bad", ["0.8", None, [], {}])
    def test_malformed_threshold_is_refused(self, bad):
        with pytest.raises(
            ValueError, match="search_modules.semantic.settings.similarity_threshold"
        ):
            SemanticSearchSettings.from_ariel_config(make_config({"similarity_threshold": bad}))

    @pytest.mark.parametrize("bad", [True, False])
    def test_boolean_threshold_is_refused(self, bad):
        with pytest.raises(
            ValueError, match="search_modules.semantic.settings.similarity_threshold"
        ):
            SemanticSearchSettings.from_ariel_config(make_config({"similarity_threshold": bad}))

    @pytest.mark.parametrize("bad", [-0.1, 1.5, -1, 2])
    def test_out_of_range_threshold_is_refused(self, bad):
        with pytest.raises(ValueError, match=r"must be a float in \[0, 1\]"):
            SemanticSearchSettings.from_ariel_config(make_config({"similarity_threshold": bad}))

    def test_settings_are_frozen(self):
        settings = SemanticSearchSettings.from_ariel_config(make_config())
        with pytest.raises(Exception):
            settings.similarity_threshold = 0.1  # type: ignore[misc]


# --------------------------------------------------------------------------
# Wiring into the query path
# --------------------------------------------------------------------------


class TestThresholdResolution:
    """What :func:`semantic_search` hands the repository."""

    @pytest.mark.asyncio
    async def test_explicit_argument_wins_over_config(self, mock_repository, mock_embedder):
        await semantic_search(
            "test",
            mock_repository,
            make_wired_config({"similarity_threshold": 0.8}),
            mock_embedder,
            similarity_threshold=0.3,
        )
        assert mock_repository.semantic_search.call_args.kwargs["similarity_threshold"] == 0.3

    @pytest.mark.asyncio
    async def test_config_value_is_used_without_an_argument(self, mock_repository, mock_embedder):
        await semantic_search(
            "test",
            mock_repository,
            make_wired_config({"similarity_threshold": 0.9}),
            mock_embedder,
        )
        assert mock_repository.semantic_search.call_args.kwargs["similarity_threshold"] == 0.9

    @pytest.mark.asyncio
    async def test_default_applies_when_settings_are_absent(self, mock_repository, mock_embedder):
        await semantic_search("test", mock_repository, make_wired_config(), mock_embedder)
        assert (
            mock_repository.semantic_search.call_args.kwargs["similarity_threshold"]
            == DEFAULT_SIMILARITY_THRESHOLD
        )

    @pytest.mark.asyncio
    async def test_malformed_settings_raise_instead_of_reaching_the_repository(
        self, mock_repository, mock_embedder
    ):
        with pytest.raises(
            ValueError, match="search_modules.semantic.settings.similarity_threshold"
        ):
            await semantic_search(
                "test",
                mock_repository,
                make_wired_config({"similarity_threshold": "high"}),
                mock_embedder,
            )
        mock_repository.semantic_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_argument_short_circuits_the_config_block(
        self, mock_repository, mock_embedder
    ):
        # A caller that named the threshold never reaches the config block, so
        # a malformed one does not fail a query that had no use for it. The
        # deployment-wide answer to a malformed block is startup validation,
        # not a per-query surprise on the one path that already has a value.
        await semantic_search(
            "test",
            mock_repository,
            make_wired_config({"similarity_threshold": "high"}),
            mock_embedder,
            similarity_threshold=0.4,
        )
        assert mock_repository.semantic_search.call_args.kwargs["similarity_threshold"] == 0.4

    @pytest.mark.asyncio
    async def test_empty_query_returns_before_parsing(self, mock_repository, mock_embedder):
        # The empty-query short circuit stays ahead of the parser, so a blank
        # box in the panel is not the thing that surfaces a config error.
        result = await semantic_search(
            "   ",
            mock_repository,
            make_wired_config({"similarity_threshold": "high"}),
            mock_embedder,
        )
        assert result == []


# --------------------------------------------------------------------------
# Parameter descriptors
# --------------------------------------------------------------------------


class TestParameterDescriptors:
    """What the capabilities API reports as the panel's starting point.

    Only the semantic module grew a ``config`` parameter alongside hybrid's.
    The keyword module's two descriptors (``include_highlights``,
    ``fuzzy_fallback``) have no counterpart in ``KeywordSearchSettings``, which
    covers ``patterns_enabled``, ``pattern_timeout_seconds`` and
    ``fuzzy_threshold`` instead, none of them a per-query parameter — there is
    no configured value for them to report, so ``keyword`` stays zero-arg
    deliberately rather than by oversight.
    """

    def test_defaults_to_the_shipped_threshold_with_no_config(self):
        descriptor = get_parameter_descriptors()[0]
        assert descriptor.name == "similarity_threshold"
        assert descriptor.default == DEFAULT_SIMILARITY_THRESHOLD

    def test_bounds_are_the_slider_the_panel_draws(self):
        descriptor = get_parameter_descriptors()[0]
        assert descriptor.param_type == "float"
        assert descriptor.min_value == 0.0
        assert descriptor.max_value == 1.0
        assert descriptor.step == 0.01
        assert descriptor.section == "Retrieval"

    def test_reports_the_configured_threshold(self):
        """The panel opens on what a query would do, not on what ships."""
        descriptor = get_parameter_descriptors(make_config({"similarity_threshold": 0.8}))[0]
        assert descriptor.default == 0.8

    def test_malformed_config_falls_back_instead_of_raising(self):
        """Describing the module survives a key the query path would refuse.

        Without the fallback the capabilities endpoint raises and the slider
        renders with no default at all; startup validation is what names the
        offending key.
        """
        descriptor = get_parameter_descriptors(make_config({"similarity_threshold": "high"}))[0]
        assert descriptor.default == DEFAULT_SIMILARITY_THRESHOLD

    @pytest.mark.parametrize("bad", [True, -0.1, 1.5, None])
    def test_every_refused_spelling_falls_back(self, bad):
        descriptor = get_parameter_descriptors(make_config({"similarity_threshold": bad}))[0]
        assert descriptor.default == DEFAULT_SIMILARITY_THRESHOLD

    def test_config_of_none_matches_the_zero_argument_call(self):
        assert get_parameter_descriptors(None) == get_parameter_descriptors()


# --------------------------------------------------------------------------
# Provider resolution
# --------------------------------------------------------------------------

_LLAMA_MODEL = "qwen3-vl-embedding-2b"


def _config(**ariel: Any) -> ARIELConfig:
    return ARIELConfig.from_dict({"database": {"uri": "postgresql://localhost/test"}, **ariel})


def _embedder_for(config: ARIELConfig):
    """The adapter the service resolves for *config*, through the real registry."""
    from osprey.services.ariel_search.service import ARIELSearchService

    return ARIELSearchService(
        config=config, pool=MagicMock(), repository=MagicMock()
    )._get_embedder()


@pytest.fixture
def no_provider_configs(monkeypatch):
    """Every provider's ``api.providers`` entry is empty; returns the names asked."""
    asked: list[str] = []

    def get_provider_config(name):
        asked.append(name)
        return {}

    monkeypatch.setattr("osprey.models.config.get_provider_config", get_provider_config)
    return asked


class TestSemanticProvider:
    """semantic_provider: the search module > the text_embedding module > embedding."""

    def test_the_search_module_provider_wins(self):
        config = _config(
            embedding={"provider": "ollama"},
            search_modules={"semantic": {"enabled": True, "provider": "openai"}},
            enhancement_modules={"text_embedding": {"provider": "llama-cpp"}},
        )
        assert semantic_provider(config) == ("openai", "ariel.search_modules.semantic.provider")

    def test_the_text_embedding_module_outranks_the_embedding_default(self):
        config = _config(
            embedding={"provider": "ollama"},
            enhancement_modules={"text_embedding": {"provider": "llama-cpp"}},
        )
        assert semantic_provider(config) == (
            "llama-cpp",
            "ariel.enhancement_modules.text_embedding.provider",
        )

    def test_the_embedding_default_is_ollama(self):
        assert semantic_provider(_config()) == ("ollama", "ariel.embedding.provider")


class TestQueryEmbeddingProvider:
    """The query is embedded by the provider that built the table it searches."""

    @pytest.mark.asyncio
    async def test_openai_semantic_provider_embeds_through_openai_without_api_base(
        self, mock_repository, no_provider_configs
    ):
        from types import SimpleNamespace
        from unittest.mock import patch

        from osprey.models.providers.openai import OpenAIProviderAdapter

        config = _config(
            embedding={"provider": "ollama"},
            search_modules={
                "semantic": {
                    "enabled": True,
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                }
            },
        )
        embedder = _embedder_for(config)
        assert type(embedder) is OpenAIProviderAdapter

        response = SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]}])
        with patch("litellm.embedding", return_value=response) as embed:
            await semantic_search("beam loss", mock_repository, config, embedder)

        embed.assert_called_once()
        assert "api_base" not in embed.call_args.kwargs
        assert "dimensions" not in embed.call_args.kwargs
        assert no_provider_configs == ["openai"]
        assert mock_repository.semantic_search.call_args.kwargs["query_embedding"] == [
            0.1,
            0.2,
            0.3,
        ]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_provider_configs")
    async def test_llama_cpp_text_table_query_is_cut_to_its_dimension(
        self, monkeypatch, mock_repository
    ):
        import requests

        from osprey.models.providers.llama_cpp import LlamaCppProviderAdapter

        monkeypatch.delenv("LLAMA_CPP_HOST", raising=False)
        seen: list[dict[str, Any]] = []
        real = LlamaCppProviderAdapter.execute_embedding

        def spy(self, texts, model_id, **kwargs):
            seen.append(kwargs)
            return real(self, texts, model_id, **kwargs)

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"embedding": [float(i % 3 + 1) for i in range(2048)]}]}

        monkeypatch.setattr(LlamaCppProviderAdapter, "execute_embedding", spy)
        monkeypatch.setattr(requests, "post", lambda url, **_kwargs: _Response())
        config = _config(
            embedding={"provider": "ollama"},
            search_modules={"semantic": {"enabled": True, "model": _LLAMA_MODEL}},
            enhancement_modules={
                "text_embedding": {
                    "provider": "llama-cpp",
                    "models": [{"name": _LLAMA_MODEL, "dimension": 1024}],
                }
            },
        )
        embedder = _embedder_for(config)
        assert type(embedder) is LlamaCppProviderAdapter

        await semantic_search("orbit kick", mock_repository, config, embedder)

        assert seen[0]["dimensions"] == 1024
        assert seen[0]["base_url"] == "http://localhost:8080"
        assert len(mock_repository.semantic_search.call_args.kwargs["query_embedding"]) == 1024

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_provider_configs")
    async def test_a_semantic_ollama_provider_still_embeds_through_ollama(
        self, monkeypatch, mock_repository
    ):
        from osprey.models.providers.ollama import OllamaProviderAdapter

        seen: list[dict[str, Any]] = []

        def record(_self, **kwargs):
            seen.append(kwargs)
            return [[0.1, 0.2, 0.3]]

        monkeypatch.setattr(OllamaProviderAdapter, "execute_embedding", record)
        config = _config(
            search_modules={
                "semantic": {"enabled": True, "provider": "ollama", "model": _LLAMA_MODEL}
            },
            enhancement_modules={
                "text_embedding": {
                    "provider": "llama-cpp",
                    "models": [{"name": _LLAMA_MODEL, "dimension": 1024}],
                }
            },
        )
        embedder = _embedder_for(config)
        assert type(embedder) is OllamaProviderAdapter

        await semantic_search("orbit kick", mock_repository, config, embedder)

        assert len(seen) == 1
        assert "dimensions" not in seen[0]
        assert seen[0]["base_url"] == OllamaProviderAdapter.default_base_url


class TestProviderConfig:
    """provider_config, when given, is used as is; None resolves it."""

    @pytest.mark.asyncio
    async def test_a_given_provider_config_is_used_without_a_lookup(
        self, mock_repository, no_provider_configs
    ):
        embedder = make_fake_embedding_provider()()

        await semantic_search(
            "q",
            mock_repository,
            make_wired_config(),
            embedder,
            provider_config={"base_url": "http://given:1", "api_key": "k"},
        )

        assert no_provider_configs == []
        assert embedder.calls[0]["base_url"] == "http://given:1"
        assert embedder.calls[0]["api_key"] == "k"

    @pytest.mark.asyncio
    async def test_none_resolves_through_semantic_provider(
        self, mock_repository, no_provider_configs
    ):
        embedder = make_fake_embedding_provider(default_base_url="http://default:2")()
        config = _config(
            search_modules={"semantic": {"enabled": True, "model": "m"}},
            enhancement_modules={"text_embedding": {"provider": "llama-cpp"}},
        )

        await semantic_search("q", mock_repository, config, embedder)

        assert no_provider_configs == ["llama-cpp"]
        assert embedder.calls[0]["base_url"] == "http://default:2"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_provider_configs")
    async def test_a_truncating_provider_without_a_model_entry_uses_the_settings_dimension(
        self, mock_repository
    ):
        embedder = make_fake_embedding_provider(truncates_to_dimensions=True)()

        await semantic_search(
            "q", mock_repository, make_wired_config({"embedding_dimension": 3}), embedder
        )

        assert embedder.calls[0]["dimensions"] == 3


# --------------------------------------------------------------------------
# Reachable server, off the event loop
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic_local_server(monkeypatch):
    """No test here probes a real port unless it says so; the cache starts empty."""
    from osprey.models.providers import _local_server

    _local_server.reset_cache()
    monkeypatch.setattr(_local_server, "probe", lambda url, path, timeout: False)
    yield
    _local_server.reset_cache()


def _fallback_config(port: int) -> ARIELConfig:
    return _config(
        search_modules={"semantic": {"enabled": True, "model": "m"}},
        enhancement_modules={
            "text_embedding": {
                "provider": {"name": "fakellama", "base_url": f"http://127.0.0.1:{port}"},
                "models": [{"name": "m", "dimension": 4}],
            }
        },
    )


class TestReachableQueryServer:
    """A llama-cpp-style provider is found through its fallbacks, off the event loop."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_provider_configs")
    async def test_a_refusing_localhost_reaches_the_fallback_with_one_walk(
        self, monkeypatch, mock_repository
    ):
        from osprey.models.providers import _local_server
        from tests.services.ariel_search.test_provider_resolver import (
            _fallback_class,
            _free_port,
            _ProbeRecorder,
            _register,
            _Stub,
        )

        monkeypatch.delenv("X_HOST", raising=False)
        stub = _Stub()
        try:
            probes = _ProbeRecorder()
            probes.docker_target = stub.url
            monkeypatch.setattr(_local_server, "probe", probes)
            cls = _fallback_class()
            _register(monkeypatch, fakellama=cls)
            port = _free_port()
            config = _fallback_config(port)
            embedder = _embedder_for(config)
            assert type(embedder) is cls

            await semantic_search("beam", mock_repository, config, embedder)
            walked = len(probes.calls)
            await semantic_search("orbit", mock_repository, config, embedder)
        finally:
            stub.stop()

        fallback = f"http://host.docker.internal:{port}"
        assert [call["base_url"] for call in cls.calls] == [fallback, fallback]
        assert len(probes.calls) == walked

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_provider_configs")
    async def test_resolving_the_embedder_does_no_network_io(self, monkeypatch):
        from osprey.models.providers import _local_server
        from tests.services.ariel_search.test_provider_resolver import (
            _fallback_class,
            _register,
        )

        def no_network(*args, **kwargs):
            raise AssertionError("embedder resolution touched the network")

        monkeypatch.setattr(_local_server, "probe", no_network)
        cls = _fallback_class()
        _register(monkeypatch, fakellama=cls)

        assert type(_embedder_for(_fallback_config(1))) is cls

    @staticmethod
    def _slow_probe(monkeypatch, seconds: float) -> None:
        from osprey.models.providers import _local_server
        from tests.services.ariel_search.test_provider_resolver import (
            _fallback_class,
            _register,
        )

        first = threading.Event()

        def probe(_url, _path, _timeout):
            if not first.is_set():
                first.set()
                time.sleep(seconds)
            return False

        monkeypatch.delenv("X_HOST", raising=False)
        monkeypatch.setattr(_local_server, "probe", probe)
        _register(monkeypatch, fakellama=_fallback_class())

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_provider_configs")
    async def test_a_hanging_probe_never_stalls_the_event_loop(self, monkeypatch, mock_repository):
        self._slow_probe(monkeypatch, 3.0)
        config = _fallback_config(1)
        embedder = _embedder_for(config)
        ticks: list[float] = []

        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.05)

        task = asyncio.create_task(ticker())
        started = time.monotonic()
        try:
            await semantic_search("beam", mock_repository, config, embedder)
        finally:
            task.cancel()

        assert time.monotonic() - started >= 2.5
        gaps = [later - earlier for earlier, later in zip(ticks, ticks[1:], strict=False)]
        assert len(ticks) > 30
        assert max(gaps) < 0.5

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_provider_configs")
    async def test_a_hanging_probe_never_delays_to_thread(self, monkeypatch, mock_repository):
        self._slow_probe(monkeypatch, 3.0)
        config = _fallback_config(1)
        embedder = _embedder_for(config)

        search = asyncio.create_task(semantic_search("beam", mock_repository, config, embedder))
        await asyncio.sleep(0.2)
        started = time.monotonic()
        assert await asyncio.to_thread(lambda: 42) == 42
        elapsed = time.monotonic() - started
        assert not search.done()
        await search

        assert elapsed < 0.5

    def test_search_calls_run_on_their_own_bounded_pool(self):
        from osprey.services.ariel_search.search._offload import SEARCH_CALL_POOL

        assert SEARCH_CALL_POOL._max_workers == 4

    @pytest.mark.asyncio
    async def test_run_search_call_times_out(self):
        from osprey.services.ariel_search.search._offload import run_search_call

        with pytest.raises(TimeoutError):
            await run_search_call(time.sleep, 0.5, timeout_s=0.05)
