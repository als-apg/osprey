"""Tests for ARIEL enhancement modules.

Tests for base enhancement module, factory, and module implementations.
"""

import logging
import os
import sys
from datetime import UTC, datetime
from unittest import mock

import pytest

from osprey.models.provider_registry import ProviderRegistry
from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig
from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule
from osprey.services.ariel_search.enhancement.factory import (
    create_enhancers_from_config,
    get_enhancer_names,
)
from osprey.services.ariel_search.enhancement.qmd_export import (
    TOUCH_MARKER_NAME,
    QmdExportModule,
)
from osprey.services.ariel_search.enhancement.semantic_processor import (
    SemanticProcessorModule,
    SemanticProcessorResult,
)
from osprey.services.ariel_search.enhancement.text_embedding import (
    TextEmbeddingModule,
)
from osprey.services.ariel_search.enhancement.text_embedding import embedder as embedder_mod
from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
    embedding_input,
    fit_to_input_limit,
    max_input_tokens,
)
from tests.services.ariel_search.conftest import _FakeConnection
from tests.services.ariel_search.fake_providers import (
    make_fake_embedding_provider,
    ollama_text_embedder,
)

TABLES_QUERY = "information_schema.tables"
COLUMNS_QUERY = "information_schema.columns"

#: The registry lookup as shipped, captured before any test routes it.
_REAL_GET_PROVIDER = ProviderRegistry.get_provider


def _route_provider(monkeypatch, provider_cls, lookups=None):
    """Answer every ``ProviderRegistry.get_provider`` lookup with *provider_cls*.

    Returns *provider_cls*; the names asked for are appended to *lookups*.
    """

    def get_provider(_self, name):
        if lookups is not None:
            lookups.append(name)
        return provider_cls

    monkeypatch.setattr(ProviderRegistry, "get_provider", get_provider)
    return provider_cls


@pytest.fixture(autouse=True)
def _input_key_absence_unlogged():
    """Start every test with no embedding table reported as lacking ``input_key``."""
    embedder_mod._INPUT_KEY_ABSENT_LOGGED.clear()
    yield
    embedder_mod._INPUT_KEY_ABSENT_LOGGED.clear()


@pytest.fixture(autouse=True)
def _fake_provider_is_registered(monkeypatch):
    """Register a default fake embedding provider under the name ``fake``.

    Every other name keeps its real registry lookup.
    """
    fake_cls = make_fake_embedding_provider()

    def get_provider(self, name):
        return fake_cls if name == "fake" else _REAL_GET_PROVIDER(self, name)

    monkeypatch.setattr(ProviderRegistry, "get_provider", get_provider)


class TestEnhancementFactory:
    """Tests for enhancement module factory."""

    def test_get_enhancer_names(self):
        """get_enhancer_names returns correct list."""
        names = get_enhancer_names()
        assert names == [
            "image_caption",
            "semantic_processor",
            "text_embedding",
            "image_embedding",
            "qmd_export",
        ]

    def test_create_enhancers_no_modules_enabled(self):
        """Factory returns empty list when no modules enabled."""
        config = ARIELConfig(
            database=DatabaseConfig(uri="postgresql://localhost:5432/test"),
            enhancement_modules={},
        )
        enhancers = create_enhancers_from_config(config)
        assert enhancers == []

    def test_create_enhancers_text_embedding_only(self):
        """Factory creates only text_embedding when configured."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "text_embedding": {
                        "enabled": True,
                        "models": [{"name": "nomic-embed-text", "dimension": 768}],
                    },
                },
            }
        )
        enhancers = create_enhancers_from_config(config)
        assert len(enhancers) == 1
        assert enhancers[0].name == "text_embedding"

    def test_create_enhancers_both_modules(self):
        """Factory creates both modules in execution order."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "semantic_processor": {"enabled": True, "provider": "ollama"},
                    "text_embedding": {
                        "enabled": True,
                        "models": [{"name": "nomic-embed-text", "dimension": 768}],
                    },
                },
            }
        )
        enhancers = create_enhancers_from_config(config)
        assert len(enhancers) == 2
        # Check order: semantic_processor first, then text_embedding
        assert enhancers[0].name == "semantic_processor"
        assert enhancers[1].name == "text_embedding"


class TestTextEmbeddingModule:
    """Tests for TextEmbeddingModule."""

    def test_module_name(self):
        """Module has correct name."""
        module = TextEmbeddingModule()
        assert module.name == "text_embedding"

    def test_module_has_migration(self):
        """Module provides migration class."""
        module = TextEmbeddingModule()
        from osprey.services.ariel_search.enhancement.text_embedding.migration import (
            TextEmbeddingMigration,
        )

        assert module.migration == TextEmbeddingMigration

    def test_configure_sets_models(self):
        """configure() sets models from config."""
        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [
                    {"name": "nomic-embed-text", "dimension": 768},
                    {"name": "all-minilm", "dimension": 384},
                ],
            }
        )
        assert len(module._models) == 2
        assert module._models[0]["name"] == "nomic-embed-text"

    def test_is_base_enhancement_module(self):
        """Module is a BaseEnhancementModule."""
        module = TextEmbeddingModule()
        assert isinstance(module, BaseEnhancementModule)


class TestSemanticProcessorModule:
    """Tests for SemanticProcessorModule."""

    def test_module_name(self):
        """Module has correct name."""
        module = SemanticProcessorModule()
        assert module.name == "semantic_processor"

    def test_module_has_migration(self):
        """Module provides migration class."""
        module = SemanticProcessorModule()
        from osprey.services.ariel_search.enhancement.semantic_processor.migration import (
            SemanticProcessorMigration,
        )

        assert module.migration == SemanticProcessorMigration

    def test_configure_sets_prompt_template(self):
        """configure() sets custom prompt template."""
        module = SemanticProcessorModule()
        module.configure(
            {
                "provider": "ollama",
                "prompt_template": "Custom prompt: {text}",
            }
        )
        assert module._prompt_template == "Custom prompt: {text}"

    def test_is_base_enhancement_module(self):
        """Module is a BaseEnhancementModule."""
        module = SemanticProcessorModule()
        assert isinstance(module, BaseEnhancementModule)


class TestSemanticProcessorResult:
    """Tests for SemanticProcessorResult model."""

    def test_basic_creation(self):
        """Can create SemanticProcessorResult."""
        result = SemanticProcessorResult(
            keywords=["vacuum", "pump", "VP-103"],
            summary="Replaced vacuum pump VP-103.",
        )
        assert result.keywords == ["vacuum", "pump", "VP-103"]
        assert result.summary == "Replaced vacuum pump VP-103."

    def test_from_dict(self):
        """Can create from dict."""
        data = {
            "keywords": ["rf", "cavity", "fault"],
            "summary": "RF cavity fault detected.",
        }
        result = SemanticProcessorResult(**data)
        assert result.keywords == data["keywords"]
        assert result.summary == data["summary"]

    def test_empty_keywords(self):
        """Can have empty keywords list."""
        result = SemanticProcessorResult(keywords=[], summary="Short entry.")
        assert result.keywords == []


class TestParseResponse:
    """Tests for SemanticProcessorModule._parse_response."""

    def test_parse_json_response(self):
        """Parses plain JSON response."""
        module = SemanticProcessorModule()
        response = '{"keywords": ["test", "entry"], "summary": "Test summary."}'
        result = module._parse_response(response)
        assert result is not None
        assert result.keywords == ["test", "entry"]
        assert result.summary == "Test summary."

    def test_parse_json_with_markdown_code_block(self):
        """Parses JSON wrapped in markdown code block."""
        module = SemanticProcessorModule()
        response = '```json\n{"keywords": ["vacuum"], "summary": "Summary"}\n```'
        result = module._parse_response(response)
        assert result is not None
        assert result.keywords == ["vacuum"]

    def test_parse_invalid_json(self):
        """Returns None for invalid JSON."""
        module = SemanticProcessorModule()
        response = "This is not JSON"
        result = module._parse_response(response)
        assert result is None

    def test_parse_missing_field(self):
        """Returns None if required field missing."""
        module = SemanticProcessorModule()
        response = '{"keywords": ["test"]}'  # Missing summary
        result = module._parse_response(response)
        assert result is None


class TestGetEnhancementModuleConfig:
    """Tests for ARIELConfig.get_enhancement_module_config."""

    def test_returns_none_for_unconfigured_module(self):
        """Returns None for module not in config."""
        config = ARIELConfig(
            database=DatabaseConfig(uri="postgresql://localhost:5432/test"),
        )
        result = config.get_enhancement_module_config("text_embedding")
        assert result is None

    def test_returns_config_dict_for_configured_module(self):
        """Returns config dict for configured module."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "text_embedding": {
                        "enabled": True,
                        "models": [{"name": "nomic-embed-text", "dimension": 768}],
                    },
                },
            }
        )
        result = config.get_enhancement_module_config("text_embedding")
        assert result is not None
        assert result["enabled"] is True
        assert len(result["models"]) == 1
        assert result["models"][0]["name"] == "nomic-embed-text"

    def test_includes_settings_in_config(self):
        """Config dict includes settings."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "semantic_processor": {
                        "enabled": True,
                        "settings": {"custom_setting": "value"},
                    },
                },
            }
        )
        result = config.get_enhancement_module_config("semantic_processor")
        assert result is not None
        assert result["custom_setting"] == "value"


class TestSemanticProcessorPromptGeneration:
    """Tests for SemanticProcessorModule prompt template."""

    def test_default_prompt_template(self):
        """Default prompt template contains expected placeholders."""
        module = SemanticProcessorModule()
        assert "{text}" in module._prompt_template
        assert "keywords" in module._prompt_template.lower()

    def test_custom_prompt_template(self):
        """Configure can set custom prompt template."""
        module = SemanticProcessorModule()
        module.configure({"provider": "ollama", "prompt_template": "Custom: {text}"})
        assert module._prompt_template == "Custom: {text}"

    def test_prompt_template_format(self):
        """Prompt template can be formatted with text."""
        module = SemanticProcessorModule()
        formatted = module._prompt_template.format(text="Entry content here")
        assert "Entry content here" in formatted


class TestBaseEnhancementModuleAbstract:
    """Tests for BaseEnhancementModule interface."""

    def test_base_module_requires_name(self):
        """BaseEnhancementModule requires name property."""
        with pytest.raises(TypeError):

            class InvalidModule(BaseEnhancementModule):
                pass

            InvalidModule()

    def test_concrete_module_implements_name(self):
        """Concrete modules implement name property."""
        module = TextEmbeddingModule()
        assert hasattr(module, "name")
        assert isinstance(module.name, str)


class TestEnhancementExports:
    """Tests for enhancement module exports."""

    def test_base_enhancement_module_exported(self):
        """BaseEnhancementModule is exported from enhancement package."""
        from osprey.services.ariel_search.enhancement import BaseEnhancementModule

        assert BaseEnhancementModule is not None

    def test_factory_functions_exported(self):
        """Factory functions are exported from enhancement package."""
        from osprey.services.ariel_search.enhancement import (
            create_enhancers_from_config,
            get_enhancer_names,
        )

        assert callable(create_enhancers_from_config)
        assert callable(get_enhancer_names)


class TestSemanticProcessorModuleConfig:
    """Tests for SemanticProcessorModule configure method."""

    def test_configure_empty_dict(self):
        """Configure with empty dict names the provider key it needs."""
        module = SemanticProcessorModule()
        with pytest.raises(ValueError, match="semantic_processor.provider"):
            module.configure({})

    def test_configure_with_model_config(self):
        """Configure sets model_config for LLM calls."""
        module = SemanticProcessorModule()
        module.configure({"provider": "ollama", "model": {"name": "llama3"}})
        assert module._model_config == {"provider": "ollama", "name": "llama3"}


class TestTextEmbeddingModuleConfig:
    """Tests for TextEmbeddingModule configure method."""

    def test_configure_empty_dict(self):
        """Configure with empty dict uses defaults."""
        module = TextEmbeddingModule()
        module.configure({})
        # Should not raise

    def test_configure_with_models(self):
        """Configure sets models."""
        module = TextEmbeddingModule()
        module.configure({"models": [{"name": "test-model", "dimension": 512}]})
        assert len(module._models) == 1


class TestSemanticProcessorHealthCheck:
    """Tests for SemanticProcessorModule health_check."""

    @pytest.mark.asyncio
    async def test_health_check_no_llm(self):
        """Health check returns unhealthy when no LLM configured."""
        module = SemanticProcessorModule()
        result = await module.health_check()
        # Without LLM configured, should be unhealthy
        assert isinstance(result.reachable, bool)
        assert isinstance(result.message, str)


class TestTextEmbeddingModuleEmbedder:
    """Tests for TextEmbeddingModule embedder configuration."""

    def test_models_initially_empty(self):
        """Models list is empty before configuration."""
        module = TextEmbeddingModule()
        assert module._models == []

    def test_configure_sets_models(self):
        """Configure with models sets the model list."""
        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [{"name": "test", "dimension": 768}],
            }
        )
        assert len(module._models) == 1
        assert module._models[0]["name"] == "test"


class TestTextEmbeddingModuleHealthCheck:
    """Tests for TextEmbeddingModule health_check."""

    @pytest.mark.asyncio
    async def test_health_check_no_embedder(self):
        """Health check returns unhealthy when no embedder configured."""
        module = TextEmbeddingModule()
        result = await module.health_check()
        # Without embedder configured, should be unhealthy
        assert isinstance(result.reachable, bool)
        assert isinstance(result.message, str)


class TestSemanticProcessorEnhanceEmptyEntry:
    """Tests for SemanticProcessorModule enhance method edge cases."""

    @pytest.mark.asyncio
    async def test_enhance_empty_text(self):
        """Enhance skips entries with empty text."""
        from unittest.mock import AsyncMock, MagicMock

        module = SemanticProcessorModule()

        # Create entry with empty text
        entry = {
            "entry_id": "test-empty",
            "source_system": "test",
            "raw_text": "",  # Empty text
            "timestamp": None,
            "author": "test",
            "attachments": [],
            "metadata": {},
        }

        # Mock repository
        repo = MagicMock()
        repo.upsert_entry = AsyncMock()

        # Should handle gracefully
        await module.enhance(entry, repo)
        # Result depends on implementation - may skip or process


class TestSemanticProcessorParseResponseEdgeCases:
    """Tests for SemanticProcessorModule._parse_response edge cases."""

    def test_parse_response_with_extra_fields(self):
        """Parse ignores extra fields in JSON."""
        module = SemanticProcessorModule()
        response = '{"keywords": ["test"], "summary": "Test.", "extra": "ignored"}'
        result = module._parse_response(response)
        assert result is not None
        assert result.keywords == ["test"]

    def test_parse_response_empty_string(self):
        """Parse handles empty string."""
        module = SemanticProcessorModule()
        result = module._parse_response("")
        assert result is None

    def test_parse_response_whitespace_only(self):
        """Parse handles whitespace-only string."""
        module = SemanticProcessorModule()
        result = module._parse_response("   \n   ")
        assert result is None


class TestTextEmbeddingModuleMoreConfig:
    """Additional tests for TextEmbeddingModule configuration."""

    def test_multiple_models_configuration(self):
        """Configure supports multiple models."""
        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [
                    {"name": "model1", "dimension": 768},
                    {"name": "model2", "dimension": 1024},
                ],
            }
        )
        assert len(module._models) == 2

    def test_configure_overwrites_models(self):
        """Configure overwrites existing model configuration."""
        module = TextEmbeddingModule()
        module.configure({"models": [{"name": "first", "dimension": 768}]})
        module.configure({"models": [{"name": "second", "dimension": 512}]})
        assert len(module._models) == 1
        assert module._models[0]["name"] == "second"


class TestTextEmbeddingModuleEnhance:
    """Tests for TextEmbeddingModule.enhance method."""

    @pytest.fixture
    def module(self):
        """Create configured module."""
        from osprey.services.ariel_search.enhancement.text_embedding import TextEmbeddingModule

        m = TextEmbeddingModule()
        m.configure(
            {
                "models": [{"name": "test-model", "dimension": 768, "max_input_tokens": 8192}],
                "provider": {"base_url": "http://localhost:11434"},
            }
        )
        return m

    @pytest.fixture
    def sample_entry(self):
        """Sample entry for testing."""
        from datetime import datetime

        return {
            "entry_id": "entry-001",
            "source_system": "Example eLog",
            "timestamp": datetime(2024, 1, 15, tzinfo=UTC),
            "author": "jsmith",
            "raw_text": "Beam current at 500mA.",
            "attachments": [],
            "metadata": {},
        }

    @pytest.mark.asyncio
    async def test_enhance_no_models_configured(self):
        """Enhance logs warning when no models configured."""
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.enhancement.text_embedding import TextEmbeddingModule

        module = TextEmbeddingModule()  # No models configured
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()

        entry = {
            "entry_id": "entry-001",
            "raw_text": "Test content",
        }

        await module.enhance(entry, mock_conn)
        mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_enhance_empty_text_skipped(self, module):
        """Enhance skips entries with empty text (no INSERT executed)."""
        from unittest.mock import AsyncMock, MagicMock

        mock_result = MagicMock()
        mock_result.fetchone = AsyncMock(return_value=(True,))
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        entry = {
            "entry_id": "entry-empty",
            "raw_text": "   ",
        }

        await module.enhance(entry, mock_conn)
        # Only the table existence check should have been called, no INSERT
        for call in mock_conn.execute.call_args_list:
            assert "INSERT" not in str(call)

    @pytest.mark.asyncio
    async def test_enhance_single_model_failure_raises(self, module, sample_entry):
        """Single model failure raises RuntimeError (not silently swallowed)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_result = MagicMock()
        mock_result.fetchone = AsyncMock(return_value=(True,))
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        with patch.object(module, "_get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.execute_embedding.side_effect = RuntimeError("Ollama 500")
            mock_provider.default_base_url = "http://localhost:11434"
            mock_get.return_value = mock_provider

            with pytest.raises(RuntimeError, match="All embedding models failed"):
                await module.enhance(sample_entry, mock_conn)

    @pytest.mark.asyncio
    async def test_enhance_all_models_fail_raises(self, sample_entry):
        """All models failing raises RuntimeError with all error details."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from osprey.services.ariel_search.enhancement.text_embedding import TextEmbeddingModule

        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [
                    {"name": "model-a", "dimension": 768},
                    {"name": "model-b", "dimension": 384},
                ],
                "provider": {"base_url": "http://localhost:11434"},
            }
        )

        mock_result = MagicMock()
        mock_result.fetchone = AsyncMock(return_value=(True,))
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        with patch.object(module, "_get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.execute_embedding.side_effect = RuntimeError("connection refused")
            mock_provider.default_base_url = "http://localhost:11434"
            mock_get.return_value = mock_provider

            with pytest.raises(RuntimeError, match="All embedding models failed") as exc_info:
                await module.enhance(sample_entry, mock_conn)
            assert "model-a" in str(exc_info.value)
            assert "model-b" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_enhance_partial_model_failure_succeeds(self, sample_entry):
        """Partial failure (some models succeed) does not raise."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from osprey.services.ariel_search.enhancement.text_embedding import TextEmbeddingModule

        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [
                    {"name": "model-a", "dimension": 768},
                    {"name": "model-b", "dimension": 384},
                ],
                "provider": {"base_url": "http://localhost:11434"},
            }
        )

        mock_result = MagicMock()
        mock_result.fetchone = AsyncMock(return_value=(True,))
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [[0.1] * 768]  # model-a succeeds
            raise RuntimeError("Ollama 500")  # model-b fails

        with patch.object(module, "_get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.execute_embedding.side_effect = side_effect
            mock_provider.default_base_url = "http://localhost:11434"
            mock_get.return_value = mock_provider

            # Should NOT raise — partial success is acceptable
            await module.enhance(sample_entry, mock_conn)

    @pytest.mark.asyncio
    async def test_enhance_empty_embedding_result_counted_as_error(self, module, sample_entry):
        """Empty embedding result is counted as an error."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_result = MagicMock()
        mock_result.fetchone = AsyncMock(return_value=(True,))
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        with patch.object(module, "_get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.execute_embedding.return_value = []  # Empty result
            mock_provider.default_base_url = "http://localhost:11434"
            mock_get.return_value = mock_provider

            with pytest.raises(RuntimeError, match="All embedding models failed"):
                await module.enhance(sample_entry, mock_conn)


class TestSemanticProcessorEnhanceWithLLM:
    """Tests for SemanticProcessorModule enhance with mocked LLM."""

    @pytest.fixture
    def module(self):
        """Create configured module."""
        m = SemanticProcessorModule()
        m.configure(
            {
                "provider": "ollama",
                "model": {"model_id": "llama3"},
            }
        )
        return m

    @pytest.mark.asyncio
    async def test_enhance_empty_text(self, module):
        """Enhance handles empty text."""
        from unittest.mock import AsyncMock, MagicMock

        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()

        entry = {
            "entry_id": "entry-empty",
            "raw_text": "",
        }

        # Should handle gracefully
        await module.enhance(entry, mock_conn)


class TestGetEnhancementModuleConfigDetails:
    """Additional tests for enhancement module config retrieval."""

    def test_text_embedding_config_with_provider(self):
        """Text embedding config includes provider settings."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "text_embedding": {
                        "enabled": True,
                        "models": [{"name": "nomic-embed-text", "dimension": 768}],
                        "settings": {"provider": {"base_url": "http://localhost:11434"}},
                    },
                },
            }
        )

        # Access the config directly
        te_config = config.enhancement_modules.get("text_embedding")
        assert te_config is not None
        assert te_config.enabled is True
        assert te_config.models is not None
        assert len(te_config.models) == 1
        assert te_config.models[0].name == "nomic-embed-text"


class TestSemanticProcessorConfigureModelResolution:
    """Tests for SemanticProcessorModule.configure model handling."""

    def test_configure_takes_the_model_id_verbatim(self):
        """A provider plus a model_id reaches the model config unchanged."""
        module = SemanticProcessorModule()
        module.configure(
            {
                "provider": "ollama",
                "model": {"model_id": "gpt-oss:20b", "temperature": 0.1},
            }
        )

        assert module._model_config == {
            "provider": "ollama",
            "model_id": "gpt-oss:20b",
            "temperature": 0.1,
        }

    def test_configure_without_provider_raises(self):
        """No provider is an error, not a silent fall-through to a default."""
        module = SemanticProcessorModule()
        with pytest.raises(ValueError, match="semantic_processor.provider"):
            module.configure({"model": {"model_id": "llama3"}})

    def test_configure_with_provider_but_no_model_id_reads_no_config(self, monkeypatch):
        """The main model is looked up at the first call, never at configure time."""

        def boom(*_args, **_kwargs):
            raise AssertionError("configure must not load config.yml")

        monkeypatch.setattr("osprey.utils.config.load_config", boom)

        module = SemanticProcessorModule()
        module.configure({"provider": "ollama", "model": {"temperature": 0.2}})

        assert module._model_config == {"provider": "ollama", "temperature": 0.2}


class TestSemanticProcessorProcessText:
    """Tests for SemanticProcessorModule._process_text with a patched LLM."""

    @pytest.fixture
    def module(self):
        """Module configured with a model config."""
        m = SemanticProcessorModule()
        m.configure({"provider": "ollama", "model": {"model_id": "llama3"}})
        return m

    @pytest.mark.asyncio
    async def test_process_text_parses_llm_json(self, module, monkeypatch):
        """A JSON completion becomes a SemanticProcessorResult."""
        calls = []

        def fake_completion(message, model_config=None):
            calls.append({"message": message, "model_config": model_config})
            return '{"keywords": ["vacuum", "pump"], "summary": "Pump replaced."}'

        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            fake_completion,
        )

        result = await module._process_text("Replaced vacuum pump VP-103.")

        assert result is not None
        assert result.keywords == ["vacuum", "pump"]
        assert result.summary == "Pump replaced."
        assert calls[0]["model_config"] == {"provider": "ollama", "model_id": "llama3"}
        assert "Replaced vacuum pump VP-103." in calls[0]["message"]

    @pytest.mark.asyncio
    async def test_process_text_truncates_entry_to_8000_chars(self, module, monkeypatch):
        """Entry text is clipped to 8000 characters before prompting."""
        calls = []

        def fake_completion(message, model_config=None):  # noqa: ARG001 - the get_chat_completion signature
            calls.append(message)
            return '{"keywords": [], "summary": "s"}'

        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            fake_completion,
        )

        await module._process_text("a" * 8000 + "TAIL")

        assert "TAIL" not in calls[0]
        assert "a" * 8000 in calls[0]

    @pytest.mark.asyncio
    async def test_process_text_stringifies_non_string_response(self, module, monkeypatch):
        """A non-str completion is coerced with str() before parsing."""

        class _JSONish:
            def __str__(self) -> str:
                return '{"keywords": ["rf"], "summary": "RF fault."}'

        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            lambda message, model_config=None: _JSONish(),
        )

        result = await module._process_text("RF cavity fault.")

        assert result is not None
        assert result.keywords == ["rf"]

    @pytest.mark.asyncio
    async def test_process_text_passes_none_model_config_when_unconfigured(self, monkeypatch):
        """An unconfigured module sends model_config=None, not an empty dict."""
        calls = []

        def fake_completion(message, model_config=None):  # noqa: ARG001 - the get_chat_completion signature
            calls.append(model_config)
            return '{"keywords": [], "summary": "s"}'

        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            fake_completion,
        )

        await SemanticProcessorModule()._process_text("some text")

        assert calls == [None]

    @pytest.mark.asyncio
    async def test_process_text_import_error_degrades_gracefully(self, module, monkeypatch, caplog):
        """An unimportable completion module yields None plus a warning."""
        monkeypatch.setitem(sys.modules, "osprey.models.completion", None)

        with caplog.at_level(logging.WARNING, logger="ariel"):
            result = await module._process_text("some text")

        assert result is None
        assert "osprey.models.completion not available" in caplog.text

    @pytest.mark.asyncio
    async def test_process_text_llm_failure_returns_none(self, module, monkeypatch, caplog):
        """An LLM exception is logged and reported as no result."""

        def boom(message, model_config=None):  # noqa: ARG001 - the get_chat_completion signature
            raise RuntimeError("model server down")

        monkeypatch.setattr("osprey.models.completion.get_chat_completion", boom)

        with caplog.at_level(logging.WARNING, logger="ariel"):
            result = await module._process_text("some text")

        assert result is None
        assert "LLM processing failed: model server down" in caplog.text


class TestSemanticProcessorEnhanceBranches:
    """Tests for SemanticProcessorModule.enhance store/skip/swallow branches."""

    @pytest.fixture
    def module(self):
        """Module configured with a model config."""
        m = SemanticProcessorModule()
        m.configure({"provider": "ollama", "model": {"model_id": "llama3"}})
        return m

    @pytest.fixture
    def entry(self):
        """Non-empty sample entry."""
        return {"entry_id": "entry-001", "raw_text": "Replaced vacuum pump VP-103."}

    @pytest.mark.asyncio
    async def test_enhance_stores_keywords_and_summary(self, module, entry, monkeypatch):
        """A parsed result is written to enhanced_entries."""
        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            lambda message, model_config=None: (
                '{"keywords": ["vacuum", "pump"], "summary": "Pump replaced."}'
            ),
        )
        conn = _FakeConnection()

        await module.enhance(entry, conn)

        assert len(conn.calls) == 1
        sql, params = conn.calls[0]
        assert "UPDATE enhanced_entries" in sql
        assert params == [["vacuum", "pump"], "Pump replaced.", "entry-001"]

    @pytest.mark.asyncio
    async def test_enhance_exposes_result_to_downstream_qmd_export(
        self, module, entry, monkeypatch, tmp_path
    ):
        """The qmd exporter receives enrichment produced earlier in the same pass."""
        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            lambda message, model_config=None: (
                '{"keywords": ["vacuum", "pump"], "summary": "Pump replaced."}'
            ),
        )
        qmd_export = QmdExportModule()
        qmd_export.configure({"mirror_path": str(tmp_path)})
        conn = _FakeConnection()

        await module.enhance(entry, conn)
        await qmd_export.enhance(entry, conn)

        (document_path,) = tmp_path.rglob("*.md")
        document = document_path.read_text(encoding="utf-8")
        assert "Summary: Pump replaced." in document
        assert "Keywords: vacuum, pump." in document

    @pytest.mark.asyncio
    async def test_enhance_skips_store_when_parsing_fails(self, module, entry, monkeypatch):
        """An unparseable completion leaves the database untouched."""
        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            lambda message, model_config=None: "not JSON at all",
        )
        conn = _FakeConnection()

        await module.enhance(entry, conn)

        assert conn.calls == []

    @pytest.mark.asyncio
    async def test_enhance_swallows_store_failure(self, module, entry, monkeypatch, caplog):
        """A failing UPDATE is logged, not raised — enhancement is best-effort."""
        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            lambda message, model_config=None: '{"keywords": ["a"], "summary": "s"}',
        )
        conn = _FakeConnection(results=[RuntimeError("connection reset")])

        with caplog.at_level(logging.WARNING, logger="ariel"):
            await module.enhance(entry, conn)

        assert "Failed to process entry entry-001: connection reset" in caplog.text


class TestSemanticProcessorParseResponseFences:
    """Tests for _parse_response markdown-fence stripping."""

    def test_parse_response_plain_triple_backtick_fence(self):
        """A bare ``` fence (no json tag) is stripped before parsing."""
        module = SemanticProcessorModule()
        response = '```\n{"keywords": ["beam"], "summary": "Beam lost."}\n```'

        result = module._parse_response(response)

        assert result is not None
        assert result.keywords == ["beam"]
        assert result.summary == "Beam lost."

    def test_parse_response_json_fence_without_closing_fence(self):
        """An unterminated ```json fence still parses."""
        module = SemanticProcessorModule()
        response = '```json\n{"keywords": ["rf"], "summary": "RF trip."}'

        result = module._parse_response(response)

        assert result is not None
        assert result.keywords == ["rf"]


class TestSemanticProcessorHealthCheckBranches:
    """Tests for SemanticProcessorModule.health_check outcomes."""

    @pytest.mark.asyncio
    async def test_health_check_ok_when_llm_responds(self, monkeypatch):
        """A truthy completion reports healthy."""
        calls = []

        def fake_completion(message, model_config=None):  # noqa: ARG001 - the get_chat_completion signature
            calls.append(message)
            return "OK"

        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            fake_completion,
        )

        result = await SemanticProcessorModule().health_check()

        assert (result.reachable, result.message, result.reason) == (True, "OK", None)
        assert calls == ["Say OK"]

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_empty_response(self, monkeypatch):
        """An empty completion reports unhealthy."""
        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            lambda message, model_config=None: "",
        )

        result = await SemanticProcessorModule().health_check()

        assert (result.reachable, result.message) == (False, "Empty response from LLM")
        assert result.reason == "unreachable"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_import_error(self, monkeypatch):
        """An unimportable completion module reports unhealthy."""
        monkeypatch.setitem(sys.modules, "osprey.models.completion", None)

        result = await SemanticProcessorModule().health_check()

        assert (result.reachable, result.message) == (
            False,
            "osprey.models.completion not available",
        )
        assert result.reason == "config"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_llm_error(self, monkeypatch):
        """An LLM exception surfaces as the unhealthy message."""

        def boom(message, model_config=None):  # noqa: ARG001 - the get_chat_completion signature
            raise RuntimeError("no route to host")

        monkeypatch.setattr("osprey.models.completion.get_chat_completion", boom)

        result = await SemanticProcessorModule().health_check()

        assert (result.reachable, result.message) == (False, "no route to host")
        assert result.reason == "unreachable"


def _embedding_module(models=None, provider=None):
    """Build a TextEmbeddingModule with an inline provider config."""
    module = TextEmbeddingModule()
    module.configure(
        {
            "models": models if models is not None else [{"name": "test-model", "dimension": 4}],
            "provider": provider
            if provider is not None
            else {"name": "fake", "base_url": "http://fake:11434"},
        }
    )
    return module


class TestTextEmbeddingCheckTablesExist:
    """Tests for TextEmbeddingModule._check_tables_exist caching and latching."""

    @pytest.mark.asyncio
    async def test_tables_present_is_cached_after_first_probe(self):
        """A positive probe is cached — the second call issues no SQL."""
        module = _embedding_module()
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        assert await module._check_tables_exist(conn) is True
        assert len(conn.calls) == 1

        assert await module._check_tables_exist(conn) is True
        assert len(conn.calls) == 1

    @pytest.mark.asyncio
    async def test_tables_absent_warns_and_caches_negative(self, caplog):
        """A negative probe warns once and caches False."""
        module = _embedding_module()
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(False,)]})

        with caplog.at_level(logging.WARNING, logger="ariel"):
            assert await module._check_tables_exist(conn) is False

        assert "Embedding tables do not exist" in caplog.text
        assert module._tables_exist is False
        assert len(conn.calls) == 1

    @pytest.mark.asyncio
    async def test_probe_stops_at_first_existing_table(self):
        """The scan short-circuits on the first model whose table exists."""
        module = _embedding_module(
            models=[{"name": "model-a", "dimension": 4}, {"name": "model-b", "dimension": 4}]
        )
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        assert await module._check_tables_exist(conn) is True
        assert len(conn.calls) == 1

    @pytest.mark.asyncio
    async def test_check_tables_exist_latches_first_outcome_permanently(self):
        """HAZARD: the tri-state latch is permanent for the module instance.

        ``_tables_exist`` starts as None and is written exactly once, by the
        first probe. Whatever that probe decided is returned forever after —
        the module never re-probes, so a table created *after* a negative probe
        stays invisible for the lifetime of the instance (and, in the service,
        for the lifetime of the process). A run that starts before the pgvector
        migration lands will skip text embedding for every entry even once the
        tables are there; only a fresh module instance clears the latch.
        """
        module = _embedding_module()
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(False,)]})

        assert await module._check_tables_exist(conn) is False
        assert len(conn.calls) == 1

        # The world changes: the migration runs and the table now exists.
        conn.recorder.rows_for[TABLES_QUERY] = [(True,)]

        assert await module._check_tables_exist(conn) is False, (
            "latched negative must survive the table appearing"
        )
        assert len(conn.calls) == 1, "no re-probe is ever issued"

        # A fresh instance is the only way back.
        assert await _embedding_module()._check_tables_exist(conn) is True

    @pytest.mark.asyncio
    async def test_enhance_returns_early_when_tables_missing(self):
        """enhance() stops at the table check — no provider, no INSERT."""
        module = _embedding_module()
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(False,)]})

        await module.enhance({"entry_id": "entry-001", "raw_text": "Beam at 500mA."}, conn)

        assert len(conn.calls) == 1
        assert not any("INSERT" in sql for sql in conn.sql)
        assert module._provider is None


class TestTextEmbeddingEnhanceWithProvider:
    """Tests for TextEmbeddingModule.enhance against a fake embedding provider."""

    @pytest.fixture
    def entry(self):
        """Non-empty sample entry."""
        return {"entry_id": "entry-001", "raw_text": "Beam current stabilized at 500mA."}

    @pytest.mark.asyncio
    async def test_enhance_stores_embedding_in_model_table(self, entry, monkeypatch):
        """A successful embedding is upserted into the model's own table."""
        provider = _route_provider(
            monkeypatch, make_fake_embedding_provider(vectors=[[0.1, 0.2, 0.3, 0.4]])
        )
        module = _embedding_module(
            provider={"name": "fake", "base_url": "http://fake:11434", "api_key": "secret"}
        )
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance(entry, conn)

        insert_sql = [sql for sql in conn.sql if "INSERT INTO" in sql]
        assert len(insert_sql) == 1
        assert "embeddings_test_model" in insert_sql[0]
        assert conn.calls[-1][1] == ["entry-001", [0.1, 0.2, 0.3, 0.4]]

        assert provider.calls == [
            {
                "texts": ["Beam current stabilized at 500mA."],
                "model_id": "test-model",
                "api_key": "secret",
                "base_url": "http://fake:11434",
            }
        ]

    @pytest.mark.asyncio
    async def test_enhance_falls_back_to_provider_default_base_url(self, entry, monkeypatch):
        """With no configured base_url the provider's own default is used."""
        provider = _route_provider(
            monkeypatch,
            make_fake_embedding_provider(default_base_url="http://provider-default:11434"),
        )
        module = _embedding_module(provider={"name": "fake"})
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance(entry, conn)

        assert provider.calls[0]["base_url"] == "http://provider-default:11434"
        assert provider.calls[0]["api_key"] is None

    @pytest.mark.asyncio
    async def test_enhance_cuts_a_long_entry_to_the_model_input_limit(self, monkeypatch):
        """A long entry is cut to the model's limit less the reserved tokens, in bytes."""
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module(
            models=[{"name": "test-model", "dimension": 4, "max_input_tokens": 40}]
        )
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance({"entry_id": "entry-001", "raw_text": "x" * 100}, conn)

        assert provider.calls[0]["texts"] == ["x" * 32]

    @pytest.mark.asyncio
    async def test_enhance_sends_an_entry_within_the_limit_whole(self, monkeypatch):
        """An entry that fits the limit is sent unchanged."""
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module(
            models=[{"name": "test-model", "dimension": 4, "max_input_tokens": 40}]
        )
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance({"entry_id": "entry-001", "raw_text": "x" * 32}, conn)

        assert provider.calls[0]["texts"] == ["x" * 32]

    @pytest.mark.asyncio
    async def test_enhance_logs_the_cut_naming_the_entry(self, monkeypatch, caplog):
        """A cut entry is logged once with its id, its length and the length embedded."""
        _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module(
            models=[{"name": "test-model", "dimension": 4, "max_input_tokens": 40}]
        )
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        with caplog.at_level(logging.INFO, logger="ariel"):
            await module.enhance({"entry_id": "entry-001", "raw_text": "x" * 100}, conn)

        records = [
            r.getMessage()
            for r in caplog.records
            if "entry-001" in r.getMessage() and "100 characters" in r.getMessage()
        ]
        assert len(records) == 1
        assert "first 32" in records[0]

    @pytest.mark.asyncio
    async def test_enhance_uses_the_default_limit_when_none_is_set(self, monkeypatch):
        """A model listed without max_input_tokens is cut to the default limit."""
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module(models=[{"name": "test-model", "dimension": 4}])
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance({"entry_id": "entry-001", "raw_text": "x" * 1000}, conn)

        assert provider.calls[0]["texts"] == ["x" * 504]

    @pytest.mark.asyncio
    async def test_enhance_loads_provider_once_across_entries(self, entry, monkeypatch):
        """The provider class is looked up once, in configure(), and then reused."""
        lookups = []
        _route_provider(monkeypatch, make_fake_embedding_provider(), lookups)
        module = _embedding_module(provider="ollama")
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance(entry, conn)
        await module.enhance(entry, conn)

        assert lookups == ["ollama"]

    @pytest.mark.asyncio
    async def test_enhance_raises_when_provider_fails_for_every_model(
        self, entry, monkeypatch, caplog
    ):
        """Every model failing escalates to RuntimeError naming each one."""
        _route_provider(
            monkeypatch, make_fake_embedding_provider(error=RuntimeError("embedding backend down"))
        )
        module = _embedding_module(
            models=[{"name": "model-a", "dimension": 4}, {"name": "model-b", "dimension": 4}]
        )
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        with caplog.at_level(logging.WARNING, logger="ariel"), pytest.raises(RuntimeError) as exc:
            await module.enhance(entry, conn)

        assert "All embedding models failed for entry entry-001" in str(exc.value)
        assert "model-a: embedding backend down" in str(exc.value)
        assert "model-b: embedding backend down" in str(exc.value)
        assert "Failed to generate embedding for entry entry-001" in caplog.text
        assert not any("INSERT" in sql for sql in conn.sql)


class TestTextEmbeddingEnhanceWithAttachmentText:
    """The embedder appends an entry's picture text to its own text."""

    @pytest.mark.asyncio
    async def test_the_caption_follows_the_entry_text_in_the_input(self, monkeypatch):
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module()
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})
        caption = "[picture a.png - upstream caption] dipole trip"

        await module.enhance(
            {"entry_id": "e1", "raw_text": "beam lost", "attachment_text": caption}, conn
        )

        assert provider.calls[0]["texts"] == [f"beam lost\n{caption}"]

    @pytest.mark.asyncio
    async def test_an_entry_without_attachment_text_gets_exactly_the_b1_input(self, monkeypatch):
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module(
            models=[{"name": "test-model", "dimension": 4, "max_input_tokens": 40}]
        )
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance({"entry_id": "e1", "raw_text": "x" * 100}, conn)
        await module.enhance(
            {"entry_id": "e2", "raw_text": "x" * 100, "attachment_text": None}, conn
        )

        assert [c["texts"] for c in provider.calls] == [
            [fit_to_input_limit("x" * 100, 40)],
            [fit_to_input_limit("x" * 100, 40)],
        ]

    @pytest.mark.asyncio
    async def test_an_entry_with_only_picture_text_is_embedded(self, monkeypatch):
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module()
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance(
            {"entry_id": "e1", "raw_text": "  ", "attachment_text": "[picture] trip"}, conn
        )

        assert provider.calls[0]["texts"] == ["  \n[picture] trip"]

    @pytest.mark.asyncio
    async def test_an_entry_blank_in_both_is_skipped(self, monkeypatch):
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module()
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance({"entry_id": "e1", "raw_text": " ", "attachment_text": " \n"}, conn)

        assert provider.calls == []


class TestEmbeddingInput:
    """The shared embed-input helper: the entry text, then its capped picture text."""

    def test_no_picture_text_is_exactly_the_cut_entry_text(self):
        text = "y" * 20
        assert embedding_input(text, None, 40) is text
        assert embedding_input(text, "", 40) is text
        assert embedding_input("x" * 100, "   ", 40) == fit_to_input_limit("x" * 100, 40)

    def test_a_short_entry_keeps_its_whole_text_before_the_picture_text(self):
        assert embedding_input("beam lost", "cap", 40) == "beam lost\ncap"

    def test_a_long_entry_fills_the_byte_budget_exactly(self):
        out = embedding_input("x" * 100, "cap", 40)

        assert len(out.encode("utf-8")) == 32
        assert out == "x" * 28 + "\ncap"

    def test_the_picture_text_is_capped_at_a_quarter_of_the_budget(self):
        out = embedding_input("x" * 100, "c" * 100, 40)

        assert out == "x" * 23 + "\n" + "c" * 8
        assert len(out.encode("utf-8")) == 32

    def test_the_cuts_count_utf8_bytes_and_never_split_a_character(self):
        out = embedding_input("加" * 100, "图" * 10, 40)

        assert out == "加" * 8 + "\n" + "图" * 2
        assert len(out.encode("utf-8")) <= 32


class TestFitToInputLimit:
    """Tests for the byte cut that fits a text to a model's input limit."""

    def test_text_within_the_budget_is_returned_as_is(self):
        """A text that fits is the same object, not a copy."""
        text = "y" * 32
        assert fit_to_input_limit(text, 40) is text

    def test_the_cut_counts_utf8_bytes(self):
        """A three-byte character costs three bytes of the budget."""
        assert fit_to_input_limit("加" * 100, 40) == "加" * 10

    def test_the_cut_never_splits_a_character(self):
        """A character the cut would split is dropped whole."""
        assert fit_to_input_limit("a" + "加" * 20, 13) == "a" + "加" * 1


class TestMaxInputTokens:
    """Tests for reading a model's input limit from its config entry."""

    def test_unset_is_the_default(self):
        """An absent or null limit reads as the default."""
        assert max_input_tokens({}) == 512
        assert max_input_tokens({"max_input_tokens": None}) == 512

    def test_a_stated_limit_is_used(self):
        """A stated integer limit is returned as is."""
        assert max_input_tokens({"name": "m", "max_input_tokens": 2048}) == 2048

    @pytest.mark.parametrize("bad", [8, 0, -1, True, "2048", 2048.0])
    def test_rejects_every_other_spelling(self, bad):
        """Anything but an integer above the reserve is refused with the model named."""
        with pytest.raises(ValueError) as exc:
            max_input_tokens({"name": "m", "max_input_tokens": bad})
        assert str(exc.value) == (
            "ariel.enhancement_modules.text_embedding.models: max_input_tokens for 'm' "
            f"must be an integer greater than 8 (got {bad!r})"
        )

    def test_configure_refuses_a_malformed_limit(self):
        """A malformed limit is refused when the module is configured."""
        with pytest.raises(
            ValueError,
            match=r"max_input_tokens for 'test-model' must be an integer greater than 8",
        ):
            TextEmbeddingModule().configure(
                {
                    "models": [{"name": "test-model", "dimension": 4, "max_input_tokens": "2048"}],
                    "provider": {"name": "fake"},
                }
            )


class TestTextEmbeddingHealthCheckBranches:
    """Tests for TextEmbeddingModule.health_check outcomes."""

    @pytest.mark.asyncio
    async def test_health_check_delegates_to_provider(self, monkeypatch):
        """A healthy verdict is returned, with credentials and the first model passed through."""
        provider = _route_provider(
            monkeypatch, make_fake_embedding_provider(healthy=(True, "ollama reachable"))
        )
        module = _embedding_module(
            provider={"name": "fake", "base_url": "http://fake:11434", "api_key": "secret"}
        )

        result = await module.health_check()
        assert (result.reachable, result.message, result.reason) == (
            True,
            "ollama reachable",
            None,
        )
        assert provider.calls == [
            {
                "check_embedding_health": True,
                "api_key": "secret",
                "base_url": "http://fake:11434",
                "model_id": "test-model",
            }
        ]

    @pytest.mark.asyncio
    async def test_health_check_with_no_models_lets_the_adapter_choose(self, monkeypatch):
        """With no models configured the probe names no model."""
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module(models=[])

        assert (await module.health_check()).reachable is True
        assert provider.calls[0]["model_id"] is None

    @pytest.mark.asyncio
    async def test_health_check_reports_provider_unhealthy(self, monkeypatch):
        """An unhealthy provider's message is surfaced verbatim."""
        _route_provider(
            monkeypatch, make_fake_embedding_provider(healthy=(False, "connection refused"))
        )

        result = await _embedding_module().health_check()
        assert (result.reachable, result.message, result.reason) == (
            False,
            "connection refused",
            "unreachable",
        )

    @pytest.mark.asyncio
    async def test_unknown_reachability_is_returned_unchanged(self, monkeypatch):
        """A verdict whose reachability is unknown (None) is passed on as it is."""
        from osprey.models.providers.health import HealthResult

        _route_provider(
            monkeypatch,
            make_fake_embedding_provider(healthy=HealthResult(None, "not checked", None)),
        )

        assert await _embedding_module().health_check() == HealthResult(None, "not checked", None)

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_the_probe_raises(self, monkeypatch):
        """A probe that raises reports unhealthy rather than raising."""
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())

        def boom(_self, *_args, **_kwargs):
            raise RuntimeError("no route to host")

        monkeypatch.setattr(provider, "check_embedding_health", boom)

        result = await _embedding_module().health_check()
        assert (result.reachable, result.message, result.reason) == (
            False,
            "no route to host",
            "unreachable",
        )


class TestTextEmbeddingProviderResolution:
    """configure() resolves the provider class once, from the provider registry."""

    @pytest.mark.parametrize(
        "provider_key",
        ["ariel.enhancement_modules.text_embedding.provider", "ariel.embedding.provider"],
    )
    def test_unknown_provider_is_refused_naming_its_key(self, monkeypatch, provider_key):
        """An unknown provider fails configure() with the key it came from."""
        _route_provider(monkeypatch, None)

        with pytest.raises(ValueError, match="nonesuch") as exc:
            TextEmbeddingModule().configure(
                {
                    "models": [{"name": "m", "dimension": 4}],
                    "provider": "nonesuch",
                    "provider_key": provider_key,
                }
            )
        assert provider_key in str(exc.value)

    def test_a_provider_without_embeddings_is_refused(self, monkeypatch):
        """A chat-only provider fails configure() with the module key."""
        from osprey.models.providers.base import BaseProvider

        class ChatOnly(BaseProvider):
            name = "chat-only"
            requires_api_key = False
            requires_base_url = False
            requires_model_id = False
            supports_proxy = False

        _route_provider(monkeypatch, ChatOnly)

        with pytest.raises(
            ValueError, match=r"ariel\.enhancement_modules\.text_embedding\.provider"
        ):
            TextEmbeddingModule().configure({"models": [], "provider": "chat-only"})

    def test_config_from_ariel_config_carries_the_key(self, monkeypatch):
        """The key from get_enhancement_module_config reaches the error."""
        _route_provider(monkeypatch, None)
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "embedding": {"provider": "nonesuch"},
                "enhancement_modules": {
                    "text_embedding": {"enabled": True, "models": [{"name": "m", "dimension": 4}]}
                },
            }
        )

        with pytest.raises(ValueError, match=r"ariel\.embedding\.provider"):
            TextEmbeddingModule().configure(config.get_enhancement_module_config("text_embedding"))


class TestTextEmbeddingDimensions:
    """Only a provider that cuts vectors to length is told the table's dimension."""

    @pytest.fixture
    def entry(self):
        return {"entry_id": "entry-001", "raw_text": "Beam current stabilized at 500mA."}

    @pytest.mark.asyncio
    async def test_a_non_truncating_provider_gets_no_dimensions_key(self, entry, monkeypatch):
        """An Ollama-like provider is called without a ``dimensions`` key."""
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module(models=[{"name": "test-model", "dimension": 4}])

        await module.enhance(entry, _FakeConnection(rows_for={TABLES_QUERY: [(True,)]}))

        assert "dimensions" not in provider.calls[0]

    @pytest.mark.asyncio
    async def test_a_truncating_provider_gets_each_models_dimension(self, entry, monkeypatch):
        """Each model's own dimension is passed to a truncating provider."""
        provider = _route_provider(
            monkeypatch, make_fake_embedding_provider(truncates_to_dimensions=True)
        )
        module = _embedding_module(
            models=[{"name": "model-a", "dimension": 4}, {"name": "model-b", "dimension": 2}]
        )

        await module.enhance(entry, _FakeConnection(rows_for={TABLES_QUERY: [(True,)]}))

        assert [call["dimensions"] for call in provider.calls] == [4, 2]

    @pytest.mark.asyncio
    async def test_llama_cpp_text_embedding_stores_the_configured_length(self, entry, monkeypatch):
        """llama-cpp answers 2048 floats; configure() + enhance() store 1024-d unit vectors."""
        import math

        import requests

        monkeypatch.setattr(ProviderRegistry, "get_provider", _REAL_GET_PROVIDER)
        monkeypatch.delenv("LLAMA_CPP_HOST", raising=False)
        posts = []

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"embedding": [float(i % 7 + 1) for i in range(2048)]}]}

        def fake_post(url, json, **_kwargs):
            posts.append((url, json))
            return _Response()

        monkeypatch.setattr(requests, "post", fake_post)
        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [{"name": "qwen3-vl-embedding-2b", "dimension": 1024}],
                "provider": "llama-cpp",
                "provider_key": "ariel.enhancement_modules.text_embedding.provider",
            }
        )
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance(entry, conn)

        assert posts[0][0] == "http://localhost:8080/v1/embeddings"
        stored = conn.calls[-1][1][1]
        assert len(stored) == 1024
        assert math.isclose(math.sqrt(sum(v * v for v in stored)), 1.0, rel_tol=1e-6)


class TestQmdExportModule:
    """Tests for QmdExportModule."""

    @pytest.fixture
    def entry(self):
        """Sample entry with a timestamp that picks a known shard."""
        return {
            "entry_id": "entry-001",
            "author": "operator",
            "source_system": "elog",
            "raw_text": "Replaced vacuum pump VP-103.",
            "timestamp": datetime(2024, 5, 17, 13, 45, 9, tzinfo=UTC),
        }

    def _module(self, mirror_root):
        """Return a module configured to mirror into an absolute root."""
        module = QmdExportModule()
        module.configure({"mirror_path": str(mirror_root)})
        return module

    def test_module_name(self):
        """Module has correct name."""
        assert QmdExportModule().name == "qmd_export"

    def test_module_has_no_migration(self):
        """The mirror lives on the filesystem, so no migration is needed."""
        assert QmdExportModule().migration is None

    def test_configure_keeps_absolute_path(self, tmp_path):
        """An absolute mirror_path is used verbatim."""
        module = self._module(tmp_path / "mirror")
        assert module._mirror_root == tmp_path / "mirror"

    def test_configure_resolves_relative_path_against_the_project_root(self, tmp_path, monkeypatch):
        """A relative mirror_path resolves against the project root, not ``build/``.

        The config lives in the render zone; the mirror must not, because the
        sidecar bind-mounts the same configured path from the repo root (the
        pinned compose project directory) and would otherwise index an empty
        tree while the exporter fills one inside ``build/``.
        """
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "build" / "config.yml"))
        module = QmdExportModule()

        module.configure({"mirror_path": "qmd-mirror"})

        assert module._mirror_root == (tmp_path / "qmd-mirror").resolve()

    def test_configure_expands_user_home(self, tmp_path, monkeypatch):
        """A tilde-prefixed mirror_path expands before the absolute check."""
        monkeypatch.setenv("HOME", str(tmp_path))
        module = QmdExportModule()

        module.configure({"mirror_path": "~/mirror"})

        assert module._mirror_root == tmp_path / "mirror"

    def test_configure_without_mirror_path_raises(self):
        """An enabled exporter with nowhere to write is a broken config."""
        with pytest.raises(ValueError, match="mirror_path is required"):
            QmdExportModule().configure({"enabled": True})

    @pytest.mark.asyncio
    async def test_enhance_writes_entry_and_touches_marker(self, tmp_path, entry):
        """A first export writes the sharded file and the freshness marker."""
        module = self._module(tmp_path)

        await module.enhance(entry, _FakeConnection())

        mirrored = tmp_path / "2024" / "05" / "entry-001.md"
        assert mirrored.is_file()
        assert "Replaced vacuum pump VP-103." in mirrored.read_text()
        assert (tmp_path / TOUCH_MARKER_NAME).is_file()

    @pytest.mark.asyncio
    async def test_enhance_leaves_marker_alone_when_nothing_changed(self, tmp_path, entry):
        """Re-exporting an unchanged entry must not manufacture reindex work."""
        module = self._module(tmp_path)
        await module.enhance(entry, _FakeConnection())

        marker = tmp_path / TOUCH_MARKER_NAME
        mirrored = tmp_path / "2024" / "05" / "entry-001.md"
        stale = 1_000_000.0
        os.utime(marker, (stale, stale))
        os.utime(mirrored, (stale, stale))

        await module.enhance(entry, _FakeConnection())

        assert marker.stat().st_mtime == stale
        assert mirrored.stat().st_mtime == stale

    @pytest.mark.asyncio
    async def test_enhance_touches_marker_when_content_changed(self, tmp_path, entry):
        """A changed entry rewrites the file and advances the marker."""
        module = self._module(tmp_path)
        await module.enhance(entry, _FakeConnection())

        marker = tmp_path / TOUCH_MARKER_NAME
        stale = 1_000_000.0
        os.utime(marker, (stale, stale))

        changed = {**entry, "raw_text": "Pump VP-103 replaced again."}
        await module.enhance(changed, _FakeConnection())

        assert marker.stat().st_mtime > stale

    @pytest.mark.asyncio
    async def test_enhance_is_a_noop_without_a_configured_mirror(self, tmp_path, entry):
        """A module that was never configured writes nothing rather than guessing."""
        module = QmdExportModule()

        await module.enhance(entry, _FakeConnection())

        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_enhance_raises_on_an_unmirrorable_entry_id(self, tmp_path):
        """An id that cannot be encoded is a failure, not a silent search gap."""
        module = self._module(tmp_path)

        with pytest.raises(ValueError, match="entry_id"):
            await module.enhance({"entry_id": "", "raw_text": "x"}, _FakeConnection())

    @pytest.mark.asyncio
    async def test_marker_failure_does_not_fail_the_entry(self, tmp_path, entry, caplog):
        """A marker that cannot be written is logged; the mirrored file still stands."""
        module = self._module(tmp_path)

        def boom(_path, _data, *, mode, sync="required"):  # noqa: ARG001
            raise OSError("read-only file system")

        target = "osprey.services.ariel_search.enhancement.qmd_export.exporter.atomic_replace"
        with caplog.at_level(logging.WARNING), mock.patch(target, boom):
            await module.enhance(entry, _FakeConnection())

        assert (tmp_path / "2024" / "05" / "entry-001.md").is_file()
        assert "freshness marker" in caplog.text

    @pytest.mark.asyncio
    async def test_health_check_reports_missing_configuration(self):
        """Health check is explicit about an unconfigured mirror."""
        result = await QmdExportModule().health_check()
        assert (result.reachable, result.message, result.reason) == (
            False,
            "mirror_path is not configured",
            "config",
        )

    @pytest.mark.asyncio
    async def test_health_check_creates_the_mirror_root(self, tmp_path):
        """A configured, creatable mirror root reports healthy."""
        module = self._module(tmp_path / "mirror")

        result = await module.health_check()

        assert (result.reachable, result.message, result.reason) == (True, "OK", None)
        assert (tmp_path / "mirror").is_dir()

    @pytest.mark.asyncio
    async def test_health_check_reports_an_unwritable_mirror_root(self, tmp_path):
        """A mirror root that cannot be created reports unhealthy rather than raising."""
        blocker = tmp_path / "mirror"
        blocker.write_text("not a directory")
        module = self._module(blocker)

        result = await module.health_check()

        assert result.reachable is False
        assert "not writable" in result.message
        assert result.reason == "config"


class TestOllamaTextEmbedder:
    """The helper the Ollama-gated ARIEL tests embed through."""

    def test_embeds_through_the_registered_ollama_adapter(self, monkeypatch):
        """The helper's adapter sends the text to Ollama's embedding endpoint via litellm."""
        from osprey.models.providers.ollama import OllamaProviderAdapter

        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        embedder = ollama_text_embedder()
        assert isinstance(embedder, OllamaProviderAdapter)

        with (
            mock.patch("osprey.models.providers._local_server.probe", return_value=True),
            mock.patch("litellm.embedding") as mock_embed,
        ):
            mock_embed.return_value = mock.MagicMock(data=[{"embedding": [0.25, 0.5]}])
            vectors = embedder.execute_embedding(["x"], "nomic-embed-text")

        assert vectors == [[0.25, 0.5]]
        call = mock_embed.call_args.kwargs
        assert call["model"] == "ollama/nomic-embed-text"
        assert call["input"] == ["x"]


class TestBaseHealthContract:
    """The base module's health answer is a typed 'not checked', never 'healthy'."""

    @pytest.mark.asyncio
    async def test_default_health_check_is_unchecked(self):
        """A module with no check of its own answers reachable None, no reason."""
        from osprey.models.providers.health import HealthResult
        from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule

        class _NoCheck(BaseEnhancementModule):
            @property
            def name(self) -> str:
                return "no_check"

            async def enhance(self, entry, conn):  # noqa: ARG002 - the module contract
                return None

        module = _NoCheck()

        assert await module.health_check() == HealthResult(None, "no health check", None)
        assert module.health_reason() is None

    @pytest.mark.asyncio
    async def test_a_sync_check_does_not_block_the_loop(self, monkeypatch):
        """The text-embedding probe runs off the loop, so a timeout can return."""
        import asyncio
        import time

        provider = _route_provider(monkeypatch, make_fake_embedding_provider())

        def slow(_self, *_args, **_kwargs):
            time.sleep(3)
            raise AssertionError("never read")

        monkeypatch.setattr(provider, "check_embedding_health", slow)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(_embedding_module().health_check(), 0.2)
        assert time.monotonic() - started < 1.5


# ---------------------------------------------------------------------------
# _fit_expanded: the cut walk for an expanded entry
# ---------------------------------------------------------------------------


class _SpyExpander:
    """Wraps an expander and records the ``limit`` of every scan."""

    def __init__(self, inner):
        self.inner = inner
        self.limits = []

    def expand(self, text, *, limit=None, tokens=None):
        self.limits.append(limit)
        return self.inner.expand(text, limit=limit, tokens=tokens)


def _fit_expander(unit_vocabulary_file):
    from osprey.services.ariel_search.vocabulary import EntryExpander
    from osprey.services.ariel_search.vocabulary.loader import load_vocabulary

    loaded, errors, _warnings = load_vocabulary(
        unit_vocabulary_file, canonical_to_acronym=True, canonical_to_shorthand=False
    )
    assert errors == []
    return _SpyExpander(
        EntryExpander(loaded, canonical_to_acronym=True, canonical_to_shorthand=False)
    )


def _fit(raw, expander, *, room, floor, scanned=None):
    from osprey.services.ariel_search.enhancement.text_embedding.embedder import _fit_expanded
    from osprey.services.ariel_search.vocabulary.model import match_tokens

    return _fit_expanded(raw, match_tokens(raw), expander, room=room, floor=floor, scanned=scanned)


def _bytes_at(raw, cut, expansion):
    added = sum(len((" " + i.text).encode()) for i in expansion.insertions if i.anchor <= cut)
    return len(raw[:cut].encode()) + added


def _word_ends(raw):
    import re

    return {m.end() for m in re.finditer(r"\S+", raw)}


def _floor(raw, room):
    """The shortest whitespace-aligned prefix of at least three quarters of ``room``."""
    return min(end for end in _word_ends(raw) if len(raw[:end].encode()) >= room * 3 // 4)


class TestFitExpanded:
    """``_fit_expanded`` picks a word-aligned cut whose expansion fits the room."""

    def test_fit_expanded_uncut_entry_keeps_whole_text(self, unit_vocabulary_file):
        raw = "RF trip at BPM 3, quad reset"
        expander = _fit_expander(unit_vocabulary_file)
        cut, expansion = _fit(raw, expander, room=200, floor=len(raw))
        assert cut == len(raw)
        assert expansion.text == (
            "RF radio frequency trip at BPM beam position monitor 3, quad quadrupole reset"
        )
        assert _bytes_at(raw, cut, expansion) <= 200

    def test_fit_expanded_uncut_entry_never_cut_when_terms_overflow(self, unit_vocabulary_file):
        # 15 acronyms in a short entry: the raw text stays whole, the caller drops
        # the trailing insertions that do not fit.
        raw = " ".join(["RF trip"] * 15)
        room = len(raw) + 40
        expander = _fit_expander(unit_vocabulary_file)
        cut, expansion = _fit(raw, expander, room=room, floor=len(raw))
        assert cut == len(raw)
        assert len(expansion.insertions) == 15
        assert _bytes_at(raw, cut, expansion) > room
        assert len(expander.limits) == 2  # walk scan + one confirming re-scan at the floor

    def test_fit_expanded_cut_entry_picks_largest_fitting_word_end(self, unit_vocabulary_file):
        raw = "BPM drift " + "filler word " * 40
        room = 200
        floor = _floor(raw, room)
        expander = _fit_expander(unit_vocabulary_file)
        cut, expansion = _fit(raw, expander, room=room, floor=floor)
        assert cut in _word_ends(raw)
        assert floor <= cut
        assert _bytes_at(raw, cut, expansion) <= room
        # One more word would overflow.
        nxt = min(end for end in _word_ends(raw) if end > cut)
        assert _bytes_at(raw, nxt, expansion) > room

    def test_fit_expanded_dense_entry_falls_back_to_floor(self, unit_vocabulary_file):
        raw = " ".join(["BPM trip"] * 100)
        room = 120
        floor = _floor(raw, room)
        expander = _fit_expander(unit_vocabulary_file)
        cut, expansion = _fit(raw, expander, room=room, floor=floor)
        assert cut == floor
        assert len(raw[:cut].encode()) >= room * 3 // 4
        assert all(i.anchor <= cut for i in expansion.insertions)

    def test_fit_expanded_expansion_is_scanned_up_to_the_cut(self, unit_vocabulary_file):
        raw = "word " * 30 + "RF " + "word " * 30
        room = 100
        floor = _floor(raw, room)
        expander = _fit_expander(unit_vocabulary_file)
        cut, expansion = _fit(raw, expander, room=room, floor=floor)
        assert expander.limits[-1] == cut
        assert all(i.anchor <= cut for i in expansion.insertions)

    def test_fit_expanded_runs_at_most_three_rescans(self, unit_vocabulary_file):
        raw = ("RF BPM quad " * 60).strip()
        room = 300
        floor = _floor(raw, room)
        expander = _fit_expander(unit_vocabulary_file)
        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            _utf8_prefix,
        )

        scanned = expander.inner.expand(raw, limit=len(_utf8_prefix(raw, room)))
        cut, _expansion = _fit(raw, expander, room=room, floor=floor, scanned=scanned)
        assert 1 <= len(expander.limits) <= 3
        assert floor <= cut

    def test_fit_expanded_reuses_given_scan(self, unit_vocabulary_file):
        raw = "RF trip"
        expander = _fit_expander(unit_vocabulary_file)
        scanned = expander.inner.expand(raw, limit=len(raw))
        _fit(raw, expander, room=100, floor=len(raw), scanned=scanned)
        assert expander.limits == [len(raw)]

    def test_fit_expanded_ignores_torn_token_at_room_end(self, unit_vocabulary_file):
        # The room ends inside the last word: that word's end is not a candidate.
        raw = "alpha beta gamma RFQ"
        room = len("alpha beta gamma RF")
        expander = _fit_expander(unit_vocabulary_file)
        cut, expansion = _fit(raw, expander, room=room, floor=len("alpha beta gamma"))
        assert cut == len("alpha beta gamma")
        assert expansion.insertions == ()

    def test_fit_expanded_no_boundary_floor_is_final(self, unit_vocabulary_file):
        raw = "RF " + "x" * 400
        room = 200
        floor = len(raw.encode()[:room].decode())
        expander = _fit_expander(unit_vocabulary_file)
        cut, expansion = _fit(raw, expander, room=room, floor=floor)
        assert cut == floor
        assert expander.limits[-1] == floor

    def test_fit_expanded_multibyte_text_counts_bytes(self, unit_vocabulary_file):
        raw = "RF " + "ünïcödé " * 40
        room = 150
        floor = _floor(raw, room)
        expander = _fit_expander(unit_vocabulary_file)
        cut, expansion = _fit(raw, expander, room=room, floor=floor)
        assert cut in _word_ends(raw)
        assert _bytes_at(raw, cut, expansion) <= room
        assert len(raw[:cut].encode()) >= room * 3 // 4

    def test_fit_expanded_random_invariants(self, unit_vocabulary_file):
        import random

        rng = random.Random(1234)
        words = ["RF", "BPM", "quad", "trip", "beam", "ünï", "orbit", "BPMs", "x" * 9, "TS"]
        expander = _fit_expander(unit_vocabulary_file)
        for _ in range(200):
            raw = " ".join(rng.choice(words) for _ in range(rng.randint(1, 120)))
            room = rng.randint(40, 400)
            if len(raw.encode()) <= room:
                floor = len(raw)
            else:
                floor = _floor(raw, room)
            expander.limits.clear()
            cut, expansion = _fit(raw, expander, room=room, floor=floor)
            assert len(expander.limits) <= 4
            assert floor <= cut
            assert cut == len(raw) or cut in _word_ends(raw)
            assert expander.limits[-1] == cut
            if cut > floor:
                assert _bytes_at(raw, cut, expansion) <= room


# ---------------------------------------------------------------------------
# build_embedding_input: the embed input with optional entry expansion
# ---------------------------------------------------------------------------


def _build(raw, att, max_tokens, expander=None):
    from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
        build_embedding_input,
    )

    return build_embedding_input(raw, att, max_tokens, expander=expander)


#: ``max_input_tokens`` whose byte room for the entry text is exactly ``room``.
def _tokens_for(room):
    from osprey.services.ariel_search.enhancement.text_embedding.embedder import RESERVED_TOKENS

    return room + RESERVED_TOKENS


class TestBuildEmbeddingInput:
    """``build_embedding_input`` keeps today's bytes without an expander and fits terms with one."""

    _SAMPLES = (
        ("beam lost", None),
        ("x" * 100, None),
        ("x" * 100, "cap"),
        ("x" * 100, "c" * 100),
        ("加" * 100, "图" * 10),
        ("y" * 20, ""),
        ("RF trip at BPM 3, quad reset", "   "),
    )

    def test_embedding_input_without_expander_is_byte_exact(self):
        for raw, att in self._SAMPLES:
            built = _build(raw, att, 40)
            assert built.text == embedding_input(raw, att, 40)
            if not (att or "").strip():
                assert built.text == fit_to_input_limit(raw, 40)

    def test_embedding_input_returns_the_built_text_object(self):
        text = "y" * 20
        assert _build(text, None, 40).text is text
        assert embedding_input(text, None, 40) is text

    def test_embedding_input_kept_chars_counts_raw_code_points_only(self):
        assert _build("beam lost", "cap", 40).kept_chars == len("beam lost")
        assert _build("x" * 100, "cap", 40).kept_chars == 28
        assert _build("加" * 100, "图" * 10, 40).kept_chars == 8
        assert _build("x" * 100, None, 40).kept_chars == 32

    def test_EmbeddingInput_is_a_frozen_value(self):
        import dataclasses

        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            EmbeddingInput,
        )

        value = EmbeddingInput(text="a", kept_chars=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            value.text = "b"

    def test_embedding_input_short_entry_embeds_every_term(self, unit_vocabulary_file):
        expander = _fit_expander(unit_vocabulary_file)
        raw = "RF trip at BPM 3, quad reset"

        built = _build(raw, None, 512, expander)

        assert built.text == expander.inner.expand(raw).text
        assert built.kept_chars == len(raw)

    def test_embedding_input_without_matches_is_today_bytes(self, unit_vocabulary_file):
        expander = _fit_expander(unit_vocabulary_file)
        short = "beam lost at injection"
        long = "beam lost at injection " * 20

        assert _build(short, None, 512, expander).text is short
        assert _build(long, None, 40, expander).text == fit_to_input_limit(long, 40)
        assert expander.limits[-1] == len(fit_to_input_limit(long, 40))

    def test_embedding_input_short_entry_is_never_cut_terms_dropped_whole(
        self, unit_vocabulary_file
    ):
        expander = _fit_expander(unit_vocabulary_file)
        raw = "RF trip at BPM 3, quad reset"
        # Room for the raw text, " radio frequency" and " quadrupole", but not
        # " beam position monitor": once a term overflows the rest are dropped.
        room = len(raw) + len(" radio frequency") + len(" quadrupole")

        built = _build(raw, None, _tokens_for(room), expander)

        assert built.text == "RF radio frequency trip at BPM 3, quad reset"
        assert built.kept_chars == len(raw)
        assert len(built.text.encode()) <= room

    def test_embedding_input_long_entry_cut_at_word_end_above_floor(self, unit_vocabulary_file):
        expander = _fit_expander(unit_vocabulary_file)
        raw = "RF trip at BPM 3, quad reset and the vertical orbit drifted " * 6
        room = 120

        built = _build(raw, None, _tokens_for(room), expander)

        assert len(built.text.encode()) <= room
        cut = built.kept_chars
        assert cut in _word_ends(raw)
        assert cut >= _floor(raw, room)
        assert len(raw[:cut].encode()) >= room * 3 // 4
        assert "radio frequency" in built.text
        # Every inserted term is matched over the text actually embedded.
        assert len(expander.limits) <= 1 + 3
        assert expander.limits[0] == len(fit_to_input_limit(raw, _tokens_for(room)))

    def test_embedding_input_output_is_raw_prefix_plus_ordered_insertions(
        self, unit_vocabulary_file
    ):
        expander = _fit_expander(unit_vocabulary_file)
        raw = "quad trip, BPM 4 noisy, RF ok, then quad again " * 5
        for room in (40, 63, 90, 150, 400):
            built = _build(raw, None, _tokens_for(room), expander)
            cut = built.kept_chars
            full = expander.inner.expand(raw[:cut])
            # The output is raw[:cut] with a prefix of the cut text's insertions.
            n = len(full.insertions)
            renders = []
            for k in range(n + 1):
                pos, parts = 0, []
                for ins in full.insertions[:k]:
                    parts.append(raw[pos : ins.anchor] + " " + ins.text)
                    pos = ins.anchor
                parts.append(raw[pos:cut])
                renders.append("".join(parts))
            assert built.text in renders
            assert len(built.text.encode()) <= room

    def test_embedding_input_no_boundary_in_window_falls_back_to_byte_cut(
        self, unit_vocabulary_file
    ):
        expander = _fit_expander(unit_vocabulary_file)
        raw = "RF " + "x" * 100
        tokens = _tokens_for(40)

        built = _build(raw, None, tokens, expander)

        assert built.text == fit_to_input_limit(raw, tokens)
        assert built.kept_chars == 40

    def test_embedding_input_torn_token_is_never_matched(self, unit_vocabulary_file):
        expander = _fit_expander(unit_vocabulary_file)
        raw = "x " * 18 + "BPM trip"  # the room ends inside "BPM"
        tokens = _tokens_for(37)

        built = _build(raw, None, tokens, expander)

        assert built.text == fit_to_input_limit(raw, tokens)
        assert "beam position monitor" not in built.text

    def test_embedding_input_picture_text_keeps_its_quarter(self, unit_vocabulary_file):
        expander = _fit_expander(unit_vocabulary_file)
        raw = "RF trip at BPM 3, quad reset " * 10
        att = "c" * 200
        budget = 120

        built = _build(raw, att, _tokens_for(budget), expander)

        assert built.text.endswith("\n" + "c" * (budget // 4))
        assert len(built.text.encode()) <= budget
        assert "radio frequency" in built.text
        assert built.kept_chars < len(raw)

    def test_embedding_input_running_key_composition(self, unit_vocabulary_file):
        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            INPUT_RULE,
            running_input_key,
        )
        from osprey.services.ariel_search.vocabulary.entry import classify_build_keys

        expander = _fit_expander(unit_vocabulary_file).inner

        assert INPUT_RULE == 1
        assert running_input_key(None) == "r1"
        assert running_input_key(expander) == f"r1:{expander.key}"
        for key in (running_input_key(None), running_input_key(expander)):
            classify_build_keys({}, key)


# ---------------------------------------------------------------------------
# TextEmbeddingModule: entry expansion bound through bind()
# ---------------------------------------------------------------------------


def _ariel_config(vocabulary=None):
    """An ARIELConfig, with entry expansion switched on against *vocabulary* when given."""
    config = {"database": {"uri": "postgresql://localhost/test"}}
    if vocabulary is not None:
        config["vocabulary"] = {"enabled": True, "path": str(vocabulary), "expand_entries": True}
    return ARIELConfig.from_dict(config)


class TestTextEmbeddingEnhanceExpander:
    """enhance() builds its input with the expander bound from the ARIEL configuration."""

    @pytest.mark.asyncio
    async def test_enhance_without_expander_sends_identical_text_and_insert(self, monkeypatch):
        """With expansion off, the provider text and the INSERT match an unbound module's."""
        entry = {"entry_id": "entry-001", "raw_text": "RF trip at BPM 3. " * 10}
        models = [{"name": "test-model", "dimension": 4, "max_input_tokens": 40}]

        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        unbound = _embedding_module(models=models)
        unbound_conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})
        await unbound.enhance(entry, unbound_conn)

        bound = _embedding_module(models=models)
        bound.bind(_ariel_config())
        bound_conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})
        await bound.enhance(entry, bound_conn)

        assert bound._expander is None
        assert provider.calls[1]["texts"] == provider.calls[0]["texts"]
        assert provider.calls[1]["texts"] == [embedding_input(entry["raw_text"], None, 40)]
        assert bound_conn.calls == unbound_conn.calls
        assert len([sql for sql in bound_conn.sql if "INSERT INTO" in sql]) == 1

    @pytest.mark.asyncio
    async def test_enhance_with_expander_sends_the_expanded_text(
        self, monkeypatch, unit_vocabulary_file
    ):
        """A bound expander's terms are inserted inline into the text the provider receives."""
        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            DEFAULT_MAX_INPUT_TOKENS,
            build_embedding_input,
        )

        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        config = _ariel_config(unit_vocabulary_file)
        module = _embedding_module()
        module.bind(config)
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)], COLUMNS_QUERY: [(1,)]})
        raw = "RF trip at BPM 3"

        await module.enhance({"entry_id": "entry-001", "raw_text": raw}, conn)

        expected = build_embedding_input(
            raw, None, DEFAULT_MAX_INPUT_TOKENS, expander=config.entry_expander()
        ).text
        assert expected == "RF radio frequency trip at BPM beam position monitor 3"
        assert provider.calls[0]["texts"] == [expected]

    @pytest.mark.asyncio
    async def test_enhance_with_expander_logs_truncation_with_kept_chars(
        self, monkeypatch, caplog, unit_vocabulary_file
    ):
        """A cut expanded entry logs how much of its own text was kept, not the input length."""
        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            build_embedding_input,
        )

        _route_provider(monkeypatch, make_fake_embedding_provider())
        config = _ariel_config(unit_vocabulary_file)
        module = _embedding_module(
            models=[{"name": "test-model", "dimension": 4, "max_input_tokens": 160}]
        )
        module.bind(config)
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)], COLUMNS_QUERY: [(1,)]})
        raw = "RF 故障 BPM 漂移 " * 20
        built = build_embedding_input(raw, None, 160, expander=config.entry_expander())
        assert built.kept_chars < len(raw)
        assert built.kept_chars != len(built.text)

        with caplog.at_level(logging.INFO, logger="ariel"):
            await module.enhance({"entry_id": "entry-cjk", "raw_text": raw}, conn)

        records = [r.getMessage() for r in caplog.records if "entry-cjk" in r.getMessage()]
        assert len(records) == 1
        assert f"{len(raw)} characters" in records[0]
        assert f"first {built.kept_chars} " in records[0]

    @pytest.mark.asyncio
    async def test_enhance_does_not_log_truncation_when_the_expanded_entry_is_whole(
        self, monkeypatch, caplog, unit_vocabulary_file
    ):
        """An entry kept whole is not logged, however long its inserted terms make the input."""
        _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module()
        module.bind(_ariel_config(unit_vocabulary_file))
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)], COLUMNS_QUERY: [(1,)]})

        with caplog.at_level(logging.INFO, logger="ariel"):
            await module.enhance({"entry_id": "entry-whole", "raw_text": "RF trip"}, conn)

        assert not [r for r in caplog.records if "entry-whole" in r.getMessage()]


# ---------------------------------------------------------------------------
# TextEmbeddingModule: the input_key column probe and write
# ---------------------------------------------------------------------------


def _absent_log_lines(caplog):
    """Log lines reporting an embedding table without an ``input_key`` column."""
    return [r.getMessage() for r in caplog.records if "no input_key column" in r.getMessage()]


class TestTextEmbeddingInputKey:
    """enhance() writes the build key into tables that have an ``input_key`` column."""

    @pytest.mark.asyncio
    async def test_input_key_column_present_binds_the_running_key(self, monkeypatch):
        """With the column present and expansion off, the INSERT binds ``input_key='r1'``."""
        _route_provider(monkeypatch, make_fake_embedding_provider(vectors=[[0.1, 0.2, 0.3, 0.4]]))
        module = _embedding_module()
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)], COLUMNS_QUERY: [(1,)]})

        await module.enhance({"entry_id": "entry-001", "raw_text": "Beam at 500mA."}, conn)

        probes = conn.recorder.matching(COLUMNS_QUERY)
        assert [params for _sql, params in probes] == [["text_embeddings_test_model"]]
        inserts = conn.recorder.matching("INSERT INTO")
        assert len(inserts) == 1
        sql, params = inserts[0]
        assert "input_key" in sql
        assert "input_key = EXCLUDED.input_key" in " ".join(sql.split())
        assert params == ["entry-001", [0.1, 0.2, 0.3, 0.4], "r1"]

    @pytest.mark.asyncio
    async def test_input_key_column_present_binds_the_expander_key(
        self, monkeypatch, unit_vocabulary_file
    ):
        """With expansion on, the bound key names the expander the input was built with."""
        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            running_input_key,
        )

        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        config = _ariel_config(unit_vocabulary_file)
        module = _embedding_module()
        module.bind(config)
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)], COLUMNS_QUERY: [(1,)]})

        await module.enhance({"entry_id": "entry-001", "raw_text": "RF trip at BPM 3"}, conn)

        ((_sql, params),) = conn.recorder.matching("INSERT INTO")
        assert params[2] == running_input_key(config.entry_expander())
        assert params[2].startswith("r1:")
        assert "radio frequency" in provider.calls[0]["texts"][0]

    @pytest.mark.asyncio
    async def test_input_key_column_absent_embeds_unexpanded_and_omits_the_column(
        self, monkeypatch, caplog, unit_vocabulary_file
    ):
        """Without the column the entry is embedded unexpanded and the INSERT omits it."""
        provider = _route_provider(monkeypatch, make_fake_embedding_provider())
        module = _embedding_module()
        module.bind(_ariel_config(unit_vocabulary_file))
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        with caplog.at_level(logging.INFO, logger="ariel"):
            await module.enhance({"entry_id": "entry-001", "raw_text": "RF trip at BPM 3"}, conn)

        assert provider.calls[0]["texts"] == ["RF trip at BPM 3"]
        ((sql, params),) = conn.recorder.matching("INSERT INTO")
        assert "input_key" not in sql
        assert len(params) == 2
        lines = _absent_log_lines(caplog)
        assert len(lines) == 1
        assert "text_embeddings_test_model" in lines[0]

    @pytest.mark.asyncio
    async def test_input_key_probe_runs_once_per_table_and_logs_once_per_process(
        self, monkeypatch, caplog
    ):
        """The probe is cached per module; the absence is logged once across modules."""
        _route_provider(monkeypatch, make_fake_embedding_provider())
        first = _embedding_module()
        second = _embedding_module()
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        with caplog.at_level(logging.INFO, logger="ariel"):
            await first.enhance({"entry_id": "e1", "raw_text": "one"}, conn)
            await first.enhance({"entry_id": "e2", "raw_text": "two"}, conn)
            await second.enhance({"entry_id": "e3", "raw_text": "three"}, conn)

        assert len(conn.recorder.matching(COLUMNS_QUERY)) == 2
        assert len(_absent_log_lines(caplog)) == 1

    @pytest.mark.asyncio
    async def test_input_key_column_on_one_of_two_tables(self, monkeypatch, caplog):
        """Each model's table is probed; only the table with the column gets ``input_key``."""
        _route_provider(monkeypatch, make_fake_embedding_provider(vectors=[[0.5, 0.5]]))
        module = _embedding_module(
            models=[{"name": "model-a", "dimension": 2}, {"name": "model-b", "dimension": 2}]
        )
        # Execute order after the rows_for table check: probe a, INSERT a, probe b, INSERT b.
        conn = _FakeConnection(
            rows_for={TABLES_QUERY: [(True,)]},
            results=[[(1,)], [], [], []],
        )

        with caplog.at_level(logging.INFO, logger="ariel"):
            await module.enhance({"entry_id": "entry-001", "raw_text": "Beam at 500mA."}, conn)

        probes = conn.recorder.matching(COLUMNS_QUERY)
        assert [params for _sql, params in probes] == [
            ["text_embeddings_model_a"],
            ["text_embeddings_model_b"],
        ]
        inserts = conn.recorder.matching("INSERT INTO")
        assert len(inserts) == 2
        assert "text_embeddings_model_a" in inserts[0][0]
        assert inserts[0][1] == ["entry-001", [0.5, 0.5], "r1"]
        assert "text_embeddings_model_b" in inserts[1][0]
        assert "input_key" not in inserts[1][0]
        assert inserts[1][1] == ["entry-001", [0.5, 0.5]]
        lines = _absent_log_lines(caplog)
        assert len(lines) == 1
        assert "text_embeddings_model_b" in lines[0]
