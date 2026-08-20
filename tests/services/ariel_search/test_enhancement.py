"""Tests for ARIEL enhancement modules.

Tests for base enhancement module, factory, and module implementations.
"""

import logging
import os
import sys
from datetime import UTC, datetime
from unittest import mock

import pytest

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
from tests.services.ariel_search.conftest import _FakeConnection, _FakeEmbeddingProvider

TABLES_QUERY = "information_schema.tables"


class TestEnhancementFactory:
    """Tests for enhancement module factory."""

    def test_get_enhancer_names(self):
        """get_enhancer_names returns correct list."""
        names = get_enhancer_names()
        assert names == ["semantic_processor", "text_embedding", "qmd_export"]

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
        healthy, message = await module.health_check()
        # Without LLM configured, should be unhealthy
        assert isinstance(healthy, bool)
        assert isinstance(message, str)


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
        healthy, message = await module.health_check()
        # Without embedder configured, should be unhealthy
        assert isinstance(healthy, bool)
        assert isinstance(message, str)


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
            "source_system": "ALS eLog",
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
    """Tests for SemanticProcessorModule.configure tier resolution."""

    def test_configure_resolves_model_id_when_provider_given(self, monkeypatch):
        """A provider plus a model_id routes the id through resolve_model_id."""
        calls = []

        def fake_resolve(provider, model_id):
            calls.append((provider, model_id))
            return f"{provider}/{model_id}-resolved"

        monkeypatch.setattr("osprey.models.tiers.resolve_model_id", fake_resolve)

        module = SemanticProcessorModule()
        module.configure(
            {
                "provider": "ollama",
                "model": {"model_id": "fast", "temperature": 0.1},
            }
        )

        assert calls == [("ollama", "fast")]
        assert module._model_config == {
            "provider": "ollama",
            "model_id": "ollama/fast-resolved",
            "temperature": 0.1,
        }

    def test_configure_without_provider_raises_before_resolution(self, monkeypatch):
        """No provider is an error, not a silent fall-through to a default."""

        def boom(provider, model_id):
            raise AssertionError("resolve_model_id must not run without a provider")

        monkeypatch.setattr("osprey.models.tiers.resolve_model_id", boom)

        module = SemanticProcessorModule()
        with pytest.raises(ValueError, match="semantic_processor.provider"):
            module.configure({"model": {"model_id": "llama3"}})

    def test_configure_with_provider_but_no_model_id_skips_resolution(self, monkeypatch):
        """A provider alone is not enough — resolution needs a model_id too."""

        def boom(provider, model_id):
            raise AssertionError("resolve_model_id must not run without a model_id")

        monkeypatch.setattr("osprey.models.tiers.resolve_model_id", boom)

        module = SemanticProcessorModule()
        module.configure({"provider": "ollama", "model": {"temperature": 0.2}})

        assert module._model_config == {"provider": "ollama", "temperature": 0.2}


class TestSemanticProcessorProcessText:
    """Tests for SemanticProcessorModule._process_text with a patched LLM."""

    @pytest.fixture
    def module(self):
        """Module configured with a model config (no tier resolution)."""
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

        def fake_completion(message, model_config=None):
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

        def fake_completion(message, model_config=None):
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

        def boom(message, model_config=None):
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

        def fake_completion(message, model_config=None):
            calls.append(message)
            return "OK"

        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            fake_completion,
        )

        healthy, message = await SemanticProcessorModule().health_check()

        assert (healthy, message) == (True, "OK")
        assert calls == ["Say OK"]

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_empty_response(self, monkeypatch):
        """An empty completion reports unhealthy."""
        monkeypatch.setattr(
            "osprey.models.completion.get_chat_completion",
            lambda message, model_config=None: "",
        )

        healthy, message = await SemanticProcessorModule().health_check()

        assert (healthy, message) == (False, "Empty response from LLM")

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_import_error(self, monkeypatch):
        """An unimportable completion module reports unhealthy."""
        monkeypatch.setitem(sys.modules, "osprey.models.completion", None)

        healthy, message = await SemanticProcessorModule().health_check()

        assert (healthy, message) == (False, "osprey.models.completion not available")

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_llm_error(self, monkeypatch):
        """An LLM exception surfaces as the unhealthy message."""

        def boom(message, model_config=None):
            raise RuntimeError("no route to host")

        monkeypatch.setattr("osprey.models.completion.get_chat_completion", boom)

        healthy, message = await SemanticProcessorModule().health_check()

        assert (healthy, message) == (False, "no route to host")


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
        provider = _FakeEmbeddingProvider(vectors=[[0.1, 0.2, 0.3, 0.4]])
        monkeypatch.setattr(
            "osprey.models.embeddings.get_embedding_provider",
            lambda name: provider,
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
        provider = _FakeEmbeddingProvider(default_base_url="http://provider-default:11434")
        monkeypatch.setattr(
            "osprey.models.embeddings.get_embedding_provider",
            lambda name: provider,
        )
        module = _embedding_module(provider={"name": "fake"})
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance(entry, conn)

        assert provider.calls[0]["base_url"] == "http://provider-default:11434"
        assert provider.calls[0]["api_key"] is None

    @pytest.mark.asyncio
    async def test_enhance_truncates_text_to_model_token_budget(self, monkeypatch):
        """Text is clipped to max_input_tokens * 4 characters."""
        provider = _FakeEmbeddingProvider()
        monkeypatch.setattr(
            "osprey.models.embeddings.get_embedding_provider",
            lambda name: provider,
        )
        module = _embedding_module(
            models=[{"name": "test-model", "dimension": 4, "max_input_tokens": 10}]
        )
        conn = _FakeConnection(rows_for={TABLES_QUERY: [(True,)]})

        await module.enhance({"entry_id": "entry-001", "raw_text": "x" * 100}, conn)

        assert provider.calls[0]["texts"] == ["x" * 40]

    @pytest.mark.asyncio
    async def test_enhance_loads_provider_once_across_entries(self, entry, monkeypatch):
        """The provider is looked up lazily and then reused."""
        lookups = []
        provider = _FakeEmbeddingProvider()

        def fake_lookup(name):
            lookups.append(name)
            return provider

        monkeypatch.setattr("osprey.models.embeddings.get_embedding_provider", fake_lookup)
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
        provider = _FakeEmbeddingProvider(error=RuntimeError("embedding backend down"))
        monkeypatch.setattr(
            "osprey.models.embeddings.get_embedding_provider",
            lambda name: provider,
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


class TestTextEmbeddingHealthCheckBranches:
    """Tests for TextEmbeddingModule.health_check outcomes."""

    @pytest.mark.asyncio
    async def test_health_check_delegates_to_provider(self, monkeypatch):
        """A healthy provider's verdict is returned, with credentials passed through."""
        provider = _FakeEmbeddingProvider(healthy=(True, "ollama reachable"))
        monkeypatch.setattr(
            "osprey.models.embeddings.get_embedding_provider",
            lambda name: provider,
        )
        module = _embedding_module(
            provider={"name": "fake", "base_url": "http://fake:11434", "api_key": "secret"}
        )

        assert await module.health_check() == (True, "ollama reachable")
        assert provider.calls == [
            {"check_health": True, "api_key": "secret", "base_url": "http://fake:11434"}
        ]

    @pytest.mark.asyncio
    async def test_health_check_reports_provider_unhealthy(self, monkeypatch):
        """An unhealthy provider's message is surfaced verbatim."""
        provider = _FakeEmbeddingProvider(healthy=(False, "connection refused"))
        monkeypatch.setattr(
            "osprey.models.embeddings.get_embedding_provider",
            lambda name: provider,
        )

        assert await _embedding_module().health_check() == (False, "connection refused")

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_provider_lookup_fails(self, monkeypatch):
        """A provider that cannot be loaded reports unhealthy rather than raising."""

        def boom(name):
            raise ValueError("unknown embedding provider 'fake'")

        monkeypatch.setattr("osprey.models.embeddings.get_embedding_provider", boom)

        healthy, message = await _embedding_module().health_check()

        assert healthy is False
        assert message == "unknown embedding provider 'fake'"


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

    def test_configure_resolves_relative_path_against_config_dir(self, tmp_path, monkeypatch):
        """A relative mirror_path resolves against the directory holding config.yml."""
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "build" / "config.yml"))
        module = QmdExportModule()

        module.configure({"mirror_path": "qmd-mirror"})

        assert module._mirror_root == (tmp_path / "build" / "qmd-mirror").resolve()

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

        def boom(path, text):
            raise OSError("read-only file system")

        target = "osprey.services.ariel_search.enhancement.qmd_export.exporter._atomic_write_text"
        with caplog.at_level(logging.WARNING), mock.patch(target, boom):
            await module.enhance(entry, _FakeConnection())

        assert (tmp_path / "2024" / "05" / "entry-001.md").is_file()
        assert "freshness marker" in caplog.text

    @pytest.mark.asyncio
    async def test_health_check_reports_missing_configuration(self):
        """Health check is explicit about an unconfigured mirror."""
        assert await QmdExportModule().health_check() == (False, "mirror_path is not configured")

    @pytest.mark.asyncio
    async def test_health_check_creates_the_mirror_root(self, tmp_path):
        """A configured, creatable mirror root reports healthy."""
        module = self._module(tmp_path / "mirror")

        healthy, message = await module.health_check()

        assert (healthy, message) == (True, "OK")
        assert (tmp_path / "mirror").is_dir()

    @pytest.mark.asyncio
    async def test_health_check_reports_an_unwritable_mirror_root(self, tmp_path):
        """A mirror root that cannot be created reports unhealthy rather than raising."""
        blocker = tmp_path / "mirror"
        blocker.write_text("not a directory")
        module = self._module(blocker)

        healthy, message = await module.health_check()

        assert healthy is False
        assert "not writable" in message
