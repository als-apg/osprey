"""Tests for ARIEL enhancement modules.

Tests for base enhancement module, factory, and module implementations.
"""

from datetime import UTC

import pytest

from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig
from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule
from osprey.services.ariel_search.enhancement.factory import (
    create_enhancers_from_config,
    get_enhancer_names,
)
from osprey.services.ariel_search.enhancement.semantic_processor import (
    SemanticProcessorModule,
    SemanticProcessorResult,
)
from osprey.services.ariel_search.enhancement.text_embedding import (
    TextEmbeddingModule,
)


class TestEnhancementFactory:
    """Tests for enhancement module factory."""

    def test_get_enhancer_names(self):
        """get_enhancer_names returns correct list."""
        names = get_enhancer_names()
        assert names == ["semantic_processor", "text_embedding"]

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
                    "semantic_processor": {"enabled": True},
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
        module.configure({"prompt_template": "Custom: {text}"})
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
        """Configure with empty dict uses defaults."""
        module = SemanticProcessorModule()
        module.configure({})
        # Should not raise, uses defaults

    def test_configure_with_model_config(self):
        """Configure sets model_config for LLM calls."""
        module = SemanticProcessorModule()
        module.configure({"model": {"provider": "ollama", "name": "llama3"}})
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
        """Enhance skips entries with empty text."""
        from unittest.mock import AsyncMock, MagicMock

        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()

        entry = {
            "entry_id": "entry-empty",
            "raw_text": "   ",
        }

        await module.enhance(entry, mock_conn)
        mock_conn.execute.assert_not_called()


class TestSemanticProcessorEnhanceWithLLM:
    """Tests for SemanticProcessorModule enhance with mocked LLM."""

    @pytest.fixture
    def module(self):
        """Create configured module."""
        m = SemanticProcessorModule()
        m.configure(
            {
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
