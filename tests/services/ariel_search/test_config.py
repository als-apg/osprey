"""Tests for ARIEL configuration classes."""

import pytest

from osprey.services.ariel_search.config import (
    ARIELConfig,
    DatabaseConfig,
    EmbeddingConfig,
    EnhancementModuleConfig,
    IngestionConfig,
    ModelConfig,
    PipelineModuleConfig,
    ReasoningConfig,
    SearchModuleConfig,
    WatchConfig,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_basic_creation(self) -> None:
        """Test basic ModelConfig creation."""
        config = ModelConfig(name="nomic-embed-text", dimension=768)
        assert config.name == "nomic-embed-text"
        assert config.dimension == 768
        assert config.max_input_tokens is None

    def test_with_max_input_tokens(self) -> None:
        """Test ModelConfig with max_input_tokens."""
        config = ModelConfig(
            name="nomic-embed-text",
            dimension=768,
            max_input_tokens=8192,
        )
        assert config.max_input_tokens == 8192

    def test_from_dict(self) -> None:
        """Test ModelConfig.from_dict()."""
        data = {"name": "nomic-embed-text", "dimension": 768, "max_input_tokens": 8192}
        config = ModelConfig.from_dict(data)
        assert config.name == "nomic-embed-text"
        assert config.dimension == 768
        assert config.max_input_tokens == 8192

    def test_from_dict_minimal(self) -> None:
        """Test ModelConfig.from_dict() with minimal data."""
        data = {"name": "test-model", "dimension": 512}
        config = ModelConfig.from_dict(data)
        assert config.name == "test-model"
        assert config.dimension == 512
        assert config.max_input_tokens is None


class TestSearchModuleConfig:
    """Tests for SearchModuleConfig."""

    def test_basic_creation(self) -> None:
        """Test basic SearchModuleConfig creation."""
        config = SearchModuleConfig(enabled=True)
        assert config.enabled is True
        assert config.model is None
        assert config.settings == {}

    def test_with_model(self) -> None:
        """Test SearchModuleConfig with model."""
        config = SearchModuleConfig(enabled=True, model="nomic-embed-text")
        assert config.model == "nomic-embed-text"

    def test_from_dict(self) -> None:
        """Test SearchModuleConfig.from_dict()."""
        data = {
            "enabled": True,
            "model": "nomic-embed-text",
            "settings": {"threshold": 0.7},
        }
        config = SearchModuleConfig.from_dict(data)
        assert config.enabled is True
        assert config.model == "nomic-embed-text"
        assert config.settings == {"threshold": 0.7}

    def test_from_dict_defaults(self) -> None:
        """Test SearchModuleConfig.from_dict() with defaults."""
        config = SearchModuleConfig.from_dict({})
        assert config.enabled is False
        assert config.model is None
        assert config.settings == {}


class TestEnhancementModuleConfig:
    """Tests for EnhancementModuleConfig."""

    def test_basic_creation(self) -> None:
        """Test basic EnhancementModuleConfig creation."""
        config = EnhancementModuleConfig(enabled=True)
        assert config.enabled is True
        assert config.models is None
        assert config.settings == {}

    def test_with_models(self) -> None:
        """Test EnhancementModuleConfig with models."""
        models = [
            ModelConfig(name="nomic-embed-text", dimension=768),
            ModelConfig(name="all-minilm", dimension=384),
        ]
        config = EnhancementModuleConfig(enabled=True, models=models)
        assert config.models == models
        assert len(config.models) == 2

    def test_from_dict(self) -> None:
        """Test EnhancementModuleConfig.from_dict()."""
        data = {
            "enabled": True,
            "models": [
                {"name": "nomic-embed-text", "dimension": 768},
                {"name": "all-minilm", "dimension": 384},
            ],
            "settings": {"batch_size": 100},
        }
        config = EnhancementModuleConfig.from_dict(data)
        assert config.enabled is True
        assert config.models is not None
        assert len(config.models) == 2
        assert config.models[0].name == "nomic-embed-text"
        assert config.settings == {"batch_size": 100}


class TestIngestionConfig:
    """Tests for IngestionConfig."""

    def test_basic_creation(self) -> None:
        """Test basic IngestionConfig creation."""
        config = IngestionConfig(adapter="als_logbook")
        assert config.adapter == "als_logbook"
        assert config.source_url is None
        assert config.poll_interval_seconds == 3600

    def test_from_dict(self) -> None:
        """Test IngestionConfig.from_dict()."""
        data = {
            "adapter": "als_logbook",
            "source_url": "https://als.example.com/api",
            "poll_interval_seconds": 1800,
        }
        config = IngestionConfig.from_dict(data)
        assert config.adapter == "als_logbook"
        assert config.source_url == "https://als.example.com/api"
        assert config.poll_interval_seconds == 1800

    def test_watch_defaults(self) -> None:
        """IngestionConfig has default WatchConfig."""
        config = IngestionConfig(adapter="generic_json")
        assert isinstance(config.watch, WatchConfig)
        assert config.watch.require_initial_ingest is True
        assert config.watch.max_consecutive_failures == 10

    def test_from_dict_with_watch(self) -> None:
        """IngestionConfig.from_dict() parses nested watch section."""
        data = {
            "adapter": "als_logbook",
            "watch": {
                "require_initial_ingest": False,
                "max_consecutive_failures": 5,
                "backoff_multiplier": 3.0,
                "max_interval_seconds": 7200,
            },
        }
        config = IngestionConfig.from_dict(data)
        assert config.watch.require_initial_ingest is False
        assert config.watch.max_consecutive_failures == 5
        assert config.watch.backoff_multiplier == 3.0
        assert config.watch.max_interval_seconds == 7200


class TestWatchConfig:
    """Tests for WatchConfig."""

    def test_defaults(self) -> None:
        """WatchConfig has sensible defaults."""
        config = WatchConfig()
        assert config.require_initial_ingest is True
        assert config.max_consecutive_failures == 10
        assert config.backoff_multiplier == 2.0
        assert config.max_interval_seconds == 3600

    def test_from_dict(self) -> None:
        """WatchConfig.from_dict() parses all fields."""
        data = {
            "require_initial_ingest": False,
            "max_consecutive_failures": 20,
            "backoff_multiplier": 1.5,
            "max_interval_seconds": 1800,
        }
        config = WatchConfig.from_dict(data)
        assert config.require_initial_ingest is False
        assert config.max_consecutive_failures == 20
        assert config.backoff_multiplier == 1.5
        assert config.max_interval_seconds == 1800

    def test_from_dict_defaults(self) -> None:
        """WatchConfig.from_dict() with empty dict gives defaults."""
        config = WatchConfig.from_dict({})
        assert config.require_initial_ingest is True
        assert config.max_consecutive_failures == 10
        assert config.backoff_multiplier == 2.0
        assert config.max_interval_seconds == 3600


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_basic_creation(self) -> None:
        """Test basic DatabaseConfig creation."""
        config = DatabaseConfig(uri="postgresql://localhost:5432/ariel")
        assert config.uri == "postgresql://localhost:5432/ariel"

    def test_from_dict(self) -> None:
        """Test DatabaseConfig.from_dict()."""
        data = {"uri": "postgresql://localhost:5432/ariel"}
        config = DatabaseConfig.from_dict(data)
        assert config.uri == "postgresql://localhost:5432/ariel"


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_defaults(self) -> None:
        """Test EmbeddingConfig defaults."""
        config = EmbeddingConfig()
        assert config.provider == "ollama"

    def test_from_dict(self) -> None:
        """Test EmbeddingConfig.from_dict()."""
        data = {"provider": "openai"}
        config = EmbeddingConfig.from_dict(data)
        assert config.provider == "openai"


class TestReasoningConfig:
    """Tests for ReasoningConfig."""

    def test_defaults(self) -> None:
        """Test ReasoningConfig defaults."""
        config = ReasoningConfig()
        assert config.max_iterations == 5
        assert config.temperature == 0.1
        assert config.tool_timeout_seconds == 30
        assert config.total_timeout_seconds == 120

    def test_from_dict(self) -> None:
        """Test ReasoningConfig.from_dict()."""
        data = {
            "max_iterations": 10,
            "temperature": 0.2,
            "total_timeout_seconds": 180,
        }
        config = ReasoningConfig.from_dict(data)
        assert config.max_iterations == 10
        assert config.temperature == 0.2
        assert config.total_timeout_seconds == 180


class TestARIELConfig:
    """Tests for ARIELConfig."""

    @pytest.fixture
    def minimal_config_dict(self) -> dict:
        """Minimal valid configuration dictionary."""
        return {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
        }

    @pytest.fixture
    def full_config_dict(self) -> dict:
        """Full configuration dictionary."""
        return {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "nomic-embed-text"},
            },
            "pipelines": {
                "rag": {
                    "enabled": True,
                    "retrieval_modules": ["keyword", "semantic"],
                },
                "agent": {
                    "enabled": True,
                    "retrieval_modules": ["keyword"],
                },
            },
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "models": [{"name": "nomic-embed-text", "dimension": 768}],
                },
                "semantic_processor": {"enabled": True},
            },
            "ingestion": {"adapter": "als_logbook"},
            "reasoning": {"max_iterations": 10},
            "embedding": {"provider": "ollama"},
            "default_max_results": 20,
            "cache_embeddings": False,
        }

    def test_from_dict_minimal(self, minimal_config_dict: dict) -> None:
        """Test ARIELConfig.from_dict() with minimal config."""
        config = ARIELConfig.from_dict(minimal_config_dict)
        assert config.database.uri == "postgresql://localhost:5432/ariel"
        assert config.search_modules == {}
        assert config.enhancement_modules == {}
        assert config.ingestion is None
        assert config.default_max_results == 10

    def test_from_dict_full(self, full_config_dict: dict) -> None:
        """Test ARIELConfig.from_dict() with full config."""
        config = ARIELConfig.from_dict(full_config_dict)
        assert config.database.uri == "postgresql://localhost:5432/ariel"
        assert len(config.search_modules) == 2
        assert len(config.pipelines) == 2
        assert len(config.enhancement_modules) == 2
        assert config.ingestion is not None
        assert config.ingestion.adapter == "als_logbook"
        assert config.reasoning.max_iterations == 10
        assert config.default_max_results == 20
        assert config.cache_embeddings is False

    def test_is_search_module_enabled(self, full_config_dict: dict) -> None:
        """Test is_search_module_enabled()."""
        config = ARIELConfig.from_dict(full_config_dict)
        assert config.is_search_module_enabled("keyword") is True
        assert config.is_search_module_enabled("semantic") is True
        assert config.is_search_module_enabled("rag") is False  # rag is a pipeline, not a module
        assert config.is_search_module_enabled("vision") is False
        assert config.is_search_module_enabled("nonexistent") is False

    def test_get_enabled_search_modules(self, full_config_dict: dict) -> None:
        """Test get_enabled_search_modules()."""
        config = ARIELConfig.from_dict(full_config_dict)
        enabled = config.get_enabled_search_modules()
        assert "keyword" in enabled
        assert "semantic" in enabled
        assert len(enabled) == 2

    def test_is_enhancement_module_enabled(self, full_config_dict: dict) -> None:
        """Test is_enhancement_module_enabled()."""
        config = ARIELConfig.from_dict(full_config_dict)
        assert config.is_enhancement_module_enabled("text_embedding") is True
        assert config.is_enhancement_module_enabled("semantic_processor") is True
        assert config.is_enhancement_module_enabled("figure_embedding") is False

    def test_get_enabled_enhancement_modules(self, full_config_dict: dict) -> None:
        """Test get_enabled_enhancement_modules()."""
        config = ARIELConfig.from_dict(full_config_dict)
        enabled = config.get_enabled_enhancement_modules()
        assert "text_embedding" in enabled
        assert "semantic_processor" in enabled
        assert len(enabled) == 2

    def test_validate_minimal(self, minimal_config_dict: dict) -> None:
        """Test validate() with minimal valid config."""
        config = ARIELConfig.from_dict(minimal_config_dict)
        errors = config.validate()
        assert errors == []

    def test_validate_full(self, full_config_dict: dict) -> None:
        """Test validate() with full valid config."""
        config = ARIELConfig.from_dict(full_config_dict)
        errors = config.validate()
        assert errors == []

    def test_validate_empty_uri(self) -> None:
        """Test validate() catches empty database URI."""
        config = ARIELConfig(database=DatabaseConfig(uri=""))
        errors = config.validate()
        assert "database.uri is required" in errors

    def test_validate_semantic_without_model(self, minimal_config_dict: dict) -> None:
        """Test validate() catches semantic search without model."""
        minimal_config_dict["search_modules"] = {"semantic": {"enabled": True}}
        config = ARIELConfig.from_dict(minimal_config_dict)
        errors = config.validate()
        assert any("semantic.model" in e for e in errors)

    def test_validate_text_embedding_without_models(self, minimal_config_dict: dict) -> None:
        """Test validate() catches text_embedding without models."""
        minimal_config_dict["enhancement_modules"] = {"text_embedding": {"enabled": True}}
        config = ARIELConfig.from_dict(minimal_config_dict)
        errors = config.validate()
        assert any("text_embedding.models" in e for e in errors)

    def test_validate_invalid_reasoning(self, minimal_config_dict: dict) -> None:
        """Test validate() catches invalid reasoning config."""
        minimal_config_dict["reasoning"] = {"max_iterations": 0}
        config = ARIELConfig.from_dict(minimal_config_dict)
        errors = config.validate()
        assert any("max_iterations" in e for e in errors)

    def test_get_search_model(self, full_config_dict: dict) -> None:
        """Test get_search_model()."""
        config = ARIELConfig.from_dict(full_config_dict)
        model = config.get_search_model()
        assert model == "nomic-embed-text"

    def test_get_search_model_disabled(self, minimal_config_dict: dict) -> None:
        """Test get_search_model() when semantic not enabled."""
        config = ARIELConfig.from_dict(minimal_config_dict)
        model = config.get_search_model()
        assert model is None

    def test_from_dict_with_pipelines(self, minimal_config_dict: dict) -> None:
        """Test ARIELConfig.from_dict() with pipelines section."""
        minimal_config_dict["pipelines"] = {
            "rag": {"enabled": True, "retrieval_modules": ["keyword"]},
            "agent": {"enabled": False},
        }
        config = ARIELConfig.from_dict(minimal_config_dict)
        assert len(config.pipelines) == 2
        assert config.pipelines["rag"].enabled is True
        assert config.pipelines["rag"].retrieval_modules == ["keyword"]
        assert config.pipelines["agent"].enabled is False

    def test_is_pipeline_enabled(self, full_config_dict: dict) -> None:
        """Test is_pipeline_enabled() for configured pipelines."""
        config = ARIELConfig.from_dict(full_config_dict)
        assert config.is_pipeline_enabled("rag") is True
        assert config.is_pipeline_enabled("agent") is True

    def test_is_pipeline_enabled_default(self, minimal_config_dict: dict) -> None:
        """Test is_pipeline_enabled() defaults to True for unconfigured pipelines."""
        config = ARIELConfig.from_dict(minimal_config_dict)
        assert config.is_pipeline_enabled("rag") is True
        assert config.is_pipeline_enabled("agent") is True

    def test_is_pipeline_disabled(self, minimal_config_dict: dict) -> None:
        """Test is_pipeline_enabled() returns False for disabled pipeline."""
        minimal_config_dict["pipelines"] = {"rag": {"enabled": False}}
        config = ARIELConfig.from_dict(minimal_config_dict)
        assert config.is_pipeline_enabled("rag") is False
        assert config.is_pipeline_enabled("agent") is True  # Not configured = default True

    def test_get_enabled_pipelines(self, full_config_dict: dict) -> None:
        """Test get_enabled_pipelines() returns correct list."""
        config = ARIELConfig.from_dict(full_config_dict)
        enabled = config.get_enabled_pipelines()
        assert "rag" in enabled
        assert "agent" in enabled

    def test_get_enabled_pipelines_with_disabled(self, minimal_config_dict: dict) -> None:
        """Test get_enabled_pipelines() excludes disabled pipelines."""
        minimal_config_dict["pipelines"] = {
            "rag": {"enabled": True},
            "agent": {"enabled": False},
        }
        config = ARIELConfig.from_dict(minimal_config_dict)
        enabled = config.get_enabled_pipelines()
        assert "rag" in enabled
        assert "agent" not in enabled

    def test_get_pipeline_retrieval_modules(self, full_config_dict: dict) -> None:
        """Test get_pipeline_retrieval_modules() returns configured list."""
        config = ARIELConfig.from_dict(full_config_dict)
        modules = config.get_pipeline_retrieval_modules("rag")
        assert modules == ["keyword", "semantic"]
        modules = config.get_pipeline_retrieval_modules("agent")
        assert modules == ["keyword"]

    def test_get_pipeline_retrieval_modules_default(self, minimal_config_dict: dict) -> None:
        """Test get_pipeline_retrieval_modules() falls back to enabled search modules."""
        minimal_config_dict["search_modules"] = {
            "keyword": {"enabled": True},
            "semantic": {"enabled": True, "model": "test"},
        }
        config = ARIELConfig.from_dict(minimal_config_dict)
        modules = config.get_pipeline_retrieval_modules("rag")
        assert "keyword" in modules
        assert "semantic" in modules


class TestPipelineModuleConfig:
    """Tests for PipelineModuleConfig."""

    def test_basic_creation(self) -> None:
        """Test PipelineModuleConfig default creation."""
        config = PipelineModuleConfig()
        assert config.enabled is True
        assert config.retrieval_modules == ["keyword", "semantic"]
        assert config.settings == {}

    def test_from_dict(self) -> None:
        """Test PipelineModuleConfig.from_dict() with full config."""
        data = {
            "enabled": True,
            "retrieval_modules": ["keyword", "semantic"],
            "settings": {"max_context_chars": 8000},
        }
        config = PipelineModuleConfig.from_dict(data)
        assert config.enabled is True
        assert config.retrieval_modules == ["keyword", "semantic"]
        assert config.settings == {"max_context_chars": 8000}

    def test_from_dict_defaults(self) -> None:
        """Test PipelineModuleConfig.from_dict() with empty dict gives defaults."""
        config = PipelineModuleConfig.from_dict({})
        assert config.enabled is True
        assert config.retrieval_modules == ["keyword", "semantic"]
        assert config.settings == {}

    def test_from_dict_minimal(self) -> None:
        """Test PipelineModuleConfig.from_dict() with only enabled flag."""
        config = PipelineModuleConfig.from_dict({"enabled": False})
        assert config.enabled is False
        assert config.retrieval_modules == ["keyword", "semantic"]
