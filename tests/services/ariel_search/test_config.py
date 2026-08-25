"""Tests for ARIEL configuration classes."""

from pathlib import Path

import pytest

from osprey.services.ariel_search.config import (
    ARIELConfig,
    DatabaseConfig,
    EmbeddingConfig,
    EnhancementModuleConfig,
    IngestionConfig,
    ModelConfig,
    SearchModuleConfig,
    VocabularyConfig,
    WatchConfig,
)
from osprey.services.ariel_search.exceptions import ConfigurationError, VocabularyError
from osprey.services.ariel_search.search.keyword import KeywordSearchSettings


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
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "models": [{"name": "nomic-embed-text", "dimension": 768}],
                },
                "semantic_processor": {"enabled": True},
            },
            "ingestion": {"adapter": "als_logbook"},
            "embedding": {"provider": "ollama"},
        }

    def test_from_dict_minimal(self, minimal_config_dict: dict) -> None:
        """Test ARIELConfig.from_dict() with minimal config."""
        config = ARIELConfig.from_dict(minimal_config_dict)
        assert config.database.uri == "postgresql://localhost:5432/ariel"
        assert config.search_modules == {}
        assert config.enhancement_modules == {}
        assert config.ingestion is None

    def test_from_dict_full(self, full_config_dict: dict) -> None:
        """Test ARIELConfig.from_dict() with full config."""
        config = ARIELConfig.from_dict(full_config_dict)
        assert config.database.uri == "postgresql://localhost:5432/ariel"
        assert len(config.search_modules) == 2
        assert len(config.enhancement_modules) == 2
        assert config.ingestion is not None
        assert config.ingestion.adapter == "als_logbook"
        assert config.embedding.provider == "ollama"

    def test_from_dict_raises_on_legacy_pipelines_key(self) -> None:
        """ARIELConfig.from_dict() rejects the deprecated pipelines section."""
        with pytest.raises(ConfigurationError, match="no longer supported"):
            ARIELConfig.from_dict(
                {
                    "database": {"uri": "postgresql://localhost:5432/ariel"},
                    "pipelines": {"rag": {"enabled": True}},
                }
            )

    def test_is_search_module_enabled(self, full_config_dict: dict) -> None:
        """Test is_search_module_enabled()."""
        config = ARIELConfig.from_dict(full_config_dict)
        assert config.is_search_module_enabled("keyword") is True
        assert config.is_search_module_enabled("semantic") is True
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

    def test_default_search_mode_unset_prefers_hybrid(self, minimal_config_dict: dict) -> None:
        """With no configured default, an enabled hybrid module is preferred."""
        minimal_config_dict["search_modules"] = {
            "keyword": {"enabled": True},
            "hybrid": {"enabled": True},
        }
        config = ARIELConfig.from_dict(minimal_config_dict)
        assert config.default_search_mode is None
        assert config.resolve_default_search_mode() == "hybrid"

    def test_default_search_mode_unset_falls_back_to_keyword(
        self, minimal_config_dict: dict
    ) -> None:
        """Without hybrid, the implicit default is the dependency-free mode."""
        minimal_config_dict["search_modules"] = {"keyword": {"enabled": True}}
        config = ARIELConfig.from_dict(minimal_config_dict)
        assert config.resolve_default_search_mode() == "keyword"

    def test_default_search_mode_is_normalized(self, minimal_config_dict: dict) -> None:
        """The configured name is case- and whitespace-normalized like any mode."""
        minimal_config_dict["search_modules"] = {"keyword": {"enabled": True}}
        minimal_config_dict["default_search_mode"] = "  KEYWORD  "
        config = ARIELConfig.from_dict(minimal_config_dict)
        assert config.default_search_mode == "keyword"
        assert config.resolve_default_search_mode() == "keyword"

    def test_validate_default_search_mode_not_enabled(self, minimal_config_dict: dict) -> None:
        """A default naming a disabled module is a config error, not a fallback."""
        minimal_config_dict["search_modules"] = {"keyword": {"enabled": True}}
        minimal_config_dict["default_search_mode"] = "hybrid"
        config = ARIELConfig.from_dict(minimal_config_dict)
        errors = config.validate()
        assert any("default_search_mode" in e for e in errors)

    def test_validate_default_search_mode_enabled(self, minimal_config_dict: dict) -> None:
        """A default naming an enabled module validates clean."""
        minimal_config_dict["search_modules"] = {
            "keyword": {"enabled": True},
            "hybrid": {"enabled": True},
        }
        minimal_config_dict["default_search_mode"] = "hybrid"
        config = ARIELConfig.from_dict(minimal_config_dict)
        assert not [e for e in config.validate() if "default_search_mode" in e]

    def test_validate_text_embedding_without_models(self, minimal_config_dict: dict) -> None:
        """Test validate() catches text_embedding without models."""
        minimal_config_dict["enhancement_modules"] = {"text_embedding": {"enabled": True}}
        config = ARIELConfig.from_dict(minimal_config_dict)
        errors = config.validate()
        assert any("text_embedding.models" in e for e in errors)

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


VALID_VOCABULARY = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - t/s
      - ts
  - canonical: beam position monitor
    kind: acronym
    forms:
      - BPM
"""


@pytest.fixture
def vocabulary_file(tmp_path: Path) -> Path:
    """A valid two-concept vocabulary file on disk."""
    path = tmp_path / "vocabulary.yml"
    path.write_text(VALID_VOCABULARY)
    return path


def _config_dict(**vocabulary: object) -> dict:
    """An otherwise-minimal ariel section carrying a vocabulary block."""
    config: dict = {"database": {"uri": "postgresql://localhost:5432/ariel"}}
    if vocabulary:
        config["vocabulary"] = dict(vocabulary)
    return config


class TestVocabularyConfig:
    """Tests for the ``ariel.vocabulary`` block itself."""

    def test_defaults_when_block_absent(self) -> None:
        """An ariel section with no vocabulary block yields the defaults."""
        config = ARIELConfig.from_dict(_config_dict())
        assert config.vocabulary == VocabularyConfig()
        assert config.vocabulary.enabled is False
        assert config.vocabulary.path is None
        assert config.vocabulary.expand_by_default is True
        assert config.vocabulary.canonical_to_acronym is True
        assert config.vocabulary.canonical_to_shorthand is False
        assert config.vocabulary.expand_modes is None
        assert config.loaded_vocabulary is None
        assert config.vocabulary_errors == []
        assert config.vocabulary_warnings == []

    def test_full_block_parse(self, vocabulary_file: Path) -> None:
        """Every key round-trips, with expand_modes normalized to a tuple."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/ariel"},
                "search_modules": {"keyword": {"enabled": True}},
                "vocabulary": {
                    "enabled": True,
                    "path": str(vocabulary_file),
                    "expand_by_default": False,
                    "canonical_to_acronym": False,
                    "canonical_to_shorthand": True,
                    "expand_modes": ["keyword"],
                },
            }
        )
        assert config.vocabulary == VocabularyConfig(
            enabled=True,
            path=str(vocabulary_file),
            expand_by_default=False,
            canonical_to_acronym=False,
            canonical_to_shorthand=True,
            expand_modes=("keyword",),
        )

    @pytest.mark.parametrize(
        "key",
        ["enabled", "expand_by_default", "canonical_to_acronym", "canonical_to_shorthand"],
    )
    def test_non_boolean_knob_is_refused(self, key: str) -> None:
        """A present-but-non-boolean knob raises, naming the full dotted key."""
        with pytest.raises(ValueError, match=rf"ariel\.vocabulary\.{key} must be a boolean"):
            ARIELConfig.from_dict(_config_dict(**{key: "yes"}))

    @pytest.mark.parametrize("value", [123, "", "   "])
    def test_bad_path_is_refused(self, value: object) -> None:
        """A non-string or blank path raises, naming ariel.vocabulary.path."""
        with pytest.raises(ValueError, match=r"ariel\.vocabulary\.path must be a non-empty"):
            ARIELConfig.from_dict(_config_dict(path=value))

    def test_expand_modes_must_be_a_list(self) -> None:
        """A scalar expand_modes raises, naming ariel.vocabulary.expand_modes."""
        with pytest.raises(ValueError, match=r"ariel\.vocabulary\.expand_modes must be a list"):
            ARIELConfig.from_dict(_config_dict(expand_modes="keyword"))

    @pytest.mark.parametrize("entry", [7, "", "  "])
    def test_expand_modes_entries_must_be_mode_names(self, entry: object) -> None:
        """A non-string or empty entry raises, naming the key."""
        with pytest.raises(ValueError, match=r"ariel\.vocabulary\.expand_modes entries must be"):
            ARIELConfig.from_dict(_config_dict(expand_modes=[entry]))

    def test_vocabulary_block_must_be_a_mapping(self) -> None:
        """A scalar vocabulary block raises, naming ariel.vocabulary."""
        with pytest.raises(ValueError, match=r"ariel\.vocabulary must be a mapping"):
            ARIELConfig.from_dict(
                {
                    "database": {"uri": "postgresql://localhost:5432/ariel"},
                    "vocabulary": "on",
                }
            )

    def test_expand_modes_are_normalized(self) -> None:
        """``[Keyword]`` normalizes to ``keyword`` at parse, like default_search_mode."""
        config = ARIELConfig.from_dict(_config_dict(expand_modes=[" Keyword "]))
        assert config.vocabulary.expand_modes == ("keyword",)

    def test_existing_positional_call_sites_are_unchanged(self) -> None:
        """from_dict(dict) and from_dict(dict, services) keep working."""
        assert ARIELConfig.from_dict({"database": {"uri": "x"}}).database.uri == "x"
        derived = ARIELConfig.from_dict({}, {"port_host": 6543})
        assert ":6543/" in derived.database.uri


class TestVocabularyValidation:
    """Tests for the vocabulary errors ``validate()`` reports."""

    def test_enabled_without_path(self) -> None:
        """enabled: true with no path is a validate() error naming the key."""
        config = ARIELConfig.from_dict(_config_dict(enabled=True))
        assert (
            "ariel.vocabulary.path is required when ariel.vocabulary.enabled is true"
            in config.validate()
        )

    def test_expand_modes_names_disabled_module(self) -> None:
        """A disabled module in expand_modes is refused with the exact text."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/ariel"},
                "search_modules": {
                    "keyword": {"enabled": True},
                    "semantic": {"enabled": True, "model": "nomic-embed-text"},
                    "hybrid": {"enabled": False},
                },
                "vocabulary": {"expand_modes": ["hybrid"]},
            }
        )
        assert (
            "ariel.vocabulary.expand_modes names no enabled search module: hybrid. "
            "Enabled modules: keyword, semantic" in config.validate()
        )

    def test_expand_modes_names_unknown_module(self) -> None:
        """A module nobody registered is refused the same way."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/ariel"},
                "search_modules": {"keyword": {"enabled": True}},
                "vocabulary": {"expand_modes": ["telepathy"]},
            }
        )
        assert (
            "ariel.vocabulary.expand_modes names no enabled search module: telepathy. "
            "Enabled modules: keyword" in config.validate()
        )

    def test_expand_modes_all_enabled_is_clean(self) -> None:
        """Naming only enabled modules produces no expand_modes error."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/ariel"},
                "search_modules": {"keyword": {"enabled": True}},
                "vocabulary": {"expand_modes": ["keyword"]},
            }
        )
        assert not [e for e in config.validate() if "expand_modes" in e]

    def test_missing_file_is_a_vocabulary_error(self, tmp_path: Path) -> None:
        """A missing file lands in vocabulary_errors, naming key and resolved path."""
        missing = tmp_path / "nope.yml"
        config = ARIELConfig.from_dict(_config_dict(enabled=True, path=str(missing)))
        assert config.loaded_vocabulary is None
        assert config.vocabulary_errors
        assert all(e.startswith("ariel.vocabulary.path: ") for e in config.vocabulary_errors)
        assert all(str(missing) in e for e in config.vocabulary_errors)
        assert config.vocabulary_active is False
        for error in config.vocabulary_errors:
            assert error in config.validate()

    def test_vocabulary_errors_are_distinct_from_the_pre_existing_classes(
        self, tmp_path: Path
    ) -> None:
        """The four pre-existing validate() classes still report separately."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": ""},
                "search_modules": {"semantic": {"enabled": True}},
                "enhancement_modules": {"text_embedding": {"enabled": True}},
                "default_search_mode": "keyword",
                "vocabulary": {"enabled": True, "path": str(tmp_path / "nope.yml")},
            }
        )
        errors = config.validate()
        assert "database.uri is required" in errors
        assert any("default_search_mode" in e for e in errors)
        assert any("search_modules.semantic.model is required" in e for e in errors)
        assert any("text_embedding.models is required" in e for e in errors)
        # The vocabulary errors are their own field, not folded into the above.
        assert config.vocabulary_errors
        assert set(config.vocabulary_errors) <= set(errors)
        assert not set(config.vocabulary_errors) & {
            e for e in errors if not e.startswith("ariel.vocabulary")
        }

    def test_malformed_file_reports_every_error(self, tmp_path: Path) -> None:
        """A malformed file loads nothing and reports the loader's errors."""
        path = tmp_path / "vocabulary.yml"
        path.write_text("concepts:\n  - canonical: ''\n    kind: acronym\n    forms: [x]\n")
        config = ARIELConfig.from_dict(_config_dict(enabled=True, path=str(path)))
        assert config.loaded_vocabulary is None
        assert config.vocabulary_errors
        assert config.vocabulary_active is False


class TestVocabularyLoading:
    """Tests for what ``from_dict`` actually reads off disk."""

    def test_valid_file_is_loaded(self, vocabulary_file: Path) -> None:
        """An enabled block with a good file yields a live vocabulary."""
        config = ARIELConfig.from_dict(
            _config_dict(enabled=True, path=str(vocabulary_file)),
        )
        assert config.vocabulary_errors == []
        assert config.loaded_vocabulary is not None
        assert config.loaded_vocabulary.concept_count == 2
        assert config.vocabulary_active is True

    def test_relative_path_resolves_against_config_dir(self, vocabulary_file: Path) -> None:
        """A relative path is resolved against the config file's directory."""
        config = ARIELConfig.from_dict(
            _config_dict(enabled=True, path="vocabulary.yml"),
            config_dir=vocabulary_file.parent,
        )
        assert config.vocabulary_errors == []
        assert config.loaded_vocabulary is not None
        assert config.loaded_vocabulary.concept_count == 2

    def test_relative_path_without_config_dir_falls_back_to_the_shared_rule(
        self, vocabulary_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitting config_dir defers to the shared helper, which lands on CWD here."""
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(vocabulary_file.parent)
        config = ARIELConfig.from_dict(_config_dict(enabled=True, path="vocabulary.yml"))
        assert config.vocabulary_errors == []
        assert config.loaded_vocabulary is not None

    def test_disabled_block_loads_nothing(self, tmp_path: Path) -> None:
        """A disabled block ignores its path entirely, missing file and all."""
        config = ARIELConfig.from_dict(
            _config_dict(enabled=False, path=str(tmp_path / "nope.yml")),
        )
        assert config.loaded_vocabulary is None
        assert config.vocabulary_errors == []
        assert config.vocabulary_active is False
        assert not [e for e in config.validate() if "vocabulary" in e]

    def test_warnings_are_kept_without_blocking(self, tmp_path: Path) -> None:
        """An ambiguous form warns but still loads."""
        path = tmp_path / "vocabulary.yml"
        path.write_text(
            "concepts:\n"
            "  - canonical: troubleshoot\n    kind: shorthand\n    forms: [ts]\n"
            "  - canonical: timing system\n    kind: acronym\n    forms: [ts]\n"
        )
        config = ARIELConfig.from_dict(_config_dict(enabled=True, path=str(path)))
        assert config.loaded_vocabulary is not None
        assert config.vocabulary_errors == []
        assert config.vocabulary_warnings
        assert config.vocabulary_active is True


class TestResolveExpandModes:
    """Tests for ``ARIELConfig.resolve_expand_modes()``."""

    def test_explicit_list_wins(self) -> None:
        """A configured list is returned verbatim."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/ariel"},
                "search_modules": {
                    "keyword": {"enabled": True},
                    "semantic": {"enabled": True, "model": "m"},
                },
                "vocabulary": {"expand_modes": ["keyword"]},
            }
        )
        assert config.resolve_expand_modes() == ("keyword",)

    def test_unset_means_every_enabled_module(self) -> None:
        """Unset resolves to all enabled search modules."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/ariel"},
                "search_modules": {
                    "keyword": {"enabled": True},
                    "semantic": {"enabled": True, "model": "m"},
                    "hybrid": {"enabled": False},
                },
            }
        )
        assert config.resolve_expand_modes() == ("keyword", "semantic")

    def test_empty_list_is_not_the_same_as_unset(self) -> None:
        """An explicit empty list disables expansion everywhere."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/ariel"},
                "search_modules": {"keyword": {"enabled": True}},
                "vocabulary": {"expand_modes": []},
            }
        )
        assert config.resolve_expand_modes() == ()


class TestVocabularyError:
    """Tests for the ``VocabularyError`` raised on the search path."""

    def test_message_names_the_first_error_and_counts_the_rest(self) -> None:
        """One message, a count of what it left out, and the full list kept."""
        error = VocabularyError(["first problem", "second problem", "third problem"])
        assert str(error) == "first problem (and 2 more)"
        assert error.errors == ["first problem", "second problem", "third problem"]

    def test_single_error_has_no_count(self) -> None:
        """A lone error is reported verbatim."""
        error = VocabularyError(["only problem"])
        assert str(error) == "only problem"

    def test_defaults_to_the_path_key_and_carries_the_remedy(self) -> None:
        """The key an operator edits, and the action that clears the failure."""
        error = VocabularyError(["broken"])
        assert error.config_key == "ariel.vocabulary.path"
        assert error.remedy == (
            "disable ariel.vocabulary.enabled or repoint ariel.vocabulary.path, then restart"
        )
        assert error.technical_details["errors"] == ["broken"]

    def test_is_a_configuration_error(self) -> None:
        """It is caught by every handler that already catches ConfigurationError."""
        assert isinstance(VocabularyError(["broken"]), ConfigurationError)

    def test_config_key_is_overridable(self) -> None:
        """A caller failing on a different key can say so."""
        error = VocabularyError(["broken"], config_key="ariel.vocabulary.enabled")
        assert error.config_key == "ariel.vocabulary.enabled"


def _keyword_config(
    settings: dict[str, object] | None = None, *, enabled: bool = True
) -> ARIELConfig:
    """Build a config whose keyword module carries the given settings block.

    Args:
        settings: The ``search_modules.keyword.settings`` mapping, or None to
            omit the block entirely.
        enabled: Whether the keyword module is enabled.

    Returns:
        The parsed configuration.
    """
    keyword: dict[str, object] = {"enabled": enabled}
    if settings is not None:
        keyword["settings"] = settings
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
            "search_modules": {"keyword": keyword},
        }
    )


class TestKeywordPatternSettings:
    """Tests for ``patterns_enabled`` / ``pattern_timeout_seconds`` resolution."""

    def test_patterns_enabled_and_pattern_timeout_default_when_block_absent(self) -> None:
        """No settings block is the normal case and yields the documented defaults."""
        settings = KeywordSearchSettings.from_ariel_config(_keyword_config())
        assert settings.patterns_enabled is True
        assert settings.pattern_timeout_seconds == 10.0

    def test_patterns_enabled_and_pattern_timeout_default_without_a_config(self) -> None:
        """``from_ariel_config(None)`` is the no-config path, not an error."""
        settings = KeywordSearchSettings.from_ariel_config(None)
        assert settings.patterns_enabled is True
        assert settings.pattern_timeout_seconds == 10.0

    def test_patterns_enabled_and_pattern_timeout_resolve_configured_values(self) -> None:
        """A well-formed block is read verbatim, no clamping in either field."""
        settings = KeywordSearchSettings.from_ariel_config(
            _keyword_config({"patterns_enabled": False, "pattern_timeout_seconds": 2.5})
        )
        assert settings.patterns_enabled is False
        assert settings.pattern_timeout_seconds == 2.5

    def test_patterns_enabled_rejects_a_string(self) -> None:
        """A quoted ``yes`` is refused, never defaulted."""
        with pytest.raises(ValueError) as exc_info:
            KeywordSearchSettings.from_ariel_config(_keyword_config({"patterns_enabled": "yes"}))
        assert (
            str(exc_info.value)
            == "search_modules.keyword.settings.patterns_enabled must be a boolean, got 'yes'"
        )

    def test_patterns_enabled_rejects_an_integer(self) -> None:
        """``1`` is not a boolean here; the YAML spelling is ``true``."""
        with pytest.raises(ValueError) as exc_info:
            KeywordSearchSettings.from_ariel_config(_keyword_config({"patterns_enabled": 1}))
        assert "search_modules.keyword.settings.patterns_enabled must be a boolean" in str(
            exc_info.value
        )

    def test_patterns_enabled_error_surfaces_from_validate(self) -> None:
        """The resolver validate() calls is the one the search path uses."""
        errors = _keyword_config({"patterns_enabled": "yes"}).validate()
        assert (
            "search_modules.keyword.settings.patterns_enabled must be a boolean, got 'yes'"
            in errors
        )

    def test_pattern_timeout_seconds_rejects_zero(self) -> None:
        """``0`` is below the floor and is named rather than clamped up to 10."""
        with pytest.raises(ValueError) as exc_info:
            KeywordSearchSettings.from_ariel_config(_keyword_config({"pattern_timeout_seconds": 0}))
        assert (
            str(exc_info.value)
            == "search_modules.keyword.settings.pattern_timeout_seconds must be a number "
            ">= 0.001, got 0"
        )

    def test_pattern_timeout_seconds_rejects_a_negative_number(self) -> None:
        """Below the floor from the other side, same refusal."""
        with pytest.raises(ValueError) as exc_info:
            KeywordSearchSettings.from_ariel_config(
                _keyword_config({"pattern_timeout_seconds": -1.0})
            )
        assert "pattern_timeout_seconds must be a number >= 0.001" in str(exc_info.value)

    def test_pattern_timeout_seconds_accepts_the_floor(self) -> None:
        """0.001 is inside the range, the value the timeout-effect test configures."""
        settings = KeywordSearchSettings.from_ariel_config(
            _keyword_config({"pattern_timeout_seconds": 0.001})
        )
        assert settings.pattern_timeout_seconds == 0.001

    def test_pattern_timeout_seconds_accepts_an_integer(self) -> None:
        """``5`` is a number; it resolves as a float."""
        settings = KeywordSearchSettings.from_ariel_config(
            _keyword_config({"pattern_timeout_seconds": 5})
        )
        assert settings.pattern_timeout_seconds == 5.0

    def test_pattern_timeout_seconds_rejects_a_string(self) -> None:
        """A quoted ``10`` is YAML text, not a number, and is refused."""
        with pytest.raises(ValueError) as exc_info:
            KeywordSearchSettings.from_ariel_config(
                _keyword_config({"pattern_timeout_seconds": "10"})
            )
        assert (
            str(exc_info.value)
            == "search_modules.keyword.settings.pattern_timeout_seconds must be a number "
            ">= 0.001, got '10'"
        )

    def test_pattern_timeout_seconds_rejects_a_boolean(self) -> None:
        """``True`` is an int subclass in Python but means nothing as a timeout."""
        with pytest.raises(ValueError) as exc_info:
            KeywordSearchSettings.from_ariel_config(
                _keyword_config({"pattern_timeout_seconds": True})
            )
        assert "pattern_timeout_seconds must be a number >= 0.001, got True" in str(exc_info.value)

    def test_pattern_timeout_seconds_error_surfaces_from_validate(self) -> None:
        """``vocab-check``, ``status`` and startup all report it, naming the key."""
        errors = _keyword_config({"pattern_timeout_seconds": 0}).validate()
        assert (
            "search_modules.keyword.settings.pattern_timeout_seconds must be a number "
            ">= 0.001, got 0" in errors
        )

    def test_pattern_timeout_seconds_is_not_validated_when_keyword_is_disabled(self) -> None:
        """A disabled module's settings reach no reader, so validate() stays quiet."""
        errors = _keyword_config({"pattern_timeout_seconds": 0}, enabled=False).validate()
        assert not [error for error in errors if "pattern_timeout_seconds" in error]

    def test_patterns_enabled_is_not_validated_when_keyword_is_disabled(self) -> None:
        """Same for the boolean knob: dead config is not refused."""
        errors = _keyword_config({"patterns_enabled": "yes"}, enabled=False).validate()
        assert not [error for error in errors if "patterns_enabled" in error]


def _hybrid_config(
    settings: dict[str, object] | None = None, *, enabled: bool = True
) -> ARIELConfig:
    """Build a config whose hybrid module carries the given settings block.

    Args:
        settings: The ``search_modules.hybrid.settings`` mapping, or None to
            omit the block entirely.
        enabled: Whether the hybrid module is enabled.

    Returns:
        The parsed configuration.
    """
    hybrid: dict[str, object] = {"enabled": enabled}
    if settings is not None:
        hybrid["settings"] = settings
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
            "search_modules": {"hybrid": hybrid},
        }
    )


def _semantic_config(
    settings: dict[str, object] | None = None, *, enabled: bool = True
) -> ARIELConfig:
    """Build a config whose semantic module carries the given settings block.

    A ``model`` is always named: semantic search without one is a separate,
    unrelated validate() error, and these tests are about the settings block.

    Args:
        settings: The ``search_modules.semantic.settings`` mapping, or None to
            omit the block entirely.
        enabled: Whether the semantic module is enabled.

    Returns:
        The parsed configuration.
    """
    semantic: dict[str, object] = {"enabled": enabled, "model": "nomic-embed-text"}
    if settings is not None:
        semantic["settings"] = settings
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
            "search_modules": {"semantic": semantic},
        }
    )


class TestHybridSettingsValidation:
    """Tests for ``search_modules.hybrid.settings`` reaching validate()."""

    def test_rerank_error_surfaces_from_validate(self) -> None:
        """``vocab-check``, ``status`` and startup all report it, naming the key."""
        errors = _hybrid_config({"rerank": "junk"}).validate()
        assert "search_modules.hybrid.settings.rerank must be a boolean, got 'junk'" in errors

    def test_candidate_limit_error_surfaces_from_validate(self) -> None:
        """The other knob in the block is validated by the same resolver call."""
        errors = _hybrid_config({"candidate_limit": 0}).validate()
        assert (
            "search_modules.hybrid.settings.candidate_limit must be a positive integer, got 0"
            in errors
        )

    def test_settings_are_not_validated_when_hybrid_is_disabled(self) -> None:
        """A disabled module's settings reach no reader, so validate() stays quiet."""
        errors = _hybrid_config({"rerank": "junk"}, enabled=False).validate()
        assert not [error for error in errors if "search_modules.hybrid.settings" in error]

    def test_absent_settings_block_is_quiet_when_hybrid_is_enabled(self) -> None:
        """No block is the normal case: the defaults resolve and nothing is reported."""
        errors = _hybrid_config().validate()
        assert not [error for error in errors if "search_modules.hybrid.settings" in error]


class TestSemanticSettingsValidation:
    """Tests for ``search_modules.semantic.settings`` reaching validate()."""

    def test_similarity_threshold_error_surfaces_from_validate(self) -> None:
        """``vocab-check``, ``status`` and startup all report it, naming the key."""
        errors = _semantic_config({"similarity_threshold": "high"}).validate()
        assert (
            "search_modules.semantic.settings.similarity_threshold must be a float in [0, 1], "
            "got 'high'" in errors
        )

    def test_out_of_range_threshold_surfaces_from_validate(self) -> None:
        """A number outside ``[0, 1]`` is named rather than clamped into range."""
        errors = _semantic_config({"similarity_threshold": 1.5}).validate()
        assert (
            "search_modules.semantic.settings.similarity_threshold must be a float in [0, 1], "
            "got 1.5" in errors
        )

    def test_settings_are_not_validated_when_semantic_is_disabled(self) -> None:
        """A disabled module's settings reach no reader, so validate() stays quiet."""
        errors = _semantic_config({"similarity_threshold": "high"}, enabled=False).validate()
        assert not [error for error in errors if "search_modules.semantic.settings" in error]

    def test_absent_settings_block_is_quiet_when_semantic_is_enabled(self) -> None:
        """No block is the normal case: the defaults resolve and nothing is reported."""
        errors = _semantic_config().validate()
        assert not [error for error in errors if "search_modules.semantic.settings" in error]
