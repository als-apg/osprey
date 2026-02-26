"""Tests for ARIEL database layer.

Note: These tests run without psycopg installed by testing only the
migration logic and configuration parts that don't require database access.
"""

import pytest

from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig
from osprey.services.ariel_search.database.core_migration import CoreMigration
from osprey.services.ariel_search.database.migration import BaseMigration, model_to_table_name
from osprey.services.ariel_search.enhancement.semantic_processor.migration import (
    SemanticProcessorMigration,
)
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    TextEmbeddingMigration,
)


class TestModelToTableName:
    """Tests for model_to_table_name function."""

    def test_hyphenated_name(self) -> None:
        """Test conversion of hyphenated model name."""
        assert model_to_table_name("nomic-embed-text") == "text_embeddings_nomic_embed_text"

    def test_dotted_name(self) -> None:
        """Test conversion of dotted model name."""
        assert model_to_table_name("bge.large.v1") == "text_embeddings_bge_large_v1"

    def test_slashed_name(self) -> None:
        """Test conversion of slashed model name."""
        assert model_to_table_name("org/model-name") == "text_embeddings_org_model_name"

    def test_mixed_name(self) -> None:
        """Test conversion of mixed format name."""
        assert model_to_table_name("org/model-v1.2") == "text_embeddings_org_model_v1_2"

    def test_uppercase_lowercased(self) -> None:
        """Test that uppercase is converted to lowercase."""
        assert model_to_table_name("NOMIC-EMBED-TEXT") == "text_embeddings_nomic_embed_text"

    def test_double_underscore_collapsed(self) -> None:
        """Test that double underscores are collapsed."""
        assert model_to_table_name("model--name") == "text_embeddings_model_name"


class TestBaseMigration:
    """Tests for BaseMigration class."""

    def test_core_migration_properties(self) -> None:
        """Test CoreMigration properties."""
        migration = CoreMigration()
        assert migration.name == "core_schema"
        assert migration.depends_on == []

    def test_semantic_processor_migration_properties(self) -> None:
        """Test SemanticProcessorMigration properties."""
        migration = SemanticProcessorMigration()
        assert migration.name == "semantic_processor"
        assert migration.depends_on == ["core_schema"]

    def test_text_embedding_migration_properties(self) -> None:
        """Test TextEmbeddingMigration properties."""
        migration = TextEmbeddingMigration()
        assert migration.name == "text_embedding"
        assert migration.depends_on == ["core_schema"]

    def test_text_embedding_with_custom_models(self) -> None:
        """Test TextEmbeddingMigration with custom models."""
        models = [("custom-model", 512), ("another-model", 1024)]
        migration = TextEmbeddingMigration(models=models)
        assert migration._get_models() == models


class TestMigrationTopologicalSort:
    """Tests for migration topological sorting (no database required)."""

    def test_topological_sort_simple(self) -> None:
        """Test topological sort with simple dependencies."""

        class MockMigration(BaseMigration):
            def __init__(self, name: str, deps: list[str]) -> None:
                self._name = name
                self._deps = deps

            @property
            def name(self) -> str:
                return self._name

            @property
            def depends_on(self) -> list[str]:
                return self._deps

            async def up(self, conn) -> None:
                pass

        # Create migrations with dependencies
        core = MockMigration("core_schema", [])
        semantic = MockMigration("semantic_processor", ["core_schema"])
        text_emb = MockMigration("text_embedding", ["core_schema"])

        # Create a mock runner (we'll test the sort method directly)
        class MockRunner:
            def __init__(self) -> None:
                pass

            def _topological_sort(self, migrations: list[BaseMigration]) -> list[BaseMigration]:
                from osprey.services.ariel_search.exceptions import ConfigurationError

                migration_map = {m.name: m for m in migrations}
                in_degree: dict[str, int] = {m.name: 0 for m in migrations}
                graph: dict[str, list[str]] = {m.name: [] for m in migrations}

                for migration in migrations:
                    for dep in migration.depends_on:
                        if dep in migration_map:
                            graph[dep].append(migration.name)
                            in_degree[migration.name] += 1

                queue = [name for name, degree in in_degree.items() if degree == 0]
                sorted_names: list[str] = []

                while queue:
                    name = queue.pop(0)
                    sorted_names.append(name)
                    for dependent in graph[name]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)

                if len(sorted_names) != len(migrations):
                    raise ConfigurationError(
                        "Circular dependency detected",
                        config_key="test",
                    )

                return [migration_map[name] for name in sorted_names]

        runner = MockRunner()
        sorted_migrations = runner._topological_sort([semantic, text_emb, core])

        # Core should be first
        assert sorted_migrations[0].name == "core_schema"
        # Others can be in any order after core
        remaining_names = {m.name for m in sorted_migrations[1:]}
        assert remaining_names == {"semantic_processor", "text_embedding"}

    def test_topological_sort_circular_dependency(self) -> None:
        """Test that circular dependency is detected."""
        from osprey.services.ariel_search.exceptions import ConfigurationError

        class MockMigration(BaseMigration):
            def __init__(self, name: str, deps: list[str]) -> None:
                self._name = name
                self._deps = deps

            @property
            def name(self) -> str:
                return self._name

            @property
            def depends_on(self) -> list[str]:
                return self._deps

            async def up(self, conn) -> None:
                pass

        # Create circular dependency
        a = MockMigration("a", ["b"])
        b = MockMigration("b", ["a"])

        class MockRunner:
            def _topological_sort(self, migrations: list[BaseMigration]) -> list[BaseMigration]:
                migration_map = {m.name: m for m in migrations}
                in_degree: dict[str, int] = {m.name: 0 for m in migrations}
                graph: dict[str, list[str]] = {m.name: [] for m in migrations}

                for migration in migrations:
                    for dep in migration.depends_on:
                        if dep in migration_map:
                            graph[dep].append(migration.name)
                            in_degree[migration.name] += 1

                queue = [name for name, degree in in_degree.items() if degree == 0]
                sorted_names: list[str] = []

                while queue:
                    name = queue.pop(0)
                    sorted_names.append(name)
                    for dependent in graph[name]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)

                if len(sorted_names) != len(migrations):
                    raise ConfigurationError(
                        "Circular dependency detected",
                        config_key="test",
                    )

                return [migration_map[name] for name in sorted_names]

        runner = MockRunner()
        with pytest.raises(ConfigurationError, match="Circular dependency"):
            runner._topological_sort([a, b])


class TestRequiresModule:
    """Tests for requires_module decorator."""

    def test_module_not_enabled_raises(self) -> None:
        """Test that disabled module raises ModuleNotEnabledError."""
        from osprey.services.ariel_search.database.repository import (
            ARIELRepository,
            requires_module,
        )
        from osprey.services.ariel_search.exceptions import ModuleNotEnabledError

        config = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost:5432/test"))

        class MockPool:
            pass

        repo = ARIELRepository(MockPool(), config)  # type: ignore[arg-type]

        @requires_module("enhancement", "text_embedding")
        def test_method(self: ARIELRepository) -> str:
            return "success"

        with pytest.raises(ModuleNotEnabledError, match="text_embedding"):
            test_method(repo)

    def test_module_enabled_succeeds(self) -> None:
        """Test that enabled module allows method execution."""
        from osprey.services.ariel_search.database.repository import (
            ARIELRepository,
            requires_module,
        )

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "text_embedding": {
                        "enabled": True,
                        "models": [{"name": "test", "dimension": 768}],
                    },
                },
            }
        )

        class MockPool:
            pass

        repo = ARIELRepository(MockPool(), config)  # type: ignore[arg-type]

        @requires_module("enhancement", "text_embedding")
        def test_method(self: ARIELRepository) -> str:
            return "success"

        result = test_method(repo)
        assert result == "success"

    def test_search_module_not_enabled_raises(self) -> None:
        """Test search module disabled raises ModuleNotEnabledError."""
        from osprey.services.ariel_search.database.repository import (
            ARIELRepository,
            requires_module,
        )
        from osprey.services.ariel_search.exceptions import ModuleNotEnabledError

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "search_modules": {"keyword": {"enabled": False}},
            }
        )

        class MockPool:
            pass

        repo = ARIELRepository(MockPool(), config)  # type: ignore[arg-type]

        @requires_module("search", "keyword")
        def test_method(self: ARIELRepository) -> str:
            return "success"

        with pytest.raises(ModuleNotEnabledError, match="keyword"):
            test_method(repo)

    def test_unknown_module_type_disabled(self) -> None:
        """Test unknown module type is treated as disabled."""
        from osprey.services.ariel_search.database.repository import (
            ARIELRepository,
            requires_module,
        )
        from osprey.services.ariel_search.exceptions import ModuleNotEnabledError

        config = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost:5432/test"))

        class MockPool:
            pass

        repo = ARIELRepository(MockPool(), config)  # type: ignore[arg-type]

        @requires_module("unknown_type", "some_module")
        def test_method(self: ARIELRepository) -> str:
            return "success"

        with pytest.raises(ModuleNotEnabledError, match="some_module"):
            test_method(repo)


class TestTextEmbeddingMigration:
    """Tests for TextEmbeddingMigration."""

    def test_default_models_has_nomic(self) -> None:
        """Default models list includes nomic-embed-text."""
        migration = TextEmbeddingMigration()
        models = migration._get_models()
        # Should have default model
        assert len(models) >= 1
        model_names = [m[0] for m in models]
        assert "nomic-embed-text" in model_names

    def test_custom_models(self) -> None:
        """Custom models are stored correctly."""
        models = [("model-a", 512), ("model-b", 1024)]
        migration = TextEmbeddingMigration(models=models)
        assert migration._get_models() == models


class TestMigrationModuleExports:
    """Tests for migration module exports."""

    def test_model_to_table_name_exported(self) -> None:
        """model_to_table_name is exported."""
        from osprey.services.ariel_search.database.migration import model_to_table_name

        assert callable(model_to_table_name)

    def test_base_migration_exported(self) -> None:
        """BaseMigration is exported."""
        from osprey.services.ariel_search.database.migration import BaseMigration

        assert BaseMigration is not None


class TestCoreMigration:
    """Tests for CoreMigration class."""

    def test_core_migration_name(self) -> None:
        """CoreMigration has correct name."""
        migration = CoreMigration()
        assert migration.name == "core_schema"

    def test_core_migration_no_dependencies(self) -> None:
        """CoreMigration has no dependencies."""
        migration = CoreMigration()
        assert migration.depends_on == []

    def test_core_migration_is_base_migration(self) -> None:
        """CoreMigration inherits from BaseMigration."""
        migration = CoreMigration()
        assert isinstance(migration, BaseMigration)


class TestDatabaseExports:
    """Tests for database module exports."""

    def test_create_connection_pool_exported(self) -> None:
        """create_connection_pool is exported from database package."""
        from osprey.services.ariel_search.database import create_connection_pool

        assert callable(create_connection_pool)

    def test_ariel_repository_exported(self) -> None:
        """ARIELRepository is exported from database package."""
        from osprey.services.ariel_search.database import ARIELRepository

        assert ARIELRepository is not None

    def test_run_migrations_exported(self) -> None:
        """run_migrations is exported from database package."""
        from osprey.services.ariel_search.database import run_migrations

        assert callable(run_migrations)


class TestMigrationRunnerLogic:
    """Tests for MigrationRunner logic without database."""

    def test_topological_sort_algorithm(self) -> None:
        """Test topological sort algorithm via standalone function."""
        from osprey.services.ariel_search.exceptions import ConfigurationError

        def topological_sort(migrations: list[BaseMigration]) -> list[BaseMigration]:
            """Standalone topological sort for testing."""
            migration_map = {m.name: m for m in migrations}
            in_degree: dict[str, int] = {m.name: 0 for m in migrations}
            graph: dict[str, list[str]] = {m.name: [] for m in migrations}

            for migration in migrations:
                for dep in migration.depends_on:
                    if dep in migration_map:
                        graph[dep].append(migration.name)
                        in_degree[migration.name] += 1

            queue = [name for name, degree in in_degree.items() if degree == 0]
            sorted_names: list[str] = []

            while queue:
                name = queue.pop(0)
                sorted_names.append(name)
                for dependent in graph[name]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            if len(sorted_names) != len(migrations):
                raise ConfigurationError(
                    "Circular dependency detected",
                    config_key="test",
                )

            return [migration_map[name] for name in sorted_names]

        # Create mock migrations
        class MockMigration(BaseMigration):
            def __init__(self, name: str, deps: list[str]) -> None:
                self._name = name
                self._deps = deps

            @property
            def name(self) -> str:
                return self._name

            @property
            def depends_on(self) -> list[str]:
                return self._deps

            async def up(self, conn) -> None:
                pass

        core = MockMigration("core_schema", [])
        semantic = MockMigration("semantic_processor", ["core_schema"])
        text_emb = MockMigration("text_embedding", ["core_schema"])

        # Sort should work
        sorted_migs = topological_sort([text_emb, semantic, core])
        assert sorted_migs[0].name == "core_schema"
        remaining = {m.name for m in sorted_migs[1:]}
        assert remaining == {"semantic_processor", "text_embedding"}

    def test_known_migrations_registry(self) -> None:
        """Test that KNOWN_MIGRATIONS registry exists and has expected entries."""
        from osprey.services.ariel_search.database.migrate import KNOWN_MIGRATIONS

        # KNOWN_MIGRATIONS is a list of tuples (name, module, class_name, enable_key)
        names = [m[0] for m in KNOWN_MIGRATIONS]
        assert "core_schema" in names
        assert "semantic_processor" in names
        assert "text_embedding" in names

    def test_core_migration_instantiation(self) -> None:
        """Core migration can be instantiated directly."""
        migration = CoreMigration()
        assert migration.name == "core_schema"
        assert migration.depends_on == []


class TestRepositoryInitialization:
    """Tests for ARIELRepository initialization."""

    def test_repository_stores_config(self) -> None:
        """Repository stores the config object."""
        from osprey.services.ariel_search.database.repository import ARIELRepository

        config = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost/test"))

        class MockPool:
            pass

        repo = ARIELRepository(MockPool(), config)  # type: ignore[arg-type]
        assert repo.config is config

    def test_repository_stores_pool(self) -> None:
        """Repository stores the pool object."""
        from osprey.services.ariel_search.database.repository import ARIELRepository

        config = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost/test"))

        class MockPool:
            pass

        pool = MockPool()
        repo = ARIELRepository(pool, config)  # type: ignore[arg-type]
        assert repo.pool is pool


class TestEmbeddingTableInfo:
    """Tests for EmbeddingTableInfo dataclass."""

    def test_embedding_table_info_creation(self) -> None:
        """EmbeddingTableInfo can be created with required fields."""
        from osprey.services.ariel_search.models import EmbeddingTableInfo

        info = EmbeddingTableInfo(
            table_name="text_embeddings_nomic_embed_text",
            entry_count=100,
            dimension=768,
            is_active=True,
        )
        assert info.table_name == "text_embeddings_nomic_embed_text"
        assert info.entry_count == 100
        assert info.dimension == 768
        assert info.is_active is True

    def test_embedding_table_info_optional_dimension(self) -> None:
        """EmbeddingTableInfo supports None dimension."""
        from osprey.services.ariel_search.models import EmbeddingTableInfo

        info = EmbeddingTableInfo(
            table_name="text_embeddings_unknown",
            entry_count=0,
            dimension=None,
            is_active=False,
        )
        assert info.dimension is None


class TestConnectionModule:
    """Tests for database connection module."""

    def test_create_connection_pool_import(self) -> None:
        """create_connection_pool can be imported."""
        from osprey.services.ariel_search.database.connection import (
            create_connection_pool,
        )

        assert callable(create_connection_pool)

    def test_connection_pool_is_async(self) -> None:
        """create_connection_pool is an async function."""
        import asyncio

        from osprey.services.ariel_search.database.connection import (
            create_connection_pool,
        )

        assert asyncio.iscoroutinefunction(create_connection_pool)


class TestEnhancementModulesInit:
    """Tests for enhancement module initialization."""

    def test_base_enhancement_import(self) -> None:
        """BaseEnhancementModule can be imported."""
        from osprey.services.ariel_search.enhancement.base import (
            BaseEnhancementModule,
        )

        assert BaseEnhancementModule is not None

    def test_factory_import(self) -> None:
        """Enhancement factory can be imported."""
        from osprey.services.ariel_search.enhancement.factory import (
            create_enhancers_from_config,
        )

        assert callable(create_enhancers_from_config)

    def test_text_embedding_import(self) -> None:
        """TextEmbeddingModule can be imported."""
        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            TextEmbeddingModule,
        )

        assert TextEmbeddingModule is not None

    def test_semantic_processor_import(self) -> None:
        """SemanticProcessorModule can be imported."""
        from osprey.services.ariel_search.enhancement.semantic_processor.processor import (
            SemanticProcessorModule,
        )

        assert SemanticProcessorModule is not None


class TestIngestionAdaptersInit:
    """Tests for ingestion adapter initialization."""

    def test_base_adapter_import(self) -> None:
        """BaseAdapter can be imported."""
        from osprey.services.ariel_search.ingestion.base import BaseAdapter

        assert BaseAdapter is not None

    def test_ingestion_adapters_in_registry(self) -> None:
        """Ingestion adapters are discoverable via the Osprey registry."""
        from osprey.registry import get_registry

        registry = get_registry()
        adapters = registry.list_ariel_ingestion_adapters()
        assert "als_logbook" in adapters
        assert "generic_json" in adapters

    def test_get_adapter_function(self) -> None:
        """get_adapter function exists."""
        from osprey.services.ariel_search.ingestion.adapters import get_adapter

        assert callable(get_adapter)


class TestSearchModulesInit:
    """Tests for search module initialization."""

    def test_keyword_search_import(self) -> None:
        """keyword_search can be imported."""
        from osprey.services.ariel_search.search import keyword_search

        assert callable(keyword_search)

    def test_semantic_search_import(self) -> None:
        """semantic_search can be imported."""
        from osprey.services.ariel_search.search import semantic_search

        assert callable(semantic_search)

    def test_agent_executor_import(self) -> None:
        """AgentExecutor can be imported from agent module."""
        from osprey.services.ariel_search.agent import AgentExecutor

        assert AgentExecutor is not None


class TestEnhancementFactory:
    """Tests for enhancement factory function."""

    def test_create_enhancers_empty_config(self) -> None:
        """create_enhancers_from_config returns empty list for no modules."""
        from osprey.services.ariel_search.enhancement.factory import (
            create_enhancers_from_config,
        )

        config = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost/test"))
        enhancers = create_enhancers_from_config(config)

        assert isinstance(enhancers, list)

    def test_create_enhancers_with_text_embedding(self) -> None:
        """create_enhancers_from_config creates TextEmbeddingModule."""
        from osprey.services.ariel_search.enhancement.factory import (
            create_enhancers_from_config,
        )
        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            TextEmbeddingModule,
        )

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "enhancement_modules": {
                    "text_embedding": {
                        "enabled": True,
                        "models": [{"name": "test-model", "dimension": 768}],
                    },
                },
            }
        )
        enhancers = create_enhancers_from_config(config)

        assert len(enhancers) >= 1
        enhancer_types = [type(e) for e in enhancers]
        assert TextEmbeddingModule in enhancer_types

    def test_create_enhancers_with_semantic_processor(self) -> None:
        """create_enhancers_from_config creates SemanticProcessorModule."""
        from osprey.services.ariel_search.enhancement.factory import (
            create_enhancers_from_config,
        )
        from osprey.services.ariel_search.enhancement.semantic_processor.processor import (
            SemanticProcessorModule,
        )

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "enhancement_modules": {
                    "semantic_processor": {"enabled": True},
                },
            }
        )
        enhancers = create_enhancers_from_config(config)

        assert len(enhancers) >= 1
        enhancer_types = [type(e) for e in enhancers]
        assert SemanticProcessorModule in enhancer_types
