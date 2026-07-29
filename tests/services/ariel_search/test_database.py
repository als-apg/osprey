"""Tests for ARIEL database layer.

Note: These tests run without psycopg installed by testing only the
migration logic and configuration parts that don't require database access.
"""

import logging

import pytest

from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig
from osprey.services.ariel_search.database import migrations as migrations_module
from osprey.services.ariel_search.database.attachment_migration import AttachmentMigration
from osprey.services.ariel_search.database.core_migration import CoreMigration
from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    MigrationRunner,
    MigrationSkippedError,
    model_to_table_name,
    run_migrations,
)
from osprey.services.ariel_search.enhancement.semantic_processor.migration import (
    SemanticProcessorMigration,
)
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    TextEmbeddingMigration,
)


class StubMigration(BaseMigration):
    """Minimal migration used to exercise dependency ordering."""

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


class BareMigration(BaseMigration):
    """Migration overriding only the two abstract members.

    Exists to exercise the base-class defaults that every other migration in
    the tree replaces: the empty ``depends_on`` and the ``down()`` that refuses.
    """

    @property
    def name(self) -> str:
        return "bare"

    async def up(self, conn) -> None:
        pass


class RecordingMigration(StubMigration):
    """Stub that records how the runner drove it and can fail on demand.

    ``is_applied``/``mark_applied``/``mark_unapplied`` are answered in memory so
    a runner test controls the applied state without scripting the connection,
    and ``events`` shows exactly which of them the runner reached. ``up_error``
    and ``down_error`` are raised from ``up()``/``down()`` to reach the runner's
    skip, failure and rollback-failure branches.
    """

    def __init__(
        self,
        name: str,
        deps: list[str] | None = None,
        *,
        applied: bool = False,
        up_error: Exception | None = None,
        down_error: Exception | None = None,
    ) -> None:
        super().__init__(name, deps or [])
        self.applied = applied
        self.up_error = up_error
        self.down_error = down_error
        self.events: list[str] = []

    async def is_applied(self, conn) -> bool:
        self.events.append("is_applied")
        return self.applied

    async def up(self, conn) -> None:
        self.events.append("up")
        if self.up_error is not None:
            raise self.up_error

    async def down(self, conn) -> None:
        self.events.append("down")
        if self.down_error is not None:
            raise self.down_error

    async def mark_applied(self, conn) -> None:
        self.events.append("mark_applied")

    async def mark_unapplied(self, conn) -> None:
        self.events.append("mark_unapplied")


def make_runner(pool=None, config: ARIELConfig | None = None) -> MigrationRunner:
    """Build a MigrationRunner that needs no real database.

    Args:
        pool: Fake pool for the tests that drive ``run``/``rollback``/``status``.
            The default ``None`` is enough for the sort- and config-only paths,
            which never open a connection.
        config: Configuration to run against; defaults to a database-only config
            with every enhancement module off.
    """
    if config is None:
        config = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost:5432/test"))
    return MigrationRunner(pool=pool, config=config)  # type: ignore[arg-type]


def sql_index(conn, needle: str) -> int:
    """Position of the first statement executed on `conn` containing `needle`.

    Both sides are whitespace-normalized so a single-line needle matches the
    multi-line DDL literals the migrations emit.
    """
    wanted = " ".join(needle.split())
    for position, statement in enumerate(conn.sql):
        if wanted in " ".join(statement.split()):
            return position
    raise AssertionError(f"no statement containing {needle!r} in {conn.sql}")


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
        core = StubMigration("core_schema", [])
        semantic = StubMigration("semantic_processor", ["core_schema"])
        text_emb = StubMigration("text_embedding", ["core_schema"])

        sorted_migrations = make_runner()._topological_sort([semantic, text_emb, core])

        # Core should be first
        assert sorted_migrations[0].name == "core_schema"
        # Others can be in any order after core
        remaining_names = {m.name for m in sorted_migrations[1:]}
        assert remaining_names == {"semantic_processor", "text_embedding"}

    def test_topological_sort_circular_dependency(self) -> None:
        """Test that circular dependency is detected."""
        from osprey.services.ariel_search.exceptions import ConfigurationError

        a = StubMigration("a", ["b"])
        b = StubMigration("b", ["a"])

        with pytest.raises(ConfigurationError, match="Circular dependency") as exc_info:
            make_runner()._topological_sort([a, b])

        assert exc_info.value.config_key == "ariel.migrations"

    def test_topological_sort_self_dependency_is_circular(self) -> None:
        """A migration depending on itself is rejected as circular."""
        from osprey.services.ariel_search.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Circular dependency"):
            make_runner()._topological_sort([StubMigration("a", ["a"])])

    def test_topological_sort_ignores_unknown_dependency(self) -> None:
        """Dependencies outside the given list do not block sorting."""
        migration = StubMigration("text_embedding", ["not_enabled"])

        sorted_migrations = make_runner()._topological_sort([migration])

        assert [m.name for m in sorted_migrations] == ["text_embedding"]

    def test_topological_sort_empty_list(self) -> None:
        """Sorting no migrations yields no migrations."""
        assert make_runner()._topological_sort([]) == []


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
        from osprey.services.ariel_search.database.migrations import model_to_table_name

        assert callable(model_to_table_name)

    def test_base_migration_exported(self) -> None:
        """BaseMigration is exported."""
        from osprey.services.ariel_search.database.migrations import BaseMigration

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
        """Test topological sort orders a transitive dependency chain."""
        core = StubMigration("core_schema", [])
        semantic = StubMigration("semantic_processor", ["core_schema"])
        text_emb = StubMigration("text_embedding", ["semantic_processor"])

        sorted_migs = make_runner()._topological_sort([text_emb, semantic, core])

        assert [m.name for m in sorted_migs] == [
            "core_schema",
            "semantic_processor",
            "text_embedding",
        ]

    def test_known_migrations_registry(self) -> None:
        """Test that KNOWN_MIGRATIONS registry exists and has expected entries."""
        from osprey.services.ariel_search.database.migrations import KNOWN_MIGRATIONS

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

    def test_get_enabled_migrations_passes_configured_models(self) -> None:
        """`osprey ariel migrate` must build the embedding migration with the
        configured (model, dimension) pairs, so a table is created per configured
        model rather than only the hardcoded default.

        Regression: MigrationRunner instantiated TextEmbeddingMigration() with no
        args, so configured models were dropped and only nomic-embed-text/768 was
        ever created.
        """
        from osprey.services.ariel_search.database.migrations import MigrationRunner

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "text_embedding": {
                        "enabled": True,
                        "models": [{"name": "mxbai-embed-large", "dimension": 1024}],
                    },
                },
            }
        )

        runner = MigrationRunner(pool=None, config=config)  # type: ignore[arg-type]
        migrations = runner._get_enabled_migrations()

        text_emb = next(m for m in migrations if m.name == "text_embedding")
        assert text_emb._get_models() == [("mxbai-embed-large", 1024)]


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


class TestBaseMigrationTracking:
    """Tests for the ariel_migrations bookkeeping on BaseMigration."""

    async def test_is_applied_false_before_tracking_table_exists(self, ddl_conn) -> None:
        """A fresh database has no ariel_migrations table, so nothing is applied.

        The bootstrap check must short-circuit: querying ariel_migrations before
        it exists would error out on the very first migration of a new install.
        """
        ddl_conn.recorder.rows_for = {"FROM information_schema.tables": [(False,)]}

        assert await StubMigration("core_schema", []).is_applied(ddl_conn) is False
        assert len(ddl_conn.sql) == 1

    async def test_is_applied_false_when_bootstrap_query_returns_no_row(self, ddl_conn) -> None:
        """A missing row from the existence probe is treated as 'not applied'."""
        assert await StubMigration("core_schema", []).is_applied(ddl_conn) is False
        assert len(ddl_conn.sql) == 1

    async def test_is_applied_false_when_migration_row_absent(self, ddl_conn) -> None:
        """With the table present but no row, the second query decides."""
        ddl_conn.recorder.rows_for = {
            "FROM information_schema.tables": [(True,)],
            "SELECT EXISTS (SELECT 1 FROM ariel_migrations": [(False,)],
        }

        assert await StubMigration("text_embedding", []).is_applied(ddl_conn) is False
        assert len(ddl_conn.sql) == 2
        assert ddl_conn.calls[1][1] == ["text_embedding"]

    async def test_is_applied_true_when_row_present(self, ddl_conn) -> None:
        """A row in ariel_migrations means the migration already ran."""
        ddl_conn.recorder.rows_for = {
            "FROM information_schema.tables": [(True,)],
            "SELECT EXISTS (SELECT 1 FROM ariel_migrations": [(True,)],
        }

        assert await StubMigration("text_embedding", []).is_applied(ddl_conn) is True

    async def test_mark_applied_is_idempotent(self, ddl_conn) -> None:
        """Marking applied twice must not raise, so re-running migrations is safe."""
        await StubMigration("core_schema", []).mark_applied(ddl_conn)

        statement = " ".join(ddl_conn.sql[0].split())
        assert "INSERT INTO ariel_migrations (name, applied_at)" in statement
        assert "ON CONFLICT (name) DO NOTHING" in statement
        assert ddl_conn.calls[0][1] == ["core_schema"]

    async def test_mark_unapplied_deletes_only_this_migration(self, ddl_conn) -> None:
        """Rollback bookkeeping is scoped by name, not a table-wide delete."""
        await StubMigration("text_embedding", []).mark_unapplied(ddl_conn)

        assert " ".join(ddl_conn.sql[0].split()) == ("DELETE FROM ariel_migrations WHERE name = %s")
        assert ddl_conn.calls[0][1] == ["text_embedding"]

    def test_depends_on_defaults_to_empty(self) -> None:
        """A migration that declares no dependencies sorts first."""
        assert BareMigration().depends_on == []

    async def test_down_defaults_to_not_implemented(self, ddl_conn) -> None:
        """Rollback is opt-in; the default refuses and names the migration."""
        with pytest.raises(NotImplementedError, match="bare"):
            await BareMigration().down(ddl_conn)

        assert ddl_conn.sql == []


class TestGetEnabledMigrations:
    """Tests for MigrationRunner._get_enabled_migrations."""

    def test_disabled_module_migration_is_not_loaded(self) -> None:
        """Only the always-run migrations load when no module is enabled."""
        names = [m.name for m in make_runner()._get_enabled_migrations()]

        assert names == ["core_schema", "attachment_files"]

    def test_unimportable_migration_is_skipped_with_warning(self, monkeypatch, caplog) -> None:
        """A migration whose module is gone must not break the whole run."""
        caplog.set_level(logging.WARNING, logger="ariel")
        monkeypatch.setattr(
            migrations_module,
            "KNOWN_MIGRATIONS",
            [("ghost", "osprey.services.ariel_search.database.no_such_module", "Ghost", None)],
        )

        assert make_runner()._get_enabled_migrations() == []
        assert "Failed to load migration ghost" in caplog.text

    def test_missing_migration_class_is_skipped_with_warning(self, monkeypatch, caplog) -> None:
        """A renamed class is reported the same way as a missing module."""
        caplog.set_level(logging.WARNING, logger="ariel")
        monkeypatch.setattr(
            migrations_module,
            "KNOWN_MIGRATIONS",
            [
                (
                    "core_schema",
                    "osprey.services.ariel_search.database.core_migration",
                    "RenamedMigration",
                    None,
                )
            ],
        )

        assert make_runner()._get_enabled_migrations() == []
        assert "Failed to load migration core_schema" in caplog.text

    def test_embedding_migration_falls_back_to_default_models(self) -> None:
        """text_embedding enabled without a models list keeps the built-in default."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {"text_embedding": {"enabled": True}},
            }
        )
        runner = make_runner(config=config)

        assert runner._configured_embedding_models() is None

        text_embedding = next(
            m for m in runner._get_enabled_migrations() if m.name == "text_embedding"
        )
        assert text_embedding._get_models() == [("nomic-embed-text", 768)]


class TestMigrationRunnerRun:
    """Tests for MigrationRunner.run."""

    async def test_applies_pending_migrations_in_dependency_order(self, fake_pool) -> None:
        """Dependencies run before dependents regardless of registry order."""
        core = RecordingMigration("core_schema")
        embedding = RecordingMigration("text_embedding", ["core_schema"])
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [embedding, core]  # type: ignore[method-assign]

        assert await runner.run() == ["core_schema", "text_embedding"]
        assert core.events == ["is_applied", "up", "mark_applied"]
        assert embedding.events == ["is_applied", "up", "mark_applied"]

    async def test_already_applied_migration_is_left_alone(self, fake_pool) -> None:
        """An applied migration is neither re-run nor re-reported."""
        migration = RecordingMigration("core_schema", applied=True)
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [migration]  # type: ignore[method-assign]

        assert await runner.run() == []
        assert migration.events == ["is_applied"]

    async def test_dry_run_reports_without_touching_the_schema(self, fake_pool, caplog) -> None:
        """Dry run names what would be applied but emits no DDL."""
        caplog.set_level(logging.INFO, logger="ariel")
        migration = RecordingMigration("core_schema")
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [migration]  # type: ignore[method-assign]

        assert await runner.run(dry_run=True) == ["core_schema"]
        assert migration.events == ["is_applied"]
        assert "Would apply migration: core_schema" in caplog.text

    async def test_skipped_migration_is_not_marked_and_run_continues(
        self, fake_pool, caplog
    ) -> None:
        """A missing prerequisite (e.g. pgvector) downgrades to a warning.

        The migration must stay unmarked so it retries once the prerequisite is
        installed, and later migrations must still get their turn.
        """
        caplog.set_level(logging.WARNING, logger="ariel")
        skipped = RecordingMigration(
            "text_embedding", up_error=MigrationSkippedError("pgvector is not available")
        )
        follower = RecordingMigration("attachment_files", ["text_embedding"])
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [skipped, follower]  # type: ignore[method-assign]

        assert await runner.run() == ["attachment_files"]
        assert skipped.events == ["is_applied", "up"]
        assert "Migration text_embedding skipped: pgvector is not available" in caplog.text

    async def test_unexpected_failure_aborts_the_run(self, fake_pool, caplog) -> None:
        """Any non-skip error propagates and stops later migrations."""
        caplog.set_level(logging.ERROR, logger="ariel")
        failing = RecordingMigration("core_schema", up_error=RuntimeError("boom"))
        follower = RecordingMigration("attachment_files", ["core_schema"])
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [failing, follower]  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="boom"):
            await runner.run()

        assert failing.events == ["is_applied", "up"]
        assert follower.events == []
        assert "Failed to apply migration core_schema: boom" in caplog.text

    async def test_run_migrations_helper_drives_a_runner(self, fake_pool, monkeypatch) -> None:
        """The module-level convenience function is a thin wrapper over run()."""
        migration = RecordingMigration("core_schema")
        monkeypatch.setattr(MigrationRunner, "_get_enabled_migrations", lambda self: [migration])
        config = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost:5432/test"))

        assert await run_migrations(fake_pool, config) == ["core_schema"]
        assert migration.events == ["is_applied", "up", "mark_applied"]


class TestMigrationRunnerRollback:
    """Tests for MigrationRunner.rollback."""

    async def test_unknown_migration_reports_failure(self, fake_pool, caplog) -> None:
        """Rolling back a name that is not enabled fails rather than no-ops silently."""
        caplog.set_level(logging.ERROR, logger="ariel")
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [RecordingMigration("core_schema")]  # type: ignore[method-assign]

        assert await runner.rollback("text_embedding") is False
        assert "Migration not found: text_embedding" in caplog.text

    async def test_unapplied_migration_rolls_back_trivially(self, fake_pool, caplog) -> None:
        """Rollback of a migration that never ran succeeds without calling down()."""
        caplog.set_level(logging.INFO, logger="ariel")
        migration = RecordingMigration("core_schema", applied=False)
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [migration]  # type: ignore[method-assign]

        assert await runner.rollback("core_schema") is True
        assert migration.events == ["is_applied"]
        assert "Migration core_schema is not applied" in caplog.text

    async def test_applied_migration_is_reverted_and_unmarked(self, fake_pool) -> None:
        """A successful rollback both reverts the schema and clears the tracking row."""
        migration = RecordingMigration("core_schema", applied=True)
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [migration]  # type: ignore[method-assign]

        assert await runner.rollback("core_schema") is True
        assert migration.events == ["is_applied", "down", "mark_unapplied"]

    async def test_migration_without_rollback_support_reports_failure(
        self, fake_pool, caplog
    ) -> None:
        """A migration that never implemented down() stays marked as applied."""
        caplog.set_level(logging.ERROR, logger="ariel")
        fake_pool.recorder.rows_for = {
            "FROM information_schema.tables": [(True,)],
            "SELECT EXISTS (SELECT 1 FROM ariel_migrations": [(True,)],
        }
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [StubMigration("text_embedding", [])]  # type: ignore[method-assign]

        assert await runner.rollback("text_embedding") is False
        assert "Rollback not implemented for migration: text_embedding" in caplog.text
        assert fake_pool.matching("DELETE FROM ariel_migrations") == []

    async def test_unexpected_rollback_failure_propagates(self, fake_pool, caplog) -> None:
        """A failing down() surfaces rather than being reported as a clean rollback."""
        caplog.set_level(logging.ERROR, logger="ariel")
        migration = RecordingMigration(
            "core_schema", applied=True, down_error=RuntimeError("drop failed")
        )
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [migration]  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="drop failed"):
            await runner.rollback("core_schema")

        assert migration.events == ["is_applied", "down"]
        assert "Failed to rollback migration core_schema: drop failed" in caplog.text


class TestMigrationRunnerStatus:
    """Tests for MigrationRunner.status."""

    async def test_status_reports_applied_state_and_dependencies(self, fake_pool) -> None:
        """Status covers every enabled migration, applied or not."""
        runner = make_runner(pool=fake_pool)
        runner._get_enabled_migrations = lambda: [  # type: ignore[method-assign]
            RecordingMigration("core_schema", applied=True),
            RecordingMigration("text_embedding", ["core_schema", "semantic_processor"]),
        ]

        assert await runner.status() == {
            "core_schema": {"applied": True, "depends_on": "(none)"},
            "text_embedding": {
                "applied": False,
                "depends_on": "core_schema, semantic_processor",
            },
        }


class TestCoreMigrationDDL:
    """Tests for the SQL CoreMigration emits."""

    async def test_up_creates_objects_in_dependency_order(self, ddl_conn) -> None:
        """Each object is created after whatever it is built on."""
        await CoreMigration().up(ddl_conn)

        # pg_trgm must exist before the GIN trigram index that uses its operators.
        assert sql_index(ddl_conn, "CREATE EXTENSION IF NOT EXISTS pg_trgm") < sql_index(
            ddl_conn, "idx_entries_raw_text_trgm"
        )
        # The table precedes every index and the trigger placed on it.
        entries_table = sql_index(ddl_conn, "CREATE TABLE IF NOT EXISTS enhanced_entries")
        assert entries_table < sql_index(ddl_conn, "idx_entries_timestamp")
        assert entries_table < sql_index(ddl_conn, "CREATE TRIGGER")
        # The trigger function is defined before the trigger references it.
        assert sql_index(ddl_conn, "CREATE OR REPLACE FUNCTION update_updated_at_column") < (
            sql_index(ddl_conn, "CREATE TRIGGER update_enhanced_entries_updated_at")
        )
        assert sql_index(ddl_conn, "CREATE TABLE IF NOT EXISTS ingestion_runs") < sql_index(
            ddl_conn, "idx_ingestion_runs_started"
        )
        # The tracking table the runner writes to is part of the core schema.
        assert sql_index(ddl_conn, "CREATE TABLE IF NOT EXISTS ariel_migrations") >= 0

    async def test_up_is_rerunnable(self, ddl_conn) -> None:
        """Every created object is guarded, so a second migrate run is a no-op."""
        await CoreMigration().up(ddl_conn)

        for statement in ddl_conn.sql:
            normalized = " ".join(statement.split())
            if normalized.startswith(("CREATE TABLE", "CREATE INDEX", "CREATE EXTENSION")):
                assert "IF NOT EXISTS" in normalized, normalized
        # The two objects that cannot take IF NOT EXISTS carry their own guards.
        assert "CREATE OR REPLACE FUNCTION" in " ".join(
            ddl_conn.sql[sql_index(ddl_conn, "update_updated_at_column()")].split()
        )
        trigger_ddl = " ".join(ddl_conn.sql[sql_index(ddl_conn, "CREATE TRIGGER")].split())
        assert "IF NOT EXISTS ( SELECT 1 FROM pg_trigger" in trigger_ddl

    async def test_down_drops_dependents_before_their_dependencies(self, ddl_conn) -> None:
        """Rollback unwinds the core schema in reverse creation order."""
        await CoreMigration().down(ddl_conn)

        assert [" ".join(s.split()) for s in ddl_conn.sql] == [
            "DROP TABLE IF EXISTS ariel_migrations CASCADE",
            "DROP TABLE IF EXISTS ingestion_runs CASCADE",
            "DROP TABLE IF EXISTS enhanced_entries CASCADE",
            "DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE",
        ]


class TestAttachmentMigrationDDL:
    """Tests for AttachmentMigration."""

    def test_properties(self) -> None:
        """Attachments hang off enhanced_entries, so core_schema comes first."""
        migration = AttachmentMigration()
        assert migration.name == "attachment_files"
        assert migration.depends_on == ["core_schema"]

    async def test_up_creates_table_then_lookup_index(self, ddl_conn) -> None:
        """The table stores file bytes and cascades from its entry."""
        await AttachmentMigration().up(ddl_conn)

        table_ddl = " ".join(ddl_conn.sql[0].split())
        assert "CREATE TABLE IF NOT EXISTS attachment_files" in table_ddl
        assert "data BYTEA NOT NULL" in table_ddl
        assert "REFERENCES enhanced_entries(entry_id) ON DELETE CASCADE" in table_ddl

        index_ddl = " ".join(ddl_conn.sql[1].split())
        assert "CREATE INDEX IF NOT EXISTS idx_attachment_files_entry_id" in index_ddl
        assert "ON attachment_files(entry_id)" in index_ddl

    async def test_down_drops_the_table(self, ddl_conn) -> None:
        """Rollback removes the table; the index goes with it."""
        await AttachmentMigration().down(ddl_conn)

        assert ddl_conn.sql == ["DROP TABLE IF EXISTS attachment_files CASCADE"]


class TestSemanticProcessorMigrationDDL:
    """Tests for SemanticProcessorMigration."""

    async def test_up_adds_columns_before_indexing_them(self, ddl_conn) -> None:
        """Columns are added first, then the indexes that read them."""
        await SemanticProcessorMigration().up(ddl_conn)

        statements = [" ".join(s.split()) for s in ddl_conn.sql]
        assert statements[0] == "ALTER TABLE enhanced_entries ADD COLUMN IF NOT EXISTS summary TEXT"
        assert statements[1] == (
            "ALTER TABLE enhanced_entries ADD COLUMN IF NOT EXISTS keywords TEXT[] DEFAULT '{}'"
        )
        assert "CREATE INDEX IF NOT EXISTS idx_entries_keywords" in statements[2]
        assert "USING GIN(keywords)" in statements[2]
        assert "CREATE INDEX IF NOT EXISTS idx_entries_text_search" in statements[3]
        assert "to_tsvector('english', raw_text || ' ' || COALESCE(summary, ''))" in statements[3]

    async def test_down_drops_indexes_before_columns(self, ddl_conn) -> None:
        """Dropping the indexed columns first would leave the drops to CASCADE."""
        await SemanticProcessorMigration().down(ddl_conn)

        assert [" ".join(s.split()) for s in ddl_conn.sql] == [
            "DROP INDEX IF EXISTS idx_entries_text_search",
            "DROP INDEX IF EXISTS idx_entries_keywords",
            "ALTER TABLE enhanced_entries DROP COLUMN IF EXISTS keywords",
            "ALTER TABLE enhanced_entries DROP COLUMN IF EXISTS summary",
        ]


class TestTextEmbeddingMigrationDDL:
    """Tests for TextEmbeddingMigration."""

    async def test_up_skips_when_pgvector_is_unavailable(self, ddl_conn) -> None:
        """Without pgvector the migration skips instead of failing the whole run.

        Skipping (rather than raising) is what keeps ARIEL usable in
        keyword-only mode on a stock PostgreSQL.
        """
        with pytest.raises(MigrationSkippedError, match="pgvector"):
            await TextEmbeddingMigration().up(ddl_conn)

        assert len(ddl_conn.sql) == 1
        assert "pg_available_extensions" in ddl_conn.sql[0]

    async def test_up_creates_a_table_and_index_per_model(self, ddl_conn) -> None:
        """Each configured model gets its own dimensioned table and vector index."""
        ddl_conn.recorder.rows_for = {"pg_available_extensions": [(True,)]}

        await TextEmbeddingMigration(models=[("model-a", 512), ("model-b", 1024)]).up(ddl_conn)

        extension = sql_index(ddl_conn, "CREATE EXTENSION IF NOT EXISTS vector")
        table_a = sql_index(ddl_conn, "CREATE TABLE IF NOT EXISTS text_embeddings_model_a")
        assert extension < table_a

        assert "embedding vector(512)" in " ".join(ddl_conn.sql[table_a].split())
        assert "REFERENCES enhanced_entries(entry_id) ON DELETE CASCADE" in " ".join(
            ddl_conn.sql[table_a].split()
        )

        index_a = sql_index(
            ddl_conn, "CREATE INDEX IF NOT EXISTS idx_text_embeddings_model_a_vector"
        )
        assert table_a < index_a
        assert "USING ivfflat (embedding vector_cosine_ops)" in " ".join(
            ddl_conn.sql[index_a].split()
        )

        table_b = sql_index(ddl_conn, "CREATE TABLE IF NOT EXISTS text_embeddings_model_b")
        assert "embedding vector(1024)" in " ".join(ddl_conn.sql[table_b].split())

    async def test_up_uses_the_default_model_when_none_configured(self, ddl_conn) -> None:
        """An unconfigured migration still creates the default nomic table."""
        ddl_conn.recorder.rows_for = {"pg_available_extensions": [(True,)]}

        await TextEmbeddingMigration().up(ddl_conn)

        table = sql_index(ddl_conn, "CREATE TABLE IF NOT EXISTS text_embeddings_nomic_embed_text")
        assert "embedding vector(768)" in " ".join(ddl_conn.sql[table].split())

    async def test_down_drops_one_table_per_model(self, ddl_conn) -> None:
        """Rollback drops the per-model tables but leaves the shared extension."""
        await TextEmbeddingMigration(models=[("model-a", 512), ("model-b", 1024)]).down(ddl_conn)

        assert ddl_conn.sql == [
            "DROP TABLE IF EXISTS text_embeddings_model_a CASCADE",
            "DROP TABLE IF EXISTS text_embeddings_model_b CASCADE",
        ]
        assert ddl_conn.recorder.matching("DROP EXTENSION") == []
