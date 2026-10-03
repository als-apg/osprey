"""Unit tests for the migration runner's lock and busy-skip contract.

Driven against the fake pool: the advisory lock goes through an injected
``lock_factory``, so nothing here opens a database connection. The same
behaviour against real PostgreSQL lives in ``integration/test_advisory_lock.py``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from contextlib import asynccontextmanager

import pytest
from psycopg import errors

from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig
from osprey.services.ariel_search.database import migrations as migrations_module
from osprey.services.ariel_search.database.connection import try_advisory_lock
from osprey.services.ariel_search.database.migrations import (
    MIGRATION_LOCK_KEY,
    ImageEmbeddingTarget,
    MigrationBusyError,
    MigrationResult,
    MigrationRunner,
    MigrationSkippedError,
    acquire_nonqueueing_lock,
    image_embedding_target,
    image_index_name,
    image_table_name,
    model_to_table_name,
    run_migrations_detailed,
)
from osprey.services.ariel_search.exceptions import ModuleConfigError
from tests.services.ariel_search.test_database import RecordingMigration, no_lock

CONFIG = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost:5432/test"))


class RecordingLock:
    """Lock factory recording how it was opened, granting or refusing on demand."""

    def __init__(self, events: list[str], *, granted: bool = True) -> None:
        self.events = events
        self.granted = granted
        self.calls: list[tuple[object, str, bool]] = []

    @asynccontextmanager
    async def __call__(self, conninfo, key, *, wait: bool = False):
        self.calls.append((conninfo, key, wait))
        self.events.append("lock")
        try:
            yield self.granted
        finally:
            self.events.append("unlock")


class EventMigration(RecordingMigration):
    """RecordingMigration that also logs into a list shared with the lock."""

    def __init__(self, name: str, shared: list[str], **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.shared = shared

    async def is_applied(self, conn) -> bool:
        self.shared.append(f"is_applied:{self.name}")
        return await super().is_applied(conn)


def runner_with(pool, migrations, lock_factory=no_lock) -> MigrationRunner:
    runner = MigrationRunner(pool, CONFIG, lock_factory=lock_factory)  # type: ignore[arg-type]
    runner._get_enabled_migrations = lambda: list(migrations)  # type: ignore[method-assign]
    return runner


class TestBusySkip:
    async def test_busy_migration_holds_back_its_dependent_until_the_next_run(
        self, fake_pool, caplog
    ) -> None:
        """A busy skip leaves the dependent's up() uncalled; the next run applies both."""
        caplog.set_level(logging.WARNING, logger="ariel")
        busy = RecordingMigration("attachment_text_columns", up_error=MigrationBusyError("busy"))
        dependent = RecordingMigration("attachment_files_copy_state", ["attachment_text_columns"])
        runner = runner_with(fake_pool, [busy, dependent])

        assert await runner.run() == ([], ["attachment_text_columns"])
        assert "up" not in dependent.events
        assert "attachment_files_copy_state waits for attachment_text_columns" in caplog.text

        busy.up_error = None
        assert await runner.run() == (
            ["attachment_text_columns", "attachment_files_copy_state"],
            [],
        )
        assert dependent.events[-2:] == ["up", "mark_applied"]

    async def test_waiting_propagates_down_a_chain(self, fake_pool, caplog) -> None:
        """A dependent of a waiting migration waits too, naming its own blocker."""
        caplog.set_level(logging.WARNING, logger="ariel")
        busy = RecordingMigration("a", up_error=MigrationBusyError("busy"))
        middle = RecordingMigration("b", ["a"])
        last = RecordingMigration("c", ["b"])
        runner = runner_with(fake_pool, [busy, middle, last])

        assert await runner.run() == ([], ["a"])
        assert "up" not in last.events
        assert "b waits for a" in caplog.text
        assert "c waits for b" in caplog.text

    async def test_unrelated_migration_still_runs_after_a_busy_skip(self, fake_pool) -> None:
        """Only dependents wait; an independent migration gets its turn."""
        busy = RecordingMigration("a", up_error=MigrationBusyError("busy"))
        other = RecordingMigration("z")
        runner = runner_with(fake_pool, [busy, other])

        assert await runner.run() == (["z"], ["a"])

    async def test_applied_dependent_of_a_skip_is_left_alone_without_a_warning(
        self, fake_pool, caplog
    ) -> None:
        """An already-applied dependent has nothing to wait for."""
        caplog.set_level(logging.WARNING, logger="ariel")
        skipped = RecordingMigration("a", up_error=MigrationSkippedError("no pgvector"))
        done = RecordingMigration("b", ["a"], applied=True)
        runner = runner_with(fake_pool, [skipped, done])

        assert await runner.run() == ([], [])
        assert "waits for" not in caplog.text

    async def test_busy_skip_rolls_its_transaction_back(self, fake_pool) -> None:
        """A busy skip unwinds like any other skip."""
        busy = RecordingMigration("a", up_error=MigrationBusyError("busy"))
        runner = runner_with(fake_pool, [busy])

        await runner.run()

        assert fake_pool.conn.transactions == ["BEGIN", "ROLLBACK"]

    def test_busy_error_is_a_skip(self) -> None:
        assert issubclass(MigrationBusyError, MigrationSkippedError)


class TestMigrationLock:
    async def test_lock_is_taken_before_the_first_is_applied_and_released_after(
        self, fake_pool
    ) -> None:
        events: list[str] = []
        lock = RecordingLock(events)
        migration = EventMigration("core_schema", events)
        fake_pool.conninfo = "postgresql://db/ariel"
        runner = runner_with(fake_pool, [migration], lock)

        assert await runner.run() == (["core_schema"], [])

        assert events == ["lock", "is_applied:core_schema", "unlock"]
        assert lock.calls == [("postgresql://db/ariel", MIGRATION_LOCK_KEY, True)]

    async def test_try_mode_does_not_wait_and_reports_the_lock_held(self, fake_pool) -> None:
        """'try' with the lock held elsewhere returns at once and touches nothing."""
        events: list[str] = []
        lock = RecordingLock(events, granted=False)
        migration = EventMigration("core_schema", events)
        runner = runner_with(fake_pool, [migration], lock)

        assert await runner.run(lock="try") == ([], [])

        assert runner.lock_held is True
        assert migration.events == []
        assert lock.calls[0][2] is False
        assert fake_pool.conn.transactions == []

    async def test_try_mode_with_the_lock_free_runs_normally(self, fake_pool) -> None:
        lock = RecordingLock([])
        runner = runner_with(fake_pool, [RecordingMigration("core_schema")], lock)

        assert await runner.run(lock="try") == (["core_schema"], [])
        assert runner.lock_held is False

    async def test_lock_is_released_when_a_migration_fails(self, fake_pool) -> None:
        events: list[str] = []
        lock = RecordingLock(events)
        failing = RecordingMigration("core_schema", up_error=RuntimeError("boom"))
        runner = runner_with(fake_pool, [failing], lock)

        with pytest.raises(RuntimeError, match="boom"):
            await runner.run()

        assert events == ["lock", "unlock"]

    async def test_unknown_lock_mode_is_refused(self, fake_pool) -> None:
        runner = runner_with(fake_pool, [])

        with pytest.raises(ValueError, match="'wait' or 'try'"):
            await runner.run(lock="never")  # type: ignore[arg-type]

    async def test_rollback_waits_for_the_lock(self, fake_pool) -> None:
        events: list[str] = []
        lock = RecordingLock(events)
        migration = EventMigration("core_schema", events, applied=True)
        runner = runner_with(fake_pool, [migration], lock)

        assert await runner.rollback("core_schema") is True

        assert events == ["lock", "is_applied:core_schema", "unlock"]
        assert lock.calls[0][2] is True

    def test_default_lock_factory_is_the_advisory_lock(self) -> None:
        assert MigrationRunner(None, CONFIG).lock_factory is try_advisory_lock  # type: ignore[arg-type]


class TestRunMigrationsDetailed:
    async def test_reports_applied_busy_and_lock_state(self, fake_pool, monkeypatch) -> None:
        busy = RecordingMigration("b", up_error=MigrationBusyError("busy"))
        monkeypatch.setattr(
            MigrationRunner,
            "_get_enabled_migrations",
            lambda self: [RecordingMigration("a"), busy],
        )

        result = await run_migrations_detailed(fake_pool, CONFIG, lock_factory=no_lock)

        assert result == MigrationResult(["a"], ["b"], False)

    async def test_try_with_the_lock_held_reports_lock_held(self, fake_pool, monkeypatch) -> None:
        monkeypatch.setattr(
            MigrationRunner, "_get_enabled_migrations", lambda self: [RecordingMigration("a")]
        )

        result = await run_migrations_detailed(
            fake_pool, CONFIG, lock="try", lock_factory=RecordingLock([], granted=False)
        )

        assert result == MigrationResult([], [], True)


class FakeLockConn:
    """Connection double for acquire_nonqueueing_lock: refuses the lock N times."""

    def __init__(self, refusals: int) -> None:
        self.refusals = refusals
        self.statements: list[object] = []
        self.transactions: list[str] = []

    @asynccontextmanager
    async def transaction(self):
        self.transactions.append("SAVEPOINT")
        try:
            yield
        except BaseException:
            self.transactions.append("ROLLBACK TO SAVEPOINT")
            raise
        self.transactions.append("RELEASE SAVEPOINT")

    async def execute(self, statement) -> None:
        self.statements.append(statement)
        if self.refusals > 0:
            self.refusals -= 1
            raise errors.LockNotAvailable("could not obtain lock on relation")


class TestAcquireNonqueueingLock:
    @pytest.fixture(autouse=True)
    def _no_delay(self, monkeypatch):
        sleeps: list[float] = []

        async def _sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(migrations_module.asyncio, "sleep", _sleep)
        return sleeps

    async def test_granted_at_once(self) -> None:
        conn = FakeLockConn(refusals=0)

        await acquire_nonqueueing_lock(conn, "enhanced_entries", "SHARE")  # type: ignore[arg-type]

        assert len(conn.statements) == 1
        assert conn.transactions == ["SAVEPOINT", "RELEASE SAVEPOINT"]
        rendered = conn.statements[0].as_string(None)
        assert rendered == 'LOCK TABLE "enhanced_entries" IN SHARE MODE NOWAIT'

    async def test_retries_until_granted(self, _no_delay) -> None:
        conn = FakeLockConn(refusals=3)

        await acquire_nonqueueing_lock(conn, "enhanced_entries", "ACCESS EXCLUSIVE")  # type: ignore[arg-type]

        assert len(conn.statements) == 4
        assert conn.transactions.count("ROLLBACK TO SAVEPOINT") == 3
        assert _no_delay == [0.5, 0.5, 0.5]

    async def test_gives_up_busy_after_ten_attempts(self, _no_delay) -> None:
        conn = FakeLockConn(refusals=100)

        with pytest.raises(MigrationBusyError, match="busy, retry"):
            await acquire_nonqueueing_lock(conn, "enhanced_entries", "SHARE")  # type: ignore[arg-type]

        assert len(conn.statements) == 10
        assert len(_no_delay) == 9

    async def test_unknown_mode_is_refused_before_any_statement(self) -> None:
        conn = FakeLockConn(refusals=0)

        with pytest.raises(ValueError, match="lock mode"):
            await acquire_nonqueueing_lock(conn, "t", "SHARE; DROP TABLE t")  # type: ignore[arg-type]

        assert conn.statements == []


class TestTryAdvisoryLockArguments:
    async def test_empty_conninfo_is_refused(self) -> None:
        """No conninfo would let libpq lock in whatever database its env names."""
        with pytest.raises(ValueError, match="conninfo"):
            async with try_advisory_lock("", MIGRATION_LOCK_KEY):
                pass  # pragma: no cover


class TestMigrationArgs:
    """``MIGRATION_ARGS`` resolves constructor arguments from the runner's config."""

    @staticmethod
    def _runner(enhancement_modules: dict | None = None) -> MigrationRunner:
        data: dict = {"database": {"uri": "postgresql://localhost:5432/test"}}
        if enhancement_modules is not None:
            data["enhancement_modules"] = enhancement_modules
        return MigrationRunner(None, ARIELConfig.from_dict(data), lock_factory=no_lock)  # type: ignore[arg-type]

    @staticmethod
    def _fold(runner: MigrationRunner):
        (fold,) = [
            m for m in runner._get_enabled_migrations() if m.name == "attachment_text_upstream_fold"
        ]
        return fold

    def test_resolver_map_replaces_the_embedding_frozenset(self) -> None:
        assert set(migrations_module.MIGRATION_ARGS) == {
            "text_embedding",
            "text_embedding_hnsw_index",
            "attachment_text_upstream_fold",
            "image_embedding",
        }
        assert not hasattr(migrations_module, "MIGRATIONS_TAKING_EMBEDDING_MODELS")

    def test_fold_is_registered_always_on_after_the_text_columns(self) -> None:
        from osprey.services.ariel_search.database.attachment_text_migration import (
            AttachmentTextUpstreamFoldMigration,
        )

        (row,) = [
            r for r in migrations_module.KNOWN_MIGRATIONS if r[0] == "attachment_text_upstream_fold"
        ]
        assert row == (
            "attachment_text_upstream_fold",
            "osprey.services.ariel_search.database.attachment_text_migration",
            "AttachmentTextUpstreamFoldMigration",
            None,
        )
        assert AttachmentTextUpstreamFoldMigration(None).depends_on == ["attachment_text_columns"]

    def test_fold_gets_the_configured_caption_model(self) -> None:
        runner = self._runner(
            {"image_caption": {"enabled": False, "provider": "ollama", "model": {"model_id": "m"}}}
        )

        assert self._fold(runner).model_id == "m"

    def test_fold_gets_none_without_a_caption_model(self) -> None:
        assert self._fold(self._runner()).model_id is None

    def test_embedding_migrations_still_get_the_configured_models(self, monkeypatch) -> None:
        captured: list[object] = []

        class Probe(RecordingMigration):
            def __init__(self, models=None) -> None:
                super().__init__("text_embedding")
                captured.append(models)

        class ProbeModule:
            TextEmbeddingMigration = Probe

        runner = self._runner()
        monkeypatch.setattr(ARIELConfig, "is_enhancement_module_enabled", lambda _self, _name: True)
        monkeypatch.setattr(runner, "_configured_embedding_models", lambda: [("nomic", 768)])
        monkeypatch.setattr(
            migrations_module,
            "KNOWN_MIGRATIONS",
            [("text_embedding", "probe.module", "TextEmbeddingMigration", "text_embedding")],
        )
        monkeypatch.setattr(migrations_module.importlib, "import_module", lambda _p: ProbeModule)

        runner._get_enabled_migrations()

        assert captured == [[("nomic", 768)]]


class TestImageEmbeddingResolver:
    """The ``image_embedding`` resolver and the runner's skip on a bad block."""

    MODEL_KEY = "ariel.enhancement_modules.image_embedding.model"
    DIMS_KEY = "ariel.enhancement_modules.image_embedding.dimensions"

    @staticmethod
    def _runner(image_block: dict | None, pool=None) -> MigrationRunner:
        modules = {} if image_block is None else {"image_embedding": image_block}
        runner = TestMigrationArgs._runner(modules)
        runner.pool = pool
        return runner

    @staticmethod
    def _recording_modules(monkeypatch) -> dict[str, RecordingMigration]:
        """Swap every registered class for a RecordingMigration, keeping real resolvers."""
        built: dict[str, RecordingMigration] = {}
        rows = {row[2]: row for row in migrations_module.KNOWN_MIGRATIONS}

        class AnyModule:
            def __getattr__(self, class_name: str):
                name = rows[class_name][0]

                def build(*_args):
                    migration = RecordingMigration(name)
                    built[name] = migration
                    return migration

                return build

        monkeypatch.setattr(migrations_module.importlib, "import_module", lambda _p: AnyModule())
        return built

    def test_registry_row_requires_the_module_after_copy_state(self) -> None:
        names = [row[0] for row in migrations_module.KNOWN_MIGRATIONS]
        (row,) = [r for r in migrations_module.KNOWN_MIGRATIONS if r[0] == "image_embedding"]
        assert row == (
            "image_embedding",
            "osprey.services.ariel_search.enhancement.image_embedding.migration",
            "ImageEmbeddingMigration",
            "image_embedding",
        )
        assert names.index("image_embedding") > names.index("attachment_files_copy_state")

    def test_resolver_reads_image_embedding_target(self) -> None:
        runner = self._runner(
            {"enabled": True, "provider": "ollama", "model": "clip-vit", "dimensions": 512}
        )

        (target,) = migrations_module.MIGRATION_ARGS["image_embedding"](runner)

        assert target == image_embedding_target({"model": "clip-vit", "dimensions": 512})
        assert target.table == image_table_name("clip-vit", 512)

    def test_resolver_raises_the_model_key_without_a_model(self) -> None:
        runner = self._runner({"enabled": True, "provider": "ollama"})

        with pytest.raises(ModuleConfigError) as exc:
            migrations_module.MIGRATION_ARGS["image_embedding"](runner)

        assert exc.value.key == self.MODEL_KEY

    def test_enabled_block_builds_the_real_migration(self) -> None:
        from osprey.services.ariel_search.enhancement.image_embedding import (
            ImageEmbeddingMigration,
        )

        runner = self._runner(
            {"enabled": True, "provider": "ollama", "model": "clip-vit", "dimensions": 512}
        )

        (migration,) = [m for m in runner._get_enabled_migrations() if m.name == "image_embedding"]

        assert isinstance(migration, ImageEmbeddingMigration)
        assert migration.depends_on == ["attachment_files_copy_state"]
        assert migration.target.table == image_table_name("clip-vit", 512)

    def test_disabled_block_never_resolves(self) -> None:
        runner = self._runner({"enabled": False, "provider": "ollama"})

        names = [m.name for m in runner._get_enabled_migrations()]

        assert "image_embedding" not in names
        assert "core_schema" in names

    @pytest.mark.parametrize(
        ("block", "key"),
        [
            ({"enabled": True, "provider": "ollama"}, MODEL_KEY),
            ({"enabled": True, "provider": "ollama", "model": "m", "dimensions": 0}, DIMS_KEY),
        ],
    )
    async def test_bad_block_skips_only_image_embedding(
        self, fake_pool, caplog, monkeypatch, block, key
    ) -> None:
        """run() applies core_schema and the rest, with one warning naming the key."""
        caplog.set_level(logging.WARNING, logger="ariel")
        built = self._recording_modules(monkeypatch)
        runner = self._runner(block, fake_pool)

        applied, busy = await runner.run()

        expected = [
            row[0]
            for row in migrations_module.KNOWN_MIGRATIONS
            if row[3] is None and row[0] != "image_embedding"
        ]
        assert "image_embedding" not in built
        assert busy == []
        assert "core_schema" in applied
        assert sorted(applied) == sorted(expected)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert key in warnings[0].getMessage()
        assert "image_embedding" in warnings[0].getMessage()

    async def test_bad_block_holds_back_a_dependent(self, fake_pool, caplog, monkeypatch) -> None:
        """A migration built on image_embedding waits for it, as after a skip in up()."""
        caplog.set_level(logging.WARNING, logger="ariel")
        built = self._recording_modules(monkeypatch)
        monkeypatch.setattr(
            migrations_module,
            "KNOWN_MIGRATIONS",
            [
                *migrations_module.KNOWN_MIGRATIONS,
                ("image_dependent", "probe.module", "ImageDependent", "image_embedding"),
            ],
        )
        real_import = migrations_module.importlib.import_module

        def import_module(path: str):
            if path == "probe.module":
                dependent = RecordingMigration("image_dependent", ["image_embedding"])
                built["image_dependent"] = dependent
                return type("Probe", (), {"ImageDependent": staticmethod(lambda: dependent)})
            return real_import(path)

        monkeypatch.setattr(migrations_module.importlib, "import_module", import_module)
        runner = self._runner({"enabled": True, "provider": "ollama"}, fake_pool)

        applied, _busy = await runner.run()

        assert "image_dependent" not in applied
        assert "up" not in built["image_dependent"].events
        assert "core_schema" in applied
        assert "image_dependent waits for image_embedding" in caplog.text

    async def test_skip_warning_is_logged_once_per_runner(
        self, fake_pool, caplog, monkeypatch
    ) -> None:
        """status() then run() on one runner warn about the bad block once."""
        caplog.set_level(logging.WARNING, logger="ariel")
        self._recording_modules(monkeypatch)
        runner = self._runner({"enabled": True, "provider": "ollama"}, fake_pool)

        await runner.status()
        await runner.run()

        skips = [r for r in caplog.records if "Migration image_embedding skipped" in r.getMessage()]
        assert len(skips) == 1

    async def test_rollback_names_the_config_cause(self, fake_pool, caplog, monkeypatch) -> None:
        """A registered migration the block cannot build is not reported as missing."""
        caplog.set_level(logging.WARNING, logger="ariel")
        self._recording_modules(monkeypatch)
        runner = self._runner({"enabled": True, "provider": "ollama"}, fake_pool)

        assert await runner.rollback("image_embedding") is False

        assert "Migration image_embedding cannot be built" in caplog.text
        assert self.MODEL_KEY in caplog.text
        assert "Migration not found" not in caplog.text

    def test_index_ddl_comes_from_the_shared_producer(self) -> None:
        """The image index is the text lane's HNSW cosine DDL under the image name."""
        from osprey.services.ariel_search.enhancement.image_embedding.migration import (
            create_image_index_sql,
        )
        from osprey.services.ariel_search.enhancement.text_embedding.migration import (
            create_vector_index_sql,
            vector_index_name,
        )

        table = image_table_name("clip-vit", 512)
        sql = create_image_index_sql(table)

        assert sql == create_vector_index_sql(table).replace(
            vector_index_name(table), image_index_name(table)
        )
        assert f"IF NOT EXISTS {image_index_name(table)} ON {table} " in sql
        assert vector_index_name(table) not in sql
        assert sql.endswith("USING hnsw (embedding vector_cosine_ops)")

    def test_enumeration_skips_a_bad_block_without_raising(self, caplog) -> None:
        """run(), status() and rollback() all enumerate through this skip."""
        caplog.set_level(logging.WARNING, logger="ariel")
        runner = self._runner({"enabled": True, "provider": "ollama"})

        names = [m.name for m in runner._get_enabled_migrations()]

        assert "image_embedding" not in names
        assert self.MODEL_KEY in caplog.text


class TestV2IndexMigrations:
    """The v2 search-index migrations: registry rows, ordering, and build shape."""

    @staticmethod
    def _order(semantic_processor: bool, text_embedding: bool) -> list[str]:
        data: dict = {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "enhancement_modules": {
                "semantic_processor": {"enabled": semantic_processor},
                "text_embedding": {"enabled": text_embedding},
            },
        }
        runner = MigrationRunner(None, ARIELConfig.from_dict(data), lock_factory=no_lock)  # type: ignore[arg-type]
        return [m.name for m in runner._topological_sort(runner._get_enabled_migrations())]

    def test_v2_rows_are_registered(self) -> None:
        rows = {r[0]: r[1:] for r in migrations_module.KNOWN_MIGRATIONS}

        assert rows["raw_text_fts_index_v2"] == (
            "osprey.services.ariel_search.database.attachment_text_migration",
            "RawTextFtsIndexV2Migration",
            None,
        )
        assert rows["semantic_processor_search_index_v2"] == (
            "osprey.services.ariel_search.enhancement.semantic_processor.search_migration",
            "SemanticProcessorSearchIndexV2Migration",
            "semantic_processor",
        )

    def test_v2_dependencies(self) -> None:
        from osprey.services.ariel_search.database.attachment_text_migration import (
            RawTextFtsIndexV2Migration,
        )
        from osprey.services.ariel_search.enhancement.semantic_processor.search_migration import (
            SemanticProcessorSearchIndexV2Migration,
        )

        assert RawTextFtsIndexV2Migration().depends_on == ["attachment_text_upstream_fold"]
        assert SemanticProcessorSearchIndexV2Migration().depends_on == [
            "attachment_text_upstream_fold",
            "semantic_processor_search_index",
        ]

    @pytest.mark.parametrize("semantic_processor", [False, True])
    @pytest.mark.parametrize("text_embedding", [False, True])
    def test_topological_order_for_each_module_combination(
        self, semantic_processor: bool, text_embedding: bool
    ) -> None:
        order = self._order(semantic_processor, text_embedding)

        assert len(order) == len(set(order))
        pos = order.index
        assert pos("core_schema") < pos("attachment_text_columns")
        assert pos("attachment_text_columns") < pos("attachment_text_upstream_fold")
        assert pos("attachment_text_upstream_fold") < pos("raw_text_fts_index_v2")
        assert pos("keyword_search_fts_index") < pos("raw_text_fts_index_v2")  # v1 kept too
        assert ("semantic_processor_search_index_v2" in order) is semantic_processor
        assert ("text_embedding" in order) is text_embedding
        if semantic_processor:
            v2 = pos("semantic_processor_search_index_v2")
            assert pos("semantic_processor") < pos("semantic_processor_search_index") < v2
            assert pos("attachment_text_upstream_fold") < v2

    async def test_v2_build_takes_share_lock_then_work_mem_then_both_indexes(self) -> None:
        from osprey.services.ariel_search.database.attachment_text_migration import (
            RawTextFtsIndexV2Migration,
        )
        from osprey.services.ariel_search.database.search_fts import RAW_TEXT_FTS_EXPRESSION_V2

        conn = FakeLockConn(refusals=0)

        await RawTextFtsIndexV2Migration().up(conn)  # type: ignore[arg-type]

        lock, work_mem, fts, trgm = conn.statements
        assert lock.as_string(None) == 'LOCK TABLE "enhanced_entries" IN SHARE MODE NOWAIT'  # type: ignore[attr-defined]
        assert work_mem == "SET LOCAL maintenance_work_mem = '256MB'"
        assert "CREATE INDEX IF NOT EXISTS idx_entries_raw_text_fts_v2" in fts
        assert f"GIN({RAW_TEXT_FTS_EXPRESSION_V2})" in fts
        assert "CREATE INDEX IF NOT EXISTS idx_entries_attachment_text_trgm" in trgm
        assert "GIN((COALESCE(attachment_text,'')) gin_trgm_ops)" in trgm
        assert not any("DROP" in str(s) for s in conn.statements)  # v1 kept

    async def test_semantic_v2_build_keeps_v1(self) -> None:
        from osprey.services.ariel_search.database.search_fts import SEMANTIC_FTS_EXPRESSION_V2
        from osprey.services.ariel_search.enhancement.semantic_processor.search_migration import (
            SemanticProcessorSearchIndexV2Migration,
        )

        conn = FakeLockConn(refusals=0)

        await SemanticProcessorSearchIndexV2Migration().up(conn)  # type: ignore[arg-type]

        lock, work_mem, index = conn.statements
        assert "IN SHARE MODE NOWAIT" in lock.as_string(None)  # type: ignore[attr-defined]
        assert work_mem == "SET LOCAL maintenance_work_mem = '256MB'"
        assert "CREATE INDEX IF NOT EXISTS idx_entries_text_search_v2" in index
        assert f"GIN({SEMANTIC_FTS_EXPRESSION_V2})" in index
        assert not any("DROP" in str(s) for s in conn.statements)

    async def test_v2_busy_table_raises_busy(self, monkeypatch) -> None:
        from osprey.services.ariel_search.database.attachment_text_migration import (
            RawTextFtsIndexV2Migration,
        )

        async def _sleep(_delay):
            return None

        monkeypatch.setattr(migrations_module.asyncio, "sleep", _sleep)
        conn = FakeLockConn(refusals=100)

        with pytest.raises(MigrationBusyError):
            await RawTextFtsIndexV2Migration().up(conn)  # type: ignore[arg-type]

        assert not any("CREATE INDEX" in str(s) for s in conn.statements)


_LONG_MODEL = "organisation/very-long-multimodal-embedding-model-name-v2.5-q8_0"


class TestImageTableName:
    """Image-embedding table naming, with text-table naming pinned unchanged."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("nomic-embed-text", "text_embeddings_nomic_embed_text"),
            ("all-MiniLM-L6-v2", "text_embeddings_all_minilm_l6_v2"),
            ("org/Model.v1--x", "text_embeddings_org_model_v1_x"),
            ("text-embedding-3-small", "text_embeddings_text_embedding_3_small"),
        ],
    )
    def test_text_table_name_unchanged(self, model, expected):
        assert model_to_table_name(model) == expected

    @pytest.mark.parametrize(
        "model", ["m", "nomic-embed-vision", _LONG_MODEL, "Ünïcödé/" * 20, "---"]
    )
    @pytest.mark.parametrize("dims", [1, 512, 1024, 2000])
    def test_image_table_name_within_identifier_limit(self, model, dims):
        name = image_table_name(model, dims)
        assert len(name.encode()) <= 63
        assert re.fullmatch(r"image_embeddings_[a-z0-9_]*_[0-9a-f]{8}_d[0-9]+", name)

    def test_image_table_name_longest_is_exactly_63(self):
        assert len(image_table_name(_LONG_MODEL, 2000).encode()) == 63

    def test_image_table_name_shape(self):
        model = "nomic-embed-vision"
        digest = hashlib.sha1(model.encode()).hexdigest()[:8]
        assert image_table_name(model, 768) == f"image_embeddings_nomic_embed_vision_{digest}_d768"

    def test_image_table_name_widths_of_long_model_differ(self):
        assert image_table_name(_LONG_MODEL, 512) != image_table_name(_LONG_MODEL, 1024)

    def test_image_table_name_punctuation_variants_differ(self):
        assert image_table_name("m-q8", 1024) != image_table_name("m.q8", 1024)

    def test_image_table_name_case_variants_differ(self):
        assert image_table_name("Model", 1024) != image_table_name("model", 1024)

    def test_image_table_name_long_prefix_variants_differ(self):
        assert image_table_name(_LONG_MODEL + "a", 1024) != image_table_name(
            _LONG_MODEL + "b", 1024
        )

    def test_image_index_name_from_table_name(self):
        table = image_table_name(_LONG_MODEL, 1024)
        name = image_index_name(table)
        assert name == f"idx_img_{hashlib.sha1(table.encode()).hexdigest()[:16]}"
        assert len(name.encode()) <= 63
        assert image_index_name(image_table_name(_LONG_MODEL, 512)) != name


class TestImageEmbeddingTarget:
    """The one reader of the image-embedding block."""

    def test_image_embedding_target_resolves_model_dims_table(self):
        target = image_embedding_target({"model": "m-q8", "dimensions": 512, "enabled": True})
        assert target == ImageEmbeddingTarget(
            model="m-q8", dims=512, table=image_table_name("m-q8", 512)
        )

    def test_image_embedding_target_defaults_dims_to_1024(self):
        target = image_embedding_target({"model": "m"})
        assert target.dims == 1024
        assert target.table == image_table_name("m", 1024)

    @pytest.mark.parametrize("dims", [1, 2000])
    def test_image_embedding_target_accepts_bounds(self, dims):
        assert image_embedding_target({"model": "m", "dimensions": dims}).dims == dims

    @pytest.mark.parametrize("cfg", [{}, {"model": None}, {"model": ""}, {"dimensions": 512}])
    def test_image_embedding_target_missing_model(self, cfg):
        with pytest.raises(ModuleConfigError) as exc:
            image_embedding_target(cfg)
        assert exc.value.key == "ariel.enhancement_modules.image_embedding.model"
        assert str(exc.value) == "ariel.enhancement_modules.image_embedding.model is required"
        assert isinstance(exc.value, ValueError)

    @pytest.mark.parametrize("dims", [0, -1, 2001, True, False, 512.0, "1024", None])
    def test_image_embedding_target_rejects_bad_dims(self, dims):
        with pytest.raises(ModuleConfigError) as exc:
            image_embedding_target({"model": "m", "dimensions": dims})
        assert exc.value.key == "ariel.enhancement_modules.image_embedding.dimensions"
        assert "dimensions" in str(exc.value)
