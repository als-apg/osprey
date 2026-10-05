"""Unit tests for the *maintenance* half of ``ariel_search.cli_operations``.

These are the operations that reshape the database rather than read it:
``run_migrate``, ``run_reembed`` (+ its ``_embed_batch`` helper),
``run_quickstart``, ``get_purge_info`` and ``execute_purge``. The pure-logic
and error-translation contracts live in ``test_cli_operations.py``; the
pipeline operations in ``test_cli_operations_pipeline.py``.

Everything here runs against the fake pool/connection/cursor family in
``conftest.py``, so the assertions are about the *SQL that gets emitted*, the
progress narration, and the accounting the functions return. Two shapes matter:

* ``conn.cursor()`` with no ``row_factory`` yields **positional tuple** rows —
  ``get_purge_info``, ``execute_purge``, ``run_reembed`` and ``_embed_batch``
  all read them positionally, so the scripts below supply tuples.
* the provider registry returns a provider *class* whose ``execute_embedding``
  is **synchronous**; ARIEL calls it from async code without awaiting.

``cli_operations`` imports its collaborators lazily inside each function, so
each is monkeypatched at its *source* module: ``create_connection_pool`` on
``...database.connection``, ``run_migrations`` on ``...database.migrations``,
``get_provider`` on ``ProviderRegistry``. No real database,
network or embedding backend is touched.

One purge contract is asserted here at the SQL level only: both
``execute_purge`` branches must delete the ``text_embedding`` row from
``ariel_migrations``, guarded by an existence check on the table itself,
otherwise a later ``osprey ariel migrate`` is a silent no-op and the dropped
tables never come back. That the statement really fires against Postgres
stays covered by the container tests in ``test_apply_seed.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.database.repository import SchemaFacts, attachment_text_md5
from tests.services.ariel_search._cli_ops_doubles import (
    _Adapter,
    _Enhancer,
    _forbid_service,
    _patch_adapter,
    _patch_enhancers,
    _patch_migrations,
    _patch_pool,
    _patch_service,
    _StubService,
)
from tests.services.ariel_search.fake_providers import make_fake_embedding_provider

# Minimal config dict accepted by ARIELConfig.from_dict.
_DB = {"database": {"uri": "postgresql://localhost/test"}}

# The table run_reembed derives for the model used throughout this module.
_MODEL = "nomic-embed-text"
_TABLE = "text_embeddings_nomic_embed_text"

# SQL fragments the code under test emits, quoted here once.
_SELECT_ENTRIES = "SELECT entry_id, raw_text FROM enhanced_entries"
_SELECT_ENTRIES_WITH_PICTURES = "SELECT entry_id, raw_text, attachment_text FROM enhanced_entries"
_SELECT_EMBEDDING_TABLES = "SELECT table_name FROM information_schema.tables"
_PGVECTOR_PROBE = "pg_available_extensions"

# Includes the existence-check guard around the DELETE, not just the DELETE
# itself: without ``IF EXISTS (...ariel_migrations...) THEN`` wrapping it, the
# statement aborts with an UndefinedTable error against a database that has
# never run migrations, instead of the intended no-op.
_UNRECORD_MIGRATION = (
    "IF EXISTS (SELECT 1 FROM information_schema.tables "
    "WHERE table_schema = 'public' AND table_name = 'ariel_migrations') "
    "THEN DELETE FROM ariel_migrations WHERE name = 'text_embedding'"
)


# ---------------------------------------------------------------------------
# Embedding seam and repository stubs -- shared doubles in ``_cli_ops_doubles``
# ---------------------------------------------------------------------------


def _patch_embedding_provider(monkeypatch, provider):
    """Route ``ProviderRegistry.get_provider``; returns the provider names requested.

    Args:
        provider: A fake provider instance (its class answers every lookup), a
            ``{name: class}`` mapping (unlisted names answer ``None``), or an
            exception to raise.
    """
    from osprey.models.provider_registry import ProviderRegistry

    asked: list[str] = []

    def _fake_get(_self, name):
        asked.append(name)
        if isinstance(provider, Exception):
            raise provider
        if isinstance(provider, dict):
            return provider.get(name)
        return type(provider)

    monkeypatch.setattr(ProviderRegistry, "get_provider", _fake_get)
    return asked


def _reembed_repo(*, tables=(), entry_count=0, has_copy_state=False):
    """Repository mock with just what ``run_reembed`` awaits (schema-behind by default)."""
    repo = MagicMock()
    repo.get_embedding_tables = AsyncMock(return_value=list(tables))
    repo.count_entries = AsyncMock(return_value=entry_count)
    repo.schema_facts = AsyncMock(return_value=SchemaFacts(False, has_copy_state))
    return repo


def _embedding_table(name):
    table = MagicMock()
    table.table_name = name
    return table


# ---------------------------------------------------------------------------
# run_migrate
# ---------------------------------------------------------------------------


class TestRunMigrate:
    async def test_progress_narrates_connect_run_and_complete(self, monkeypatch, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        calls = _patch_migrations(monkeypatch, applied=["core_schema"])

        messages: list[str] = []
        await ops.run_migrate(
            {"database": {"uri": "postgresql://user:secret@dbhost:5432/ariel"}},
            progress=messages.append,
        )

        assert messages == [
            "Connecting to database: dbhost:5432/ariel",
            "Running migrations...",
            "Migrations complete.",
        ]
        # Credentials are stripped from the connection banner.
        assert not any("secret" in m for m in messages)
        assert len(calls) == 1
        assert calls[0][0] is fake_pool
        assert fake_pool.closed

    async def test_runs_and_closes_without_a_progress_callback(self, monkeypatch, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        calls = _patch_migrations(monkeypatch)

        assert await ops.run_migrate(dict(_DB)) == []

        assert len(calls) == 1
        assert fake_pool.closed

    async def test_returns_the_busy_skipped_migrations(self, monkeypatch, fake_pool):
        """The names the runner skipped as busy are what the CLI reports."""
        import osprey.services.ariel_search.database.migrations as mig_mod

        _patch_pool(monkeypatch, fake_pool)

        async def _busy(_pool, _config, **_kwargs):
            return mig_mod.MigrationResult(["core_schema"], ["attachment_text_columns"], False)

        monkeypatch.setattr(mig_mod, "run_migrations_detailed", _busy)

        assert await ops.run_migrate(dict(_DB)) == ["attachment_text_columns"]
        assert fake_pool.closed

    async def test_pool_is_closed_when_migrations_raise(self, monkeypatch, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, error=RuntimeError("migration exploded"))

        messages: list[str] = []
        with pytest.raises(RuntimeError, match="migration exploded"):
            await ops.run_migrate(dict(_DB), progress=messages.append)

        # The completion line is never reached, but the pool is still released.
        assert "Migrations complete." not in messages
        assert fake_pool.closed


# ---------------------------------------------------------------------------
# run_reembed — non-dry-run
# ---------------------------------------------------------------------------


class TestRunReembed:
    async def test_missing_table_is_created_before_embedding(
        self, monkeypatch, fake_pool_factory, fake_embedding_provider
    ):
        pool = fake_pool_factory(
            rows_for={
                _PGVECTOR_PROBE: [(True,)],
                _SELECT_ENTRIES: [("E1", "first text")],
                f"SELECT 1 FROM {_TABLE}": [],
            }
        )
        repo = _reembed_repo(tables=[], entry_count=1)
        _patch_service(monkeypatch, _StubService(repo, pool))
        _patch_embedding_provider(monkeypatch, fake_embedding_provider)

        messages: list[str] = []
        result = await ops.run_reembed(
            dict(_DB),
            model=_MODEL,
            dimension=4,
            batch_size=10,
            dry_run=False,
            force=False,
            progress=messages.append,
        )

        assert result.dry_run is False
        assert result.processed == 1
        # The real TextEmbeddingMigration ran against the fake connection.
        assert pool.matching(f"CREATE TABLE IF NOT EXISTS {_TABLE}")
        assert pool.matching("CREATE EXTENSION IF NOT EXISTS vector")
        assert f"Creating embedding table: {_TABLE}" in messages
        assert f"  Table created: {_TABLE}" in messages

    async def test_existing_table_is_not_recreated(
        self, monkeypatch, fake_pool_factory, fake_embedding_provider
    ):
        pool = fake_pool_factory(
            rows_for={
                _SELECT_ENTRIES: [("E1", "first text")],
                f"SELECT 1 FROM {_TABLE}": [],
            }
        )
        repo = _reembed_repo(tables=[_embedding_table(_TABLE)], entry_count=1)
        _patch_service(monkeypatch, _StubService(repo, pool))
        _patch_embedding_provider(monkeypatch, fake_embedding_provider)

        messages: list[str] = []
        result = await ops.run_reembed(
            dict(_DB),
            model=_MODEL,
            dimension=4,
            batch_size=10,
            dry_run=False,
            force=False,
            progress=messages.append,
        )

        assert result.processed == 1
        assert not pool.matching("CREATE TABLE IF NOT EXISTS")
        assert not any("Creating embedding table" in m for m in messages)

    async def test_zero_entries_short_circuits_before_the_provider(self, monkeypatch, fake_pool):
        repo = _reembed_repo(tables=[_embedding_table(_TABLE)], entry_count=0)
        _patch_service(monkeypatch, _StubService(repo, fake_pool))
        # The provider lookup sits *after* the count check; reaching it is a bug.
        _patch_embedding_provider(monkeypatch, AssertionError("provider must not be resolved"))

        messages: list[str] = []
        result = await ops.run_reembed(
            dict(_DB),
            model=_MODEL,
            dimension=4,
            batch_size=10,
            dry_run=False,
            force=False,
            progress=messages.append,
        )

        assert (result.processed, result.skipped, result.errors) == (0, 0, 0)
        assert result.dry_run is False
        assert "Found 0 entries to embed" in messages
        assert "No entries to embed." in messages
        # Nothing was read from or written to the database.
        assert fake_pool.calls == []

    async def test_force_skips_the_existence_probe_and_upserts(
        self, monkeypatch, fake_pool_factory, fake_embedding_provider
    ):
        rows = [("E1", "one"), ("E2", "two"), ("E3", "three")]
        pool = fake_pool_factory(rows_for={_SELECT_ENTRIES: rows})
        repo = _reembed_repo(tables=[_embedding_table(_TABLE)], entry_count=3)
        _patch_service(monkeypatch, _StubService(repo, pool))
        _patch_embedding_provider(monkeypatch, fake_embedding_provider)

        result = await ops.run_reembed(
            dict(_DB),
            model=_MODEL,
            dimension=4,
            batch_size=2,
            dry_run=False,
            force=True,
            progress=None,
        )

        assert (result.processed, result.skipped, result.errors) == (3, 0, 0)
        # force=True means the "already embedded?" probe is never issued.
        assert not pool.matching(f"SELECT 1 FROM {_TABLE} WHERE entry_id")
        # batch_size=2 over 3 rows -> a mid-loop flush plus the trailing flush.
        assert [len(call["texts"]) for call in fake_embedding_provider.calls] == [2, 1]
        assert len(pool.matching(f"INSERT INTO {_TABLE}")) == 3
        assert pool.matching("ON CONFLICT (entry_id) DO UPDATE SET embedding = EXCLUDED.embedding")

    async def test_already_embedded_rows_are_skipped_when_not_forcing(
        self, monkeypatch, fake_pool_factory, fake_embedding_provider
    ):
        rows = [("E1", "one"), ("E2", "two")]
        pool = fake_pool_factory(
            rows_for={
                _SELECT_ENTRIES: rows,
                f"SELECT 1 FROM {_TABLE}": [(1,)],
            }
        )
        repo = _reembed_repo(tables=[_embedding_table(_TABLE)], entry_count=2)
        _patch_service(monkeypatch, _StubService(repo, pool))
        _patch_embedding_provider(monkeypatch, fake_embedding_provider)

        result = await ops.run_reembed(
            dict(_DB),
            model=_MODEL,
            dimension=4,
            batch_size=10,
            dry_run=False,
            force=False,
            progress=None,
        )

        assert (result.processed, result.skipped, result.errors) == (0, 2, 0)
        assert fake_embedding_provider.calls == []
        assert not pool.matching(f"INSERT INTO {_TABLE}")

    async def test_non_force_insert_uses_do_nothing_and_provider_defaults(
        self, monkeypatch, fake_pool_factory, fake_embedding_provider
    ):
        pool = fake_pool_factory(
            rows_for={
                _SELECT_ENTRIES: [("E1", None)],
                f"SELECT 1 FROM {_TABLE}": [],
            }
        )
        repo = _reembed_repo(tables=[_embedding_table(_TABLE)], entry_count=1)
        _patch_service(monkeypatch, _StubService(repo, pool))
        asked = _patch_embedding_provider(monkeypatch, fake_embedding_provider)

        result = await ops.run_reembed(
            dict(_DB),
            model=_MODEL,
            dimension=4,
            batch_size=10,
            dry_run=False,
            force=False,
            progress=None,
        )

        assert result.processed == 1
        assert pool.matching("ON CONFLICT (entry_id) DO NOTHING")
        assert not pool.matching("DO UPDATE SET")
        # With no text_embedding block the provider comes from
        # config.embedding.provider; base_url falls back to the provider's own
        # default because ARIEL config carries none.
        assert asked == ["ollama"]
        call = fake_embedding_provider.calls[0]
        assert call["model_id"] == _MODEL
        assert call["base_url"] == fake_embedding_provider.default_base_url
        assert "dimensions" not in call
        # A NULL raw_text is embedded as the empty string, not None.
        assert call["texts"] == [""]

    async def test_a_long_entry_is_cut_to_the_configured_model_limit(
        self, monkeypatch, fake_pool_factory, fake_embedding_provider
    ):
        pool = fake_pool_factory(
            rows_for={
                _SELECT_ENTRIES: [("E1", "x" * 100)],
                f"SELECT 1 FROM {_TABLE}": [],
            }
        )
        repo = _reembed_repo(tables=[_embedding_table(_TABLE)], entry_count=1)
        _patch_service(monkeypatch, _StubService(repo, pool))
        _patch_embedding_provider(monkeypatch, fake_embedding_provider)
        config = {
            **_DB,
            "enhancement_modules": {
                "text_embedding": {
                    "models": [{"name": _MODEL, "dimension": 4, "max_input_tokens": 40}]
                }
            },
        }

        result = await ops.run_reembed(
            config,
            model=_MODEL,
            dimension=4,
            batch_size=10,
            dry_run=False,
            force=False,
            progress=None,
        )

        assert result.processed == 1
        assert fake_embedding_provider.calls[0]["texts"] == ["x" * 32]

    async def test_an_unlisted_model_is_cut_to_the_default_limit(
        self, monkeypatch, fake_pool_factory, fake_embedding_provider
    ):
        pool = fake_pool_factory(
            rows_for={
                _SELECT_ENTRIES: [("E1", "x" * 1000)],
                f"SELECT 1 FROM {_TABLE}": [],
            }
        )
        repo = _reembed_repo(tables=[_embedding_table(_TABLE)], entry_count=1)
        _patch_service(monkeypatch, _StubService(repo, pool))
        _patch_embedding_provider(monkeypatch, fake_embedding_provider)

        result = await ops.run_reembed(
            dict(_DB),
            model=_MODEL,
            dimension=4,
            batch_size=10,
            dry_run=False,
            force=False,
            progress=None,
        )

        assert result.processed == 1
        assert fake_embedding_provider.calls[0]["texts"] == ["x" * 504]


class TestRunReembedAttachmentText:
    """On a store with copy state the picture text joins the input of entries that have it."""

    _CONFIG = {
        **_DB,
        "enhancement_modules": {
            "text_embedding": {"models": [{"name": _MODEL, "dimension": 4, "max_input_tokens": 40}]}
        },
    }

    async def _reembed(self, monkeypatch, pool, repo, provider):
        _patch_service(monkeypatch, _StubService(repo, pool))
        _patch_embedding_provider(monkeypatch, provider)
        return await ops.run_reembed(
            self._CONFIG,
            model=_MODEL,
            dimension=4,
            batch_size=10,
            dry_run=False,
            force=True,
            progress=None,
        )

    async def test_only_entries_with_attachment_text_get_the_helper_input(
        self, monkeypatch, fake_pool_factory, fake_embedding_provider
    ):
        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            embedding_input,
        )

        pool = fake_pool_factory(
            rows_for={
                _SELECT_ENTRIES_WITH_PICTURES: [
                    ("E1", "x" * 100, "[picture a.png - upstream caption] trip"),
                    ("E2", "y" * 100, None),
                    ("E3", None, "  "),
                ],
            }
        )
        repo = _reembed_repo(tables=[_embedding_table(_TABLE)], entry_count=3, has_copy_state=True)

        result = await self._reembed(monkeypatch, pool, repo, fake_embedding_provider)

        assert result.processed == 3
        assert fake_embedding_provider.calls[0]["texts"] == [
            embedding_input("x" * 100, "[picture a.png - upstream caption] trip", 40),
            "y" * 32,
            "",
        ]
        assert fake_embedding_provider.calls[0]["texts"][0] == "x" * 23 + "\n[picture"
        assert any(_SELECT_ENTRIES_WITH_PICTURES in sql for sql in pool.sql)

    async def test_a_schema_behind_store_keeps_the_b1_query(
        self, monkeypatch, fake_pool_factory, fake_embedding_provider
    ):
        pool = fake_pool_factory(rows_for={_SELECT_ENTRIES: [("E1", "x" * 100)]})
        repo = _reembed_repo(tables=[_embedding_table(_TABLE)], entry_count=1)

        await self._reembed(monkeypatch, pool, repo, fake_embedding_provider)

        assert fake_embedding_provider.calls[0]["texts"] == ["x" * 32]
        assert not any("attachment_text" in sql for sql in pool.sql)


class TestRunEnhanceMarks:
    """run_enhance marks text modules under the md5 of the picture text they read."""

    def _repo(self, has_copy_state, entry):
        repo = MagicMock()
        repo.schema_facts = AsyncMock(return_value=SchemaFacts(False, has_copy_state))
        repo.get_incomplete_entries = AsyncMock(return_value=[entry])
        repo.mark_enhancement_complete = AsyncMock()
        repo.mark_enhancement_failed = AsyncMock(return_value=1)
        return repo

    async def _enhance(self, monkeypatch, fake_pool, repo, names):
        import osprey.services.ariel_search.enhancement as enhancement_pkg

        enhancers = [_Enhancer(name) for name in names]
        monkeypatch.setattr(
            enhancement_pkg, "create_enhancers_from_config", lambda *_a, **_k: enhancers
        )
        _patch_service(monkeypatch, _StubService(repo, fake_pool))
        return await ops.run_enhance(dict(_DB), module=None, force=False, limit=10)

    async def test_copy_state_text_modules_mark_with_md5_others_without(
        self, monkeypatch, fake_pool
    ):
        entry = {"entry_id": "E1", "raw_text": "t", "attachment_text": "[picture] trip"}
        repo = self._repo(True, entry)

        await self._enhance(
            monkeypatch, fake_pool, repo, ["text_embedding", "qmd_export", "semantic_processor"]
        )

        calls = {c.args[1]: c for c in repo.mark_enhancement_complete.await_args_list}
        digest = attachment_text_md5(entry)
        assert calls["text_embedding"].args == ("E1", "text_embedding")
        assert calls["text_embedding"].kwargs == {"md5": digest}
        assert calls["qmd_export"].kwargs == {"md5": digest}
        assert calls["semantic_processor"].kwargs == {}

    async def test_schema_behind_store_marks_exactly_as_b1(self, monkeypatch, fake_pool):
        entry = {"entry_id": "E1", "raw_text": "t"}
        repo = self._repo(False, entry)

        await self._enhance(monkeypatch, fake_pool, repo, ["text_embedding"])

        repo.mark_enhancement_complete.assert_awaited_once_with("E1", "text_embedding")


class TestRunReembedProvider:
    """run_reembed resolves its provider exactly as the text_embedding module does."""

    @staticmethod
    def _wire(monkeypatch, fake_pool_factory):
        pool = fake_pool_factory(
            rows_for={_SELECT_ENTRIES: [("E1", "first text")], f"SELECT 1 FROM {_TABLE}": []}
        )
        repo = _reembed_repo(tables=[_embedding_table(_TABLE)], entry_count=1)
        _patch_service(monkeypatch, _StubService(repo, pool))

    async def test_the_text_embedding_module_provider_wins_and_gets_the_cli_dimension(
        self, monkeypatch, fake_pool_factory
    ):
        self._wire(monkeypatch, fake_pool_factory)
        llama = make_fake_embedding_provider(
            name="llama-cpp", truncates_to_dimensions=True, default_base_url="http://llama:8080"
        )
        ollama = make_fake_embedding_provider()
        asked = _patch_embedding_provider(monkeypatch, {"llama-cpp": llama, "ollama": ollama})
        config = {
            **_DB,
            "embedding": {"provider": "ollama"},
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "provider": "llama-cpp",
                    "models": [{"name": _MODEL, "dimension": 768}],
                }
            },
        }

        result = await ops.run_reembed(
            config,
            model=_MODEL,
            dimension=1024,
            batch_size=10,
            dry_run=False,
            force=False,
            progress=None,
        )

        assert result.processed == 1
        assert asked == ["llama-cpp"]
        assert ollama.calls == []
        assert llama.calls[0]["dimensions"] == 1024
        assert llama.calls[0]["base_url"] == "http://llama:8080"

    async def test_no_text_embedding_block_embeds_through_ollama_without_dimensions(
        self, monkeypatch, fake_pool_factory
    ):
        self._wire(monkeypatch, fake_pool_factory)
        ollama = make_fake_embedding_provider()
        asked = _patch_embedding_provider(monkeypatch, {"ollama": ollama})

        await ops.run_reembed(
            {**_DB, "embedding": {"provider": "ollama"}},
            model=_MODEL,
            dimension=4,
            batch_size=10,
            dry_run=False,
            force=False,
            progress=None,
        )

        assert asked == ["ollama"]
        assert len(ollama.calls) == 1
        assert "dimensions" not in ollama.calls[0]

    @pytest.mark.parametrize(
        ("config", "key"),
        [
            (
                {
                    **_DB,
                    "enhancement_modules": {
                        "text_embedding": {"enabled": True, "provider": "nonesuch"}
                    },
                },
                "ariel.enhancement_modules.text_embedding.provider",
            ),
            ({**_DB, "embedding": {"provider": "nonesuch"}}, "ariel.embedding.provider"),
        ],
    )
    async def test_an_unknown_provider_is_refused_naming_its_key(
        self, monkeypatch, fake_pool_factory, config, key
    ):
        self._wire(monkeypatch, fake_pool_factory)
        _patch_embedding_provider(monkeypatch, {})

        with pytest.raises(ValueError, match="nonesuch") as exc:
            await ops.run_reembed(
                config,
                model=_MODEL,
                dimension=4,
                batch_size=10,
                dry_run=False,
                force=False,
                progress=None,
            )
        assert key in str(exc.value)


# ---------------------------------------------------------------------------
# _embed_batch
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    async def test_force_emits_do_update_conflict_clause(self, fake_pool, fake_embedding_provider):
        cur = fake_pool.conn.cursor()

        messages: list[str] = []
        processed, errors = await ops._embed_batch(
            cur,
            fake_embedding_provider,
            ["alpha", "beta"],
            ["E1", "E2"],
            _MODEL,
            "http://embed.invalid",
            _TABLE,
            True,
            messages.append,
        )

        assert (processed, errors) == (2, 0)
        inserts = fake_pool.matching(f"INSERT INTO {_TABLE} (entry_id, embedding)")
        assert len(inserts) == 2
        assert [params[0] for _, params in inserts] == ["E1", "E2"]
        assert fake_pool.matching(
            "ON CONFLICT (entry_id) DO UPDATE SET embedding = EXCLUDED.embedding"
        )
        assert messages == ["  Processed 2 entries in batch..."]
        assert fake_embedding_provider.calls[0]["base_url"] == "http://embed.invalid"

    async def test_non_force_emits_do_nothing_conflict_clause(
        self, fake_pool, fake_embedding_provider
    ):
        cur = fake_pool.conn.cursor()

        processed, errors = await ops._embed_batch(
            cur,
            fake_embedding_provider,
            ["alpha"],
            ["E1"],
            _MODEL,
            "http://embed.invalid",
            _TABLE,
            False,
            None,
        )

        assert (processed, errors) == (1, 0)
        assert fake_pool.matching("ON CONFLICT (entry_id) DO NOTHING")
        assert not fake_pool.matching("DO UPDATE SET")

    async def test_embedding_failure_counts_the_whole_batch_as_errors(self, fake_pool):
        provider = make_fake_embedding_provider(error=RuntimeError("embedding backend down"))()
        cur = fake_pool.conn.cursor()

        messages: list[str] = []
        processed, errors = await ops._embed_batch(
            cur,
            provider,
            ["alpha", "beta", "gamma"],
            ["E1", "E2", "E3"],
            _MODEL,
            "http://embed.invalid",
            _TABLE,
            False,
            messages.append,
        )

        assert (processed, errors) == (0, 3)
        assert fake_pool.calls == []
        assert messages == ["  Error in batch: embedding backend down"]

    async def test_vector_count_mismatch_is_an_error_not_a_partial_write(self, fake_pool):
        # Provider returns fewer vectors than texts -> zip(strict=True) rejects it.
        provider = make_fake_embedding_provider(vectors=[[0.1, 0.2]])()
        cur = fake_pool.conn.cursor()

        processed, errors = await ops._embed_batch(
            cur,
            provider,
            ["alpha", "beta"],
            ["E1", "E2"],
            _MODEL,
            "http://embed.invalid",
            _TABLE,
            False,
            None,
        )

        assert (processed, errors) == (0, 2)
        # The first row was written before the mismatch surfaced; the batch is
        # still reported as failed rather than partially processed.
        assert len(fake_pool.matching(f"INSERT INTO {_TABLE}")) == 1

    async def test_failure_without_progress_callback_is_silent(self, fake_pool):
        provider = make_fake_embedding_provider(error=RuntimeError("boom"))()
        cur = fake_pool.conn.cursor()

        assert await ops._embed_batch(
            cur, provider, ["a"], ["E1"], _MODEL, "http://x", _TABLE, False, None
        ) == (0, 1)


# ---------------------------------------------------------------------------
# run_quickstart
# ---------------------------------------------------------------------------


class TestRunQuickstart:
    async def test_source_argument_selects_the_generic_json_adapter(
        self, monkeypatch, fake_pool, mock_repository
    ):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=["core_schema"])
        _patch_adapter(monkeypatch, _Adapter([{"entry_id": "E1"}, {"entry_id": "E2"}]))
        _patch_enhancers(monkeypatch, [])
        _patch_service(monkeypatch, _StubService(mock_repository, mock_repository.pool))

        config_dict = dict(_DB)
        messages: list[str] = []
        result = await ops.run_quickstart(
            config_dict,
            source="file:///entries.json",
            progress=messages.append,
        )

        assert config_dict["ingestion"] == {
            "source_url": "file:///entries.json",
            "adapter": "generic_json",
        }
        assert result.count == 2
        assert result.enhanced_count == 0
        assert result.failed_count == 0
        assert result.migrations_applied == 1
        assert mock_repository.upsert_entry.await_count == 2
        assert "  Entries: 2 ingested" in messages
        assert fake_pool.closed

    async def test_unreadable_entries_are_reported_in_progress(
        self, monkeypatch, fake_pool, mock_repository
    ):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        _patch_adapter(monkeypatch, _Adapter([{"entry_id": "E1"}], unreadable=3))
        _patch_enhancers(monkeypatch, [])
        _patch_service(monkeypatch, _StubService(mock_repository, mock_repository.pool))

        messages: list[str] = []
        await ops.run_quickstart(dict(_DB), source="file:///entries.json", progress=messages.append)

        assert "  Skipped: 3 entries that could not be read" in messages

    async def test_no_source_configured_skips_ingestion_entirely(self, monkeypatch, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        _patch_adapter(monkeypatch, AssertionError("adapter must not be built"))
        _forbid_service(monkeypatch, "service must not be created")

        messages: list[str] = []
        result = await ops.run_quickstart(dict(_DB), source=None, progress=messages.append)

        assert result.count == 0
        assert result.migrations_applied == 0
        assert any("Skipping data ingestion" in m for m in messages)
        assert "  Tables: already up to date" in messages
        assert fake_pool.closed

    async def test_a_demo_narrative_is_seeded_then_enhanced_instead_of_ingested(
        self, monkeypatch, fake_pool
    ):
        """No ingestion source and an ``ariel.demo_narrative``: the narrative goes in
        the way a deploy writes it, then the enhancement pass adds what it lacks."""
        import osprey.simulation.apply as apply_mod

        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        _patch_adapter(monkeypatch, AssertionError("adapter must not be built"))
        calls: dict = {}

        monkeypatch.setattr(apply_mod, "demo_narrative_logbook", lambda config: ["E1", "E2", "E3"])

        async def _seed(config, logbook, anchor):
            calls["seeded"] = (config["demo_narrative"], list(logbook), anchor.tzinfo)
            return len(logbook)

        async def _enhance(_config, module, force, limit, _progress=None, **_kwargs):
            calls["enhance"] = (module, force, limit)
            return ops.EnhanceResult(entries_processed=3, module_names=["text_embedding"])

        monkeypatch.setattr(apply_mod, "seed_narrative_if_empty", _seed)
        monkeypatch.setattr(ops, "run_enhance", _enhance)

        messages: list[str] = []
        result = await ops.run_quickstart(
            {**_DB, "demo_narrative": "data/logbook_seed"}, source=None, progress=messages.append
        )

        assert calls["seeded"][:2] == ("data/logbook_seed", ["E1", "E2", "E3"])
        assert calls["seeded"][2] is not None
        assert calls["enhance"] == (None, False, 3)
        assert (result.count, result.enhanced_count) == (3, 3)
        assert "  Entries: 3 seeded" in messages

    async def test_an_explicit_source_wins_over_the_demo_narrative(
        self, monkeypatch, fake_pool, mock_repository
    ):
        import osprey.simulation.apply as apply_mod

        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        _patch_adapter(monkeypatch, _Adapter([{"entry_id": "E1"}]))
        _patch_enhancers(monkeypatch, [])
        _patch_service(monkeypatch, _StubService(mock_repository, mock_repository.pool))

        async def _never(*_args, **_kwargs):
            raise AssertionError("the narrative must not be seeded over an explicit source")

        monkeypatch.setattr(apply_mod, "seed_narrative_if_empty", _never)

        result = await ops.run_quickstart(
            {**_DB, "demo_narrative": "data/logbook_seed"}, source="file:///entries.json"
        )

        assert result.count == 1

    async def test_applied_migrations_are_reported_as_created(self, monkeypatch, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=["core_schema", "text_embedding"])
        _forbid_service(monkeypatch, "service must not be created")

        messages: list[str] = []
        result = await ops.run_quickstart(dict(_DB), source=None, progress=messages.append)

        assert result.migrations_applied == 2
        assert "  Tables: created (2 migrations applied)" in messages

    async def test_enhancers_run_per_entry_and_successes_are_counted(
        self, monkeypatch, fake_pool, mock_repository
    ):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        _patch_adapter(monkeypatch, _Adapter([{"entry_id": "E1"}, {"entry_id": "E2"}]))
        enhancer = _Enhancer(name="text_embedding")
        _patch_enhancers(monkeypatch, [enhancer])
        _patch_service(monkeypatch, _StubService(mock_repository, mock_repository.pool))

        messages: list[str] = []
        result = await ops.run_quickstart(
            {
                **_DB,
                "ingestion": {"adapter": "generic_json", "source_url": "file:///entries.json"},
            },
            source=None,
            progress=messages.append,
        )

        assert result.enhanced_count == 2
        assert result.failed_count == 0
        assert enhancer.seen == ["E1", "E2"]
        assert mock_repository.mark_enhancement_complete.await_count == 2
        assert "  Enhancement modules: ['text_embedding']" in messages
        assert "  Enhancements: 2 applied" in messages

    async def test_enhancer_failures_are_counted_marked_and_logged(
        self, monkeypatch, fake_pool, mock_repository, caplog
    ):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        _patch_adapter(monkeypatch, _Adapter([{"entry_id": "E1"}]))
        _patch_enhancers(
            monkeypatch,
            [_Enhancer(name="text_embedding", error=RuntimeError("no embedding backend"))],
        )
        _patch_service(monkeypatch, _StubService(mock_repository, mock_repository.pool))

        messages: list[str] = []
        with caplog.at_level(logging.DEBUG, logger="ariel"):
            result = await ops.run_quickstart(
                {
                    **_DB,
                    "ingestion": {"adapter": "generic_json", "source_url": "file:///entries.json"},
                },
                source=None,
                progress=messages.append,
            )

        assert result.count == 1
        assert result.enhanced_count == 0
        assert result.failed_count == 1
        mock_repository.mark_enhancement_failed.assert_awaited_once_with(
            "E1", "text_embedding", "no embedding backend"
        )
        assert "Enhancement failed for E1" in caplog.text
        # The failure count rides along on the enhancement summary line.
        assert "  Enhancements: 0 applied, 1 failed" in messages

    async def test_enabled_search_modules_are_reported(self, monkeypatch, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        _forbid_service(monkeypatch, "service must not be created")

        messages: list[str] = []
        result = await ops.run_quickstart(
            {**_DB, "search_modules": {"keyword": {"enabled": True}}},
            source=None,
            progress=messages.append,
        )

        assert "keyword" in result.enabled_search
        assert any("Search modules: " in m for m in messages)
        assert any("osprey ariel search" in m for m in messages)

    async def test_runs_without_a_progress_callback(self, monkeypatch, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=["core_schema"])
        _forbid_service(monkeypatch, "service must not be created")

        result = await ops.run_quickstart(dict(_DB), source=None)

        assert result.migrations_applied == 1
        assert fake_pool.closed

    async def test_pool_is_closed_when_migrations_fail(self, monkeypatch, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, error=RuntimeError("no such database"))
        _forbid_service(monkeypatch, "service must not be created")

        with pytest.raises(RuntimeError, match="no such database"):
            await ops.run_quickstart(dict(_DB), source=None)

        assert fake_pool.closed


# ---------------------------------------------------------------------------
# get_purge_info
# ---------------------------------------------------------------------------


class TestGetPurgeInfo:
    async def test_reports_entry_count_and_embedding_tables(self, monkeypatch, fake_pool_factory):
        pool = fake_pool_factory(
            rows_for={
                "SELECT COUNT(*) FROM enhanced_entries": [(7,)],
                _SELECT_EMBEDDING_TABLES: [
                    ("text_embeddings_nomic_embed_text",),
                    ("text_embeddings_mxbai",),
                ],
            }
        )
        _patch_pool(monkeypatch, pool)

        info = await ops.get_purge_info(dict(_DB))

        assert info.entry_count == 7
        assert info.embedding_tables == [
            "text_embeddings_nomic_embed_text",
            "text_embeddings_mxbai",
        ]
        assert pool.closed

    async def test_missing_count_row_defaults_to_zero(self, monkeypatch, fake_pool):
        _patch_pool(monkeypatch, fake_pool)

        info = await ops.get_purge_info(dict(_DB))

        assert info.entry_count == 0
        assert info.embedding_tables == []

    async def test_pool_is_closed_when_the_count_query_fails(self, monkeypatch, fake_pool_factory):
        pool = fake_pool_factory(
            rows_for={"SELECT COUNT(*) FROM enhanced_entries": RuntimeError("table is gone")}
        )
        _patch_pool(monkeypatch, pool)

        with pytest.raises(RuntimeError, match="table is gone"):
            await ops.get_purge_info(dict(_DB))

        assert pool.closed


# ---------------------------------------------------------------------------
# execute_purge
# ---------------------------------------------------------------------------


class TestExecutePurge:
    """Both branches must unrecord the ``text_embedding`` migration.

    Purging drops the migration-owned ``text_embeddings_*`` tables. If the
    migration stays recorded as applied, the next ``osprey ariel migrate`` is a
    no-op and the tables are never recreated — so both the ``DELETE FROM
    ariel_migrations`` and the ``IF EXISTS`` guard wrapping it (dropping the
    guard would abort with UndefinedTable on a database that never migrated)
    are asserted on both paths.
    """

    async def test_embeddings_only_drops_tables_and_preserves_entries(
        self, monkeypatch, fake_pool_factory
    ):
        pool = fake_pool_factory(
            rows_for={
                _SELECT_EMBEDDING_TABLES: [
                    ("text_embeddings_nomic_embed_text",),
                    ("text_embeddings_mxbai",),
                ]
            }
        )
        _patch_pool(monkeypatch, pool)

        messages: list[str] = []
        await ops.execute_purge(dict(_DB), embeddings_only=True, progress=messages.append)

        assert pool.matching("DROP TABLE IF EXISTS text_embeddings_nomic_embed_text CASCADE")
        assert pool.matching("DROP TABLE IF EXISTS text_embeddings_mxbai CASCADE")
        assert pool.matching(_UNRECORD_MIGRATION)
        # Entries survive an embeddings-only purge.
        assert not pool.matching("TRUNCATE enhanced_entries")
        assert not pool.matching("TRUNCATE ingestion_runs")
        assert "  Dropped text_embeddings_mxbai" in messages
        assert any("Embedding tables purged" in m for m in messages)
        assert pool.closed

    async def test_full_purge_truncates_drops_and_unrecords_migration(
        self, monkeypatch, fake_pool_factory
    ):
        pool = fake_pool_factory(
            rows_for={_SELECT_EMBEDDING_TABLES: [("text_embeddings_nomic_embed_text",)]}
        )
        _patch_pool(monkeypatch, pool)

        messages: list[str] = []
        await ops.execute_purge(dict(_DB), embeddings_only=False, progress=messages.append)

        assert pool.matching("TRUNCATE enhanced_entries CASCADE")
        assert pool.matching("TRUNCATE ingestion_runs CASCADE")
        assert pool.matching("DROP TABLE IF EXISTS text_embeddings_nomic_embed_text CASCADE")
        assert pool.matching(_UNRECORD_MIGRATION)
        assert any("All ARIEL data purged" in m for m in messages)
        # The full purge does not narrate individual drops.
        assert not any(m.startswith("  Dropped ") for m in messages)
        assert pool.closed

    async def test_full_purge_unrecords_migration_even_with_no_embedding_tables(
        self, monkeypatch, fake_pool
    ):
        _patch_pool(monkeypatch, fake_pool)

        await ops.execute_purge(dict(_DB), embeddings_only=False, progress=None)

        assert not fake_pool.matching("DROP TABLE IF EXISTS")
        assert fake_pool.matching(_UNRECORD_MIGRATION)

    async def test_embeddings_only_unrecords_migration_with_no_tables_present(
        self, monkeypatch, fake_pool
    ):
        _patch_pool(monkeypatch, fake_pool)

        await ops.execute_purge(dict(_DB), embeddings_only=True, progress=None)

        assert not fake_pool.matching("DROP TABLE IF EXISTS")
        assert fake_pool.matching(_UNRECORD_MIGRATION)

    async def test_pool_is_closed_when_a_drop_fails(self, monkeypatch, fake_pool_factory):
        pool = fake_pool_factory(
            rows_for={
                _SELECT_EMBEDDING_TABLES: [("text_embeddings_nomic_embed_text",)],
                "DROP TABLE IF EXISTS": RuntimeError("permission denied"),
            }
        )
        _patch_pool(monkeypatch, pool)

        with pytest.raises(RuntimeError, match="permission denied"):
            await ops.execute_purge(dict(_DB), embeddings_only=True)

        assert pool.closed
        # The migration row is left recorded because the purge never finished.
        assert not pool.matching(_UNRECORD_MIGRATION)
