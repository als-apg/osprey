"""``text_embedding`` build-key stale on a real store: one verdict on every surface.

One scratch database holds text-embedding rows built under a NULL key (read as
``r1``), the running key and two older keys, plus an entry with no row. The
per-entry view (:meth:`EnhancementContext.view` over one batched read), the
five-state SQL and ``osprey ariel status`` (``classify_build_keys``) must agree
on how many are stale; with an unusable vocabulary none is; and the owed-work
query never owes ``text_embedding`` for a build key.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from osprey.services.ariel_search.cli_operations import get_status
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.migrations import model_to_table_name
from osprey.services.ariel_search.enhancement.status import (
    enhancement_state_case_sql,
    store_enhancement_context,
)
from osprey.services.ariel_search.vocabulary.entry import classify_build_keys
from tests.services.ariel_search.integration._support import migrated_repository

# xdist_group("docker"): every container-starting test file shares one worker.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

MODEL = "fake-embed"
TABLE = model_to_table_name(MODEL)

#: entry id -> the stored ``input_key`` of its row (``...`` = no row at all).
ROWS: dict[str, Any] = {
    "bk-null": None,  # COALESCE -> r1, the running key: current
    "bk-current": "r1",
    "bk-expanded": "r1:e1:0123456789abcdef",  # built with expansion: stale
    "bk-rule": "r2",  # another input rule: stale
    "bk-no-row": ...,
}
STALE = {"bk-expanded", "bk-rule"}


def _config_dict(database_uri: str, vocabulary: dict[str, Any] | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "database": {"uri": database_uri},
        "search_modules": {"keyword": {"enabled": True}, "semantic": {"enabled": False}},
        "enhancement_modules": {
            "semantic_processor": {"enabled": False},
            "text_embedding": {
                "enabled": True,
                "provider": {"name": "fake", "base_url": "http://fake-embeddings.invalid:11434"},
                "models": [{"name": MODEL, "dimension": 4}],
            },
        },
    }
    if vocabulary is not None:
        raw["vocabulary"] = vocabulary
    return raw


async def _seed(repository, seed_entry_factory) -> None:
    for entry_id, key in ROWS.items():
        await repository.upsert_entry(
            seed_entry_factory(
                entry_id=entry_id,
                raw_text=f"build key {entry_id}",
                enhancement_status={"text_embedding": {"status": "complete"}},
            )
        )
        if key is ...:
            continue
        async with repository.pool.connection() as conn:
            await conn.execute(
                f"INSERT INTO {TABLE} (entry_id, input_key) VALUES (%s, %s)", [entry_id, key]
            )


async def _stale_by_sql(repository, ctx) -> set[str]:
    sql, params = ctx.state_sql("text_embedding", "stale", "e")
    async with repository.pool.connection() as conn:
        result = await conn.execute(
            f"SELECT e.entry_id FROM enhanced_entries e WHERE {sql}", params
        )
        return {row[0] for row in await result.fetchall()}


async def _states_by_case(repository, ctx) -> dict[str, str]:
    case, params = enhancement_state_case_sql(
        "text_embedding", "e", marker=ctx.text_key, text_table=ctx.text_table
    )
    async with repository.pool.connection() as conn:
        result = await conn.execute(f"SELECT e.entry_id, {case} FROM enhanced_entries e", params)
        return dict(await result.fetchall())


async def test_build_key_stale_agrees_with_classify_build_keys_on_every_surface(
    scratch_database: str, seed_entry_factory
):
    config_dict = _config_dict(scratch_database)
    config = ARIELConfig.from_dict(copy.deepcopy(config_dict))
    async with migrated_repository(config) as repository:
        assert await repository.has_input_key_column(TABLE), "migration made no text table"
        await _seed(repository, seed_entry_factory)

        counts = await repository.get_embedding_build_counts(TABLE)
        assert counts == {"r1": 2, "r1:e1:0123456789abcdef": 1, "r2": 1}
        classified = sum(classify_build_keys(counts, "r1").values())

        ids = list(ROWS)
        ctx = await store_enhancement_context(config, repository, entry_ids=ids)
        assert (ctx.text_table, ctx.text_key) == (TABLE, "r1")
        assert ctx.build_keys == {
            entry_id: ("r1" if key is None else key)
            for entry_id, key in ROWS.items()
            if key is not ...
        }
        entries = await repository.get_entries_by_ids(ids)
        view_stale = {e["entry_id"] for e in entries if ctx.view(e)["text_embedding"] == "stale"}

        assert view_stale == STALE
        assert len(view_stale) == classified == 2
        assert await _stale_by_sql(repository, ctx) == STALE
        states = await _states_by_case(repository, ctx)
        assert states == {e["entry_id"]: ctx.view(e)["text_embedding"] for e in entries}

        # Re-embedding stays with reembed: the owed-work query never owes it.
        owed = await repository.get_incomplete_entries("text_embedding")
        assert not {e["entry_id"] for e in owed} & set(ROWS)

    status = await get_status(copy.deepcopy(config_dict), config_dir=None)
    assert status["status"] != "error", status
    (table,) = [t for t in status["embedding_tables"] if t["table"] == TABLE]
    assert sum(table["stale"].values()) == len(view_stale)


async def test_build_key_unusable_vocabulary_gives_no_stale(
    scratch_database: str, seed_entry_factory
):
    unusable = _config_dict(scratch_database, {"enabled": True, "expand_entries": True})
    config = ARIELConfig.from_dict(copy.deepcopy(unusable))
    async with migrated_repository(config) as repository:
        await _seed(repository, seed_entry_factory)

        ctx = await store_enhancement_context(config, repository, entry_ids=list(ROWS))
        assert (ctx.text_table, ctx.text_key, dict(ctx.build_keys)) == (None, None, {})
        entries = await repository.get_entries_by_ids(list(ROWS))
        assert [e for e in entries if ctx.view(e)["text_embedding"] == "stale"] == []
        assert await _stale_by_sql(repository, ctx) == set()
        assert "stale" not in (await _states_by_case(repository, ctx)).values()
        owed = await repository.get_incomplete_entries("text_embedding")
        assert not {e["entry_id"] for e in owed} & set(ROWS)
