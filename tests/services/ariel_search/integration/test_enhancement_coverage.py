"""Enhancement coverage counts on a real store: one tally, the badge strip's states.

A scratch database holds entries in every state of every module -- complete,
pending (missing or stored pending), failed (status, gave up, attempts spent),
stale (an older marker, an older text build key) and ``n/a`` (a picture module
on a picture-less entry). :meth:`ARIELRepository.get_enhancement_coverage` must
give, per enabled module, counts that sum to the number of entries and equal a
tally of :meth:`EnhancementContext.view` over the same rows.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.migrations import model_to_table_name
from osprey.services.ariel_search.enhancement.status import (
    STATES,
    EnhancementContext,
    store_enhancement_context,
)
from tests.services.ariel_search.integration._support import insert_picture, migrated_repository

# xdist_group("docker"): every container-starting test file shares one worker.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

MODEL = "fake-embed"
TABLE = model_to_table_name(MODEL)
CAPTION_MODEL = "vis-b"
SEMANTIC_MARKER = "sp-2"

MODULES = (
    "semantic_processor",
    "text_embedding",
    "image_caption",
    "image_embedding",
    "qmd_export",
)

#: entry id -> (enhancement_status, has a viewable picture, text build key or ... for no row)
ROWS: dict[str, tuple[dict[str, Any], bool, Any]] = {
    "cov-empty": ({}, False, ...),
    "cov-empty-pic": ({}, True, ...),
    "cov-all-done": (
        {
            "semantic_processor": {"status": "complete", "marker": SEMANTIC_MARKER},
            "text_embedding": {"status": "complete"},
            "image_caption": {"status": "complete", "marker": CAPTION_MODEL},
            "image_embedding": {"status": "complete", "marker": "img-1"},
            "qmd_export": {"status": "complete"},
        },
        True,
        "r1",
    ),
    "cov-stale": (
        {
            "semantic_processor": {"status": "complete", "marker": "sp-1"},
            "text_embedding": {"status": "complete"},
            "image_caption": {"status": "complete", "marker": "vis-a"},
            "image_embedding": {"status": "complete", "marker": "img-0"},
            "qmd_export": {"status": "pending"},
        },
        True,
        "r2",
    ),
    "cov-failed": (
        {
            "semantic_processor": {"status": "failed", "attempts": 1},
            "text_embedding": {"status": "pending", "attempts": 99},
            "image_caption": {"status": "pending", "gave_up": True, "marker": CAPTION_MODEL},
            "image_embedding": {"status": "failed", "marker": "img-1"},
            "qmd_export": {"status": "failed"},
        },
        True,
        None,
    ),
    "cov-no-pic": (
        {
            "semantic_processor": {"status": "complete"},
            "text_embedding": {"status": "complete"},
            "image_caption": {"status": "complete", "marker": CAPTION_MODEL},
            "qmd_export": {"status": "complete"},
        },
        False,
        "r1:e1:0123456789abcdef",
    ),
}


def _config_dict(database_uri: str) -> dict[str, Any]:
    return {
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


async def _seed(repository, seed_entry_factory, uri: str) -> None:
    for entry_id, (status, picture, key) in ROWS.items():
        await repository.upsert_entry(
            seed_entry_factory(
                entry_id=entry_id,
                raw_text=f"coverage {entry_id}",
                enhancement_status=status,
            )
        )
        if picture:
            insert_picture(uri, f"{entry_id}-a", entry_id, copy_status="copied")
        if key is ...:
            continue
        async with repository.pool.connection() as conn:
            await conn.execute(
                f"INSERT INTO {TABLE} (entry_id, input_key) VALUES (%s, %s)", [entry_id, key]
            )


async def _context(config: ARIELConfig, repository) -> EnhancementContext:
    """The store's context with every built-in module enabled and a marker for each."""
    ctx = await store_enhancement_context(config, repository, entry_ids=list(ROWS))
    assert (ctx.text_table, ctx.text_key) == (TABLE, "r1"), "no text build-key verdict"
    return replace(
        ctx,
        enabled=MODULES,
        markers={
            **ctx.markers,
            "semantic_processor": SEMANTIC_MARKER,
            "image_caption": CAPTION_MODEL,
            "image_embedding": "img-1",
            "qmd_export": None,
        },
    )


def _tally(ctx: EnhancementContext, entries) -> dict[str, Counter]:
    tally: dict[str, Counter] = {module: Counter() for module in ctx.enabled}
    for entry in entries:
        for module, state in ctx.view(entry).items():
            tally[module][state] += 1
    return tally


async def test_coverage_counts_equal_a_tally_of_the_view(scratch_database: str, seed_entry_factory):
    config_dict = _config_dict(scratch_database)
    config = ARIELConfig.from_dict(copy.deepcopy(config_dict))
    async with migrated_repository(config) as repository:
        await _seed(repository, seed_entry_factory, scratch_database)
        ctx = await _context(config, repository)

        coverage = await repository.get_enhancement_coverage(ctx)

        assert list(coverage) == list(MODULES)
        entries = await repository.get_entries_by_ids(list(ROWS))
        assert len(entries) == len(ROWS)
        tally = _tally(ctx, entries)
        for module, counts in coverage.items():
            assert set(counts) == {"label", *STATES}
            assert isinstance(counts["label"], str) and counts["label"]
            assert sum(counts[s] for s in STATES) == len(ROWS), module
            assert {s: counts[s] for s in STATES} == {s: tally[module][s] for s in STATES}, module

        # Every state is exercised somewhere, so the equality is not vacuous.
        seen = {s for counts in coverage.values() for s in STATES if counts[s]}
        assert seen == set(STATES)
        assert coverage["text_embedding"]["stale"] == 2
        assert coverage["image_caption"]["n/a"] == 2
        assert coverage["semantic_processor"]["stale"] == 1


async def test_coverage_labels_and_order_follow_the_registry(
    scratch_database: str, seed_entry_factory
):
    from osprey.registry import get_registry

    config = ARIELConfig.from_dict(_config_dict(scratch_database))
    async with migrated_repository(config) as repository:
        await _seed(repository, seed_entry_factory, scratch_database)
        ctx = await _context(config, repository)
        order = [m for m in get_registry().list_ariel_enhancement_modules() if m in MODULES]
        ctx = replace(ctx, enabled=tuple(order))

        coverage = await repository.get_enhancement_coverage(ctx)

        assert list(coverage) == order
        for module in order:
            _cls, reg = get_registry().get_ariel_enhancement_module(module)
            assert coverage[module]["label"] == (reg.label or module)


async def test_coverage_on_an_empty_store_lists_every_module_at_zero(scratch_repo):
    ctx = EnhancementContext(enabled=("semantic_processor", "image_caption"))

    coverage = await scratch_repo.get_enhancement_coverage(ctx)

    assert list(coverage) == ["semantic_processor", "image_caption"]
    for counts in coverage.values():
        assert {s: counts[s] for s in STATES} == dict.fromkeys(STATES, 0)


async def test_coverage_with_no_enabled_module_is_empty(scratch_repo):
    assert await scratch_repo.get_enhancement_coverage(EnhancementContext()) == {}
