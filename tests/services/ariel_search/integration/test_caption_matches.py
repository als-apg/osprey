"""Caption matches against a real database.

``ARIELRepository.caption_matches`` answers which attachments of a set of
entries carry a caption that satisfies a keyword query: model captions read
from ``attachment_captions`` under one model id, and upstream captions read
from the ``attachments`` items joined to ``attachment_files`` by URL. The
keyword path passes its tsquery and pattern bodies; the hybrid path passes the
typed and flattened query text with a minimum lexeme coverage. Every database
test runs on a fresh scratch database.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import psycopg
import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import (
    ARIELRepository,
    SchemaFacts,
    named_tsquery,
)
from osprey.services.ariel_search.database.search_fts import build_expanded_tsquery
from osprey.services.ariel_search.search.base import ExpansionGroup, QueryExpansion
from osprey.services.ariel_search.search.keyword import build_tsquery, parse_keyword_query

from .conftest import RecordingPool

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

MODEL = "vis-a"
OTHER_MODEL = "vis-b"


@pytest.fixture
async def repo(scratch_database: str) -> AsyncIterator[ARIELRepository]:
    """A repository over a freshly migrated scratch database."""
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    config = ARIELConfig.from_dict({"database": {"uri": scratch_database}})
    pool = await create_connection_pool(config.database)
    try:
        await run_migrations(pool, config)
        yield ARIELRepository(pool, config)
    finally:
        await pool.close()


def _seed(
    uri: str,
    entry_id: str,
    *,
    captions: dict[str, Any] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    files: list[tuple[str, str]] = (),  # type: ignore[assignment]
) -> None:
    """Insert one entry, its upstream attachment items, and its attachment rows.

    Args:
        uri: Scratch database URI.
        entry_id: The entry.
        captions: The ``attachment_captions`` object, or None for SQL NULL.
        attachments: The ``attachments`` items (``url``, ``caption`` …).
        files: ``(attachment_id, source_url)`` rows for ``attachment_files``.
    """
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, raw_text, attachments,
                attachment_captions
            ) VALUES (%s, 'test', NOW(), 'text', %s::jsonb, %s::jsonb)
            """,
            (
                entry_id,
                json.dumps(attachments or []),
                None if captions is None else json.dumps(captions),
            ),
        )
        for attachment_id, source_url in files:
            conn.execute(
                """
                INSERT INTO attachment_files (
                    attachment_id, entry_id, filename, source_url, copy_status
                ) VALUES (%s, %s, 'pic.png', %s, 'pending')
                """,
                (attachment_id, entry_id, source_url),
            )


def _caption(text: str, visible_text: str = "") -> dict[str, str]:
    return {"caption": text, "visible_text": visible_text}


def _tsquery(query: str) -> tuple[str, list[str]]:
    """The keyword path's unexpanded tsquery and its bind values for `query`."""
    parsed = parse_keyword_query(query)
    params = ([parsed.search_text] if parsed.search_text.strip() else []) + list(parsed.phrases)
    return build_tsquery(parsed.search_text, list(parsed.phrases)), params


# ---------------------------------------------------------------------------
# named_tsquery (pure)
# ---------------------------------------------------------------------------


async def test_named_tsquery_rewrites_build_tsquery_output() -> None:
    sql, params = _tsquery('beam dump "orbit feedback"')
    named, bound = named_tsquery(sql, params)
    assert "%s" not in named
    assert named.count("%(tq0)s") == 1 and named.count("%(tq1)s") == 1
    assert bound == {"tq0": params[0], "tq1": params[1]}


async def test_named_tsquery_numbers_every_expanded_alternative() -> None:
    parsed = parse_keyword_query("orbit at BPM")
    expansion = QueryExpansion(
        groups=(
            ExpansionGroup(
                original="bpm", alternatives=("beam position monitor", "beam position monitors")
            ),
        ),
        flattened_text="orbit at bpm beam position monitor beam position monitors",
    )
    sql, params = build_expanded_tsquery(parsed, expansion)
    named, bound = named_tsquery(sql, params)
    assert "%s" not in named
    assert list(bound) == [f"tq{i}" for i in range(len(params))]
    assert list(bound.values()) == params
    assert len(params) == 4


async def test_named_tsquery_leaves_escaped_percent_and_takes_a_prefix() -> None:
    named, bound = named_tsquery("a ~* '%%s' OR b ~* %s OR c LIKE '%%%s'", ["x", "y"], prefix="pat")
    assert named == "a ~* '%%s' OR b ~* %(pat0)s OR c LIKE '%%%(pat1)s'"
    assert bound == {"pat0": "x", "pat1": "y"}


async def test_named_tsquery_refuses_a_placeholder_count_mismatch() -> None:
    with pytest.raises(ValueError):
        named_tsquery("plainto_tsquery('english', %s)", [])
    with pytest.raises(ValueError):
        named_tsquery("plainto_tsquery('english', %s)", ["a", "b"])


# ---------------------------------------------------------------------------
# Schema behind code (no database)
# ---------------------------------------------------------------------------


async def test_schema_behind_returns_empty_without_touching_the_database() -> None:
    pool = MagicMock()
    pool.connection.side_effect = AssertionError("database touched")
    repository = ARIELRepository(pool, ARIELConfig.from_dict({"database": {"uri": "x"}}))

    async def _facts() -> SchemaFacts:
        return SchemaFacts(has_v2_fts=True, has_copy_state=False)

    repository.schema_facts = _facts  # type: ignore[method-assign]
    sql, params = _tsquery("orbit")
    assert (
        await repository.caption_matches(["e1"], MODEL, tsquery_sql=sql, tsquery_params=params)
        == {}
    )
    assert (
        await repository.caption_matches(
            ["e1"], MODEL, query_original="orbit", query_flattened="orbit", min_fraction=0.5
        )
        == {}
    )
    pool.connection.assert_not_called()


# ---------------------------------------------------------------------------
# Keyword form: tsquery and pattern spans
# ---------------------------------------------------------------------------


async def test_glob_pattern_yields_the_captioned_attachment(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(
        scratch_database,
        "e1",
        captions={
            "att-1": {MODEL: _caption("QX-77B magnet current trend")},
            "att-2": {MODEL: _caption("a cat")},
        },
    )
    parsed = parse_keyword_query("QX-77*")
    bodies = tuple(span.body for span in parsed.pattern_spans)
    assert bodies, "the glob must parse to a pattern span"
    assert await repo.caption_matches(["e1"], MODEL, pattern_bodies=bodies) == {"e1": ["att-1"]}


async def test_visible_text_counts_as_caption_text(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(scratch_database, "e1", captions={"att-1": {MODEL: _caption("a screen", "QX-77B 12 A")}})
    bodies = tuple(span.body for span in parse_keyword_query("QX-77*").pattern_spans)
    assert await repo.caption_matches(["e1"], MODEL, pattern_bodies=bodies) == {"e1": ["att-1"]}


async def test_caption_beside_an_error_for_the_same_id_succeeds(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(
        scratch_database,
        "e1",
        captions={
            "att-1": {
                MODEL: _caption("orbit correction plot"),
                OTHER_MODEL: {"error": "timeout", "attempts": 2},
            },
            "att-2": {MODEL: {"error": "orbit failed", "attempts": 3}},
        },
    )
    sql, params = _tsquery("orbit")
    result = await repo.caption_matches(["e1"], MODEL, tsquery_sql=sql, tsquery_params=params)
    assert result == {"e1": ["att-1"]}


async def test_other_models_captions_do_not_match(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(scratch_database, "e1", captions={"att-1": {OTHER_MODEL: _caption("orbit plot")}})
    sql, params = _tsquery("orbit")
    assert await repo.caption_matches(["e1"], MODEL, tsquery_sql=sql, tsquery_params=params) == {}


async def test_stopword_query_matches_nothing(repo: ARIELRepository, scratch_database: str) -> None:
    _seed(
        scratch_database,
        "e1",
        captions={"att-1": {MODEL: _caption("what was it about the orbit")}},
        attachments=[{"url": "https://up/1.png", "caption": "what was it"}],
        files=[("att-up", "https://up/1.png")],
    )
    sql, params = _tsquery("what was it")
    assert await repo.caption_matches(["e1"], MODEL, tsquery_sql=sql, tsquery_params=params) == {}


async def test_vocabulary_expansion_matches_bpm_orbit_plot(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(scratch_database, "e1", captions={"att-1": {MODEL: _caption("BPM orbit plot")}})
    _seed(
        scratch_database,
        "e2",
        captions={"att-2": {MODEL: _caption("beam position monitor orbit plot")}},
    )
    _seed(scratch_database, "e3", captions={"att-3": {MODEL: _caption("orbit plot")}})
    parsed = parse_keyword_query("orbit at BPM")
    expansion = QueryExpansion(
        groups=(
            ExpansionGroup(
                original="bpm", alternatives=("beam position monitor", "beam position monitors")
            ),
        ),
        flattened_text="orbit at bpm beam position monitor beam position monitors",
    )
    sql, params = build_expanded_tsquery(parsed, expansion)
    result = await repo.caption_matches(
        ["e1", "e2", "e3"], MODEL, tsquery_sql=sql, tsquery_params=params
    )
    assert result == {"e1": ["att-1"], "e2": ["att-2"]}


async def test_tsquery_and_pattern_spans_are_or_ed(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(
        scratch_database,
        "e1",
        captions={
            "att-1": {MODEL: _caption("orbit plot")},
            "att-2": {MODEL: _caption("QX-77B trend")},
            "att-3": {MODEL: _caption("a cat")},
        },
    )
    sql, params = _tsquery("orbit")
    bodies = tuple(span.body for span in parse_keyword_query("QX-77*").pattern_spans)
    await repo.schema_facts()  # cached, so only caption_matches' own statements are recorded
    pool = RecordingPool(repo.pool)
    repo.pool = pool  # type: ignore[assignment]
    result = await repo.caption_matches(
        ["e1"], MODEL, tsquery_sql=sql, tsquery_params=params, pattern_bodies=bodies
    )
    assert result == {"e1": ["att-1", "att-2"]}
    # One statement, under the pattern statement_timeout.
    selects = [s for s in pool.statements if "set_config" not in s]
    assert len(selects) == 1
    assert pool.opened_transaction
    assert pool.timeouts_inside and pool.timeouts_inside[0] not in ("0", "")


async def test_upstream_caption_joined_by_url(repo: ARIELRepository, scratch_database: str) -> None:
    _seed(
        scratch_database,
        "e1",
        attachments=[
            {"url": "https://up/1.png", "caption": "orbit drift overnight"},
            {"url": "https://up/2.png", "caption": "coffee machine"},
            {"url": "https://up/3.png", "caption": "orbit with no stored row"},
        ],
        files=[("att-up1", "https://up/1.png"), ("att-up2", "https://up/2.png")],
    )
    sql, params = _tsquery("orbit")
    assert await repo.caption_matches(["e1"], MODEL, tsquery_sql=sql, tsquery_params=params) == {
        "e1": ["att-up1"]
    }


async def test_no_model_id_omits_model_captions_but_keeps_upstream(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(
        scratch_database,
        "e1",
        captions={"att-1": {MODEL: _caption("orbit plot")}},
        attachments=[{"url": "https://up/1.png", "caption": "orbit drift"}],
        files=[("att-up1", "https://up/1.png")],
    )
    sql, params = _tsquery("orbit")
    assert await repo.caption_matches(["e1"], None, tsquery_sql=sql, tsquery_params=params) == {
        "e1": ["att-up1"]
    }
    assert await repo.caption_matches(["e1"], MODEL, tsquery_sql=sql, tsquery_params=params) == {
        "e1": ["att-1", "att-up1"]
    }


async def test_only_the_named_entries_are_searched(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(scratch_database, "e1", captions={"att-1": {MODEL: _caption("orbit plot")}})
    _seed(scratch_database, "e2", captions={"att-2": {MODEL: _caption("orbit plot")}})
    sql, params = _tsquery("orbit")
    assert await repo.caption_matches(["e2"], MODEL, tsquery_sql=sql, tsquery_params=params) == {
        "e2": ["att-2"]
    }
    assert await repo.caption_matches([], MODEL, tsquery_sql=sql, tsquery_params=params) == {}


async def test_null_captions_and_attachments_are_tolerated(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(scratch_database, "e1", captions=None)
    sql, params = _tsquery("orbit")
    assert await repo.caption_matches(["e1"], MODEL, tsquery_sql=sql, tsquery_params=params) == {}


# ---------------------------------------------------------------------------
# Hybrid form: lexeme coverage
# ---------------------------------------------------------------------------


async def test_coverage_half_matches_a_close_caption(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(
        scratch_database,
        "e1",
        captions={
            "att-1": {MODEL: _caption("orbit kick at BPM 7")},
            "att-2": {MODEL: _caption("vacuum pressure trend")},
        },
    )
    query = "orbit kick near BPM 7"
    result = await repo.caption_matches(
        ["e1"], MODEL, query_original=query, query_flattened=query, min_fraction=0.5
    )
    assert result == {"e1": ["att-1"]}


async def test_coverage_with_no_query_lexemes_matches_nothing(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(scratch_database, "e1", captions={"att-1": {MODEL: _caption("what was it")}})
    result = await repo.caption_matches(
        ["e1"],
        MODEL,
        query_original="what was it",
        query_flattened="what was it",
        min_fraction=0.5,
    )
    assert result == {}


async def test_coverage_counts_an_expansion_alternative_in_the_numerator(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(scratch_database, "e1", captions={"att-1": {MODEL: _caption("BPM drift trace")}})
    original = "beam position monitor drift"
    # n(q_orig) = 4 (beam, posit, monitor, drift): ceil(0.5 * 4) = 2 lexemes needed.
    unexpanded = await repo.caption_matches(
        ["e1"], MODEL, query_original=original, query_flattened=original, min_fraction=0.5
    )
    assert unexpanded == {}
    expanded = await repo.caption_matches(
        ["e1"],
        MODEL,
        query_original=original,
        query_flattened=f"{original} bpm bpms",
        min_fraction=0.5,
    )
    assert expanded == {"e1": ["att-1"]}


async def test_coverage_mode_emits_only_the_coverage_predicate(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(
        scratch_database,
        "e1",
        captions={
            "att-1": {MODEL: _caption("orbit kick at BPM 7")},
            "att-2": {MODEL: _caption("QX-77B trend")},
        },
    )
    sql, params = _tsquery("QX")
    bodies = tuple(span.body for span in parse_keyword_query("QX-77*").pattern_spans)
    await repo.schema_facts()  # cached, so only caption_matches' own statements are recorded
    pool = RecordingPool(repo.pool)
    repo.pool = pool  # type: ignore[assignment]
    query = "orbit kick near BPM 7"
    result = await repo.caption_matches(
        ["e1"],
        MODEL,
        tsquery_sql=sql,
        tsquery_params=params,
        pattern_bodies=bodies,
        query_original=query,
        query_flattened=query,
        min_fraction=0.5,
    )
    assert result == {"e1": ["att-1"]}
    selects = [s for s in pool.statements if "set_config" not in s]
    assert len(selects) == 1
    assert "@@" not in selects[0] and "~*" not in selects[0]
    assert "INTERSECT" in selects[0]


async def test_coverage_applies_to_upstream_captions(
    repo: ARIELRepository, scratch_database: str
) -> None:
    _seed(
        scratch_database,
        "e1",
        attachments=[{"url": "https://up/1.png", "caption": "orbit kick at BPM 7"}],
        files=[("att-up1", "https://up/1.png")],
    )
    query = "orbit kick near BPM 7"
    result = await repo.caption_matches(
        ["e1"], None, query_original=query, query_flattened=query, min_fraction=0.5
    )
    assert result == {"e1": ["att-up1"]}
