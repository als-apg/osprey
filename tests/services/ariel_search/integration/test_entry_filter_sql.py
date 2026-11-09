"""The SQL form of the filter agrees with its Python form on real Postgres.

Every row of the shared case table (``MATCH_CASES`` in ``test_filters.py``)
is seeded as one ``enhanced_entries`` row -- a row with ``has_images`` gets a
real, viewable ``attachment_files`` row, never a JSON key -- and its filter is
run through :meth:`EntryFilter.to_sql`. The SQL must select the row exactly
when :meth:`EntryFilter.matches` accepts the entry. All rows share one table,
so a guard that only works on a lone row (the ``tags`` array check) is
exercised next to rows it must not trip over.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg
import pytest

from osprey.services.ariel_search.filters import EntryFilter
from tests.services.ariel_search.integration._support import insert_picture
from tests.services.ariel_search.test_filters import FILTER_CTX, FILTER_DIMS, MATCH_CASES

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(120)]

#: Rows the schema cannot hold, so parity is pinned by the Python form only.
#: ``timestamp`` is ``TIMESTAMPTZ NOT NULL``: no stored entry lacks one.
UNSEEDABLE = {"time-bound-and-no-timestamp"}


def _cases() -> list[tuple[str, Any, dict[str, Any], bool]]:
    return [(p.id, *p.values) for p in MATCH_CASES]


def _seed(uri: str, entry_id: str, entry: dict[str, Any]) -> None:
    status = entry.get("enhancement_status")
    if not isinstance(status, str):
        status = json.dumps(status or {})
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, author, raw_text, metadata,
                enhancement_status
            ) VALUES (%s, %s, %s, %s, 'text', %s::jsonb, %s::jsonb)
            """,
            (
                entry_id,
                entry["source_system"],
                entry["timestamp"],
                # author is NOT NULL DEFAULT ''; '' matches no non-empty value,
                # as None does in the Python form
                entry["author"] if entry["author"] is not None else "",
                json.dumps(entry["metadata"]),
                status,
            ),
        )
    if entry.get("has_images") is True:
        insert_picture(uri, f"a-{entry_id}", entry_id, copy_status="copied")


async def test_the_case_table_covers_the_unseedable_rows_it_names() -> None:
    """A renamed case must not silently drop out of (or into) the exclusion."""
    assert UNSEEDABLE <= {case_id for case_id, *_ in _cases()}


@pytest.mark.usefixtures("scratch_repo")  # brings the scratch database to today's schema
async def test_to_sql_selects_exactly_what_matches_accepts(scratch_database: str) -> None:
    cases = [c for c in _cases() if c[0] not in UNSEEDABLE]
    for case_id, _raw, entry, _expected in cases:
        _seed(scratch_database, case_id, entry)

    mismatches: dict[str, dict[str, bool]] = {}
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        for case_id, raw, entry, expected in cases:
            flt = EntryFilter.parse(raw, FILTER_DIMS)
            python = flt.matches(entry, FILTER_CTX)
            clauses, params = flt.to_sql("e", FILTER_CTX)
            where = " AND ".join(["e.entry_id = %s", *clauses])
            row = conn.execute(
                f"SELECT e.entry_id FROM enhanced_entries e WHERE {where}",
                [case_id, *params],
            ).fetchone()
            sql = row is not None
            if not (python is sql is expected):
                mismatches[case_id] = {"expected": expected, "python": python, "sql": sql}

    assert mismatches == {}


@pytest.mark.usefixtures("scratch_repo")  # brings the scratch database to today's schema
async def test_the_whole_table_under_each_filter(scratch_database: str) -> None:
    """Each case's filter over every seeded row, not just its own: same set both ways."""
    cases = [c for c in _cases() if c[0] not in UNSEEDABLE]
    entries = {case_id: entry for case_id, _raw, entry, _expected in cases}
    for case_id, entry in entries.items():
        _seed(scratch_database, case_id, entry)

    mismatches: dict[str, dict[str, list[str]]] = {}
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        for case_id, raw, _entry, _expected in cases:
            flt = EntryFilter.parse(raw, FILTER_DIMS)
            python = sorted(i for i, e in entries.items() if flt.matches(e, FILTER_CTX))
            clauses, params = flt.to_sql("e", FILTER_CTX)
            where = " AND ".join(clauses) or "TRUE"
            rows = conn.execute(
                f"SELECT e.entry_id FROM enhanced_entries e WHERE {where} ORDER BY 1", params
            ).fetchall()
            sql = sorted(r[0] for r in rows)
            if python != sql:
                mismatches[case_id] = {"python": python, "sql": sql}

    assert mismatches == {}
