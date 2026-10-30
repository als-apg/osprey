"""The hello-world tutorial's archive query runs on what the record exports.

The tutorial shows ``hello-world-tutorial.sql`` and tells the reader to run it
with DuckDB from the repo root, over the days the record has exported to
``var/archive/<day>/agent_record/entries.parquet``. It promises one row per
identity and kind, with the number of entries, the first and the last entry
time, the most frequent first; and, on the first day, that DuckDB reports no
file matches. The query runs here unchanged over a real export.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import duckdb
import pytest

from osprey.agent_record import archive
from osprey.agent_record import contract as c
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.utils.workspace import ARCHIVE_DIR_RELPATH
from tests.agent_record.contract_suite import _kit as k

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL = REPO_ROOT / "docs/source/getting-started/hello-world-tutorial.sql"

COLUMNS = ["identity", "kind", "entries", "first_seen", "last_seen"]


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    root = project / ARCHIVE_DIR_RELPATH
    root.mkdir(parents=True)
    os.chmod(root, 0o750)
    return project, root


def _chunks(identity: str, key: str, kind: str, t: float) -> list[c.Entry]:
    """One entry whose body was split in two: two rows, one entry."""
    return [
        k.entry(identity, key, kind, t, {c.CHUNK_PART_FIELD: part}, chunk=c.Chunk(i, 2))
        for i, part in enumerate(('{"te', 'xt":1}'))
    ]


def test_hello_world_tutorial_sql_counts_each_identity_and_kind(tmp_path, monkeypatch):
    project, root = _project(tmp_path)
    with EmbeddedBackend(tmp_path / "record.duckdb") as backend:
        backend.append(
            [
                k.cc(k.ALICE, "s1", 1, c.KIND_FEEDBACK, 10),
                k.cc(k.ALICE, "s1", 2, c.KIND_FEEDBACK, 20),
                *_chunks(k.ALICE, "sp:alice:big", c.KIND_FEEDBACK, 30),
                k.cc(k.ALICE, "s1", 3, c.KIND_RUN, 40),
                k.cc(k.BOB, "s2", 1, c.KIND_FEEDBACK, 50),
                k.cc(k.BOB, "s2", 2, c.KIND_FEEDBACK, 60),
                k.cc(k.BOB, "s2", 3, c.KIND_FEEDBACK, 70),
                k.cc(k.W1, "s3", 1, c.KIND_RUN, 80),
            ]
        )
        # The day after: still open, so it is not exported and not counted.
        backend.append([k.cc(k.ALICE, "s4", 1, c.KIND_FEEDBACK, 24 * 3600 + 5)])

        assert archive.run_export(backend, k.T0 + timedelta(days=1), root) == [k.T0.date()]

    monkeypatch.chdir(project)
    result = duckdb.sql(SQL.read_text())

    assert result.columns == COLUMNS
    assert result.fetchall() == [
        (k.ALICE, c.KIND_FEEDBACK, 3, k.at(10), k.at(30)),
        (k.BOB, c.KIND_FEEDBACK, 3, k.at(50), k.at(70)),
        (k.ALICE, c.KIND_RUN, 1, k.at(40), k.at(40)),
        (k.W1, c.KIND_RUN, 1, k.at(80), k.at(80)),
    ]


def test_hello_world_tutorial_sql_reports_no_file_on_the_first_day(tmp_path, monkeypatch):
    project, _ = _project(tmp_path)
    monkeypatch.chdir(project)

    with pytest.raises(duckdb.IOException, match="No files found"):
        duckdb.sql(SQL.read_text()).fetchall()
