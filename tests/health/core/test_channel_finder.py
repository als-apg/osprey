"""Tests for the core ``channel_finder`` health category.

Exercises the presence gate (a top-level ``channel_finder`` block), the
pipeline row, the pipeline database file checks (present / missing / empty /
unconfigured), the informational freshness row, and the middle-layer-only DuckDB
channel count against a tiny real DuckDB fixture. Skips the DuckDB-count cases
when the ``duckdb`` package is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.health.core.channel_finder import channel_finder
from osprey.health.models import CheckResult, Status


def _cf(
    *,
    mode: str | None = "hierarchical",
    path: str | None = None,
    duckdb_path: str | None = None,
    with_pipeline_block: bool = True,
) -> dict:
    """Build a config with a top-level ``channel_finder`` block."""
    cf: dict = {}
    if mode is not None:
        cf["pipeline_mode"] = mode
    if with_pipeline_block and mode is not None:
        database: dict = {}
        if path is not None:
            database["path"] = path
        if duckdb_path is not None:
            database["duckdb_path"] = duckdb_path
        cf["pipelines"] = {mode: {"database": database}}
    return {"channel_finder": cf}


async def _run(config, *, cwd: Path | None = None) -> dict[str, CheckResult]:
    results = await channel_finder(config, cwd=cwd)()
    assert isinstance(results, list)
    return {r.name: r for r in results}


def _write_json_db(path: Path, content: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_duckdb(path: Path, channels: list[str]) -> None:
    import duckdb

    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE TABLE channels (channel_name TEXT PRIMARY KEY, system TEXT)")
        if channels:
            con.executemany("INSERT INTO channels VALUES (?, ?)", [(c, "SR") for c in channels])
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Presence gate
# --------------------------------------------------------------------------- #


async def test_no_rows_when_no_block() -> None:
    assert await _run({"deployment": {"bind_address": "127.0.0.1"}}) == {}


async def test_no_rows_when_block_empty() -> None:
    assert await _run({"channel_finder": {}}) == {}


async def test_no_rows_when_config_none() -> None:
    assert await _run(None) == {}


# --------------------------------------------------------------------------- #
# Pipeline row
# --------------------------------------------------------------------------- #


async def test_pipeline_row_reports_mode(tmp_path) -> None:
    db = tmp_path / "hierarchical.json"
    _write_json_db(db)
    by_name = await _run(_cf(mode="hierarchical", path=str(db)))
    row = by_name["channel_finder_pipeline"]
    assert row.status is Status.OK
    assert row.value == "hierarchical"


async def test_pipeline_row_warns_when_mode_unset() -> None:
    by_name = await _run({"channel_finder": {"pipelines": {}}})
    assert by_name["channel_finder_pipeline"].status is Status.WARNING


async def test_pipeline_row_warns_when_block_absent() -> None:
    by_name = await _run(_cf(mode="middle_layer", with_pipeline_block=False))
    row = by_name["channel_finder_pipeline"]
    assert row.status is Status.WARNING
    assert row.value == "middle_layer"


# --------------------------------------------------------------------------- #
# Database row + freshness
# --------------------------------------------------------------------------- #


async def test_fresh_hierarchical_build_has_no_duckdb_warning(tmp_path) -> None:
    # A fresh control-assistant defaults to the hierarchical pipeline, which
    # ships a JSON database and no DuckDB — the tile must not warn about DuckDB.
    db = tmp_path / "data" / "channel_databases" / "hierarchical.json"
    _write_json_db(db, '{"SR": {}}')
    by_name = await _run(_cf(mode="hierarchical", path=str(db)))
    assert set(by_name) == {
        "channel_finder_pipeline",
        "channel_finder_database",
        "channel_finder_freshness",
    }
    assert by_name["channel_finder_database"].status is Status.OK
    assert by_name["channel_finder_freshness"].status is Status.OK
    assert by_name["channel_finder_freshness"].value.startswith("built ")
    assert "channel_finder_channels" not in by_name  # not middle_layer


async def test_missing_database_path_is_error(tmp_path) -> None:
    missing = tmp_path / "nope.json"
    by_name = await _run(_cf(mode="hierarchical", path=str(missing)))
    assert by_name["channel_finder_database"].status is Status.ERROR
    assert "channel_finder_freshness" not in by_name


async def test_empty_database_file_is_error(tmp_path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    by_name = await _run(_cf(mode="hierarchical", path=str(empty)))
    assert by_name["channel_finder_database"].status is Status.ERROR


async def test_no_database_path_configured_warns() -> None:
    by_name = await _run(_cf(mode="hierarchical", path=None))
    assert by_name["channel_finder_database"].status is Status.WARNING


async def test_relative_database_path_resolved_against_cwd(tmp_path) -> None:
    rel = "data/channel_databases/hierarchical.json"
    _write_json_db(tmp_path / rel, '{"SR": {}}')
    by_name = await _run(_cf(mode="hierarchical", path=rel), cwd=tmp_path)
    assert by_name["channel_finder_database"].status is Status.OK


# --------------------------------------------------------------------------- #
# Channel count (middle_layer + duckdb only)
# --------------------------------------------------------------------------- #


async def test_no_channels_row_without_duckdb_path(tmp_path) -> None:
    db = tmp_path / "middle_layer.json"
    _write_json_db(db)
    by_name = await _run(_cf(mode="middle_layer", path=str(db)))
    assert "channel_finder_channels" not in by_name


class TestDuckDBCount:
    """DuckDB-backed channel counting (skipped when duckdb is unavailable)."""

    @pytest.fixture(autouse=True)
    def _require_duckdb(self):
        pytest.importorskip("duckdb")

    async def test_counts_channels(self, tmp_path) -> None:
        js = tmp_path / "middle_layer.json"
        _write_json_db(js)
        duck = tmp_path / "middle_layer.duckdb"
        _make_duckdb(duck, ["SR:BPM1:X", "SR:BPM1:Y", "SR:HCM1:Setpoint"])
        by_name = await _run(_cf(mode="middle_layer", path=str(js), duckdb_path=str(duck)))
        channels = by_name["channel_finder_channels"]
        assert channels.status is Status.OK
        assert channels.value == "3 channels"

    async def test_zero_channels_warns(self, tmp_path) -> None:
        js = tmp_path / "middle_layer.json"
        _write_json_db(js)
        duck = tmp_path / "empty.duckdb"
        _make_duckdb(duck, [])
        by_name = await _run(_cf(mode="middle_layer", path=str(js), duckdb_path=str(duck)))
        assert by_name["channel_finder_channels"].status is Status.WARNING

    async def test_unreadable_duckdb_degrades_to_warning(self, tmp_path) -> None:
        js = tmp_path / "middle_layer.json"
        _write_json_db(js)
        garbage = tmp_path / "garbage.duckdb"
        garbage.write_bytes(b"this is not a duckdb file at all")
        by_name = await _run(_cf(mode="middle_layer", path=str(js), duckdb_path=str(garbage)))
        assert by_name["channel_finder_channels"].status is Status.WARNING

    async def test_relative_duckdb_path_resolved_against_cwd(self, tmp_path) -> None:
        _write_json_db(tmp_path / "data" / "middle_layer.json")
        _make_duckdb(tmp_path / "data" / "middle_layer.duckdb", ["SR:BPM1:X"])
        by_name = await _run(
            _cf(
                mode="middle_layer",
                path="data/middle_layer.json",
                duckdb_path="data/middle_layer.duckdb",
            ),
            cwd=tmp_path,
        )
        assert by_name["channel_finder_channels"].value == "1 channels"
