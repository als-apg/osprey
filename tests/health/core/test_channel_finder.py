"""Tests for the core ``channel_finder`` health category.

Exercises the presence gate (a top-level ``channel_finder`` block), the
pipeline row, the rejection of a pipeline mode that is not a real paradigm, the
pipeline database file checks (present / missing / empty / unconfigured), the
informational freshness row, the middle-layer-only DuckDB channel count against
a tiny real DuckDB fixture, and the graph pipeline's store-backed rows against a
fake neo4j driver. Skips the DuckDB-count cases when the ``duckdb`` package is
unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.health.core.channel_finder import channel_finder
from osprey.health.models import CheckResult, Status
from osprey.services.channel_finder.core.exceptions import PipelineModeError


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


async def test_unknown_pipeline_mode_raises() -> None:
    """A mode that is not a real paradigm is a config defect, not a warning row.

    A missing ``pipelines.<mode>`` block warns because a paradigm can legitimately
    be configured before its block is filled in. A mode name that no paradigm
    answers to is different: nothing downstream can serve it, so the category
    refuses to report on it rather than emitting a row that reads as a mere gap.
    """
    with pytest.raises(PipelineModeError) as excinfo:
        await _run(_cf(mode="quantum"))
    message = str(excinfo.value)
    assert "quantum" in message
    for mode in VALID_CHANNEL_FINDER_MODES:
        assert mode in message


@pytest.mark.parametrize("mode", VALID_CHANNEL_FINDER_MODES)
async def test_every_valid_mode_reports_a_row(mode: str, tmp_path) -> None:
    db = tmp_path / f"{mode}.json"
    _write_json_db(db, '{"SR": {}}')
    by_name = await _run(_cf(mode=mode, path=str(db)))
    row = by_name["channel_finder_pipeline"]
    assert row.status is Status.OK
    # Every file-backed paradigm reports its own name; the graph paradigm reports
    # the name plus where it answers from, since it reads no database file.
    assert row.value == ("graph (store-backed)" if mode == "graph" else mode)


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


async def test_every_database_row_names_the_one_file_it_stats(tmp_path) -> None:
    """The row stats ``database.path`` and nothing else, so it must say so.

    A middle-layer deployment holds a second channel database — the DuckDB
    ``channel_finder_channels`` reports on — and a row saying "channel database
    present" while stat-ing only the pipeline's own file reads as a verdict on
    both. It would be vouching for an artifact it never opened, next to a row
    correctly reporting that artifact absent.
    """
    present = tmp_path / "hierarchical.json"
    _write_json_db(present, '{"SR": {}}')
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    missing = tmp_path / "nope.json"

    ok = (await _run(_cf(path=str(present))))["channel_finder_database"]
    assert ok.message == "Active pipeline's channel database file present"

    fresh = (await _run(_cf(path=str(present))))["channel_finder_freshness"]
    assert fresh.message == "Active pipeline's channel database file build age"

    assert (await _run(_cf(path=str(empty))))[
        "channel_finder_database"
    ].message == f"Active pipeline's channel database file is empty ({empty})"

    assert (await _run(_cf(path=str(missing))))[
        "channel_finder_database"
    ].message == f"Active pipeline's channel database file missing at {missing}"

    unset = (await _run(_cf(path=None)))["channel_finder_database"]
    assert unset.message == "No database.path configured for the active pipeline"


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


# --------------------------------------------------------------------------- #
# Graph pipeline (store-backed rows)
# --------------------------------------------------------------------------- #


class _FakeRecord(dict):
    """A driver record: mapping access by result key."""


class _FakeEagerResult:
    """The ``EagerResult`` shape ``Driver.execute_query`` returns."""

    def __init__(self, records: list[_FakeRecord]) -> None:
        self.records = records


class _FakeDriver:
    """Answers the ping and the ``(:Resource)`` count from a script."""

    def __init__(self, *, count: int = 7, connect_error: Exception | None = None) -> None:
        self.count = count
        self.connect_error = connect_error
        self.queries: list[str] = []

    def execute_query(self, query: str, *args, **kwargs) -> _FakeEagerResult:
        self.queries.append(query)
        if len(self.queries) == 1:
            if self.connect_error is not None:
                raise self.connect_error
            return _FakeEagerResult([_FakeRecord(ok=1)])
        return _FakeEagerResult([_FakeRecord(count=self.count)])

    def close(self) -> None:
        pass


class TestGraphPipeline:
    """The ``graph`` paradigm answers from the store, so it reads store rows.

    Nothing here touches a real store: the neo4j driver factory is replaced, so
    the cases are hermetic and the "unreachable" case is instantaneous.
    """

    @pytest.fixture(autouse=True)
    def _no_ambient_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep the resolver off whatever ``GRAPHDB_PASSWORD`` the host carries."""
        monkeypatch.delenv("GRAPHDB_PASSWORD", raising=False)

    @staticmethod
    def _install_driver(monkeypatch: pytest.MonkeyPatch, driver: _FakeDriver) -> list[tuple]:
        """Point ``GraphDatabase.driver`` at ``driver``; return captured call args."""
        import neo4j

        calls: list[tuple] = []

        def _factory(uri, *, auth=None, **config):
            calls.append((uri, auth))
            return driver

        monkeypatch.setattr(neo4j.GraphDatabase, "driver", _factory)
        return calls

    @staticmethod
    def _cfg(*, pipelines: dict | None = None, **graphdb_block) -> dict:
        """Graph-mode config: no ``pipelines.graph`` block unless one is asked for."""
        cf: dict = {"pipeline_mode": "graph"}
        if pipelines is not None:
            cf["pipelines"] = pipelines
        block = {"path": "./services/graphdb"}
        block.update(graphdb_block)
        return {"channel_finder": cf, "services": {"graphdb": block}}

    async def test_reports_pipeline_reachability_and_resources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_driver(monkeypatch, _FakeDriver(count=2908))
        by_name = await _run(self._cfg())
        assert set(by_name) == {
            "channel_finder_pipeline",
            "channel_finder_store",
            "channel_finder_resources",
        }
        assert by_name["channel_finder_pipeline"].status is Status.OK
        assert by_name["channel_finder_pipeline"].value == "graph (store-backed)"
        assert by_name["channel_finder_store"].status is Status.OK
        assert by_name["channel_finder_resources"].status is Status.OK
        assert by_name["channel_finder_resources"].value == "2,908 Resource nodes"

    async def test_needs_no_pipelines_block_and_never_warns_about_a_database_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A graph build configures no ``database.path``, so none may be missed.

        The pipeline row is derived from the mode alone: unlike the file-backed
        paradigms, ``graph`` has nothing to put in a ``pipelines.graph`` block,
        so requiring one would make every correct graph build warn.
        """
        self._install_driver(monkeypatch, _FakeDriver())
        by_name = await _run(self._cfg())
        assert "channel_finder_database" not in by_name
        assert "channel_finder_freshness" not in by_name
        assert by_name["channel_finder_pipeline"].status is Status.OK
        assert not any("database.path" in r.message for r in by_name.values())

    async def test_rows_belong_to_the_channel_finder_category(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Restated readings, not the graphdb category's rows leaking sideways.

        The ``graphdb`` tile reports the same store next to this one. These rows
        answer a different question — whether channel search has anything to
        answer from — so they carry this category's name and row ids.
        """
        self._install_driver(monkeypatch, _FakeDriver())
        by_name = await _run(self._cfg())
        assert all(r.category == "channel_finder" for r in by_name.values())
        assert not any(name.startswith("graphdb_") for name in by_name)

    async def test_unreachable_store_warns_and_never_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from neo4j.exceptions import ServiceUnavailable

        self._install_driver(
            monkeypatch, _FakeDriver(connect_error=ServiceUnavailable("Connection refused"))
        )
        by_name = await _run(self._cfg())
        assert set(by_name) == {"channel_finder_pipeline", "channel_finder_store"}
        row = by_name["channel_finder_store"]
        assert row.status is Status.WARNING
        assert "unreachable" in row.message
        assert row.details
        assert all(r.status is not Status.ERROR for r in by_name.values())

    async def test_empty_store_warns_and_names_the_seed_verb(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_driver(monkeypatch, _FakeDriver(count=0))
        row = (await _run(self._cfg()))["channel_finder_resources"]
        assert row.status is Status.WARNING
        assert "osprey knowledge seed-graph" in f"{row.message} {row.details}"

    async def test_external_store_is_dialed_at_its_own_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A facility-hosted store is resolved exactly as the graphdb tile does."""
        calls = self._install_driver(monkeypatch, _FakeDriver())
        monkeypatch.setenv("GRAPHDB_PASSWORD", "s3cret")
        await _run(self._cfg(uri="bolt://graph.example.org:7687", username="reader"))
        assert calls == [("bolt://graph.example.org:7687", ("reader", "s3cret"))]

    async def test_deployed_store_is_dialed_on_the_published_bolt_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._install_driver(monkeypatch, _FakeDriver())
        await _run(self._cfg(port_host=17687))
        assert calls[0][0] == "bolt://localhost:17687"

    async def test_no_graphdb_block_warns_and_opens_no_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Graph mode with no store configured names the block that is missing."""
        calls = self._install_driver(monkeypatch, _FakeDriver())
        by_name = await _run({"channel_finder": {"pipeline_mode": "graph"}})
        assert set(by_name) == {"channel_finder_pipeline", "channel_finder_store"}
        row = by_name["channel_finder_store"]
        assert row.status is Status.WARNING
        assert "services.graphdb" in f"{row.message} {row.details}"
        assert calls == []

    async def test_a_stale_pipelines_block_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The store is the reading, whatever a hand-written block claims."""
        self._install_driver(monkeypatch, _FakeDriver())
        by_name = await _run(
            self._cfg(pipelines={"graph": {"database": {"path": "data/nope.json"}}})
        )
        assert set(by_name) == {
            "channel_finder_pipeline",
            "channel_finder_store",
            "channel_finder_resources",
        }
        assert by_name["channel_finder_pipeline"].value == "graph (store-backed)"
