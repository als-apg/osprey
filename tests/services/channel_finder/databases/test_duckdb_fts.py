"""Tests for the shared DuckDB FTS extension helper.

The regression scenario: a machine (fresh CI runner, new user install) that has
never installed the FTS extension. A bare ``LOAD fts`` fails there — ensure_fts
must fall back to installing before loading.
"""

import duckdb
import pytest

from osprey.services.channel_finder.databases import duckdb_fts
from osprey.services.channel_finder.databases.duckdb_fts import ensure_fts


class FakeConnection:
    """Records executed SQL; optionally fails the first LOAD."""

    def __init__(self, fail_first_load: bool = False):
        self.fail_first_load = fail_first_load
        self.executed: list[str] = []

    def execute(self, sql: str):
        self.executed.append(sql)
        if sql == "LOAD fts" and self.fail_first_load and self.executed.count("LOAD fts") == 1:
            raise duckdb.IOException("Extension fts.duckdb_extension not found")


def test_load_succeeds_without_install():
    con = FakeConnection()
    ensure_fts(con)
    assert con.executed == ["LOAD fts"]


def test_missing_extension_installs_then_loads():
    con = FakeConnection(fail_first_load=True)
    ensure_fts(con)
    assert con.executed == ["LOAD fts", "INSTALL fts", "LOAD fts"]


def test_missing_extension_prefers_bundled_file(tmp_path, monkeypatch):
    bundled = tmp_path / "fts.duckdb_extension"
    bundled.write_bytes(b"")
    monkeypatch.setattr(duckdb_fts, "_BUNDLED_FTS", bundled)
    con = FakeConnection(fail_first_load=True)
    ensure_fts(con)
    assert con.executed == ["LOAD fts", f"INSTALL '{bundled.resolve()}'", "LOAD fts"]


def test_missing_extension_applies_proxy_env(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")
    con = FakeConnection(fail_first_load=True)
    ensure_fts(con)
    assert "SET http_proxy = 'http://proxy.example:3128'" in con.executed
    assert con.executed[-2:] == ["INSTALL fts", "LOAD fts"]


def test_install_failure_propagates():
    class AlwaysFailing(FakeConnection):
        def execute(self, sql: str):
            self.executed.append(sql)
            raise duckdb.IOException("no network")

    with pytest.raises(duckdb.Error):
        ensure_fts(AlwaysFailing())
