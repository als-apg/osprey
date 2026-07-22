"""Tests for importing Middle Layer JSON into a DuckDB database.

The import reuses ``MiddleLayerDatabase`` flattening, then writes systems,
families, channels and a device map into DuckDB. These tests pin the data
transforms (list subfield / MemberOf joining, device-map extraction) and the
idempotency contract: re-running replaces ``source='mml'`` rows while
preserving ``source='runtime'`` rows.

The FTS extension helpers are stubbed out so the import never touches the
network or a bundled extension file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from osprey.services.channel_finder.databases import duckdb_import as dimp  # noqa: E402


@pytest.fixture(autouse=True)
def _no_fts(monkeypatch: pytest.MonkeyPatch):
    """Disable FTS install/index so imports stay hermetic (no network/file)."""
    monkeypatch.setattr(dimp, "_install_fts", lambda con: None)
    monkeypatch.setattr(dimp, "_create_fts_index", lambda con: None)


@pytest.fixture()
def mml_json(tmp_path: Path) -> str:
    """A small MML JSON with a flat field, a nested subfield, and a device map."""
    data = {
        "SR": {
            "_description": "Storage Ring",
            "BPM": {
                "_description": "Beam position monitors",
                "Monitor": {
                    "ChannelNames": ["SR01:BPM:X", "SR01:BPM:Y"],
                    "Units": "mm",
                    "DataType": "double",
                    "MemberOf": ["BPM", "Diagnostics"],
                },
                "Setpoint": {
                    "X": {"ChannelNames": ["SR01:BPM:XSet"]},
                },
                "setup": {
                    "DeviceList": [[1, 1], [1, 2]],
                    "CommonNames": ["BPM1", "BPM2"],
                },
            },
        },
    }
    path = tmp_path / "middle_layer.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


class TestImportStats:
    def test_row_counts(self, mml_json: str, tmp_path: Path):
        out = str(tmp_path / "out.duckdb")
        stats = dimp.import_to_duckdb(mml_json, out)

        assert stats["systems"] == 1
        assert stats["families"] == 1
        assert stats["channels"] == 3  # X, Y, XSet
        assert stats["device_map_entries"] == 2
        assert stats["duckdb_path"] == out


class TestImportedContent:
    def test_subfield_list_is_colon_joined(self, mml_json: str, tmp_path: Path):
        out = str(tmp_path / "out.duckdb")
        dimp.import_to_duckdb(mml_json, out)

        con = duckdb.connect(out)
        try:
            (subfield,) = con.execute(
                "SELECT subfield FROM channels WHERE channel_name = 'SR01:BPM:XSet'"
            ).fetchone()
            (field,) = con.execute(
                "SELECT field FROM channels WHERE channel_name = 'SR01:BPM:XSet'"
            ).fetchone()
        finally:
            con.close()
        assert field == "Setpoint"
        assert subfield == "X"

    def test_member_of_list_and_units_preserved(self, mml_json: str, tmp_path: Path):
        out = str(tmp_path / "out.duckdb")
        dimp.import_to_duckdb(mml_json, out)

        con = duckdb.connect(out)
        try:
            member_of, units = con.execute(
                "SELECT member_of, units FROM channels WHERE channel_name = 'SR01:BPM:X'"
            ).fetchone()
        finally:
            con.close()
        assert member_of == "BPM, Diagnostics"
        assert units == "mm"

    def test_device_map_pairs_index_to_common_name(self, mml_json: str, tmp_path: Path):
        out = str(tmp_path / "out.duckdb")
        dimp.import_to_duckdb(mml_json, out)

        con = duckdb.connect(out)
        try:
            rows = con.execute(
                "SELECT device_index, sector, device, common_name "
                "FROM device_map ORDER BY device_index"
            ).fetchall()
        finally:
            con.close()
        assert rows == [(0, 1, 1, "BPM1"), (1, 1, 2, "BPM2")]


class TestIdempotency:
    def test_reimport_preserves_runtime_rows(self, mml_json: str, tmp_path: Path):
        out = str(tmp_path / "out.duckdb")
        dimp.import_to_duckdb(mml_json, out)

        # An agent adds a runtime channel between rebuilds.
        con = duckdb.connect(out)
        try:
            con.execute(
                "INSERT INTO channels (channel_name, system, family, source) "
                "VALUES ('SR01:RUNTIME:1', 'SR', 'BPM', 'runtime')"
            )
        finally:
            con.close()

        # Rebuild from the same JSON.
        stats = dimp.import_to_duckdb(mml_json, out)
        assert stats["channels"] == 3  # only mml rows re-inserted

        con = duckdb.connect(out)
        try:
            (runtime_count,) = con.execute(
                "SELECT COUNT(*) FROM channels WHERE source = 'runtime'"
            ).fetchone()
            (mml_count,) = con.execute(
                "SELECT COUNT(*) FROM channels WHERE source = 'mml'"
            ).fetchone()
        finally:
            con.close()
        assert runtime_count == 1  # runtime row survived the rebuild
        assert mml_count == 3  # mml rows replaced, not duplicated
