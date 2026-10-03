"""Tests for importing Middle Layer JSON into a DuckDB database.

The import reuses ``MiddleLayerDatabase`` flattening, then writes systems,
families, channels and a device map into DuckDB. These tests pin the data
transforms (list subfield / MemberOf joining, device-map extraction), one
``channels`` row per place a channel is listed, and the idempotency
contract: re-running replaces ``source='mml'`` rows while preserving
``source='runtime'`` rows.

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
    monkeypatch.setattr(dimp, "ensure_fts", lambda con: None)
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


#: Member families and the umbrella families that repeat their channels.
_UMBRELLA = {
    "SR": {
        "QF": {"Monitor": {"ChannelNames": ["SR:QF1:I", "SR:QF2:I"]}},
        "HCM": {"Monitor": {"ChannelNames": ["SR:HCM1:I"]}},
        "BPM": {"X": {"ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"]}},
        "MAG": {
            "SR:QF1:I": {"ChannelNames": ["SR:QF1:I"]},
            "SR:QF2:I": {"ChannelNames": ["SR:QF2:I"]},
            "SR:HCM1:I": {"ChannelNames": ["SR:HCM1:I"]},
        },
        "DIAG": {"X": {"ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"]}},
    }
}


class TestEveryFamilyAChannelBelongsTo:
    """A channel is one row per family it belongs to."""

    @pytest.fixture()
    def umbrella(self, tmp_path: Path):
        src = tmp_path / "ml.json"
        src.write_text(json.dumps(_UMBRELLA))
        out = str(tmp_path / "out.duckdb")
        dimp.import_to_duckdb(str(src), out)
        con = duckdb.connect(out, read_only=True)
        try:
            yield con
        finally:
            con.close()

    @staticmethod
    def _family(con, family: str) -> list[str]:
        rows = con.execute(
            "SELECT channel_name FROM channels WHERE family = ? ORDER BY channel_name", [family]
        ).fetchall()
        return [name for (name,) in rows]

    def test_a_member_family_and_its_umbrella_both_return_the_bpms(self, umbrella):
        assert self._family(umbrella, "BPM") == ["SR:BPM1:X", "SR:BPM2:X"]
        assert self._family(umbrella, "DIAG") == ["SR:BPM1:X", "SR:BPM2:X"]

    def test_a_member_family_returns_its_magnets_and_the_umbrella_every_magnet(self, umbrella):
        assert self._family(umbrella, "QF") == ["SR:QF1:I", "SR:QF2:I"]
        assert self._family(umbrella, "MAG") == ["SR:HCM1:I", "SR:QF1:I", "SR:QF2:I"]

    def test_rows_count_memberships_and_distinct_names_count_channels(self, umbrella):
        assert umbrella.execute(
            "SELECT COUNT(*), COUNT(DISTINCT channel_name) FROM channels"
        ).fetchone() == (10, 5)


#: One channel listed under Fields X and Y of Family BPM, under X twice with
#: different Subfields, and twice at the exact path X:Raw.
_PER_FIELD = {
    "SR": {
        "BPM": {
            "X": {
                "Raw": {"ChannelNames": ["SR01:BPM:A", "SR01:BPM:A"]},
                "Cal": {"ChannelNames": ["SR01:BPM:A"]},
            },
            "Y": {"ChannelNames": ["SR01:BPM:A"]},
        }
    }
}


class TestEveryFieldAChannelIsListedUnder:
    """A channel is one row per (System, Family, Field, Subfield) path that lists it."""

    @pytest.fixture()
    def per_field(self, tmp_path: Path):
        src = tmp_path / "ml.json"
        src.write_text(json.dumps(_PER_FIELD))
        out = str(tmp_path / "out.duckdb")
        dimp.import_to_duckdb(str(src), out)
        con = duckdb.connect(out)
        try:
            yield con
        finally:
            con.close()

    def test_each_listing_is_one_row_and_the_channel_counts_once(self, per_field):
        rows = per_field.execute(
            "SELECT family, field, subfield FROM channels ORDER BY row_id"
        ).fetchall()
        assert rows == [("BPM", "X", "Raw"), ("BPM", "X", "Cal"), ("BPM", "Y", "")]
        assert per_field.execute(
            "SELECT COUNT(DISTINCT channel_name), COUNT(DISTINCT (channel_name, family)) "
            "FROM channels"
        ).fetchone() == (1, 1)
        assert per_field.execute(
            "SELECT DISTINCT family FROM channels WHERE family = 'BPM'"
        ).fetchall() == [("BPM",)]

    def test_the_key_refuses_a_second_row_at_the_same_path(self, per_field):
        with pytest.raises(duckdb.ConstraintException):
            per_field.execute(
                "INSERT INTO channels (channel_name, system, family, field, subfield) "
                "VALUES ('SR01:BPM:A', 'SR', 'BPM', 'Y', '')"
            )


class TestEngineeringUnit:
    """``channels.units`` holds the unit a field is served in, never the MML mode word."""

    @pytest.mark.parametrize(
        ("meta", "expected"),
        [
            ({"Units": "Hardware", "HWUnits": "Amps", "PhysicsUnits": "rad"}, "Amps"),
            ({"Units": "Physics", "HWUnits": "Amps", "PhysicsUnits": "rad"}, "rad"),
            ({"HWUnits": "mm"}, "mm"),
            ({"Units": "mm"}, "mm"),
            ({"Units": "Hardware", "HWUnits": ["nm", "nm", "nm"]}, "nm"),
            ({"Units": "Hardware", "HWUnits": ["nm", "", " "]}, "nm"),
            ({"Units": "Hardware", "HWUnits": ["nm", "mm"]}, ""),
            ({"Units": "Hardware", "HWUnits": []}, ""),
            ({"Units": "Hardware"}, ""),
            ({"Units": "Physics", "HWUnits": "Amps"}, ""),
            ({}, ""),
        ],
    )
    def test_unit_per_metadata_shape(self, meta: dict, expected: str):
        """A mode word selects the unit key; any other ``Units`` string is the unit."""
        assert dimp._engineering_unit(meta) == expected

    def test_mode_word_never_reaches_the_column(self, tmp_path: Path):
        """A field served in hardware units lands its ``HWUnits`` in ``units``."""
        data = {
            "SR": {
                "HCM": {
                    "Setpoint": {
                        "ChannelNames": ["SR01:HCM:SP"],
                        "Units": "Hardware",
                        "HWUnits": "Amps",
                        "PhysicsUnits": "rad",
                    }
                }
            }
        }
        src = tmp_path / "ml.json"
        src.write_text(json.dumps(data))
        out = str(tmp_path / "out.duckdb")
        dimp.import_to_duckdb(str(src), out)

        con = duckdb.connect(out)
        try:
            (units,) = con.execute("SELECT units FROM channels").fetchone()
        finally:
            con.close()
        assert units == "Amps"

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
