"""The relations import as a whole: refuse-or-replace one layer (CF-2, CF-11, p5-r5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.layers import LayerSourceInvalid
from osprey.facility.layers import relations as relations_layer
from osprey.facility.layers.relations import import_relations

HEADER = "type,subject,object,subject_kind,object_kind,subject_class,document,version,imported\n"


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, document in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = document if isinstance(document, str) else yaml.safe_dump(document)
        path.write_text(text, encoding="utf-8")
    return root


def _tree(tmp_path: Path, extra: dict[str, Any] | None = None) -> Path:
    files: dict[str, Any] = {
        "records/places.yaml": [{"id": "SR"}, {"id": "SR/S01"}],
        "records/devices.yaml": [
            {"id": "QF1", "class": "Quadrupole"},
            {"id": "QF2", "class": "Quadrupole"},
            {"id": "PS1", "class": "PowerSupply"},
        ],
        "records/groups.yaml": [{"id": "QUADS", "members": ["QF1", "QF2"]}],
    }
    files.update(extra or {})
    return _write(tmp_path / "facility", files)


def _csv(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / "in" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _relations(tmp_path: Path, *rows: str, name: str = "cables.csv") -> Path:
    return _csv(tmp_path, name, HEADER + "".join(rows))


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None)
        for path in sorted(root.rglob("*"))
    }


ROWS = (
    "powered_by,QF1,PS1,,,,DOC-1,A,2026-01-01\n",
    "powered_by,QF2,PS1,,,,DOC-1,A,2026-01-01\n",
    "powered_by,QF3,PS1,,,Quadrupole,DOC-1,A,2026-01-01\n",
)


class TestWrite:
    def test_writes_the_layer_and_copies_the_tables(self, tmp_path: Path) -> None:
        facility = _tree(tmp_path)
        table = _relations(tmp_path, *ROWS)
        places = _csv(tmp_path, "where.csv", "place,within\nRACK1,SR/S01\n")
        members = _csv(tmp_path, "who.csv", "group,member\nRACKED,QF3\n")
        names = _csv(tmp_path, "aka.csv", "thing,name,scheme\nQF3,Q-THREE,ctl\n")
        written = import_relations(
            table, facility, "cables", names=names, places=places, members=members
        )
        layer = facility / "imported" / "cables"
        assert sorted(p.name for p in written) == sorted(
            [
                "relations.yaml",
                "devices.yaml",
                "places.yaml",
                "groups.yaml",
                "relations.csv",
                "names.csv",
                "places.csv",
                "members.csv",
            ]
        )
        assert all(p.parent == layer for p in written)
        assert sorted(p.name for p in layer.iterdir()) == sorted(p.name for p in written)
        assert (layer / "relations.csv").read_bytes() == table.read_bytes()
        assert (layer / "places.csv").read_bytes() == places.read_bytes()
        assert (layer / "members.csv").read_bytes() == members.read_bytes()
        assert (layer / "names.csv").read_bytes() == names.read_bytes()
        claims = yaml.safe_load((layer / "relations.yaml").read_text(encoding="utf-8"))
        assert [c["id"] for c in claims] == sorted(c["id"] for c in claims)
        assert len(claims) == 3
        devices = yaml.safe_load((layer / "devices.yaml").read_text(encoding="utf-8"))
        assert [d["id"] for d in devices] == ["QF3"]
        assert devices[0]["scheme_names"][0]["name"] == "Q-THREE"
        assert yaml.safe_load((layer / "places.yaml").read_text(encoding="utf-8")) == [
            {"id": "SR/S01/RACK1"}
        ]
        assert yaml.safe_load((layer / "groups.yaml").read_text(encoding="utf-8")) == [
            {"id": "RACKED", "members": ["QF3"]}
        ]

    def test_only_non_empty_record_files_are_written(self, tmp_path: Path) -> None:
        facility = _tree(tmp_path)
        written = import_relations(_relations(tmp_path, ROWS[0]), facility, "cables")
        assert [p.name for p in written] == ["relations.yaml", "relations.csv"]
        layer = facility / "imported" / "cables"
        assert sorted(p.name for p in layer.iterdir()) == ["relations.csv", "relations.yaml"]

    def test_no_staging_dir_is_left(self, tmp_path: Path) -> None:
        facility = _tree(tmp_path)
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        assert sorted(p.name for p in facility.iterdir()) == ["imported", "records"]

    def test_the_build_loader_reads_the_layer(self, tmp_path: Path) -> None:
        from osprey.facility.sources import load_sources

        facility = _tree(tmp_path)
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        result = load_sources(facility)
        assert "cables" in result.sources.relations_layers


class TestReplace:
    def test_stale_file_of_the_previous_import_is_gone(self, tmp_path: Path) -> None:
        facility = _tree(tmp_path)
        members = _csv(tmp_path, "who.csv", "group,member\nRACKED,QF3\n")
        import_relations(_relations(tmp_path, *ROWS), facility, "cables", members=members)
        layer = facility / "imported" / "cables"
        assert (layer / "groups.yaml").is_file()
        assert (layer / "members.csv").is_file()
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        assert not (layer / "groups.yaml").exists()
        assert not (layer / "members.csv").exists()

    def test_re_import_resolves_without_its_own_previous_output(self, tmp_path: Path) -> None:
        facility = _tree(tmp_path)
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        first = _snapshot(facility)
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        assert _snapshot(facility) == first

    def test_second_import_on_a_different_date_is_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import datetime

        facility = _tree(tmp_path)
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        first = _snapshot(facility)

        class _Later(datetime.date):
            @classmethod
            def today(cls) -> datetime.date:
                return datetime.date(2031, 6, 1)

        monkeypatch.setattr(datetime, "date", _Later)
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        assert _snapshot(facility) == first
        text = (facility / "imported/cables/relations.yaml").read_text(encoding="utf-8")
        assert "2031" not in text

    def test_reordered_rows_keep_the_bytes_of_relations_yaml(self, tmp_path: Path) -> None:
        facility = _tree(tmp_path)
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        first = (facility / "imported/cables/relations.yaml").read_bytes()
        first_devices = (facility / "imported/cables/devices.yaml").read_bytes()
        import_relations(_relations(tmp_path, *reversed(ROWS)), facility, "cables")
        assert (facility / "imported/cables/relations.yaml").read_bytes() == first
        assert (facility / "imported/cables/devices.yaml").read_bytes() == first_devices

    def test_other_layers_are_untouched(self, tmp_path: Path) -> None:
        facility = _tree(
            tmp_path, {"imported/controls/devices.yaml": [{"id": "PS9", "class": "PowerSupply"}]}
        )
        before = (facility / "imported/controls/devices.yaml").read_bytes()
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        assert (facility / "imported/controls/devices.yaml").read_bytes() == before
        assert sorted(p.name for p in (facility / "imported/controls").iterdir()) == [
            "devices.yaml"
        ]


class TestRefused:
    def test_refused_run_leaves_the_whole_tree_byte_identical(self, tmp_path: Path) -> None:
        facility = _tree(tmp_path)
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        before = _snapshot(facility)
        bad = _relations(tmp_path, ROWS[0], "powered_by,NOSUCH,PS1,,,,DOC-1,A,\n", name="bad.csv")
        with pytest.raises(LayerSourceInvalid) as caught:
            import_relations(bad, facility, "cables")
        assert any("NOSUCH" in e.format_message() for e in caught.value.errors)
        assert _snapshot(facility) == before

    def test_refused_first_run_creates_nothing(self, tmp_path: Path) -> None:
        facility = _tree(tmp_path)
        before = _snapshot(facility)
        bad = _relations(tmp_path, "no_such_type,QF1,PS1,,,,,,\n")
        with pytest.raises(LayerSourceInvalid):
            import_relations(bad, facility, "cables")
        assert _snapshot(facility) == before

    def test_refusals_of_every_table_are_reported_together(self, tmp_path: Path) -> None:
        facility = _tree(tmp_path)
        before = _snapshot(facility)
        bad = _relations(tmp_path, "powered_by,NOSUCH,PS1,,,,,,\n")
        members = _csv(tmp_path, "who.csv", "group,member\nQUADS,PS1\n")
        with pytest.raises(LayerSourceInvalid) as caught:
            import_relations(bad, facility, "cables", members=members)
        text = "\n".join(e.format_message() for e in caught.value.errors)
        assert "NOSUCH" in text
        assert "QUADS" in text or "PS1" in text
        assert _snapshot(facility) == before


class TestLayerGuard:
    @pytest.mark.parametrize("layer", ["list", "mml", "authored"])
    def test_reserved_names_are_refused(self, tmp_path: Path, layer: str) -> None:
        facility = _tree(tmp_path)
        before = _snapshot(facility)
        with pytest.raises(LayerSourceInvalid) as caught:
            import_relations(_relations(tmp_path, *ROWS), facility, layer)
        assert caught.value.errors[0].kind == "source-invalid"
        assert _snapshot(facility) == before

    @pytest.mark.parametrize("layer", ["", "Cables", "-x", "a/b", "..", ".hidden", "a b"])
    def test_malformed_names_are_refused(self, tmp_path: Path, layer: str) -> None:
        facility = _tree(tmp_path)
        before = _snapshot(facility)
        with pytest.raises(LayerSourceInvalid):
            import_relations(_relations(tmp_path, *ROWS), facility, layer)
        assert _snapshot(facility) == before

    @pytest.mark.parametrize(
        "foreign",
        ["channels.yaml", "models.yaml", "mapping.yaml", "rows.json", "decks/SR.lat"],
    )
    def test_another_formats_layer_is_refused(self, tmp_path: Path, foreign: str) -> None:
        facility = _tree(
            tmp_path,
            {
                "imported/ctl/devices.yaml": [{"id": "PS9", "class": "PowerSupply"}],
                f"imported/ctl/{foreign}": "x: 1\n" if foreign.endswith(".yaml") else "x\n",
            },
        )
        before = _snapshot(facility)
        with pytest.raises(LayerSourceInvalid) as caught:
            import_relations(_relations(tmp_path, *ROWS), facility, "ctl")
        text = "\n".join(e.format_message() for e in caught.value.errors)
        assert foreign.split("/")[0] in text
        assert _snapshot(facility) == before

    def test_name_pattern_accepts_digits_dash_underscore(self, tmp_path: Path) -> None:
        facility = _tree(tmp_path)
        written = import_relations(_relations(tmp_path, *ROWS), facility, "0cables_v2-a")
        assert written[0].parent == facility / "imported" / "0cables_v2-a"


class TestWriteFailure:
    def test_exception_during_write_leaves_no_staging_and_old_layer_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        facility = _tree(tmp_path)
        members = _csv(tmp_path, "who.csv", "group,member\nRACKED,QF3\n")
        import_relations(_relations(tmp_path, *ROWS), facility, "cables", members=members)
        before = _snapshot(facility)

        real = relations_layer._dump
        calls: list[Path] = []

        def failing(path: Path, rows: list[dict[str, Any]]) -> Path:
            calls.append(path)
            if len(calls) == 2:
                raise OSError("disk full")
            return real(path, rows)

        monkeypatch.setattr(relations_layer, "_dump", failing)
        with pytest.raises(OSError, match="disk full"):
            import_relations(_relations(tmp_path, *ROWS[1:]), facility, "cables", members=members)
        assert len(calls) == 2
        assert _snapshot(facility) == before
        assert not [p for p in facility.iterdir() if p.name.startswith(".")]

    def test_failed_rename_restores_the_old_layer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        facility = _tree(tmp_path)
        import_relations(_relations(tmp_path, *ROWS), facility, "cables")
        before = _snapshot(facility)

        import os

        real = os.rename
        calls: list[tuple[str, str]] = []

        def failing(source: Any, target: Any) -> None:
            calls.append((str(source), str(target)))
            # The first rename moves the old layer aside; fail the one after it.
            if len(calls) == 2:
                raise OSError("rename failed")
            real(source, target)

        monkeypatch.setattr(os, "rename", failing)
        with pytest.raises(OSError, match="rename failed"):
            import_relations(_relations(tmp_path, ROWS[0]), facility, "cables")
        # aside, the failed swap, then the old layer moved back
        assert len(calls) == 3
        assert calls[2] == calls[0][::-1]
        assert _snapshot(facility) == before
        assert not [p for p in facility.iterdir() if p.name.startswith(".")]
