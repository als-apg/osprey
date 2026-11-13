"""``osprey facility import relations``: relation tables written as one layer's sources.

Each repo case copies one initialised demo repo, whose ``data/facility/`` holds
the bundled example facility. The verb never runs the import's replace-or-refuse
step (FR10): the bundled files stay, the tables resolve against them, and no
file outside ``imported/<layer>/`` changes. The orchestrator's rules are proved
in ``tests/facility/test_relations_import*.py``; these cases prove the verb.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.main import cli
from tests._builds import init_project

LAYER = "data/facility/imported/cables"


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A demo repo, initialised once for this module and never edited."""
    return init_project(tmp_path_factory.mktemp("rel"), "demo", "demo")


@pytest.fixture
def repo(initialised: Path, tmp_path: Path) -> Path:
    """A copy of the initialised repo this test may edit."""
    copy = tmp_path / initialised.name
    shutil.copytree(initialised, copy, symlinks=True)
    return copy


def _csv(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / "in" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes() if path.is_file() else None)
        for path in sorted(root.rglob("*"))
    }


def _outside_layer(snapshot: dict[str, bytes | None]) -> dict[str, bytes | None]:
    """Every entry but the layer's own and the directories that hold it."""
    return {
        key: value
        for key, value in snapshot.items()
        if not key.startswith(LAYER) and not LAYER.startswith(f"{key}/")
    }


def _invoke(repo: Path, table: Path, *args: str) -> Result:
    return CliRunner().invoke(
        cli,
        ["facility", "import", "relations", str(table), *args, "--repo", str(repo)],
    )


def _load(path: Path) -> list[dict]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document["records"] if isinstance(document, dict) else document


RELATIONS = (
    "type,subject,object,object_kind,object_class,document,version\n"
    "powered_by,BR/QD-RAMP01,PS-QD01,,PowerSupply,PS-LIST,A\n"
    "located_in,PS-QD01,RACK01,place,,RACK-ROW3,B\n"
)


def _tables(tmp_path: Path) -> dict[str, Path]:
    return {
        "table": _csv(tmp_path, "cables.csv", RELATIONS),
        "places": _csv(tmp_path, "where.csv", "place,within\nRACK01,SR/SECT3\n"),
        "members": _csv(tmp_path, "who.csv", "group,member\nQD-SUPPLIES,PS-QD01\n"),
        "names": _csv(tmp_path, "aka.csv", "thing,name,scheme\nPS-QD01,QD-PS-1,ctl\n"),
    }


class TestSuccess:
    def test_writes_the_layer_prints_each_file_and_exits_0(
        self, repo: Path, tmp_path: Path
    ) -> None:
        tables = _tables(tmp_path)
        before = _snapshot(repo)
        result = _invoke(
            repo,
            tables["table"],
            "--layer",
            "cables",
            "--names",
            str(tables["names"]),
            "--places",
            str(tables["places"]),
            "--members",
            str(tables["members"]),
        )
        assert result.exit_code == 0, result.output
        assert result.stdout == "".join(
            f"wrote {LAYER}/{name}\n"
            for name in (
                "relations.yaml",
                "devices.yaml",
                "places.yaml",
                "groups.yaml",
                "relations.csv",
                "names.csv",
                "places.csv",
                "members.csv",
            )
        )
        layer = repo / LAYER
        assert (layer / "relations.csv").read_bytes() == tables["table"].read_bytes()
        stubs = _load(layer / "devices.yaml")
        assert [(stub["id"], stub["class"]) for stub in stubs] == [("PS-QD01", "PowerSupply")]
        assert [place["id"] for place in _load(layer / "places.yaml")] == ["SR/SECT3/RACK01"]
        assert len(_load(layer / "relations.yaml")) == 2
        # FR10: nothing outside imported/<layer>/ changes.
        assert _outside_layer(_snapshot(repo)) == _outside_layer(before)

    def test_bundled_example_facility_keeps_its_files_and_resolves_against_them(
        self, repo: Path, tmp_path: Path
    ) -> None:
        facility = repo / "data" / "facility"
        bundled = _snapshot(facility / "records")
        assert bundled, "the demo repo holds the bundled example records"
        table = _csv(
            tmp_path,
            "cables.csv",
            "type,subject,object,document\npowered_by,BR/QD-RAMP01,BR/DIPOLE01,PS-LIST\n",
        )
        result = _invoke(repo, table, "--layer", "cables")
        assert result.exit_code == 0, result.output
        assert result.stdout == (f"wrote {LAYER}/relations.yaml\nwrote {LAYER}/relations.csv\n")
        assert result.stderr == ""
        assert _snapshot(facility / "records") == bundled
        assert not (repo / LAYER / "devices.yaml").exists()

    def test_authored_relations_and_places_coexist(self, repo: Path, tmp_path: Path) -> None:
        records = repo / "data" / "facility" / "records"
        places = records / "places.yaml"
        places.write_text(
            places.read_text(encoding="utf-8") + "- id: SR/SECT3/HALL\n", encoding="utf-8"
        )
        authored = records / "relations.yaml"
        authored.write_text(
            yaml.safe_dump(
                [
                    {
                        "id": "hand-written",
                        "type": "located_in",
                        "subject": {"device": "BR/DIPOLE01"},
                        "object": {"place": "SR/SECT3/HALL"},
                    }
                ]
            ),
            encoding="utf-8",
        )
        before = _snapshot(repo)
        table = _csv(
            tmp_path,
            "cables.csv",
            "type,subject,object,object_kind,document\n"
            "located_in,BR/QD-RAMP01,SR/SECT3/HALL,place,RACK-ROW3\n",
        )
        result = _invoke(repo, table, "--layer", "cables")
        assert result.exit_code == 0, result.output
        assert result.stdout == (f"wrote {LAYER}/relations.yaml\nwrote {LAYER}/relations.csv\n")
        claims = _load(repo / LAYER / "relations.yaml")
        assert [claim["object"] for claim in claims] == [{"place": "SR/SECT3/HALL"}]
        assert _outside_layer(_snapshot(repo)) == _outside_layer(before)

    def test_stub_unknown_stubs_a_class_less_end_as_unidentified(
        self, repo: Path, tmp_path: Path
    ) -> None:
        table = _csv(
            tmp_path,
            "cables.csv",
            "type,subject,object,document\npowered_by,BR/QD-RAMP01,PS-NEW,PS-LIST\n",
        )
        result = _invoke(repo, table, "--layer", "cables", "--stub-unknown")
        assert result.exit_code == 0, result.output
        stubs = _load(repo / LAYER / "devices.yaml")
        assert [(stub["id"], stub["class"]) for stub in stubs] == [("PS-NEW", "Unidentified")]

    def test_a_second_import_replaces_the_layer_whole(self, repo: Path, tmp_path: Path) -> None:
        tables = _tables(tmp_path)
        first = _invoke(
            repo, tables["table"], "--layer", "cables", "--places", str(tables["places"])
        )
        assert first.exit_code == 0, first.output
        assert (repo / LAYER / "places.csv").exists()
        table = _csv(
            tmp_path,
            "again.csv",
            "type,subject,object,document\npowered_by,BR/QD-RAMP01,BR/DIPOLE01,PS-LIST\n",
        )
        second = _invoke(repo, table, "--layer", "cables")
        assert second.exit_code == 0, second.output
        assert sorted(path.name for path in (repo / LAYER).iterdir()) == [
            "relations.csv",
            "relations.yaml",
        ]


class TestRefusal:
    def test_a_refused_table_exits_1_with_folded_lines_and_writes_nothing(
        self, repo: Path, tmp_path: Path
    ) -> None:
        rows = "".join(
            f"powered_by,BR/QD-RAMP01,PS-MISSING{index:02d},PS-LIST\n" for index in range(25)
        )
        # One unknown end repeated on two lines folds into one refusal.
        rows += "powered_by,BR/DIPOLE01,PS-MISSING00,PS-LIST\n"
        table = _csv(tmp_path, "cables.csv", "type,subject,object,document\n" + rows)
        before = _snapshot(repo)
        result = _invoke(repo, table, "--layer", "cables")
        assert result.exit_code == 1
        assert result.stdout == ""
        lines = result.stderr.splitlines()
        assert len(lines) == 21
        assert all(line.startswith("facility: source-invalid: ") for line in lines)
        assert sum("PS-MISSING00" in line for line in lines) == 1
        assert "5 more refused values" in lines[-1]
        assert not (repo / LAYER).exists()
        assert _snapshot(repo) == before

    @pytest.mark.parametrize("layer", ["list", "mml", "authored", "Bad Name"])
    def test_a_refused_layer_name_exits_1_and_writes_nothing(
        self, repo: Path, tmp_path: Path, layer: str
    ) -> None:
        table = _csv(
            tmp_path,
            "cables.csv",
            "type,subject,object,document\npowered_by,BR/QD-RAMP01,BR/DIPOLE01,PS-LIST\n",
        )
        before = _snapshot(repo)
        result = _invoke(repo, table, "--layer", layer)
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr.startswith("facility: source-invalid: ")
        assert _snapshot(repo) == before

    def test_the_layer_option_is_required(self, repo: Path, tmp_path: Path) -> None:
        table = _csv(tmp_path, "cables.csv", "type,subject,object\n")
        result = _invoke(repo, table)
        assert result.exit_code == 2
        assert "--layer" in result.stderr


class TestHelp:
    def test_help_lists_every_tables_columns(self) -> None:
        result = CliRunner().invoke(cli, ["facility", "import", "relations", "--help"])
        assert result.exit_code == 0
        for column in (
            "type, subject, object",
            "subject_kind",
            "object_class",
            "thing, name, scheme",
            "place; within",
            "group, member; member_kind, group_kind",
            "document, version, state",
            "locator and imported",
            "machine:",
            "machine root",
            "--stub-unknown",
        ):
            assert column in result.output, column
