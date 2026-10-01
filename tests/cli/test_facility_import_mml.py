"""``osprey facility import mml``: exports written as the mml layer's sources.

Each repo case copies one initialised control-assistant repo, whose
``data/facility/`` holds the demo's authored tree, and imports the synthetic
fixture tree's export under that tree's mapping.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from osprey.cli.main import cli
from osprey.facility.layers.mml.importer import LAYER_DIR
from osprey.facility.layers.mml.mapping import MAPPING_FILE
from osprey.facility.layers.mml.seed import HEADER
from tests._builds import init_project

REPO_ROOT = Path(__file__).resolve().parents[2]
TREE = REPO_ROOT / "tests" / "fixtures" / "mml" / "synthetic"
EXPORT = TREE / "quokka.sr.ao.json"
EXPORTER = REPO_ROOT / "src" / "osprey" / "facility" / "layers" / "mml" / "mml_export.m"

#: The stop's first line, with the count of the files it then names.
STOP = "import mml: authored-present: {} files"


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A control-assistant repo, initialised once for this module and never edited."""
    return init_project(tmp_path_factory.mktemp("ca"), "control-assistant", "demo")


@pytest.fixture
def repo(initialised: Path, tmp_path: Path) -> Path:
    """A copy of the initialised repo this test may edit."""
    copy = tmp_path / initialised.name
    shutil.copytree(initialised, copy, symlinks=True)
    return copy


def _import(repo: Path) -> Result:
    return CliRunner().invoke(cli, ["facility", "import", "mml", str(EXPORT), "--repo", str(repo)])


def _removals(result: Result) -> list[str]:
    """The paths a stop names, one per ``rm`` line."""
    return [line[3:] for line in result.stderr.splitlines() if line.startswith("rm ")]


def _facility(repo: Path) -> Path:
    return repo / "data" / "facility"


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _authored(facility: Path) -> dict[str, bytes]:
    """Every file outside the layer directory, by its path under ``data/facility``."""
    return {
        name: data
        for name, data in _snapshot(facility).items()
        if not name.startswith(f"{LAYER_DIR}/")
    }


def _written(result: Result) -> list[str]:
    return [line[6:] for line in result.stdout.splitlines() if line.startswith("wrote ")]


@pytest.fixture
def cleared(repo: Path) -> Path:
    """A repo with the removals the stop prints applied and the tree's mapping in place."""
    for path in _removals(_import(repo)):
        (repo / path).unlink()
    target = _facility(repo) / MAPPING_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TREE / MAPPING_FILE, target)
    return repo


def test_print_exporter_needs_no_repo_and_no_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["facility", "import", "mml", "--print-exporter"])

    assert result.exit_code == 0, result.output
    assert result.stdout == EXPORTER.read_text(encoding="utf-8")
    assert list(tmp_path.iterdir()) == []


def test_an_authored_tree_stops_the_import_and_names_each_file(repo: Path) -> None:
    before = _snapshot(repo)

    result = _import(repo)

    assert result.exit_code == 1
    removals = _removals(result)
    assert result.stderr.splitlines() == [
        STOP.format(len(removals)),
        *(f"rm {path}" for path in removals),
    ]
    assert "data/facility/records/channels.yaml" in removals
    assert removals == sorted(removals)
    assert all((repo / path).is_file() for path in removals)
    assert result.stdout == ""
    assert _snapshot(repo) == before


def test_fixes_classes_and_knowledge_are_never_named(repo: Path) -> None:
    facility = _facility(repo)
    (facility / "fixes.yaml").write_text("fixes: []\n", encoding="utf-8")
    (facility / "classes.yaml").write_text("classes: []\n", encoding="utf-8")
    (facility / "knowledge").mkdir(exist_ok=True)
    (facility / "knowledge" / "note.md").write_text("A note.\n", encoding="utf-8")

    removals = _removals(_import(repo))

    assert removals
    kept = ("fixes.yaml", "classes.yaml", "knowledge/")
    assert [path for path in removals if any(name in path for name in kept)] == []


def test_the_import_runs_once_the_named_files_are_removed(cleared: Path) -> None:
    pytest.importorskip("at")

    result = _import(cleared)

    assert result.exit_code == 0, result.output
    written = _written(result)
    assert f"data/facility/{LAYER_DIR}/channels.yaml" in written
    assert "data/facility/limits.yaml" in written
    assert all((cleared / path).is_file() for path in written)
    assert result.stdout.count("golden skipped:") == 1


def test_a_second_import_leaves_the_seeded_files_byte_unchanged(cleared: Path) -> None:
    pytest.importorskip("at")
    first = _import(cleared)
    assert first.exit_code == 0, first.output
    assert first.stdout.count("golden skipped:") == 1
    facility = _facility(cleared)
    assert (facility / "seeds.yaml").is_file()
    seeded = [path for path in _written(first) if not path.startswith(f"data/facility/{LAYER_DIR}")]
    assert seeded
    before = _authored(facility)

    second = _import(cleared)

    assert second.exit_code == 0, second.output
    assert _authored(facility) == before
    # The importer reports skipped nominals only when it seeds ``seeds.yaml``,
    # and the verb prints no seeding line of its own.
    assert "golden skipped:" not in second.stdout
    assert [path for path in _written(second) if path in seeded] == []
    assert _written(second)


def test_an_import_beside_a_fix_file_runs(cleared: Path) -> None:
    pytest.importorskip("at")
    assert _import(cleared).exit_code == 0
    fixes = _facility(cleared) / "fixes.yaml"
    fixes.write_text("schema: osprey.facility.fixes/1\nfixes: []\n", encoding="utf-8")

    result = _import(cleared)

    assert result.exit_code == 0, result.output
    assert fixes.read_text(encoding="utf-8") == "schema: osprey.facility.fixes/1\nfixes: []\n"


def test_a_limits_file_under_the_layer_header_is_not_refused(cleared: Path) -> None:
    pytest.importorskip("at")
    limits = _facility(cleared) / "limits.yaml"
    limits.write_text(f"{HEADER}\nrecords: []\n", encoding="utf-8")

    result = _import(cleared)

    assert result.exit_code == 0, result.output
    assert limits.read_text(encoding="utf-8") == f"{HEADER}\nrecords: []\n"


@pytest.mark.parametrize("text", ["records: []\n", ""], ids=["no-header", "empty"])
def test_a_limits_file_without_the_layer_header_is_refused_and_named(
    cleared: Path, text: str
) -> None:
    (_facility(cleared) / "limits.yaml").write_text(text, encoding="utf-8")

    result = _import(cleared)

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [STOP.format(1), "rm data/facility/limits.yaml"]
    assert not (_facility(cleared) / LAYER_DIR / "channels.yaml").exists()


def test_a_mapping_that_fails_its_check_stops_before_any_record(cleared: Path) -> None:
    mapping = _facility(cleared) / MAPPING_FILE
    text = mapping.read_text(encoding="utf-8")
    assert text.count("    name: SR\n") == 1
    mapping.write_text(text.replace("    name: SR\n", "    name: S R\n"), encoding="utf-8")

    result = _import(cleared)

    assert result.exit_code == 1
    lines = result.stderr.splitlines()
    assert lines[0] == "models.SR.name: 'S R' is not PN_LOCAL"
    assert lines[-1] == f"{len(lines) - 1} problems in {mapping}; fix each and check again."
    assert result.stdout == ""
    assert sorted(path.name for path in (_facility(cleared) / LAYER_DIR).iterdir()) == [
        "mapping.yaml"
    ]
