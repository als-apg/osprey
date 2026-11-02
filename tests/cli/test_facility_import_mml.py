"""``osprey facility import mml``: exports written as the mml layer's sources.

Each repo case copies one initialised control-assistant repo, whose
``data/facility/`` holds the bundled example facility, and imports the synthetic
fixture tree's export under that tree's mapping. The list verb's half of the
import's one step is proved in ``tests/facility/test_layer_list.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from osprey.cli import build_profile_resolve
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.facility_cmd import KNOWLEDGE_PLACEHOLDER_LINE, _writes, import_group
from osprey.cli.main import cli
from osprey.facility.bundled import classify
from osprey.facility.layers import HEADER
from osprey.facility.layers.mml.importer import LAYER_DIR
from osprey.facility.layers.mml.mapping import MAPPING_FILE
from tests._builds import init_project

REPO_ROOT = Path(__file__).resolve().parents[2]
TREE = REPO_ROOT / "tests" / "fixtures" / "mml" / "synthetic"
EXPORT = TREE / "quokka.sr.ao.json"
EXPORTER = REPO_ROOT / "src" / "osprey" / "facility" / "layers" / "mml" / "mml_export.m"

TEMPLATE_ROOT = REPO_ROOT / "src" / "osprey" / "templates"
INDEX = "knowledge/index.md"
REMEDY = (
    "move corrections into fixes.yaml and remove the record sources, "
    "or start a new repo: osprey init <name> --preset control-assistant"
)


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
    """A fresh repo with the tree's mapping in place, which the import's step keeps."""
    target = _facility(repo) / MAPPING_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TREE / MAPPING_FILE, target)
    return repo


def _refusal(*paths: str, kinds: str) -> list[str]:
    return [
        f"import mml: data/facility/ holds hand-written {kinds} records, which this import writes:",
        *paths,
        REMEDY,
    ]


def _replaced(count: int) -> str:
    return f"import mml: replaced the bundled example facility ({count} files)"


def _drop_header(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0].startswith("# Bundled with OSPREY")
    path.write_text("\n".join(lines[1:]), encoding="utf-8")


def test_print_exporter_needs_no_repo_and_no_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["facility", "import", "mml", "--print-exporter"])

    assert result.exit_code == 0, result.output
    assert result.stdout == EXPORTER.read_text(encoding="utf-8")
    assert list(tmp_path.iterdir()) == []


def test_every_subcommand_of_the_import_group_states_what_it_writes() -> None:
    assert set(_writes()) == set(import_group.commands)


def test_a_wholly_bundled_facility_is_replaced_in_one_line(cleared: Path) -> None:
    pytest.importorskip("at")
    facility = _facility(cleared)
    count = len(classify(facility, TEMPLATE_ROOT).bundled)
    outside = {
        name: data
        for name, data in _snapshot(cleared).items()
        if not name.startswith("data/facility/")
    }

    result = _import(cleared)

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines()[0] == _replaced(count)
    assert result.stderr.count("replaced the bundled") == 1
    kinds = classify(facility, TEMPLATE_ROOT)
    assert (kinds.bundled, kinds.authored) == ((), ())
    assert (facility / MAPPING_FILE).is_file()
    assert (facility / INDEX).is_file()
    remaining = _snapshot(cleared)
    assert {
        name: data for name, data in remaining.items() if not name.startswith("data/facility/")
    } == outside


def test_a_wholly_bundled_facility_is_replaced_and_builds(repo: Path) -> None:
    pytest.importorskip("at")
    from tests.cli.test_mml_build_recipes import NO_START_SET, invoke

    tree = REPO_ROOT / "tests" / "fixtures" / "mml" / "nsls2"
    target = _facility(repo) / MAPPING_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tree / MAPPING_FILE, target)
    runner = CliRunner()
    exports = sorted(str(path) for path in tree.glob("*.ao.json"))

    invoke(runner, "facility", "import", "mml", *exports, "--repo", str(repo))
    invoke(runner, "set", "--repo", str(repo), NO_START_SET)

    invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")


def test_a_hand_written_record_of_a_kind_the_import_writes_refuses(repo: Path) -> None:
    _drop_header(_facility(repo) / "records" / "devices.yaml")
    before = _snapshot(repo)

    result = _import(repo)

    assert result.exit_code == 1
    assert result.stderr.splitlines() == _refusal(
        "data/facility/records/devices.yaml", kinds="device"
    )
    assert result.stdout == ""
    assert _snapshot(repo) == before


def test_each_kind_it_writes_is_named_in_the_order_of_the_table(repo: Path) -> None:
    facility = _facility(repo)
    _drop_header(facility / "limits.yaml")
    _drop_header(facility / "records" / "devices.yaml")
    _drop_header(facility / "records" / "channels.yaml")

    result = _import(repo)

    assert result.exit_code == 1
    assert result.stderr.splitlines()[0] == (
        "import mml: data/facility/ holds hand-written channel, device and limit records, "
        "which this import writes:"
    )


def test_a_hand_written_record_of_another_kind_survives_the_import(cleared: Path) -> None:
    pytest.importorskip("at")
    facility = _facility(cleared)
    count = len(classify(facility, TEMPLATE_ROOT).bundled)
    places = facility / "records" / "places.yaml"
    _drop_header(places)
    kept = places.read_bytes()

    result = _import(cleared)

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines()[0] == _replaced(count - 1)
    assert places.read_bytes() == kept
    assert (facility / LAYER_DIR / "channels.yaml").is_file()


def test_fixes_classes_and_knowledge_never_stop_the_import(cleared: Path) -> None:
    pytest.importorskip("at")
    facility = _facility(cleared)
    (facility / "fixes.yaml").write_text(
        "schema: osprey.facility.fixes/1\nfixes: []\n", encoding="utf-8"
    )
    (facility / "knowledge" / "note.md").write_text("A note.\n", encoding="utf-8")

    result = _import(cleared)

    assert result.exit_code == 0, result.output
    assert (facility / "knowledge" / "note.md").is_file()


def test_the_import_runs_on_a_fresh_repo(cleared: Path) -> None:
    pytest.importorskip("at")

    result = _import(cleared)

    assert result.exit_code == 0, result.output
    written = _written(result)
    assert written[0] == f"data/facility/{INDEX}"
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
    assert result.stderr.splitlines() == _refusal("data/facility/limits.yaml", kinds="limit")
    assert not (_facility(cleared) / LAYER_DIR / "channels.yaml").exists()


def test_a_mapping_that_fails_its_check_stops_before_any_record(cleared: Path) -> None:
    count = len(classify(_facility(cleared), TEMPLATE_ROOT).bundled)
    mapping = _facility(cleared) / MAPPING_FILE
    text = mapping.read_text(encoding="utf-8")
    assert text.count("    name: SR\n") == 1
    mapping.write_text(text.replace("    name: SR\n", "    name: S R\n"), encoding="utf-8")

    result = _import(cleared)

    assert result.exit_code == 1
    lines = result.stderr.splitlines()
    assert lines[0] == _replaced(count)
    assert lines[1] == "models.SR.name: 'S R' is not PN_LOCAL"
    assert lines[-1] == (
        f"{len(lines) - 2} problems in data/facility/{MAPPING_FILE}; fix each and check again."
    )
    assert result.stdout == f"wrote data/facility/{INDEX}\n{KNOWLEDGE_PLACEHOLDER_LINE}\n"
    assert sorted(path.name for path in (_facility(cleared) / LAYER_DIR).iterdir()) == [
        "mapping.yaml"
    ]


def test_a_mapping_with_the_wrong_structure_stops_with_its_key(cleared: Path) -> None:
    count = len(classify(_facility(cleared), TEMPLATE_ROOT).bundled)
    mapping = _facility(cleared) / MAPPING_FILE
    mapping.write_text("models: 3\n", encoding="utf-8")

    result = _import(cleared)

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.stderr.splitlines() == [
        _replaced(count),
        f"✗ data/facility/{MAPPING_FILE} is not a valid mapping document.",
        "  directions: required key is missing",
    ]
    assert result.stdout == f"wrote data/facility/{INDEX}\n{KNOWLEDGE_PLACEHOLDER_LINE}\n"
    assert sorted(path.name for path in (_facility(cleared) / LAYER_DIR).iterdir()) == [
        "mapping.yaml"
    ]


def test_a_profile_that_does_not_resolve_stops_the_import(repo: Path) -> None:
    profile = repo / "profile.yml"
    profile.write_text(profile.read_text(encoding="utf-8") + "no_such_key: 1\n", encoding="utf-8")
    before = _snapshot(_facility(repo))

    result = _import(repo)

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "The profile does not resolve." in result.stderr
    assert _snapshot(_facility(repo)) == before


def test_a_profile_that_names_no_data_root_stops_the_import(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Profile validation requires ``data:``, so a resolved profile without a
    # data root is reached only past it: the root goes missing once the
    # profile has resolved.
    resolve = build_profile_resolve.resolve_build_document

    def resolved_without_a_data_root(*args: Any, **kwargs: Any) -> Any:
        resolved = resolve(*args, **kwargs)
        monkeypatch.setattr(BuildProfile, "resolved_data_root", lambda self, profile_dir: None)
        return resolved

    monkeypatch.setattr(
        build_profile_resolve, "resolve_build_document", resolved_without_a_data_root
    )
    before = _snapshot(repo)

    result = _import(repo)

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.stderr.splitlines() == [
        "✗ The profile does not resolve.",
        "  a resolved profile names no data root",
    ]
    assert result.stdout == ""
    assert _snapshot(repo) == before


@pytest.mark.parametrize("separator", ["\f", " "], ids=["form-feed", "line-separator"])
def test_a_header_line_that_runs_on_past_the_header_is_refused(
    cleared: Path, separator: str
) -> None:
    limits = _facility(cleared) / "limits.yaml"
    limits.write_text(f"{HEADER}{separator}\nrecords: []\n", encoding="utf-8")

    result = _import(cleared)

    assert result.exit_code == 1
    assert result.stderr.splitlines() == _refusal("data/facility/limits.yaml", kinds="limit")


def test_the_seeded_readout_scenario_passes_a_second_import_unchanged(cleared: Path) -> None:
    pytest.importorskip("at")
    readout = _facility(cleared) / "scenarios" / "readout.yaml"

    first = _import(cleared)
    assert first.exit_code == 0, first.output
    seeded = readout.read_bytes()
    assert seeded.decode("utf-8").splitlines()[0] == HEADER

    second = _import(cleared)
    assert second.exit_code == 0, second.output
    assert "hand-written" not in second.output
    assert readout.read_bytes() == seeded


def test_the_knowledge_bundle_validates_after_an_import_that_keeps_a_page(cleared: Path) -> None:
    pytest.importorskip("at")
    knowledge = _facility(cleared) / "knowledge"
    (knowledge / "note.md").write_text(
        "---\ntype: Note\ntitle: A note\ndescription: Kept.\n---\n\nA note.\n", encoding="utf-8"
    )

    assert _import(cleared).exit_code == 0

    assert "/note.md" in (knowledge / "index.md").read_text(encoding="utf-8")
    result = CliRunner().invoke(cli, ["knowledge", "validate", str(knowledge)])
    assert result.exit_code == 0, result.output


def test_the_knowledge_bundle_validates_after_an_import_that_leaves_no_page(cleared: Path) -> None:
    pytest.importorskip("at")

    assert _import(cleared).exit_code == 0

    index = _facility(cleared) / INDEX
    assert index.read_bytes() == b""
    result = CliRunner().invoke(cli, ["knowledge", "validate", str(index.parent)])
    assert result.exit_code == 0, result.output
