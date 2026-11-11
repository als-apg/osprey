"""``osprey facility import mml``: exports written as the mml layer's sources.

Each repo case copies one initialised demo repo, whose
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
from osprey.facility.layers import FORMER_HEADER, HEADER
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
    "move corrections into fixes.yaml and remove these records, "
    "or start a new repo: osprey init <name> --preset demo"
)


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A demo repo, initialised once for this module and never edited."""
    return init_project(tmp_path_factory.mktemp("ca"), "demo", "demo")


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


#: Hand-written records the export's import writes too, and one of each kind it does not.
COLLIDING = {
    "records/devices.yaml": "- id: LINAC/GUN\n  class: Gun\n- id: SR/qk_bdm_1\n  class: Bend\n",
    "records/channels.yaml": "- id: QK:BDM:1:CUR:SP\n- id: LN:GUN:V\n",
    "records/groups.yaml": "- id: BEND\n- id: Guns\n",
    "models.yaml": "- name: SR\n  engine: pyat\n",
    "decks/SR.json": "{}\n",
    "measurement/SR.yaml": "model: SR\n",
}

#: Hand-written records of every kind the import writes, none of which it writes.
APART = {
    "records/devices.yaml": "- id: LINAC/GUN\n  class: Gun\n",
    "records/channels.yaml": "- id: LN:GUN:V\n",
    "records/groups.yaml": "- id: Guns\n",
    "models.yaml": "- name: LINAC\n  engine: pyat\n",
    "decks/LINAC.json": "{}\n",
    "measurement/LINAC.yaml": "model: LINAC\n",
}


def _write(facility: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (facility / name).parent.mkdir(parents=True, exist_ok=True)
        (facility / name).write_text(text, encoding="utf-8")


def test_a_hand_written_record_the_import_writes_refuses(cleared: Path) -> None:
    pytest.importorskip("at")
    _write(_facility(cleared), {name: COLLIDING[name] for name in ("records/devices.yaml",)})
    before = _snapshot(cleared)

    result = _import(cleared)

    assert result.exit_code == 1
    assert result.stderr.splitlines() == _refusal(
        "data/facility/records/devices.yaml: SR/qk_bdm_1", kinds="device"
    )
    assert result.stdout == ""
    assert _snapshot(cleared) == before


def test_every_kind_of_record_the_import_writes_refuses_by_its_id(cleared: Path) -> None:
    pytest.importorskip("at")
    _write(_facility(cleared), COLLIDING)
    before = _snapshot(cleared)

    result = _import(cleared)

    assert result.exit_code == 1
    assert result.stderr.splitlines() == _refusal(
        "data/facility/decks/SR.json",
        "data/facility/measurement/SR.yaml",
        "data/facility/models.yaml: SR",
        "data/facility/records/channels.yaml: QK:BDM:1:CUR:SP",
        "data/facility/records/devices.yaml: SR/qk_bdm_1",
        "data/facility/records/groups.yaml: BEND",
        kinds="channel, device, group, model, deck and measurement",
    )
    assert _snapshot(cleared) == before


def test_a_refusal_names_five_ids_of_a_file_and_counts_the_rest(cleared: Path) -> None:
    pytest.importorskip("at")
    facility = _facility(cleared)
    assert _import(cleared).exit_code == 0
    layer = (facility / LAYER_DIR / "devices.yaml").read_text(encoding="utf-8")
    ids = [line[6:] for line in layer.splitlines() if line.startswith("- id: ")]
    assert len(ids) > 5
    _write(facility, {"records/devices.yaml": "".join(f"- id: {name}\n" for name in ids)})
    before = _snapshot(cleared)

    result = _import(cleared)

    assert result.exit_code == 1
    assert result.stderr.splitlines()[1] == (
        f"data/facility/records/devices.yaml: {', '.join(sorted(ids)[:5])} and {len(ids) - 5} more"
    )
    assert _snapshot(cleared) == before


def test_hand_written_records_the_import_does_not_write_stay(cleared: Path) -> None:
    pytest.importorskip("at")
    facility = _facility(cleared)
    _write(facility, APART)

    first = _import(cleared)
    again = _import(cleared)

    assert first.exit_code == 0, first.output
    assert again.exit_code == 0, again.output
    assert "hand-written" not in first.output + again.output
    assert {name: (facility / name).read_text(encoding="utf-8") for name in APART} == APART
    assert (facility / LAYER_DIR / "decks" / "SR.json").is_file()
    assert (facility / "measurement" / "SR.yaml").is_file()


def test_hand_written_records_without_a_mapping_stop_at_the_draft_with_nothing_replaced(
    repo: Path,
) -> None:
    facility = _facility(repo)
    _drop_header(facility / "records" / "devices.yaml")
    before = _snapshot(repo)

    result = _import(repo)

    assert result.exit_code == 1
    assert "replaced the bundled" not in result.output
    after = _snapshot(repo)
    assert set(after) - set(before) == {f"data/facility/{MAPPING_FILE}"}
    assert {name: after[name] for name in before} == before


def test_each_kind_it_writes_is_named_in_the_order_of_the_table(cleared: Path) -> None:
    pytest.importorskip("at")
    facility = _facility(cleared)
    _drop_header(facility / "limits.yaml")
    _write(facility, {name: COLLIDING[name] for name in COLLIDING if name.startswith("records/")})

    result = _import(cleared)

    assert result.exit_code == 1
    assert result.stderr.splitlines()[0] == (
        "import mml: data/facility/ holds hand-written channel, device, group and limit "
        "records, which this import writes:"
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


def _former(*paths: str) -> list[str]:
    return [
        "import mml: data/facility/ holds files seeded under the old header, "
        "which this import writes:",
        *paths,
        f"set line 1 of each to: {HEADER}",
    ]


def test_a_file_under_the_former_header_is_refused_with_the_header_fix(cleared: Path) -> None:
    facility = _facility(cleared)
    (facility / "limits.yaml").write_text(f"{FORMER_HEADER}\nrecords: []\n", encoding="utf-8")
    (facility / "seeds.yaml").write_text(f"{FORMER_HEADER}\nseeds: []\n", encoding="utf-8")
    before = _snapshot(cleared)

    result = _import(cleared)

    assert result.exit_code == 1
    assert result.stderr.splitlines() == _former(
        "data/facility/limits.yaml", "data/facility/seeds.yaml"
    )
    assert result.stdout == ""
    assert _snapshot(cleared) == before


def test_a_former_header_set_to_the_current_one_imports_and_keeps_the_edits(
    cleared: Path,
) -> None:
    pytest.importorskip("at")
    limits = _facility(cleared) / "limits.yaml"
    limits.write_text(f"{FORMER_HEADER}\nrecords: []\n", encoding="utf-8")
    assert _import(cleared).exit_code == 1
    limits.write_text(f"{HEADER}\nrecords: []\n", encoding="utf-8")

    result = _import(cleared)

    assert result.exit_code == 0, result.output
    assert limits.read_text(encoding="utf-8") == f"{HEADER}\nrecords: []\n"


def test_hand_written_and_former_header_files_are_named_in_their_own_groups(
    cleared: Path,
) -> None:
    pytest.importorskip("at")
    facility = _facility(cleared)
    _write(facility, {"records/devices.yaml": COLLIDING["records/devices.yaml"]})
    (facility / "limits.yaml").write_text(f"{FORMER_HEADER}\nrecords: []\n", encoding="utf-8")
    before = _snapshot(cleared)

    result = _import(cleared)

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        *_refusal("data/facility/records/devices.yaml: SR/qk_bdm_1", kinds="device"),
        *_former("data/facility/limits.yaml"),
    ]
    assert _snapshot(cleared) == before


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


def test_the_knowledge_bundle_validates_after_an_import_of_a_tree_with_no_bundled_file(
    repo: Path,
) -> None:
    pytest.importorskip("at")
    facility = _facility(repo)
    shutil.rmtree(facility)
    (facility / MAPPING_FILE).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TREE / MAPPING_FILE, facility / MAPPING_FILE)
    knowledge = facility / "knowledge"
    knowledge.mkdir()
    (knowledge / "note.md").write_text(
        "---\ntype: Note\ntitle: A note\ndescription: Kept.\n---\n\nA note.\n", encoding="utf-8"
    )

    imported = _import(repo)

    assert imported.exit_code == 0, imported.output
    assert "replaced the bundled" not in imported.stderr
    assert "/note.md" in (knowledge / "index.md").read_text(encoding="utf-8")
    result = CliRunner().invoke(cli, ["knowledge", "validate", str(knowledge)])
    assert result.exit_code == 0, result.output
