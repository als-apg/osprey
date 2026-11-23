"""``osprey-panels write``, ``check`` and ``classes``: operator screens from a facility file."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from osprey_panels import cli
from osprey_panels.view import VIEW_KINDS
from osprey_panels_tests._fixture import PANELS_DIR, add_part, write_fixture_facility
from osprey_panels_tests._harness import REPO_ROOT

#: The demo theme, read by its path in the checkout (a file, not an import).
DEMO_THEME = REPO_ROOT / "src/osprey/templates/apps/demo/data/canopy/theme/osprey.json"
#: The demo theme's MINOR light colour: far from the fallback MINOR shades.
THEME_ONLY_MINOR = "#b5710d"


@dataclass(frozen=True)
class Run:
    """One run of the command: its exit code and what it printed where."""

    code: int
    out: str
    err: str

    @property
    def output(self) -> str:
        return self.out + self.err


@pytest.fixture
def run(capsys: pytest.CaptureFixture[str]):
    def invoke(*argv: str | Path) -> Run:
        capsys.readouterr()
        code = cli.main([str(arg) for arg in argv])
        captured = capsys.readouterr()
        return Run(code, captured.out, captured.err)

    return invoke


@pytest.fixture(scope="module")
def facility_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_fixture_facility(tmp_path_factory.mktemp("panels-cli") / "facility.json")


@pytest.fixture
def panels(run, facility_file: Path):
    """The command on the fixture facility and its tables."""

    def invoke(verb: str, *args: str | Path, tables: Path = PANELS_DIR) -> Run:
        return run(verb, "--facility", facility_file, "--tables", tables, *args)

    return invoke


def test_write_prints_the_counts_the_total_and_the_largest(panels, tmp_path):
    result = panels("write", "--out", tmp_path / "out", "--top", "3")
    assert result.code == 0, result.output
    index = json.loads((tmp_path / "out" / "index.json").read_text())
    assert f"{len(index['screens'])} screens" in result.out
    for kind in ("card", "place", "rollup", "checkout"):
        assert f"  {kind}: " in result.out
    assert result.out.count(".screen.json") == 3
    biggest = max(index["screens"], key=lambda e: e["bytes"])
    assert biggest["file"] in result.out
    devices = sum(e["kind"] == "card" and e["thing_kind"] == "device" for e in index["screens"])
    assert f"  device cards: {devices} with the default recipe" in result.out  # no recipe rows


def test_check_after_write_exits_0(panels, tmp_path):
    out = tmp_path / "out"
    assert panels("write", "--out", out).code == 0
    result = panels("check", "--out", out)
    assert result.code == 0, result.output
    assert "match the model" in result.out
    # the classes by how their pictures are stated, before the problems
    assert "pictures, drawing: 0 classes" in result.out
    assert "pictures, parts only: 1 class (Quadrupole)" in result.out
    assert "pictures, none: " in result.out
    assert "pictures, parts not placed: none" in result.out  # information, not a problem
    assert "0 left out (no value stated) on 0 cards" in result.out
    assert "default card without tiles: " in result.out  # no roles: no tile is guessed


def test_check_counts_the_default_and_the_devices_each_cards_row_decides(panels, tmp_path):
    tables = tmp_path / "panels"
    shutil.copytree(PANELS_DIR, tables)
    (tables / "cards.csv").write_text(
        "action,class,under,devices,attribute,when,owner_relation,owner_direction,owner_to,"
        "reason,source\n"
        "drop,Pump,XR/XR01,,,,,,,drawn on its rack,S\n"
        "keep,*,,XR02:*,,no-channel,,,,worth a card,S\n"
    )
    out = tmp_path / "out"
    assert panels("write", "--out", out, tables=tables).code == 0
    result = panels("check", "--out", out, tables=tables)
    assert "  by default (no channel, no drawing): " in result.out
    assert "  cards.csv row 1, drop (class Pump, under XR/XR01): 2 devices" in result.out
    assert "  cards.csv row 2, keep (class *, devices XR02:*, when no-channel): " in result.out


def test_check_exits_1_and_counts_each_kind(panels, tmp_path):
    out = tmp_path / "out"
    panels("write", "--out", out)
    victims = sorted(out.glob("card/*.screen.json"))[:3]
    for victim in victims:
        victim.write_text("{broken")
    (out / "card" / "foreign.txt").write_text("ignored")
    result = panels("check", "--out", out, "--limit", "2")
    assert result.code == 1
    assert "invalid: 3" in result.out
    assert "... and 1 more" in result.out
    assert "foreign" not in result.output
    assert "3 problems in the written screens." in result.err


def test_check_reports_a_screen_the_schema_refuses_as_failing(panels, tmp_path):
    out = tmp_path / "out"
    assert panels("write", "--out", out).code == 0
    victim = sorted(out.glob("card/*.screen.json"))[0]
    document = json.loads(victim.read_text(encoding="utf-8"))
    document["schemaVersion"] = "0.0"  # still JSON, no longer Screen-IR
    victim.write_text(json.dumps(document), encoding="utf-8")
    for args in ((), ("--strict",)):
        result = panels("check", "--out", out, *args)
        assert result.code == 1, result.output
        assert "schema: 1" in result.out
        assert f"  {victim.relative_to(out).as_posix()}: fails the Screen-IR schema" in result.out


def test_kind_and_under_write_a_subset_and_check_notes_the_rest(panels, tmp_path):
    out = tmp_path / "out"
    written = panels("write", "--out", out, "--kind", "card", "--under", "XR/XR01")
    assert written.code == 0, written.output
    assert "not written" in written.out
    clean = panels("check", "--out", out)
    assert clean.code == 0, clean.output
    assert "outside_set:" in clean.out
    strict = panels("check", "--out", out, "--strict")
    assert strict.code == 1


def test_max_bytes_check_lists_the_oversize(panels, tmp_path):
    out = tmp_path / "out"
    assert panels("write", "--out", out).code == 0
    result = panels("check", "--out", out, "--max-bytes", "1")
    assert result.code == 1 and "oversize:" in result.out


def test_write_refuses_a_non_empty_directory_without_an_index(panels, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "mine.txt").write_text("x")
    result = panels("write", "--out", tmp_path / "out")
    assert result.code == 1
    assert "index.json" in result.err
    assert (tmp_path / "out" / "mine.txt").read_text() == "x"


def test_an_unknown_place_is_refused(panels, tmp_path):
    result = panels("write", "--out", tmp_path / "out", "--under", "NOWHERE")
    assert result.code == 1 and "NOWHERE" in result.err
    assert not (tmp_path / "out" / "index.json").exists()


# -- inputs ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verb", "option"),
    [
        ("write", "--facility"),
        ("write", "--tables"),
        ("write", "--out"),
        ("check", "--facility"),
        ("check", "--tables"),
        ("check", "--out"),
        ("classes", "--out"),
    ],
)
def test_an_omitted_required_input_is_a_usage_error_naming_it(run, facility_file, verb, option):
    given = {"--facility": facility_file, "--tables": PANELS_DIR, "--out": "out"}
    if verb == "classes":
        given = {"--out": "out"}
    argv = [part for name, value in given.items() if name != option for part in (name, value)]
    result = run(verb, *argv)
    assert result.code == 2
    assert "required" in result.err and option in result.err
    assert result.out == ""


def test_classes_takes_no_facility_input(run, facility_file, tmp_path):
    result = run("classes", "--out", tmp_path, "--facility", facility_file)
    assert result.code == 2 and "--facility" in result.err


def test_no_verb_is_a_usage_error(run):
    assert run().code == 2


def test_a_missing_facility_file_says_so(run, tmp_path):
    missing = tmp_path / "none.json"
    result = run("check", "--facility", missing, "--tables", PANELS_DIR, "--out", tmp_path)
    assert result.code == 1
    assert "The facility cannot be read." in result.err
    assert str(missing) in result.err


def test_a_missing_tables_directory_is_named(run, facility_file, tmp_path):
    missing = tmp_path / "no-tables"
    for verb in ("write", "check"):
        result = run(
            verb, "--facility", facility_file, "--tables", missing, "--out", tmp_path / "o"
        )
        assert result.code == 1, result.output
        assert "The rule tables cannot be read." in result.err
        assert str(missing) in result.err
        assert "Traceback" not in result.output
    assert not (tmp_path / "o").exists()


def test_a_missing_draw_file_is_named(panels, tmp_path):
    missing = tmp_path / "none-draw.json"
    result = panels("write", "--out", tmp_path / "out", "--draw", missing)
    assert result.code == 1
    assert str(missing) in result.err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "content", [None, "{not json", '{"roles": []}'], ids=["missing", "not-json", "no-roles"]
)
def test_an_unreadable_theme_is_named(panels, tmp_path, content):
    theme = tmp_path / "theme.json"
    if content is not None:
        theme.write_text(content)
    for verb in ("write", "check"):
        result = panels(verb, "--out", tmp_path / "out", "--theme", theme)
        assert result.code == 1, result.output
        assert "The theme cannot be used." in result.err
        assert str(theme) in result.err
    assert not (tmp_path / "out").exists()


def _assert_refused(result: Run) -> None:
    """Refused as an unreadable facility: the reason and the rebuild hint, exit 1, no traceback."""
    assert result.code == 1, result.output
    assert "The facility cannot be read." in result.err
    assert "rebuild with osprey build" in result.err
    assert "Traceback" not in result.output


def test_a_facility_3_document_is_refused_with_the_rebuild_hint(run, tmp_path):
    def older(document) -> None:
        document["schema"] = "osprey.facility.facility/3"

    facility = write_fixture_facility(tmp_path / "facility.json", older)
    for verb in ("check", "write"):
        out = tmp_path / f"out-{verb}"
        result = run(verb, "--facility", facility, "--tables", PANELS_DIR, "--out", out)
        _assert_refused(result)
        assert "osprey.facility.facility/3" in result.err
    assert not (tmp_path / "out-write" / "index.json").exists()


def test_a_facility_4_document_without_relation_types_is_refused(run, tmp_path):
    def without(document) -> None:
        document.pop("relation_types", None)

    facility = write_fixture_facility(tmp_path / "facility.json", without)
    result = run("check", "--facility", facility, "--tables", PANELS_DIR, "--out", tmp_path / "o")
    _assert_refused(result)
    assert "relation_types" in result.err


def test_a_corrupt_index_stops_write_and_check(panels, tmp_path):
    out = tmp_path / "out"
    assert panels("write", "--out", out).code == 0
    (out / "index.json").write_text("{nope")
    written = panels("write", "--out", out)
    assert written.code == 1 and "index.json" in written.err
    checked = panels("check", "--out", out)
    assert checked.code == 1 and "invalid: 1" in checked.out


# -- draw and theme -------------------------------------------------------------------


def test_a_draw_file_is_read_and_part_of_the_stamp(panels, tmp_path):
    draw = tmp_path / "draw.json"
    draw.write_text(json.dumps({"colours": {"quadrupole": "#2f9e52"}}))
    out = tmp_path / "out"
    assert panels("write", "--out", out, "--draw", draw).code == 0
    assert panels("check", "--out", out, "--draw", draw).code == 0
    plain = panels("check", "--out", out)
    assert plain.code == 1
    assert "stamp:" in plain.out and "draw settings" in plain.out


def test_draw_json_beside_the_tables_is_the_default(panels, tmp_path):
    tables = tmp_path / "tables"
    shutil.copytree(PANELS_DIR, tables)
    (tables / "draw.json").unlink()  # the fixture's own settings; this test writes its own
    result = panels("write", "--out", tmp_path / "a", tables=tables)
    assert result.code == 0, result.output
    assert "no draw.json" in result.out
    (tables / "draw.json").write_text(json.dumps({"colours": {"quadrupole": "#2f9e52"}}))
    result = panels("write", "--out", tmp_path / "b", tables=tables)
    assert result.code == 0 and "no draw.json" not in result.output
    index = json.loads((tmp_path / "b" / "index.json").read_text())
    other = json.loads((tmp_path / "a" / "index.json").read_text())
    assert index["draw_hash"] != other["draw_hash"]


def test_a_bad_draw_file_stops_the_command(panels, tmp_path):
    draw = tmp_path / "draw.json"
    draw.write_text(json.dumps({"colours": {"no_such_symbol": "#2f9e52"}}))
    result = panels("write", "--out", tmp_path / "out", "--draw", draw)
    assert result.code == 1 and "no_such_symbol" in result.err
    assert not (tmp_path / "out").exists()


def test_a_draw_colour_near_a_state_colour_stops_the_command(panels, tmp_path):
    draw = tmp_path / "draw.json"
    draw.write_text(json.dumps({"colours": {"bend": "#c4162a"}}))  # the bad state's red
    result = panels("write", "--out", tmp_path / "out", "--draw", draw)
    assert result.code == 1 and "too near the state colour MAJOR" in result.err
    checked = panels("check", "--out", tmp_path / "out", "--draw", draw)
    assert checked.code == 1 and "too near the state colour MAJOR" in checked.err


def test_a_draw_colour_near_a_theme_state_colour_is_refused_only_with_the_theme(panels, tmp_path):
    draw = tmp_path / "draw.json"
    draw.write_text(json.dumps({"colours": {"quadrupole": THEME_ONLY_MINOR}}))
    themed = panels("write", "--out", tmp_path / "themed", "--draw", draw, "--theme", DEMO_THEME)
    assert themed.code == 1 and "too near the state colour MINOR" in themed.err
    assert not (tmp_path / "themed").exists()
    plain = panels("write", "--out", tmp_path / "plain", "--draw", draw)
    assert plain.code == 0, plain.output


def test_the_demo_theme_writes_and_checks_the_same_screens(panels, tmp_path):
    themed, plain = tmp_path / "themed", tmp_path / "plain"
    assert panels("write", "--out", themed, "--theme", DEMO_THEME).code == 0
    assert panels("write", "--out", plain).code == 0
    checked = panels("check", "--out", themed, "--theme", DEMO_THEME)
    assert checked.code == 0, checked.output

    def files(root: Path) -> dict[str, bytes]:
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    assert files(themed) == files(plain)  # the theme is a reference, never drawn from


# -- reports --------------------------------------------------------------------------


def test_a_tampered_index_entry_is_reported_on_stderr_and_left_alone(panels, tmp_path):
    out = tmp_path / "out"
    assert panels("write", "--out", out).code == 0
    (out / "mine.txt").write_text("1")
    index = json.loads((out / "index.json").read_text())
    index["screens"].append({"file": "mine.txt"})
    (out / "index.json").write_text(json.dumps(index))
    result = panels("write", "--out", out)
    assert result.code == 0 and "mine.txt" in result.err
    assert (out / "mine.txt").read_text() == "1"


def test_a_value_the_graph_cannot_read_is_counted_and_nothing_stops(run, tmp_path):
    def odd_unit(document) -> None:
        for claim in document["relations"]:
            if claim["id"] == "xr02-ch5-rack2":
                claim["object"]["unit"] = "A5"

    facility = write_fixture_facility(tmp_path / "facility.json", odd_unit)
    result = run("write", "--facility", facility, "--tables", PANELS_DIR, "--out", tmp_path / "o")
    assert result.code == 0, result.output
    assert "1 values of the facility's rows cannot be read" in result.out


def test_a_part_of_an_excluded_thing_is_counted_as_withheld(run, tmp_path):
    def part(document) -> None:
        add_part(document, "XR02:SAF1:CARD", "XR02:SAF1")

    facility = write_fixture_facility(tmp_path / "facility.json", part)
    result = run("write", "--facility", facility, "--tables", PANELS_DIR, "--out", tmp_path / "o")
    assert result.code == 0, result.output
    assert "1 things withheld as part of an excluded thing" in result.out


def test_every_view_kind_of_the_engine_can_be_chosen(panels, tmp_path):
    assert cli.VIEW_CHOICES == VIEW_KINDS
    out = tmp_path / "out"
    written = panels("write", "--out", out, "--kind", "calibration")
    assert written.code == 0, written.output
    assert "calibration:" in written.out
    assert list((out / "calibration").iterdir())
    refused = panels("write", "--out", out, "--kind", "no-such-view")
    assert refused.code == 2 and "--kind" in refused.err


def test_check_notes_the_rows_that_restate_the_facility_file(panels, facility_file, tmp_path):
    from osprey_panels import load_graph
    from osprey_panels.tables import COLUMNS

    tables = tmp_path / "panels"
    shutil.copytree(PANELS_DIR, tables)
    roles_header = ",".join(COLUMNS["channel_roles.csv"])
    restating = "Quadrupole,current_readback,*:Current-RB,device,value,,,,,stated,S,1,l"
    # drawn as its signal's default is, but a stated entry is a press, and a default
    # role is none: without the row the card loses the entry
    pressing = "Quadrupole,current_setpoint,*:Current-SP,device,entry,,,,,stated,S,1,l"
    # last: without it, its channel plays the same role in the same place of the order
    (tables / "channel_roles.csv").write_text(f"{roles_header}\n{pressing}\n{restating}\n")

    out = tmp_path / "out"
    assert panels("write", "--out", out, tables=tables).code == 0
    result = panels("check", "--out", out, tables=tables)
    assert "redundant: channel_roles.csv: 1 rows restate the facility file or the default" in (
        result.out
    )
    # a record type is read: only the drawn_as that is the role's may go
    assert "redundant: record_types.csv:" not in result.out
    assert "redundant: record_types.csv drawn_as: " in result.out
    # deleting exactly what the notes name leaves every screen as it was
    graph = load_graph(facility_file, tables)
    rows = (tables / "record_types.csv").read_text().splitlines()
    kept = [rows[0]]
    for line in rows[1:]:
        cells = line.split(",")
        try:
            default = "setpoint" if graph.channel(cells[0]).writable else "value"
        except KeyError:  # a channel the graph does not hold: the row is not read
            default = None
        if cells[2] == default:
            cells[2] = ""
        kept.append(",".join(cells))
    (tables / "record_types.csv").write_text("\n".join(kept) + "\n")
    (tables / "channel_roles.csv").write_text(f"{roles_header}\n{pressing}\n")
    again = tmp_path / "again"
    assert panels("write", "--out", again, tables=tables).code == 0
    result = panels("check", "--out", again, tables=tables)
    assert "redundant: record_types.csv" not in result.out
    assert "redundant: channel_roles.csv" not in result.out

    def screens(root: Path) -> dict[str, str]:
        """Every screen, its stamp of the facility and tables masked."""
        return {
            str(p.relative_to(root)): re.sub(
                r"(Facility |model )[0-9a-f]{8,64}", r"\1-", p.read_text(encoding="utf-8")
            )
            for p in root.rglob("*.screen.json")
        }

    assert screens(again) == screens(out)
    # the row the note does not name: deleting it changes the screens
    (tables / "channel_roles.csv").write_text(f"{roles_header}\n")
    without = tmp_path / "without"
    assert panels("write", "--out", without, tables=tables).code == 0
    assert screens(without) != screens(out)


# -- classes --------------------------------------------------------------------------


def test_classes_prints_the_table_and_the_summary_a_gate_reads(run, panels, tmp_path):
    from osprey_panels.classes import summary, summary_line

    out = tmp_path / "out"
    written = panels("write", "--out", out)
    assert written.code == 0, written.output
    index = json.loads((out / "index.json").read_text())
    table = index["classes"]
    assert index["class_summary"] == summary(table)
    line = f"classes: {summary_line(summary(table))}"
    assert line in written.out
    checked = panels("check", "--out", out)
    assert line in checked.out
    result = run("classes", "--out", out, "--limit", "2", "--differs")
    assert result.code == 0, result.output
    assert result.out.rstrip().endswith(line)
    biggest = min(table, key=lambda name: (-table[name]["members"], name))  # the first row
    assert biggest in result.out
    assert f"... and {len(table) - 2} more" in result.out
    # the table per view kind
    assert re.search(r"^card +\d+ +\d+ +\d+ +\d+ +\d+$", result.out, re.MULTILINE)


def test_classes_differs_lists_each_screen_off_its_standard(run, panels, tmp_path):
    out = tmp_path / "out"
    assert panels("write", "--out", out).code == 0
    # the fixture's members differ only in blocks, which split classes; the verb
    # prints what the index says, so one entry is made off standard here
    path = out / "index.json"
    index = json.loads(path.read_text())
    off = index["screens"][0]
    off["differs"] = ["a calibration section, which the standard has not"]
    path.write_text(json.dumps(index))
    result = run("classes", "--out", out, "--differs")
    assert result.code == 0, result.output
    assert f"{off['file']} ({off['class']}):" in result.out
    assert "a calibration section, which the standard has not" in result.out


def test_classes_without_an_index_fails(run, tmp_path):
    result = run("classes", "--out", tmp_path)
    assert result.code == 1
    assert "No class table" in result.err


def test_the_module_runs_as_the_command(tmp_path):
    import subprocess
    import sys

    from osprey_panels_tests._harness import PACKAGE_SRC

    result = subprocess.run(
        [sys.executable, "-m", "osprey_panels", "classes", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(PACKAGE_SRC)},
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert "No class table" in result.stderr
