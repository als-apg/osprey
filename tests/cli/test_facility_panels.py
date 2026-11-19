"""``osprey facility panels write`` and ``check``: operator screens from the built facility."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from osprey.cli.main import cli
from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility


@pytest.fixture(scope="module")
def facility_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_fixture_facility(tmp_path_factory.mktemp("panels-cli") / "facility.json")


def _run(facility_file: Path, verb: str, *args: str) -> Result:
    return CliRunner().invoke(
        cli,
        [
            "facility",
            "panels",
            verb,
            "--facility",
            str(facility_file),
            "--tables",
            str(PANELS_DIR),
            *args,
        ],
    )


def test_write_prints_the_counts_the_total_and_the_largest(facility_file, tmp_path):
    result = _run(facility_file, "write", "--out", str(tmp_path / "out"), "--top", "3")
    assert result.exit_code == 0, result.output
    index = json.loads((tmp_path / "out" / "index.json").read_text())
    assert f"{len(index['screens'])} screens" in result.output
    for kind in ("card", "place", "rollup", "checkout"):
        assert f"  {kind}: " in result.output
    assert result.output.count(".screen.json") == 3
    biggest = max(index["screens"], key=lambda e: e["bytes"])
    assert biggest["file"] in result.output
    devices = sum(e["kind"] == "card" and e["thing_kind"] == "device" for e in index["screens"])
    assert f"  device cards: {devices} with the default recipe" in result.output  # no recipe rows


def test_check_after_write_exits_0(facility_file, tmp_path):
    out = str(tmp_path / "out")
    assert _run(facility_file, "write", "--out", out).exit_code == 0
    result = _run(facility_file, "check", "--out", out)
    assert result.exit_code == 0, result.output
    assert "match the model" in result.output
    # the classes by how their pictures are stated, before the problems
    assert "pictures, drawing: 0 classes" in result.output
    assert "pictures, parts only: 1 class (Quadrupole)" in result.output
    assert "pictures, none: " in result.output
    assert "pictures, parts not placed: none" in result.output  # information, not a problem
    assert "0 left out (no value stated) on 0 cards" in result.output
    assert "default card without tiles: " in result.output  # no roles: no tile is guessed


def test_check_counts_the_default_and_the_devices_each_cards_row_decides(facility_file, tmp_path):
    import shutil

    tables = tmp_path / "panels"
    shutil.copytree(PANELS_DIR, tables)
    (tables / "cards.csv").write_text(
        "action,class,under,devices,attribute,when,owner_relation,owner_direction,owner_to,"
        "reason,source\n"
        "drop,Pump,XR/XR01,,,,,,,drawn on its rack,S\n"
        "keep,*,,XR02:*,,no-channel,,,,worth a card,S\n"
    )

    def run(verb: str, *args: str) -> Result:
        return CliRunner().invoke(
            cli,
            ["facility", "panels", verb, "--facility", str(facility_file), "--tables", str(tables)]
            + list(args),
        )

    out = str(tmp_path / "out")
    assert run("write", "--out", out).exit_code == 0
    result = run("check", "--out", out)
    assert "  by default (no channel, no drawing): " in result.output
    assert "  cards.csv row 1, drop (class Pump, under XR/XR01): 2 devices" in result.output
    assert "  cards.csv row 2, keep (class *, devices XR02:*, when no-channel): " in result.output


def test_check_exits_1_and_counts_each_kind(facility_file, tmp_path):
    out = tmp_path / "out"
    _run(facility_file, "write", "--out", str(out))
    victims = sorted(out.glob("card/*.screen.json"))[:3]
    for victim in victims:
        victim.write_text("{broken")
    (out / "card" / "foreign.txt").write_text("ignored")
    result = _run(facility_file, "check", "--out", str(out), "--limit", "2")
    assert result.exit_code == 1
    assert "invalid: 3" in result.output
    assert "... and 1 more" in result.output
    assert "foreign" not in result.output


def test_kind_and_under_write_a_subset_and_check_notes_the_rest(facility_file, tmp_path):
    out = str(tmp_path / "out")
    written = _run(facility_file, "write", "--out", out, "--kind", "card", "--under", "XR/XR01")
    assert written.exit_code == 0, written.output
    assert "not written" in written.output
    clean = _run(facility_file, "check", "--out", out)
    assert clean.exit_code == 0, clean.output
    assert "outside_set:" in clean.output
    strict = _run(facility_file, "check", "--out", out, "--strict")
    assert strict.exit_code == 1


def test_max_bytes_check_lists_the_oversize(facility_file, tmp_path):
    out = str(tmp_path / "out")
    assert _run(facility_file, "write", "--out", out).exit_code == 0
    result = _run(facility_file, "check", "--out", out, "--max-bytes", "1")
    assert result.exit_code == 1 and "oversize:" in result.output


def test_write_refuses_a_non_empty_directory_without_an_index(facility_file, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "mine.txt").write_text("x")
    result = _run(facility_file, "write", "--out", str(tmp_path / "out"))
    assert result.exit_code == 1
    assert "index.json" in result.output
    assert (tmp_path / "out" / "mine.txt").read_text() == "x"


def test_an_unknown_place_is_refused(facility_file, tmp_path):
    result = _run(facility_file, "write", "--out", str(tmp_path / "out"), "--under", "NOWHERE")
    assert result.exit_code == 1 and "NOWHERE" in result.output
    assert not (tmp_path / "out" / "index.json").exists()


def test_a_missing_facility_file_says_so(tmp_path):
    result = _run(tmp_path / "none.json", "check", "--out", str(tmp_path))
    assert result.exit_code == 1
    assert "cannot be read" in result.output


def test_a_corrupt_index_stops_write_and_check(facility_file, tmp_path):
    out = tmp_path / "out"
    assert _run(facility_file, "write", "--out", str(out)).exit_code == 0
    (out / "index.json").write_text("{nope")
    written = _run(facility_file, "write", "--out", str(out))
    assert written.exit_code == 1 and "index.json" in written.output
    checked = _run(facility_file, "check", "--out", str(out))
    assert checked.exit_code == 1 and "invalid: 1" in checked.output


def test_a_draw_file_is_read_and_part_of_the_stamp(facility_file, tmp_path):
    draw = tmp_path / "draw.json"
    draw.write_text(json.dumps({"colours": {"quadrupole": "#2f9e52"}}))
    out = str(tmp_path / "out")
    assert _run(facility_file, "write", "--out", out, "--draw", str(draw)).exit_code == 0
    assert _run(facility_file, "check", "--out", out, "--draw", str(draw)).exit_code == 0
    plain = _run(facility_file, "check", "--out", out)
    assert plain.exit_code == 1
    assert "stamp:" in plain.output and "draw settings" in plain.output


def test_draw_json_beside_the_tables_is_the_default(facility_file, tmp_path):
    import shutil

    tables = tmp_path / "tables"
    shutil.copytree(PANELS_DIR, tables)
    (tables / "draw.json").unlink()  # the fixture's own settings; this test writes its own
    result = CliRunner().invoke(
        cli,
        [
            "facility",
            "panels",
            "write",
            "--facility",
            str(facility_file),
            "--tables",
            str(tables),
            "--out",
            str(tmp_path / "a"),
        ],
    )
    assert "no draw.json" in result.output
    (tables / "draw.json").write_text(json.dumps({"colours": {"quadrupole": "#2f9e52"}}))
    result = CliRunner().invoke(
        cli,
        [
            "facility",
            "panels",
            "write",
            "--facility",
            str(facility_file),
            "--tables",
            str(tables),
            "--out",
            str(tmp_path / "b"),
        ],
    )
    assert result.exit_code == 0 and "no draw.json" not in result.output
    index = json.loads((tmp_path / "b" / "index.json").read_text())
    other = json.loads((tmp_path / "a" / "index.json").read_text())
    assert index["draw_hash"] != other["draw_hash"]


def test_a_bad_draw_file_stops_the_command(facility_file, tmp_path):
    draw = tmp_path / "draw.json"
    draw.write_text(json.dumps({"colours": {"no_such_symbol": "#2f9e52"}}))
    result = _run(facility_file, "write", "--out", str(tmp_path / "out"), "--draw", str(draw))
    assert result.exit_code == 1 and "no_such_symbol" in result.output
    assert not (tmp_path / "out").exists()


def test_a_tampered_index_entry_is_reported_on_stderr_and_left_alone(facility_file, tmp_path):
    out = tmp_path / "out"
    assert _run(facility_file, "write", "--out", str(out)).exit_code == 0
    (out / "mine.txt").write_text("1")
    index = json.loads((out / "index.json").read_text())
    index["screens"].append({"file": "mine.txt"})
    (out / "index.json").write_text(json.dumps(index))
    result = _run(facility_file, "write", "--out", str(out))
    assert result.exit_code == 0 and "mine.txt" in result.output
    assert (out / "mine.txt").read_text() == "1"


def test_a_value_the_graph_cannot_read_is_counted_and_nothing_stops(tmp_path):
    def odd_unit(document) -> None:
        for claim in document["relations"]:
            if claim["id"] == "xr02-ch5-rack2":
                claim["object"]["unit"] = "A5"

    facility = write_fixture_facility(tmp_path / "facility.json", odd_unit)
    result = _run(facility, "write", "--out", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert "1 values of the facility's rows cannot be read" in result.output


def test_a_part_of_an_excluded_thing_is_counted_as_withheld(tmp_path):
    from tests.facility.panels._fixture import add_part

    def part(document) -> None:
        add_part(document, "XR02:SAF1:CARD", "XR02:SAF1")

    facility = write_fixture_facility(tmp_path / "facility.json", part)
    result = _run(facility, "write", "--out", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert "1 things withheld as part of an excluded thing" in result.output


def test_a_draw_colour_near_a_state_colour_stops_the_command(facility_file, tmp_path):
    draw = tmp_path / "draw.json"
    draw.write_text(json.dumps({"colours": {"bend": "#c4162a"}}))  # the bad state's red
    result = _run(facility_file, "write", "--out", str(tmp_path / "out"), "--draw", str(draw))
    assert result.exit_code == 1 and "too near the state colour MAJOR" in result.output
    checked = _run(facility_file, "check", "--out", str(tmp_path / "out"), "--draw", str(draw))
    assert checked.exit_code == 1 and "too near the state colour MAJOR" in checked.output


def test_every_view_kind_of_the_engine_can_be_chosen(facility_file, tmp_path):
    from osprey.cli.facility_cmd import _PANEL_VIEWS
    from osprey.facility.panels.view import VIEW_KINDS

    assert tuple(_PANEL_VIEWS) == VIEW_KINDS
    out = tmp_path / "out"
    written = _run(facility_file, "write", "--out", str(out), "--kind", "calibration")
    assert written.exit_code == 0, written.output
    assert "calibration:" in written.output
    assert list((out / "calibration").iterdir())


def test_check_notes_the_rows_that_restate_the_facility_file(facility_file, tmp_path):
    import shutil

    tables = tmp_path / "panels"
    shutil.copytree(PANELS_DIR, tables)
    from osprey.facility.panels.tables import COLUMNS

    roles_header = ",".join(COLUMNS["channel_roles.csv"])
    restating = "Quadrupole,current_readback,*:Current-RB,device,value,,,,,stated,S,1,l"
    # drawn as its signal's default is, but a stated entry is a press, and a default
    # role is none: without the row the card loses the entry
    pressing = "Quadrupole,current_setpoint,*:Current-SP,device,entry,,,,,stated,S,1,l"
    # last: without it, its channel plays the same role in the same place of the order
    (tables / "channel_roles.csv").write_text(f"{roles_header}\n{pressing}\n{restating}\n")

    def run(verb: str, *args: str) -> Result:
        return CliRunner().invoke(
            cli,
            ["facility", "panels", verb, "--facility", str(facility_file), "--tables", str(tables)]
            + list(args),
        )

    out = str(tmp_path / "out")
    assert run("write", "--out", out).exit_code == 0
    result = run("check", "--out", out)
    assert "redundant: channel_roles.csv: 1 rows restate the facility file or the default" in (
        result.output
    )
    # a record type is read: only the drawn_as that is the role's may go
    assert "redundant: record_types.csv:" not in result.output
    assert "redundant: record_types.csv drawn_as: " in result.output
    # deleting exactly what the notes name leaves every screen as it was
    from osprey.facility.panels import load_graph

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
    again = str(tmp_path / "again")
    assert run("write", "--out", again).exit_code == 0
    result = run("check", "--out", again)
    assert "redundant: record_types.csv" not in result.output
    assert "redundant: channel_roles.csv" not in result.output

    def screens(root: str) -> dict[str, str]:
        """Every screen, its stamp of the facility and tables masked."""
        return {
            str(p.relative_to(root)): re.sub(
                r"(Facility |model )[0-9a-f]{8,64}", r"\1-", p.read_text(encoding="utf-8")
            )
            for p in Path(root).rglob("*.screen.json")
        }

    assert screens(again) == screens(out)
    # the row the note does not name: deleting it changes the screens
    (tables / "channel_roles.csv").write_text(f"{roles_header}\n")
    without = str(tmp_path / "without")
    assert run("write", "--out", without).exit_code == 0
    assert screens(without) != screens(out)
