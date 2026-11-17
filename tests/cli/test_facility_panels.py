"""``osprey facility panels write`` and ``check``: operator screens from the built facility."""

from __future__ import annotations

import json
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


def test_check_after_write_exits_0(facility_file, tmp_path):
    out = str(tmp_path / "out")
    assert _run(facility_file, "write", "--out", out).exit_code == 0
    result = _run(facility_file, "check", "--out", out)
    assert result.exit_code == 0, result.output
    assert "match the model" in result.output


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
