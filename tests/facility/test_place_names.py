"""The flat-name rule for place ids, as one pure function."""

from __future__ import annotations

import subprocess
import sys

import pytest

from osprey.facility.places import resolve_place_name

IDS = {"SR", "SR/S01", "BR", "BR/S01", "SR/S01/RACK1"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RACK1", "SR/S01/RACK1"),
        ("S01", ["BR/S01", "SR/S01"]),
        ("SR", "SR"),
        ("01", None),
        ("ACK1", None),
        ("NOWHERE", None),
        ("SR/S01", "SR/S01"),
        ("SR/S01/RACK1", "SR/S01/RACK1"),
        ("S01/RACK1", None),
        ("SR/S02", None),
    ],
)
def test_resolves_names(name, expected):
    assert resolve_place_name(name, IDS) == expected


def test_flat_name_matches_both_the_id_and_a_last_segment():
    assert resolve_place_name("SR", {"SR", "LTB/SR"}) == ["LTB/SR", "SR"]


def test_accepts_any_iterable_including_a_one_shot_generator():
    assert resolve_place_name("S01", iter(sorted(IDS))) == ["BR/S01", "SR/S01"]
    assert resolve_place_name("SR/S01", (i for i in IDS)) == "SR/S01"


def test_duplicate_ids_count_once():
    assert resolve_place_name("RACK1", ["SR/S01/RACK1", "SR/S01/RACK1"]) == "SR/S01/RACK1"


def test_no_ids_resolves_nothing():
    assert resolve_place_name("SR", []) is None


def test_does_not_mutate_the_ids():
    ids = set(IDS)
    resolve_place_name("S01", ids)
    assert ids == IDS


def test_module_does_not_import_the_importer():
    code = (
        "import sys, osprey.facility.places; "
        "bad = [m for m in sys.modules if m.startswith(('osprey.facility.layers', "
        "'osprey.facility.sources', 'osprey.facility.build'))]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
