"""The example facility's transfer line ``LINE`` in its records and models.

The generator's ``line`` mode merges the line's place, devices, channels and
groups into ``records/``, its model into ``models.yaml`` and its measurement
file into ``measurement/LINE.yaml``. On a copy of the committed tree with the
line removed it writes the committed tree back, every other record byte-equal;
run twice it changes nothing. It writes no limits record and no seed, and the
hello-world tree holds no line record. The line's groups hold only its own
devices, and its losing kick loses the particle on the deck.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.facility.sources import read_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts/facility_demo/generate.py"
FACILITIES = REPO_ROOT / "src/osprey/templates/facilities"
EXAMPLE = FACILITIES / "example"
HELLO_WORLD = FACILITIES / "hello_world"

#: The files the ``line`` mode merges into, relative to the tree.
MERGED = [
    "records/places.yaml",
    "records/devices.yaml",
    "records/channels.yaml",
    "records/groups.yaml",
    "models.yaml",
]

#: The files the ``line`` mode writes whole.
WRITTEN = ["decks/LINE.json", "measurement/LINE.yaml"]


def _is_line(record_id: str) -> bool:
    return record_id == "LINE" or record_id.startswith(("LINE/", "LINE:"))


def _chunks(text: str) -> list[str]:
    """A top-level YAML list's items, each as its exact text."""
    chunks: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("- ") or not chunks:
            chunks.append(line)
        else:
            chunks[-1] += line
    return chunks


def _chunk_id(chunk: str) -> str:
    (record,) = read_yaml(chunk)
    return str(record.get("id", record.get("name")))


def _without_line(text: str) -> list[str]:
    return [chunk for chunk in _chunks(text) if not _is_line(_chunk_id(chunk))]


def _run_line(tree: Path) -> None:
    subprocess.run(
        [sys.executable, str(GENERATOR), "line", str(tree)],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def stripped(tmp_path: Path) -> Path:
    """A copy of the committed example tree with the line removed."""
    tree = tmp_path / "example"
    shutil.copytree(EXAMPLE, tree)
    for rel in MERGED:
        path = tree / rel
        path.write_text("".join(_without_line(path.read_text(encoding="utf-8"))), "utf-8")
    for rel in WRITTEN:
        (tree / rel).unlink()
    return tree


def test_the_line_mode_writes_the_committed_tree(stripped: Path) -> None:
    _run_line(stripped)
    for rel in [*MERGED, *WRITTEN]:
        assert (stripped / rel).read_bytes() == (EXAMPLE / rel).read_bytes(), rel


def test_the_line_mode_leaves_every_other_record_byte_equal(stripped: Path) -> None:
    before = {rel: (stripped / rel).read_text(encoding="utf-8") for rel in MERGED}
    _run_line(stripped)
    for rel in MERGED:
        after = (stripped / rel).read_text(encoding="utf-8")
        assert _without_line(after) == _chunks(before[rel]), rel


def test_the_line_mode_is_idempotent(tmp_path: Path) -> None:
    tree = tmp_path / "example"
    shutil.copytree(EXAMPLE, tree)
    _run_line(tree)
    _run_line(tree)
    for rel in [*MERGED, *WRITTEN]:
        assert (tree / rel).read_bytes() == (EXAMPLE / rel).read_bytes(), rel


def test_no_limits_record_names_the_line(stripped: Path) -> None:
    limits = read_yaml((EXAMPLE / "limits.yaml").read_text(encoding="utf-8"))
    assert [r["address"] for r in limits["records"] if r["address"].startswith("LINE:")] == []
    before = (stripped / "limits.yaml").read_bytes()
    _run_line(stripped)
    assert (stripped / "limits.yaml").read_bytes() == before


def test_no_seed_names_the_line() -> None:
    seeds = read_yaml((EXAMPLE / "seeds.yaml").read_text(encoding="utf-8"))
    assert [address for address in seeds if address.startswith("LINE:")] == []


def test_hello_world_holds_no_line_record() -> None:
    texts = [path.read_text(encoding="utf-8") for path in sorted(HELLO_WORLD.rglob("*.yaml"))]
    assert texts
    assert not any("LINE" in text for text in texts)
    assert not (HELLO_WORLD / "decks").exists()


def test_the_line_holds_forty_channels_on_twenty_devices() -> None:
    channels = read_yaml((EXAMPLE / "records/channels.yaml").read_text(encoding="utf-8"))
    devices = read_yaml((EXAMPLE / "records/devices.yaml").read_text(encoding="utf-8"))
    line_channels = [c for c in channels if _is_line(c["id"])]
    line_devices = {d["id"] for d in devices if _is_line(d["id"])}
    assert len(line_channels) == 40
    assert len(line_devices) == 20
    assert sum(c.get("role") == "setpoint" for c in line_channels) == 16
    assert {c["on"]["device"] for c in line_channels} == line_devices
    assert not any("attributes" in d or "place" in d for d in devices if _is_line(d["id"]))
    assert not any("names" in c for c in line_channels)


def test_the_line_groups_hold_only_line_devices_and_name_the_measurement() -> None:
    groups = read_yaml((EXAMPLE / "records/groups.yaml").read_text(encoding="utf-8"))
    line_groups = {g["id"]: g["members"] for g in groups if _is_line(g["id"])}
    assert sorted(line_groups) == ["LINE/BPM", "LINE/HCM", "LINE/VCM"]
    assert all(member.startswith("LINE/") for members in line_groups.values() for member in members)
    assert not any(
        member.startswith("LINE/")
        for g in groups
        if not _is_line(g["id"])
        for member in g["members"]
    )
    measurement = read_yaml((EXAMPLE / "measurement/LINE.yaml").read_text(encoding="utf-8"))
    assert measurement["kinds"] == ["orm"]
    assert measurement["groups"] == {"bpm": "LINE/BPM", "hcor": "LINE/HCM", "vcor": "LINE/VCM"}


def test_the_line_place_spans_the_deck_from_its_start_marker() -> None:
    sys.path.insert(0, str(GENERATOR.parent))
    import _line

    places = read_yaml((EXAMPLE / "records/places.yaml").read_text(encoding="utf-8"))
    (place,) = [p for p in places if p["id"] == "LINE"]
    assert place["level"] == "machine"
    assert place["span"] == {"model": "LINE", "from_marker": _line.START_MARKER}


def test_the_losing_kick_loses_the_particle_on_the_deck() -> None:
    at = pytest.importorskip("at")
    import numpy as np

    sys.path.insert(0, str(GENERATOR.parent))
    import _line

    lattice = at.load_lattice(str(EXAMPLE / "decks/LINE.json"))
    lattice.disable_6d()
    names = [element.FamName for element in lattice]

    def lost(kick: float) -> bool:
        line = lattice.deepcopy()
        line[names.index("HCM01")].KickAngle = np.array([kick * _line.CORRECTOR_GAIN, 0.0])
        _, _, info = line.track(np.zeros((6, 1)), nturns=1, refpts=len(line), losses=True)
        return bool(info["loss_map"]["islost"][0])

    assert lost(_line.LOSING_KICK)
    assert not lost(0.0)
