"""The facility-overview generator reproduces the committed screens and refuses what it cannot draw.

``scripts/facility_demo/facility_overview.py`` renders the control-assistant
library's ``facility/overview``, ``br/overview``, the twelve ``sr/sector-NN``
screens and their vacuum and absorber details from the example facility's
channel and device records. A re-run must write the committed bytes, ``check`` must say so, and a
record set the drawing cannot hold must stop the run with a named error before
anything is written.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "facility_demo" / "facility_overview.py"
_SPEC = importlib.util.spec_from_file_location("facility_overview", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
gen = importlib.util.module_from_spec(_SPEC)
# import-time required because scripts/ is not a package: the generator is loaded
# by path and registered before exec so its dataclasses resolve their module.
sys.modules[_SPEC.name] = gen
_SPEC.loader.exec_module(gen)

LIBRARY = (
    REPO_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "canopy"
    / "library"
)
# The straights whose absorbers carry thermocouples, each with an absorber detail.
ABSORBER_STRAIGHTS = (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)
GENERATED = (
    "screens/facility/overview.screen.json",
    "screens/br/overview.screen.json",
    *(f"screens/sr/sector-{n:02d}.screen.json" for n in range(1, 13)),
    *(f"screens/sr/_vacuum-{n:02d}.screen.json" for n in range(1, 13)),
    *(f"screens/sr/_absorbers-{n:02d}.screen.json" for n in ABSORBER_STRAIGHTS),
)
DEVICES = gen.EXAMPLE_RECORDS.with_name(gen.DEVICES_FILE)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return yaml.safe_load(gen.EXAMPLE_RECORDS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def devices() -> dict[str, dict]:
    return {row["id"]: row for row in yaml.safe_load(DEVICES.read_text(encoding="utf-8"))}


def _records(rows: list[dict], drop=lambda name: False, devices=None) -> object:
    if devices is None:
        devices = {row["id"]: row for row in yaml.safe_load(DEVICES.read_text(encoding="utf-8"))}
    return gen.Records({row["id"]: row for row in rows if not drop(row["id"])}, devices)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120
    )


def test_a_rerun_reproduces_the_committed_bytes(rows):
    files = gen.render(_records(rows), LIBRARY)
    assert sorted(files) == sorted(GENERATED)
    for rel, text in files.items():
        assert (LIBRARY / rel).read_bytes() == text.encode("utf-8"), rel


def test_check_passes_on_the_committed_library():
    result = _run("check", "--library", str(LIBRARY))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""


def test_check_names_a_stale_file_and_writes_nothing(tmp_path):
    copy = tmp_path / "library"
    shutil.copytree(LIBRARY, copy)
    target = copy / GENERATED[0]
    target.write_bytes(target.read_bytes() + b"\n")
    before = target.read_bytes()
    result = _run("check", "--library", str(copy))
    assert result.returncode == 1
    assert f"stale: {GENERATED[0]}" in result.stderr
    assert f"stale: {GENERATED[1]}" not in result.stderr
    assert target.read_bytes() == before


def test_write_restores_the_committed_bytes(tmp_path):
    copy = tmp_path / "library"
    shutil.copytree(LIBRARY, copy)
    for rel in GENERATED:
        (copy / rel).unlink()
    result = _run("write", "--library", str(copy))
    assert result.returncode == 0, result.stderr
    for rel in GENERATED:
        assert (copy / rel).read_bytes() == (LIBRARY / rel).read_bytes(), rel


def test_the_output_is_deterministic_ascii_json(rows):
    first = gen.render(_records(rows), LIBRARY)
    assert first == gen.render(_records(list(reversed(rows))), LIBRARY)
    for text in first.values():
        assert text.isascii()
        assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=True) + "\n"


def _ids(node: dict, widgets: list[str], rules: list[str]) -> None:
    widgets.append(node["id"])
    rules.extend(rule["id"] for rule in node.get("rules", []))
    for child in node.get("children", []):
        _ids(child, widgets, rules)


def test_widget_and_rule_ids_are_sequential(rows):
    for text in gen.render(_records(rows), LIBRARY).values():
        widgets: list[str] = []
        rules: list[str] = []
        _ids(json.loads(text)["root"], widgets, rules)
        assert sorted(widgets) == [f"w-{i:03d}" for i in range(1, len(widgets) + 1)]
        assert sorted(rules) == [f"r-{i:03d}" for i in range(1, len(rules) + 1)]


def test_the_pinned_canopy_verdicts_are_the_generator_constants():
    """The viewer appends a tab for ``tab`` and hands a labelled channel's index to rules."""
    assert gen.NAVIGATION_TARGET == "tab"
    assert gen.LABELLED_INPUT_FORM == "index"


@pytest.mark.parametrize(
    ("dropped", "error", "named"),
    [
        (lambda n: n == "SR:OPS:MODE:RB", "MissingRecordError", "SR:OPS:MODE:RB"),
        (
            lambda n: n == "LN:RF:WATER:01:STATUS:ON",
            "MissingRecordError",
            "LN:RF:WATER:01:STATUS:ON",
        ),
        (lambda n: n == "SR:DIAG:BPM:72:STATUS:VALID", "GeometryMismatchError", "71 SR BPMs"),
        (lambda n: n.startswith("SR:MAG:QFA:24:"), "GeometryMismatchError", "23 SR QFA"),
        (
            lambda n: n == "SR:VAC:GAUGE:SR05:PRESSURE:RB",
            "MissingRecordError",
            "SR:VAC:GAUGE:SR05:PRESSURE:RB",
        ),
        (lambda n: n.startswith("SR:MAG:VCM:72:"), "GeometryMismatchError", "71 SR VCM"),
    ],
)
def test_what_the_drawing_cannot_hold_is_a_named_error(rows, tmp_path, dropped, error, named):
    with pytest.raises(getattr(gen, error), match=named):
        gen.render(_records(rows, dropped), LIBRARY)
    records = tmp_path / "channels.yaml"
    records.write_text(yaml.safe_dump([r for r in rows if not dropped(r["id"])]), encoding="utf-8")
    shutil.copy(DEVICES, tmp_path / gen.DEVICES_FILE)
    copy = tmp_path / "library"
    shutil.copytree(LIBRARY, copy)
    before = {rel: (copy / rel).read_bytes() for rel in GENERATED}
    result = _run("write", "--library", str(copy), "--records", str(records))
    assert result.returncode == 2
    assert f"{error}:" in result.stderr and named in result.stderr
    assert {rel: (copy / rel).read_bytes() for rel in GENERATED} == before


def test_a_device_without_its_record_is_a_named_error(rows, devices):
    """A sector places a device by its record; a device the records lack stops the run."""
    without = {d: r for d, r in devices.items() if d != "SR/ION-PUMP40"}
    with pytest.raises(gen.MissingRecordError, match="SR/ION-PUMP40"):
        gen.render(_records(rows, devices=without), LIBRARY)


def test_an_area_target_the_library_lacks_is_refused(rows, tmp_path):
    library = tmp_path / "library"
    shutil.copytree(LIBRARY, library)
    shutil.rmtree(library / "screens" / "bts")
    with pytest.raises(gen.MissingScreenError, match="bts/BTS"):
        gen.render(_records(rows), library)


def test_the_scripts_readme_lists_the_generator():
    readme = (REPO_ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    assert "`facility_demo/facility_overview.py`" in readme
    assert "facility_overview.py check --library" in readme
