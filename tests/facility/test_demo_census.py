"""Every pinned demo-facility count, named in one census.

``golden/demo_census.json`` lists each file that states a count of the demo
facility -- channels, devices, groups, setpoints, moving channels, benchmark
queries, logbook entries -- with the text that states it and the value it
states. Each file must still state its value, and the built demo facility
must hold the census' own counts. A test constant does not restate a count: it
reads it from the census by name, through ``tests/_census.py``. A change to the
demo's size therefore updates the census and every pin in the same commit; a
pin that is moved without the census, or a census that is moved without the
pins, fails here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests._census import census
from tests.facility.test_cf_view_parity import load_golden

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]
CENSUS = "demo_census.json"


def _json_value(document: Any, key: str) -> Any:
    match = re.fullmatch(r"len\((\w*)\)", key)
    if match is None:
        return document[key]
    return len(document[match.group(1)] if match.group(1) else document)


def _text_pattern(fragment: str, value: Any) -> re.Pattern[str]:
    before, placeholder, after = fragment.partition("{value}")
    if not placeholder:
        raise ValueError(f"text pin {fragment!r} has no {{value}} placeholder")
    return re.compile(re.escape(before) + rf"(?<!\d){value}(?!\d)" + re.escape(after))


def stated(text: str, pin: str, value: Any, *, suffix: str) -> bool:
    """Whether a file's ``text`` states ``value`` the way ``pin`` says it does."""
    kind, _, spec = pin.partition(":")
    if kind == "json":
        if suffix != ".json":
            raise ValueError(f"json pin {pin!r} on a {suffix} file")
        try:
            return _json_value(json.loads(text), spec) == value
        except (KeyError, TypeError):
            return False
    if kind == "text":
        return _text_pattern(spec, value).search(text) is not None
    raise ValueError(f"unknown pin kind in {pin!r}")


def _census() -> dict[str, Any]:
    return load_golden(CENSUS)


def _entry_id(entry: dict[str, Any]) -> str:
    return f"{entry['file']}::{entry['pin']}"


def test_the_census_is_well_formed() -> None:
    census = _census()
    assert census["_reproduce"] == "hand-maintained: update with the pin in the same commit"
    entries = census["entries"]
    assert entries
    for entry in entries:
        assert set(entry) == {"file", "pin", "value"}, entry
        assert entry["pin"].split(":", 1)[0] in {"json", "text"}, entry
    ids = [_entry_id(entry) for entry in entries]
    assert len(ids) == len(set(ids)), "an entry is listed twice"
    assert set(census["facility"]) == {"channels", "devices", "groups", "setpoints", "readables"}
    assert census["facts"]
    assert not set(census["facts"]) & set(census["facility"]), "a count is held twice"


def test_census_reads_a_fact_and_refuses_an_unknown_name() -> None:
    document = _census()
    assert census("channels") == document["facility"]["channels"]
    assert census("typed_devices") == document["facts"]["typed_devices"]
    with pytest.raises(KeyError, match=r"demo_census\.json holds no count called 'no_such_count'"):
        census("no_such_count")


@pytest.mark.parametrize("entry", _census()["entries"], ids=_entry_id)
def test_every_listed_file_still_states_its_value(entry: dict[str, Any]) -> None:
    path = REPO_ROOT / entry["file"]
    assert path.is_file(), f"{entry['file']} is gone; drop or move its census entries"
    text = path.read_text(encoding="utf-8")
    assert stated(text, entry["pin"], entry["value"], suffix=path.suffix), (
        f"{entry['file']} no longer states {entry['pin']} = {entry['value']!r}; "
        "update the pin and golden/demo_census.json together"
    )


def test_the_census_agrees_with_itself() -> None:
    """The facility block, the facts and the goldens' counts relate as the demo does."""
    census = _census()
    facility = census["facility"]
    facts = census["facts"]
    assert facility["channels"] == facility["setpoints"] + facility["readables"]
    # The graph view types every device; the store and the index count the
    # devices a channel binds, which the facility's unbound devices are not.
    assert facts["typed_devices"] < facility["devices"]
    by_pin = {(entry["file"], entry["pin"]): entry["value"] for entry in census["entries"]}
    fingerprint = "tests/facility/golden/demo_fingerprint.json"
    additions = "tests/facility/golden/demo_fingerprint_additions.json"
    retired = "tests/facility/golden/demo_fingerprint_retired.json"
    assert by_pin[(fingerprint, "json:count")] == facts["manifest_channels"]
    assert by_pin[(retired, "json:len(rows)")] == facts["retired"]
    assert (
        facts["manifest_channels"] - facts["retired"] + by_pin[(additions, "json:len(rows)")]
        == facility["channels"]
    )


@pytest.mark.slow
def test_the_built_facility_holds_the_census_counts(
    built_demo: BuiltProject,
) -> None:
    built = built_demo.facility
    channels = built["channels"]
    setpoints = [channel for channel in channels if channel["role"] == "setpoint"]
    assert {
        "channels": len(channels),
        "devices": len(built["devices"]),
        "groups": len(built["groups"]),
        "setpoints": len(setpoints),
        "readables": len(channels) - len(setpoints),
    } == _census()["facility"]


# --- the checker itself ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "pin", "value", "suffix", "expected"),
    [
        ("    assert len(groups) == 29\n", "text:assert len(groups) == {value}", 29, ".py", True),
        ("    assert len(groups) == 30\n", "text:assert len(groups) == {value}", 29, ".py", False),
        ("    assert len(groups) == 290\n", "text:assert len(groups) == {value}", 29, ".py", False),
        ("the 1532 devices", "text:the {value} devices", 532, ".py", False),
        ('{"count": 2908, "rows": [1, 2]}', "json:count", 2908, ".json", True),
        ('{"count": 2908, "rows": [1, 2]}', "json:len(rows)", 2, ".json", True),
        ('{"count": 2908, "rows": [1, 2]}', "json:len(rows)", 3, ".json", False),
        ("[1, 2, 3]", "json:len()", 3, ".json", True),
        ('{"rows": []}', "json:count", 0, ".json", False),
    ],
)
def test_the_checker_catches_a_moved_value(
    text: str, pin: str, value: Any, suffix: str, expected: bool
) -> None:
    assert stated(text, pin, value, suffix=suffix) is expected


def test_the_checker_refuses_a_malformed_pin() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        stated("x", "text:no placeholder", 1, suffix=".py")
    with pytest.raises(ValueError, match="unknown pin kind"):
        stated("x", "grep:x", 1, suffix=".py")
