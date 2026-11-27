"""Every pinned demo-facility count, named in one census.

``golden/demo_census.json`` lists each file that states a count of the demo
facility -- channels, devices, groups, setpoints, moving channels, benchmark
queries, logbook entries -- with the text that states it and the value it
states. Each file must still state its value, and the built demo facility
must hold the census' own counts. A test constant does not restate a count: it
reads it from the census by name, through ``tests/_census.py``. ``REPRODUCE``
rewrites the census's counts from a built demo, and with them the in-process
nominal golden; a text pin moves by hand. A change to the demo's size therefore
updates the census and every pin in the same commit; a pin that is moved
without the census, or a census that is moved without the pins, fails here.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests._census import census
from tests.facility.test_cf_view_parity import load_golden

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]
CENSUS = "demo_census.json"
CENSUS_PATH = REPO_ROOT / "tests/facility/golden" / CENSUS

#: The command that rewrites the census's counts from a built demo.
REPRODUCE = "uv run python -m tests.facility.test_demo_census --write"

#: The NARAD property namespace the graph view binds as ``narad_p:``.
NARAD_PROPERTY = "https://narad.example.org/property/"

#: The class every magnet's type rolls up to through ``rdfs:subClassOf``.
MAGNET_CLASS = "https://narad.example.org/schema/shared_semantics/Magnet"

#: The devices whose type rolls up to a class, bound or not, as the store counts them.
MAGNET_ROLLUP = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT (COUNT(DISTINCT ?device) AS ?n)
WHERE { ?cls rdfs:subClassOf* ?magnet . ?device a ?cls }
"""

#: The device prose predicates whose distinct texts the census counts.
DEVICE_TEXT_PREDICATES = ("familyDescription", "systemDescription")


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
    assert census["_reproduce"] == REPRODUCE
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


def facility_counts(facility: dict[str, Any]) -> dict[str, int]:
    """The census's ``facility`` block, counted off a built facility file."""
    channels = facility["channels"]
    setpoints = [channel for channel in channels if channel["role"] == "setpoint"]
    return {
        "channels": len(channels),
        "devices": len(facility["devices"]),
        "groups": len(facility["groups"]),
        "setpoints": len(setpoints),
        "readables": len(channels) - len(setpoints),
    }


def family_facts(facility: dict[str, Any]) -> dict[str, Any]:
    """The census facts the middle-layer families state, by System.

    ``groups_by_system`` names the Families a group makes, ``derived_by_system``
    the Families a class makes for the devices no single-class group holds.
    """
    from osprey.facility.views.channel_finder import _families_by_system
    from osprey_facility.records import FacilityRecords

    families = _families_by_system(FacilityRecords(facility))
    return {
        f"{kind}_by_system": {
            system: sorted(family.name for family in by_system if family.record_kind == record_kind)
            for system, by_system in sorted(families.items())
        }
        for kind, record_kind in (("groups", "group"), ("derived", "class"))
    }


def graph_facts(view: Path) -> dict[str, Any]:
    """The census facts the demo's graph view states, parsed as the index parses it."""
    from rdflib import Graph, URIRef

    from osprey.services.channel_finder.graph_index.builder import EDGE_READS, parse_corpus

    parsed = parse_corpus(view.read_text(encoding="utf-8"))
    classes = {row.name: row for row in parsed.class_rows}
    graph = Graph()
    graph.parse(view, format="turtle")

    def pairs(local_name: str) -> list[tuple[Any, Any]]:
        return list(graph.subject_objects(URIRef(NARAD_PROPERTY + local_name)))

    (magnets,) = graph.query(MAGNET_ROLLUP, initBindings={"magnet": URIRef(MAGNET_CLASS)})

    return {
        "bound_devices": len({row.device_uri for row in parsed.binding_rows}),
        "bound_magnets": classes["Magnet"].rollup_devices,
        "distinct_device_texts": {
            predicate: len({str(text) for _, text in pairs(predicate)})
            for predicate in DEVICE_TEXT_PREDICATES
        },
        "family_described": len({subject for subject, _ in pairs("familyDescription")}),
        "graph_classes": len(parsed.class_rows),
        "graph_read_only": sum(1 for row in parsed.binding_rows if row.edges == [EDGE_READS]),
        "graph_sections": len(parsed.section_codes),
        "graph_signals": parsed.signal_count,
        "magnets": int(magnets[0]),
        "system_described": len({subject for subject, _ in pairs("systemDescription")}),
        "typed_devices": classes["AcceleratorDevice"].rollup_devices,
    }


@pytest.mark.slow
def test_the_built_facility_holds_the_census_counts(
    built_demo: BuiltProject,
) -> None:
    assert facility_counts(built_demo.facility) == _census()["facility"]


@pytest.mark.slow
def test_the_built_demo_states_the_census_facts(built_demo: BuiltProject) -> None:
    document = _census()
    recaptured = json.loads(json.dumps(document))
    moved = [line for line in recapture(built_demo, recaptured) if not line.startswith("? ")]
    assert moved == [], f"the census is stale; run {REPRODUCE}"
    assert recaptured == document


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


# --- the recapture -----------------------------------------------------------


def recapture(built: BuiltProject, document: dict[str, Any]) -> list[str]:
    """Rewrite ``document``'s counts from ``built``; one line per moved or kept value."""
    lines: list[str] = []
    produced = {
        "facility": facility_counts(built.facility),
        "facts": {
            **family_facts(built.facility),
            **graph_facts(built.build_dir / "data" / "graph" / "facility.ttl"),
        },
    }
    for block, values in produced.items():
        for name, value in values.items():
            old = document[block].get(name)
            if old != value:
                lines.append(f"~ {name} {json.dumps(old)} -> {json.dumps(value)}")
    lines.extend(
        f"? {name} kept (no producer here)"
        for name in sorted(set(document["facts"]) - set(produced["facts"]))
    )
    document["_reproduce"] = REPRODUCE
    document["facility"] = produced["facility"]
    document["facts"] = dict(sorted({**document["facts"], **produced["facts"]}.items()))
    return lines


def _write() -> None:
    from tests._builds import build_demo
    from tests.facility.test_nominal_goldens import GOLDEN, capture

    document = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="demo-census-") as root:
        built = build_demo(Path(root))
        lines = recapture(built, document)
        nominals = capture(built)
    CENSUS_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    GOLDEN.write_text(json.dumps(nominals, indent=2) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    print(f"wrote {CENSUS_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {len(nominals['channels'])} addresses to {GOLDEN.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit(f"usage: {REPRODUCE}")
    _write()
