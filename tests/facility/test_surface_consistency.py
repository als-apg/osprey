"""Every consumer surface of the demo facility agrees with it, section by section.

The facility file is the one source each surface is derived from; this module
holds, per section of the demo machine, the joins no single-surface test makes:

* the recorder's channel set, as the archiver recorder resolves it from the
  simulator view, records exactly the section's scalar channels as numbers and
  exactly its waveform channels in the waveform collection;
* every benchmark target of the section is a channel of the facility, and the
  section answers the number of added benchmark queries its row declares (a
  query is *added* when it targets a channel beyond the frozen fingerprint);
* every class a section's devices carry, every machine-level place and every
  family of the section has a description, and the section's new classes,
  places and families are present.

The expectations live in :data:`SECTIONS`, one row per section, so a section
the demo gains is one more row. Each check is a function that returns what it
finds wrong, and each is shown to fire on a mutated copy of its input.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from osprey.facility.validate import vocabulary
from osprey.services.archiver_recorder.config import ChannelSet, resolve_channel_addresses
from osprey_connectors.simulation.values import is_waveform
from tests.facility.test_cf_view_parity import addition_addresses, pre_line_queries

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

pytestmark = [pytest.mark.slow]

#: The served waveform channel of the storage ring's SEK injection septum.
SEK_WAVEFORM = "SR:MAG:SEPTUM:01:CURRENT:WAVEFORM"

#: The render-relative directory of the simulator view the recorder reads.
SIMULATOR_PREFIX = "data/simulator/"


@dataclass(frozen=True)
class Section:
    """What one section of the demo machine must show on every surface.

    Attributes:
        waveforms: The section's channels the recorder records as traces.
        added_queries: How many benchmark queries target a channel of the
            section that lies beyond the frozen fingerprint.
        new_classes: Device classes the section introduces.
        new_places: Places the section introduces.
        new_families: Families (groups) the section introduces.
    """

    waveforms: frozenset[str] = frozenset()
    added_queries: int = 0
    new_classes: tuple[str, ...] = ()
    new_places: tuple[str, ...] = ()
    new_families: tuple[str, ...] = ()


#: The per-section expectations, keyed by the section's address prefix.
SECTIONS: dict[str, Section] = {
    "SR": Section(
        waveforms=frozenset({SEK_WAVEFORM}),
        added_queries=1,
        new_classes=("Septum",),
        new_families=("SR/SEPTUM",),
    ),
    "BR": Section(
        new_classes=("Kicker", "Septum", "ScreenDiagnostic"),
        new_families=(
            "BR/DIPOLE-RAMP",
            "BR/KICKER",
            "BR/QD-RAMP",
            "BR/QF-RAMP",
            "BR/SEPTUM",
            "BR/TV",
        ),
    ),
    # Two queries ask about BTS channels the transfer line gained; two older
    # aggregate queries (all corrector setpoints, all quadrupole readbacks) now
    # also reach those channels, so four queries count as added.
    "BTS": Section(
        added_queries=4,
        new_classes=("ScreenDiagnostic",),
        new_families=(
            "BTS/DIPOLE",
            "BTS/ICT",
            "BTS/PADDLES",
            "BTS/QD",
            "BTS/QF",
            "BTS/TV",
        ),
    ),
    "LINE": Section(
        added_queries=1,
        new_families=("LINE/QD", "LINE/QF"),
    ),
    "GTL": Section(
        added_queries=2,
        new_classes=(
            "BuckingCoil",
            "ElectronGun",
            "ScreenDiagnostic",
            "Solenoid",
            "SubHarmonicBuncher",
        ),
        new_places=("GTL",),
        new_families=(
            "GTL/BC",
            "GTL/BPM",
            "GTL/EGUN",
            "GTL/HCM",
            "GTL/ICT",
            "GTL/SHB",
            "GTL/SOL",
            "GTL/TV",
            "GTL/VCM",
        ),
    ),
    "LN": Section(
        added_queries=2,
        new_classes=(
            "BunchingCavity",
            "RadioFrequency",
            "ScreenDiagnostic",
            "Solenoid",
            "TravellingWaveCavity",
        ),
        new_places=("LN",),
        new_families=(
            "LN/ACC",
            "LN/BPM",
            "LN/BUNCHER",
            "LN/HCM",
            "LN/ICT",
            "LN/MASTER",
            "LN/QD",
            "LN/QF",
            "LN/SOL",
            "LN/TV",
            "LN/VCM",
            "LN/WATER",
        ),
    ),
    "LTB": Section(
        added_queries=3,
        new_classes=("ScreenDiagnostic",),
        new_places=("LTB",),
        new_families=(
            "LTB/BPM",
            "LTB/DIPOLE",
            "LTB/HCM",
            "LTB/ICT",
            "LTB/QD",
            "LTB/QF",
            "LTB/TV",
            "LTB/VCM",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Sections of a record
# ---------------------------------------------------------------------------


def address_section(address: str) -> str:
    """The section of a channel address: its first ``:`` segment."""
    return address.split(":", 1)[0]


def id_section(record_id: str) -> str:
    """The section of a device, place or group id: its first ``/`` segment."""
    return record_id.split("/", 1)[0]


# ---------------------------------------------------------------------------
# Checks: each returns what it finds wrong, empty when the section agrees
# ---------------------------------------------------------------------------


def recorder_gaps(
    section: str, expected: Section, channel_set: ChannelSet, facility: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Where the recorder's set disagrees with the section's facility channels."""
    channels = [c for c in facility["channels"] if address_section(c["id"]) == section]
    waveforms = {c["id"] for c in channels if is_waveform(c.get("value_type"))}
    scalars = {c["id"] for c in channels} - waveforms
    recorded_scalars = {a for a in channel_set.scalars if address_section(a) == section}
    recorded_waveforms = {a for a in channel_set.waveforms if address_section(a) == section}
    gaps = {
        "scalars not recorded": sorted(scalars - recorded_scalars),
        "recorded as scalars but not scalar channels": sorted(recorded_scalars - scalars),
        "waveforms not recorded as traces": sorted(waveforms - recorded_waveforms),
        "recorded as traces but not waveform channels": sorted(recorded_waveforms - waveforms),
        "declared waveforms the facility lacks": sorted(expected.waveforms - waveforms),
        "waveform channels the row does not declare": sorted(waveforms - expected.waveforms),
    }
    return {problem: found for problem, found in gaps.items() if found}


def benchmark_gaps(
    section: str,
    expected: Section,
    queries: Iterable[Mapping[str, Any]],
    facility: Mapping[str, Any],
    additions: set[str],
) -> dict[str, Any]:
    """Where the section's benchmark queries disagree with the facility."""
    channels = {c["id"] for c in facility["channels"]}
    missing: list[str] = []
    added = 0
    for query in queries:
        targets = [t for t in query["targeted_pv"] if address_section(t) == section]
        missing.extend(f"{query['user_query']}: {t}" for t in targets if t not in channels)
        if any(t in additions for t in targets):
            added += 1
    gaps: dict[str, Any] = {}
    if missing:
        gaps["targets the facility lacks"] = missing
    if added != expected.added_queries:
        gaps["added queries"] = {"expected": expected.added_queries, "found": added}
    return gaps


def description_gaps(
    section: str,
    expected: Section,
    facility: Mapping[str, Any],
    class_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """The section's classes, places and families that lack a description."""
    described = {
        str(row["name"]): bool(str(row.get("description") or "").strip()) for row in class_rows
    }
    for row in facility.get("classes", []):
        described[str(row["class"])] = bool(str(row.get("description") or "").strip())

    def has_text(record: Mapping[str, Any]) -> bool:
        return bool(str(record.get("description") or "").strip())

    classes = {d["class"] for d in facility["devices"] if id_section(d["id"]) == section} | set(
        expected.new_classes
    )
    places = {p["id"]: p for p in facility["places"] if id_section(p["id"]) == section}
    families = {g["id"]: g for g in facility["groups"] if id_section(g["id"]) == section}
    machine_places = {pid for pid, p in places.items() if p.get("level") == "machine"}

    gaps = {
        "classes without a description": sorted(c for c in classes if not described.get(c)),
        "places without a description": sorted(
            pid
            for pid in machine_places | set(expected.new_places)
            if pid not in places or not has_text(places[pid])
        ),
        "families without a description": sorted(
            gid
            for gid in set(families) | set(expected.new_families)
            if gid not in families or not has_text(families[gid])
        ),
    }
    return {problem: found for problem, found in gaps.items() if found}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recorder_set(built_control_assistant: BuiltProject, tmp_path_factory) -> ChannelSet:
    """The channel set the recorder resolves from the build's simulator view."""
    root = tmp_path_factory.mktemp("surface") / "simulator"
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith(SIMULATOR_PREFIX):
            target = root / name[len(SIMULATOR_PREFIX) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return resolve_channel_addresses(Path(root))


SECTION_IDS = list(SECTIONS)


# ---------------------------------------------------------------------------
# Every section agrees on every surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section", SECTION_IDS)
def test_the_recorder_records_the_sections_scalars_and_traces_its_waveforms(
    section: str, recorder_set: ChannelSet, built_control_assistant: BuiltProject
) -> None:
    gaps = recorder_gaps(section, SECTIONS[section], recorder_set, built_control_assistant.facility)
    assert gaps == {}, f"{section}: {gaps}"


@pytest.mark.parametrize("section", SECTION_IDS)
def test_the_sections_benchmark_targets_exist_and_its_added_queries_match(
    section: str, built_control_assistant: BuiltProject
) -> None:
    gaps = benchmark_gaps(
        section,
        SECTIONS[section],
        pre_line_queries(),
        built_control_assistant.facility,
        addition_addresses(),
    )
    assert gaps == {}, f"{section}: {gaps}"


@pytest.mark.parametrize("section", SECTION_IDS)
def test_every_class_place_and_family_of_the_section_has_a_description(
    section: str, built_control_assistant: BuiltProject
) -> None:
    gaps = description_gaps(
        section, SECTIONS[section], built_control_assistant.facility, vocabulary()["classes"]
    )
    assert gaps == {}, f"{section}: {gaps}"


# ---------------------------------------------------------------------------
# Each check fires when its surface drifts
# ---------------------------------------------------------------------------


def test_a_waveform_missing_from_the_recorders_trace_set_is_reported(
    recorder_set: ChannelSet, built_control_assistant: BuiltProject
) -> None:
    untraced = ChannelSet(
        list(recorder_set), [a for a in recorder_set.waveforms if a != SEK_WAVEFORM]
    )
    gaps = recorder_gaps("SR", SECTIONS["SR"], untraced, built_control_assistant.facility)
    assert gaps["waveforms not recorded as traces"] == [SEK_WAVEFORM]
    assert gaps["recorded as scalars but not scalar channels"] == [SEK_WAVEFORM]


def test_an_absent_benchmark_target_is_reported(built_control_assistant: BuiltProject) -> None:
    facility = copy.deepcopy(built_control_assistant.facility)
    facility["channels"] = [c for c in facility["channels"] if c["id"] != SEK_WAVEFORM]
    queries = [*pre_line_queries(), {"user_query": "the SEK trace", "targeted_pv": [SEK_WAVEFORM]}]
    gaps = benchmark_gaps("SR", SECTIONS["SR"], queries, facility, addition_addresses())
    assert gaps["targets the facility lacks"] == [f"the SEK trace: {SEK_WAVEFORM}"]


def test_a_dropped_added_query_is_reported(built_control_assistant: BuiltProject) -> None:
    additions = addition_addresses()
    queries = [
        q
        for q in pre_line_queries()
        if not any(t in additions and address_section(t) == "SR" for t in q["targeted_pv"])
    ]
    gaps = benchmark_gaps(
        "SR", SECTIONS["SR"], queries, built_control_assistant.facility, additions
    )
    assert gaps == {"added queries": {"expected": 1, "found": 0}}


def test_a_septum_class_without_a_description_is_reported(
    built_control_assistant: BuiltProject,
) -> None:
    class_rows = copy.deepcopy(vocabulary()["classes"])
    for row in class_rows:
        if row["name"] == "Septum":
            row["description"] = ""
    gaps = description_gaps("SR", SECTIONS["SR"], built_control_assistant.facility, class_rows)
    assert gaps == {"classes without a description": ["Septum"]}


def test_a_new_family_without_a_description_is_reported(
    built_control_assistant: BuiltProject,
) -> None:
    facility = copy.deepcopy(built_control_assistant.facility)
    for group in facility["groups"]:
        if group["id"] == "SR/SEPTUM":
            del group["description"]
    gaps = description_gaps("SR", SECTIONS["SR"], facility, vocabulary()["classes"])
    assert gaps == {"families without a description": ["SR/SEPTUM"]}
