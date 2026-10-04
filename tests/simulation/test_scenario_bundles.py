"""Per-scenario bundle contract: each bundle owns the right logbook entries.

The control_assistant scenarios are self-contained bundles — each owns its
telemetry overlay (``scenario.json``) and, optionally, its logbook narrative
(``logbook.json``). These fast tests pin logbook ownership and entry fields
straight from the parsed bundles (no DB), so a misfiled entry or a malformed
relative timestamp is caught here rather than in the expensive e2e judge layer.
"""

import io
import json
import struct
from datetime import datetime
from datetime import time as dtime
from zoneinfo import ZoneInfo

import pytest
from PIL import Image

from osprey.simulation.machine import PlotSpec, ScenarioLogEntry, parse_machine
from osprey.simulation.plots import render_plot_spec
from osprey.utils.relative_time import RelativeTimestamp, resolve_relative_timestamp
from tests.simulation.conftest import TEMPLATE_SIM

#: Upper bound on a picture: a shipped one travels in every wheel and image, and
#: a drawn one is stored in the logbook for every seeded entry.
_MAX_PICTURE_BYTES = 80 * 1024

#: The anchor a drawn picture is checked at; any instant draws the same size.
_ANCHOR = datetime(2026, 9, 28, 9, 30, tzinfo=ZoneInfo("America/Los_Angeles"))


def _shipped_entries() -> dict[str, ScenarioLogEntry]:
    """Entry id -> entry, for every shipped entry that carries pictures."""
    machine_path = TEMPLATE_SIM / "machine.json"
    parsed = parse_machine(json.loads(machine_path.read_text()), machine_path)
    return {
        entry.entry_id: entry
        for scenario in parsed.scenarios.values()
        for entry in scenario.logbook
        if entry.attachments
    }


def _picture_bytes(entry: ScenarioLogEntry) -> list[tuple[str, bytes]]:
    """``(file name, PNG bytes)`` of each picture, drawing plot specs as seeding does."""
    pictures = []
    for item in entry.attachments:
        if isinstance(item, PlotSpec):
            end = resolve_relative_timestamp(entry.when, _ANCHOR)
            pictures.append((item.filename, render_plot_spec(item, end)))
        else:
            pictures.append((item.name, item.read_bytes()))
    return pictures


def _png_chunk_types(data: bytes) -> list[bytes]:
    chunks, offset = [], 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunks.append(data[offset + 4 : offset + 8])
        offset += 12 + length
    return chunks


class TestShippedPictures:
    def test_the_pictured_entries(self):
        entries = _shipped_entries()
        assert {
            entry_id: [
                item.filename if isinstance(item, PlotSpec) else item.name
                for item in entry.attachments
            ]
            for entry_id, entry in entries.items()
        } == {
            "DEMO-011": ["orbit_rms.png"],
            "DEMO-027": ["cavity_temperatures.png"],
            "DEMO-031": ["corrector_bump_test.png"],
        }
        # The two time series are drawn at seed time so they carry the entry's
        # dates; the bump test has no time axis and ships as a file.
        kinds = {
            entry_id: [isinstance(item, PlotSpec) for item in entry.attachments]
            for entry_id, entry in entries.items()
        }
        assert kinds == {"DEMO-011": [True], "DEMO-027": [True], "DEMO-031": [False]}

    @pytest.mark.parametrize("entry_id", ["DEMO-011", "DEMO-027", "DEMO-031"])
    def test_each_picture_is_a_small_clean_png(self, entry_id):
        for name, data in _picture_bytes(_shipped_entries()[entry_id]):
            assert len(data) < _MAX_PICTURE_BYTES, f"{name} is {len(data)} bytes"
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
                assert image.format == "PNG"
            text_chunks = {b"tEXt", b"iTXt", b"zTXt", b"tIME"} & set(_png_chunk_types(data))
            assert not text_chunks, f"{name} carries metadata chunks {text_chunks}"

    def test_the_reversed_bpm_is_named_only_in_the_picture(self):
        spec = json.loads((TEMPLATE_SIM / "scenarios/bpm-polarity/scenario.json").read_text())
        reversed_bpms = [
            name for name, err in spec["physics"]["bpm_errors"].items() if err["polarity"] == -1
        ]
        entries = json.loads((TEMPLATE_SIM / "scenarios/bpm-polarity/logbook.json").read_text())
        for entry in entries:
            words = " ".join([entry["title"], entry["text"], *entry["tags"]]).upper()
            for name in reversed_bpms:
                number = name.removeprefix("BPM")
                assert name not in words.replace(" ", "")
                assert f"BPM {number}" not in words
                assert f"BPM {int(number)}" not in words


class TestLogbookOwnership:
    def test_nominal_owns_ambient_entries(self, engine_factory):
        engine = engine_factory("nominal")
        ids = [e.entry_id for e in engine.scenario_logbook("nominal")]
        assert len(ids) == 25
        assert ids[0] == "DEMO-001"
        assert ids[-1] == "DEMO-025"

    def test_rf_thermal_owns_the_incident_arc(self, engine_factory):
        engine = engine_factory("nominal")
        ids = [e.entry_id for e in engine.scenario_logbook("rf-thermal")]
        assert ids == ["DEMO-026", "DEMO-027", "DEMO-028"]

    def test_vacuum_burst_is_telemetry_only(self, engine_factory):
        engine = engine_factory("nominal")
        assert engine.scenario_logbook("vacuum-burst") == ()


class TestRelativeTimestamps:
    def test_rf_incident_lands_at_documented_offsets(self, engine_factory):
        engine = engine_factory("nominal")
        by_id = {e.entry_id: e for e in engine.scenario_logbook("rf-thermal")}
        assert by_id["DEMO-026"].when == RelativeTimestamp(days_ago=4, time=dtime(3, 20, 0))
        assert by_id["DEMO-027"].when == RelativeTimestamp(days_ago=3, time=dtime(10, 0, 0))
        # Newest incident entry lands at now-2d.
        assert by_id["DEMO-028"].when == RelativeTimestamp(days_ago=2, time=dtime(14, 0, 0))

    def test_all_relative_timestamps_are_well_formed(self, engine_factory):
        engine = engine_factory("nominal")
        for name in ("nominal", "rf-thermal"):
            for entry in engine.scenario_logbook(name):
                assert entry.when.days_ago >= 0
                assert isinstance(entry.when.time, dtime)


class TestEntryFields:
    def test_incident_entry_fields(self, engine_factory):
        engine = engine_factory("nominal")
        demo026 = next(e for e in engine.scenario_logbook("rf-thermal") if e.entry_id == "DEMO-026")
        assert demo026.author == "M. Chen"
        assert "CAVITY01" in demo026.title
        assert "reflected power" in demo026.text.lower()
        assert "rf" in demo026.tags
        assert demo026.categories == ("Operations",)
        assert demo026.loto_tag is None

    def test_loto_tag_preserved_where_present(self, engine_factory):
        engine = engine_factory("nominal")
        loto = {e.entry_id: e.loto_tag for e in engine.scenario_logbook("nominal") if e.loto_tag}
        # DEMO-005 / 020 / 022 carry LOTO tags in the ambient log.
        assert loto["DEMO-005"] == "LOTO-2024-0312"
        assert "DEMO-020" in loto


class TestActiveLogbookComposition:
    def test_active_logbook_concatenates_active_scenarios(self, engine_factory):
        engine = engine_factory("rf-thermal")
        ids = [e.entry_id for e in engine.active_logbook()]
        # nominal (25) + rf-thermal (3), nominal-first.
        assert len(ids) == 28
        assert ids[:25] == [f"DEMO-{i:03d}" for i in range(1, 26)]
        assert ids[25:] == ["DEMO-026", "DEMO-027", "DEMO-028"]

    def test_telemetry_only_scenario_adds_no_entries(self, engine_factory):
        engine = engine_factory("vacuum-burst")
        ids = [e.entry_id for e in engine.active_logbook()]
        assert len(ids) == 25  # only nominal's ambient entries
