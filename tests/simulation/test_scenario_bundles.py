"""Per-scenario bundle contract: each bundle owns the right logbook entries.

The control_assistant scenarios are self-contained bundles — each owns its
telemetry overlay (``scenario.json``) and, optionally, its logbook narrative
(``logbook.json``). These fast tests pin logbook ownership and entry fields
straight from the parsed bundles (no DB), so a misfiled entry or a malformed
relative timestamp is caught here rather than in the expensive e2e judge layer.
"""

from datetime import time as dtime

from osprey.utils.relative_time import RelativeTimestamp


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
        # Newest incident entry lands at now-2d, matching the old rebase landing.
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
        assert "RF cavity C1" in demo026.title
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
