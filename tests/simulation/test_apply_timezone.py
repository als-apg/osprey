"""Regression (review Finding 1): seeded logbook timestamps land in the FACILITY
zone, coherent with where telemetry ``at_time`` events are placed.

``apply_scenarios`` resolves each bundle entry's ``{days_ago, time}`` relative
timestamp against the apply-time anchor; the anchor's tzinfo decides the zone the
narrative time-of-day lands in. Before the fix the *default* anchor was
``datetime.now(UTC)``, so on a non-UTC facility a logbook entry documented at
03:20 landed at 03:20 UTC while the archiver spike it describes (a daily
``at_time`` event, placed facility-local by the simulation engine) landed at
03:20 facility-local — hours apart, breaking the correlation the scenario exists
to exercise.

DB-free: ``_seed_logbook`` is stubbed to capture the resolved entries, so the
contract is pinned without a Postgres dependency and runs in the fast suite.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from osprey.simulation.apply import apply_scenarios

TEMPLATE_SIM = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/simulation"
)
LA = ZoneInfo("America/Los_Angeles")  # non-UTC facility; -7h (PDT) / -8h (PST)


def _make_project(tmp_path: Path) -> Path:
    """Stage a minimal sim-backed project with a non-UTC facility timezone."""
    sim_dst = tmp_path / "data" / "simulation"
    sim_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_SIM, sim_dst)
    config = {
        "control_system": {
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}}
        },
        # URI is never dialed: _seed_logbook is stubbed below.
        "ariel": {"database": {"uri": "postgresql://unused-mocked/none"}},
        "system": {"timezone": "America/Los_Angeles"},
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return tmp_path


def test_default_anchor_seeds_logbook_in_facility_zone(tmp_path, monkeypatch):
    project = _make_project(tmp_path)

    captured: dict = {}

    async def _fake_seed(ariel_config, entries):
        captured["entries"] = entries
        return len(entries), True

    monkeypatch.setattr("osprey.simulation.apply._seed_logbook", _fake_seed)
    # Pin the facility zone regardless of host config-loading quirks.
    monkeypatch.setattr("osprey.simulation.apply.get_facility_timezone", lambda: LA, raising=False)

    # now=None -> exercise the DEFAULT anchor (the path the bug lives on).
    apply_scenarios(project, ["rf-thermal"], now=None)

    entries = captured.get("entries")
    assert entries, "no logbook entries were seeded"

    # Every seeded timestamp is expressed in the facility zone (LA), not UTC.
    # A UTC anchor makes utcoffset() == 0 while LA is -7/-8h, so this is unequal
    # before the fix and equal after (DST-robust: compares the entry's own zone).
    for e in entries:
        ts = e["timestamp"]
        assert ts.tzinfo is not None, f"entry {e['entry_id']} has a naive timestamp"
        assert ts.utcoffset() == ts.astimezone(LA).utcoffset(), (
            f"entry {e['entry_id']} not anchored in the facility zone: {ts.isoformat()}"
        )

    # DEMO-026 is documented at days_ago=4, 03:20 — that time-of-day is the
    # FACILITY-LOCAL clock time, matching the archiver narrative it accompanies.
    demo026 = next(e["timestamp"] for e in entries if e["entry_id"] == "DEMO-026")
    local = demo026.astimezone(LA)
    assert (local.hour, local.minute) == (3, 20)
    # ...and NOT 03:20 UTC (the pre-fix placement).
    utc = demo026.astimezone(ZoneInfo("UTC"))
    assert (utc.hour, utc.minute) != (3, 20)
