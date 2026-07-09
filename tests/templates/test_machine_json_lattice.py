"""Consistency tests for the lattice-channel augmentation to machine.json
(task 5.3).

Task 3.1's namespace-union manifest
(src/osprey/services/virtual_accelerator/manifest/channel_manifest.json)
classifies every namespace address into partition
(a) ``pyat-coupled`` (SR magnet CURRENT SP/RB + SR BPM POSITION X/Y --
backed by the AT lattice model), (b) ``sp-echo`` (writable but
physics-free: BR/BTS transport-line magnets, plus a handful of SR RF/VAC
setpoint+readback pairs), and (c) ``static-noisy`` (everything else).

Before this task, mock-mode reads of partition (a)/(b) channels were
connection-refused fictions -- absent from machine.json entirely, so the
mock connector had nothing to serve and the mock archiver had no history to
synthesize for them. This adds partition (a) in full, plus the BR/BTS slice
of (b), to machine.json. It deliberately excludes the SR RF/VAC slice of
(b) (28 channels) -- out of this task's scope; some of those already exist
in machine.json by hand (e.g. ``SR:RF:CAVITY:*``).
"""

import json
from pathlib import Path

import pytest

from osprey.simulation.machine import parse_machine

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "src/osprey/services/virtual_accelerator/manifest/channel_manifest.json"
MACHINE_PATH = (
    REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation/machine.json"
)


@pytest.fixture(scope="module")
def manifest_channels():
    return json.loads(MANIFEST_PATH.read_text())["channels"]


@pytest.fixture(scope="module")
def machine():
    return json.loads(MACHINE_PATH.read_text())


@pytest.fixture(scope="module")
def machine_channels(machine):
    return machine["channels"]


def test_every_partition_a_channel_exists_in_machine_json(manifest_channels, machine_channels):
    """Primary consistency gate: every pyat-coupled (partition (a)) address
    from the manifest must have a machine.json entry."""
    partition_a = [c["address"] for c in manifest_channels if c["partition"] == "pyat-coupled"]
    assert len(partition_a) == 280  # SR magnet CURRENT SP/RB + SR BPM POSITION X/Y

    missing = [addr for addr in partition_a if addr not in machine_channels]
    assert not missing, (
        f"{len(missing)} partition (a) channels missing from machine.json: {missing[:10]}"
    )


def test_br_bts_echo_setpoint_families_exist_in_machine_json(manifest_channels, machine_channels):
    """BR/BTS sp-echo (transport-line magnet) channels must also be present,
    per this task's scope -- but NOT the SR RF/VAC sp-echo slice, which is
    out of scope here."""
    br_bts_echo = [
        c["address"]
        for c in manifest_channels
        if c["partition"] == "sp-echo" and c["ring"] in ("BR", "BTS")
    ]
    assert len(br_bts_echo) == 118

    missing = [addr for addr in br_bts_echo if addr not in machine_channels]
    assert not missing, (
        f"{len(missing)} BR/BTS sp-echo channels missing from machine.json: {missing[:10]}"
    )


def test_sr_rf_vac_sp_echo_channels_out_of_scope_not_required(manifest_channels, machine_channels):
    """Documents the scope boundary: the 28 SR RF/VAC sp-echo channels are
    NOT this task's responsibility. This is not an exhaustiveness assertion
    against machine.json (some already exist there by hand) -- just a
    record that their absence/presence isn't gated here."""
    sr_echo = [
        c["address"] for c in manifest_channels if c["partition"] == "sp-echo" and c["ring"] == "SR"
    ]
    assert len(sr_echo) == 28


def test_pre_existing_channels_unchanged(machine_channels):
    """The 78 channels present before this task must be untouched."""
    pre_existing = {
        "SR:VAC:GAUGE:SR01:PRESSURE:RB": {
            "value": 5e-08,
            "noise": 0.03,
            "units": "Torr",
            "description": "Cold-cathode gauge pressure, storage-ring sector 1 (nominal ~5e-8 Torr)",
            "min": 0.0,
        },
        "SR:DIAG:DCCT:01:CURRENT:RB": {
            "value": 500.0,
            "noise": 0.0001,
            "units": "mA",
            "description": "Stored beam current from the DC current transformer (nominal 500 mA, top-up)",
            "min": 0.0,
        },
        "SR:RF:CAVITY:01:POWER:NET": {
            "expr": "ch('SR:RF:CAVITY:01:POWER:FWD') - ch('SR:RF:CAVITY:01:POWER:REV')",
            "noise": 0,
            "units": "kW",
            "description": "Cavity 01 net power delivered to the cavity (forward minus reflected)",
        },
        "SR:RF:KLYSTRON:02:STATUS:FAULT": {
            "value": 0,
            "noise": 0,
            "description": "Klystron 02 fault summary (0=ok, 1=faulted)",
        },
    }
    for addr, expected in pre_existing.items():
        assert machine_channels[addr] == expected


def test_machine_json_channel_count():
    """78 pre-existing + 280 partition (a) + 118 BR/BTS sp-echo = 476."""
    machine = json.loads(MACHINE_PATH.read_text())
    assert len(machine["channels"]) == 476


def test_setpoints_noiseless_readbacks_noisy(machine_channels, manifest_channels):
    """New CURRENT:SP entries are noise-free (deterministic commanded value,
    matching the file's existing SP convention); CURRENT:RB entries carry
    positive noise."""
    added = [
        c["address"]
        for c in manifest_channels
        if c["partition"] == "pyat-coupled"
        or (c["partition"] == "sp-echo" and c["ring"] in ("BR", "BTS"))
    ]
    for addr in added:
        if addr.endswith(":CURRENT:SP"):
            assert machine_channels[addr]["noise"] == 0, addr
        elif addr.endswith(":CURRENT:RB") or ":POSITION:" in addr:
            assert machine_channels[addr]["noise"] > 0, addr
        elif addr.endswith(":STATUS:FAULT") or addr.endswith(":STATUS:READY"):
            assert machine_channels[addr]["noise"] == 0, addr


def test_machine_json_parses_as_valid_machine_description():
    """The augmented file must still validate against the simulation
    engine's real schema loader (osprey.simulation.machine.parse_machine),
    not just be syntactically valid JSON."""
    machine = json.loads(MACHINE_PATH.read_text())
    parsed = parse_machine(machine, MACHINE_PATH)
    assert len(parsed.channels) == 476
