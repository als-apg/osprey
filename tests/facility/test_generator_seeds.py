"""The demo generator's ``seeds.yaml`` against the machine file and the nominal goldens.

``scripts/facility_demo/generate.py`` writes one seed per channel that moves,
is clamped, or starts at a value no model computes. These tests read the seeds
back from the YAML the generator writes and hold them to the demo's machine
file, to the procedural taxonomy's classes, and to the per-address nominal and
sigma the mock and the virtual accelerator serve today
(``tests/facility/golden/nominal_mock.json`` and ``nominal_va.json``).
"""

from __future__ import annotations

import json
import math
import shutil
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from tests.facility.test_generator_records import generated, generated_files, generator

REPO_ROOT = Path(__file__).resolve().parents[2]
CA_DATA = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"
MACHINE_JSON = CA_DATA / "simulation/machine.json"
SR_DECK = CA_DATA / "facility/decks/SR.json"
GOLDEN = REPO_ROOT / "tests/facility/golden"

#: A seed's value tolerance against a golden.
TOLERANCE = 1e-12

#: The labels of a bool channel with no ``options``.
TRUE, FALSE = "TRUE", "FALSE"

#: The deck machine's instruments the golden captures predate.
ADDITIONS = frozenset({"SR:DIAG:CHROM:X", "SR:DIAG:CHROM:Y", "SR:DIAG:TUNE:X", "SR:DIAG:TUNE:Y"})

#: Each RF net-power readback and the forward and reflected readbacks it sums.
NET_POWER = {
    f"SR:RF:CAVITY:{n}:POWER:NET": {
        f"SR:RF:CAVITY:{n}:POWER:FWD": 1.0,
        f"SR:RF:CAVITY:{n}:POWER:REV": -1.0,
    }
    for n in ("01", "02")
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def machine() -> dict[str, dict[str, Any]]:
    """The machine file's channels."""
    channels: dict[str, dict[str, Any]] = _json(MACHINE_JSON)["channels"]
    return channels


@cache
def golden(substrate: str) -> dict[str, dict[str, float]]:
    """``nominal_<substrate>.json``'s channels."""
    channels: dict[str, dict[str, float]] = _json(GOLDEN / f"nominal_{substrate}.json")["channels"]
    return channels


def seeds() -> dict[str, dict[str, Any]]:
    """``seeds.yaml`` as the facility loader parses it."""
    document: dict[str, dict[str, Any]] = generated("seeds.yaml")
    return document


def channels() -> dict[str, dict[str, Any]]:
    """The generated channel records keyed by id."""
    return {c["id"]: c for c in generated("records/channels.yaml")}


@cache
def wired() -> frozenset[str]:
    """Every address the generated model wires."""
    return frozenset(r["address"] for m in generated("models.yaml") for r in m["wiring"])


def value_type(address: str) -> str:
    return str(channels()[address].get("value_type", "float"))


def role(address: str) -> str:
    return str(channels()[address].get("role", "readback"))


def moves(entry: dict[str, Any]) -> bool:
    """A machine-file channel moves on its own: noise or a texture."""
    return bool(entry.get("noise")) or bool(entry.get("noise_abs")) or "texture" in entry


def machine_sigma(entry: dict[str, Any]) -> float:
    """The machine file's relative and absolute noise, combined in quadrature."""
    return math.hypot(abs(entry["value"]) * entry.get("noise", 0.0), entry.get("noise_abs", 0.0))


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def test_seeds_are_sorted_by_address() -> None:
    addresses = list(seeds())
    assert addresses == sorted(addresses)
    assert set(addresses) <= set(channels())


def test_every_unwired_channel_has_a_seed() -> None:
    unwired = set(channels()) - wired()
    assert len(unwired) == 2066
    assert sorted(unwired - set(seeds())) == []


def test_wired_channels_carry_no_nominal_and_setpoints_no_seed() -> None:
    seeded = [a for a in wired() if a in seeds()]
    assert [a for a in seeded if "nominal" in seeds()[a]] == []
    assert [a for a in seeded if role(a) == "setpoint"] == []
    assert {k for a in seeded for k in seeds()[a]} <= {"noise", "drift", "clamp"}


def test_machine_motion_is_carried_on_every_moving_channel() -> None:
    moving = sorted(a for a, entry in machine().items() if moves(entry))
    assert len(moving) == 555
    assert sum(a in wired() for a in moving) == 493
    for address in moving:
        entry, seed = machine()[address], seeds().get(address, {})
        sigma = machine_sigma(entry)
        if sigma:
            assert close(seed["noise"], sigma), address
        else:
            assert "noise" not in seed, address
        texture = entry.get("texture")
        if texture is None:
            assert "drift" not in seed, address
        else:
            assert texture["kind"] == "wander"
            expected = {"amplitude": texture["amplitude"], "period_s": texture["period_s"]}
            assert seed["drift"] == expected, address


def test_still_machine_channels_carry_no_motion() -> None:
    still = [a for a, entry in machine().items() if not moves(entry)]
    assert [a for a in still if {"noise", "drift"} & set(seeds().get(a, {}))] == []


def test_readback_floors_and_ceilings_become_clamps() -> None:
    bounded = {a: e for a, e in machine().items() if "min" in e or "max" in e}
    readbacks = sorted(a for a in bounded if role(a) == "readback")
    assert len(readbacks) == 401
    for address in readbacks:
        entry = bounded[address]
        assert seeds()[address]["clamp"] == [entry.get("min"), entry.get("max")], address
    assert [a for a, seed in seeds().items() if "clamp" in seed and a not in readbacks] == []


def test_unwired_machine_channels_start_at_their_value() -> None:
    unwired = [a for a in machine() if a not in wired() and a not in NET_POWER]
    assert len(unwired) == 192
    for address in unwired:
        value = machine()[address]["value"]
        if value_type(address) == "bool":
            assert seeds()[address]["nominal"] == (TRUE if value else FALSE), address
        else:
            assert seeds()[address]["nominal"] == value, address


def test_declared_bools_are_labels() -> None:
    declared = [a for a in machine() if value_type(a) == "bool"]
    assert len(declared) == 87
    labels = [seeds()[a]["nominal"] for a in declared]
    assert (labels.count(TRUE), labels.count(FALSE)) == (65, 22)
    assert [a for a in declared if set(seeds()[a]) != {"nominal"}] == []


def test_channels_absent_from_the_machine_file_fall_into_three_classes() -> None:
    absent = sorted(set(channels()) - set(machine()) - ADDITIONS)
    assert len(absent) == 1872
    bools = [a for a in absent if value_type(a) == "bool"]
    readbacks = [a for a in absent if value_type(a) == "float" and role(a) == "readback"]
    setpoints = [a for a in absent if role(a) == "setpoint"]
    assert (len(bools), len(readbacks), len(setpoints)) == (1159, 707, 6)
    labels = [seeds()[a]["nominal"] for a in bools]
    assert (labels.count(TRUE), labels.count(FALSE)) == (1135, 24)
    assert [a for a in bools if set(seeds()[a]) != {"nominal"}] == []
    assert [a for a in readbacks if set(seeds()[a]) != {"nominal", "noise"}] == []
    assert setpoints == [f"SR:VAC:ION-PUMP:0{n}:VOLTAGE:SP" for n in range(1, 7)]
    assert [seeds()[a] for a in setpoints] == [{"nominal": 5000.0}] * 6


def test_procedural_motion_is_for_float_readbacks_only() -> None:
    procedural = generator()._seeds._procedural_seed
    absent = sorted(set(channels()) - set(machine()) - ADDITIONS)
    readbacks = [a for a in absent if value_type(a) == "float" and role(a) == "readback"]
    assert all("noise" in procedural(a, channels()[a]) for a in readbacks)
    as_int = [procedural(a, {**channels()[a], "value_type": "int"}) for a in readbacks]
    assert [seed for seed in as_int if "noise" in seed] == []


def test_noise_counts() -> None:
    noisy = [a for a, seed in seeds().items() if "noise" in seed]
    from_machine = [a for a in noisy if a in machine()]
    assert (len(noisy), len(from_machine)) == (555 + 707, 555)
    assert sum(a in wired() for a in from_machine) == 493


def test_net_power_is_forward_minus_reflected() -> None:
    for address, terms in NET_POWER.items():
        assert seeds()[address] == {"linear": terms}, address
        assert not set(terms) & wired()
        total = sum(seeds()[source]["nominal"] * k for source, k in terms.items())
        assert close(total, golden("mock")[address]["nominal"]), address


def test_float_nominals_equal_the_mock_golden() -> None:
    assert set(channels()) - set(golden("mock")) == ADDITIONS
    floats = [a for a in seeds() if value_type(a) == "float" and "nominal" in seeds()[a]]
    assert len(floats) == 707 + 6 + 105
    for address in floats:
        assert close(seeds()[address]["nominal"], golden("mock")[address]["nominal"]), address


def test_float_readback_noise_equals_the_mock_golden_sigma() -> None:
    readbacks = [
        a
        for a in golden("mock")
        if value_type(a) == "float" and role(a) == "readback" and a not in NET_POWER
    ]
    for address in readbacks:
        noise = seeds().get(address, {}).get("noise", 0.0)
        assert close(noise, golden("mock")[address]["sigma"]), address


def test_bool_labels_follow_both_goldens() -> None:
    bools = [a for a in channels() if value_type(a) == "bool"]
    assert len(bools) == 1246
    for address in bools:
        label = seeds()[address]["nominal"]
        assert (golden("mock")[address]["nominal"] != 0.0) == (label == TRUE), address
        assert golden("va")[address]["nominal"] == (1.0 if label == TRUE else 0.0), address


def test_banded_unwired_nominals_sit_inside_their_band() -> None:
    records = _json(GOLDEN / "limits.json")["channels"]
    setpoint_of = {
        str(c["pair"]): c["id"]
        for c in channels().values()
        if c.get("role") == "setpoint" and c.get("pair", c["id"]) != c["id"]
    }

    def nominal(address: str) -> Any:
        seed = seeds().get(address, {})
        if "nominal" in seed:
            return seed["nominal"]
        setpoint = setpoint_of.get(address)
        if setpoint is not None and "nominal" in seeds().get(setpoint, {}):
            return seeds()[setpoint]["nominal"]
        return 0.0

    banded = [
        (address, record)
        for address, record in sorted(records.items())
        if record.get("min_value") is not None
        and record.get("max_value") is not None
        and address not in wired()
        and "linear" not in seeds().get(address, {})
    ]
    assert banded
    outside = [
        address
        for address, record in banded
        if not record["min_value"] <= nominal(address) <= record["max_value"]
    ]
    assert outside == []


FLOAT_READBACK = {"value_type": "float", "role": "readback"}


@pytest.mark.parametrize(
    ("entry", "channel", "refusal"),
    [
        (
            {"value": 1.0, "texture": {"kind": "step", "amplitude": 1.0, "period_s": 1.0}},
            FLOAT_READBACK,
            "is not wander",
        ),
        ({"expr": "ch('A') + ch('B')"}, FLOAT_READBACK, "is not a difference"),
        ({"value": 2}, {"value_type": "bool", "role": "readback"}, "is not 0 or 1"),
        ({"value": 1.0, "noise": 0.1}, {"value_type": "float", "role": "setpoint"}, "motion on"),
        ({"noise_abs": 0.1}, FLOAT_READBACK, "no value and no expression"),
    ],
)
def test_malformed_machine_entries_are_refused(
    entry: dict[str, Any], channel: dict[str, Any], refusal: str
) -> None:
    module = generator()._seeds
    with pytest.raises(module._records.RecordsError, match=refusal):
        module._machine_seed("X:Y", entry, channel, wired=False)


def test_a_machine_channel_with_no_record_is_refused() -> None:
    module = generator()._seeds
    with pytest.raises(module._records.RecordsError, match="channels with no record"):
        module.build_seeds([], [])


def test_written_tree_passes_every_build_stage(tmp_path: Path) -> None:
    pytest.importorskip("at")
    from osprey.facility.build import LATER_STAGES
    from osprey.facility.validate import run_stages

    assert generator().main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "seeds.yaml").read_text(encoding="utf-8") == generated_files()["seeds.yaml"]
    (tmp_path / "decks").mkdir()
    shutil.copyfile(SR_DECK, tmp_path / "decks/SR.json")
    report = run_stages(tmp_path, project_name="ca", later=LATER_STAGES)
    assert report.errors == []
