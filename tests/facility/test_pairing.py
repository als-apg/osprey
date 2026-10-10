"""The rule that pairs a setpoint with the one readback of its quantity."""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.pairing import derive_pairs, signal_quantity
from osprey.facility.validate import StageReport, run_stages, vocabulary
from tests.facility._mml_built import TREES, BuiltModel

EXAMPLE = Path(__file__).parents[2] / "src" / "osprey" / "templates" / "facilities" / "example"

#: Every (setpoint signal, readback signal) the vocabulary lets the rule pair.
PAIRINGS = {
    ("current_setpoint", "current_readback"),
    ("current_x_setpoint", "current_x_readback"),
    ("current_y_setpoint", "current_y_readback"),
    ("field_setpoint", "field_readback"),
    ("frequency_setpoint", "frequency_readback"),
    ("gradient_setpoint", "gradient_readback"),
    ("gun_voltage_setpoint", "gun_voltage_readback"),
    ("phase_setpoint", "phase_readback"),
    ("rf_hv_setpoint", "rf_hv_readback"),
    ("tuner_position_setpoint", "tuner_position_readback"),
    ("voltage_setpoint", "voltage_readback"),
    ("lamp_command", "lamp_status"),
    ("power_on_command", "power_on_status"),
    ("screen_in_command", "screen_in_status"),
}

#: The write-side signals no readback of the vocabulary shares a quantity with.
WRITE_ONLY = {
    "bulk_supply_command",
    "modulator_state_command",
    "pulse_amplitude_setpoint",
    "pulse_delay_setpoint",
    "pulse_width_setpoint",
    "pulsing_command",
    "ramp_rate_setpoint",
    "reset_command",
    "valve_open_command",
}


def _sp(address: str, signal: str | None = "current_setpoint", **fields: Any) -> dict[str, Any]:
    channel: dict[str, Any] = {"role": "setpoint", "value_type": "float", "on": {"device": "D"}}
    if signal is not None:
        channel["signal"] = signal
    return {address: {**channel, **fields}}


def _rb(address: str, signal: str | None = "current_readback", **fields: Any) -> dict[str, Any]:
    channel: dict[str, Any] = {"role": "readback", "value_type": "float", "on": {"device": "D"}}
    if signal is not None:
        channel["signal"] = signal
    return {address: {**channel, **fields}}


# --- the quantity table ----------------------------------------------------------------


def test_the_vocabulary_pairings_are_the_table() -> None:
    names = [row["name"] for row in vocabulary()["signal_roles"]]
    writers = {signal_quantity(n, "setpoint"): n for n in names if signal_quantity(n, "setpoint")}
    readers = {signal_quantity(n, "readback"): n for n in names if signal_quantity(n, "readback")}
    assert {(writers[q], readers[q]) for q in writers if q in readers} == PAIRINGS
    assert {writers[q] for q in writers if q not in readers} == WRITE_ONLY


def test_a_signal_without_a_role_word_of_its_side_has_no_quantity() -> None:
    assert signal_quantity("current_setpoint", "setpoint") == "current"
    assert signal_quantity("current_readback", "readback") == "current"
    assert signal_quantity("current_readback", "setpoint") is None
    assert signal_quantity("position_offset", "readback") is None
    assert signal_quantity("status", "readback") is None
    assert signal_quantity(None, "setpoint") is None


# --- derive_pairs ----------------------------------------------------------------------


def test_one_setpoint_and_one_readback_pair() -> None:
    assert derive_pairs({**_sp("SP"), **_rb("RB")}) == {"SP": "RB"}


def test_a_qualified_quantity_never_pairs_with_the_plain_one() -> None:
    channels = {**_sp("SP"), **_rb("GOLD", "current_golden_readback")}
    assert derive_pairs(channels) == {}


def test_each_coil_pairs_with_the_readback_of_its_plane() -> None:
    channels = {
        **_sp("SPX", "current_x_setpoint"),
        **_sp("SPY", "current_y_setpoint"),
        **_rb("RBX", "current_x_readback"),
        **_rb("RBY", "current_y_readback"),
    }
    assert derive_pairs(channels) == {"SPX": "RBX", "SPY": "RBY"}


def test_two_setpoints_of_one_readback_pair_neither() -> None:
    assert derive_pairs({**_sp("SP1"), **_sp("SP2"), **_rb("RB")}) == {}


def test_one_setpoint_of_two_readbacks_pairs_neither() -> None:
    assert derive_pairs({**_sp("SP"), **_rb("RB1"), **_rb("RB2")}) == {}


def test_an_explicit_pair_takes_its_readback_out_of_the_candidates() -> None:
    channels = {**_sp("SP1", pair="RB1"), **_sp("SP2"), **_rb("RB1"), **_rb("RB2")}
    assert derive_pairs(channels) == {"SP2": "RB2"}


def test_a_setpoint_naming_itself_opts_out() -> None:
    assert derive_pairs({**_sp("SP", pair="SP"), **_rb("RB")}) == {}


def test_a_shared_supply_pairs_only_with_the_same_devices() -> None:
    channels = {
        "SP": {
            "role": "setpoint",
            "value_type": "float",
            "signal": "current_setpoint",
            "endpoint_of": ["B", "A"],
        },
        "RB": {
            "role": "readback",
            "value_type": "float",
            "signal": "current_readback",
            "endpoint_of": ["A", "B"],
        },
        "OTHER": {
            "role": "readback",
            "value_type": "float",
            "signal": "current_readback",
            "endpoint_of": ["A", "C"],
        },
    }
    assert derive_pairs(channels) == {"SP": "RB"}


def test_a_channel_on_nothing_pairs_only_with_a_channel_on_nothing() -> None:
    loose = {
        "SP": {"role": "setpoint", "value_type": "float", "signal": "current_setpoint"},
        "RB": {"role": "readback", "value_type": "float", "signal": "current_readback"},
    }
    assert derive_pairs({**loose, **_rb("ON_D")}) == {"SP": "RB"}
    assert derive_pairs({"SP": loose["SP"], **_rb("ON_D")}) == {}


def test_a_value_type_mismatch_pairs_neither() -> None:
    assert derive_pairs({**_sp("SP"), **_rb("RB", value_type="int")}) == {}


def test_a_readback_word_on_a_setpoint_never_pairs() -> None:
    assert derive_pairs({**_sp("SP", "current_readback"), **_rb("RB")}) == {}


# --- through the build -------------------------------------------------------------------


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


def _run(tmp_path: Path, files: dict[str, Any]) -> StageReport:
    return run_stages(_write(tmp_path / "facility", files), project_name="my proj")


def _channels(result: StageReport) -> dict[str, dict[str, Any]]:
    assert result.ok, [error.format_message() for error in result.errors]
    return {channel["id"]: channel for channel in result.validated.document["channels"]}


def _magnet(**setpoint: Any) -> dict[str, Any]:
    on = {"device": "SR/Q1"}
    return {
        "records/devices.yaml": [{"id": "SR/Q1", "class": "Quadrupole"}],
        "records/channels.yaml": [
            {"id": "Q1:SP", "role": "setpoint", "signal": "current_setpoint", "on": on, **setpoint},
            {"id": "Q1:RB", "signal": "current_readback", "on": on},
            {"id": "Q1:RB2", "on": on},
        ],
    }


def test_a_setpoint_pairs_with_the_one_readback_of_its_quantity(tmp_path: Path) -> None:
    setpoint = _channels(_run(tmp_path, _magnet()))["Q1:SP"]
    assert setpoint["pair"] == "Q1:RB"
    assert "pair" in setpoint["provenance"]["defaults"]


def test_an_explicit_pair_wins(tmp_path: Path) -> None:
    setpoint = _channels(_run(tmp_path, _magnet(pair="Q1:RB2")))["Q1:SP"]
    assert setpoint["pair"] == "Q1:RB2"
    assert "pair" not in setpoint["provenance"]["defaults"]


def test_a_setpoint_with_no_readback_of_its_quantity_is_its_own_pair(tmp_path: Path) -> None:
    files = _magnet(signal="voltage_setpoint")
    setpoint = _channels(_run(tmp_path, files))["Q1:SP"]
    assert setpoint["pair"] == "Q1:SP"
    assert "pair" in setpoint["provenance"]["defaults"]


def test_a_derived_pair_stop_names_the_explicit_way_out(tmp_path: Path) -> None:
    files = _magnet()
    files["models.yaml"] = [
        {"name": "one", "engine": "pyat", "wiring": [{"address": "Q1:SP", "element": "Q1"}]},
        {"name": "two", "engine": "pyat", "wiring": [{"address": "Q1:RB", "element": "Q1"}]},
    ]
    result = _run(tmp_path, files)
    assert result.failed == "records"
    (error,) = result.errors
    assert error.kind == "pair-invalid"
    assert "wire the setpoint and its pair in the same model" in error.remedy
    assert "or state `pair: Q1:SP` on Q1:SP to leave it unpaired" in error.remedy


# --- the demo ----------------------------------------------------------------------------


def test_the_demo_pairs_are_the_derived_pairs(tmp_path: Path) -> None:
    stated = tmp_path / "stated"
    derived = tmp_path / "derived"
    shutil.copytree(EXAMPLE, stated)
    shutil.copytree(EXAMPLE, derived)
    channels_file = derived / "records" / "channels.yaml"
    text = channels_file.read_text(encoding="utf-8")
    stripped = re.sub(r"^  pair: .*\n", "", text, flags=re.MULTILINE)
    assert stripped != text
    channels_file.write_text(stripped, encoding="utf-8")

    def pairs(root: Path) -> dict[str, str]:
        return {
            address: channel["pair"]
            for address, channel in _channels(run_stages(root, project_name="demo")).items()
            if channel["role"] == "setpoint"
        }

    authored = pairs(stated)
    assert len(authored) == 412
    assert pairs(derived) == authored


# --- the fixture trees ---------------------------------------------------------------------

#: Each model of the trees whose build derives pairs, as (tree, export stem).
FIXTURE_MODELS = [(tree, stem) for tree in ("spear3", "nsls2") for stem in TREES[tree]]


@pytest.mark.xdist_group("mml_built")
@pytest.mark.parametrize(("tree", "stem"), FIXTURE_MODELS, ids=[s for _t, s in FIXTURE_MODELS])
def test_view_reader_stamp_a1_holds(
    tree: str, stem: str, mml_built: Callable[[str, str], BuiltModel]
) -> None:
    model = mml_built(tree, stem)
    derived = {
        channel["pair"]
        for channel in model.document["channels"]
        if channel["role"] == "setpoint"
        and channel["pair"] != channel["id"]
        and "pair" in channel["provenance"]["defaults"]
    }
    named = [entry for entry in model.wiring if entry["address"] in derived]
    assert named
    assert {(entry["direction"], entry["role"]) for entry in named} == {("read", "readback")}
