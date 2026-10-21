"""The devices of a pyAML view: magnets, BPMs, the RF plant and the tune monitor.

A magnet is one setpoint address, of the pyAML type its group role is, driving
every deck element its wiring slices name; a BPM is one device with the
readbacks the model wires on it as monitors, each reference carrying its
channel's unit; the RF plant drives the ``instruments.rf`` address; the tune
monitor reads the ``instruments.tune`` output on both planes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.views.pyaml import CONFIGURATION_FILE
from tests.facility._pyaml_trees import measured_tree, write_view

pytest.importorskip("pyaml")


def _devices(tmp_path: Path, tree: dict[str, Any], model: str) -> dict[str, dict[str, Any]]:
    directory, _ = write_view(tmp_path, tree)
    configuration = yaml.safe_load((directory / model / CONFIGURATION_FILE).read_text())
    return {device["name"]: device for device in configuration["devices"]}


def test_each_corrector_plane_is_one_magnet_on_the_shared_element(tmp_path: Path) -> None:
    devices = _devices(tmp_path, measured_tree(), "LINE")
    assert devices["LCOR:H:SP"]["type"] == "pyaml.magnet.hcorrector"
    assert devices["LCOR:V:SP"]["type"] == "pyaml.magnet.vcorrector"
    assert devices["LCOR:H:SP"]["lattice_names"] == "list(COR1)"
    assert devices["LCOR:V:SP"]["lattice_names"] == "list(COR1)"


def test_a_split_magnet_drives_every_slice_element(tmp_path: Path) -> None:
    devices = _devices(tmp_path, measured_tree(), "SR")
    assert devices["QF:SP"]["type"] == "pyaml.magnet.quadrupole"
    assert devices["QF:SP"]["lattice_names"] == "list(QFA,QFB)"


def test_a_paired_setpoint_reads_its_readback(tmp_path: Path) -> None:
    devices = _devices(tmp_path, measured_tree(), "SR")
    assert devices["QF:SP"]["model"]["powerconverter"] == "(QF:RB, QF:SP)"
    assert devices["QD:SP"]["model"]["powerconverter"] == "QD:SP"


def test_a_bpm_reads_both_planes_in_its_channels_unit(tmp_path: Path) -> None:
    devices = _devices(tmp_path, measured_tree(), "SR")
    assert devices["SR_BPM1"] == {
        "type": "pyaml.bpm.bpm",
        "name": "SR_BPM1",
        "lattice_names": "list(BPM1)",
        "x_pos": "BPM1:X[mm]",
        "y_pos": "BPM1:Y[mm]",
    }


def test_a_scalar_tune_output_is_paired_with_the_other_plane(tmp_path: Path) -> None:
    devices = _devices(tmp_path, measured_tree(), "SR")
    assert devices["BETATRON_TUNE"] == {
        "type": "pyaml.diagnostics.tune_monitor",
        "name": "BETATRON_TUNE",
        "tune_h": "SR:TUNE:X",
        "tune_v": "SR:TUNE:Y",
    }


def test_a_vertical_tune_instrument_still_reads_x_first(tmp_path: Path) -> None:
    tree = measured_tree()
    tree["measurement/SR.yaml"]["instruments"]["tune"] = "SR:TUNE:Y"
    monitor = _devices(tmp_path, tree, "SR")["BETATRON_TUNE"]
    assert (monitor["tune_h"], monitor["tune_v"]) == ("SR:TUNE:X", "SR:TUNE:Y")


def test_a_whole_tune_waveform_is_read_by_element(tmp_path: Path) -> None:
    tree = measured_tree()
    channels = tree["records/channels.yaml"]
    channels[:] = [c for c in channels if not str(c["id"]).startswith("SR:TUNE")]
    channels.append(
        {"id": "SR:TUNES", "value_type": "waveform", "shape": [3], "on": {"place": "SR"}}
    )
    (sr,) = [model for model in tree["models.yaml"] if model["name"] == "SR"]
    sr["wiring"] = [w for w in sr["wiring"] if not str(w["address"]).startswith("SR:TUNE")]
    sr["wiring"].append({"address": "SR:TUNES", "engine": {"attribute": "tune"}})
    tree["measurement/SR.yaml"]["instruments"]["tune"] = "SR:TUNES"
    monitor = _devices(tmp_path, tree, "SR")["BETATRON_TUNE"]
    assert (monitor["tune_h"], monitor["tune_v"]) == ("SR:TUNES@0", "SR:TUNES@1")


def test_the_rf_plant_drives_the_rf_instrument_in_its_unit(tmp_path: Path) -> None:
    tree = measured_tree()
    tree["records/devices.yaml"].append({"id": "SR/CAV", "class": "AcceleratingCavity"})
    tree["records/channels.yaml"].append(
        {"id": "RF:SP", "role": "setpoint", "unit": "MHz", "on": {"device": "SR/CAV"}}
    )
    tree["measurement/SR.yaml"]["instruments"]["rf"] = "RF:SP"
    devices = _devices(tmp_path, tree, "SR")
    assert devices["DEFAULT_RF_PLANT"] == {
        "type": "pyaml.rf.rf_plant",
        "name": "DEFAULT_RF_PLANT",
        "masterclock": "RF:SP[MHz]",
    }
    assert devices["BETATRON_TUNE"]["rf_plant_name"] == "DEFAULT_RF_PLANT"


def test_a_supply_shared_by_two_devices_is_one_magnet(tmp_path: Path) -> None:
    """A setpoint naming two member devices as endpoints drives both their elements."""
    tree = measured_tree()
    channels = tree["records/channels.yaml"]
    channels[:] = [c for c in channels if c["id"] not in ("QF:SP", "QF:RB", "QD:SP")]
    channels.append({"id": "Q:FAMILY:SP", "role": "setpoint", "endpoint_of": ["SR/QF", "SR/QD"]})
    (sr,) = [model for model in tree["models.yaml"] if model["name"] == "SR"]
    sr["wiring"] = [w for w in sr["wiring"] if w["address"] not in ("QF:SP", "QF:RB", "QD:SP")]
    sr["wiring"].append(
        {
            "address": "Q:FAMILY:SP",
            "slices": [
                {"element": "QFA", "device": "SR/QF"},
                {"element": "QD", "device": "SR/QD"},
                {"element": "QFB", "device": "SR/QF"},
            ],
            "engine": {"attribute": "PolynomB", "index": 1},
        }
    )
    devices = _devices(tmp_path, tree, "SR")
    magnets = [name for name, device in devices.items() if "model" in device]
    assert magnets == ["Q:FAMILY:SP"]
    assert devices["Q:FAMILY:SP"]["lattice_names"] == "list(QFA,QD,QFB)"
