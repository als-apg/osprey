"""The arrays and measurement tools of a pyAML view.

One array per group the measurement file names, its members in the order the
group's addresses are taken; one tool per measurement kind the file allows,
carrying the step and settle keys of that kind verbatim. The chromaticity
monitor's momentum step comes from the design momentum compaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.views.pyaml import CONFIGURATION_FILE, E_DELTA_CAP, E_DELTA_ORBIT
from tests.facility._pyaml_trees import (
    LINE_GROUPS,
    SR_GROUPS,
    measured_tree,
    with_chromaticity,
    with_correctors,
    with_rf,
    write_view,
)

pytest.importorskip("pyaml")


def _configuration(tmp_path: Path, tree: dict[str, Any], model: str) -> dict[str, Any]:
    directory, _ = write_view(tmp_path, tree)
    loaded: dict[str, Any] = yaml.safe_load((directory / model / CONFIGURATION_FILE).read_text())
    return loaded


def _tools(configuration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        device["name"]: device
        for device in configuration["devices"]
        if str(device["type"]).startswith("pyaml.tuning_tools.")
    }


def test_one_array_per_named_group(tmp_path: Path) -> None:
    arrays = _configuration(tmp_path, measured_tree(), "LINE")["arrays"]
    assert [(array["type"], array["name"]) for array in arrays] == [
        ("pyaml.arrays.bpm", "LINE_BPM"),
        ("pyaml.arrays.magnet", "LINE_HCM"),
        ("pyaml.arrays.magnet", "LINE_VCM"),
        ("pyaml.arrays.magnet", "LINE_Q"),
    ]
    assert len(arrays) == len(LINE_GROUPS)


def test_array_members_follow_the_devices_s_order(tmp_path: Path) -> None:
    """SR/QF, split over both ends of the deck, sits at 4.25 m; SR/QD at 1.95 m comes first."""
    arrays = {a["name"]: a for a in _configuration(tmp_path, measured_tree(), "SR")["arrays"]}
    assert len(arrays) == len(SR_GROUPS)
    assert arrays["SR_BPM"]["elements"] == ["SR_BPM1"]
    assert arrays["SR_Q"]["elements"] == ["QD:SP", "QF:SP"]


def test_the_orbit_response_tool_reads_the_bpm_and_corrector_arrays(tmp_path: Path) -> None:
    tools = _tools(_configuration(tmp_path, measured_tree(), "LINE"))
    assert tools == {
        "DEFAULT_ORBIT_RESPONSE_MATRIX": {
            "type": "pyaml.tuning_tools.orbit_response_matrix",
            "name": "DEFAULT_ORBIT_RESPONSE_MATRIX",
            "bpm_array_name": "LINE_BPM",
            "hcorr_array_name": "LINE_HCM",
            "vcorr_array_name": "LINE_VCM",
            "corrector_delta": 1.0e-5,
        }
    }


def test_the_tune_response_tool_steps_the_quadrupole_array(tmp_path: Path) -> None:
    tools = _tools(_configuration(tmp_path, measured_tree(), "SR"))
    assert tools["DEFAULT_TUNE_RESPONSE_MATRIX"] == {
        "type": "pyaml.tuning_tools.tune_response_matrix",
        "name": "DEFAULT_TUNE_RESPONSE_MATRIX",
        "quad_array_name": "SR_Q",
        "betatron_tune_name": "BETATRON_TUNE",
        "quad_delta": 0.001,
    }


def test_a_key_another_kind_takes_is_not_carried(tmp_path: Path) -> None:
    """``n_step`` belongs to the chromaticity monitor; the tune response tool does not take it."""
    tools = _tools(_configuration(tmp_path, measured_tree(), "SR"))
    assert "n_step" not in tools["DEFAULT_TUNE_RESPONSE_MATRIX"]


def test_dispersion_steps_the_rf_plant(tmp_path: Path) -> None:
    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"]["kinds"] = ["dispersion"]
    tree["measurement/SR.yaml"]["frequency_delta"] = 50.0
    tools = _tools(_configuration(tmp_path, tree, "SR"))
    assert tools == {
        "DEFAULT_DISPERSION": {
            "type": "pyaml.tuning_tools.dispersion",
            "name": "DEFAULT_DISPERSION",
            "bpm_array_name": "SR_BPM",
            "rf_plant_name": "DEFAULT_RF_PLANT",
            "frequency_delta": 50.0,
        }
    }


def test_the_chromaticity_monitor_steps_the_momentum_by_the_compaction(tmp_path: Path) -> None:
    tree = with_rf(measured_tree())
    tree["measurement/SR.yaml"]["kinds"] = ["chromaticity_monitor"]
    tree["measurement/SR.yaml"] |= {
        "n_avg_meas": 1,
        "fit_order": 2,
        "sleep_between_step": 0.5,
        "sleep_between_meas": 0.0,
    }
    configuration = _configuration(tmp_path, tree, "SR")
    monitor = _tools(configuration)["CHROMATICITY_MONITOR"]
    alphac = configuration["alphac"]
    e_delta = min(E_DELTA_CAP, E_DELTA_ORBIT / abs(alphac))
    assert monitor == {
        "type": "pyaml.tuning_tools.chromaticity_monitor",
        "name": "CHROMATICITY_MONITOR",
        "betatron_tune_name": "BETATRON_TUNE",
        "rf_plant_name": "DEFAULT_RF_PLANT",
        "e_delta": pytest.approx(e_delta),
        "max_e_delta": pytest.approx(2 * e_delta),
        "n_step": 3,
        "n_avg_meas": 1,
        "fit_order": 2,
        "sleep_between_step": 0.5,
        "sleep_between_meas": 0.0,
    }


def test_the_chromaticity_response_tool_reads_a_chromaticity_monitor(tmp_path: Path) -> None:
    tree = with_chromaticity(with_rf(measured_tree()))
    tree["records/devices.yaml"].append({"id": "SR/SX", "class": "Sextupole"})
    tree["records/channels.yaml"].append(
        {"id": "SX:SP", "role": "setpoint", "on": {"device": "SR/SX"}}
    )
    tree["records/groups.yaml"].append({"id": "SR/S", "members": ["SR/SX"]})
    (sr,) = [model for model in tree["models.yaml"] if model["name"] == "SR"]
    sr["wiring"].append(
        {"address": "SX:SP", "element": "SX", "engine": {"attribute": "PolynomB", "index": 2}}
    )
    tree["measurement/SR.yaml"] |= {"kinds": ["crm"], "sextu_delta": 0.01}
    tree["measurement/SR.yaml"]["groups"]["sext"] = "SR/S"
    tools = _tools(_configuration(tmp_path, tree, "SR"))
    assert tools["DEFAULT_CHROMATICITY_RESPONSE_MATRIX"] == {
        "type": "pyaml.tuning_tools.chromaticity_response_matrix",
        "name": "DEFAULT_CHROMATICITY_RESPONSE_MATRIX",
        "sextu_array_name": "SR_S",
        "chromaticity_name": "CHROMATICITY_MONITOR",
        "sextu_delta": 0.01,
    }
    assert tools["CHROMATICITY_MONITOR"]["type"] == "pyaml.tuning_tools.chromaticity_monitor"


def test_the_measurement_tools_load_in_pyaml(tmp_path: Path) -> None:
    from pyaml.accelerator import Accelerator

    tree = with_rf(measured_tree())
    tree["measurement/SR.yaml"]["kinds"] = ["trm", "chromaticity_monitor"]
    tree["measurement/SR.yaml"] |= {
        "n_avg_meas": 1,
        "fit_order": 2,
        "sleep_between_step": 0.0,
        "sleep_between_meas": 0.0,
    }
    directory, _ = write_view(tmp_path, tree)
    sr = Accelerator.load(str(directory / "SR" / CONFIGURATION_FILE))
    assert sr.design.trm is not None
    assert sr.design.get_chromaticity_monitor("CHROMATICITY_MONITOR") is not None
