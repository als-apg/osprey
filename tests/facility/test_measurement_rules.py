"""The measurement file's rules and the addresses each of its groups stands for.

``kinds`` is authored and carried verbatim; each kind needs its groups and
instruments named in the file and wired by the model, a ``single_pass`` model
allows ``orm`` alone, and each kind's tool needs its step and settle keys.

A group role stands for the setpoints (or, for ``bpm``, the monitored
readbacks) its model wires on the group's member devices: ``hcor`` those the
engine describes as plane ``x``, ``vcor`` plane ``y``, so a combined corrector
keeps the two planes apart.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.facility.views.pyaml import measurement_groups
from tests.facility._pyaml_trees import built_document, measured_tree, with_rf

if TYPE_CHECKING:
    from tests.facility._mml_built import BuiltModel
    from tests.facility.conftest import BuiltProject

# xdist_group("mml_built"): the session ``mml_built`` fixture builds each
# fixture tree once per worker; the group keeps its readers on one worker.
pytestmark = pytest.mark.xdist_group("mml_built")

#: The demo SR model's kinds, sorted as the facility file carries them.
DEMO_KINDS = ["chromaticity_monitor", "crm", "dispersion", "orm", "trm"]


def _model(document: dict[str, Any], name: str) -> dict[str, Any]:
    (record,) = [model for model in document["models"] if model["name"] == name]
    return record


def _setpoints_on(document: dict[str, Any], model: str, group: str) -> set[str]:
    """The setpoint addresses ``model`` wires on the devices of ``group``."""
    (members,) = [g["members"] for g in document["groups"] if g["id"] == group]
    on = {
        channel["id"]
        for channel in document["channels"]
        if (channel.get("on") or {}).get("device") in members
    }
    return {
        entry["address"]
        for entry in _model(document, model)["wiring"]
        if entry["address"] in on and entry.get("direction") == "write"
    }


def _stop(tmp_path: Path, tree: dict[str, Any]) -> FacilityBuildError:
    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path, tree)
    return stopped.value


# --- what builds --------------------------------------------------------------------


def test_the_demo_sr_model_builds_all_five_kinds(built_control_assistant: BuiltProject) -> None:
    measurement = _model(built_control_assistant.facility, "SR")["measurement"]
    assert measurement["kinds"] == DEMO_KINDS
    assert set(measurement["instruments"]) == {"tune", "rf"}


def test_the_demo_corrector_groups_are_their_planes(built_control_assistant: BuiltProject) -> None:
    document = built_control_assistant.facility
    groups = measurement_groups(document, "SR")
    dipoles = {
        channel["id"]
        for channel in document["channels"]
        if str((channel.get("on") or {}).get("device", "")).startswith("SR/DIPOLE")
    }
    assert set(groups["hcor"]) == _setpoints_on(document, "SR", "SR/HCM")
    assert set(groups["vcor"]) == _setpoints_on(document, "SR", "SR/VCM")
    assert not set(groups["hcor"]) & set(groups["vcor"])
    assert not (set(groups["hcor"]) | set(groups["vcor"])) & dipoles
    assert set(groups["quad"]) == _setpoints_on(document, "SR", "SR/QF")
    assert set(groups["sext"]) == _setpoints_on(document, "SR", "SR/SF")
    assert len(groups["hcor"]) == len(groups["vcor"]) == 72


def test_the_demo_bpm_group_is_both_planes_device_by_device(
    built_control_assistant: BuiltProject,
) -> None:
    bpm = measurement_groups(built_control_assistant.facility, "SR")["bpm"]
    assert bpm[:4] == [
        "SR:DIAG:BPM:01:POSITION:X",
        "SR:DIAG:BPM:01:POSITION:Y",
        "SR:DIAG:BPM:02:POSITION:X",
        "SR:DIAG:BPM:02:POSITION:Y",
    ]
    assert len(bpm) == len(set(bpm)) == 144


def test_a_line_builds_orm(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, measured_tree())
    assert _model(document, "LINE")["measurement"]["kinds"] == ["orm"]
    assert measurement_groups(document, "LINE") == {
        "bpm": ["LBPM:X", "LBPM:Y"],
        "hcor": ["LCOR:H:SP"],
        "vcor": ["LCOR:V:SP"],
        "quad": ["LQ:SP"],
    }


def test_a_setpoint_shared_by_two_members_is_listed_once(tmp_path: Path) -> None:
    tree = measured_tree()
    channels = tree["records/channels.yaml"]
    channels.append({"id": "Q:FAMILY:SP", "role": "setpoint", "endpoint_of": ["SR/QF", "SR/QD"]})
    (sr,) = [model for model in tree["models.yaml"] if model["name"] == "SR"]
    sr["wiring"].append(
        {
            "address": "Q:FAMILY:SP",
            "slices": [
                {"element": "QFA", "device": "SR/QF"},
                {"element": "QD", "device": "SR/QD"},
            ],
            "engine": {"attribute": "PolynomB", "index": 1},
        }
    )
    document, _ = built_document(tmp_path, tree)
    assert measurement_groups(document, "SR")["quad"] == ["Q:FAMILY:SP", "QD:SP", "QF:SP"]


@pytest.mark.slow
def test_the_nsls2_storage_correctors_are_180_per_plane_and_disjoint(
    mml_built: Callable[[str, str], BuiltModel],
) -> None:
    """NSLS-II's storage-lattice correctors are combined: each element is steered from both planes."""
    built = mml_built("nsls2", "nsls2.storagering")
    groups = measurement_groups(built.document, built.name)
    assert len(groups["hcor"]) == len(set(groups["hcor"])) == 180
    assert len(groups["vcor"]) == len(set(groups["vcor"])) == 180
    assert not set(groups["hcor"]) & set(groups["vcor"])


@pytest.mark.slow
def test_the_spear3_correctors_keep_their_planes_apart(
    mml_built: Callable[[str, str], BuiltModel],
) -> None:
    """Each SPEAR3 group of 78 lists six correctors of the other plane, which stay in theirs.

    The ``HCM`` family holds 9SCY1 and 9SCY2, steered vertically; ``VCM`` holds
    9SCX1 to 9SCX4, steered horizontally.
    """
    built = mml_built("spear3", "spear3.storagering")
    document = built.document
    (hcm,) = [g["members"] for g in document["groups"] if g["id"] == "HCM"]
    (vcm,) = [g["members"] for g in document["groups"] if g["id"] == "VCM"]
    groups = measurement_groups(document, built.name)
    assert (len(hcm), len(vcm)) == (78, 78)
    assert (len(groups["hcor"]), len(groups["vcor"])) == (76, 74)
    assert not set(groups["hcor"]) & set(groups["vcor"])


# --- what stops --------------------------------------------------------------------


def test_a_file_naming_a_missing_group_stops(tmp_path: Path) -> None:
    tree = measured_tree()
    tree["measurement/LINE.yaml"]["groups"]["hcor"] = "LINE/NONE"
    stop = _stop(tmp_path, tree)
    assert (stop.kind, stop.record_kind, stop.record_id) == (
        "reference-missing",
        "measurement",
        "LINE",
    )
    assert "group LINE/NONE" in stop.detail


def test_a_kind_whose_member_the_file_does_not_name_stops(tmp_path: Path) -> None:
    tree = measured_tree()
    del tree["measurement/LINE.yaml"]["groups"]["vcor"]
    stop = _stop(tmp_path, tree)
    assert str(stop.format_message()) == (
        "facility: reference-missing: measurement LINE — kind orm needs `groups.vcor`, which "
        "the file does not name; fix: name `groups.vcor`, or remove orm from `kinds`"
    )


def test_a_group_the_model_does_not_wire_stops(tmp_path: Path) -> None:
    """LINE/HCM's corrector is steered in x only once its y setpoint is unwired from vcor."""
    tree = measured_tree()
    tree["measurement/LINE.yaml"]["groups"]["vcor"] = "LINE/HCM"
    (line,) = [model for model in tree["models.yaml"] if model["name"] == "LINE"]
    line["wiring"] = [w for w in line["wiring"] if w["address"] != "LCOR:V:SP"]
    stop = _stop(tmp_path, tree)
    assert str(stop.format_message()) == (
        "facility: reference-missing: measurement LINE — kind orm needs `groups.vcor`; group "
        "LINE/HCM holds no y-plane setpoint model LINE wires; fix: name a group holding a "
        "y-plane setpoint model LINE wires as `groups.vcor`"
    )


def test_an_instrument_the_model_does_not_wire_stops(tmp_path: Path) -> None:
    tree = measured_tree()
    tree["records/channels.yaml"].append({"id": "SR:TUNE:S", "on": {"place": "SR"}})
    tree["measurement/SR.yaml"]["instruments"]["tune"] = "SR:TUNE:S"
    stop = _stop(tmp_path, tree)
    assert str(stop.format_message()) == (
        "facility: reference-missing: measurement SR — kind trm needs `instruments.tune`; "
        "model SR does not wire SR:TUNE:S; fix: wire SR:TUNE:S in model SR, or name a "
        "channel it wires as `instruments.tune`"
    )


def _with_sextupole_group(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with SR's sextupole SX wired and named ``groups.sext``."""
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
    return tree


def test_a_crm_file_without_rf_stops_naming_it(tmp_path: Path) -> None:
    """pyAML measures chromaticity from the tunes and an RF step, so crm needs both."""
    stop = _stop(tmp_path, _with_sextupole_group(measured_tree()))
    assert str(stop.format_message()) == (
        "facility: reference-missing: measurement SR — kind crm needs `instruments.rf`, which "
        "the file does not name; fix: name `instruments.rf`, or remove crm from `kinds`"
    )


def test_a_crm_file_without_tune_stops_naming_it(tmp_path: Path) -> None:
    tree = with_rf(_with_sextupole_group(measured_tree()))
    del tree["measurement/SR.yaml"]["instruments"]["tune"]
    stop = _stop(tmp_path, tree)
    assert str(stop.format_message()) == (
        "facility: reference-missing: measurement SR — kind crm needs `instruments.tune`, "
        "which the file does not name; fix: name `instruments.tune`, or remove crm from `kinds`"
    )


def test_a_crm_file_with_sext_tune_and_rf_builds(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, with_rf(_with_sextupole_group(measured_tree())))
    assert _model(document, "SR")["measurement"]["kinds"] == ["crm"]


def test_a_kind_other_than_orm_on_a_single_pass_model_stops(tmp_path: Path) -> None:
    tree = measured_tree()
    tree["measurement/LINE.yaml"] |= {"kinds": ["orm", "trm"], "quad_delta": 0.001}
    stop = _stop(tmp_path, tree)
    assert str(stop.format_message()) == (
        "facility: reference-missing: measurement LINE — kind trm needs a periodic solve; "
        "model LINE solves single_pass; fix: remove trm from `kinds`; a single_pass model "
        "allows orm alone"
    )


def test_a_step_key_the_kind_needs_missing_stops(tmp_path: Path) -> None:
    tree = measured_tree()
    del tree["measurement/LINE.yaml"]["corrector_delta"]
    stop = _stop(tmp_path, tree)
    assert str(stop.format_message()) == (
        "facility: value-invalid: measurement LINE — kind orm needs `corrector_delta`, which "
        "the file does not state; fix: state `corrector_delta`, or remove orm from `kinds`"
    )


def test_one_group_named_for_both_corrector_planes_builds_split_by_plane(
    tmp_path: Path,
) -> None:
    """``hcor`` stands for the group's x-plane setpoints and ``vcor`` for its y-plane ones."""
    tree = measured_tree()
    tree["measurement/LINE.yaml"]["groups"]["vcor"] = "LINE/HCM"
    document, _ = built_document(tmp_path, tree)
    assert _model(document, "LINE")["measurement"]["groups"]["vcor"] == "LINE/HCM"
    groups = measurement_groups(document, "LINE")
    assert groups["hcor"] == ["LCOR:H:SP"]
    assert groups["vcor"] == ["LCOR:V:SP"]


def test_a_kind_is_carried_into_the_facility_file_as_authored(tmp_path: Path) -> None:
    tree = measured_tree()
    tree["measurement/SR.yaml"]["kinds"] = ["trm"]
    document, _ = built_document(tmp_path, tree)
    assert _model(document, "SR")["measurement"]["kinds"] == ["trm"]
    assert _model(document, "SR")["measurement"]["quad_delta"] == 0.001
