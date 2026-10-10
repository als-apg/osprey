"""Every name a pyAML view holds maps to one address or id, and back."""

from __future__ import annotations

import pytest

from pyaml_cs_osprey.names import (
    NAME_RE,
    RF_PLANT_NAME,
    TUNE_MONITOR_NAME,
    UnmappedName,
    ViewNames,
    pyaml_name,
)

#: The demo SR model: two correctors, a BPM device, its groups and its RF setpoint.
SR_HCOR = "SR:MAG:HCM:01:CURRENT:SP"
SR_VCOR = "SR:MAG:VCM:01:CURRENT:SP"
SR_RF = "SR:RF:CAVITY:01:FREQUENCY:SP"
SR_GROUPS = {"bpm": "SR/BPM", "hcor": "SR/HCM", "vcor": "SR/VCM"}

#: One NSLS-II transfer-line corrector: both planes on one combined-corrector element.
NSLS2_H = "LTB-MG{Cor:1}I:Sp1-SP"
NSLS2_V = "LTB-MG{Cor:1}I:Sp2-SP"


def _sr() -> ViewNames:
    return ViewNames.build(
        magnets=[SR_HCOR, SR_VCOR], bpms=["SR/BPM01"], groups=SR_GROUPS, rf=SR_RF
    )


@pytest.mark.parametrize(
    ("identifier", "name"),
    [
        ("SR:MAG:HCM:01:CURRENT:SP", "SR:MAG:HCM:01:CURRENT:SP"),
        ("SR/BPM", "SR_BPM"),
        ("LQ:SP", "LQ:SP"),
        ("SR:C02-MG{PS:QH1A}I:Sp1-SP", "SR:C02-MG_PS:QH1A_I:Sp1-SP"),
        ("10G-COR1H:CurrSetpt", "10G-COR1H:CurrSetpt"),
    ],
)
def test_a_name_keeps_name_characters_and_folds_the_rest(identifier: str, name: str) -> None:
    assert pyaml_name(identifier) == name
    assert NAME_RE.fullmatch(name)


@pytest.mark.parametrize("identifier", ["", "{}", "//"])
def test_an_identifier_with_no_name_character_is_refused(identifier: str) -> None:
    with pytest.raises(ValueError, match="pyAML name"):
        pyaml_name(identifier)


def test_sr_model_names_round_trip() -> None:
    names = _sr()
    for address in (SR_HCOR, SR_VCOR):
        assert names.magnet_address(names.magnet_name(address)) == address
    assert names.bpm_device(names.bpm_name("SR/BPM01")) == "SR/BPM01"
    assert names.rf_address(names.rf_plant_name(SR_RF)) == SR_RF
    assert names.rf_plant_name(SR_RF) == RF_PLANT_NAME


def test_the_sr_model_group_arrays_map_back_to_their_ids() -> None:
    names = _sr()
    arrays = {group: names.array_name(role, group) for role, group in SR_GROUPS.items()}
    assert arrays == {"SR/BPM": "SR_BPM", "SR/HCM": "SR_HCM", "SR/VCM": "SR_VCM"}
    for group, array in arrays.items():
        assert "/" not in array
        assert NAME_RE.fullmatch(array)
        assert names.array_group(array) == group


def test_a_line_setpoint_round_trips() -> None:
    names = ViewNames.build(magnets=["LQ:SP"], bpms=["LINE/BPM1"], groups={"quad": "LINE/Q"})
    assert names.magnet_address(names.magnet_name("LQ:SP")) == "LQ:SP"
    assert names.array_group(names.array_name("quad", "LINE/Q")) == "LINE/Q"
    assert names.rf is None


def test_a_combined_corrector_gets_one_magnet_per_plane() -> None:
    """Both planes of one NSLS-II corrector element are two magnets, each by its address."""
    names = ViewNames.build(magnets=[NSLS2_H, NSLS2_V], groups={"hcor": "HCM", "vcor": "VCM"})
    h, v = names.magnet_name(NSLS2_H), names.magnet_name(NSLS2_V)
    assert h != v
    assert (names.magnet_address(h), names.magnet_address(v)) == (NSLS2_H, NSLS2_V)
    assert all(NAME_RE.fullmatch(name) for name in (h, v))


def test_one_group_named_for_both_corrector_planes_is_one_array_per_plane() -> None:
    names = ViewNames.build(groups={"bpm": "SR/BPM", "hcor": "SR/COR", "vcor": "SR/COR"})
    horizontal = names.array_name("hcor", "SR/COR")
    vertical = names.array_name("vcor", "SR/COR")
    assert (horizontal, vertical) == ("SR_COR_h", "SR_COR_v")
    assert names.array_group(horizontal) == names.array_group(vertical) == "SR/COR"
    assert names.array_name("bpm", "SR/BPM") == "SR_BPM"


def test_a_plane_array_name_another_group_would_take_is_refused() -> None:
    with pytest.raises(ValueError, match="SR/COR_h and SR/COR would both be named SR_COR_h"):
        ViewNames.build(groups={"bpm": "SR/COR_h", "hcor": "SR/COR", "vcor": "SR/COR"})


@pytest.mark.parametrize(("role", "group"), [("quad", "SR/QF"), ("vcor", "SR/HCM")])
def test_an_array_is_looked_up_by_the_role_naming_its_group(role: str, group: str) -> None:
    with pytest.raises(UnmappedName, match=group) as refused:
        _sr().array_name(role, group)
    assert refused.value.key == group


@pytest.mark.parametrize(
    ("lookup", "key"),
    [
        ("magnet_name", "SR:MAG:QF:01:CURRENT:SP"),
        ("bpm_name", "SR/BPM02"),
        ("rf_plant_name", "SR:RF:CAVITY:02:FREQUENCY:SP"),
        ("magnet_address", "SR_QF01"),
        ("bpm_device", "SR_BPM02"),
        ("array_group", "SR_QF"),
        ("rf_address", TUNE_MONITOR_NAME),
    ],
)
def test_an_unmapped_address_id_or_name_is_refused_naming_it(lookup: str, key: str) -> None:
    with pytest.raises(UnmappedName, match=key.replace("{", r"\{")) as refused:
        getattr(_sr(), lookup)(key)
    assert refused.value.key == key


def test_a_view_without_rf_maps_no_rf_address() -> None:
    names = ViewNames.build(magnets=[SR_HCOR])
    with pytest.raises(UnmappedName, match=SR_RF):
        names.rf_plant_name(SR_RF)


def test_two_addresses_sharing_a_name_are_refused_naming_both() -> None:
    with pytest.raises(ValueError, match=r"A\{1\} and A_1_"):
        ViewNames.build(magnets=["A{1}", "A_1_"])


def test_a_magnet_and_a_bpm_sharing_a_name_are_refused() -> None:
    with pytest.raises(ValueError, match="magnet setpoint SR_X and BPM device SR/X"):
        ViewNames.build(magnets=["SR_X"], bpms=["SR/X"])


def test_a_name_pyaml_reserves_is_refused() -> None:
    with pytest.raises(ValueError, match="pyAML reserves"):
        ViewNames.build(magnets=[RF_PLANT_NAME])


def test_a_repeated_address_is_one_magnet() -> None:
    names = ViewNames.build(magnets=[SR_HCOR, SR_HCOR])
    assert dict(names.magnets) == {SR_HCOR: SR_HCOR}
