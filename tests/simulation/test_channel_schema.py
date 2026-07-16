"""Tests for :mod:`osprey.simulation.channel_schema`.

Schema completeness/consistency checks (task 2.1 gate): every
:data:`ALS_U_AR` family has exactly one schema entry, systems/fields/
per-subfield metadata match the shipped tier-3 DB conventions the schema is
sourced from (including the full ``STATUS``/BPM ``OFFSET`` field set), and
the genuinely-new families (QFA/SHF/SHD) are structural clones of their
closest existing analog with concrete, non-placeholder templated
descriptions.

Deterministic and offline: pure dataclass/dict introspection, no DB I/O, no
EPICS / softioc / MATLAB / network / LLM dependency.
"""

from __future__ import annotations

import dataclasses

import pytest

from osprey.simulation.channel_schema import (
    CHANNEL_SCHEMA,
    FamilyChannelSchema,
    SubfieldSchema,
    schema_for,
)
from osprey.simulation.facility_spec import ALS_U_AR

_NEW_FAMILIES = ("QFA", "SHF", "SHD")
_ANALOG_ANALOGS = {"QFA": "QF", "SHF": "SF", "SHD": "SD"}
_PLACEHOLDER_MARKERS = ("TODO", "FIXME", "XXX")

_MAGNETS_WITH_FAULT = ("QF", "QD", "QFA", "DIPOLE", "SF", "SD", "SHF", "SHD")
_CORRECTORS_NO_FAULT = ("HCM", "VCM")


def test_schema_covers_every_declared_family_exactly():
    assert set(CHANNEL_SCHEMA) == set(ALS_U_AR.family_names())


def test_bpm_system_is_diag():
    assert CHANNEL_SCHEMA["BPM"].system == "DIAG"


def test_magnets_and_correctors_system_is_mag():
    for fam in ALS_U_AR.families:
        if fam.kind in ("magnet", "corrector"):
            assert CHANNEL_SCHEMA[fam.name].system == "MAG", fam.name


def test_magnets_and_correctors_declare_current_sp_rb_golden():
    for fam in ALS_U_AR.families:
        if fam.kind in ("magnet", "corrector"):
            subfields = CHANNEL_SCHEMA[fam.name].fields["CURRENT"]
            assert set(subfields) == {"SP", "RB", "GOLDEN"}, fam.name


@pytest.mark.parametrize("family", _MAGNETS_WITH_FAULT)
def test_magnets_declare_status_ready_on_fault(family):
    subfields = CHANNEL_SCHEMA[family].fields["STATUS"]
    assert set(subfields) == {"READY", "ON", "FAULT"}


@pytest.mark.parametrize("family", _CORRECTORS_NO_FAULT)
def test_correctors_declare_status_ready_on_without_fault(family):
    subfields = CHANNEL_SCHEMA[family].fields["STATUS"]
    assert set(subfields) == {"READY", "ON"}
    assert "FAULT" not in subfields


def test_bpm_declares_full_field_set():
    assert set(CHANNEL_SCHEMA["BPM"].fields) == {"POSITION", "GOLDEN", "OFFSET", "STATUS"}


def test_bpm_position_golden_offset_are_x_y():
    for field_name in ("POSITION", "GOLDEN", "OFFSET"):
        assert set(CHANNEL_SCHEMA["BPM"].fields[field_name]) == {"X", "Y"}


def test_bpm_declares_status_valid_connected():
    assert set(CHANNEL_SCHEMA["BPM"].fields["STATUS"]) == {"VALID", "CONNECTED"}


# ── Per-subfield metadata (DataType / HWUnits / description) ────────────────


def test_analog_current_subfields_are_double_amps():
    for fam in ALS_U_AR.families:
        if fam.kind in ("magnet", "corrector"):
            for sub, meta in CHANNEL_SCHEMA[fam.name].fields["CURRENT"].items():
                assert meta.data_type == "double", (fam.name, sub)
                assert meta.hw_units == "A", (fam.name, sub)
                assert meta.description.strip()


def test_analog_bpm_subfields_are_double_mm():
    for field_name in ("POSITION", "GOLDEN", "OFFSET"):
        for sub, meta in CHANNEL_SCHEMA["BPM"].fields[field_name].items():
            assert meta.data_type == "double", (field_name, sub)
            assert meta.hw_units == "mm", (field_name, sub)
            assert meta.description.strip()


def test_status_subfields_are_enum_unitless():
    for fam_name, schema in CHANNEL_SCHEMA.items():
        for sub, meta in schema.fields["STATUS"].items():
            assert meta.data_type == "enum", (fam_name, sub)
            assert meta.hw_units == "", (fam_name, sub)
            assert meta.description.strip()


def test_status_subfield_description_is_lowercased_word():
    # e.g. READY -> "ready", ON -> "on", FAULT -> "fault", VALID -> "valid".
    for schema in CHANNEL_SCHEMA.values():
        for sub, meta in schema.fields["STATUS"].items():
            assert meta.description == sub.lower()


def test_current_subfield_descriptions_match_shipped_db():
    expected = {"SP": "setpoint", "RB": "readback", "GOLDEN": "golden setpoint"}
    for fam in ALS_U_AR.families:
        if fam.kind in ("magnet", "corrector"):
            current = CHANNEL_SCHEMA[fam.name].fields["CURRENT"]
            for sub, desc in expected.items():
                assert current[sub].description == desc, fam.name


def test_bpm_xy_descriptions_are_horizontal_vertical():
    for field_name in ("POSITION", "GOLDEN", "OFFSET"):
        assert CHANNEL_SCHEMA["BPM"].fields[field_name]["X"].description == "horizontal"
        assert CHANNEL_SCHEMA["BPM"].fields[field_name]["Y"].description == "vertical"


# ── Family-level display metadata ────────────────────────────────────────


@pytest.mark.parametrize("family", sorted(ALS_U_AR.family_names()))
def test_every_family_has_complete_display_metadata(family):
    schema = CHANNEL_SCHEMA[family]
    assert schema.display_name.strip()
    assert schema.description_template.strip()
    assert schema.common_name_template.strip()
    assert schema.fields


@pytest.mark.parametrize("family", sorted(ALS_U_AR.family_names()))
def test_descriptions_render_with_a_device_id(family):
    schema = CHANNEL_SCHEMA[family]
    rendered = schema.describe(3)
    assert rendered
    assert "{id" not in rendered
    common = schema.common_name(3)
    assert common
    assert "{id" not in common


@pytest.mark.parametrize("family", _NEW_FAMILIES)
def test_new_families_have_concrete_templated_descriptions(family):
    schema = CHANNEL_SCHEMA[family]
    rendered = schema.describe(1)
    upper = rendered.upper()
    for marker in _PLACEHOLDER_MARKERS:
        assert marker not in upper, f"{family} description looks like a placeholder: {rendered!r}"
    # Deterministic: the same id always renders the same description (no
    # LLM/network dependency, no per-run randomness).
    assert schema.describe(1) == rendered


@pytest.mark.parametrize("family,analog", sorted(_ANALOG_ANALOGS.items()))
def test_new_families_are_structural_clones_of_their_analog(family, analog):
    """QFA/SHF/SHD must be token-independent structural clones of QF/SF/SD:
    identical fields/subfields and identical per-subfield metadata -- only
    the family token and the family-level display prose differ.
    """
    new_schema = CHANNEL_SCHEMA[family]
    analog_schema = CHANNEL_SCHEMA[analog]
    assert new_schema.system == analog_schema.system
    assert new_schema.fields == analog_schema.fields
    # Display prose must differ (else it's not a distinct family).
    assert new_schema.display_name != analog_schema.display_name
    assert new_schema.describe(1) != analog_schema.describe(1)
    assert new_schema.common_name(1) != analog_schema.common_name(1)


def test_schema_for_missing_family_raises_keyerror():
    with pytest.raises(KeyError):
        schema_for("NOPE")


def test_schema_for_returns_declared_entry():
    assert schema_for("BPM") is CHANNEL_SCHEMA["BPM"]


def test_family_schema_is_frozen():
    schema = CHANNEL_SCHEMA["QF"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        schema.display_name = "mutated"  # type: ignore[misc]


def test_subfield_schema_is_frozen():
    meta = CHANNEL_SCHEMA["QF"].fields["CURRENT"]["SP"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.data_type = "mutated"  # type: ignore[misc]


def test_family_channel_schema_is_a_dataclass():
    assert dataclasses.is_dataclass(FamilyChannelSchema)
    assert dataclasses.is_dataclass(SubfieldSchema)
