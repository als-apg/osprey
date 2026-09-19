"""Tests for the VA bindings document schema (``bindings.py``).

The document is the contract between the emit lane that writes it and the
virtual accelerator that serves from it, so the tests below pin the rules
rather than the implementation: which keys a binding carries, what each kind
may and may not say, which cross-binding collisions are refused, and that a
document dumped from code is exactly the document that loads back.
"""

from __future__ import annotations

import json

import pytest

from osprey.services.virtual_accelerator.bindings import (
    Binding,
    BindingsDocument,
    BindingsError,
    Linear,
    Slice,
    Table,
    dump_bindings,
    load_bindings,
    parse_bindings,
    setpoints,
)

SHA = "a" * 64


def _linear(gain: float = 2.0, offset: float = 0.5) -> dict:
    return {"kind": "linear", "gain": gain, "offset": offset}


def _table(grid: list | None = None, values: list | None = None) -> dict:
    return {
        "kind": "table",
        "grid": [-1.0, 0.0, 1.0] if grid is None else grid,
        "values": [-2.0, 0.5, 3.0] if values is None else values,
    }


def _strength(**overrides) -> dict:
    body = {
        "kind": "strength",
        "family": "fam_a",
        "setpoint_address": "one:sp",
        "readback_address": "one:rb",
        "readback": "identity",
        "element": "fam_a_01_1",
        "attribute": "PolynomB",
        "index": 1,
        "slices": [{"element": "fam_a_01_1", "weight": 1.0}],
        "owner": "fam_a",
        "calibration": _linear(),
        "monitor_inverse": None,
        "nominal": 12.5,
        "energy_scaling": "brho",
        "energy_table": None,
    }
    body.update(overrides)
    return body


def _kick(**overrides) -> dict:
    body = _strength(
        kind="kick",
        family="fam_b",
        setpoint_address="two:sp",
        readback_address="two:rb",
        readback="inverse",
        element="fam_b_02_1_1",
        attribute="KickAngle",
        index=0,
        slices=[
            {"element": "fam_b_02_1_1", "weight": 0.5},
            {"element": "fam_b_02_1_3", "weight": 0.5},
        ],
        owner="fam_b",
        monitor_inverse=_table(),
        nominal=0.0,
    )
    body.update(overrides)
    return body


def _monitor(**overrides) -> dict:
    body = _strength(
        kind="monitor",
        family="fam_c",
        setpoint_address="three:x",
        readback_address=None,
        readback="inverse",
        element="fam_c_03_1",
        attribute="x",
        index=None,
        slices=[{"element": "fam_c_03_1", "weight": 1.0}],
        owner="fam_c",
        calibration=_linear(gain=1.0e-3, offset=0.0),
        monitor_inverse=_linear(gain=1.0e3, offset=0.0),
        nominal=None,
        energy_scaling="none",
    )
    body.update(overrides)
    return body


def _energy(**overrides) -> dict:
    body = _strength(
        kind="energy",
        family="fam_d",
        setpoint_address="four:sp",
        readback_address="four:rb",
        readback="identity",
        element=None,
        attribute=None,
        index=None,
        slices=[],
        owner=None,
        calibration=None,
        monitor_inverse=None,
        nominal=300.0,
        energy_scaling="none",
        energy_table=_table(grid=[280.0, 300.0, 320.0], values=[2.8, 3.0, 3.2]),
    )
    body.update(overrides)
    return body


def _rf(**overrides) -> dict:
    body = _strength(
        kind="rf",
        family="fam_e",
        setpoint_address="five:sp",
        readback_address=None,
        readback="same_as_setpoint",
        element="fam_e_00_1",
        attribute="Frequency",
        index=None,
        slices=[{"element": "fam_e_00_1", "weight": 1.0}],
        owner="fam_e",
        calibration=_linear(gain=1.0e6, offset=0.0),
        monitor_inverse=None,
        nominal=499.64,
        energy_scaling="none",
    )
    body.update(overrides)
    return body


def _document(bindings: list | None = None, **overrides) -> dict:
    body = {
        "system": "StorageRing",
        "energy_gev": 3.0,
        "lattice_sha256": SHA,
        "bindings": [_strength()] if bindings is None else bindings,
    }
    body.update(overrides)
    return body


def _refused(document) -> BindingsError:
    with pytest.raises(BindingsError) as excinfo:
        parse_bindings(document)
    return excinfo.value


class TestDocumentHead:
    """The document-level facts every consumer stamps or checks."""

    def test_parses_the_head_and_its_bindings(self):
        doc = parse_bindings(_document())
        assert doc.system == "StorageRing"
        assert doc.energy_gev == 3.0
        assert doc.lattice_sha256 == SHA
        assert doc.provenance is None
        assert len(doc.bindings) == 1
        assert doc.bindings[0].family == "fam_a"

    def test_keeps_an_optional_provenance_stamp(self):
        doc = parse_bindings(_document(_provenance="exporter=2.0.0"))
        assert doc.provenance == "exporter=2.0.0"

    def test_refuses_an_unknown_top_level_key(self):
        error = _refused(_document(timestamp="2026-01-01"))
        assert error.key == "timestamp"
        assert "unknown key" in error.message

    def test_refuses_a_missing_top_level_key(self):
        body = _document()
        del body["energy_gev"]
        assert _refused(body).key == "energy_gev"

    def test_refuses_a_top_level_that_is_not_an_object(self):
        assert _refused([]).key == ""

    def test_refuses_a_digest_that_is_not_lowercase_hex_sha256(self):
        assert _refused(_document(lattice_sha256=SHA.upper())).key == "lattice_sha256"
        assert _refused(_document(lattice_sha256="a" * 63)).key == "lattice_sha256"

    def test_refuses_an_energy_that_is_not_positive(self):
        assert _refused(_document(energy_gev=0.0)).key == "energy_gev"
        assert _refused(_document(energy_gev="3.0")).key == "energy_gev"

    def test_refuses_an_empty_system(self):
        assert _refused(_document(system="")).key == "system"

    def test_accepts_a_document_with_no_bindings(self):
        assert parse_bindings(_document(bindings=[])).bindings == ()

    def test_refuses_bindings_that_are_not_a_list(self):
        assert _refused(_document(bindings={})).key == "bindings"


class TestBindingStructure:
    """Every binding key is present, spelled once, and typed."""

    def test_parses_each_kind(self):
        doc = parse_bindings(_document([_strength(), _kick(), _monitor(), _energy(), _rf()]))
        assert [binding.kind for binding in doc.bindings] == [
            "strength",
            "kick",
            "monitor",
            "energy",
            "rf",
        ]

    def test_refuses_a_missing_binding_key(self):
        body = _strength()
        del body["nominal"]
        assert _refused(_document([body])).key == "bindings[0].nominal"

    def test_refuses_an_unknown_binding_key(self):
        assert _refused(_document([_strength(axis="x")])).key == "bindings[0].axis"

    def test_refuses_an_unknown_kind(self):
        error = _refused(_document([_strength(kind="skew")]))
        assert error.key == "bindings[0].kind"
        assert "'skew'" in error.message

    def test_refuses_an_unknown_readback_rule(self):
        assert _refused(_document([_strength(readback="echo")])).key == "bindings[0].readback"

    def test_refuses_an_unknown_energy_scaling(self):
        assert (
            _refused(_document([_strength(energy_scaling="brho2")])).key
            == "bindings[0].energy_scaling"
        )

    def test_refuses_an_empty_family_or_address(self):
        assert _refused(_document([_strength(family="")])).key == "bindings[0].family"
        assert (
            _refused(_document([_strength(setpoint_address="")])).key
            == "bindings[0].setpoint_address"
        )

    def test_reports_the_position_of_the_offending_binding(self):
        assert _refused(_document([_strength(), _kick(index=7)])).key == "bindings[1].index"


class TestSlices:
    """A split device writes every slice and reads the first."""

    def test_keeps_slice_order_and_weights(self):
        binding = parse_bindings(_document([_kick()])).bindings[0]
        assert binding.slices == (
            Slice(element="fam_b_02_1_1", weight=0.5),
            Slice(element="fam_b_02_1_3", weight=0.5),
        )

    def test_refuses_an_element_kind_with_no_slice(self):
        assert _refused(_document([_strength(slices=[])])).key == "bindings[0].slices"

    def test_refuses_a_first_slice_that_is_not_the_bound_element(self):
        body = _strength(slices=[{"element": "other_01_1", "weight": 1.0}])
        assert _refused(_document([body])).key == "bindings[0].slices[0].element"

    def test_refuses_a_repeated_slice_element(self):
        body = _strength(
            slices=[
                {"element": "fam_a_01_1", "weight": 0.5},
                {"element": "fam_a_01_1", "weight": 0.5},
            ]
        )
        assert _refused(_document([body])).key == "bindings[0].slices[1].element"

    def test_refuses_a_zero_or_non_finite_weight(self):
        body = _strength(slices=[{"element": "fam_a_01_1", "weight": 0.0}])
        assert _refused(_document([body])).key == "bindings[0].slices[0].weight"

    def test_refuses_an_unknown_slice_key(self):
        body = _strength(slices=[{"element": "fam_a_01_1", "weight": 1.0, "plane": "x"}])
        assert _refused(_document([body])).key == "bindings[0].slices[0].plane"


class TestCurves:
    """Calibrations and inverses are linear pairs or sampled tables."""

    def test_parses_a_linear_curve(self):
        binding = parse_bindings(_document([_strength()])).bindings[0]
        assert binding.calibration == Linear(gain=2.0, offset=0.5)

    def test_parses_a_table_curve(self):
        binding = parse_bindings(_document([_kick()])).bindings[0]
        assert binding.monitor_inverse == Table(grid=(-1.0, 0.0, 1.0), values=(-2.0, 0.5, 3.0))

    def test_accepts_a_descending_grid(self):
        body = _kick(monitor_inverse=_table(grid=[1.0, 0.0, -1.0], values=[3.0, 0.5, -2.0]))
        binding = parse_bindings(_document([body])).bindings[0]
        assert binding.monitor_inverse.grid == (1.0, 0.0, -1.0)

    def test_refuses_an_unknown_curve_kind(self):
        assert (
            _refused(_document([_strength(calibration={"kind": "poly"})])).key
            == "bindings[0].calibration.kind"
        )

    def test_refuses_a_zero_gain(self):
        assert (
            _refused(_document([_strength(calibration=_linear(gain=0.0))])).key
            == "bindings[0].calibration.gain"
        )

    def test_refuses_a_grid_and_values_of_different_length(self):
        body = _kick(monitor_inverse=_table(values=[1.0, 2.0]))
        assert _refused(_document([body])).key == "bindings[0].monitor_inverse.values"

    def test_refuses_a_grid_that_is_not_strictly_monotonic(self):
        body = _kick(monitor_inverse=_table(grid=[0.0, 1.0, 1.0]))
        assert _refused(_document([body])).key == "bindings[0].monitor_inverse.grid[2]"

    def test_refuses_a_grid_of_one_point(self):
        body = _kick(monitor_inverse=_table(grid=[1.0], values=[2.0]))
        assert _refused(_document([body])).key == "bindings[0].monitor_inverse.grid"

    def test_refuses_a_non_numeric_grid_entry(self):
        body = _kick(monitor_inverse=_table(grid=[0.0, "1.0", 2.0]))
        assert _refused(_document([body])).key == "bindings[0].monitor_inverse.grid[1]"


class TestKindRules:
    """What each kind must say, and what it may not."""

    def test_strength_takes_a_polynomial_attribute_and_an_index(self):
        assert _refused(_document([_strength(attribute="KickAngle")])).key == (
            "bindings[0].attribute"
        )
        assert _refused(_document([_strength(index=None)])).key == "bindings[0].index"
        assert _refused(_document([_strength(index=-1)])).key == "bindings[0].index"
        assert _refused(_document([_strength(index=1.0)])).key == "bindings[0].index"

    def test_kick_takes_one_of_the_two_kick_components(self):
        assert parse_bindings(_document([_kick(index=1)])).bindings[0].index == 1
        assert _refused(_document([_kick(index=2)])).key == "bindings[0].index"

    def test_monitor_takes_a_transverse_axis_and_no_index(self):
        assert parse_bindings(_document([_monitor()])).bindings[0].attribute == "x"
        assert _refused(_document([_monitor(attribute="PolynomB")])).key == "bindings[0].attribute"
        assert _refused(_document([_monitor(index=0)])).key == "bindings[0].index"

    def test_monitor_serves_its_reading_on_its_own_address_through_the_inverse(self):
        assert _refused(_document([_monitor(readback="identity", monitor_inverse=None)])).key == (
            "bindings[0].readback"
        )
        assert _refused(_document([_monitor(readback_address="three:rb")])).key == (
            "bindings[0].readback_address"
        )

    def test_energy_binds_no_element_and_carries_a_table(self):
        assert _refused(_document([_energy(energy_table=None)])).key == "bindings[0].energy_table"
        assert _refused(_document([_energy(element="fam_d_01_1")])).key == "bindings[0].element"
        assert _refused(_document([_energy(owner="fam_d")])).key == "bindings[0].owner"
        assert _refused(_document([_energy(slices=[{"element": "e", "weight": 1.0}])])).key == (
            "bindings[0].slices"
        )

    def test_energy_converts_through_its_table_only(self):
        assert _refused(_document([_energy(calibration=_linear())])).key == (
            "bindings[0].calibration"
        )
        assert _refused(_document([_energy(readback="inverse", monitor_inverse=_table())])).key == (
            "bindings[0].readback"
        )
        assert _refused(_document([_energy(nominal=None)])).key == "bindings[0].nominal"

    def test_only_a_strength_or_a_kick_scales_with_rigidity(self):
        assert _refused(_document([_monitor(energy_scaling="brho")])).key == (
            "bindings[0].energy_scaling"
        )
        assert _refused(_document([_rf(energy_scaling="brho")])).key == "bindings[0].energy_scaling"
        assert parse_bindings(_document([_kick(energy_scaling="brho")])).bindings

    def test_an_energy_table_belongs_to_the_energy_kind_alone(self):
        assert _refused(_document([_strength(energy_table=_table())])).key == (
            "bindings[0].energy_table"
        )

    def test_rf_writes_the_cavity_frequency(self):
        assert parse_bindings(_document([_rf()])).bindings[0].attribute == "Frequency"
        assert _refused(_document([_rf(attribute="Voltage")])).key == "bindings[0].attribute"

    def test_a_writable_kind_needs_a_nominal_and_a_calibration(self):
        assert _refused(_document([_strength(nominal=None)])).key == "bindings[0].nominal"
        assert _refused(_document([_strength(calibration=None)])).key == "bindings[0].calibration"

    def test_an_element_kind_names_an_element_and_its_owner(self):
        assert _refused(_document([_strength(element=None)])).key == "bindings[0].element"
        assert _refused(_document([_strength(owner=None)])).key == "bindings[0].owner"


class TestReadbackRules:
    """How the readback value is produced, and where it is served."""

    def test_identity_serves_the_written_value_on_its_own_address(self):
        binding = parse_bindings(_document([_strength()])).bindings[0]
        assert binding.readback == "identity"
        assert binding.readback_address == "one:rb"
        assert binding.monitor_inverse is None

    def test_identity_refuses_an_inverse_it_would_never_apply(self):
        assert _refused(_document([_strength(monitor_inverse=_table())])).key == (
            "bindings[0].monitor_inverse"
        )

    def test_inverse_needs_the_exported_inverse(self):
        assert _refused(_document([_kick(monitor_inverse=None)])).key == (
            "bindings[0].monitor_inverse"
        )

    def test_same_as_setpoint_carries_no_second_address(self):
        binding = parse_bindings(_document([_rf()])).bindings[0]
        assert binding.readback_address is None
        assert _refused(_document([_rf(readback_address="five:rb")])).key == (
            "bindings[0].readback_address"
        )

    def test_a_second_address_is_required_unless_it_collapses(self):
        assert _refused(_document([_strength(readback_address=None)])).key == (
            "bindings[0].readback_address"
        )

    def test_refuses_a_readback_address_equal_to_the_setpoint(self):
        assert _refused(_document([_strength(readback_address="one:sp")])).key == (
            "bindings[0].readback_address"
        )


class TestReadout:
    """The one optional key: how a monitor's own reading is calibrated."""

    def test_parses_the_four_numbers_a_reading_is_corrected_by(self):
        body = _monitor(readout={"gain": 1.02, "offset": 0.12, "roll": 0.001, "crunch": 0.002})
        readout = parse_bindings(_document([body])).bindings[0].readout
        assert (readout.gain, readout.offset, readout.roll, readout.crunch) == (
            1.02,
            0.12,
            0.001,
            0.002,
        )
        assert readout.stated == ("gain", "offset", "roll", "crunch")

    def test_a_number_the_facility_does_not_state_stays_absent(self):
        # Absent is not zero. A consumer reading a missing offset as 0.0 and a
        # missing gain as 1.0 gets the same reading either way; one reading it
        # as a stated number would correct by something nobody measured.
        readout = parse_bindings(_document([_monitor(readout={"gain": 0.99})])).bindings[0].readout
        assert readout.stated == ("gain",)
        assert (readout.offset, readout.roll, readout.crunch) == (None, None, None)

    def test_a_binding_that_states_none_of_it_carries_none(self):
        assert parse_bindings(_document([_monitor()])).bindings[0].readout is None

    def test_refuses_a_readout_stating_nothing(self):
        error = _refused(_document([_monitor(readout={})]))
        assert error.key == "bindings[0].readout"
        assert "leave the key out" in error.message

    def test_refuses_a_readout_on_anything_but_a_reading(self):
        # A driven family's exported gains calibrate its setpoint, which the
        # calibration states; there is no published reading to correct.
        error = _refused(_document([_strength(readout={"gain": 1.0})]))
        assert error.key == "bindings[0].readout"
        assert "publishes no reading" in error.message

    def test_refuses_an_unknown_correction(self):
        error = _refused(_document([_monitor(readout={"gain": 1.0, "tilt": 0.5})]))
        assert error.key == "bindings[0].readout.tilt"

    def test_refuses_a_correction_that_is_not_a_finite_number(self):
        for value in ("1.02", None, float("nan")):
            error = _refused(_document([_monitor(readout={"gain": value})]))
            assert error.key == "bindings[0].readout.gain"

    def test_refuses_a_readout_that_is_not_an_object(self):
        assert _refused(_document([_monitor(readout=1.02)])).key == "bindings[0].readout"


class TestCrossBindingRules:
    """One address, one writer; one element field, one family."""

    def test_refuses_two_bindings_on_one_setpoint_address(self):
        error = _refused(_document([_strength(), _kick(setpoint_address="one:sp")]))
        assert error.key == "bindings[1].setpoint_address"
        assert "fam_a" in error.message

    def test_refuses_a_readback_address_another_binding_already_uses(self):
        body = _kick(readback_address="one:rb")
        assert _refused(_document([_strength(), body])).key == "bindings[1].readback_address"

    def test_refuses_two_bindings_writing_one_element_field(self):
        body = _strength(
            family="fam_z",
            setpoint_address="nine:sp",
            readback_address="nine:rb",
            owner="fam_a",
        )
        error = _refused(_document([_strength(), body]))
        assert error.key == "bindings[1].slices[0].element"
        assert "fam_a" in error.message

    def test_allows_two_families_on_one_element_through_different_fields(self):
        body = _strength(
            family="fam_z",
            setpoint_address="nine:sp",
            readback_address="nine:rb",
            attribute="PolynomA",
            owner="fam_a",
        )
        assert len(parse_bindings(_document([_strength(), body])).bindings) == 2

    def test_refuses_a_second_energy_knob(self):
        body = _energy(family="fam_f", setpoint_address="six:sp", readback_address="six:rb")
        assert _refused(_document([_energy(), body])).key == "bindings[1].kind"


class TestSetpoints:
    """The writable addresses the manifest partition is checked against."""

    def test_lists_writable_addresses_in_document_order(self):
        doc = parse_bindings(_document([_rf(), _strength(), _monitor(), _energy()]))
        assert setpoints(doc) == ("five:sp", "one:sp", "four:sp")

    def test_is_empty_for_a_document_of_monitors(self):
        assert setpoints(parse_bindings(_document([_monitor()]))) == ()


class TestDump:
    """The emitted bytes are a pure function of the document."""

    def test_round_trips_through_text(self):
        doc = parse_bindings(_document([_strength(), _kick(), _monitor(), _energy(), _rf()]))
        assert parse_bindings(json.loads(dump_bindings(doc))) == doc

    def test_writes_sorted_keys_and_one_trailing_newline(self):
        text = dump_bindings(parse_bindings(_document()))
        assert text.endswith("}\n")
        assert not text.endswith("}\n\n")
        body = json.loads(text)
        assert list(body) == sorted(body)
        assert list(body["bindings"][0]) == sorted(body["bindings"][0])

    def test_is_byte_stable_and_carries_no_stamp(self):
        doc = parse_bindings(_document(_provenance="exporter=2.0.0"))
        text = dump_bindings(doc)
        assert text == dump_bindings(doc)
        assert "timestamp" not in text
        assert '"_provenance": "exporter=2.0.0"' in text

    def test_writes_every_key_including_the_null_ones(self):
        body = json.loads(dump_bindings(parse_bindings(_document())))["bindings"][0]
        assert body["monitor_inverse"] is None
        assert body["energy_table"] is None
        assert set(body) == set(_strength())

    def test_writes_a_readout_only_where_one_is_stated_and_only_what_it_states(self):
        stated = _monitor(readout={"gain": 1, "roll": 0.001})
        body = json.loads(dump_bindings(parse_bindings(_document([stated, _strength()]))))
        assert body["bindings"][0]["readout"] == {"gain": 1.0, "roll": 0.001}
        assert "readout" not in body["bindings"][1]

    def test_round_trips_a_readout_through_text(self):
        doc = parse_bindings(_document([_monitor(readout={"gain": 1.02, "crunch": -0.002})]))
        assert parse_bindings(json.loads(dump_bindings(doc))) == doc

    def test_writes_whole_numbers_as_floats(self):
        doc = parse_bindings(_document(energy_gev=3, bindings=[_strength(nominal=12)]))
        assert '"energy_gev": 3.0' in dump_bindings(doc)
        assert '"nominal": 12.0' in dump_bindings(doc)

    def test_refuses_a_document_built_in_code_that_would_not_load(self):
        doc = BindingsDocument(
            system="StorageRing",
            energy_gev=3.0,
            lattice_sha256=SHA,
            bindings=(
                Binding(
                    kind="kick",
                    family="fam_b",
                    setpoint_address="two:sp",
                    readback_address="two:rb",
                    readback="identity",
                    element="fam_b_02_1_1",
                    attribute="KickAngle",
                    index=2,
                    slices=(Slice(element="fam_b_02_1_1", weight=1.0),),
                    owner="fam_b",
                    calibration=Linear(gain=1.0, offset=0.0),
                    monitor_inverse=None,
                    nominal=0.0,
                    energy_scaling="none",
                    energy_table=None,
                ),
            ),
        )
        with pytest.raises(BindingsError) as excinfo:
            dump_bindings(doc)
        assert excinfo.value.key == "bindings[0].index"


class TestLoad:
    """Reading the file back, with the file named in every refusal."""

    def _write(self, tmp_path, text: str):
        path = tmp_path / "va_bindings.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_what_dump_wrote(self, tmp_path):
        doc = parse_bindings(_document([_strength(), _monitor()]))
        path = self._write(tmp_path, dump_bindings(doc))
        assert load_bindings(path) == doc

    def test_refuses_a_missing_file_by_name(self, tmp_path):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_bindings(tmp_path / "absent.json")
        assert "absent.json" in str(excinfo.value)

    def test_names_the_file_in_a_schema_refusal(self, tmp_path):
        path = self._write(tmp_path, json.dumps(_document([_strength(index=None)])))
        with pytest.raises(BindingsError) as excinfo:
            load_bindings(path)
        assert excinfo.value.key == "bindings[0].index"
        assert "va_bindings.json" in str(excinfo.value)

    def test_refuses_text_that_is_not_json(self, tmp_path):
        path = self._write(tmp_path, "{")
        with pytest.raises(BindingsError) as excinfo:
            load_bindings(path)
        assert "va_bindings.json" in str(excinfo.value)

    def test_refuses_a_non_finite_number(self, tmp_path):
        path = self._write(tmp_path, json.dumps(_document(energy_gev=float("inf"))))
        with pytest.raises(BindingsError):
            load_bindings(path)
