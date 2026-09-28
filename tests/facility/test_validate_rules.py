"""Stages S3 to S5 of the facility build and the stage runner, on synthetic trees."""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.errors import FacilityBuildError
from osprey.facility.sources import load_sources
from osprey.facility.validate import (
    SET_VALUED_SLOTS,
    STAGES,
    StageReport,
    known_classes,
    ordered_slots,
    run_stages,
    schema_document,
    signal_roles,
    sort_errors,
    validate,
)

CORE_YAML = Path(__file__).parents[2] / "src" / "osprey" / "facility" / "schema" / "core.yaml"

QUAD = {"id": "SR/Q1", "class": "Quadrupole"}
BPM = {"id": "SR/BPM1", "class": "BeamPositionMonitor"}
SP = {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB", "on": {"device": "SR/Q1"}}
RB = {"id": "Q1:RB", "on": {"device": "SR/Q1"}}
X = {"id": "BPM1:X", "on": {"device": "SR/BPM1"}}


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False)
        path.write_text(text, encoding="utf-8")
    return root


def _tree(**extra: Any) -> dict[str, Any]:
    """A clean tree: one quadrupole with a setpoint/readback pair, one BPM."""
    files: dict[str, Any] = {
        "records/devices.yaml": [QUAD, BPM],
        "records/channels.yaml": [SP, RB, X],
    }
    files.update(extra)
    return files


def _with_channels(*channels: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return _tree(**{"records/channels.yaml": [SP, RB, X, *channels]}, **extra)


def _run(tmp_path: Path, files: dict[str, Any]) -> StageReport:
    return run_stages(_write(tmp_path / "facility", files), project_name="my proj")


def _lines(result: StageReport) -> list[str]:
    return [error.format_message() for error in result.errors]


def _one(tmp_path: Path, files: dict[str, Any], stage: str) -> FacilityBuildError:
    result = _run(tmp_path, files)
    assert result.failed == stage, _lines(result)
    assert len(result.errors) == 1, _lines(result)
    return result.errors[0]


def _wired(address: str, model: str = "optics", **wiring: Any) -> dict[str, Any]:
    entry = {"address": address, **(wiring or {"element": "Q1"})}
    return {"name": model, "engine": "pyat", "wiring": [entry]}


# --- the stage runner ---------------------------------------------------------------


class TestStageRunner:
    def test_the_stages_are_fixed(self) -> None:
        assert STAGES == (
            "load",
            "combine",
            "schema",
            "references",
            "records",
            "compute",
            "views",
        )

    def test_a_clean_tree_passes_every_stage(self, tmp_path: Path) -> None:
        result = _run(tmp_path, _tree())
        assert result.ok and result.failed is None and result.errors == []

    def test_a_missing_directory_passes_every_stage(self, tmp_path: Path) -> None:
        result = run_stages(tmp_path / "absent", project_name="my proj")
        assert result.ok
        assert result.validated.document is not None
        assert result.validated.document["identity"] == {"code": "my_proj", "name": "my proj"}

    def test_a_tree_without_limits_passes_every_stage(self, tmp_path: Path) -> None:
        files = _tree(**{"models.yaml": [_wired("Q1:SP")]})
        assert not (tmp_path / "facility" / "limits.yaml").exists()
        result = _run(tmp_path, files)
        assert result.ok, _lines(result)
        assert "limits" not in result.validated.document

    def test_a_load_error_stops_before_combine(self, tmp_path: Path) -> None:
        files = _tree(**{"fixes.yaml": {"schema": "wrong"}, "records/extra.yaml": []})
        result = _run(tmp_path, files)
        assert result.failed == "load"
        assert result.validated.combined is None
        assert [e.kind for e in result.errors] == ["source-invalid"]

    def test_a_combine_error_stops_before_the_schema(self, tmp_path: Path) -> None:
        files = _tree(**{"imported/mml/devices.yaml": [{"id": "SR/Q1", "class": "Sextupole"}]})
        result = _run(tmp_path, files)
        assert result.failed == "combine"
        assert result.validated.document is None
        assert [e.kind for e in result.errors] == ["layer-conflict"]

    def test_a_missing_reference_stops_before_the_record_rules(self, tmp_path: Path) -> None:
        # The pair names a missing channel: reference-missing only, never pair-invalid.
        setpoint = {**SP, "pair": "Q1:GONE"}
        files = _tree(**{"records/channels.yaml": [setpoint, RB, X]})
        error = _one(tmp_path, files, "references")
        assert error.kind == "reference-missing"

    def test_later_stages_run_only_when_the_earlier_ones_are_clean(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def compute(_validated: Any) -> list[FacilityBuildError]:
            seen.append("compute")
            return [
                FacilityBuildError(
                    "span-invalid", "P", ["x"], "fix", record_kind="place", detail="bad"
                )
            ]

        def views(_validated: Any) -> list[FacilityBuildError]:
            seen.append("views")
            return []

        root = _write(tmp_path / "facility", _tree())
        result = run_stages(root, project_name="p", later=[("compute", compute), ("views", views)])
        assert result.failed == "compute" and seen == ["compute"]

        bad = _write(tmp_path / "bad", _tree(**{"seeds.yaml": {"NOPE": {"noise": 1.0}}}))
        seen.clear()
        result = run_stages(bad, project_name="p", later=[("compute", compute)])
        assert result.failed == "references" and seen == []

    def test_raise_first_raises_the_first_sorted_error(self, tmp_path: Path) -> None:
        files = _tree(**{"seeds.yaml": {"B:GONE": {"noise": 1.0}, "A:GONE": {"noise": 1.0}}})
        result = _run(tmp_path, files)
        with pytest.raises(FacilityBuildError) as caught:
            result.raise_first()
        assert caught.value is result.errors[0]
        assert caught.value.record_id == "A:GONE"


class TestGather:
    def test_errors_sort_by_table_row_then_record_kind_then_id(self) -> None:
        def error(kind: str, record_kind: str, rid: str) -> FacilityBuildError:
            return FacilityBuildError(kind, rid, ["f"], "r", record_kind=record_kind, detail="d")

        errors = [
            error("limit-invalid", "path", "limits.defaults"),
            error("source-invalid", "path", "b"),
            error("class-unknown", "device", "A"),
            error("reference-missing", "seed", "A"),
            error("reference-missing", "channel", "Z"),
            error("source-invalid", "path", "a"),
        ]
        ordered = [(e.kind, e.record_kind, e.record_id) for e in sort_errors(errors)]
        assert ordered == [
            ("source-invalid", "path", "a"),
            ("source-invalid", "path", "b"),
            ("reference-missing", "channel", "Z"),
            ("reference-missing", "seed", "A"),
            ("class-unknown", "device", "A"),
            ("limit-invalid", "path", "limits.defaults"),
        ]

    def test_validate_prints_every_error_of_the_first_failing_stage(self, tmp_path: Path) -> None:
        files = _tree(
            **{
                "seeds.yaml": {"B:GONE": {"noise": 1.0}, "A:GONE": {"noise": 1.0}},
                "records/devices.yaml": [QUAD, {**BPM, "class": "NoSuchClass"}],
            }
        )
        stream = io.StringIO()
        code = validate(_write(tmp_path / "facility", files), project_name="p", file=stream)
        assert code == 1
        assert stream.getvalue().splitlines() == [
            "facility: reference-missing: seed A:GONE — seeds.yaml `address` names channel "
            "A:GONE, which does not exist; fix: add channel A:GONE or correct `address`",
            "facility: reference-missing: seed B:GONE — seeds.yaml `address` names channel "
            "B:GONE, which does not exist; fix: add channel B:GONE or correct `address`",
            "facility: class-unknown: device SR/BPM1 — class NoSuchClass is in neither the "
            "vocabulary nor classes.yaml; fix: use a vocabulary class or add it to classes.yaml",
        ]

    def test_validate_of_a_clean_tree_prints_nothing(self, tmp_path: Path) -> None:
        stream = io.StringIO()
        assert validate(_write(tmp_path / "f", _tree()), project_name="p", file=stream) == 0
        assert stream.getvalue() == ""

    def test_load_stops_keep_pn_local_and_case_rules(self, tmp_path: Path) -> None:
        files = _tree(
            **{
                "models.yaml": [
                    {"name": "Optics", "engine": "pyat"},
                    {"name": "optics", "engine": "pyat"},
                    {"name": "bad-name", "engine": "pyat"},
                ],
                "identity.yaml": {"code": "1bad"},
            }
        )
        result = _run(tmp_path, files)
        assert result.failed == "load"
        assert [(e.kind, e.record_kind, e.record_id) for e in result.errors] == [
            ("source-invalid", "model", "Optics"),
            ("source-invalid", "model", "bad-name"),
            ("source-invalid", "path", "identity.yaml.code"),
        ]


# --- S3: schema ---------------------------------------------------------------------


class TestSchemaStage:
    def test_the_document_gets_a_header_and_the_folded_identity(self, tmp_path: Path) -> None:
        result = _run(tmp_path, _tree())
        document = result.validated.document
        assert list(document)[:2] == ["schema", "identity"]
        assert document["schema"] == "osprey.facility.facility/1"
        assert document["identity"] == {"code": "my_proj", "name": "my proj"}

    def test_identity_yaml_supplies_the_identity(self, tmp_path: Path) -> None:
        files = _tree(**{"identity.yaml": {"code": "demo", "name": "Demo"}})
        result = _run(tmp_path, files)
        assert result.ok
        assert result.validated.document["identity"] == {"code": "demo", "name": "Demo"}

    def test_the_sources_and_the_combined_document_are_not_modified(self, tmp_path: Path) -> None:
        loaded = load_sources(_write(tmp_path / "facility", _tree()))
        before = copy.deepcopy(loaded.sources)
        document = {"channels": [], "identity": {"code": "x"}}
        filled = schema_document(document, loaded.sources, project_name="p")
        assert document == {"channels": [], "identity": {"code": "x"}}
        assert loaded.sources == before
        assert filled["identity"] == {"code": "p", "name": "p"}

    def test_a_pydantic_failure_is_one_line_per_path(self, tmp_path: Path) -> None:
        files = _tree(**{"seeds.yaml": {"Q1:RB": {"noise": "loud", "drift": {"amplitude": 1.0}}}})
        result = _run(tmp_path, files)
        assert result.failed == "schema"
        assert [(e.kind, e.record_kind, e.record_id) for e in result.errors] == [
            ("source-invalid", "path", "channels.1.simulation.drift.period_s"),
            ("source-invalid", "path", "channels.1.simulation.noise"),
        ]
        assert result.errors[0].format_message() == (
            "facility: source-invalid: path channels.1.simulation.drift.period_s — "
            "seeds.yaml: field required; fix: add `period_s` to seeds.yaml"
        )
        assert result.errors[0].sources == ("seeds.yaml",)

    def test_a_union_slot_failure_is_one_line(self, tmp_path: Path) -> None:
        files = _tree(**{"seeds.yaml": {"Q1:RB": {"linear": {"BPM1:X": "much"}}}})
        error = _one(tmp_path, files, "schema")
        assert (error.kind, error.record_id) == (
            "source-invalid",
            "channels.1.simulation.linear.BPM1:X",
        )

    def test_a_bad_value_names_the_record_file(self, tmp_path: Path) -> None:
        files = _with_channels({"id": "T", "value_type": "complex"})
        error = _one(tmp_path, files, "schema")
        assert error.record_id == "channels.3.value_type"
        assert error.sources == ("records/channels.yaml",)
        assert error.remedy == "correct `value_type` in records/channels.yaml"

    def test_limits_without_defaults_is_limit_invalid(self, tmp_path: Path) -> None:
        files = _tree(**{"limits.yaml": {"records": [{"address": "Q1:SP"}]}})
        error = _one(tmp_path, files, "schema")
        assert error.format_message() == (
            "facility: limit-invalid: path limits.defaults — limits.yaml has no `defaults` "
            "block; fix: add `defaults: {writable: <bool>, confirm: <bool>}` to limits.yaml,"
            " or delete limits.yaml to run without limits"
        )

    def test_limits_without_defaults_prints_beside_other_schema_errors(
        self, tmp_path: Path
    ) -> None:
        files = _tree(
            **{
                "limits.yaml": {"records": [{"address": "Q1:SP", "max_step": "big"}]},
            }
        )
        result = _run(tmp_path, files)
        assert [(e.kind, e.record_id) for e in result.errors] == [
            ("source-invalid", "limits.records.0.max_step"),
            ("limit-invalid", "limits.defaults"),
        ]

    @pytest.mark.parametrize(
        "text",
        ["", "# no limits yet\n", "schema: osprey.facility.limits/1\n"],
        ids=["empty", "comment-only", "header-only"],
    )
    def test_a_limits_file_holding_no_limits_counts_as_absent(
        self, tmp_path: Path, text: str
    ) -> None:
        result = _run(tmp_path, _tree(**{"limits.yaml": text}))
        assert result.ok, _lines(result)
        assert "limits" not in result.validated.document

    def test_other_limit_defaults_failures_stay_source_invalid(self, tmp_path: Path) -> None:
        files = _tree(**{"limits.yaml": {"defaults": {"writable": False}}})
        error = _one(tmp_path, files, "schema")
        assert (error.kind, error.record_id) == ("source-invalid", "limits.defaults.confirm")

    @pytest.mark.parametrize(
        ("slot", "value"),
        [("overrides", ["Q1:SP", 1.0]), ("faults", {"optics": 3})],
    )
    def test_scenario_maps_have_their_shape(self, tmp_path: Path, slot: str, value: Any) -> None:
        files = _tree(**{"scenarios/s.yaml": {slot: value}})
        error = _one(tmp_path, files, "schema")
        assert (error.kind, error.record_id) == ("source-invalid", f"scenarios.0.{slot}")
        assert error.sources == ("scenarios/s.yaml",)


# --- S4: references -----------------------------------------------------------------


def _missing(error: FacilityBuildError, record_kind: str, rid: str, fragment: str) -> None:
    assert error.kind == "reference-missing", error.format_message()
    assert (error.record_kind, error.record_id) == (record_kind, rid)
    assert fragment in error.detail, error.format_message()


class TestReferences:
    @pytest.mark.parametrize(
        ("channel", "fragment"),
        [
            ({"id": "T", "on": {"device": "SR/GONE"}}, "`on.device` names device SR/GONE"),
            ({"id": "T", "on": {"place": "SR"}}, "`on.place` names place SR"),
            ({"id": "T", "role": "setpoint", "pair": "GONE"}, "`pair` names channel GONE"),
            ({"id": "T", "endpoint_of": ["SR/GONE"]}, "`endpoint_of` names device SR/GONE"),
        ],
        ids=["on-device", "on-place", "pair", "endpoint_of"],
    )
    def test_channel_slots(self, tmp_path: Path, channel: dict[str, Any], fragment: str) -> None:
        error = _one(tmp_path, _with_channels(channel), "references")
        _missing(error, "channel", "T", fragment)
        assert error.sources == ("records/channels.yaml",)

    def test_line_names_file_and_field(self, tmp_path: Path) -> None:
        files = _with_channels({"id": "T", "on": {"device": "SR/GONE"}})
        assert _one(tmp_path, files, "references").format_message() == (
            "facility: reference-missing: channel T — records/channels.yaml `on.device` names "
            "device SR/GONE, which does not exist; fix: add device SR/GONE or correct "
            "`on.device`"
        )

    def test_linear_input(self, tmp_path: Path) -> None:
        files = _tree(**{"seeds.yaml": {"Q1:RB": {"linear": {"GONE": 1.0}}}})
        error = _one(tmp_path, files, "references")
        _missing(error, "channel", "Q1:RB", "seeds.yaml `simulation.linear` names channel GONE")

    def test_group_member(self, tmp_path: Path) -> None:
        files = _tree(**{"records/groups.yaml": [{"id": "G", "members": ["SR/Q1", "SR/GONE"]}]})
        error = _one(tmp_path, files, "references")
        _missing(error, "group", "G", "`members` names device SR/GONE")

    def test_place_parent_path(self, tmp_path: Path) -> None:
        files = _tree(**{"records/places.yaml": [{"id": "SR/S01"}]})
        error = _one(tmp_path, files, "references")
        _missing(error, "place", "SR/S01", "names place SR,")

    def test_span_model(self, tmp_path: Path) -> None:
        span = {"model": "optics", "from_marker": "M1"}
        files = _tree(**{"records/places.yaml": [{"id": "SR", "span": span}]})
        error = _one(tmp_path, files, "references")
        _missing(error, "place", "SR", "`span.model` names model optics")

    def test_device_place(self, tmp_path: Path) -> None:
        files = _tree(**{"records/devices.yaml": [{**QUAD, "place": "SR"}, BPM]})
        error = _one(tmp_path, files, "references")
        _missing(error, "device", "SR/Q1", "`place` names place SR")

    def test_wiring_address(self, tmp_path: Path) -> None:
        files = _tree(**{"models.yaml": [_wired("GONE")]})
        error = _one(tmp_path, files, "references")
        _missing(error, "wiring", "optics/GONE", "models.yaml `address` names channel GONE")

    def test_slice_device(self, tmp_path: Path) -> None:
        slices = [{"element": "Q1"}, {"element": "Q2", "device": "SR/GONE"}]
        files = _tree(**{"models.yaml": [_wired("Q1:SP", slices=slices)]})
        error = _one(tmp_path, files, "references")
        _missing(error, "wiring", "optics/Q1:SP", "`slices.device` names device SR/GONE")

    def test_a_filled_slice_device_is_reported_once_on_the_channel(self, tmp_path: Path) -> None:
        channel = {"id": "T", "on": {"device": "SR/GONE"}}
        files = _with_channels(
            channel, **{"models.yaml": [_wired("T", slices=[{"element": "Q1"}])]}
        )
        error = _one(tmp_path, files, "references")
        _missing(error, "channel", "T", "`on.device` names device SR/GONE")

    def test_seed_address(self, tmp_path: Path) -> None:
        files = _tree(**{"seeds.yaml": {"GONE": {"noise": 0.1}}})
        _missing(_one(tmp_path, files, "references"), "seed", "GONE", "seeds.yaml")

    def test_limit_address(self, tmp_path: Path) -> None:
        limits = {"defaults": {"writable": False, "confirm": True}, "records": [{"address": "G"}]}
        files = _tree(**{"limits.yaml": limits})
        error = _one(tmp_path, files, "references")
        _missing(error, "limit", "G", "limits.yaml `records.address` names channel G")

    @pytest.mark.parametrize(
        ("scenario", "fragment"),
        [
            ({"overrides": {"GONE": 1.0}}, "`overrides` names channel GONE"),
            ({"archiver": [{"channel": "GONE", "events": []}]}, "`archiver.channel` names"),
            ({"faults": {"nomodel": {"Q1:SP": 1.0}}}, "`faults` names model nomodel"),
            ({"faults": {"optics": {"GONE": 1.0}}}, "`faults.optics` names channel GONE"),
        ],
        ids=["override", "archiver", "fault-model", "fault-target"],
    )
    def test_scenario_entries(
        self, tmp_path: Path, scenario: dict[str, Any], fragment: str
    ) -> None:
        files = _tree(**{"scenarios/s.yaml": scenario, "models.yaml": [_wired("Q1:SP")]})
        error = _one(tmp_path, files, "references")
        _missing(error, "scenario", "s", fragment)
        assert error.sources == ("scenarios/s.yaml",)

    @pytest.mark.parametrize(
        ("measurement", "rid", "fragment"),
        [
            ({"kinds": ["orm"], "groups": {"bpm": "NOGROUP"}}, "optics", "names group NOGROUP"),
            ({"kinds": ["orm"], "instruments": {"tune": "GONE"}}, "optics", "names channel GONE"),
        ],
        ids=["group", "instrument"],
    )
    def test_measurement_references(
        self, tmp_path: Path, measurement: dict[str, Any], rid: str, fragment: str
    ) -> None:
        files = _tree(**{"models.yaml": [_wired("Q1:SP")], "measurement/optics.yaml": measurement})
        _missing(_one(tmp_path, files, "references"), "measurement", rid, fragment)

    def test_measurement_file_of_a_missing_model(self, tmp_path: Path) -> None:
        files = _tree(**{"measurement/nomodel.yaml": {"kinds": ["orm"]}})
        error = _one(tmp_path, files, "references")
        _missing(error, "measurement", "nomodel", "names model nomodel")

    def test_a_former_address_names_the_new_address(self, tmp_path: Path) -> None:
        renamed = {**X, "former_addresses": ["OLD:X"]}
        files = _tree(
            **{
                "records/channels.yaml": [SP, RB, renamed],
                "seeds.yaml": {"OLD:X": {"noise": 0.1}},
            }
        )
        error = _one(tmp_path, files, "references")
        assert error.format_message() == (
            "facility: reference-missing: seed OLD:X — seeds.yaml `address` names channel "
            "OLD:X, which does not exist; channel BPM1:X lists it in former_addresses; "
            "fix: point `address` at BPM1:X"
        )


def _dropping_tree(**extra: Any) -> dict[str, Any]:
    """The tree plus an imported channel X2 that fixes.yaml drops."""
    fixes = {
        "schema": "osprey.facility.fixes/1",
        "fixes": [{"op": "drop", "kind": "channel", "id": "X2", "why": "A duplicate."}],
    }
    return _tree(
        **{
            "imported/mml/channels.yaml": [{"id": "X2", "on": {"device": "SR/BPM1"}}],
            "fixes.yaml": fixes,
            "models.yaml": [_wired("Q1:SP")],
        },
        **extra,
    )


class TestDroppedReferences:
    DROP = "names channel X2, dropped by fix drop channel X2: A duplicate."

    def _check(self, error: FacilityBuildError, record_kind: str, rid: str) -> None:
        _missing(error, record_kind, rid, self.DROP)
        assert error.remedy == "remove the entry or drop the fix"
        assert "fixes.yaml" in error.sources

    def test_limits_record(self, tmp_path: Path) -> None:
        limits = {"defaults": {"writable": False, "confirm": True}, "records": [{"address": "X2"}]}
        error = _one(tmp_path, _dropping_tree(**{"limits.yaml": limits}), "references")
        self._check(error, "limit", "X2")

    def test_seed(self, tmp_path: Path) -> None:
        error = _one(
            tmp_path, _dropping_tree(**{"seeds.yaml": {"X2": {"noise": 0.1}}}), "references"
        )
        self._check(error, "seed", "X2")

    def test_scenario_override(self, tmp_path: Path) -> None:
        files = _dropping_tree(**{"scenarios/s.yaml": {"overrides": {"X2": 1.0}}})
        self._check(_one(tmp_path, files, "references"), "scenario", "s")

    def test_scenario_fault(self, tmp_path: Path) -> None:
        files = _dropping_tree(**{"scenarios/s.yaml": {"faults": {"optics": {"X2": 1.0}}}})
        self._check(_one(tmp_path, files, "references"), "scenario", "s")

    def test_the_drop_alone_builds(self, tmp_path: Path) -> None:
        result = _run(tmp_path, _dropping_tree())
        assert result.ok, _lines(result)


class TestClasses:
    def test_known_classes_join_vocabulary_and_classes_yaml(self) -> None:
        known = known_classes([{"class": "Kicker", "parent": "Magnet"}])
        assert {"Quadrupole", "Magnet", "Kicker"} <= known

    def test_unknown_device_class(self, tmp_path: Path) -> None:
        files = _tree(**{"records/devices.yaml": [QUAD, {**BPM, "class": "Kicker"}]})
        error = _one(tmp_path, files, "references")
        assert (error.kind, error.record_kind, error.record_id) == (
            "class-unknown",
            "device",
            "SR/BPM1",
        )

    def test_a_facility_added_class_is_known(self, tmp_path: Path) -> None:
        files = _tree(
            **{
                "classes.yaml": [
                    {"class": "Kicker", "parent": "Magnet"},
                    {"class": "FastKicker", "parent": "Kicker"},
                ],
                "records/devices.yaml": [QUAD, {**BPM, "class": "FastKicker"}],
            }
        )
        assert _run(tmp_path, files).ok

    def test_a_parent_must_be_listed_earlier(self, tmp_path: Path) -> None:
        files = _tree(
            **{
                "classes.yaml": [
                    {"class": "FastKicker", "parent": "Kicker"},
                    {"class": "Kicker", "parent": "Magnet"},
                ]
            }
        )
        error = _one(tmp_path, files, "references")
        assert (error.kind, error.record_kind, error.record_id) == (
            "class-unknown",
            "class",
            "FastKicker",
        )

    def test_an_unknown_parent(self, tmp_path: Path) -> None:
        files = _tree(**{"classes.yaml": [{"class": "Kicker", "parent": "Widget"}]})
        assert _one(tmp_path, files, "references").kind == "class-unknown"

    def test_a_channel_signal_must_be_a_vocabulary_role(self, tmp_path: Path) -> None:
        role = sorted(signal_roles())[0]
        clean = _with_channels({"id": "T", "signal": role})
        assert _run(tmp_path / "clean", clean).ok
        files = _with_channels({"id": "T", "signal": "hcm_current_sp"})
        error = _one(tmp_path / "bad", files, "references")
        assert error.format_message() == (
            "facility: class-unknown: channel T — signal hcm_current_sp is not a vocabulary "
            "signal role; fix: use a vocabulary signal role or remove `signal`"
        )


# --- S5: record rules ---------------------------------------------------------------


def _rule(tmp_path: Path, files: dict[str, Any], kind: str) -> FacilityBuildError:
    error = _one(tmp_path, files, "records")
    assert error.kind == kind, error.format_message()
    return error


class TestPairRules:
    def test_pair_on_a_readback(self, tmp_path: Path) -> None:
        files = _with_channels({"id": "T", "pair": "Q1:RB"})
        error = _rule(tmp_path, files, "pair-invalid")
        assert error.detail == "a `pair` on a readback channel"

    def test_pair_on_a_none_channel(self, tmp_path: Path) -> None:
        files = _with_channels({"id": "T", "role": "none", "pair": "Q1:RB"})
        assert _rule(tmp_path, files, "pair-invalid").detail == "a `pair` on a none channel"

    def test_pair_naming_another_setpoint(self, tmp_path: Path) -> None:
        files = _with_channels({"id": "T", "role": "setpoint", "pair": "Q1:SP"})
        error = _rule(tmp_path, files, "pair-invalid")
        assert error.detail == "`pair` names setpoint channel Q1:SP"

    def test_a_readback_paired_twice(self, tmp_path: Path) -> None:
        files = _with_channels({"id": "T", "role": "setpoint", "pair": "Q1:RB"})
        error = _rule(tmp_path, files, "pair-invalid")
        assert (error.record_id, error.detail) == (
            "Q1:RB",
            "the pair of setpoints Q1:SP, T",
        )

    def test_a_setpoint_without_pair_pairs_itself(self, tmp_path: Path) -> None:
        files = _with_channels({"id": "T", "role": "setpoint"})
        assert _run(tmp_path, files).ok

    @pytest.mark.parametrize(
        ("setpoint", "readback", "slots"),
        [
            ({"value_type": "int"}, {}, "`value_type`"),
            (
                {"value_type": "enum", "options": ["A", "B"]},
                {"value_type": "enum", "options": ["A", "C"]},
                "`options`",
            ),
            (
                {"value_type": "waveform", "shape": [2]},
                {"value_type": "waveform", "shape": [3]},
                "`shape`",
            ),
        ],
        ids=["value_type", "options", "shape"],
    )
    def test_setpoint_and_pair_share_their_type(
        self, tmp_path: Path, setpoint: dict[str, Any], readback: dict[str, Any], slots: str
    ) -> None:
        files = _tree(**{"records/channels.yaml": [{**SP, **setpoint}, {**RB, **readback}, X]})
        error = _rule(tmp_path, files, "pair-invalid")
        assert error.detail == f"setpoint and pair Q1:RB differ in {slots}"

    def test_setpoint_and_pair_are_owned_by_one_model(self, tmp_path: Path) -> None:
        files = _tree(**{"models.yaml": [_wired("Q1:SP"), _wired("Q1:RB", model="other")]})
        error = _rule(tmp_path, files, "pair-invalid")
        assert error.record_id == "Q1:SP"

    def test_element_and_slices_both_present(self, tmp_path: Path) -> None:
        files = _tree(
            **{"models.yaml": [_wired("Q1:SP", element="Q1", slices=[{"element": "Q1"}])]}
        )
        error = _rule(tmp_path, files, "pair-invalid")
        assert (error.record_kind, error.record_id) == ("wiring", "optics/Q1:SP")

    @pytest.mark.parametrize("weight", [0, float("inf"), float("nan")], ids=["zero", "inf", "nan"])
    def test_slice_weight(self, tmp_path: Path, weight: float) -> None:
        slices = [{"element": "Q1", "weight": weight}]
        files = _tree(**{"models.yaml": [_wired("Q1:SP", slices=slices)]})
        assert "zero or not finite" in _rule(tmp_path, files, "pair-invalid").detail

    def test_a_wired_endpoint_device_is_named_by_a_slice(self, tmp_path: Path) -> None:
        channel = {"id": "PS", "endpoint_of": ["SR/Q1", "SR/BPM1"]}
        slices = [{"element": "Q1", "device": "SR/Q1"}]
        files = _with_channels(channel, **{"models.yaml": [_wired("PS", slices=slices)]})
        error = _rule(tmp_path, files, "pair-invalid")
        assert error.detail == "`endpoint_of` device SR/BPM1 is named by no slice"

    def test_an_unwired_endpoint_channel_is_free(self, tmp_path: Path) -> None:
        files = _with_channels({"id": "PS", "endpoint_of": ["SR/Q1", "SR/BPM1"]})
        assert _run(tmp_path, files).ok

    def test_a_split_magnet_covers_every_endpoint(self, tmp_path: Path) -> None:
        channel = {"id": "PS", "endpoint_of": ["SR/Q1", "SR/BPM1"]}
        slices = [
            {"element": "Q1", "device": "SR/Q1", "weight": 2.0},
            {"element": "B1", "device": "SR/BPM1"},
        ]
        files = _with_channels(channel, **{"models.yaml": [_wired("PS", slices=slices)]})
        assert _run(tmp_path, files).ok


class TestValueRules:
    @pytest.mark.parametrize(
        ("channel", "detail"),
        [
            ({"value_type": "enum"}, "a enum channel has no `options`"),
            ({"value_type": "enum", "options": ["ONLY"]}, "fewer than 2 `options`"),
            ({"value_type": "bool", "options": ["A", "B", "C"]}, "has 3 `options`, not 2"),
            (
                {"value_type": "enum", "options": [f"L{i}" for i in range(17)]},
                "more than 16",
            ),
            ({"value_type": "enum", "options": ["A", "X" * 26]}, "at most 25 characters"),
            ({"value_type": "enum", "options": ["A", "A"]}, "repeats a label"),
            ({"options": ["A", "B"]}, "a float channel carries `options`"),
            ({"value_type": "waveform"}, "needs `shape`"),
            ({"value_type": "waveform", "shape": [0]}, "needs `shape`"),
            ({"value_type": "int", "shape": [2]}, "a int channel carries `shape`"),
        ],
        ids=[
            "enum-no-options",
            "enum-one-label",
            "bool-three-labels",
            "enum-too-many",
            "label-too-long",
            "label-repeated",
            "options-on-float",
            "waveform-no-shape",
            "waveform-zero-dim",
            "shape-on-int",
        ],
    )
    def test_options_and_shape_presence(
        self, tmp_path: Path, channel: dict[str, Any], detail: str
    ) -> None:
        error = _rule(tmp_path, _with_channels({"id": "T", **channel}), "value-invalid")
        assert detail in error.detail

    def test_a_bool_gets_its_default_labels(self, tmp_path: Path) -> None:
        result = _run(tmp_path, _with_channels({"id": "T", "value_type": "bool"}))
        assert result.ok

    @pytest.mark.parametrize("seed", [{"noise": 0.1}, {"drift": {"amplitude": 1, "period_s": 5}}])
    def test_motion_on_a_non_float_is_value_invalid_only(
        self, tmp_path: Path, seed: dict[str, Any]
    ) -> None:
        # A non-float setpoint with motion: value-invalid, never also seed-invalid.
        files = _tree(
            **{
                "records/channels.yaml": [
                    {**SP, "value_type": "int"},
                    {**RB, "value_type": "int"},
                    X,
                ],
                "seeds.yaml": {"Q1:SP": seed},
            }
        )
        error = _rule(tmp_path, files, "value-invalid")
        assert "apply to float channels only" in error.detail
        assert error.sources == ("seeds.yaml",)

    def test_drift_period_must_be_positive(self, tmp_path: Path) -> None:
        files = _tree(**{"seeds.yaml": {"Q1:RB": {"drift": {"amplitude": 1, "period_s": 0}}}})
        assert "period_s" in _rule(tmp_path, files, "value-invalid").detail

    @pytest.mark.parametrize(
        "clamp",
        [[2.0, 1.0], [1.0], [None, float("inf")], ["low", 1.0]],
        ids=["low-above-high", "one-side", "infinite", "string"],
    )
    def test_clamp_shape(self, tmp_path: Path, clamp: list[Any]) -> None:
        files = _tree(**{"seeds.yaml": {"Q1:RB": {"clamp": clamp}}})
        result = _run(tmp_path, files)
        assert result.failed in ("schema", "records"), _lines(result)
        if result.failed == "records":
            assert [e.kind for e in result.errors] == ["value-invalid"]

    def test_clamp_with_open_sides_builds(self, tmp_path: Path) -> None:
        files = _tree(**{"seeds.yaml": {"Q1:RB": {"clamp": [None, 5.0]}}})
        assert _run(tmp_path, files).ok

    def test_clamp_on_a_non_float(self, tmp_path: Path) -> None:
        files = _with_channels(
            {"id": "T", "value_type": "int"}, **{"seeds.yaml": {"T": {"clamp": [0, 1]}}}
        )
        assert "`clamp`" in _rule(tmp_path, files, "value-invalid").detail

    def test_linear_with_nominal(self, tmp_path: Path) -> None:
        seeds = {"T": {"linear": {"BPM1:X": 1.0}, "nominal": 2.0}}
        files = _with_channels({"id": "T"}, **{"seeds.yaml": seeds})
        assert "`nominal`" in _rule(tmp_path, files, "value-invalid").detail

    def test_linear_on_a_non_float(self, tmp_path: Path) -> None:
        seeds = {"T": {"linear": {"BPM1:X": 1.0}}}
        files = _with_channels({"id": "T", "value_type": "int"}, **{"seeds.yaml": seeds})
        assert "`linear`" in _rule(tmp_path, files, "value-invalid").detail

    def test_linear_with_a_wired_input(self, tmp_path: Path) -> None:
        seeds = {"T": {"linear": {"Q1:SP": 1.0}}}
        files = _with_channels(
            {"id": "T"}, **{"seeds.yaml": seeds, "models.yaml": [_wired("Q1:SP")]}
        )
        error = _rule(tmp_path, files, "value-invalid")
        assert error.detail == "`linear` input Q1:SP is wired"

    def test_linear_with_a_non_float_input(self, tmp_path: Path) -> None:
        seeds = {"T": {"linear": {"S": 1.0}}}
        files = _with_channels(
            {"id": "T"}, {"id": "S", "value_type": "string"}, **{"seeds.yaml": seeds}
        )
        assert "not float" in _rule(tmp_path, files, "value-invalid").detail

    def test_linear_cycle(self, tmp_path: Path) -> None:
        seeds = {"A": {"linear": {"B": 1.0}}, "B": {"linear": {"A": -1.0}}}
        files = _with_channels({"id": "A"}, {"id": "B"}, **{"seeds.yaml": seeds})
        error = _rule(tmp_path, files, "value-invalid")
        assert (error.record_id, error.detail) == ("A", "`linear` inputs form a cycle: A -> B -> A")

    def test_linear_over_unwired_inputs_builds(self, tmp_path: Path) -> None:
        seeds = {"NET": {"linear": {"FWD": 1.0, "REV": -1.0}}}
        files = _with_channels({"id": "FWD"}, {"id": "REV"}, {"id": "NET"}, **{"seeds.yaml": seeds})
        assert _run(tmp_path, files).ok

    @pytest.mark.parametrize(
        ("channel", "nominal"),
        [
            ({}, True),
            ({}, "high"),
            ({"value_type": "int"}, "3"),
            ({"value_type": "bool"}, "MAYBE"),
            ({"value_type": "enum", "options": ["A", "B"]}, 5),
            ({"value_type": "string"}, 1),
            ({"value_type": "waveform", "shape": [2]}, [1.0, 2.0, 3.0]),
        ],
        ids=[
            "float-bool",
            "float-str",
            "int-str",
            "bool-label",
            "enum-index",
            "string",
            "waveform",
        ],
    )
    def test_a_nominal_is_coerced_by_value_type(
        self, tmp_path: Path, channel: dict[str, Any], nominal: Any
    ) -> None:
        files = _with_channels(
            {"id": "T", **channel}, **{"seeds.yaml": {"T": {"nominal": nominal}}}
        )
        error = _rule(tmp_path, files, "value-invalid")
        assert error.detail.startswith("`nominal` of channel T: ")
        assert error.sources == ("seeds.yaml",)

    def test_a_coercible_nominal_builds(self, tmp_path: Path) -> None:
        channels = [
            {"id": "I", "value_type": "int"},
            {"id": "E", "value_type": "enum", "options": ["A", "B"]},
            {"id": "W", "value_type": "waveform", "shape": [2, 2]},
        ]
        seeds = {"I": {"nominal": 3.0}, "E": {"nominal": 1}, "W": {"nominal": [[1, 2], [3, 4]]}}
        assert _run(tmp_path, _with_channels(*channels, **{"seeds.yaml": seeds})).ok

    def test_limit_bounds_on_a_non_numeric_channel(self, tmp_path: Path) -> None:
        limits = {
            "defaults": {"writable": False, "confirm": True},
            "records": [{"address": "T", "min_value": 0.0}],
        }
        files = _with_channels({"id": "T", "value_type": "string"}, **{"limits.yaml": limits})
        error = _rule(tmp_path, files, "value-invalid")
        assert (error.record_kind, error.record_id) == ("limit", "T")

    def test_override_is_coerced(self, tmp_path: Path) -> None:
        files = _tree(**{"scenarios/s.yaml": {"overrides": {"Q1:SP": "high"}}})
        error = _rule(tmp_path, files, "value-invalid")
        assert (error.record_kind, error.record_id) == ("scenario", "s")
        assert error.detail.startswith("`overrides.Q1:SP` of channel Q1:SP: ")

    def test_fault_value_is_coerced(self, tmp_path: Path) -> None:
        files = _tree(
            **{
                "models.yaml": [_wired("Q1:SP")],
                "scenarios/s.yaml": {"faults": {"optics": {"Q1:RB": [1, 2]}}},
            }
        )
        error = _rule(tmp_path, files, "value-invalid")
        assert error.detail.startswith("`faults.optics.Q1:RB` of channel Q1:RB: ")

    def test_stuck_only_on_a_setpoint(self, tmp_path: Path) -> None:
        models = {"models.yaml": [_wired("Q1:SP")]}
        ok = _tree(**models, **{"scenarios/s.yaml": {"faults": {"optics": {"Q1:SP": "stuck"}}}})
        assert _run(tmp_path / "ok", ok).ok
        bad = _tree(**models, **{"scenarios/s.yaml": {"faults": {"optics": {"Q1:RB": "stuck"}}}})
        error = _rule(tmp_path / "bad", bad, "value-invalid")
        assert error.detail == "`faults.optics.Q1:RB` is `stuck` on a readback channel"

    def test_a_fault_field_map_is_left_to_the_engine(self, tmp_path: Path) -> None:
        files = _tree(
            **{
                "models.yaml": [_wired("Q1:SP")],
                "scenarios/s.yaml": {"faults": {"optics": {"BPM1:X": {"offset": 1e-4}}}},
            }
        )
        assert _run(tmp_path, files).ok


class TestLimitRules:
    DEFAULTS = {"writable": False, "confirm": True}

    def _limits(self, *records: dict[str, Any]) -> dict[str, Any]:
        return {"limits.yaml": {"defaults": self.DEFAULTS, "records": list(records)}}

    def test_writable_only_on_a_setpoint(self, tmp_path: Path) -> None:
        files = _tree(**self._limits({"address": "Q1:RB", "writable": True}))
        error = _rule(tmp_path, files, "limit-invalid")
        assert error.format_message() == (
            "facility: limit-invalid: limit Q1:RB — `writable: true` on a readback channel; "
            "fix: remove `writable`; only a setpoint is writable"
        )

    def test_a_locked_setpoint_builds(self, tmp_path: Path) -> None:
        files = _tree(
            **self._limits(
                {"address": "Q1:SP", "writable": False, "min_value": 0.0, "max_value": 1.0}
            )
        )
        assert _run(tmp_path, files).ok

    def test_integral_bounds_on_an_int_channel(self, tmp_path: Path) -> None:
        files = _with_channels(
            {"id": "T", "role": "setpoint", "value_type": "int"},
            **self._limits({"address": "T", "min_value": 0, "max_value": 2.5}),
        )
        error = _rule(tmp_path, files, "limit-invalid")
        assert error.detail == "`max_value` 2.5 on an int channel is not integral"


class TestSeedRules:
    def test_noise_on_a_setpoint(self, tmp_path: Path) -> None:
        files = _tree(**{"seeds.yaml": {"Q1:SP": {"noise": 0.1}}})
        error = _rule(tmp_path, files, "seed-invalid")
        assert error.detail == "a setpoint carries `noise`"

    def test_noise_on_a_wired_readback_builds(self, tmp_path: Path) -> None:
        files = _tree(**{"models.yaml": [_wired("BPM1:X")], "seeds.yaml": {"BPM1:X": {"noise": 1}}})
        assert _run(tmp_path, files).ok

    def test_nominal_on_a_wired_channel(self, tmp_path: Path) -> None:
        files = _tree(
            **{"models.yaml": [_wired("BPM1:X")], "seeds.yaml": {"BPM1:X": {"nominal": 1.0}}}
        )
        error = _rule(tmp_path, files, "seed-invalid")
        assert error.detail == "a `nominal` on a channel model optics wires"

    def test_non_integral_nominal_on_an_int_channel(self, tmp_path: Path) -> None:
        files = _with_channels(
            {"id": "T", "value_type": "int"}, **{"seeds.yaml": {"T": {"nominal": 2.5}}}
        )
        error = _rule(tmp_path, files, "seed-invalid")
        assert error.detail == "`nominal` of int channel T is 2.5, not integral"

    def test_non_integral_override_on_an_int_channel(self, tmp_path: Path) -> None:
        files = _with_channels(
            {"id": "T", "value_type": "int"}, **{"scenarios/s.yaml": {"overrides": {"T": 0.5}}}
        )
        error = _rule(tmp_path, files, "seed-invalid")
        assert (error.record_kind, error.record_id) == ("scenario", "s")

    def test_a_paired_readback_starts_at_its_setpoint(self, tmp_path: Path) -> None:
        agree = _tree(**{"seeds.yaml": {"Q1:SP": {"nominal": 2.0}, "Q1:RB": {"nominal": 2}}})
        assert _run(tmp_path / "agree", agree).ok
        differ = _tree(**{"seeds.yaml": {"Q1:SP": {"nominal": 2.0}, "Q1:RB": {"nominal": 3.0}}})
        error = _rule(tmp_path / "differ", differ, "seed-invalid")
        assert error.record_id == "Q1:RB"
        assert error.detail == "`nominal` 3.0 differs from its setpoint Q1:SP's 2.0"

    def test_a_paired_readback_against_an_unseeded_setpoint(self, tmp_path: Path) -> None:
        files = _tree(**{"seeds.yaml": {"Q1:RB": {"nominal": 1.0}}})
        error = _rule(tmp_path, files, "seed-invalid")
        assert error.detail == "`nominal` 1.0 differs from its setpoint Q1:SP's 0.0"


# --- ordered slots ------------------------------------------------------------------


class TestOrderedSlots:
    def test_ordered_slots_are_every_multivalued_slot_but_the_sets(self) -> None:
        core = yaml.safe_load(CORE_YAML.read_text(encoding="utf-8"))
        multivalued = {
            (name, slot)
            for name, cls in core["classes"].items()
            for slot, spec in (cls.get("attributes") or {}).items()
            if (spec or {}).get("multivalued")
        }
        assert SET_VALUED_SLOTS <= multivalued
        assert ordered_slots() == multivalued - SET_VALUED_SLOTS

    @pytest.mark.parametrize(
        "slot",
        [
            ("Wiring", "slices"),
            ("Channel", "names"),
            ("Channel", "options"),
            ("Channel", "shape"),
            ("Seed", "clamp"),
        ],
    )
    def test_the_named_lists_are_ordered(self, slot: tuple[str, str]) -> None:
        assert slot in ordered_slots()
