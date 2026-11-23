"""Tests for the facility build's error line and identity-code folding."""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from osprey.cli.main import cli, facility_errors_as_click
from osprey.facility import PN_LOCAL, fold_code
from osprey.facility.errors import (
    KINDS,
    WARNING_KINDS,
    FacilityBuildError,
    FacilityBuildWarning,
    FacilityError,
    FacilityInputError,
    FacilityUsageError,
    ImportStop,
    LayerSourceInvalid,
    MappingProblems,
)
from osprey_connectors.simulation.engines import FacilityDefectError


def _error(**overrides: object) -> FacilityBuildError:
    fields: dict[str, object] = {
        "kind": "reference-missing",
        "record_id": "QF1",
        "sources": ["records/devices.yaml"],
        "remedy": "add place S01 or correct the device's place",
        "record_kind": "device",
        "detail": "place S01 does not exist",
    }
    fields.update(overrides)
    return FacilityBuildError(**fields)  # type: ignore[arg-type]


class TestLineGrammar:
    def test_record_line(self) -> None:
        assert _error().format_message() == (
            "facility: reference-missing: device QF1 — place S01 does not exist; "
            "fix: add place S01 or correct the device's place"
        )

    def test_pydantic_failure_names_the_dotted_path(self) -> None:
        err = _error(
            kind="source-invalid",
            record_kind="path",
            record_id="channels.QF1_SP.value_type",
            detail="input should be 'float', 'int', 'bool', 'enum', 'string' or 'waveform'",
            remedy="use one of the listed value types",
        )
        assert err.format_message() == (
            "facility: source-invalid: path channels.QF1_SP.value_type — input should be "
            "'float', 'int', 'bool', 'enum', 'string' or 'waveform'; "
            "fix: use one of the listed value types"
        )

    def test_message_is_one_line(self) -> None:
        assert "\n" not in _error().format_message()
        assert str(_error()) == _error().format_message()

    def test_fields_are_kept(self) -> None:
        err = _error(sources=["records/a.yaml", "imported/mml/b.yaml"])
        assert err.kind == "reference-missing"
        assert err.record_kind == "device"
        assert err.record_id == "QF1"
        assert err.sources == ("records/a.yaml", "imported/mml/b.yaml")
        assert err.remedy == "add place S01 or correct the device's place"
        assert err.detail == "place S01 does not exist"

    def test_it_is_a_plain_facility_error(self) -> None:
        assert isinstance(_error(), FacilityError)
        assert isinstance(_error(), FacilityDefectError)
        assert not isinstance(_error(), click.ClickException)
        assert not isinstance(_error(), ValueError)

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no-such-kind"):
            _error(kind="no-such-kind")


def _warning(**overrides: str) -> FacilityBuildWarning:
    fields = {
        "kind": "place-wrapped",
        "record_kind": "device",
        "record_id": "SR/SPARE",
        "detail": "layer authored states s -0.5 in periodic model SR, outside its deck of "
        "length 3.8; the device is placed at s 3.3",
        "remedy": "state an s within [0, 3.8], or drop it",
    }
    fields.update(overrides)
    return FacilityBuildWarning(**fields)


class TestWarningLine:
    def test_summary_has_the_stop_line_shape_without_the_remedy(self) -> None:
        assert _warning().summary == (
            "facility: place-wrapped: device SR/SPARE — layer authored states s -0.5 in "
            "periodic model SR, outside its deck of length 3.8; the device is placed at s 3.3"
        )

    def test_line_appends_the_remedy(self) -> None:
        assert (
            _warning().line == f"{_warning().summary}; fix: state an s within [0, 3.8], or drop it"
        )
        assert "\n" not in _warning().line

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no-such-kind"):
            _warning(kind="no-such-kind")

    def test_a_stop_kind_is_not_a_warning_kind(self) -> None:
        assert not set(WARNING_KINDS) & set(KINDS)
        with pytest.raises(ValueError, match="place-conflict"):
            _warning(kind="place-conflict")

    def test_a_dangling_link_is_a_warning_kind(self) -> None:
        warning = _warning(kind="link-dangling", record_kind="page", record_id="knowledge/gone.md")

        assert warning.kind == "link-dangling"


class TestKinds:
    def test_kinds_follow_the_two_word_grammar(self) -> None:
        for kind in KINDS:
            thing, _, problem = kind.partition("-")
            assert thing and problem and "-" not in problem, kind
            assert kind == kind.lower(), kind

    def test_kinds_are_unique(self) -> None:
        assert len(KINDS) == len(set(KINDS))

    def test_first_kinds_are_registered(self) -> None:
        assert set(KINDS) >= {
            "source-invalid",
            "layer-conflict",
            "layer-duplicate",
            "fix-missing",
            "fix-stale",
            "fix-duplicate",
            "fix-computed",
            "fix-authored",
            "fix-referenced",
            "reference-missing",
            "class-unknown",
            "pair-invalid",
            "value-invalid",
            "seed-invalid",
            "limit-invalid",
            "place-conflict",
            "span-invalid",
            "wiring-conflict",
            "engine-missing",
            "engine-invalid",
            "model-conflict",
        }


def _run(error: Exception) -> click.testing.Result:
    """Raise *error* from a command run as the command line runs it."""

    @click.command()
    def cmd() -> None:
        with facility_errors_as_click():
            raise error

    return CliRunner().invoke(cmd, [])


class TestShow:
    def test_show_writes_the_line_alone_to_stderr(self) -> None:
        result = _run(_error())
        assert result.exit_code == 1
        assert result.stderr == _error().format_message() + "\n"
        assert "Error: " not in result.stderr
        assert result.stdout == ""

    def test_show_honours_an_explicit_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        import io

        buffer = io.StringIO()
        _error().show(buffer)
        assert buffer.getvalue() == _error().format_message() + "\n"
        assert capsys.readouterr().err == ""

    def test_show_defaults_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        FacilityError("one\ntwo").show()
        captured = capsys.readouterr()
        assert captured.err == "one\ntwo\n"
        assert captured.out == ""


class TestFacilityErrorFamily:
    def test_the_base_is_a_plain_exception(self) -> None:
        error = FacilityError("broken")
        assert error.format_message() == "broken"
        assert str(error) == "broken"
        assert not isinstance(error, ValueError)
        assert not isinstance(error, click.ClickException)

    @pytest.mark.parametrize(
        "error",
        [
            LayerSourceInvalid([_error()]),
            ImportStop("mapping-draft", ["x"]),
            MappingProblems(Path("m.yaml"), ["p"]),
            FacilityInputError("cannot read"),
            FacilityUsageError("no such system"),
        ],
    )
    def test_every_error_derives_from_the_base(self, error: FacilityError) -> None:
        assert isinstance(error, FacilityError)
        assert not isinstance(error, click.ClickException)

    def test_layer_source_invalid_prints_each_line(self) -> None:
        second = _error(record_id="QF2")
        result = _run(LayerSourceInvalid([_error(), second]))
        assert result.exit_code == 1
        assert result.stderr == f"{_error().format_message()}\n{second.format_message()}\n"

    def test_import_stop_prints_one_line_per_finding(self) -> None:
        stop = ImportStop("export-invalid", ["a", "b"])
        assert stop.problem == "export-invalid"
        assert stop.lines == ("a", "b")
        result = _run(stop)
        assert result.exit_code == 1
        assert result.stderr == "import mml: export-invalid: a\nimport mml: export-invalid: b\n"

    def test_mapping_problems_count_their_problems(self) -> None:
        one = MappingProblems(Path("m.yaml"), ["p1"])
        two = MappingProblems(Path("m.yaml"), ["p1", "p2"])
        assert one.format_message() == "p1\n1 problem in m.yaml; fix each and check again."
        assert two.format_message().endswith("2 problems in m.yaml; fix each and check again.")
        assert two.problems == ("p1", "p2")

    def test_an_input_error_keeps_the_error_prefix(self) -> None:
        result = _run(FacilityInputError("cannot read x.mat"))
        assert result.exit_code == 1
        assert result.stderr == "Error: cannot read x.mat\n"

    def test_a_usage_error_left_unconverted_prints_bare(self) -> None:
        result = _run(FacilityUsageError("no such system"))
        assert result.exit_code == 1
        assert result.stderr == "no such system\n"

    def test_other_exceptions_pass_through(self) -> None:
        result = _run(RuntimeError("boom"))
        assert result.exit_code == 1
        assert isinstance(result.exception, RuntimeError)
        assert result.stderr == ""


class TestRootGroup:
    def test_the_root_group_encloses_its_subcommands(self, monkeypatch: pytest.MonkeyPatch) -> None:
        @click.command("boom")
        def boom() -> None:
            raise _error()

        monkeypatch.setattr(type(cli), "get_command", lambda self, ctx, name: boom)
        result = CliRunner().invoke(cli, ["boom"])
        assert result.exit_code == 1
        assert result.stderr == _error().format_message() + "\n"


class TestFold:
    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("my proj", "my_proj"),
            ("1st-lab", "x1st_lab"),
            ("als.u", "als_u"),
            ("already_ok", "already_ok"),
            ("_lead", "_lead"),
            ("", "x"),
            ("é", "_"),
        ],
    )
    def test_fold(self, name: str, code: str) -> None:
        assert fold_code(name) == code
        assert PN_LOCAL.fullmatch(fold_code(name))

    def test_pn_local(self) -> None:
        assert PN_LOCAL.pattern == r"[A-Za-z_][A-Za-z0-9_]*"
        assert PN_LOCAL.fullmatch("demo_1")
        assert not PN_LOCAL.fullmatch("1demo")
        assert not PN_LOCAL.fullmatch("de-mo")


class TestRelationKinds:
    def test_relation_invalid_is_a_stop_kind(self) -> None:
        assert "relation-invalid" in KINDS
        assert "relation-invalid" not in WARNING_KINDS

    def test_relation_invalid_prints_directly_after_reference_missing(self) -> None:
        from osprey.facility.validate import _ROW, _TABLE_ROWS

        row = _ROW["relation-invalid"]
        assert _TABLE_ROWS[row] == ("relation-invalid",)
        assert row == _ROW["reference-missing"] + 1
        assert _ROW["class-unknown"] == row + 1

    def test_the_six_build_warnings_are_registered(self) -> None:
        assert set(WARNING_KINDS) >= {
            "relation-contested",
            "relation-coarse",
            "alias-shared",
            "identity-ambiguous",
            "part-cycle",
            "relation-stub",
        }

    def test_the_place_warning_is_registered_with_its_emitter(self) -> None:
        assert "place-unplaced" in WARNING_KINDS
        assert "place-parents-disagree" not in WARNING_KINDS

    def test_warning_kinds_are_unique(self) -> None:
        assert len(WARNING_KINDS) == len(set(WARNING_KINDS))

    @pytest.mark.parametrize(
        "kind",
        [
            "relation-contested",
            "relation-coarse",
            "alias-shared",
            "identity-ambiguous",
            "part-cycle",
            "relation-stub",
        ],
    )
    def test_each_warning_kind_constructs(self, kind: str) -> None:
        assert _warning(kind=kind).kind == kind

    def test_relation_invalid_constructs_an_error(self) -> None:
        assert _error(kind="relation-invalid").kind == "relation-invalid"
