"""Tests for the facility build's error line and identity-code folding."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from osprey.facility import PN_LOCAL, fold_code
from osprey.facility.errors import KINDS, WARNING_KINDS, FacilityBuildError, FacilityBuildWarning


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

    def test_exit_code_is_one(self) -> None:
        assert _error().exit_code == 1
        assert isinstance(_error(), click.ClickException)

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
        "remedy": "state s 3.3, or drop it",
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
        assert _warning().line == f"{_warning().summary}; fix: state s 3.3, or drop it"
        assert "\n" not in _warning().line

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no-such-kind"):
            _warning(kind="no-such-kind")

    def test_a_stop_kind_is_not_a_warning_kind(self) -> None:
        assert not set(WARNING_KINDS) & set(KINDS)
        with pytest.raises(ValueError, match="place-conflict"):
            _warning(kind="place-conflict")


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


class TestShow:
    def test_show_writes_the_line_alone_to_stderr(self) -> None:
        @click.command()
        def cmd() -> None:
            raise _error()

        result = CliRunner().invoke(cmd, [])
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
