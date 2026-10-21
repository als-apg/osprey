"""Tests for the pyaml-cs-osprey channel reference grammar."""

import dataclasses

import pytest
from pyaml.common.exception import PyAMLException

from pyaml_cs_osprey.catalog import ChannelReference, parse_reference


def test_bare_address_reads_and_writes_one_address():
    ref = parse_reference("TUNEZR:rdH[kHz]")
    assert ref.mode == "rw"
    assert ref.address == "TUNEZR:rdH"
    assert ref.readback is None
    assert ref.unit == "kHz"
    assert ref.index is None
    assert ref.text == "TUNEZR:rdH[kHz]"


def test_parenthesised_single_address_is_write_only():
    ref = parse_reference("(RF:freq:set)[Hz]")
    assert ref.mode == "w"
    assert ref.address == "RF:freq:set"
    assert ref.readback is None
    assert ref.unit == "Hz"
    assert ref.index is None


def test_pair_is_read_write_with_separate_addresses():
    ref = parse_reference("(QF_001:Cm:rdbk, QF_001:Cm:set)[1/m]")
    assert ref.mode == "rw"
    assert ref.address == "QF_001:Cm:set"
    assert ref.readback == "QF_001:Cm:rdbk"
    assert ref.unit == "1/m"
    assert ref.index is None


def test_unit_with_exponent_is_kept_verbatim():
    ref = parse_reference("(SF_001:Cm:rdbk, SF_001:Cm:set)[1/m**2]")
    assert ref.unit == "1/m**2"


def test_array_index_on_bare_reference():
    ref = parse_reference("beam:orbit:x@3[m]")
    assert ref.mode == "rw"
    assert ref.readback is None
    assert ref.address == "beam:orbit:x"
    assert ref.index == 3
    assert ref.unit == "m"


def test_array_index_on_write_only_reference():
    ref = parse_reference("(wave:set)@0[A]")
    assert ref.mode == "w"
    assert ref.address == "wave:set"
    assert ref.index == 0


def test_array_index_on_read_write_reference():
    ref = parse_reference("(wave:rdbk, wave:set)@15[A]")
    assert ref.mode == "rw"
    assert ref.readback == "wave:rdbk"
    assert ref.address == "wave:set"
    assert ref.index == 15
    assert ref.unit == "A"


def test_array_index_without_unit():
    ref = parse_reference("beam:orbit:y@27")
    assert ref.index == 27
    assert ref.unit == ""


@pytest.mark.parametrize(
    ("text", "mode"),
    [("master_clock:freq", "rw"), ("(RF:set)", "w"), ("(RF:rdbk, RF:set)", "rw")],
)
def test_missing_unit_yields_empty_string(text, mode):
    ref = parse_reference(text)
    assert ref.unit == ""
    assert ref.mode == mode


def test_empty_unit_brackets_yield_empty_string():
    assert parse_reference("TUNE:x[]").unit == ""


@pytest.mark.parametrize(
    "text",
    [
        "  TUNEZR:rdH[kHz]  ",
        "TUNEZR:rdH [kHz]",
        "TUNEZR:rdH[ kHz ]",
    ],
)
def test_whitespace_around_parts_is_ignored(text):
    ref = parse_reference(text)
    assert ref.address == "TUNEZR:rdH"
    assert ref.unit == "kHz"
    assert ref.mode == "rw"


def test_whitespace_inside_pair_is_ignored():
    ref = parse_reference(" ( QF:rdbk ,QF:set ) @ 2 [ 1/m ] ")
    assert ref.readback == "QF:rdbk"
    assert ref.address == "QF:set"
    assert ref.index == 2
    assert ref.unit == "1/m"
    assert ref.mode == "rw"


def test_text_is_stripped_original():
    assert parse_reference("  (A, B)[m] ").text == "(A, B)[m]"


def test_reference_is_frozen():
    ref = parse_reference("A[m]")
    assert isinstance(ref, ChannelReference)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.address = "B"  # type: ignore[misc]


def test_equal_texts_give_equal_references():
    assert parse_reference("(A, B)[m]") == parse_reference("(A,B)[m]")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "[m]",
        "()[m]",
        "(A, B, C)[m]",
        "(A, )[m]",
        "(, B)[m]",
        "(A[m]",
        "A)[m]",
        "A[m",
        "A m]",
        "A[m]extra",
        "A@[m]",
        "A@x[m]",
        "A@-1[m]",
        "A B[m]",
        "A[m][n]",
        "(A)(B)[m]",
    ],
)
def test_malformed_reference_raises_pyaml_exception_naming_text(text):
    with pytest.raises(PyAMLException) as info:
        parse_reference(text)
    assert repr(text) in str(info.value)


def test_non_string_reference_raises_pyaml_exception():
    with pytest.raises(PyAMLException):
        parse_reference(3)  # type: ignore[arg-type]
