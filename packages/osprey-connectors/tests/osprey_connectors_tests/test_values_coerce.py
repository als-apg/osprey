"""Value coercion per channel ``value_type``: one case per accept/refuse rule."""

import math
import subprocess
import sys

import pytest

from osprey_connectors.simulation.values import coerce, zero

ENUM = ["OFF", "ON", "FAULT"]


# float --------------------------------------------------------------------


def test_float_accepts_int_and_float():
    assert coerce(3, "float") == 3.0
    assert isinstance(coerce(3, "float"), float)
    assert coerce(2.5, "float") == 2.5


def test_float_is_the_default_type():
    assert coerce(1, None) == 1.0


@pytest.mark.parametrize(
    ("bad", "value_type"),
    [
        # float: bool, non-finite and non-numbers
        pytest.param(True, "float", id="float-bool-true"),
        pytest.param(False, "float", id="float-bool-false"),
        pytest.param(math.nan, "float", id="float-nan"),
        pytest.param(math.inf, "float", id="float-inf"),
        pytest.param(-math.inf, "float", id="float-negative-inf"),
        pytest.param("1.0", "float", id="float-string"),
        pytest.param(None, "float", id="float-none"),
        pytest.param([1.0], "float", id="float-list"),
        # int: fractional, non-finite, bool and non-numbers
        pytest.param(4.5, "int", id="int-fractional"),
        pytest.param(math.nan, "int", id="int-nan"),
        pytest.param(math.inf, "int", id="int-inf"),
        pytest.param(True, "int", id="int-bool"),
        pytest.param("4", "int", id="int-string"),
        pytest.param(None, "int", id="int-none"),
        # bool: out-of-range index, float and unknown label
        pytest.param(2, "bool", id="bool-index-out-of-range"),
        pytest.param(-1, "bool", id="bool-negative-index"),
        pytest.param("maybe", "bool", id="bool-unknown-label"),
        pytest.param(0.5, "bool", id="bool-fractional-float"),
        pytest.param(1.0, "bool", id="bool-integral-float"),
        pytest.param(None, "bool", id="bool-none"),
        pytest.param([0], "bool", id="bool-list"),
        # string: anything that is not a str
        pytest.param(1, "string", id="string-int"),
        pytest.param(1.0, "string", id="string-float"),
        pytest.param(True, "string", id="string-bool"),
        pytest.param(None, "string", id="string-none"),
        pytest.param(b"x", "string", id="string-bytes"),
        pytest.param(["a"], "string", id="string-list"),
        # enum without options
        pytest.param(0, "enum", id="enum-without-options"),
        # unknown value type
        pytest.param(1.0, "complex", id="unknown-value-type"),
    ],
)
def test_a_value_the_type_cannot_hold_is_refused(bad, value_type):
    with pytest.raises(ValueError, match=value_type):
        coerce(bad, value_type)


# int ----------------------------------------------------------------------


def test_int_accepts_int_and_integral_float():
    assert coerce(4, "int") == 4
    got = coerce(4.0, "int")
    assert got == 4 and type(got) is int


# bool ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "label"),
    [(0, "FALSE"), (1, "TRUE"), (False, "FALSE"), (True, "TRUE"), ("TRUE", "TRUE")],
)
def test_bool_default_labels_store_the_label(value, label):
    assert coerce(value, "bool") == label


def test_bool_custom_labels():
    assert coerce(True, "bool", options=["Open", "Closed"]) == "Closed"
    assert coerce("Open", "bool", options=["Open", "Closed"]) == "Open"


# enum ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "label"),
    [(0, "OFF"), (2, "FAULT"), (True, "ON"), (False, "OFF"), ("FAULT", "FAULT")],
)
def test_enum_accepts_index_bool_or_label_and_stores_the_label(value, label):
    assert coerce(value, "enum", options=ENUM) == label


@pytest.mark.parametrize("bad", [3, -1, "on", 1.5, 2.0, None])
def test_enum_refuses_out_of_range_index_float_and_unknown_label(bad):
    with pytest.raises(ValueError, match="enum"):
        coerce(bad, "enum", options=ENUM)


# string -------------------------------------------------------------------


def test_string_accepts_str_only():
    assert coerce("hello", "string") == "hello"
    assert coerce("", "string") == ""


# waveform -----------------------------------------------------------------


def test_waveform_accepts_flat_numeric_list_of_the_shape_product():
    assert coerce([1, 2.5, 3], "waveform", shape=[3]) == [1.0, 2.5, 3.0]


def test_waveform_accepts_nested_list_with_matching_flattened_length():
    assert coerce([[1, 2, 3], [4, 5, 6]], "waveform", shape=[2, 3]) == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]
    assert coerce((1, 2), "waveform", shape=[2]) == [1.0, 2.0]


@pytest.mark.parametrize(
    "bad",
    [[1, 2], [1, 2, 3, 4], [1, "2", 3], [1, True, 3], [1, math.nan, 3], 1.0, "abc", None],
)
def test_waveform_refuses_wrong_length_or_non_numeric(bad):
    with pytest.raises(ValueError, match="waveform"):
        coerce(bad, "waveform", shape=[3])


def test_waveform_without_shape_is_refused():
    with pytest.raises(ValueError, match="waveform"):
        coerce([1.0], "waveform")


# numpy scalars -----------------------------------------------------------


def test_numpy_scalars_are_numbers():
    np = pytest.importorskip("numpy")
    got = coerce(np.int64(3), "float")
    assert got == 3.0 and type(got) is float
    got = coerce(np.float32(2.5), "float")
    assert got == 2.5 and type(got) is float
    got = coerce(np.int64(4), "int")
    assert got == 4 and type(got) is int
    got = coerce(np.float64(4.0), "int")
    assert got == 4 and type(got) is int
    assert coerce(np.int64(2), "enum", options=ENUM) == "FAULT"
    assert coerce([np.int64(1), np.float32(2.0)], "waveform", shape=[2]) == [1.0, 2.0]


def test_numpy_bool_and_non_finite_are_refused_as_numbers():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="float"):
        coerce(np.bool_(True), "float")
    with pytest.raises(ValueError, match="float"):
        coerce(np.float32("nan"), "float")
    with pytest.raises(ValueError, match="enum"):
        coerce(np.float64(2.0), "enum", options=ENUM)


# unknown type -------------------------------------------------------------


def test_refusal_names_the_type_and_the_value():
    with pytest.raises(ValueError) as info:
        coerce("abc", "float")
    assert "float" in str(info.value) and "'abc'" in str(info.value)


# zero ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value_type", "options", "shape", "expected"),
    [
        ("float", None, None, 0.0),
        ("int", None, None, 0),
        ("bool", None, None, "FALSE"),
        ("bool", ["Open", "Closed"], None, "Open"),
        ("enum", ENUM, None, "OFF"),
        ("string", None, None, ""),
        ("waveform", None, [2, 3], [0.0] * 6),
    ],
)
def test_zero_per_type(value_type, options, shape, expected):
    got = zero(value_type, options, shape)
    assert got == expected and type(got) is type(expected)


def test_zero_is_accepted_by_coerce():
    for value_type, options, shape in [
        ("float", None, None),
        ("int", None, None),
        ("bool", None, None),
        ("enum", ENUM, None),
        ("string", None, None),
        ("waveform", None, [4]),
    ]:
        z = zero(value_type, options, shape)
        assert coerce(z, value_type, options, shape) == z


def test_zero_refuses_unknown_type():
    with pytest.raises(ValueError, match="complex"):
        zero("complex", None, None)


# import isolation ---------------------------------------------------------


def test_values_module_imports_without_a_framework():
    code = (
        "import sys, osprey_connectors.simulation.values;"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('lume', 'lume_base', 'osprey'));"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
