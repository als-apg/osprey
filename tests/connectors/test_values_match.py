"""Tests for `values_match`, the one comparison rule behind a confirmed write.

A write is confirmed when the channel holds the value that was sent — exactly,
with no configurable tolerance. Every row below is a shape a control system
actually reads back (boxed scalars, float32 stores, enum indices, waveforms),
and the last test asserts the whole matrix is symmetric: swapping "sent" and
"observed" can never change the answer, so no caller can get a different verdict
by holding the pair the other way round.
"""

import inspect

import numpy as np
import pytest

from osprey.connectors.control_system.base import values_match


class _Unmentionable:
    """An object whose equality raises — rule 7's "cannot be compared"."""

    def __eq__(self, other):
        raise TypeError("these do not compare")

    __hash__ = None


# (id, sent, observed, enum_label, expected)
MATRIX = [
    # --- plain scalars -----------------------------------------------------
    ("zero_equals_zero", 0.0, 0.0, None, True),
    ("zero_vs_tiny", 0.0, 5e-10, None, False),
    ("relative_difference_above_tolerance", 1.0, 1.00001, None, False),
    ("relative_difference_below_tolerance", 1.0, 1.0000001, None, True),
    ("int_and_float_of_the_same_number", 5, 5.0, None, True),
    ("different_numbers", 5.0, 6.0, None, False),
    # --- float32 stores ----------------------------------------------------
    ("float32_scalar_round_trip", 1e6 / 3, np.float32(1e6 / 3), None, True),
    ("float32_array_round_trip", [0.1, 0.2], np.array([0.1, 0.2], dtype=np.float32), None, True),
    ("float32_array_that_differs", [0.1, 0.2], np.array([0.1, 0.3], dtype=np.float32), None, False),
    # --- length-1 unwrapping, both ways ------------------------------------
    ("boxed_sent_vs_scalar", [5.0], 5.0, None, True),
    ("scalar_vs_boxed_observed", 5.0, np.array([5.0]), None, True),
    ("boxed_both_sides", [5.0], np.array([5.0]), None, True),
    ("boxed_values_that_differ", np.array([0.1]), np.array([0.2]), None, False),
    # --- a scalar is not a vector (rule 3) ---------------------------------
    ("vector_vs_scalar", np.array([5.0, 5.0]), 5.0, None, False),
    ("scalar_vs_vector", 5.0, np.array([5.0, 5.0]), None, False),
    # --- sequences ---------------------------------------------------------
    ("equal_waveforms", [1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0]), None, True),
    ("tuple_matches_list", (1.0, 2.0), [1.0, 2.0], None, True),
    ("unequal_lengths", [1.0, 2.0], [1.0, 2.0, 3.0], None, False),
    ("empty_sequences", [], np.array([]), None, True),
    # --- 0-d arrays are scalars -------------------------------------------
    ("zero_d_array_vs_scalar", np.array(5.0), 5.0, None, True),
    ("zero_d_array_vs_boxed", np.array(5.0), np.array([5.0]), None, True),
    ("zero_d_array_vs_vector", np.array(5.0), np.array([5.0, 5.0]), None, False),
    # --- enum channels -----------------------------------------------------
    ("enum_label_matches_index", "Open", 1, "Open", True),
    ("enum_label_differs", "Closed", 1, "Open", False),
    ("no_enum_label_reported", "Open", 1, None, False),
    ("enum_index_written_directly", 1, 1, "Open", True),
    # --- text and non-numerics --------------------------------------------
    ("equal_strings", "Open", "Open", None, True),
    ("different_strings", "Open", "Closed", None, False),
    ("string_vs_int_array", "Open", np.array([1, 2]), None, False),
    ("string_vs_number", "5.0", 5.0, None, False),
    ("equal_bools", True, True, None, True),
    ("different_bools", True, False, None, False),
    # --- mappings and sets are not vectors ---------------------------------
    # A mapping has a length, but iterating one yields its KEYS: compared as a
    # vector, two dicts with the same keys and different values would zip to a
    # match and report a write as confirmed. They compare by `==` instead.
    ("dicts_with_equal_keys_and_different_values", {"a": 1, "b": 2}, {"a": 3, "b": 4}, None, False),
    ("equal_dicts", {"a": 1}, {"a": 1}, None, True),
    ("dict_vs_sequence", {"a": 1, "b": 2}, [1, 2], None, False),
    ("sets_of_equal_length_that_differ", {1, 2}, {3, 4}, None, False),
    ("equal_sets", {1, 2}, {1, 2}, None, True),
    # --- incomparable ------------------------------------------------------
    ("distinct_objects", object(), object(), None, False),
    ("equality_raises", _Unmentionable(), 5.0, None, False),
]

_IDS = [row[0] for row in MATRIX]
_CASES = [row[1:] for row in MATRIX]


@pytest.mark.parametrize("sent,observed,enum_label,expected", _CASES, ids=_IDS)
def test_values_match_matrix(sent, observed, enum_label, expected):
    assert values_match(sent, observed, enum_label=enum_label) is expected


@pytest.mark.parametrize("sent,observed,enum_label,expected", _CASES, ids=_IDS)
def test_values_match_is_symmetric(sent, observed, enum_label, expected):
    """Swapping the arguments cannot change the verdict.

    The comparison is between two readings of the same channel, so which one the
    caller calls "sent" is bookkeeping, not meaning.
    """
    assert values_match(sent, observed, enum_label=enum_label) == values_match(
        observed, sent, enum_label=enum_label
    )


class TestNoConfigurableTolerance:
    def test_signature_takes_only_the_two_values_and_an_enum_label(self):
        # A tolerance parameter is the thing this contract exists to remove:
        # a setpoint the machine clamped must be reported, never tolerated.
        assert list(inspect.signature(values_match).parameters) == [
            "sent",
            "observed",
            "enum_label",
        ]

    def test_absolute_tolerance_is_zero(self):
        # rel_tol only: nothing is close to zero except zero itself.
        assert values_match(0.0, 1e-300) is False

    def test_relative_tolerance_admits_a_float32_store_at_scale(self):
        # ~8x float32 epsilon, so a value stored as float32 and read back as a
        # double still confirms.
        assert values_match(1234.5678, float(np.float32(1234.5678))) is True
