"""Tests for the MML family-body normaliser.

Each rule of the normaliser has one case here, plus a mixed family. The
normaliser is what makes the ``.mat`` lane and the JSON lane produce the same
``ao.json``, so every spelling difference between the two lanes is pinned.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.io.matlab import MatlabFunction

from osprey.services.mml.loaders import LoadedInput
from osprey.services.mml.normalize import normalize_family


class TestChannelKeysAndFamilyArrays:
    """Bare strings and blank slots under channel keys and family arrays."""

    def test_bare_channel_name_becomes_one_slot_list(self):
        """A non-empty bare string under ChannelNames becomes a one-slot list."""
        assert normalize_family({"Monitor": {"ChannelNames": "PV"}}) == {
            "Monitor": {"ChannelNames": ["PV"]}
        }

    def test_tango_names_follow_the_same_rule(self):
        """TangoNames is a channel key like ChannelNames."""
        assert normalize_family({"Monitor": {"TangoNames": "a/b/c"}}) == {
            "Monitor": {"TangoNames": ["a/b/c"]}
        }

    def test_empty_bare_channel_string_becomes_empty_list(self):
        """An empty bare string is the JSON exporter's spelling of an empty char."""
        assert normalize_family({"Monitor": {"ChannelNames": ""}}) == {
            "Monitor": {"ChannelNames": []}
        }

    def test_whitespace_bare_channel_string_becomes_empty_list(self):
        """A whitespace-only bare string is also an empty char."""
        assert normalize_family({"Monitor": {"ChannelNames": "   "}}) == {
            "Monitor": {"ChannelNames": []}
        }

    def test_blank_slots_become_none_with_index_preserved(self):
        """Empty and whitespace slots inside a list become None in place."""
        body = {"Monitor": {"ChannelNames": ["A", "", "B", "   "]}}
        assert normalize_family(body) == {"Monitor": {"ChannelNames": ["A", None, "B", None]}}

    def test_char_matrix_rows_are_deblanked(self):
        """Padded char-matrix rows lose their trailing blanks, row by row."""
        body = {"Monitor": {"ChannelNames": ["SR01:BPM1  ", "SR01:BPM10 "]}}
        assert normalize_family(body) == {"Monitor": {"ChannelNames": ["SR01:BPM1", "SR01:BPM10"]}}

    @pytest.mark.parametrize(
        "key",
        [
            "DeviceList",
            "CommonNames",
            "ElementList",
            "Position",
            "Status",
            "DeviceType",
            "MemberOf",
        ],
    )
    def test_family_array_bare_string_becomes_list(self, key):
        """Every family array wraps a bare string into a one-slot list."""
        assert normalize_family({key: "x"}) == {key: ["x"]}

    def test_family_arrays_in_setup_are_scoped_too(self):
        """Family arrays under setup follow the same rule."""
        body = {"setup": {"CommonNames": "BPM1", "MemberOf": ["BPM", ""]}}
        assert normalize_family(body) == {
            "setup": {"CommonNames": ["BPM1"], "MemberOf": ["BPM", None]}
        }

    def test_n_row_list_stays_n_row(self):
        """An N-row DeviceList keeps its row structure."""
        body = {"DeviceList": [[1, 1], [1, 2], [2, 1]]}
        assert normalize_family(body) == {"DeviceList": [[1, 1], [1, 2], [2, 1]]}

    def test_one_row_list_stays_one_row(self):
        """A 1-row DeviceList is not flattened."""
        assert normalize_family({"DeviceList": [[1, 1]]}) == {"DeviceList": [[1, 1]]}

    def test_numeric_scalar_array_is_not_wrapped(self):
        """Only strings are wrapped; a numeric scalar keeps its shape."""
        assert normalize_family({"ElementList": 3}) == {"ElementList": 3}


class TestOtherStrings:
    """Strings outside the scoped keys pass through unchanged."""

    def test_units_string_passes_through(self):
        """Units is a TEXT scalar and stays a string."""
        assert normalize_family({"Monitor": {"Units": "Hardware"}}) == {
            "Monitor": {"Units": "Hardware"}
        }

    def test_empty_scalar_stays_empty_string(self):
        """An empty DataType stays '' rather than becoming a list."""
        assert normalize_family({"Monitor": {"DataType": ""}}) == {"Monitor": {"DataType": ""}}

    def test_description_with_trailing_blank_is_untouched(self):
        """A scalar string outside the scoped keys is not deblanked."""
        assert normalize_family({"Description": "Beam position "}) == {
            "Description": "Beam position "
        }

    def test_char_matrix_rows_outside_scoped_keys_are_deblanked(self):
        """Char-matrix rows are deblanked under any key, but blank rows stay strings."""
        assert normalize_family({"Description": ["line one  ", "   "]}) == {
            "Description": ["line one", ""]
        }

    def test_unknown_keys_pass_through_verbatim(self):
        """Keys the normaliser does not know keep their values."""
        body = {"WeirdKey": {"nested": [1, "two", {"x": None}]}, "Mode": "Simulator"}
        assert normalize_family(body) == body


class TestScalars:
    """MATLAB logicals and non-finite numbers."""

    def test_bool_becomes_int(self):
        """A MATLAB logical becomes an int."""
        assert normalize_family({"Monitor": {"Special": True, "Other": False}}) == {
            "Monitor": {"Special": 1, "Other": 0}
        }

    def test_numpy_bool_becomes_int(self):
        """A numpy logical becomes an int."""
        out = normalize_family({"Flag": np.bool_(True)})
        assert out == {"Flag": 1}
        assert type(out["Flag"]) is int

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(math.inf, "Inf"), (-math.inf, "-Inf"), (math.nan, "NaN")],
    )
    def test_python_non_finite_floats(self, value, expected):
        """Python non-finite floats become their canonical strings."""
        assert normalize_family({"Range": [value]}) == {"Range": [expected]}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(np.float64("inf"), "Inf"), (np.float32("-inf"), "-Inf"), (np.float64("nan"), "NaN")],
    )
    def test_numpy_non_finite_floats(self, value, expected):
        """numpy non-finite floats become their canonical strings."""
        assert normalize_family({"Tol": value}) == {"Tol": expected}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("inf", "Inf"), ("-inf", "-Inf"), ("NaN", "NaN"), ("Infinity", "Inf")],
    )
    def test_non_finite_string_spellings(self, value, expected):
        """Every quoted spelling of a non-finite value is canonicalised."""
        assert normalize_family({"Monitor": {"Range": [value]}}) == {
            "Monitor": {"Range": [expected]}
        }

    @pytest.mark.parametrize(
        ("token", "expected"),
        [("-Infinity", "-Inf"), ("+inf", "Inf"), ("nan", "NaN")],
    )
    def test_json_parse_constant_tokens(self, token, expected):
        """The bare JSON tokens parse_constant hands over are canonicalised."""
        assert normalize_family({"Range": token}) == {"Range": expected}

    def test_non_matching_string_is_not_a_number(self):
        """A string that merely contains inf is left alone."""
        assert normalize_family({"Description": "info"}) == {"Description": "info"}

    def test_finite_numbers_pass_through(self):
        """Finite numbers keep their value."""
        assert normalize_family({"Range": [-10.5, 10]}) == {"Range": [-10.5, 10]}

    def test_integral_floats_in_index_arrays_become_ints(self):
        """A MATLAB double index is written as the JSON lane writes it: an ``int``."""
        body = {
            "DeviceList": np.array([[1.0, 1.0], [1.0, 2.0]]),
            "ElementList": [1.0, 2.0],
            "Status": [1.0, 0.0],
            "setup": {"DeviceList": [[2.0, 1.0]]},
        }
        out = normalize_family(body)
        assert out["DeviceList"] == [[1, 1], [1, 2]]
        assert out["ElementList"] == [1, 2]
        assert out["Status"] == [1, 0]
        assert out["setup"]["DeviceList"] == [[2, 1]]
        assert all(type(v) is int for row in out["DeviceList"] for v in row)

    def test_non_integral_and_other_floats_keep_their_type(self):
        """Only integral values under index or status arrays change; every other float stays."""
        out = normalize_family({"ElementList": [1.5, 2.0], "Position": [1.0, 2.0], "Gain": 2.0})
        assert out["ElementList"] == [1.5, 2]
        assert type(out["ElementList"][0]) is float
        assert out["Position"] == [1.0, 2.0] and type(out["Position"][0]) is float
        assert type(out["Gain"]) is float


class TestFunctionHandles:
    """Function handles in every shape they arrive in."""

    def test_function_handle_dict_under_typo_key(self):
        """A dict carrying function_handle is a handle whatever its key."""
        body = {
            "SetpointGolden": {
                "HW2PhysicSDcn": {
                    "function_handle": {
                        "function": "hw2at",
                        "type": "simple",
                        "file": "/mml/hw2at.m",
                    },
                    "matlabroot": "/opt/matlab",
                    "sentinel": "@",
                    "separator": "/",
                }
            }
        }
        assert normalize_family(body) == {
            "SetpointGolden": {"HW2PhysicSDcn": {"$fn": "hw2at", "file": "/mml/hw2at.m"}}
        }

    def test_function_handle_as_string(self):
        """A string function_handle is the function name."""
        body = {"Monitor": {"HW2PhysicsFcn": {"function_handle": "amp2k"}}}
        assert normalize_family(body) == {
            "Monitor": {"HW2PhysicsFcn": {"$fn": "amp2k", "file": None}}
        }

    def test_dollar_fn_dict_is_canonicalised(self):
        """A dict already carrying $fn is reduced to the canonical pair."""
        body = {"Anything": {"$fn": "k2amp", "file": "/x/k2amp.m", "extra": 1}}
        assert normalize_family(body) == {"Anything": {"$fn": "k2amp", "file": "/x/k2amp.m"}}

    def test_dollar_fn_dict_without_file(self):
        """A $fn dict with no file gets file None."""
        assert normalize_family({"Anything": {"$fn": "k2amp"}}) == {
            "Anything": {"$fn": "k2amp", "file": None}
        }

    def test_bare_string_under_fcn_key(self):
        """A bare string under a *Fcn key is the handle's name."""
        assert normalize_family({"Setpoint": {"Physics2HWFcn": "k2amp"}}) == {
            "Setpoint": {"Physics2HWFcn": {"$fn": "k2amp", "file": None}}
        }

    def test_integer_one_under_fcn_key(self):
        """The integer 1 under a *Fcn key is an anonymous handle."""
        assert normalize_family({"Setpoint": {"HW2PhysicsFcn": 1}}) == {
            "Setpoint": {"HW2PhysicsFcn": {"$fn": None, "file": None}}
        }

    def test_bare_string_under_other_key_is_not_a_handle(self):
        """The key-suffix rule applies only to keys ending in Fcn."""
        assert normalize_family({"Setpoint": {"Units": "k2amp", "Count": 1}}) == {
            "Setpoint": {"Units": "k2amp", "Count": 1}
        }

    def test_matlab_function_value(self):
        """A scipy MatlabFunction is decoded by its value shape."""
        inner = np.empty((1, 1), dtype=[("function", object), ("type", object), ("file", object)])
        inner[0, 0] = (np.array(["amp2k"]), np.array(["simple"]), np.array(["/mml/amp2k.m"]))
        outer = np.empty(
            (1, 1),
            dtype=[
                ("matlabroot", object),
                ("separator", object),
                ("sentinel", object),
                ("function_handle", object),
            ],
        )
        outer[0, 0] = (np.array(["/opt/matlab"]), np.array(["/"]), np.array(["@"]), inner)
        body = {"Monitor": {"HW2PhysicsFcn": MatlabFunction(outer)}}
        assert normalize_family(body) == {
            "Monitor": {"HW2PhysicsFcn": {"$fn": "amp2k", "file": "/mml/amp2k.m"}}
        }

    def test_handles_key_is_dropped(self):
        """Handles (graphics handles) never reach the output."""
        assert normalize_family({"Monitor": {"Handles": [1.0, 2.0], "Units": "Hardware"}}) == {
            "Monitor": {"Units": "Hardware"}
        }


class TestPurity:
    """The normaliser is a pure function."""

    def test_input_is_not_mutated(self):
        """Normalising leaves the caller's body untouched."""
        body = {"Monitor": {"ChannelNames": "PV", "Handles": 1, "Status": True}}
        before = copy.deepcopy(body)
        normalize_family(body)
        assert body == before


class TestMixedFamily:
    """A family carrying many rules at once."""

    def test_the_documented_example(self):
        """The example from the task description holds."""
        out = normalize_family(
            {"Monitor": {"ChannelNames": "PV", "Units": "Amps", "DataType": "", "Status": True}}
        )
        assert out == {
            "Monitor": {"ChannelNames": ["PV"], "Units": "Amps", "DataType": "", "Status": 1}
        }
        assert type(out["Monitor"]["Status"]) is int

    def test_mixed_family(self):
        """Channel keys, family arrays, handles and non-finite values together."""
        body = {
            "FamilyName": "HCM",
            "MemberOf": ["HCM", "Magnet", "  "],
            "DeviceList": [[1, 1], [1, 2]],
            "Status": [True, False],
            "CommonNames": ["HCM1 ", "HCM2 "],
            "Monitor": {
                "Mode": "Simulator",
                "ChannelNames": ["SR:HCM1:AM ", "   "],
                "Units": "Hardware",
                "HW2PhysicsFcn": "amp2k",
                "Range": ["-inf", "inf"],
                "Handles": [0.0],
            },
            "Setpoint": {
                "TangoNames": "",
                "Physics2HWFcn": 1,
                "Tol": math.nan,
            },
            "Unknown": {"keep": "me "},
        }
        assert normalize_family(body) == {
            "FamilyName": "HCM",
            "MemberOf": ["HCM", "Magnet", None],
            "DeviceList": [[1, 1], [1, 2]],
            "Status": [1, 0],
            "CommonNames": ["HCM1", "HCM2"],
            "Monitor": {
                "Mode": "Simulator",
                "ChannelNames": ["SR:HCM1:AM", None],
                "Units": "Hardware",
                "HW2PhysicsFcn": {"$fn": "amp2k", "file": None},
                "Range": ["-Inf", "Inf"],
            },
            "Setpoint": {
                "TangoNames": [],
                "Physics2HWFcn": {"$fn": None, "file": None},
                "Tol": "NaN",
            },
            "Unknown": {"keep": "me "},
        }


class TestLoadedInput:
    """The value object both loaders return."""

    def test_fields(self, tmp_path: Path):
        """LoadedInput carries ao, ad, export, system_keyed and source."""
        li = LoadedInput(
            ao={"BPM": {}}, ad=None, export=None, system_keyed=False, source=tmp_path / "x.mat"
        )
        assert li.ao == {"BPM": {}}
        assert li.ad is None
        assert li.export is None
        assert li.system_keyed is False
        assert li.source == tmp_path / "x.mat"
