"""Tests for :mod:`pyaml_cs_osprey.units`."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

import pytest

from pyaml_cs_osprey import units
from pyaml_cs_osprey.units import UnitError


@dataclass(frozen=True)
class _Ref:
    unit: str


# spelling -> (factor, SI word)
_EXPECTED = {
    "ampere": (1.0, "A"),
    "a": (1.0, "A"),
    "1/m": (1.0, "1/m"),
    "1/m**2": (1.0, "1/m**2"),
    "rad": (1.0, "rad"),
    "m": (1.0, "m"),
    "hz": (1.0, "Hz"),
    "mm": (1e-3, "m"),
    "um": (1e-6, "m"),
    "urad": (1e-6, "rad"),
    "mrad": (1e-3, "rad"),
    "khz": (1e3, "Hz"),
    "mhz": (1e6, "Hz"),
}


@pytest.mark.parametrize("spelling", sorted(_EXPECTED))
def test_factor_table_covers_tree_spellings(spelling: str) -> None:
    want_factor, want_word = _EXPECTED[spelling]
    for variant in (spelling, spelling.upper(), spelling.capitalize(), f" {spelling} "):
        assert units.factor(variant) == pytest.approx(want_factor, rel=1e-15)
        assert units.si_unit(variant) == want_word


@pytest.mark.parametrize("spelling", ["Ampere", "A", "MHz", "kHz", "Hz", "mm", "urad"])
def test_tree_case_spellings(spelling: str) -> None:
    assert units.factor(spelling) == units.factor(spelling.casefold())


def test_no_suffix_is_identity() -> None:
    assert units.factor("") == 1.0
    assert units.si_unit("") == ""
    assert units.to_si(0.27, _Ref("")) == 0.27
    assert units.to_native(0.27, _Ref("")) == 0.27


def test_mhz_max_step_si() -> None:
    ref_mhz = _Ref("MHz")
    assert units.max_step_si(0.01, ref_mhz) == pytest.approx(1e4, rel=1e-12)
    assert units.si_unit(ref_mhz.unit) == "Hz"


def test_max_step_none_stays_none() -> None:
    assert units.max_step_si(None, _Ref("MHz")) is None


def test_to_si_and_back() -> None:
    ref = _Ref("mm")
    assert units.to_si(2.5, ref) == pytest.approx(2.5e-3)
    assert units.to_native(2.5e-3, ref) == pytest.approx(2.5)
    for suffix in _EXPECTED:
        r = _Ref(suffix)
        assert units.to_native(units.to_si(1.234, r), r) == pytest.approx(1.234, rel=1e-14)


def test_reference_may_be_a_suffix_string() -> None:
    assert units.to_si(476.3, "MHz") == pytest.approx(476.3e6)
    assert units.max_step_si(0.5, "kHz") == pytest.approx(500.0)


@pytest.mark.parametrize("suffix", ["furlong", "Amps", "T*m^-1", "mA"])
def test_unknown_suffix_refused_by_name(suffix: str) -> None:
    for call in (
        lambda: units.factor(suffix),
        lambda: units.si_unit(suffix),
        lambda: units.to_si(1.0, _Ref(suffix)),
        lambda: units.to_native(1.0, _Ref(suffix)),
        lambda: units.max_step_si(1.0, _Ref(suffix)),
    ):
        with pytest.raises(UnitError, match=suffix.replace("*", r"\*").replace("^", r"\^")) as err:
            call()
        assert err.value.suffix == suffix
        assert isinstance(err.value, ValueError)


def test_module_imports_nothing_from_osprey_or_pyaml() -> None:
    tree = ast.parse(inspect.getsource(units))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & {"osprey", "pyaml", "pyaml_cs_osprey"}
