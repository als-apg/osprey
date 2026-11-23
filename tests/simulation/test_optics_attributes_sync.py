"""The description's copy of pyAT's optics attributes equals the engine's own.

``osprey.facility.validate`` decides whether a single-pass ``pyat`` model
reads tunes or chromaticity without importing the lattice engine, so it holds
the attribute names as a private literal. This test pins that literal to
:data:`osprey.simulation.engines.pyat.OPTICS_ATTRIBUTES`; the engine never
imports ``validate``.
"""

from __future__ import annotations

from pathlib import Path

from osprey.facility import validate
from osprey.simulation.engines import pyat


def test_validate_holds_the_engines_optics_attributes() -> None:
    assert validate._PYAT_OPTICS_ATTRIBUTES == pyat.OPTICS_ATTRIBUTES


def test_the_engine_does_not_import_validate() -> None:
    assert "osprey.facility.validate" not in Path(pyat.__file__).read_text(encoding="utf-8")
