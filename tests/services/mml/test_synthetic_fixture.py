"""The committed synthetic MML fixture is exactly what its generator writes.

``tests/fixtures/mml/synthetic/`` is a generated 2.0 export: a small invented
ring, the virtual-accelerator facts sampled off it, and the orbit response
measured on it. Several suites read those bytes as ground truth, which only
holds while the bytes and the script agree. Hand-editing the export, or
changing the generator without rerunning it, breaks that silently -- every
test keeps passing against a fixture that no longer describes what the script
says it describes.

So the generator's own ``--check`` mode runs here: it rebuilds the fixture
into a temporary directory and compares every committed byte. Everything a
clock or a machine would otherwise decide is pinned in the script, so a
rebuild on another day on another machine writes the same bytes and a failure
here is a real disagreement rather than a timestamp.
"""

from __future__ import annotations

import math
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

GENERATOR = Path(__file__).resolve().parents[2] / "fixtures" / "mml" / "synthetic" / "build.py"

#: Every argument the fixture hands its curve lies inside this span.
CURVE_SPAN = np.linspace(-1.0, 1.0, 4001)


def _generator() -> dict[str, Any]:
    return runpy.run_path(str(GENERATOR))


def test_the_committed_fixture_regenerates_byte_for_byte() -> None:
    """Run the generator's determinism gate the way its docstring documents it."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "the committed synthetic fixture is not what the generator writes; "
        "rerun tests/fixtures/mml/synthetic/build.py rather than editing the "
        f"files by hand.\n{result.stdout}{result.stderr}"
    )


def test_the_generator_s_curve_is_sinh_to_its_last_digits() -> None:
    """The generator's series agrees with the library ``sinh`` to a few ulp."""
    sinh = _generator()["_sinh"]
    np.testing.assert_array_max_ulp(
        sinh(CURVE_SPAN), np.array([math.sinh(v) for v in CURVE_SPAN]), maxulp=4
    )


def test_the_generator_s_inverse_undoes_its_curve() -> None:
    """The generator's inverse is ``asinh`` to a few ulp and undoes its series."""
    generator = _generator()
    sinh, arcsinh = generator["_sinh"], generator["_arcsinh"]
    targets = np.sinh(CURVE_SPAN)
    np.testing.assert_array_max_ulp(
        arcsinh(targets), np.array([math.asinh(v) for v in targets]), maxulp=4
    )
    np.testing.assert_array_max_ulp(arcsinh(sinh(CURVE_SPAN)), CURVE_SPAN, maxulp=2)
