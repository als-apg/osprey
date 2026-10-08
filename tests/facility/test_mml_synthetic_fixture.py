"""The committed synthetic MML fixture is exactly what its generator writes.

``tests/fixtures/mml/synthetic/`` is a generated 2.0 export: a small invented
ring, the virtual-accelerator facts sampled off it, and the orbit response
measured on it. Several suites read those bytes as ground truth, which only
holds while the bytes and the script agree. Hand-editing the export, or
changing the generator without rerunning it, breaks that silently -- every
test keeps passing against a fixture that no longer describes what the script
says it describes.

So the generator's own ``--check`` mode runs here: it rebuilds the fixture
into a temporary directory and compares every committed byte, save the model
file's tracking-derived numbers, which it holds to a relative tolerance because
tracking runs through the platform's own libm and BLAS. Everything else a clock
or a machine would otherwise decide is pinned in the script, so a failure here
is a real disagreement rather than a timestamp or a platform's last digits.
"""

from __future__ import annotations

import json
import math
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

GENERATOR = Path(__file__).resolve().parents[1] / "fixtures" / "mml" / "synthetic" / "build.py"

#: Every argument the fixture hands its curve lies inside this span.
CURVE_SPAN = np.linspace(-1.0, 1.0, 4001)


def _generator() -> dict[str, Any]:
    return runpy.run_path(str(GENERATOR))


def test_the_committed_fixture_regenerates_byte_for_byte() -> None:
    """Run the generator's determinism gate the way its docstring documents it.

    Every committed byte must come back, save the model file's tracking-derived
    numbers, which must agree within the generator's ``TRACKED_RTOL``.
    """
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


#: The model file's first chromaticity as Linux tracking writes it, where macOS
#: writes the committed value: the widest cross-platform spread in the file.
LINUX_CHROMATICITY = -0.806096917623


def _rebuilt_model(tmp_path: Path, generator: dict[str, Any], chromaticity: float) -> Path:
    """The committed model file rewritten with its first chromaticity replaced."""
    committed = GENERATOR.parent / f"{generator['STEM']}.model.json"
    body = json.loads(committed.read_text(encoding="utf-8"))
    del body["_export"]
    body["chromaticity"]["physics"][0] = chromaticity
    rebuilt = tmp_path / committed.name
    rebuilt.write_text(generator["document"](body) + "\n", encoding="utf-8")
    return rebuilt


def test_the_model_file_comparison_holds_across_platforms(tmp_path: Path) -> None:
    """A rebuilt model file whose tracking differs as Linux's does from macOS's is the same file."""
    generator = _generator()
    committed = GENERATOR.parent / f"{generator['STEM']}.model.json"
    rebuilt = _rebuilt_model(tmp_path, generator, LINUX_CHROMATICITY)
    assert generator["same_file"](committed, rebuilt)


def test_the_model_file_comparison_refuses_a_real_change(tmp_path: Path) -> None:
    """A chromaticity that moved by a part in ten thousand is a different file."""
    generator = _generator()
    committed = GENERATOR.parent / f"{generator['STEM']}.model.json"
    rebuilt = _rebuilt_model(tmp_path, generator, LINUX_CHROMATICITY * (1.0 + 1.0e-4))
    assert not generator["same_file"](committed, rebuilt)


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


def test_the_generator_anchors_a_stepped_setpoint_grid_the_way_the_exporter_does() -> None:
    """Each anchor lands as a point; one that cannot takes the row's widest gap."""
    anchored_grid = _generator()["_anchored_grid"]
    grid = np.array([[0.0, 1.0, 2.0, 4.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    anchors = np.array([[0.5, 2.5], [1.0, 9.0], [np.nan, np.nan]])

    rows = anchored_grid(grid, anchors)

    np.testing.assert_array_equal(rows[0], [0.0, 0.5, 1.0, 2.0, 2.5, 4.0])
    np.testing.assert_array_equal(rows[1], [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    np.testing.assert_array_equal(rows[2], [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    assert np.all(np.diff(rows, axis=1) > 0)


def test_the_generator_keeps_a_grid_with_no_step_uniform() -> None:
    """A family the Middle Layer states no DeltaRespMat for keeps every point it had."""
    anchored_grid = _generator()["_anchored_grid"]
    grid = np.linspace(-1.0, 1.0, 33)[None, :]

    np.testing.assert_array_equal(anchored_grid(grid, np.full((1, 2), np.nan)), grid)
