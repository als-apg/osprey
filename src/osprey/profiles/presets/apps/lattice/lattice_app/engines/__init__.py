"""The engines the application carries, and the few lines every engine starts from.

An engine is a command the backend runs as ``<command> <job.json> <out.json>``. The job
file names the analysis, the deck, the analysis's settings, the model's
``engine_settings`` and the energy; the engine writes the result document
(:mod:`lattice_app.catalog`) to the output file and exits 0, or says why not on stderr
and exits non-zero. A facility's own engine needs nothing from this package: any
executable that keeps that contract will do.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

Compute = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def run_engine(analyses: Mapping[str, Compute], argv: list[str] | None = None) -> int:
    """An engine's ``main``: read the job, run its analysis, write the result.

    Args:
        analyses: The analyses this engine computes, each a function from the job to the
            result document.
        argv: ``[job.json, out.json]``; the process's arguments when omitted.

    Returns:
        The process's exit status.
    """
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        sys.stderr.write("usage: <engine> <job.json> <out.json>\n")
        return 2
    job = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    compute = analyses.get(job.get("analysis"))
    if compute is None:
        sys.stderr.write(f"this engine does not compute {job.get('analysis')}\n")
        return 2
    try:
        result = compute(job)
    except Exception as exc:  # whatever the physics code raised is the failure's reason
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    # NaN is written as NaN here; the backend turns it into null when it reads the result.
    Path(arguments[1]).write_text(json.dumps(result), encoding="utf-8")
    return 0


def tune_vs_dp(
    dp_pct: Sequence[float], tune_x: Sequence[float], tune_y: Sequence[float]
) -> dict[str, Any]:
    """The ``chromaticity`` result of a tune curve, whichever engine solved its points.

    The chromatic orders are the derivatives of the tune at zero offset, from a cubic
    fitted to the stable points: ``nu = nu0 + xi1 d + xi2 d^2 / 2 + xi3 d^3 / 6`` with
    ``d`` the fractional momentum offset. ``chromaticity_<plane>`` is ``xi1``,
    ``chromaticity2_<plane>`` is ``xi2`` and ``chromaticity3_<plane>`` is ``xi3``:
    elegant's ``dnux/dp``, ``dnux/dp2`` and ``dnux/dp3``.

    Args:
        dp_pct: The momentum offsets, in percent.
        tune_x: The horizontal tune at each offset; NaN where the optics are unstable.
        tune_y: The vertical tune at each offset.

    Returns:
        The result document. Fewer than four stable points fit no orders.
    """
    import numpy as np

    offsets = np.asarray(dp_pct, dtype=float)
    tunes = np.column_stack([np.asarray(tune_x, dtype=float), np.asarray(tune_y, dtype=float)])
    scalars: dict[str, float] = {}
    stable = np.all(np.isfinite(tunes), axis=1)
    if np.count_nonzero(stable) >= 4:
        for plane, column in (("x", 0), ("y", 1)):
            cubic, square, linear, _ = np.polyfit(offsets[stable] / 100.0, tunes[stable, column], 3)
            scalars[f"chromaticity_{plane}"] = float(linear)
            scalars[f"chromaticity2_{plane}"] = float(2.0 * square)
            scalars[f"chromaticity3_{plane}"] = float(6.0 * cubic)
    return {
        "scalars": scalars,
        "tables": {
            "tune_vs_dp": {
                "columns": {
                    "dp_pct": offsets.tolist(),
                    "tune_x": tunes[:, 0].tolist(),
                    "tune_y": tunes[:, 1].tolist(),
                },
                "units": {"dp_pct": "%"},
            }
        },
    }
