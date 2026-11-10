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
from collections.abc import Callable, Mapping
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
