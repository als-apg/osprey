"""The served process's health record, kept current after every publishing pass.

:class:`ServingHealth` counts the outcome of each publishing pass and decides
the process's ``state`` from how many of the latest passes failed in a row:

* ``serving`` -- the latest pass succeeded;
* ``degraded`` -- between one and ``failed_pass_tolerance`` consecutive passes
  failed;
* ``failed`` -- more than ``failed_pass_tolerance`` consecutive passes failed.

A successful pass returns the record to ``serving``; the counters and the last
failed pass stay as history. :meth:`ServingHealth.document` is the record as
the model RPC's ``status`` reports it and :func:`write_document` puts it on
disk, replaced atomically, for a probe outside the process to read.

Nothing here imports a server library, so the record is decided and tested in
process.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: The states a record is in, in order of what a pass outcome can bring.
STATE_SERVING = "serving"
STATE_DEGRADED = "degraded"
STATE_FAILED = "failed"

OUTCOME_OK = "ok"
OUTCOME_FAILED = "failed"


class ServingHealth:
    """The outcome of every publishing pass, and the state they add up to.

    Args:
        failed_pass_tolerance: how many consecutive failed passes still count
            as ``degraded``; one more is ``failed``. An int, zero or above.
        clock: seconds on a monotonic scale; ``uptime_s`` is read from it.
        started: the ``clock`` reading ``uptime_s`` counts from.

    Raises:
        ValueError: ``failed_pass_tolerance`` is a bool, not an int, or below
            zero.
    """

    def __init__(
        self, failed_pass_tolerance: int, clock: Callable[[], float], started: float
    ) -> None:
        if (
            isinstance(failed_pass_tolerance, bool)
            or not isinstance(failed_pass_tolerance, int)
            or failed_pass_tolerance < 0
        ):
            raise ValueError(
                f"failed_pass_tolerance must be an int of 0 or more, not {failed_pass_tolerance!r}"
            )
        self._tolerance = failed_pass_tolerance
        self._clock = clock
        self._started = started
        self._passes_ok = 0
        self._passes_failed = 0
        self._consecutive_failed = 0
        self._last_pass: dict[str, Any] | None = None
        self._last_failed_pass: dict[str, Any] | None = None

    @property
    def state(self) -> str:
        """``serving``, ``degraded`` or ``failed``, from the consecutive failed passes."""
        if self._consecutive_failed == 0:
            return STATE_SERVING
        if self._consecutive_failed <= self._tolerance:
            return STATE_DEGRADED
        return STATE_FAILED

    @property
    def last_failed_pass(self) -> dict[str, Any] | None:
        """The ``error`` and ``uptime_s`` of the latest failed pass, or ``None``."""
        return None if self._last_failed_pass is None else dict(self._last_failed_pass)

    def record_pass(self, error: str | None) -> None:
        """Record one publishing pass: ``None`` for a success, else the error it raised."""
        uptime_s = self._clock() - self._started
        if error is None:
            self._passes_ok += 1
            self._consecutive_failed = 0
            outcome = OUTCOME_OK
        else:
            self._passes_failed += 1
            self._consecutive_failed += 1
            self._last_failed_pass = {"error": str(error), "uptime_s": uptime_s}
            outcome = OUTCOME_FAILED
        self._last_pass = {"outcome": outcome, "uptime_s": uptime_s, "at": time.time()}

    def document(self) -> dict[str, Any]:
        """The record as plain JSON-able data; ``last_pass`` is ``None`` before any pass.

        ``last_pass.at`` is wall-clock seconds since the epoch; every
        ``uptime_s`` is on the monotonic scale the record was built with.
        """
        return {
            "state": self.state,
            "last_pass": None if self._last_pass is None else dict(self._last_pass),
            "passes_ok": self._passes_ok,
            "passes_failed": self._passes_failed,
            "consecutive_failed": self._consecutive_failed,
            "failed_pass_tolerance": self._tolerance,
            "last_failed_pass": self.last_failed_pass,
        }


def write_document(path: Path, document: dict[str, Any]) -> None:
    """Write ``document`` to ``path`` as sorted JSON, replacing it atomically.

    The document is written to a temporary file in ``path``'s directory,
    which is created if absent, and moved over ``path`` with
    :func:`os.replace`, so a reader sees the previous record or this one
    and never a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


__all__ = [
    "STATE_DEGRADED",
    "STATE_FAILED",
    "STATE_SERVING",
    "ServingHealth",
    "write_document",
]
