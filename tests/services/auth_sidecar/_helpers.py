"""Plain helpers shared by the auth sidecar suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from osprey.audit import writer
from osprey.services.auth_sidecar import audit


class FakeClock:
    """A hand-advanced stand-in for ``time.time`` or ``time.monotonic``.

    The sidecar's windows and lifetimes are seconds long, so tests that let them
    lapse in real time would be pinning the machine's speed rather than the
    behaviour, and tests that stay inside one would be assuming two calls land
    within a second of each other.
    """

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def ledger_records(zone: Path) -> list[dict[str, Any]]:
    """Every record the sidecar filed in ``zone`` so far, in append order."""
    path = zone / f"{audit.SURFACE}{writer.LEDGER_SUFFIX}"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def set_cookie_header(response: httpx.Response, name: str) -> str:
    """The raw ``Set-Cookie`` header for ``name``; attributes are asserted on it."""
    for header in response.headers.get_list("set-cookie"):
        if header.startswith(f"{name}="):
            return header
    raise AssertionError(f"no Set-Cookie for {name!r} in {response.headers.get_list('set-cookie')}")


def osprey_log(caplog: pytest.LogCaptureFixture) -> str:
    """Only this service's log records.

    The test client's own HTTP logger echoes full request URLs, so asserting
    against the whole capture would test httpx rather than what the sidecar
    writes.
    """
    return "\n".join(
        record.getMessage() for record in caplog.records if record.name.startswith("osprey.")
    )
