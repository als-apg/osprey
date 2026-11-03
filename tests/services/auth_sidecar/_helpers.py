"""Plain helpers shared by the auth sidecar suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from osprey.agent_record.contract import KIND_AUDIT
from tests.conftest import SpoolZone

#: The signing secrets and session lifetime every suite configures the sidecar with.
SESSION_SECRET = "session-secret-value"
STATE_SECRET = "state-secret-value"
SESSION_LIFETIME = 3600


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


#: The identity the sidecar's container is given, and therefore the name of the
#: spool directory its entries are written in.
SIDECAR_IDENTITY = "sidecar"


@dataclass(frozen=True)
class SidecarZone:
    """The sidecar's own spool and state directories, and the record they feed."""

    spool: Path
    state: Path
    spool_zone: SpoolZone

    def entries(self) -> list[Any]:
        """Every ``audit`` entry the record holds so far, in the order it was written."""
        return list(self.spool_zone.recorded_entries(KIND_AUDIT))

    def records(self) -> list[dict[str, Any]]:
        """The body of every ``audit`` entry the record holds so far, in write order."""
        return [dict(entry.body) for entry in self.entries()]


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
