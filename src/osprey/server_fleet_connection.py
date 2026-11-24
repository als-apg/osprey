"""The fleet connection: how every reader reaches the server fleet service (C6).

Every reader of the fleet resolves the service through this module and nowhere
else: IOC facts (``osprey.services.channel_directory.facts``), the ``server_fleet``
tool, the IOCs view's panel route and the Reach Contract
(``osprey.deployment.reach``), whose ``reach`` health probe knocks on what the
readers would dial. A top-level leaf module, like
``osprey.optimizer_connection``: it imports nothing from ``osprey.mcp_server``
or ``osprey.deployment``, and the snapshot model loads only when a snapshot is
fetched.

A reader never falls back to reading the control system: every failure to
fetch becomes an answer whose ``outcome`` says so, with a reason, and every
surface renders a non-``ok`` outcome as a statement.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import httpx

from osprey.port_layout import default_port, resolve_port_base

if TYPE_CHECKING:
    from osprey.services.channel_directory.fleet.snapshot import ServerFleetRow, Snapshot

__all__ = [
    "DEFAULT_SERVER_FLEET_PORT",
    "SERVER_FLEET_TIMEOUT_S",
    "SERVER_FLEET_URL_ENV_VAR",
    "STALE_FACTOR",
    "FleetAnswer",
    "FleetEntry",
    "FleetOutcome",
    "entry_for",
    "fetch_snapshot",
    "fleet_url",
]

#: The port the fleet service publishes at the layout's default base.
DEFAULT_SERVER_FLEET_PORT = default_port("server_fleet")

#: Names the fleet service's base URL outright, overriding the rendered config.
#: A bridged dispatch worker is handed the compose-name URL under this name.
SERVER_FLEET_URL_ENV_VAR = "SERVER_FLEET_URL"

#: Seconds a reader waits for the service. The service only reads a file per
#: request, so a longer wait means it is wedged; a short bound keeps a slow
#: service from stalling an IOC page or a tool call.
SERVER_FLEET_TIMEOUT_S = 3.0

#: A snapshot older than this many times the larger of its period and its last
#: cycle's duration is ``stale``: its refresher has stopped refreshing it.
STALE_FACTOR = 3

_SNAPSHOT_PATH = "/v1/snapshot"


class FleetOutcome(StrEnum):
    """C5's outcomes, plus what only a reader of the service can see."""

    OK = "ok"
    NO_SNAPSHOT = "no_snapshot"
    NOT_REFRESHED = "not_refreshed"
    CANNOT_LIST = "cannot_list"
    #: The service could not be reached or gave no usable answer.
    UNAVAILABLE = "unavailable"
    #: The snapshot is older than the stale bound; refreshes have stopped.
    STALE = "stale"
    #: Only from :func:`entry_for`: the snapshot is fresh but has no such IOC.
    NOT_IN_SNAPSHOT = "not_in_snapshot"


def fleet_url(config: Mapping[str, Any] | None) -> str | None:
    """The fleet service's base URL, trailing slash stripped; ``None`` when the fleet is off.

    The fleet is on when ``channel_directory.fleet`` is present (the
    ``channel_directory_fleet`` predicate; a malformed section is refused at build
    and reads as off here). Then :data:`SERVER_FLEET_URL_ENV_VAR` when set and
    non-empty; else ``services.server_fleet.port`` on loopback; else the layout's
    ``server_fleet`` slot at this config's ``deployment.port_base``.
    """
    section = (config or {}).get("channel_directory")
    if not isinstance(section, Mapping) or section.get("fleet") is None:
        return None
    full = os.environ.get(SERVER_FLEET_URL_ENV_VAR)
    if full:
        return full.rstrip("/")
    services = (config or {}).get("services")
    block = services.get("server_fleet") if isinstance(services, Mapping) else None
    port = block.get("port") if isinstance(block, Mapping) else None
    if not port:
        port = default_port("server_fleet", base=resolve_port_base(config))
    return f"http://127.0.0.1:{port}"


@dataclass(frozen=True, kw_only=True)
class FleetAnswer:
    """What a reader got for one target.

    ``snapshot`` is the decoded snapshot on ``ok`` and ``stale`` (so a view can
    still show a stale one, marked, with its time) and ``None`` otherwise;
    every non-``ok`` answer carries a ``reason``.
    """

    outcome: FleetOutcome
    target: str
    reason: str | None = None
    generated_at: float | None = None
    snapshot: Snapshot | None = None

    def __post_init__(self) -> None:
        if self.outcome is FleetOutcome.NOT_IN_SNAPSHOT:
            raise ValueError("not_in_snapshot is an entry's outcome, not an answer's")
        if self.outcome is FleetOutcome.OK:
            if self.snapshot is None or self.reason is not None:
                raise ValueError("an ok answer carries its snapshot and no reason")
        elif not self.reason:
            raise ValueError(f"a {self.outcome.value} answer needs a reason")

    @classmethod
    def unavailable(cls, target: str, reason: str) -> FleetAnswer:
        """The service gave no usable answer for ``target``."""
        return cls(outcome=FleetOutcome.UNAVAILABLE, target=target, reason=reason)

    def to_json(self) -> dict[str, Any]:
        """The answer as JSON; ``snapshot`` is the C5 form or ``None``."""
        return {
            "outcome": self.outcome.value,
            "target": self.target,
            "reason": self.reason,
            "generated_at": self.generated_at,
            "snapshot": self.snapshot.to_json() if self.snapshot is not None else None,
        }


@dataclass(frozen=True, kw_only=True)
class FleetEntry:
    """One IOC's fleet entry: its row on a fresh snapshot, else an outcome with a reason."""

    outcome: FleetOutcome
    target: str
    reason: str | None = None
    generated_at: float | None = None
    row: ServerFleetRow | None = None

    def to_json(self) -> dict[str, Any]:
        """The entry as JSON; ``row`` is the IOC's C5 row or ``None``."""
        return {
            "outcome": self.outcome.value,
            "target": self.target,
            "reason": self.reason,
            "generated_at": self.generated_at,
            "row": None if self.row is None else self.row.to_json(),
        }


async def fetch_snapshot(
    config: Mapping[str, Any] | None,
    target: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    now: float | None = None,
) -> FleetAnswer:
    """The service's snapshot of ``target``, as an answer; never raises on the service's behaviour.

    Args:
        config: The rendered config the URL is resolved from.
        target: The control target whose snapshot is asked for.
        transport: The HTTP transport; tests hand in a mock one.
        now: The current time in epoch seconds, for the stale rule; the clock
            when ``None``.

    Returns:
        ``ok`` with the snapshot; the service's own ``no_snapshot``,
        ``not_refreshed`` or ``cannot_list`` with its reason; ``stale`` with the
        snapshot when it is older than :data:`STALE_FACTOR` times the larger of
        its period and cycle duration; else ``unavailable`` with the reason.
    """
    url = fleet_url(config)
    if url is None:
        return FleetAnswer.unavailable(
            target, "the server fleet is not configured in this deployment"
        )
    try:
        async with httpx.AsyncClient(timeout=SERVER_FLEET_TIMEOUT_S, transport=transport) as client:
            response = await client.get(f"{url}{_SNAPSHOT_PATH}", params={"target": target})
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        return FleetAnswer.unavailable(
            target, f"could not reach the server fleet service at {url} ({type(exc).__name__})"
        )
    except httpx.TimeoutException:
        return FleetAnswer.unavailable(
            target, f"the server fleet service did not answer within {SERVER_FLEET_TIMEOUT_S:g} s"
        )
    except httpx.HTTPError as exc:
        return FleetAnswer.unavailable(
            target, f"the request to the server fleet service failed ({type(exc).__name__})"
        )
    if not response.is_success:
        return FleetAnswer.unavailable(
            target, f"the server fleet service answered HTTP {response.status_code}"
        )
    try:
        document = response.json()
    except ValueError:
        return FleetAnswer.unavailable(target, "the server fleet service's answer is not JSON")
    return _decide(document, target, time.time() if now is None else now)


def _decide(document: Any, target: str, now: float) -> FleetAnswer:
    from osprey.services.channel_directory.fleet.snapshot import (
        Outcome,
        Snapshot,
        SnapshotFormatError,
    )

    try:
        snapshot = Snapshot.from_json(document)
    except SnapshotFormatError as exc:
        return FleetAnswer.unavailable(
            target, f"the server fleet service's answer is not a snapshot ({exc})"
        )
    if snapshot.target != target:
        return FleetAnswer.unavailable(
            target, f"the server fleet service answered for target {snapshot.target!r}"
        )
    if snapshot.outcome is not Outcome.OK:
        return FleetAnswer(
            outcome=FleetOutcome(snapshot.outcome.value), target=target, reason=snapshot.reason
        )
    assert snapshot.generated_at is not None  # an ok snapshot has all of these
    assert snapshot.period_s is not None
    assert snapshot.cycle_duration_s is not None
    bound = STALE_FACTOR * max(snapshot.period_s, snapshot.cycle_duration_s)
    age = now - snapshot.generated_at
    if age > bound:
        return FleetAnswer(
            outcome=FleetOutcome.STALE,
            target=target,
            reason=(
                f"the snapshot is {age:.0f} s old, more than {bound:g} s: "
                "the server fleet service has stopped refreshing this target"
            ),
            generated_at=snapshot.generated_at,
            snapshot=snapshot,
        )
    return FleetAnswer(
        outcome=FleetOutcome.OK,
        target=target,
        generated_at=snapshot.generated_at,
        snapshot=snapshot,
    )


def entry_for(answer: FleetAnswer, ioc: str) -> FleetEntry:
    """IOC ``server``'s entry in ``answer``: its row when the answer is ``ok``.

    A fresh snapshot without the IOC gives ``not_in_snapshot``; a non-``ok``
    answer gives its own outcome and reason, never a row.
    """
    if answer.outcome is not FleetOutcome.OK or answer.snapshot is None:
        return FleetEntry(
            outcome=answer.outcome,
            target=answer.target,
            reason=answer.reason,
            generated_at=answer.generated_at,
        )
    row = answer.snapshot.row(ioc)
    if row is None:
        return FleetEntry(
            outcome=FleetOutcome.NOT_IN_SNAPSHOT,
            target=answer.target,
            reason=f"{ioc} is not in the {answer.target} snapshot",
            generated_at=answer.generated_at,
        )
    return FleetEntry(
        outcome=FleetOutcome.OK,
        target=answer.target,
        generated_at=answer.generated_at,
        row=row,
    )
