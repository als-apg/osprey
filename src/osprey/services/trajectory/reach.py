"""Whether the trajectory service's target answers, asked live on every call.

Reachability is one short-timeout read of the selected model's
``injection_busy`` slot through the service's one connector. The ``/health``
route asks it on every call and the action worker asks it before every
writing action; control-path actions (release, take-over) never ask it, so an
unreachable target never blocks a release. Nothing about reachability is kept
between calls: the connector's ``connect()`` contacts no server, so a target
that came up after startup answers on the next call, and one that went away
fails on the next call.

:func:`probe` answers ``None`` when the read came back, and otherwise the code
the caller replies with: :data:`TRAJECTORY_UNAVAILABLE` when the build gave
the service no trajectory model (there is no slot to read), and
:data:`TARGET_UNREACHABLE` when the read failed or did not come back in time.
"""

from __future__ import annotations

import asyncio
import logging

from osprey.services.trajectory.startup import TrajectoryStartup

logger = logging.getLogger(__name__)

__all__ = [
    "PROBE_TIMEOUT_S",
    "REACHABLE",
    "TARGET_UNREACHABLE",
    "TRAJECTORY_UNAVAILABLE",
    "probe",
]

#: How long one reachability read may take, in seconds.
PROBE_TIMEOUT_S = 2.0

#: What ``/health`` reports for the target when the read came back.
REACHABLE = "reachable"
#: The read of ``injection_busy`` failed or did not come back in time.
TARGET_UNREACHABLE = "target_unreachable"
#: The build gave the service no trajectory model, so there is nothing to read.
TRAJECTORY_UNAVAILABLE = "trajectory_unavailable"


async def probe(startup: TrajectoryStartup, *, timeout: float = PROBE_TIMEOUT_S) -> str | None:
    """Read the selected model's ``injection_busy`` slot once, now.

    The read is bounded twice: the connector is asked to give up after
    ``timeout``, and the await is cancelled shortly after that in case the
    connector does not honour its own timeout. The value read is not
    interpreted; that it came back is the answer.

    Args:
        startup: The started service: its connector and command set.
        timeout: How long the read may take, in seconds.

    Returns:
        ``None`` when the target answered; :data:`TRAJECTORY_UNAVAILABLE` when
        the service has no trajectory model; :data:`TARGET_UNREACHABLE` when
        the read raised or timed out.
    """
    commands = startup.commands
    if commands is None:
        return TRAJECTORY_UNAVAILABLE
    address = commands.injection_busy
    try:
        await asyncio.wait_for(
            startup.connector.read_channel(address, timeout=timeout),
            timeout=timeout + 0.5,
        )
    except Exception as exc:  # every failure of the read means the same thing
        logger.debug(
            "trajectory target %s unreachable: read of %s failed: %r", startup.target, address, exc
        )
        return TARGET_UNREACHABLE
    return None
