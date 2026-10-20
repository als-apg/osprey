"""The trajectory service's keys, their defaults, and the bounds derived from them.

It also spells the service's wire names a client needs (its doors and its
session header), so the agent's MCP client and the service share one
definition; this module imports nothing beyond the standard library at load
time, and :func:`connector_call_timeout_s` imports the connector package only
when it is called.

Every ``services.trajectory.*`` key the service reads beside its target and
model is spelled here once, with the value a deployment gets when it states
none. The bounds that follow from a sequence module's longest pause, the
connector's call timeout and the stop-checkpoint poll are computed by one
function, :func:`budgets`, so the service, its state snapshot and the compose
render cannot disagree about how long an action or a stop may take.

Units:

- ``loss_cutoff_fraction`` and ``shot_threshold_fraction`` are fractions in
  ``[0, 1]`` of a shot's injected-charge scale, the largest rise of any BPM's
  sum (:mod:`~osprey.services.trajectory.extraction`), so they hold in any
  facility's sum units.
- ``*_s`` keys are seconds.
- ``raw_window_turns`` counts turns.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ACTION_TIMEOUT_KEY",
    "AGENT_DOOR",
    "ARMED_TIMEOUT_KEY",
    "CHECKPOINT_POLL_S",
    "CLIENT_HEADER",
    "CONTINUOUS_MIN_PERIOD_KEY",
    "DEFAULT_ACTION_TIMEOUT_S",
    "DEFAULT_ARMED_TIMEOUT_S",
    "DEFAULT_CONTINUOUS_MIN_PERIOD_S",
    "DEFAULT_LOSS_CUTOFF_FRACTION",
    "DEFAULT_RAW_RETENTION",
    "DEFAULT_RAW_WINDOW_TURNS",
    "DEFAULT_SHOT_THRESHOLD_FRACTION",
    "LOSS_CUTOFF_FRACTION_KEY",
    "MAX_PAUSE_CEILING_S",
    "PANEL_DOOR",
    "RAW_RETENTION_KEY",
    "RAW_WINDOW_TURNS_KEY",
    "SEQUENCE_DIR_ENV_VAR",
    "SEQUENCE_DIR_KEY",
    "SEQUENCE_MOUNT",
    "SERVICE_KEY",
    "SHOT_THRESHOLD_FRACTION_KEY",
    "SHUTDOWN_MARGIN_S",
    "VA_SEQUENCE_DIR",
    "Budgets",
    "budgets",
    "connector_call_timeout_s",
    "stop_grace_period_s",
]

#: The door ``/api/*`` admits through: the agent's, behind the launch token.
AGENT_DOOR = "agent"
#: The door ``/panel/api/*`` admits through: the panel's.
PANEL_DOOR = "panel"
#: The header a session holder's later calls carry its ``client_id`` in.
CLIENT_HEADER = "X-Trajectory-Client"

#: The service's block in the rendered ``config.yml``.
SERVICE_KEY = "services.trajectory"

#: The rise of a BPM's sum, as a fraction of the injected-charge scale, above
#: which the beam reached it: the SolveConfig's default ``loss_cutoff``.
LOSS_CUTOFF_FRACTION_KEY = f"{SERVICE_KEY}.loss_cutoff_fraction"
#: The first BPM's first-turn level, as a fraction of the injected-charge
#: scale, below which a shot failed.
SHOT_THRESHOLD_FRACTION_KEY = f"{SERVICE_KEY}.shot_threshold_fraction"
#: Seconds each wait on the BPMs' ``armed`` flags may take.
ARMED_TIMEOUT_KEY = f"{SERVICE_KEY}.armed_timeout_s"
#: Turns of raw waveform kept after the first turn.
RAW_WINDOW_TURNS_KEY = f"{SERVICE_KEY}.raw_window_turns"
#: The host directory holding the facility's sequence module.
SEQUENCE_DIR_KEY = f"{SERVICE_KEY}.sequence_dir"
#: Seconds one admitted action may run before it is asked to stop.
ACTION_TIMEOUT_KEY = f"{SERVICE_KEY}.action_timeout_s"
#: Shots whose raw buffers the service keeps, the latest always among them.
RAW_RETENTION_KEY = f"{SERVICE_KEY}.raw_retention"
#: Seconds at least between the starts of two continuous iterations; set it no
#: shorter than the facility's injector cycle.
CONTINUOUS_MIN_PERIOD_KEY = f"{SERVICE_KEY}.continuous_min_period_s"

DEFAULT_LOSS_CUTOFF_FRACTION = 0.5
DEFAULT_SHOT_THRESHOLD_FRACTION = 0.5
DEFAULT_ARMED_TIMEOUT_S = 10.0
DEFAULT_RAW_WINDOW_TURNS = 20
DEFAULT_ACTION_TIMEOUT_S = 300.0
DEFAULT_RAW_RETENTION = 100
DEFAULT_CONTINUOUS_MIN_PERIOD_S = 1.0

#: The longest pause, in seconds, a sequence module may declare (``max_pause_s``).
MAX_PAUSE_CEILING_S = 10.0
#: Seconds between two looks at the stop event while the service waits.
CHECKPOINT_POLL_S = 0.1
#: Seconds the container is given beyond the computed shutdown, for the
#: process itself to exit.
SHUTDOWN_MARGIN_S = 5.0

#: Where a container finds the sequence module directory; set by the compose
#: render beside the read-only bind at :data:`SEQUENCE_MOUNT`.
SEQUENCE_DIR_ENV_VAR = "OSPREY_TRAJECTORY_SEQUENCE_DIR"
#: The fixed in-container target the sequence module directory is bound at.
SEQUENCE_MOUNT = "/app/project/sequence"
#: The shipped virtual-accelerator module's directory in a built project,
#: relative to the repo root: the build's default for a VA target.
VA_SEQUENCE_DIR = "./build/services/trajectory/sequence"


@dataclass(frozen=True)
class Budgets:
    """How long an action and a stop may take, in seconds.

    Attributes:
        stop_budget_s: The longest an action takes to end after Stop: the
            module's longest pause, one connector call, and one poll of the
            stop event.
        action_budget_s: The longest an admitted action takes: its timeout,
            then the stop that timeout asks for.
    """

    stop_budget_s: float
    action_budget_s: float


def budgets(
    *,
    max_pause_s: float,
    call_timeout_s: float,
    action_timeout_s: float = DEFAULT_ACTION_TIMEOUT_S,
    poll_s: float = CHECKPOINT_POLL_S,
) -> Budgets:
    """The stop and action budgets of one service.

    Args:
        max_pause_s: The sequence module's declared longest pause.
        call_timeout_s: The service connector's call timeout.
        action_timeout_s: The configured action timeout.
        poll_s: The stop-checkpoint poll.

    Returns:
        Both budgets.
    """
    stop = max_pause_s + call_timeout_s + poll_s
    return Budgets(stop_budget_s=stop, action_budget_s=action_timeout_s + stop)


def stop_grace_period_s(call_timeout_s: float) -> int:
    """Whole seconds a container stop must allow the service, for any module.

    A stop runs, in order: ``Sessions.shutdown()``, which ends the action in
    flight within the stop budget; the server's graceful drain, bounded by the
    same budget; and the connector's disconnect, one call. The stop budget is
    taken at :data:`MAX_PAUSE_CEILING_S`, the longest pause any loadable
    module may declare, so the bound holds whichever module is deployed.

    Args:
        call_timeout_s: The service connector's call timeout.

    Returns:
        The grace period, rounded up to a whole second.
    """
    stop = budgets(max_pause_s=MAX_PAUSE_CEILING_S, call_timeout_s=call_timeout_s).stop_budget_s
    return math.ceil(2 * stop + call_timeout_s + SHUTDOWN_MARGIN_S)


def connector_call_timeout_s(config: Mapping[str, Any], target: str) -> float:
    """The call timeout the service connector's own block declares.

    The service runs under this bound and the compose render derives the
    container's stop grace period from it, so both read it here.

    Args:
        config: The deployment's ``config.yml`` contents.
        target: The service's control target.

    Returns:
        The block's ``timeout_s`` in seconds, or the connector default when
        the config has no ``control_system`` section, the target does not
        resolve to a connector type, or that type's block declares none.

    Raises:
        ValueError: If the resolved block's ``timeout_s`` is invalid.
    """
    from osprey_connectors.control_system.call_timeout import DEFAULT_TIMEOUT_S, call_timeout_s
    from osprey_connectors.types import resolve_target

    section = config.get("control_system")
    if not isinstance(section, Mapping):
        return DEFAULT_TIMEOUT_S
    try:
        connector_type = resolve_target(section, target)
    except ValueError:
        return DEFAULT_TIMEOUT_S
    blocks = section.get("connector")
    block = blocks.get(connector_type) if isinstance(blocks, Mapping) else None
    return call_timeout_s(block if isinstance(block, Mapping) else {}, connector_type)
