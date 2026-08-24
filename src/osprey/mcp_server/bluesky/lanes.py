"""Which plan lane a Bluesky tool call is addressed to.

A **plan lane** is a whole Bluesky stack — bridge, queue manager, worker — wired
at render time to ONE control-system target. Every deployment renders one; a
deployment whose build profile set ``bluesky.second_lane`` renders two, one per
target, so a session switched away from the deployment baseline still has a lane
to queue plans on.

This module is the host's half of the producer split the lane axis is built on:

* each **bridge** publishes what it statically is — its lane key and the target
  that lane serves (``queue_backend.resolve_lane_identity``). It cannot do more:
  the session's target lives in a state file the controls MCP server writes on
  the host, outside every bridge container's filesystem;
* the **host** — here — compares that against the session target it alone can
  see, and composes the *active/inactive* view: the active lane is the lane
  whose target equals the session target.

Two deployment shapes, one code path
------------------------------------
The branch point is the LANE COUNT, and it is the only one:

* **Single lane** (every deployment until a second one is opted into). There is
  nothing to route to, so a session switched away from the lane's target is
  refused outright — the blanket refusal ``queue.py`` has carried since the plan
  stack shipped. :attr:`LaneSituation.active` is ``None`` in exactly that case,
  so the refusal and the routing decision are the same computed fact rather than
  two rules that could drift.
* **Two lanes.** The switch is no longer a refusal but an address: the operation
  is routed to the lane serving the session's target, and refused only when no
  single lane serves it — a config that renders two lanes for the same target,
  or an ambiguity — which fails closed for the same reason a wrong-machine write
  is unrecoverable.

Nothing here mints an error envelope: the refusal wording belongs with the tool
that refuses (``tools/queue.py``), so an agent meets one voice per surface. This
module answers only "which lanes exist, which one is active, and why not".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from osprey.bluesky_bridge_connection import (
    LANE_ONE,
    discover_lane_keys,
    lane_declared_target,
)
from osprey.mcp_server.control_system import target_banner
from osprey.mcp_server.control_system.target_banner import TargetSituation

logger = logging.getLogger("osprey.mcp_server.bluesky.lanes")

#: A lane was named that this deployment does not render. Host-only: no bridge
#: can mint it, because a bridge knows only its own lane and never the set.
REASON_UNKNOWN_LANE = "unknown_bluesky_lane"

#: A two-lane deployment was asked to start a queue without naming which lane.
#: Host-only, and deliberately not defaulted: on a deployment with two lanes,
#: "start the queue" without a lane is an ambiguous instruction about hardware
#: motion, and guessing which machine was meant is the failure this axis exists
#: to prevent.
REASON_LANE_REQUIRED = "lane_required"

#: The named lane is not the one the session is currently pointed at — the queue
#: item was bound to a lane the session has since switched away from, or the
#: caller simply named the other one. Host-only.
REASON_LANE_MISMATCH = "lane_mismatch"


@dataclass(frozen=True)
class Lane:
    """One rendered plan lane: its service key and the target it serves.

    Attributes:
        key: The lane's ``services.<lane>`` key — ``bluesky``, ``bluesky_va`` or
            ``bluesky_live``. This is the lane id every surface names it by.
        target: ``live`` or ``va``. Declared by the lane's own config block on a
            two-lane deployment; the deployment baseline on a single-lane one,
            whose block has never had a reason to name a target.
    """

    key: str
    target: str


@dataclass(frozen=True)
class LaneSituation:
    """Every lane this deployment renders, plus which one the session is on.

    Attributes:
        lanes: The rendered lanes, in render order. Never empty — lane 1 always
            exists, including when the config cannot be read at all.
        active: The lane serving the session's target, or ``None`` when no
            single lane does. ``None`` is the refusal case on BOTH deployment
            shapes: on a single-lane deployment it means the session is switched
            away from the only lane there is, and on a two-lane one it means the
            rendered lanes do not cover the session's target unambiguously.
        target_situation: The session/baseline pair the active lane was resolved
            against, kept so a caller renders its refusal from the same facts
            rather than resolving them a second time.
    """

    lanes: tuple[Lane, ...]
    active: Lane | None
    target_situation: TargetSituation

    @property
    def multi_lane(self) -> bool:
        """Whether this deployment renders more than one plan lane."""
        return len(self.lanes) > 1

    @property
    def session_target(self) -> str:
        """The control target this session is pointed at."""
        return self.target_situation.session_target

    @property
    def baseline_target(self) -> str:
        """The control target this deployment's config declares."""
        return self.target_situation.baseline_target

    @property
    def lane_keys(self) -> tuple[str, ...]:
        """Every rendered lane's service key, in render order."""
        return tuple(lane.key for lane in self.lanes)

    def lane(self, key: str) -> Lane | None:
        """The rendered lane with this service key, or ``None`` if unrendered."""
        for lane in self.lanes:
            if lane.key == key:
                return lane
        return None


def discover_lanes(baseline_target: str) -> tuple[Lane, ...]:
    """Every plan lane this deployment renders, with the target each serves.

    A lane's target comes from its own ``services.<lane>.target`` key, which is
    written only on a two-lane deploy. A lane without one gets *baseline_target*
    — the same fallback the bridge applies to its own record
    (``queue_backend.resolve_lane_identity``), so host and bridge answer "which
    target does this lane serve" identically.

    Never raises: an unreadable config yields the single-lane answer rather than
    a half-built lane set.

    :param baseline_target: The deployment baseline, ``live`` or ``va``.
    """
    try:
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
    except Exception:
        logger.debug("Could not load the project config for lane discovery", exc_info=True)
        config = {}
    if not isinstance(config, dict):
        config = {}

    lanes: list[Lane] = []
    for key in discover_lane_keys(config):
        target = lane_declared_target(key, config)
        lanes.append(Lane(key=key, target=target or baseline_target))
    return tuple(lanes)


def resolve_lane_situation() -> LaneSituation:
    """Resolve the rendered lanes and which of them the session is on.

    Never raises. Every failure — an unreadable config, unreadable session state
    — collapses to the same answer an unswitched single-lane deployment gets,
    which is the direction that keeps a deployment that never switches (today,
    nearly all of them) behaving exactly as it always has.

    The active lane is the lane whose target equals the session target, and it
    is deliberately resolved by an EXACT match on exactly one lane: zero matches
    (no rendered lane serves where the session is pointed) and more than one
    (two lanes rendered for the same target) are both "no answer", and no answer
    fails closed into a refusal rather than into a guess about which machine a
    plan would run on.
    """
    situation = target_banner.resolve_target_situation()
    lanes = discover_lanes(situation.baseline_target)
    matches = [lane for lane in lanes if lane.target == situation.session_target]
    if len(matches) != 1:
        if matches:
            logger.debug(
                "Ambiguous plan lanes: %d lanes serve target %r",
                len(matches),
                situation.session_target,
            )
        return LaneSituation(lanes=lanes, active=None, target_situation=situation)
    return LaneSituation(lanes=lanes, active=matches[0], target_situation=situation)


def compose_lane_capability(capability: object, *, active: bool) -> dict:
    """The host's active/inactive view composed onto a bridge's static capability.

    The bridge's record says what the lane IS (``lane``, ``lane_target``,
    ``can_execute``, ``reason``, ``detail``); this adds the one field only the
    host can know — whether the session is currently pointed at that lane. The
    bridge's fields are copied through untouched, so a consumer that already
    reads the capability shape gains a field rather than losing one.

    :param capability: The ``capability`` object from a lane's ``/health``.
    :param active: Whether this lane serves the session's target.
    """
    composed = dict(capability) if isinstance(capability, dict) else {}
    composed["active"] = active
    return composed


def lane_roster(situation: LaneSituation) -> list[dict]:
    """The rendered lanes as plain dicts, each marked active or not.

    The roster a refusal carries so whoever reads it can see the whole board:
    which lanes exist, which machine each drives, and which one the session is
    on right now. Exactly one entry is ``active`` whenever
    :attr:`LaneSituation.active` is set, and none is when it is not.
    """
    active_key = situation.active.key if situation.active else None
    return [
        {"lane": lane.key, "lane_target": lane.target, "active": lane.key == active_key}
        for lane in situation.lanes
    ]


__all__ = [
    "LANE_ONE",
    "REASON_LANE_MISMATCH",
    "REASON_LANE_REQUIRED",
    "REASON_UNKNOWN_LANE",
    "Lane",
    "LaneSituation",
    "compose_lane_capability",
    "discover_lanes",
    "lane_roster",
    "resolve_lane_situation",
]
