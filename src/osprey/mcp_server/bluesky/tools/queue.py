"""MCP tools: the agent's side of the bridge's plan queue.

Execution is two steps, deliberately: ``queue_add`` puts the pinned draft into
the queue (which starts nothing), and ``queue_start`` begins draining it (which
starts a real plan). Splitting them is what lets a human review a composed queue
before
anything moves, and it is why the arming gate sits on *start* rather than on
composition.

==========================  =================================================
Tool                        Bridge endpoint
==========================  =================================================
queue_status                GET    /health   (capability record)
queue_list                  GET    /queue
queue_add                   POST   /queue/items
queue_start                 POST   /queue/start
queue_stop                  POST   /queue/stop
==========================  =================================================

``stop_run`` (``tools/stop.py``) completes the halting surface with
``POST /queue/abort`` — the emergency stop for a plan already in motion. It
lives in its own module but shares this one's refusal relay and hint table, so
the two answer the same bridge code the same way.

**Arming.** Two operations can put hardware in motion. Two local safety layers
guard them, both enforced BEFORE any HTTP call — but only the first is carried
everywhere:

1. A ``control_system.writes_enabled`` re-read straight from config, mirroring
   ``ControlSystemConnector._writes_enabled``
   (``osprey/connectors/control_system/base.py``) exactly — same key, same
   fail-closed except clause. Re-read fresh on every call, never cached, so a
   hook-bypassed invocation carrying a valid launch token is still refused
   while writes are disabled.
2. A client-side launch-token presence check, so an unarmed server refuses
   locally with no network call.

3. A session-target check, on ``queue_add`` and ``queue_start`` only. A
   deployment's plan lane is wired at build time to one control target, while
   an agent session can be switched to the other one at run time
   (``control_target_set``). Queuing or starting a Bluesky PLAN while the two
   disagree would run it against a machine the session is not pointed at. This
   layer is enforced HERE rather than at the bridge because the session target
   is host state: a bridge serves its lane's target and never learns which
   target the session is on. Halting is never gated by it, for the same reason
   it is never gated by the kill switch.

   What that check DOES depends on how many lanes the deployment renders, and
   the lane count is the only branch point (see ``bluesky/lanes.py``):

   * **One lane** — every deployment until ``bluesky.second_lane`` is opted
     into. There is no other lane to route to, so a switched session is refused
     outright: the blanket refusal this module has always carried, unchanged.
   * **Two lanes** — one per control target. The switch stops being a refusal
     and becomes an ADDRESS: the operation is routed to the lane serving the
     session's target, ``queue_add`` returns the lane it bound the item to, and
     ``queue_start`` must name that lane and is refused when it no longer
     matches the active one (a session that switched between the add and the
     start). It still refuses when NO single rendered lane serves the session's
     target, which is a misrendered deployment rather than a switch.

   Both branches come out of one resolved fact — ``LaneSituation.active``, the
   lane whose target equals the session's — so the refusal and the routing
   decision can never drift apart.

``queue_start`` carries layer 1 unconditionally, then delegates the token
decision to the bridge: it posts with its launch token when it holds one and
with no token header when it does not, and the bridge answers
``launch_token_required`` for absence and mismatch alike. Keeping that one
verdict in one place is deliberate — a local presence check would let the agent
hear a different answer from the tool than the bridge would have given.
``queue_add`` is armed only *sometimes* — enqueuing onto an idle queue moves
nothing, while enqueuing onto a queue that is already draining hands the item
straight to hardware — so it too carries layer 1 and then attaches the launch
token to the request only when writes are enabled, leaving the bridge (which
alone knows the manager's live state, under a lock, with a post-add re-check)
to decide whether this particular enqueue needed it. A writes-disabled
deployment therefore keeps composing queues and is refused exactly at the point
where composing would become executing. ``queue_stop`` inverts the asymmetry: a
plain stop is ungated in every layer because halting is always allowed, while
``cancel=True`` withdraws a human's pending halt and is gated like a start — it
is the one operation that carries layer 2.

**Refusals are relayed, never rewritten.** Every refusal the bridge issues is
an HTTP 4xx/5xx whose ``detail`` is ``{"code", "detail", ...extras}``, where
``code`` is the machine-readable branch key. These tools surface the bridge's
``code`` as the envelope's ``error_type``, its sentence as the
``error_message``, and the entire detail object — including extras like
``capability``, ``manager_state``, ``revision``, ``plan`` and
``item_left_behind`` — as ``details``. Nothing is reworded or reclassified: a
tool that renamed a refusal would put the agent and the panel on different
vocabularies for the same event. The only envelopes minted here are the three
local refusals (``writes_disabled``, ``launch_token_required``,
``session_target_mismatch``) and a last-resort ``bluesky_bridge_error`` for a
bridge response that carried no structured detail at all.

**No reason-code constants are imported here, and exactly one is spelled out.**
The capability reason codes are defined in
``osprey.services.bluesky_bridge.queue_backend``, which pulls in
``bluesky-queueserver-api`` and the runner wiring; importing it would break this
server's standing invariant of making no bluesky/ophyd/tiled imports (see
``bluesky/server.py``). For every code the bridge mints, these tools never
branch on it — they pass the whole capability record through — so they need no
copy of the vocabulary and cannot drift from it. The one exception is
:data:`REASON_SESSION_TARGET_MISMATCH`, which no bridge can mint because the
session target is host state; its string is spelled here and pinned equal to
``queue_backend``'s constant by a test, so the vocabulary stays single even
though the two layers cannot import each other. The docstrings below name the
codes as prose, for the agent reading them.
"""

from __future__ import annotations

import json
from typing import NoReturn

import anyio
from fastmcp.exceptions import ToolError

from osprey.bluesky_bridge_connection import unwrap_bridge_conflict_detail
from osprey.mcp_server.bluesky.lanes import (
    LANE_ONE,
    REASON_LANE_MISMATCH,
    REASON_LANE_REQUIRED,
    REASON_UNKNOWN_LANE,
    Lane,
    LaneSituation,
    compose_lane_capability,
    lane_roster,
    resolve_lane_situation,
)
from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.server_context import (
    _http_get_json,
    _http_post_json,
    bridge_error_message,
    get_server_context,
)
from osprey.mcp_server.control_system.target_banner import (
    TargetSituation,
    baseline_refusal,
)
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async

# The capability reason code for "the session is pointed at a control target
# this deployment's single plan lane does not serve". Defined as a constant in
# `osprey.services.bluesky_bridge.queue_backend` beside the codes the bridge
# mints; spelled again here because this module may not import that one (see
# the module docstring), and pinned equal to it by
# `tests/services/test_single_lane_switch_refusal.py`.
#
# Deliberately NOT `target_banner.BASELINE_REFUSAL_ERROR_TYPE`, which is the
# generic "you are switched away" category every baseline-pinned holder refuses
# with. The queue surface has its own machine-readable vocabulary — the
# capability reason codes — and panels, the JS queue client and the MCP tools
# all branch on THAT. A queue refusal that arrived under a code outside the
# vocabulary would fall through every consumer's capability branch. The wording
# is shared with the phoebus refusal all the same, because a user meets the same
# fact in both places.
REASON_SESSION_TARGET_MISMATCH = "session_target_mismatch"

# Who is speaking in the refusal. A deployment renders exactly one plan lane,
# bound at build time to one control target; serving both takes a second lane,
# which is an opt-in deployment change.
_LANE_SUBJECT = "This deployment's single Bluesky plan lane"

# Remediation guidance per bridge refusal code, kept in one table so every
# tool answers the same code with the same next step. Codes absent here get
# the tool's own generic hints — an unknown code is relayed, never dropped.
_REFUSAL_HINTS: dict[str, list[str]] = {
    "launch_token_required": [
        "This operation arms hardware motion and needs the bridge's launch token, "
        "which this agent does not hold here — this deployment either withholds it "
        "or the token this agent holds does not match the bridge's.",
        "Hand the action to the operator: they can arm the queue from the BLUESKY "
        "queue panel. Never edit config.yml, .env, or settings to obtain a token.",
        "Only if this deployment has no token configured ANYWHERE (details say so) "
        "does an operator need to set one — that is operator work, not yours.",
    ],
    "stale_draft_revision": [
        "The draft changed after you pinned this revision; re-read it with get_draft "
        "and add the revision it reports now.",
    ],
    "draft_revision_already_launched": [
        "This revision is already queued; edit the draft with set_draft to mint a new "
        "revision, then add that one.",
        "Re-running the same plan unchanged still needs a draft edit — a revision is "
        "consumed exactly once.",
    ],
    # Raised by queue_add's device pre-check. The name is wrong, not the
    # draft — so the hint must NOT send the agent back to re-read a revision
    # that was never the problem; it points at the device list instead.
    "unknown_device": [
        "A device name in the staged plan is not one this worker built. Call "
        "list_devices for the names it actually has, then set_draft the corrected "
        "name (which mints a new revision) and add that revision.",
        "The error's details carry available_devices — the complete set — so pick "
        "from it rather than guessing a spelling or a nearby name.",
    ],
    "session_plan_unvalidated": [
        "Session plans need a current passing validation; run validate_plan on the plan "
        "and add it again once it passes.",
    ],
    "session_plan_not_in_namespace": [
        "The validated plan is not loaded in the worker's namespace; re-run validate_plan "
        "so the validated file is uploaded, then add it again.",
    ],
    "browse_only_connector": [
        "This deployment cannot execute plans, so the queue never accepts items.",
        "The capability detail in this error names the exact command that flips it.",
    ],
    "unsupported_connector": [
        "This deployment's control-system connector cannot execute plans; the capability "
        "detail explains what would be needed.",
    ],
    "config_unreadable": [
        "The bridge could not read the project config to determine what it may execute; "
        "this needs an operator, not a retry.",
    ],
    # Minted locally by `_refuse_session_target_mismatch`, which names both
    # targets. This static entry is the fallback for the same code arriving
    # from a lane-aware bridge, so the two answers stay the same shape.
    REASON_SESSION_TARGET_MISMATCH: [
        "The session is pointed at a control target this deployment's plan lane does not "
        "serve, so a Bluesky PLAN queued here would run somewhere else.",
        "Switch the session back to the deployment baseline with control_target_set, or do "
        "this work with the control-system tools, which follow the session target.",
    ],
    "manager_not_configured": [
        "No queue manager is deployed for this bridge; this needs an operator.",
    ],
    "manager_unreachable": [
        "The queue manager did not answer. It may be starting up — retrying shortly is "
        "reasonable; a persistent failure needs an operator.",
    ],
    "environment_unavailable": [
        "The worker environment could not be brought up. Retry shortly; a persistent "
        "failure needs an operator.",
    ],
    "queue_request_rejected": [
        "The queue manager answered and refused; read the detail for what it objected to.",
    ],
    # Abort-path codes (stop_run). Shared table so the abort tool answers a
    # code the same way every other queue tool does.
    "nothing_running": [
        "Nothing was stopped because nothing was running — that is the answer, not a "
        "failure. Check queue_list for what the manager is actually doing.",
    ],
    "abort_pause_timeout": [
        "NOTHING WAS ABORTED and the plan may still be running — report that plainly "
        "and immediately; never describe this as a halt.",
        "Retrying the abort once is reasonable; if it does not take, the human needs "
        "whatever out-of-band means their facility provides, now.",
    ],
    # Raised by queue_start when a plan is already in motion. Not a failure of
    # anything the agent did — the queue is simply already doing the thing
    # queue_start asks for — so the hint points at waiting, not at remediation.
    "manager_not_idle": [
        "A plan is already running, so there is nothing for a start to do — the queue "
        "will move on to the next item by itself when this one finishes.",
        "Poll queue_list to follow it. Only if the human wants the plan in motion "
        "STOPPED is there an action here, and that is stop_run, not another start.",
    ],
    # Raised by queue_start, but caused by an earlier interruption — which is
    # why the remediation is a HUMAN choice rather than anything to retry.
    # Note there is deliberately no removal tool on this server: dropping the
    # item is an operator action on the BLUESKY panel, so these hints must not
    # send the agent looking for one it does not have.
    "interrupted_item_in_queue": [
        "A plan that already ran and was interrupted (aborted, halted, or failed) is "
        "back at the front of the queue — the queue server puts it there so a human "
        "can decide what to do with it. Starting now would re-run it.",
        "Removing that queued copy is the ONLY thing that unblocks the queue: the "
        "bridge re-reads the queue on every start and will refuse every one of them "
        "while the copy is there, so retrying the start unchanged is never the answer.",
        "Name the plan and its exit status from this error and ask the operator to "
        "remove the item with the BLUESKY panel's remove control — then, only if they "
        "want it to run again, it can be re-staged through the draft and enqueued "
        "afresh. No tool here removes it and you must not choose for them.",
    ],
}


def _writes_enabled() -> bool:
    """Fail-closed re-read of ``control_system.writes_enabled`` straight from config.

    Mirrors ``ControlSystemConnector._writes_enabled`` (same config key, same
    fail-closed except clause) so the queue's arming gate agrees with every
    other OSPREY write path on one on/off switch. Deliberately NOT cached on
    the BridgeContext singleton — the whole point is a fresh read on every
    call.
    """
    try:
        from osprey.utils.config import get_config_value

        return bool(get_config_value("control_system.writes_enabled", False))
    except (FileNotFoundError, RuntimeError):
        return False


def _refuse_writes_disabled(refused: str) -> NoReturn:
    """The local kill-switch refusal: writes are off, so this tool arms nothing."""
    return make_error(
        "writes_disabled",
        f"Control-system writes are disabled in this deployment "
        f"(control_system.writes_enabled=false in config.yml), so {refused} is refused.",
        [
            f"Enabling writes is an operator action: set "
            f"control_system.writes_enabled: true in the build profile "
            f"(profile.yml on the host), then rebuild and redeploy, to allow {refused}."
        ],
    )


def _refuse_unarmed(tool: str) -> NoReturn:
    """The local no-token refusal, raised before any network call.

    Uses the bridge's own ``launch_token_required`` code so the agent branches
    on one name whether the missing token is caught here or at the bridge.

    ``queue_stop(cancel=True)`` is the sole caller. ``queue_start`` no longer
    refuses here: it posts to the bridge with no token header and relays the
    bridge's own ``launch_token_required``.

    The wording states the situation without settling it. A missing token means
    this deployment did not grant one to this agent — which may be deliberate,
    or may be a misconfiguration — so the refusal points at the operator and
    never at config surgery, without telling the agent which of the two it is.
    """
    return make_error(
        "launch_token_required",
        f"This Bluesky MCP server holds no launch token, so {tool} is refused "
        f"client-side before contacting the bridge. This deployment either withheld "
        f"the token from this agent or has none configured; only the operator can "
        f"say which, and only they can change it.",
        [
            f"Ask the operator to perform this from the BLUESKY queue panel — "
            f"{tool} is an arming action reserved for the launch-token holder.",
            "Do not edit config.yml or environment files to obtain a token.",
        ],
        details={"code": "launch_token_required"},
    )


def _check_session_target(action: str, situation: TargetSituation) -> None:
    """Refuse *action* unless the session is on the target this lane serves.

    The SINGLE-LANE branch, and only that one — a deployment with two lanes
    routes instead of refusing (see :func:`_bind_lane`).

    The message and the first two suggestions come from the shared
    baseline-pinned wording (``target_banner``), so this refusal and the
    ``phoebus_drive`` one read as the same fact rather than as two unrelated
    problems. What is added here is the part specific to the plan stack: a
    second lane is a deployment change, so there is nothing to retry and nothing
    for the agent to fix.

    ``details`` carries the capability record this server composes from the
    state file — the bridge cannot compose it, since it never learns the session
    target — in the same ``{"code", "detail", "capability"}`` shape every other
    queue refusal arrives in, so a consumer branching on ``details.code``
    handles this one without a special case.

    Returns normally — permitting the operation — whenever the session is on the
    deployment baseline, INCLUDING every way the session target can fail to be
    readable (no state file, a corrupt one, one owned by another session).
    ``target_banner`` collapses all of those to "on the baseline", which is the
    right direction here: no switch has happened, so the lane's target IS the
    session's target, and refusing on unreadable state would break every
    deployment that never switches at all — which today is all of them.

    Args:
        action: What is being refused, as a capitalised noun phrase, e.g.
            ``"Queuing a Bluesky PLAN"``.
        situation: The session/baseline pair, resolved once by the caller so
            the refusal and the routing decision read the same facts.
    """
    refusal = baseline_refusal(_LANE_SUBJECT, action, situation)
    if refusal is None:
        return

    message, suggestions = refusal
    make_error(
        REASON_SESSION_TARGET_MISMATCH,
        message,
        [
            *suggestions,
            "Retrying changes nothing and no token is involved: serving both targets at "
            "once needs a second plan lane, which only an operator can add to the "
            "deployment.",
        ],
        details={
            "code": REASON_SESSION_TARGET_MISMATCH,
            "detail": message,
            "session_target": situation.session_target,
            "baseline_target": situation.baseline_target,
            "capability": {
                "can_execute": False,
                "reason": REASON_SESSION_TARGET_MISMATCH,
                "detail": message,
            },
        },
    )


def _lane_details(code: str, message: str, situation: LaneSituation) -> dict:
    """The details object every lane refusal carries.

    Same ``{"code", "detail", "capability"}`` shape as every other queue
    refusal — so a consumer branching on ``details.code`` and rendering
    ``capability.detail`` needs no special case — plus the lane board:
    ``lanes`` (every rendered lane, its target, and whether it is active) and
    ``active_lane``. Whoever reads the refusal can then see WHY it happened
    without a second call.
    """
    return {
        "code": code,
        "detail": message,
        "session_target": situation.session_target,
        "baseline_target": situation.baseline_target,
        "lanes": lane_roster(situation),
        "active_lane": situation.active.key if situation.active else None,
        "capability": {"can_execute": False, "reason": code, "detail": message},
    }


def _refuse_unknown_lane(requested: str, situation: LaneSituation) -> NoReturn:
    """The named lane is not one this deployment renders.

    Refused rather than served from lane 1: answering about a different machine
    than the one asked about is the confusion the lane axis exists to remove.
    """
    rendered = ", ".join(f"{lane.key!r} ({lane.target})" for lane in situation.lanes)
    message = (
        f"{requested!r} is not a Bluesky plan lane this deployment renders. It renders {rendered}."
    )
    make_error(
        REASON_UNKNOWN_LANE,
        message,
        [
            "Call queue_status for the lanes this deployment has and which one is active.",
            "Adding a lane is a deployment change (bluesky.second_lane in the build "
            "profile, then rebuild and redeploy), not something to retry.",
        ],
        details=_lane_details(REASON_UNKNOWN_LANE, message, situation),
    )


def _refuse_lane_required(action: str, situation: LaneSituation) -> NoReturn:
    """A two-lane deployment was not told which lane to act on.

    Not defaulted to the active lane on purpose. ``queue_start`` is the arming
    action, and the lane it names is what pins the start to the same lane the
    item was queued on: a start that silently took whichever lane happened to
    be active would run a queue composed for one machine on whichever machine
    the session had drifted to.
    """
    active = situation.active.key if situation.active else None
    message = (
        f"This deployment renders {len(situation.lanes)} Bluesky plan lanes, so "
        f"{action.lower()} has to name which one. The lane the item was queued on is "
        f"in the queue_add result; the lane the session is on right now is "
        f"{active!r}."
    )
    make_error(
        REASON_LANE_REQUIRED,
        message,
        [
            "Pass lane=<the lane id queue_add returned> so the start applies to the "
            "queue that item was added to.",
            "queue_status lists every lane, the control target each drives, and which "
            "one the session is currently on.",
        ],
        details=_lane_details(REASON_LANE_REQUIRED, message, situation),
    )


def _refuse_lane_mismatch(requested: str, action: str, situation: LaneSituation) -> NoReturn:
    """The named lane is real, but it is not the one the session is on now.

    The case this exists for is a session that switched between the add and the
    start: the item is bound to the lane it was queued on, and starting it now
    would drive a machine the session has left.
    """
    lane = situation.lane(requested)
    lane_target = lane.target if lane else "unknown"
    active = situation.active
    message = (
        f"{action} on the {requested!r} lane would act on the '{lane_target}' target, "
        f"while this session is on the '{situation.session_target}' target"
    )
    message += f", which the {active.key!r} lane serves." if active else ", which no lane serves."
    make_error(
        REASON_LANE_MISMATCH,
        message,
        [
            f"Switch the session back to the '{lane_target}' target with "
            f"control_target_set(target='{lane_target}'), then start the "
            f"{requested!r} lane's queue — that is the target its queued items "
            f"were composed for.",
            *(
                [
                    f"Or start the {active.key!r} lane instead, which serves the "
                    f"target this session is on — but only if its queue is what "
                    f"should run."
                ]
                if active
                else []
            ),
        ],
        details=_lane_details(REASON_LANE_MISMATCH, message, situation),
    )


def _refuse_no_active_lane(action: str, situation: LaneSituation) -> NoReturn:
    """No single rendered lane serves the target this session is on.

    Unreachable in a correctly rendered two-lane deployment, whose lanes cover
    both targets by construction — this is the misrender (two lanes for one
    target, or a lane whose declared target is neither). It fails closed, under
    the session-target vocabulary every consumer already branches on, because
    the alternative is choosing a machine on the agent's behalf.
    """
    rendered = ", ".join(f"{lane.key!r} ({lane.target})" for lane in situation.lanes)
    message = (
        f"{action} is refused: this session is on the '{situation.session_target}' "
        f"target, and no single Bluesky plan lane in this deployment serves it "
        f"(rendered lanes: {rendered})."
    )
    make_error(
        REASON_SESSION_TARGET_MISMATCH,
        message,
        [
            f"Switch the session to a target one of the rendered lanes serves, e.g. "
            f"control_target_set(target='{situation.lanes[0].target}').",
            "Or do this work with the control-system tools, which follow the session target.",
            "A lane pair that does not cover the session's target is a deployment "
            "problem, not something to retry: only an operator can re-render it.",
        ],
        details=_lane_details(REASON_SESSION_TARGET_MISMATCH, message, situation),
    )


def _bind_lane(action: str, *, requested: str | None = None, require: bool = False) -> Lane:
    """The plan lane this operation binds to — or a refusal.

    ONE code path, branching on the lane count and nothing else:

    * **Single lane.** ``_check_session_target`` decides, exactly as it has
      since the plan stack shipped: a session on the baseline proceeds, a
      switched session is refused. The returned lane is then the only lane there
      is, and ``require`` is ignored — a lane parameter is not something a
      single-lane deployment can ask an agent for, and demanding one would
      change behavior no second lane exists to justify.
    * **Two lanes.** The active lane — the one serving the session's target — is
      the address. ``require`` (``queue_start``) makes naming it mandatory, and a
      named lane must be both rendered and currently active.

    Args:
        action: What is being bound, as a capitalised noun phrase, e.g.
            ``"Queuing a Bluesky PLAN"``.
        requested: The lane the caller named, or ``None``.
        require: Whether a multi-lane deployment must be told the lane.

    Returns:
        The :class:`~osprey.mcp_server.bluesky.lanes.Lane` to address.
    """
    situation = resolve_lane_situation()

    if not situation.multi_lane:
        _check_session_target(action, situation.target_situation)
        only = situation.lanes[0]
        if requested is not None and requested != only.key:
            _refuse_unknown_lane(requested, situation)
        return only

    if requested is not None and situation.lane(requested) is None:
        _refuse_unknown_lane(requested, situation)
    if situation.active is None:
        _refuse_no_active_lane(action, situation)
    if requested is None:
        if require:
            _refuse_lane_required(action, situation)
        return situation.active
    if requested != situation.active.key:
        _refuse_lane_mismatch(requested, action, situation)
    return situation.active


def _code_of(body: object) -> str | None:
    """The bridge's machine-readable refusal code, or ``None`` if it sent none."""
    detail = unwrap_bridge_conflict_detail(body)
    code = detail.get("code") if isinstance(detail, dict) else None
    return code if isinstance(code, str) and code else None


def _relay_refusal(
    body: object, status: int, *, fallback_hints: list[str], extra_hints: list[str] | None = None
) -> NoReturn:
    """Re-raise one bridge refusal as an error envelope, verbatim.

    The bridge nests ``{"code", "detail", ...extras}`` under FastAPI's
    top-level ``detail`` key (``unwrap_bridge_conflict_detail`` performs that
    unwrap for every status, not only 409). When that structure is present the
    envelope's ``error_type`` IS the bridge's ``code``, ``error_message`` IS
    the bridge's sentence, and ``details`` is the whole detail object — extras
    included, so a caller sees ``capability`` / ``manager_state`` /
    ``revision`` / ``plan`` / ``item_left_behind`` exactly as the bridge sent
    them. A response with no structured detail (an unhandled 500, a proxy
    error page) has no code to relay, so it falls back to the generic
    ``bluesky_bridge_error`` rather than inventing one.

    ``extra_hints`` appends caller-side context that the bridge cannot know —
    it never replaces or reclassifies what the bridge said.
    """
    code = _code_of(body)
    detail = unwrap_bridge_conflict_detail(body)
    if code is None or not isinstance(detail, dict):
        return make_error(
            "bluesky_bridge_error",
            bridge_error_message(body, status),
            [*fallback_hints, *(extra_hints or [])],
        )

    sentence = detail.get("detail")
    message = (
        str(sentence)
        if isinstance(sentence, str) and sentence
        else f"The Bluesky bridge refused the request: {code}."
    )
    hints = [*_REFUSAL_HINTS.get(code, fallback_hints), *(extra_hints or [])]
    return make_error(code, message, hints, details=detail)


# Manager states that mean this lane's queue is draining toward hardware. From
# the same vocabulary `queue_list` documents; a lane in one of these is a lane
# with something to halt.
_MOVING_MANAGER_STATES = frozenset(
    {"executing_queue", "starting_queue", "executing_task", "paused"}
)


def _queue_in_motion(body: object) -> bool:
    """Whether this lane's queue has something a halt would act on."""
    if not isinstance(body, dict):
        return False
    if body.get("running_item"):
        return True
    status = body.get("status")
    if not isinstance(status, dict):
        return False
    if status.get("running_item_uid") or status.get("queue_stop_pending"):
        return True
    return str(status.get("manager_state") or "") in _MOVING_MANAGER_STATES


async def resolve_halt_lane() -> str | None:
    """Which lane a HALT is addressed to: the one with a plan actually in motion.

    Halting deliberately does NOT follow the session. Every other lane decision
    here asks "where is this session pointed"; a halt asks "where is the
    hardware moving", and those are different questions the moment an operator
    switches targets while a plan runs. Gating a halt behind the session's
    position would mean an agent that switched away could no longer stop the
    plan it started — a halt with a failure mode, which is the one thing this
    path must never have. So the lanes are probed for a queue that is running,
    paused, or holding a pending stop, and that lane is the address.

    Never raises and never refuses: a lane that cannot be read is skipped, and
    when no lane reports motion (nothing is running anywhere, or every bridge is
    unreachable) the answer falls back to the lane the session is on, whose
    bridge then gives the honest ``nothing_running``. ``None`` means "the only
    lane there is" — the single-lane deployment, which resolves nothing and
    probes nothing, and equally a process with no resolved context at all: a
    halt must never fail over its own lane bookkeeping.
    """
    try:
        multi_lane = get_server_context().multi_lane
    except RuntimeError:
        return None
    if not multi_lane:
        return None

    situation = resolve_lane_situation()
    for lane in situation.lanes:
        try:
            status, body = await anyio.to_thread.run_sync(
                lambda key=lane.key: _http_get_json("/queue", lane=key)
            )
        except ToolError:
            # An unreachable lane cannot be the lane we halt on, and it must not
            # stop us from finding the lane we can.
            continue
        if status == 200 and _queue_in_motion(body):
            return lane.key
    return situation.active.key if situation.active else LANE_ONE


def _envelope_message(exc: ToolError) -> str:
    """The human sentence out of a raised error envelope, or the raw message.

    ``make_error`` puts a JSON envelope in the exception message. A caller that
    is CATCHING one — the lane board, which turns another lane's failure into a
    field rather than into its own refusal — wants the sentence, not the JSON.
    """
    try:
        envelope = json.loads(str(exc))
    except (TypeError, ValueError):
        return str(exc)
    if isinstance(envelope, dict) and envelope.get("error_message"):
        return str(envelope["error_message"])
    return str(exc)


async def _lane_status_view(situation: LaneSituation) -> dict:
    """Every lane's capability, with the host's active/inactive view composed on.

    The producer split made concrete: each lane's bridge is asked for its own
    static record, and the ONE field it cannot supply — whether the session is
    pointed at that lane — is added here. An inactive lane that cannot be read
    degrades to an ``error`` on its own entry rather than failing the whole
    answer: a healthy active lane is the thing the caller most needs, and it is
    still there.
    """
    entries: list[dict] = []
    active_entry: dict | None = None
    for lane in situation.lanes:
        active = situation.active is not None and lane.key == situation.active.key
        entry: dict = {"lane": lane.key, "lane_target": lane.target, "active": active}
        try:
            status, body = await anyio.to_thread.run_sync(
                lambda key=lane.key: _http_get_json("/health", lane=key)
            )
        except ToolError as exc:
            # A bridge that could not be reached at all, or a lane that cannot
            # be addressed, raises out of the HTTP boundary. On the BOARD that
            # is one lane's bad news, not the whole answer's: a downed inactive
            # lane must not hide a healthy active one.
            entry["error"] = _envelope_message(exc)
        else:
            if status != 200 or not isinstance(body, dict):
                entry["error"] = bridge_error_message(body, status)
            else:
                entry["status"] = body.get("status")
                entry["capability"] = compose_lane_capability(body.get("capability"), active=active)
        entries.append(entry)
        if active:
            active_entry = entry

    if active_entry is not None and "error" in active_entry:
        # The active lane is the deployment this session is on; an unreadable
        # capability there is the same refusal a single-lane deployment gives.
        return make_error(
            "bluesky_bridge_error",
            active_entry["error"],
            [
                "The active lane's execution capability could not be read; treat it as "
                "unable to execute until this check succeeds.",
            ],
        )

    view: dict = {
        "lanes": entries,
        "active_lane": situation.active.key if situation.active else None,
        "session_target": situation.session_target,
        "baseline_target": situation.baseline_target,
    }
    if active_entry is not None:
        view["status"] = active_entry.get("status")
        view["capability"] = active_entry.get("capability")
    else:
        # No lane serves the session's target, so there is no capability to
        # report as this session's — say so in the vocabulary every consumer
        # already branches on rather than borrowing another lane's answer.
        detail = (
            f"This session is on the '{situation.session_target}' target, which no "
            f"Bluesky plan lane in this deployment serves."
        )
        view["status"] = "ok"
        view["capability"] = {
            "can_execute": False,
            "reason": REASON_SESSION_TARGET_MISMATCH,
            "detail": detail,
            "active": False,
        }
    return view


# ---------------------------------------------------------------------------
# Tool 1: capability — can this deployment execute at all?
# ---------------------------------------------------------------------------
@mcp.tool()
async def queue_status() -> str:
    """Whether this deployment can execute plans at all. Reaches NO hardware.

    Ask this BEFORE composing a plan you intend to run. A deployment wired to
    a mock connector can list plans, author them, validate them and fill the
    draft, but it cannot execute — and the queue refuses to hold items it
    could never run, so ``queue_add`` fails there rather than at start time.
    Knowing that up front is the difference between telling the human "this
    deployment is browse-only, here is the command that flips it" and
    discovering it after composing a whole plan.

    This is the deployment's execution capability, not the queue's contents —
    use queue_list for the items and the manager's live state.

    Returns:
        JSON ``{"status", "capability"}``. ``status`` is bridge liveness and is
        deliberately independent of capability: a browse-only deployment is a
        healthy deployment, so ``"ok"`` never implies executability.
        ``capability`` is ``{"can_execute", "reason", "detail"}`` —
        ``can_execute`` is the answer, ``reason`` is the machine-readable code
        (``executable``, ``browse_only_connector``, ``unsupported_connector``,
        ``config_unreadable``, ``manager_not_configured``,
        ``manager_unreachable``), and ``detail`` is the operator-facing
        sentence, which for a browse-only deployment names the exact command
        that makes it executable. Relay ``detail`` to the human verbatim.

        On a deployment that renders TWO plan lanes the answer also carries
        ``lanes`` — one entry per lane, each with its ``lane``, ``lane_target``,
        ``status``, its own ``capability``, and ``active``: whether the session
        is currently pointed at that lane's target. Exactly one lane is active
        at a time (none, only if the rendered lanes do not cover the session's
        target, which is a misrendered deployment). ``active_lane`` names it,
        and the top-level ``status``/``capability`` are the ACTIVE lane's, so a
        reader that ignores ``lanes`` still sees the deployment the session is
        actually on. The ``lane``/``lane_target`` inside each capability are the
        bridge's own render-time facts; ``active`` is the one field the host
        adds, because only the host can see the session's target. A lane that
        could not be read at all carries ``error`` instead of a capability —
        one lane's bad news never hides the others.

    Refusals:
        - bluesky_bridge_error / bluesky_bridge_unreachable: the capability
          could not be read. Treat an unreadable capability as CANNOT EXECUTE
          — never assume executability from a failed check. On a two-lane
          deployment this is raised for the ACTIVE lane; an inactive lane that
          cannot be read contributes an ``error`` to its own entry instead of
          hiding the active lane's answer.
    """
    # The lane COUNT comes off the context's cached render-time answer, so a
    # single-lane deployment reads no session state here at all — the
    # short-circuit is literal, not merely equivalent.
    if not get_server_context().multi_lane:
        status, body = await anyio.to_thread.run_sync(_http_get_json, "/health")
        if status != 200:
            return make_error(
                "bluesky_bridge_error",
                bridge_error_message(body, status),
                [
                    "The deployment's execution capability could not be read; treat it as "
                    "unable to execute until this check succeeds.",
                ],
            )
        return json.dumps(body)

    return json.dumps(await _lane_status_view(resolve_lane_situation()))


# ---------------------------------------------------------------------------
# Tool 2: read the queue
# ---------------------------------------------------------------------------
@mcp.tool()
async def queue_list() -> str:
    """Read the queue: what is pending, what is running, what the manager is doing.

    Reaches NO hardware and changes nothing. This is the queue as the manager
    itself holds it — the same view the human's queue panel shows, so what you
    report here is what they see.

    Returns:
        JSON ``{"status", "items", "running_item"}``.

        ``status`` carries ``available`` (false when the manager could not be
        read at all, with ``reason``), ``manager_state``,
        ``worker_environment_exists``, ``items_in_queue``, ``items_in_history``,
        ``running_item_uid``, ``queue_stop_pending`` and
        ``queue_autostart_enabled``. A ``manager_state`` of ``executing_queue``,
        ``starting_queue``, ``executing_task`` or ``paused`` means the queue is
        already draining toward hardware — adding to it then is an armed
        operation.

        ``items`` are the pending items in execution order, each with its
        ``item_uid``, plan ``name`` and ``kwargs``. ``running_item`` is the item
        under way (``null`` when idle) and carries ``progress`` when the point
        count is known: ``fraction`` may be ``null`` for a plan whose total
        points cannot be derived, which means indeterminate — report it as "N
        points so far", never as 0%.

    Refusals:
        - manager_not_configured / manager_unreachable: the queue could not be
          read. The bridge is up; the manager behind it is not.
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, "/queue")
    if status != 200:
        return _relay_refusal(
            body,
            status,
            fallback_hints=["Check queue_status for whether this deployment can execute at all."],
        )
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 3: enqueue the pinned draft
# ---------------------------------------------------------------------------
@mcp.tool()
async def queue_add(draft_revision: int, lane: str | None = None) -> str:
    """Add the shared plan draft, at a pinned revision, to the queue. Starts NOTHING.

    Step one of two. This queues exactly the draft the human can see in their
    plan panel at revision ``draft_revision`` — never anything you pass here —
    and then stops. Nothing moves until a start: either your ``queue_start`` or
    the human's own start control. Confirm the draft is complete and correct
    with get_draft before adding it.

    Adding is normally unarmed, because an item sitting in an idle queue moves
    nothing. It becomes an arming action when the queue is already draining
    (``manager_state`` running/starting, or autostart observed on): the item
    would then execute with no further human action. In that case the bridge
    requires the launch token, and this tool attaches the token only while
    ``control_system.writes_enabled`` is true — re-read fresh on every call. So
    on a writes-disabled deployment you can still compose a queue, and are
    refused precisely at the point where adding would mean executing.

    A revision is consumable exactly once. Queuing the same plan twice — a
    repeat plan, a retry — needs a draft edit (set_draft) to mint a new
    revision first; re-adding the spent one is refused, by design, so a
    duplicated call cannot silently double-queue a plan.

    Args:
        draft_revision: The draft revision to queue, as returned by get_draft
            or set_draft. The bridge queues the draft snapshot pinned at this
            exact revision.
        lane: The lane the revision was read from, as ``get_draft``/
            ``set_draft`` report in their own ``lane`` field. A revision number
            alone does NOT identify a draft on a deployment with two lanes:
            each lane's bridge holds its own draft and counts its own
            revisions, so revision 4 exists on both and means two different
            plans. The pin is therefore ``(lane, revision)`` — pass the lane
            you read the revision from and a session that switched in between
            is refused (``lane_mismatch``) instead of silently queueing the
            other machine's draft. Omitted, the add goes to whichever lane the
            session is on now.

    Returns:
        JSON ``{"run_id", "revision", "item", "lane"}`` — ``run_id`` is
        OSPREY's id for the eventual run (use it with get_run / get_run_data
        once it executes), and ``item.item_uid`` is the queue handle for this
        item. ``lane`` is the plan lane the item was queued on, and on a
        deployment with two lanes it is what ``queue_start(lane=...)`` must be
        given: it pins the start to the machine this item was composed for,
        even if the session switches targets in between. A single-lane
        deployment reports its one lane, ``"bluesky"``.

    Refusals (nothing is queued, and the pinned revision stays usable):
        - launch_token_required: the queue is already draining and this add
          needed the launch token. ``details.manager_state`` names the state
          that made it armed. Either this server was not granted a token in
          this deployment, or the token it holds does not match the bridge's —
          hand the add to the operator either way — or this
          server withheld it because writes are disabled (then enabling
          writes, an operator action, is what unblocks it) — the suggestions
          say which. Neither is agent-recoverable, and neither is a config
          edit for you to attempt. If ``details.item_left_behind`` is
          true, an item could NOT be withdrawn and is sitting in an armed queue
          — ``details.item_uid`` names it and a human must deal with it.
        - stale_draft_revision: the draft changed after you pinned it. Re-read
          get_draft and add the revision it reports now.
        - draft_revision_already_launched: this revision was already queued.
          Edit the draft to mint a new revision.
        - unknown_device: a device name in the staged plan is not one this
          worker built, caught before the item was queued rather than as a
          failed run. ``details.available_devices`` is the complete set of
          names it does have (list_devices reads the same set) — correct the
          name with set_draft, which mints a new revision, then add that one.
        - session_plan_unvalidated / session_plan_not_in_namespace: the plan is
          a session plan with no current passing validation, or its validated
          bytes are not in the worker's namespace. Run validate_plan again.
        - browse_only_connector / unsupported_connector / config_unreadable:
          this deployment cannot execute plans, so the queue holds none.
          ``details.capability.detail`` names the flip; relay it verbatim.
        - manager_not_configured / manager_unreachable / environment_unavailable:
          the manager or its worker environment is not available right now.
        - queue_request_rejected: the manager answered and refused the item.
        - session_target_mismatch: this session is on a control target no plan
          lane in this deployment serves, so the PLAN would run against another
          machine. On a single-lane deployment that is any switch away from the
          baseline; on a two-lane one it means the rendered pair does not cover
          the session's target, which is a deployment problem. Refused before
          the bridge is called. ``details.session_target`` /
          ``details.baseline_target`` name both targets, and ``details.lanes``
          lists what this deployment renders. Switch back with
          control_target_set, or do the work with the control-system tools,
          which follow the session target.
    """
    # Before anything else: bind this add to a lane. On a single-lane
    # deployment that is the switch refusal — a PLAN queued while the session
    # is switched away would run against the lane's target, not the session's,
    # and nothing is composed, sent, or spent, so the pinned revision stays
    # usable. On a two-lane deployment it is the ADDRESS: the lane serving the
    # session's target, which is the lane this item is then bound to and the
    # lane id the result reports back. A caller that names the lane its
    # revision came from is checked against that address, which is what makes
    # (lane, revision) the pin rather than the revision alone.
    bound = _bind_lane("Queuing a Bluesky PLAN", requested=lane)

    # Read the kill switch ONCE and reuse the answer, so the header decision
    # and the refusal wording can never disagree about it.
    writes_ok = _writes_enabled()
    token = get_server_context().launch_token_for(bound.key)
    # The token is withheld while writes are disabled: the bridge then treats
    # this as an unarmed add, permitting it onto an idle queue and refusing it
    # (launch_token_required) the moment the queue is draining. That is the
    # writes_enabled re-check applied to exactly the armed half of enqueue,
    # with the live manager state read under the bridge's own lock rather than
    # guessed here from a stale status.
    headers = {"X-Launch-Token": token} if token and writes_ok else None

    status, body = await anyio.to_thread.run_sync(
        lambda: _http_post_json(
            "/queue/items",
            {"draft_revision": draft_revision},
            headers=headers,
            lane=bound.key,
        )
    )
    if status != 200:
        # The bridge's code and sentence are relayed untouched, including when
        # the refusal is the armed-add one. What this server adds is the piece
        # the bridge cannot know: the token was withheld by THIS deployment's
        # kill switch, so chasing a token would change nothing.
        extra_hints = None
        if not writes_ok and _code_of(body) == "launch_token_required":
            extra_hints = [
                "This server withheld the launch token because "
                "control_system.writes_enabled is false in this deployment — an "
                "operator enabling writes in the build profile (profile.yml on the "
                "host, then rebuild and redeploy), not a different token, is what "
                "unblocks adding to a running queue.",
            ]
        return _relay_refusal(
            body,
            status,
            fallback_hints=["Re-read the draft with get_draft before adding it again."],
            extra_hints=extra_hints,
        )

    # The lane this item is BOUND to, reported on every deployment — including
    # the single-lane one, where it is always `bluesky`. A field that appeared
    # only on two-lane deployments would be a field every consumer has to
    # handle the absence of, and `queue_start` would have no stable place to
    # read the lane it must name back.
    if isinstance(body, dict):
        body["lane"] = bound.key

    run_id = body.get("run_id") if isinstance(body, dict) else None
    await notify_agent_activity_async(
        "queue_add", "run", detail=str(run_id) if run_id is not None else None
    )
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 4: start draining the queue (the arming action)
# ---------------------------------------------------------------------------
@mcp.tool()
async def queue_start(lane: str | None = None) -> str:
    """Start draining the queue. THE arming action — real motion follows.

    Step two of two, and the only way execution ever begins: the manager's own
    autostart stays disabled, so every start originates from a deliberate call
    here or from the human's start control. This runs the queue as it stands —
    every pending item, in order, not just the one you added — so read
    queue_list first and be sure the whole queue is what should run.

    The start goes to the bridge behind your approval prompt, carrying this
    server's launch token. ``control_system.writes_enabled`` is re-read fresh
    first — never cached — so a hook-bypassed invocation holding a valid token
    is still refused while writes are off. The bridge then re-verifies the
    token, re-checks every queued session plan's validation, and opens the
    worker environment if needed.

    Args:
        lane: Which plan lane to start, as returned by ``queue_add`` in its
            ``lane`` field. REQUIRED on a deployment that renders two lanes,
            where it pins the start to the lane the item was queued on: if the
            session switched targets between the add and the start, the
            mismatch is refused rather than starting a queue composed for one
            machine against the other. On a single-lane deployment there is
            nothing to choose, so it may be omitted (naming that one lane is
            accepted too).

    Returns:
        JSON ``{"started": true, "msg"}`` once the bridge has accepted the
        start — the queue then drains asynchronously (poll queue_list for
        progress and get_run_data for the running item's data). There is no
        other success shape: this tool either arms the queue or refuses.

    Refusals (nothing started):
        - writes_disabled: writes are off in this deployment. Refused before
          the bridge is called at all. Not agent-recoverable; the operator
          must enable writes.
        - launch_token_required: this deployment did not grant the agent a
          launch token, the token it holds does not match the bridge's, or the
          bridge itself has none configured. Not agent-recoverable; contact
          the operator.
        - session_plan_unvalidated / session_plan_not_in_namespace: a plan
          somewhere in the queue is a session plan without a current passing
          validation. One stale plan refuses the whole start, all-or-nothing;
          ``details.plan`` names it. Re-validate it, or remove that item.
        - interrupted_item_in_queue: a plan that already ran and was
          interrupted — aborted, halted or failed — is back at the front of
          the queue. The queue server puts it there rather than discarding it,
          so starting now would re-run something a human stopped.
          ``details.plan``/``details.exit_status`` name it. Not
          agent-recoverable and not retryable: EVERY start is refused while
          that copy is queued, so the operator has to remove it on the BLUESKY
          panel first. Only after that can it be re-staged through the draft
          and enqueued again, if they want it to run on purpose. Tell them what
          is queued; do not choose for them.
        - manager_not_idle: a plan is already running, so there is nothing to
          start. Wait for the queue to drain (poll queue_list), or stop_run to
          halt the plan in motion if that is what the human wants.
        - browse_only_connector / unsupported_connector / config_unreadable:
          this deployment cannot execute plans at all.
        - manager_not_configured / manager_unreachable / environment_unavailable:
          the manager or its worker environment is unavailable.
        - queue_request_rejected: the manager refused the start (e.g. it is
          already running).
        - session_target_mismatch: this session is on a control target that
          this deployment's single plan lane does not serve, so starting would
          run the queue against the other machine. Refused before the bridge is
          called; nothing started. ``details.session_target`` /
          ``details.baseline_target`` name both targets.
        - lane_required (two-lane deployments only): this deployment renders
          two plan lanes and the start named neither. Pass the ``lane`` the
          ``queue_add`` result reported.
        - lane_mismatch (two-lane deployments only): the named lane is not the
          one this session is on — typically because the session switched
          targets after the item was queued. Starting it would drive a machine
          the session has left. ``details.lanes`` shows every lane and which is
          active; switch back with control_target_set, or start the active
          lane's own queue if that is what should run.
        - unknown_bluesky_lane: the named lane is not one this deployment
          renders at all. Never answered from another lane's bridge.
    """
    if not _writes_enabled():
        return _refuse_writes_disabled("queue_start")

    # The queue drains against a LANE's target, not the session's. On a
    # single-lane deployment a start issued while the two differ is refused
    # outright; on a two-lane one the named lane must still be the lane the
    # session is on, so an item queued before a switch cannot be started after
    # it.
    bound = _bind_lane("Starting the Bluesky plan queue", requested=lane, require=True)

    # A server without a token still goes to the bridge, sending no header:
    # the bridge is the one authority on arming, and it answers
    # launch_token_required either way.
    token = get_server_context().launch_token_for(bound.key)
    headers = {"X-Launch-Token": token} if token else None

    # anyio's run_sync only forwards positional args, and `headers` is
    # keyword-only on `_http_post_json`, hence the lambda.
    status, body = await anyio.to_thread.run_sync(
        lambda: _http_post_json("/queue/start", {}, headers=headers, lane=bound.key)
    )
    if status != 200:
        return _relay_refusal(
            body,
            status,
            fallback_hints=["Check queue_list for the queue's current state."],
        )

    if isinstance(body, dict):
        body["lane"] = bound.key

    await notify_agent_activity_async("queue_start", "run", detail="queue")
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 5: stop the queue (asymmetric: halting is free, un-halting is armed)
# ---------------------------------------------------------------------------
@mcp.tool()
async def queue_stop(cancel: bool = False) -> str:
    """Stop the queue after the running item finishes — or withdraw a pending stop.

    Deliberately asymmetric, because the two directions are not equally safe.

    A plain stop (``cancel`` false, the default) is ungated everywhere: no
    writes check, no launch token, at this tool and at the bridge. Halting is
    the safe direction and must stay reachable even when the kill switch has
    disabled writes.

    Know its limit before you rely on it: it stops the queue AFTER the running
    item finishes, so a plan already moving hardware keeps going. If someone
    needs THAT stopped now, ``stop_run`` is the tool — it aborts the running
    plan immediately and is ungated for the same reason this one is. Say which
    of the two you are doing rather than letting "stop" cover both.

    ``cancel=True`` is the opposite operation: it WITHDRAWS a stop that a human
    (or you) already requested and lets the queue keep draining toward
    hardware. Reversing someone's halt is an arming action, so it is gated
    twice over locally: ``control_system.writes_enabled`` is re-read fresh, and
    a missing launch token is refused here as well, both before any HTTP call —
    then the bridge checks both again. That local token check is MORE than
    ``queue_start`` does: a tokenless start goes to the bridge and relays the
    bridge's refusal, while a tokenless withdrawal never leaves this process.
    Only withdraw a stop when you know why it was requested.

    Args:
        cancel: False (default) requests the stop. True withdraws a pending
            stop, which is an arming action.

    On a deployment with two plan lanes this goes to the lane whose queue is
    actually draining, not to the lane the session happens to be on. A stop is
    about the hardware that is moving, and an operator who switched targets
    mid-run must still be able to halt what they started.

    Returns:
        JSON ``{"stop_pending", "msg"}`` — ``stop_pending`` is true after a
        stop request, false after a withdrawal.

    Refusals:
        - writes_disabled (cancel=True only): writes are off, so a pending stop
          cannot be withdrawn. A plain stop is never refused for this reason.
        - launch_token_required (cancel=True only): no launch token here or at
          the bridge, or the two do not match. This deployment did not grant
          this agent a usable token — ask the operator to withdraw the stop
          from their queue panel instead; never chase the token in config.
        - manager_not_configured / manager_unreachable: the manager could not
          be reached. The queue was NOT stopped — say so plainly; do not report
          an unconfirmed halt as done.
        - queue_request_rejected: the manager refused (e.g. no stop is pending
          to withdraw).
    """
    # The kill switch is read BEFORE anything reaches the network, so a
    # withdrawal refused for writes-off is still refused with no request made
    # at all — the ordering the local-refusal contract rests on.
    if cancel and not _writes_enabled():
        return _refuse_writes_disabled("queue_stop(cancel=True)")

    # Halting follows the RUN, not the session: on a two-lane deployment the
    # stop is addressed to the lane whose queue is actually draining, so a stop
    # still reaches a plan started before the session switched targets. Never
    # gated by that comparison — see `resolve_halt_lane`.
    halt_lane = await resolve_halt_lane()

    headers = None
    if cancel:
        token = get_server_context().launch_token_for(halt_lane)
        if not token:
            return _refuse_unarmed("queue_stop(cancel=True)")
        headers = {"X-Launch-Token": token}

    status, body = await anyio.to_thread.run_sync(
        lambda: _http_post_json("/queue/stop", {"cancel": cancel}, headers=headers, lane=halt_lane)
    )
    if status != 200:
        return _relay_refusal(
            body,
            status,
            fallback_hints=["Check queue_list for whether a stop is already pending."],
        )

    # The two directions are opposite operations, so they must not share a
    # label: rendering a withdrawal as "stop" would tell the operator the queue
    # is halting when it has just been released to keep draining.
    await notify_agent_activity_async(
        "queue_stop", "run", detail="stop-withdrawn" if cancel else "stop"
    )
    return json.dumps(body)
