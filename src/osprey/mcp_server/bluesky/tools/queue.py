"""MCP tools: the agent's side of the bridge's plan queue.

The queue replaces the old launch-a-single-draft path. Execution is now two
steps, deliberately: ``queue_add`` puts the pinned draft into the queue (which
starts nothing), and ``queue_start`` begins draining it (which starts a real
plan). Splitting them is what lets a human review a composed queue before
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

**Arming.** Two operations can put hardware in motion, and each carries two
local safety layers, both enforced BEFORE any HTTP call:

1. A ``control_system.writes_enabled`` re-read straight from config, mirroring
   ``ControlSystemConnector._writes_enabled``
   (``osprey/connectors/control_system/base.py``) exactly — same key, same
   fail-closed except clause. Re-read fresh on every call, never cached, so a
   hook-bypassed invocation carrying a valid launch token is still refused
   while writes are disabled.
2. A client-side launch-token presence check, so an unarmed server refuses
   locally with no network call.

``queue_start`` carries both unconditionally. ``queue_add`` is armed only
*sometimes* — enqueuing onto an idle queue moves nothing, while enqueuing onto
a queue that is already draining hands the item straight to hardware — so it
carries them conditionally: the launch token is attached to the request only
when writes are enabled, and the bridge (which alone knows the manager's live
state, under a lock, with a post-add re-check) decides whether this particular
enqueue needed it. A writes-disabled deployment therefore keeps composing
queues and is refused exactly at the point where composing would become
executing. ``queue_stop`` inverts the asymmetry: a plain stop is ungated in
every layer because halting is always allowed, while ``cancel=True`` withdraws
a human's pending halt and is gated like a start.

**Refusals are relayed, never rewritten.** Every refusal the bridge issues is
an HTTP 4xx/5xx whose ``detail`` is ``{"code", "detail", ...extras}``, where
``code`` is the machine-readable branch key. These tools surface the bridge's
``code`` as the envelope's ``error_type``, its sentence as the
``error_message``, and the entire detail object — including extras like
``capability``, ``manager_state``, ``revision``, ``plan`` and
``item_left_behind`` — as ``details``. Nothing is reworded or reclassified: a
tool that renamed a refusal would put the agent and the panel on different
vocabularies for the same event. The only envelopes minted here are the two
local refusals (``writes_disabled``, ``launch_token_required``) and a
last-resort ``bluesky_bridge_error`` for a bridge response that carried no
structured detail at all.

**No reason-code constants are imported or hardcoded here.** The capability
reason codes are defined in ``osprey.services.bluesky_bridge.queue_backend``,
which pulls in ``bluesky-queueserver-api`` and the runner wiring; importing it
would break this server's standing invariant of making no bluesky/ophyd/tiled
imports (see ``bluesky/server.py``). These tools never branch on a reason code
— they pass the whole capability record through — so they need no copy of the
vocabulary and cannot drift from it. The docstrings below name the codes as
prose, for the agent reading them.
"""

from __future__ import annotations

import functools
import json
from typing import NoReturn

import anyio

from osprey.bluesky_bridge_connection import unwrap_bridge_conflict_detail
from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.server_context import (
    _http_get_json,
    _http_post_json,
    bridge_error_message,
    get_server_context,
)
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity

# Remediation guidance per bridge refusal code, kept in one table so every
# tool answers the same code with the same next step. Codes absent here get
# the tool's own generic hints — an unknown code is relayed, never dropped.
_REFUSAL_HINTS: dict[str, list[str]] = {
    "launch_token_required": [
        "This operation arms hardware motion and needs the bridge's launch token.",
        "If the deployment has no BLUESKY_LAUNCH_TOKEN configured, the operator must set one.",
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
        [f"Set control_system.writes_enabled: true in config.yml to allow {refused}."],
    )


def _refuse_unarmed(tool: str) -> NoReturn:
    """The local no-token refusal, raised before any network call.

    Uses the bridge's own ``launch_token_required`` code so the agent branches
    on one name whether the missing token is caught here or at the bridge.
    """
    return make_error(
        "launch_token_required",
        f"This Bluesky MCP server has no BLUESKY_LAUNCH_TOKEN configured — "
        f"{tool} is refused client-side before contacting the bridge.",
        [
            "Set BLUESKY_LAUNCH_TOKEN (or bluesky.launch_token in config.yml) "
            "for this bridge instance."
        ],
        details={"code": "launch_token_required"},
    )


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

    Refusals:
        - bluesky_bridge_error / bluesky_bridge_unreachable: the capability
          could not be read. Treat an unreadable capability as CANNOT EXECUTE
          — never assume executability from a failed check.
    """
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
async def queue_add(draft_revision: int) -> str:
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
    repeat scan, a retry — needs a draft edit (set_draft) to mint a new
    revision first; re-adding the old one is refused, by design, so a
    duplicated call cannot silently double-queue a scan.

    Args:
        draft_revision: The draft revision to queue, as returned by get_draft
            or set_draft. The bridge queues the draft snapshot pinned at this
            exact revision.

    Returns:
        JSON ``{"run_id", "revision", "item"}`` — ``run_id`` is OSPREY's id for
        the eventual run (use it with get_run / get_run_data once it executes),
        and ``item.item_uid`` is the queue handle for this item.

    Refusals (nothing is queued, and the pinned revision stays usable):
        - launch_token_required: the queue is already draining and this add
          needed the launch token. ``details.manager_state`` names the state
          that made it armed. Either the deployment has no token, or this
          server withheld it because writes are disabled — the suggestions say
          which, and neither is agent-recoverable; the operator must enable
          writes or configure the token. If ``details.item_left_behind`` is
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
    """
    # Read the kill switch ONCE and reuse the answer, so the header decision
    # and the refusal wording can never disagree about it.
    writes_ok = _writes_enabled()
    token = get_server_context().launch_token
    # The token is withheld while writes are disabled: the bridge then treats
    # this as an unarmed add, permitting it onto an idle queue and refusing it
    # (launch_token_required) the moment the queue is draining. That is the
    # writes_enabled re-check applied to exactly the armed half of enqueue,
    # with the live manager state read under the bridge's own lock rather than
    # guessed here from a stale status.
    headers = {"X-Launch-Token": token} if token and writes_ok else None

    status, body = await anyio.to_thread.run_sync(
        lambda: _http_post_json("/queue/items", {"draft_revision": draft_revision}, headers=headers)
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
                "control_system.writes_enabled is false in this deployment — enabling "
                "writes in config.yml, not a different token, is what unblocks adding "
                "to a running queue.",
            ]
        return _relay_refusal(
            body,
            status,
            fallback_hints=["Re-read the draft with get_draft before adding it again."],
            extra_hints=extra_hints,
        )

    run_id = body.get("run_id") if isinstance(body, dict) else None
    await anyio.to_thread.run_sync(
        functools.partial(
            notify_agent_activity,
            "queue_add",
            "run",
            detail=str(run_id) if run_id is not None else None,
        )
    )
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 4: start draining the queue (the arming action)
# ---------------------------------------------------------------------------
@mcp.tool()
async def queue_start() -> str:
    """Start draining the queue. THE arming action — real motion follows.

    Step two of two, and the only way execution ever begins: the manager's own
    autostart stays disabled, so every start originates from a deliberate call
    here or from the human's start control. This runs the queue as it stands —
    every pending item, in order, not just the one you added — so read
    queue_list first and be sure the whole queue is what should run.

    Two local gates pass before any network call: this deployment's
    ``control_system.writes_enabled`` must re-read true (fresh every call,
    never cached, so a hook-bypassed invocation holding a valid token is still
    refused while writes are off), and this server must hold a launch token.
    The bridge re-verifies the token, re-checks every queued session plan's
    validation, and opens the worker environment if needed. The human also sees
    an approval prompt for the start.

    Returns:
        JSON ``{"started": true, "msg"}``. The queue then drains
        asynchronously — poll queue_list for progress and get_run_data for the
        running item's data.

    Refusals (nothing started):
        - writes_disabled: writes are off in this deployment. Not
          agent-recoverable; the operator must enable writes.
        - launch_token_required: no launch token here or at the bridge, or the
          two do not match. Not agent-recoverable; contact the operator.
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
    """
    if not _writes_enabled():
        return _refuse_writes_disabled("queue_start")

    token = get_server_context().launch_token
    if not token:
        return _refuse_unarmed("queue_start")

    # anyio's run_sync only forwards positional args, and `headers` is
    # keyword-only on `_http_post_json`, hence the lambda.
    status, body = await anyio.to_thread.run_sync(
        lambda: _http_post_json("/queue/start", {}, headers={"X-Launch-Token": token})
    )
    if status != 200:
        return _relay_refusal(
            body,
            status,
            fallback_hints=["Check queue_list for the queue's current state."],
        )

    await anyio.to_thread.run_sync(
        functools.partial(notify_agent_activity, "queue_start", "run", detail="queue")
    )
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
    hardware. Reversing someone's halt is an arming action, so it carries the
    same gates as queue_start — ``control_system.writes_enabled`` re-read fresh,
    plus the launch token — at this tool and again at the bridge. Only withdraw
    a stop when you know why it was requested.

    Args:
        cancel: False (default) requests the stop. True withdraws a pending
            stop, which is an arming action.

    Returns:
        JSON ``{"stop_pending", "msg"}`` — ``stop_pending`` is true after a
        stop request, false after a withdrawal.

    Refusals:
        - writes_disabled (cancel=True only): writes are off, so a pending stop
          cannot be withdrawn. A plain stop is never refused for this reason.
        - launch_token_required (cancel=True only): no launch token here or at
          the bridge, or the two do not match.
        - manager_not_configured / manager_unreachable: the manager could not
          be reached. The queue was NOT stopped — say so plainly; do not report
          an unconfirmed halt as done.
        - queue_request_rejected: the manager refused (e.g. no stop is pending
          to withdraw).
    """
    headers = None
    if cancel:
        if not _writes_enabled():
            return _refuse_writes_disabled("queue_stop(cancel=True)")
        token = get_server_context().launch_token
        if not token:
            return _refuse_unarmed("queue_stop(cancel=True)")
        headers = {"X-Launch-Token": token}

    status, body = await anyio.to_thread.run_sync(
        lambda: _http_post_json("/queue/stop", {"cancel": cancel}, headers=headers)
    )
    if status != 200:
        return _relay_refusal(
            body,
            status,
            fallback_hints=["Check queue_list for whether a stop is already pending."],
        )
    return json.dumps(body)
