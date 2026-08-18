"""MCP tool: stop_run — the emergency abort of the plan already in motion.

``queue_stop`` halts the queue AFTER the running item finishes. This tool is
the other half: ``POST /queue/abort`` stops the plan that is running right now,
discarding the rest of its points and leaving the queue stopped.

**Completely ungated, on purpose.** No ``control_system.writes_enabled``
re-read, no launch token, at this tool and at the bridge — exactly the posture
of a plain ``queue_stop``, and for the same reason: halting must never have a
failure mode, so it must keep working when the kill switch has disabled writes,
which is precisely when an operator is most likely to want a plan stopped. The
tool is approval-gated (registry ``permissions_ask``) so a human still sees
every abort, and the approval hook's prompt states what an abort costs; the
kill-switch hook must NEVER attach to it (pinned in
``tests/registry/test_gate_wiring.py`` and the killswitch tests).

**It takes no run id.** The queue server runs exactly one plan at a time, and
this aborts that plan. There is nothing to select, and a run-id parameter would
only add a way for a halt to be refused — the one thing this path must not
have.

The bridge composes the abort (immediate pause, bounded wait for the paused
state, then abort) and owns every refusal; see
``services/bluesky_bridge/queue_backend.py``'s ``QueueBackend.abort``. Nothing
here re-derives that: refusals are relayed verbatim through the queue tools'
shared relay, so an ``abort_pause_timeout`` reaches the agent saying plainly
that nothing was aborted.
"""

from __future__ import annotations

import functools
import json

import anyio

from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.server_context import _http_post_json

# The refusal relay and its hint table live with the other queue tools: the
# abort answers to the same bridge vocabulary, and a second copy here would be
# a second thing to keep in step.
from osprey.mcp_server.bluesky.tools.queue import _relay_refusal
from osprey.mcp_server.http import notify_agent_activity_async

# The abort is the one bridge call whose server-side work is a COMPOSITION of
# manager calls, so ``server_context._TIMEOUT`` (15s, sized for single-call
# routes) is far too short for it. Worst case, from the bridge's own constants
# (``QueueBackend.__init__``: ``abort_pause_polls=20``,
# ``abort_poll_interval=0.5``) and the queueserver zmq client's per-call recv
# timeout (``bluesky_queueserver_api`` default, 2.0s):
#
#     sleep budget      20 polls x 0.5s                              = 10s
#     manager calls     1 initial status + 1 initial re_pause
#                       + 20 poll statuses + 19 re_pause re-issues
#                       + 1 re_abort + 1 final status  = 43 round trips
#                       43 x 2.0s                                    = 86s
#     ------------------------------------------------------------------
#     true bound                                                     ~96s
#
# That is the pathological "manager answers, but only just inside its own
# timeout, every time" case; a manager that stops answering outright raises on
# the first call and the abort fails fast. 120s sits ~25% above the bound, so a
# slow-but-answering manager can never make this tool report
# ``bluesky_bridge_unreachable`` while the bridge is mid-abort -- telling the
# agent the bridge is down when the halt is actually in progress, and inviting a
# retry race on the emergency path. Per-call, never global: every other bluesky
# tool must keep failing fast on a dead bridge. Keep in step with
# ``queue_backend.QueueBackend.abort``'s docstring and the bluesky-web sidecar's
# ``queue_relay._ABORT_TIMEOUT``.
_ABORT_TIMEOUT = 120.0  # seconds


@mcp.tool()
async def stop_run() -> str:
    """Abort the plan running RIGHT NOW. Ungated — this is the emergency stop.

    Use this when a plan already moving hardware has to stop immediately.
    ``queue_stop`` is the gentler sibling and does NOT do this: it halts the
    queue only after the running item finishes.

    What an abort costs, and what to tell the human: the running plan's
    remaining points are discarded, the data collected so far is kept, and the
    hardware is left wherever the plan had moved it — an abort does not return
    anything to a starting position. Say that when you report the abort, and
    say it when you propose one.

    Nothing gates this call — no writes check, no launch token, at this tool or
    at the bridge — because a halt that can be refused for a policy reason is a
    halt with a failure mode. It is still approval-gated, so a human sees it.

    Returns:
        JSON ``{"aborted": true, "abort_pending", "paused_first",
        "manager_state", "msg"}``. ``abort_pending: true`` means the manager
        accepted the abort and is still unwinding — the abort landed; it is not
        a failure. ``paused_first`` records whether the Run Engine had to be
        paused first (the bridge does that for you). Report ``msg`` verbatim
        when it is non-empty.

    Refusals — each says exactly what did and did not happen:
        - nothing_running: no plan was under way, or it ended while the abort
          was being prepared. Nothing was stopped because there was nothing to
          stop; say that, do not report a halt.
        - abort_pause_timeout: the Run Engine never reached the paused state an
          abort requires, so NOTHING WAS ABORTED and the plan may still be
          running. Tell the human immediately and plainly; retrying once is
          reasonable, and beyond that they need their facility's out-of-band
          means.
        - manager_unreachable: the queue manager did not answer at all. The
          plan was NOT stopped.
        - queue_request_rejected: the manager reached the paused state and then
          refused the abort itself.
    """
    status, body = await anyio.to_thread.run_sync(
        functools.partial(_http_post_json, "/queue/abort", {}, timeout=_ABORT_TIMEOUT)
    )
    if status != 200:
        return _relay_refusal(
            body,
            status,
            fallback_hints=[
                "The abort did not land. Check queue_list for what the manager is doing, "
                "and tell the human the plan may still be running.",
            ],
        )

    await notify_agent_activity_async("stop_run", "run", detail="abort")
    return json.dumps(body)
