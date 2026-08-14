"""One copy of the queue flow every real-stack e2e uses to run a plan.

The bridge has no direct-execute route any more: ``POST /runs`` and
``POST /runs/{id}/launch`` both answer 410 ``use_the_queue``. Running a plan is
now four steps -- stage it in the shared draft, enqueue the exact revision that
was staged, arm the queue with the launch token, poll the run record -- and
this module is the single place they are spelled out, so a change to the flow
lands in one file rather than in every deploy proof that happens to run a scan.

``tests/e2e/test_bluesky_queue_e2e.py`` deliberately does NOT import this: that
module is the acceptance instrument for the queue surface itself, and a proof
of the flow must not be written in terms of a helper that assumes it. Every
OTHER e2e -- the ones whose subject is a substrate, a plan, or a physics
round trip, and for which the queue is merely transport -- should use this.

WHY EACH STEP IS SHAPED THE WAY IT IS:

* The draft is CLEARED before it is staged. A ``PATCH /draft`` that changes
  nothing is a documented no-op: no revision bump. Re-running the same scan
  with the same arguments (which is exactly what ``pytest.mark.flaky`` does on
  a rerun) would therefore hand back the revision the first attempt already
  consumed, and the enqueue would be refused ``draft_revision_already_launched``
  -- a confusing failure with nothing to do with the code under test. Deleting
  first makes every stage a genuine plan-change, so the revision always moves.
* The pending queue is DRAINED before staging. ``POST /queue/start`` starts the
  whole queue, not one item, and the queue lives in a Redis named volume that
  ``osprey down`` deliberately keeps -- so a previous run's leftovers would go
  back on the hardware under this run's start. Removal goes through the
  bridge's own ``DELETE /queue/items/{uid}``, never into Redis.
* The enqueue is UNARMED (no token). The queue is idle at that point, so an
  added item just sits there until the armed start -- the designed
  compose-now/arm-later split.

``wait_for_worker_environment`` is the one piece here that belongs to a deploy
FIXTURE rather than to the flow: it is the readiness gate an enqueue depends on
and that ``/health`` does not give you. See its docstring.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

LAUNCH_TOKEN_HEADER = "X-Launch-Token"

#: Terminal ``GET /runs/{id}`` statuses. ``stopped`` and ``error`` are terminal
#: too -- a caller asserting "completed" wants to see them reported promptly,
#: not to wait out its whole deadline first.
TERMINAL_STATUSES = ("completed", "error", "stopped")

#: How long a deploy fixture waits for the RE worker environment to come up.
#: Generous on purpose: opening it connects every substrate device over Channel
#: Access, which takes tens of seconds on a cold stack (and longer under QEMU
#: emulation on Apple Silicon).
WORKER_ENV_TIMEOUT_SEC = 300.0


def request(
    base_url: str,
    path: str,
    method: str,
    body: dict[str, Any] | None = None,
    *,
    token: str | None = None,
    timeout: float = 20.0,
) -> tuple[int, Any]:
    """One HTTP call, returning ``(status, parsed_body)`` for 2xx AND 4xx/5xx.

    Refusals carry the ``{"detail": {"code", "detail", ...}}`` body callers
    assert on, so an ``HTTPError`` is a normal result here, never an exception
    to propagate.
    """
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers: dict[str, str] = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    elif method != "GET":
        data = b""
    if token:
        headers[LAUNCH_TOKEN_HEADER] = token
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{base_url}{path}", data=data, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except ValueError:
                return resp.status, raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw.decode("utf-8"))
        except ValueError:
            return exc.code, raw.decode("utf-8", errors="replace")


def wait_for_worker_environment(
    base_url: str, *, timeout: float = WORKER_ENV_TIMEOUT_SEC, poll: float = 2.0
) -> dict[str, Any]:
    """Block until the manager reports an OPEN RE worker environment.

    The gate every deploy fixture in this suite owes its own enqueue. HTTP
    readiness on ``/health`` does NOT imply it: the bridge opens the worker
    environment in a background task that is deliberately excluded from
    readiness (device connect takes tens of seconds), so a stack can answer 200
    with an empty worker namespace. ``POST /queue/items`` validates against
    ``plans_allowed``, which the manager downloads from the worker only when
    that environment opens -- so enqueueing too early is refused 409 "not in
    the list of allowed plans", a message that reads like a permissions problem
    and is nothing of the sort (the shipped ``user_group_permissions.yaml``
    allows ``[":.*"]``; the list was empty because the namespace was).

    ``manager_state`` is NOT the signal to wait on. It reads ``idle`` both
    before the environment has ever been opened and after it is up, so gating
    on it would be very nearly vacuous. ``worker_environment_exists`` is the
    one field that separates the two.

    Returns the manager status document that satisfied the wait. Raises
    ``AssertionError`` naming the last status seen -- a failure here says the
    worker environment never came up, which points at the queueserver rather
    than at the plan the caller was about to run.
    """
    deadline = time.monotonic() + timeout
    last: Any = "(no answer yet)"
    while time.monotonic() < deadline:
        http_status, body = request(base_url, "/queue", "GET")
        if http_status == 200 and isinstance(body, dict):
            last = body.get("status") or {}
            if isinstance(last, dict) and last.get("worker_environment_exists"):
                return dict(last)
        else:
            last = body
        time.sleep(poll)
    raise AssertionError(
        f"the RE worker environment never opened within {timeout:.0f}s -- the queue "
        f"cannot accept plans until it does (last manager status: {last!r})"
    )


def drain_pending_queue(base_url: str) -> None:
    """Remove every PENDING item, so this run's start drains only its own work.

    History is deliberately left alone: it is append-only and every caller
    looks up its own run ids in it.
    """
    status, queue = request(base_url, "/queue", "GET")
    assert status == 200, f"GET /queue failed: {status} {queue}"
    for item in queue.get("items") or []:
        uid = item.get("item_uid")
        if not isinstance(uid, str):
            continue
        removed_status, removed = request(base_url, f"/queue/items/{uid}", "DELETE")
        assert removed_status == 200, (
            f"could not drain queue item {uid}: {removed_status} {removed}"
        )


def stage_draft(base_url: str, plan_name: str, plan_args: dict[str, Any], *, client_id: str) -> int:
    """Clear the draft, stage ``plan_name``/``plan_args`` into it, return the revision."""
    request(base_url, f"/draft?client_id={client_id}", "DELETE")
    status, body = request(
        base_url,
        "/draft",
        "PATCH",
        {"plan_name": plan_name, "plan_args_patch": plan_args, "client_id": client_id},
    )
    assert status == 200, f"PATCH /draft failed: {status} {body}"
    assert body["plan_name"] == plan_name, f"the draft did not take the plan name: {body}"
    revision = body["revision"]
    assert isinstance(revision, int), f"no integer revision on the PATCH response: {body}"
    return revision


def enqueue(base_url: str, revision: int) -> str:
    """Enqueue the plan staged at ``revision``; return its OSPREY run id."""
    status, body = request(base_url, "/queue/items", "POST", {"draft_revision": revision})
    assert status == 200, f"POST /queue/items failed: {status} {body}"
    run_id = body.get("run_id")
    assert run_id, f"no run_id in the enqueue response: {body}"
    assert body.get("revision") == revision, f"the enqueue pinned a different revision: {body}"
    return str(run_id)


def start_queue(base_url: str, token: str) -> None:
    """Arm and start the queue -- the one action that puts plans on hardware."""
    status, body = request(base_url, "/queue/start", "POST", token=token, timeout=120.0)
    assert status == 200, f"armed POST /queue/start failed: {status} {body}"
    assert body.get("started") is True, f"start did not report started: {body}"


def stage_and_enqueue(
    base_url: str, plan_name: str, plan_args: dict[str, Any], *, client_id: str
) -> str:
    """Drain leftovers, stage the plan, enqueue it. Returns the OSPREY run id."""
    drain_pending_queue(base_url)
    revision = stage_draft(base_url, plan_name, plan_args, client_id=client_id)
    return enqueue(base_url, revision)


def run_record(base_url: str, run_id: str) -> tuple[int, Any]:
    return request(base_url, f"/runs/{run_id}", "GET")


def wait_for_terminal_status(
    base_url: str, run_id: str, *, timeout: float, poll: float = 0.5
) -> dict[str, Any]:
    """Poll ``GET /runs/{id}`` until it reports a terminal status, or the deadline.

    Returns the LAST record seen either way -- a caller's own
    ``status == "completed"`` assertion is what turns a timeout into its own
    (much more informative) failure message.
    """
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, body = run_record(base_url, run_id)
        if status == 200 and isinstance(body, dict):
            last = body
            if body.get("status") in TERMINAL_STATUSES:
                return last
        time.sleep(poll)
    return last


def run_scan(
    base_url: str,
    plan_name: str,
    plan_args: dict[str, Any],
    *,
    token: str,
    client_id: str,
    timeout: float,
    poll: float = 0.5,
) -> tuple[str, dict[str, Any]]:
    """The whole flow: stage -> enqueue -> armed start -> poll.

    Returns ``(run_id, last_run_record)``. The record is returned rather than
    asserted on so each proof can say what a non-completion means in ITS terms.
    """
    run_id = stage_and_enqueue(base_url, plan_name, plan_args, client_id=client_id)
    start_queue(base_url, token)
    return run_id, wait_for_terminal_status(base_url, run_id, timeout=timeout, poll=poll)
