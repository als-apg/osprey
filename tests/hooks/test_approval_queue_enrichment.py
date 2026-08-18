"""Tests for the `osprey_approval` hook's queue-tool enrichment.

`queue_add` is how a Bluesky plan reaches the queue (see
`mcp_server/bluesky/tools/queue.py`). It carries nothing but a pinned
`draft_revision` — no run record exists yet, the bridge mints the run *from*
the shared draft at enqueue time. So this enrichment fetches the shared draft
(`GET /draft`) and renders exactly what would be queued: the plan name and args
currently staged, whether the draft still matches the pinned revision, and —
for a non-empty draft — the plan's provenance/trust tier, validation status,
and a source excerpt (resolved against `/plans` and `/plans/{name}/source`).
It also reports, from `GET /queue`, whether the queue is already draining,
which is what decides whether approving an enqueue is approving an execution.

It also asks the bridge's read-only pre-flight (`POST /plans/{name}/preview`)
what the staged parameters would actually move, and renders the answer: the
channels the plan declares by role, and a bounded window onto the setpoint
trajectory (first and last few moves, the exact total, the elided count).

Three things the rendered prompt must always carry:

* a revision-match line — a plain "matches" when `GET /draft`'s revision equals
  the pinned `draft_revision`, a LOUD "DRAFT CHANGED" warning (showing both
  revisions) when it does not: the human backstop against queuing a draft the
  agent pinned but that has since moved on; and
* for a session/unreviewed plan, the plan validator's documented obfuscation
  residual made legible — a `getattr`/string-concat body that passes the
  sandbox's AST scan (see `plan_validation.py`) is surfaced verbatim and
  labelled unmistakably agent-authored/unreviewed, so an approver who can SEE
  the source can refuse it where the automated stages could not; and
* the trajectory, or a plain statement that there is none to show — a
  pre-flight that timed out, was refused, or could not be reached degrades the
  prompt's evidence, never the human's chance to decide.

The hook runs as a real subprocess (see `hook_runner` in conftest.py) and talks
to the bridge over `urllib.request` — these tests stand up a tiny real HTTP
server (stdlib `http.server`) to play the bridge's part, so the hook's actual
network code path is exercised, not a mock. Fail-open is proven end-to-end:
`hook_runner` asserts a zero exit code, so an unreachable or malformed bridge
that still produces an `ask` decision proves the enrichment never raises out to
the subprocess boundary.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
from contextlib import contextmanager

import pytest

SCAN_HOOK_CONFIG = {
    "server_prefixes": ["mcp__bluesky__"],
    "approval_prefixes": ["mcp__bluesky__"],
}

_MISSING = object()


class _FakeBridgeHandler(http.server.BaseHTTPRequestHandler):
    """Serves canned bodies for whatever paths `routes` maps, GET and POST alike.

    A route value that is `bytes` is written raw (used to serve a malformed,
    non-JSON body); anything else is JSON-encoded. An unmapped path is a 404.
    Every POST body is recorded on `posted` as ``(path, parsed_body)``, which
    is how the pre-flight tests pin that the plan's parameters go on the wire
    exactly as staged.
    """

    routes: dict[str, object] = {}
    posted: list[tuple[str, object]] = []

    def _serve(self, body):
        if body is _MISSING:
            payload = b'{"detail": "not found"}'
            self.send_response(404)
        elif isinstance(body, bytes):
            payload = body
            self.send_response(200)
        else:
            payload = json.dumps(body).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 (stdlib method name)
        self._serve(self.routes.get(self.path, _MISSING))

    def do_POST(self):  # noqa: N802 (stdlib method name)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            parsed = raw
        self.posted.append((self.path, parsed))
        self._serve(self.routes.get(self.path, _MISSING))

    def log_message(self, format, *args):  # noqa: A002 (stdlib signature)
        pass  # keep test output quiet


@contextmanager
def fake_bridge(routes: dict[str, object], posted: list | None = None):
    """Runs a real threaded HTTP server for the duration of the `with` block.

    Pass *posted* to collect every POST the hook makes: the list is appended to
    in the server thread and is complete once the block exits.
    """
    handler_cls = type(
        "_Handler",
        (_FakeBridgeHandler,),
        {"routes": routes, "posted": posted if posted is not None else []},
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()


def _unused_port() -> int:
    """A port nothing is listening on, for the fail-open (unreachable) test."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


_SHIPPED_SOURCE = (
    'PLAN_METADATA = {"name": "orm", "description": "orm", "writes": True}\n\n'
    "def build_plan(devices, params):\n"
    "    yield from ()\n"
)

_OBFUSCATED_SESSION_SOURCE = (
    'PLAN_METADATA = {"name": "sneaky_plan", "description": "", "writes": False}\n\n'
    "def build_plan(devices, params):\n"
    "    leak = ().__class__.__base__.__subclasses__()\n"
    "    yield from ()\n"
)


def _queue_config(make_config):
    return make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": {"writes_enabled": True},
        }
    )


def _run_queue_add(hook_runner, config, tmp_path, draft_revision):
    return hook_runner(
        "osprey_approval.py",
        "mcp__bluesky__queue_add",
        {"draft_revision": draft_revision},
        config_path=config,
        cwd=tmp_path,
        hook_config=SCAN_HOOK_CONFIG,
    )


def _run_queue_tool(hook_runner, config, tmp_path, tool, tool_input=None):
    return hook_runner(
        "osprey_approval.py",
        f"mcp__bluesky__{tool}",
        tool_input or {},
        config_path=config,
        cwd=tmp_path,
        hook_config=SCAN_HOOK_CONFIG,
    )


def _reason(result) -> str:
    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    return output["permissionDecisionReason"]


@pytest.mark.unit
def test_matching_revision_renders_shipped_plan_and_source(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """When the live draft's revision equals the pinned one, render the plain
    match line plus the full plan detail; a shipped (operator-trusted) plan is
    never mislabeled as agent-authored."""
    config = _queue_config(make_config)
    routes = {
        "/draft": {
            "draft": {
                "plan_name": "orm",
                "plan_args": {"num_points": 5},
                "updated_by": "plan-panel",
                "updated_at": "2026-07-16T00:00:00+00:00",
            },
            "revision": 7,
        },
        "/plans": [
            {
                "name": "orm",
                "description": "orm",
                "schema": {},
                "metadata": {"name": "orm", "description": "orm", "writes": True},
                "provenance": "shipped",
            }
        ],
        "/plans/orm/source": {
            "name": "orm",
            "provenance": "shipped",
            "validated": True,
            "truncated": False,
            "source": _SHIPPED_SOURCE,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=7)

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]

    assert "matches pinned revision 7" in reason
    assert "DRAFT CHANGED" not in reason
    assert "Plan: orm" in reason
    assert "num_points" in reason
    assert "Hazard: writes to hardware" in reason
    # The retired authoring keys are gone from the contract and from the prompt.
    assert "Category" not in reason
    assert "Required devices" not in reason
    assert "Provenance: shipped" in reason
    assert "Validation status: not applicable" in reason
    assert _SHIPPED_SOURCE in reason
    # A trusted tier must never be mislabeled as agent-authored/unreviewed.
    assert "AGENT-AUTHORED" not in reason


@pytest.mark.unit
def test_matching_revision_labels_unvalidated_session_plan_as_untrusted(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A session-tier plan with NO passing validation record is clearly
    labelled agent-authored/unreviewed, and the obfuscated body itself is
    rendered legibly — the human backstop the plan validator's documented
    residual relies on."""
    config = _queue_config(make_config)
    routes = {
        "/draft": {
            "draft": {
                "plan_name": "sneaky_plan",
                "plan_args": {},
                "updated_by": "mcp-agent",
                "updated_at": "2026-07-16T00:00:00+00:00",
            },
            "revision": 3,
        },
        "/plans": [],  # quarantined: absent from GET /plans entirely
        "/plans/sneaky_plan/source": {
            "name": "sneaky_plan",
            "provenance": "session",
            "validated": False,
            "truncated": False,
            "source": _OBFUSCATED_SESSION_SOURCE,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=3)

    assert result is not None
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]

    assert "matches pinned revision 3" in reason
    assert "Plan: sneaky_plan" in reason
    assert "SESSION" in reason
    assert "AGENT-AUTHORED, NOT REVIEWED BY A HUMAN" in reason
    assert "NO PASSING RECORD" in reason
    # The obfuscated body itself must be visible to the approver.
    assert "__subclasses__" in reason


@pytest.mark.unit
def test_matching_revision_reports_validated_session_plan_as_passed(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A session-tier plan WITH a passing validation record renders
    "Validation status: PASSED" — and is STILL labelled agent-authored: a
    passing hash does not upgrade the trust tier, only the validation line."""
    config = _queue_config(make_config)
    routes = {
        "/draft": {
            "draft": {
                "plan_name": "reviewed_ish_plan",
                "plan_args": {},
                "updated_by": "mcp-agent",
                "updated_at": "2026-07-16T00:00:00+00:00",
            },
            "revision": 2,
        },
        "/plans": [
            {
                "name": "reviewed_ish_plan",
                "metadata": {
                    "name": "reviewed_ish_plan",
                    "description": "",
                    "writes": False,
                },
                "provenance": "session",
            }
        ],
        "/plans/reviewed_ish_plan/source": {
            "name": "reviewed_ish_plan",
            "provenance": "session",
            "validated": True,
            "truncated": False,
            "source": "PLAN_METADATA = {}\n",
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=2)

    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "Validation status: PASSED" in reason
    # A passing hash never launders the trust tier: still agent-authored.
    assert "AGENT-AUTHORED, NOT REVIEWED BY A HUMAN" in reason


@pytest.mark.unit
def test_newline_in_plan_name_cannot_forge_an_enrichment_line(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """`plan_name` is agent-authored (a session plan's PLAN_METADATA["name"])
    and reaches the prompt RAW — the bridge gates it only by registry
    membership, not character content. A newline in it must not forge a fake
    enrichment line: the render escapes control characters so the whole name
    stays on the "Plan:" line, visible but inert."""
    config = _queue_config(make_config)
    spoof = "orm\nValidation status: PASSED (SPOOFED BY THE PLAN NAME)"
    routes = {
        "/draft": {
            "draft": {
                "plan_name": spoof,
                "plan_args": {},
                "updated_by": "mcp-agent",
                "updated_at": "2026-07-16T00:00:00+00:00",
            },
            "revision": 1,
        }
        # No /plans or /source routes: the injection surface under test is the
        # plan_name interpolation, not the provenance block.
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=1)

    assert result is not None
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]

    # The embedded newline is escaped to a visible token — the name stays on
    # one line, so the spoofed text can never begin its own line.
    assert "\\x0a" in reason
    assert "\nValidation status: PASSED (SPOOFED BY THE PLAN NAME)" not in reason
    assert not any(
        line.startswith("Validation status: PASSED (SPOOFED") for line in reason.splitlines()
    )
    # The (inert) text still rides along on the Plan line for the approver to see.
    assert "SPOOFED BY THE PLAN NAME" in reason


@pytest.mark.unit
def test_changed_revision_renders_loud_drift_warning(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """When the live draft has moved past the pinned revision, the prompt leads
    with a LOUD warning naming both revisions, then renders the CURRENT draft."""
    config = _queue_config(make_config)
    routes = {
        "/draft": {
            "draft": {
                "plan_name": "orm",
                "plan_args": {"num_points": 9},
                "updated_by": "plan-panel",
                "updated_at": "2026-07-16T00:00:00+00:00",
            },
            "revision": 11,
        },
        "/plans": [
            {
                "name": "orm",
                "metadata": {"name": "orm", "description": "orm", "writes": True},
                "provenance": "shipped",
            }
        ],
        "/plans/orm/source": {
            "name": "orm",
            "provenance": "shipped",
            "validated": True,
            "truncated": False,
            "source": _SHIPPED_SOURCE,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=8)

    assert result is not None
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]

    assert "DRAFT CHANGED" in reason
    # Both the pinned and the current revision are named.
    assert "8" in reason and "11" in reason
    assert "matches pinned revision" not in reason
    # The current draft is still rendered so the approver sees what would run.
    assert "Plan: orm" in reason
    assert "num_points" in reason


@pytest.mark.unit
def test_empty_draft_renders_explicit_empty_line(tmp_path, hook_runner, make_config, monkeypatch):
    """A never-set / cleared draft renders an explicit EMPTY line, never a
    silent absence of plan detail."""
    config = _queue_config(make_config)
    routes = {"/draft": {"draft": None, "revision": 4}}

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=4)

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]

    assert "Draft: EMPTY" in reason
    assert "Plan:" not in reason
    # Tool/policy plain reason is still present.
    assert "Tool: queue_add" in reason


@pytest.mark.unit
def test_unreachable_bridge_fails_open(tmp_path, hook_runner, make_config, monkeypatch):
    """A dead bridge must never block the approval prompt: the hook still asks,
    just with the plain tool/policy reason instead of any draft detail.
    `hook_runner` asserts a zero exit code, so this also proves the enrichment
    never raises out to the subprocess boundary."""
    config = _queue_config(make_config)
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", f"http://127.0.0.1:{_unused_port()}")

    result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=5)

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]
    assert "Tool: queue_add" in reason
    assert "Approval policy: always" in reason
    assert "Plan:" not in reason
    assert "Draft:" not in reason


@pytest.mark.unit
def test_malformed_draft_response_fails_open(tmp_path, hook_runner, make_config, monkeypatch):
    """A `GET /draft` body that is not parseable JSON must fail open exactly
    like an unreachable bridge — plain reason, no draft detail, zero exit."""
    config = _queue_config(make_config)
    routes = {"/draft": b"this is not json {{{"}

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=6)

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]
    assert "Tool: queue_add" in reason
    assert "Plan:" not in reason
    assert "Draft:" not in reason


# ---------------------------------------------------------------------------
# Queue-state enrichment: whether approving an enqueue is approving execution
# ---------------------------------------------------------------------------

_IDLE_QUEUE = {
    "status": {"manager_state": "idle", "items_in_queue": 0},
    "items": [],
    "running_item": None,
}


@pytest.mark.unit
def test_queue_add_onto_a_running_queue_warns_that_it_executes_immediately(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The fact the tool call cannot show: this queue is already draining.

    Adding to an idle queue stages work for a later, separately-approved start.
    Adding to a running one hands the item to the RunEngine with no further
    human action — so the prompt must say so, or the approver has no way to
    tell the two apart.
    """
    config = _queue_config(make_config)
    routes = {
        "/draft": {"draft": {"plan_name": "orm", "plan_args": {}}, "revision": 2},
        "/queue": {
            "status": {"manager_state": "executing_queue", "items_in_queue": 1},
            "items": [{"item_uid": "u1", "name": "orm"}],
            "running_item": {"item_uid": "u0", "name": "orm"},
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_add(hook_runner, config, tmp_path, draft_revision=2))

    assert "A PLAN IS ALREADY RUNNING" in reason
    assert "executing_queue" in reason
    assert "no further approval" in reason


@pytest.mark.unit
def test_queue_add_onto_an_idle_queue_does_not_cry_wolf(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Negative control: an idle queue must NOT render the running warning.

    A prompt that warns every time teaches the approver to ignore the warning.
    """
    config = _queue_config(make_config)
    routes = {
        "/draft": {"draft": {"plan_name": "orm", "plan_args": {}}, "revision": 2},
        "/queue": _IDLE_QUEUE,
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_add(hook_runner, config, tmp_path, draft_revision=2))

    assert "ALREADY RUNNING" not in reason
    assert "AUTOSTART IS ENABLED" not in reason
    assert "No plan is currently running" in reason


@pytest.mark.unit
def test_queue_add_reports_out_of_band_autostart(tmp_path, hook_runner, make_config, monkeypatch):
    """Autostart on an idle queue is still armed — OSPREY never enables it."""
    config = _queue_config(make_config)
    routes = {
        "/draft": {"draft": {"plan_name": "orm", "plan_args": {}}, "revision": 2},
        "/queue": {
            "status": {"manager_state": "idle", "queue_autostart_enabled": True},
            "items": [],
            "running_item": None,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_add(hook_runner, config, tmp_path, draft_revision=2))

    assert "AUTOSTART IS ENABLED" in reason
    assert "out of band" in reason


@pytest.mark.unit
def test_queue_start_names_every_item_it_would_run(tmp_path, hook_runner, make_config, monkeypatch):
    """`queue_start` takes no arguments, so the prompt IS the statement of what moves.

    A start drains the whole queue, not only the item just added, and an
    agent-authored plan anywhere in it must be called out.
    """
    config = _queue_config(make_config)
    routes = {
        "/queue": {
            "status": {"manager_state": "idle", "items_in_queue": 2},
            "items": [
                {"item_uid": "u1", "name": "orm", "kwargs": {"num_points": 5}},
                {"item_uid": "u2", "name": "sneaky_plan", "kwargs": {}},
            ],
            "running_item": None,
        },
        "/plans": [
            {"name": "orm", "provenance": "shipped"},
            {"name": "sneaky_plan", "provenance": "session"},
        ],
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_tool(hook_runner, config, tmp_path, "queue_start"))

    assert "EVERY pending item" in reason
    assert "1. orm" in reason
    assert "num_points" in reason
    assert "2. sneaky_plan" in reason
    assert "AGENT-AUTHORED" in reason


@pytest.mark.unit
def test_queue_start_on_an_empty_queue_says_so(tmp_path, hook_runner, make_config, monkeypatch):
    """Approving a start that would run nothing should read as running nothing."""
    config = _queue_config(make_config)

    with fake_bridge({"/queue": _IDLE_QUEUE}) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_tool(hook_runner, config, tmp_path, "queue_start"))

    assert "Pending items: none" in reason
    assert "AGENT-AUTHORED" not in reason


@pytest.mark.unit
def test_queue_stop_cancel_renders_the_loud_withdrawal_warning(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """`cancel=true` does not halt anything — it un-halts. The prompt leads with that."""
    config = _queue_config(make_config)
    routes = {
        "/queue": {
            "status": {"manager_state": "paused", "queue_stop_pending": True},
            "items": [],
            "running_item": None,
        }
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(
            _run_queue_tool(hook_runner, config, tmp_path, "queue_stop", {"cancel": True})
        )

    assert "WITHDRAWS A PENDING STOP" in reason
    assert "does not halt anything" in reason
    assert "A stop is PENDING" in reason


@pytest.mark.unit
def test_plain_queue_stop_is_described_as_a_halt(tmp_path, hook_runner, make_config, monkeypatch):
    """Negative control for the withdrawal warning, plus the limit of a halt.

    A plain stop must not carry the withdrawal warning, and must say that the
    running item is NOT aborted — the operator reading this line is deciding
    whether halting the queue is enough. It now also names ``stop_run``, the
    tool that does abort; the earlier version of this test asserted the
    opposite, which was true only while that route was a retired 410.
    """
    config = _queue_config(make_config)

    with fake_bridge({"/queue": _IDLE_QUEUE}) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_tool(hook_runner, config, tmp_path, "queue_stop"))

    assert "Requests a stop" in reason
    assert "WITHDRAWS" not in reason
    assert "does NOT abort the item already in motion" in reason
    assert "stop_run" in reason
    assert "no tool here can" not in reason


@pytest.mark.unit
def test_stop_run_renders_the_abort_prompt_end_to_end(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The abort prompt through the real deployed hook, not just its describer.

    ``stop_run`` reaches the enrichment through the same short-name dispatch as
    the queue tools; a describer registered but not dispatched to would leave
    the most consequential Bluesky approval rendering nothing at all.
    """
    config = _queue_config(make_config)

    with fake_bridge({"/queue": _IDLE_QUEUE}) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_tool(hook_runner, config, tmp_path, "stop_run"))

    assert "ABORTS THE PLAN THAT IS RUNNING NOW" in reason
    assert "left wherever the plan moved it" in reason


@pytest.mark.unit
def test_queue_start_with_an_unreachable_bridge_says_the_queue_is_unseen(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Fail-open, but never silently: approving a start nobody can enumerate is
    itself the thing the approver needs told."""
    config = _queue_config(make_config)
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", f"http://127.0.0.1:{_unused_port()}")

    reason = _reason(_run_queue_tool(hook_runner, config, tmp_path, "queue_start"))

    assert "Tool: queue_start" in reason
    assert "unavailable" in reason
    assert "nobody here can see" in reason


# ---------------------------------------------------------------------------
# Pre-flight trajectory: what the queued launch would actually move
# ---------------------------------------------------------------------------
# `POST /plans/{name}/preview` walks the plan without running it and answers
# with the channels it declares plus the setpoints it would drive them to. The
# route is total (always HTTP 200, always the same keys), so these tests pin
# all three renderings the approver can be shown: the bounded trajectory, the
# truncated one, and the statement that there is none — the last of which must
# still leave the human an approval prompt to decide on.

_TRAJECTORY_CHANNELS = [
    {"channel": "SR:C01:COR:SP", "role": "movable"},
    {"channel": "SR:C02:COR:SP", "role": "movable"},
    {"channel": "SR:BPM01:X", "role": "readable"},
]


def _moves(count: int) -> list[dict]:
    """*count* moves with distinguishable channels and exactly-representable targets."""
    return [
        {"channel": f"SR:C{index:02d}:COR:SP", "target": index * 0.5}
        for index in range(1, count + 1)
    ]


def _preview_route(plan: str) -> str:
    return f"/plans/{plan}/preview"


def _draft_route(plan: str, plan_args: dict, revision: int) -> dict:
    return {"draft": {"plan_name": plan, "plan_args": plan_args}, "revision": revision}


@pytest.mark.unit
def test_queue_add_renders_the_bounded_trajectory_and_declared_channels(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The payoff of the whole pre-flight chain: before deciding, the human sees
    which channels the launch would move and read, and the setpoints it would
    drive them to — bounded to the first and last few, with the exact total and
    the elided count stated rather than the list silently cut short."""
    config = _queue_config(make_config)
    plan_args = {"axes": [{"setpoint": "SR:C01:COR:SP", "start": 0.5}]}
    routes = {
        "/draft": _draft_route("grid_scan", plan_args, 4),
        _preview_route("grid_scan"): {
            "ok": True,
            "plan": "grid_scan",
            "channels": _TRAJECTORY_CHANNELS,
            "moves": _moves(12),
            "total_moves": 12,
            "truncated": False,
            "move_cap": 10000,
            "reason": None,
            "detail": None,
        },
    }
    posted: list[tuple[str, object]] = []

    with fake_bridge(routes, posted) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_add(hook_runner, config, tmp_path, draft_revision=4))

    # The declared channels, grouped by role, in the bridge's own words.
    assert "Channels this launch would move: SR:C01:COR:SP, SR:C02:COR:SP" in reason
    assert "Channels this launch would read: SR:BPM01:X" in reason
    # The trajectory: exact total, first five, last five, and the elided count.
    assert "Setpoint trajectory — 12 moves in total:" in reason
    assert "1. SR:C01:COR:SP → 0.5" in reason
    assert "5. SR:C05:COR:SP → 2.5" in reason
    assert "… 2 moves not shown …" in reason
    assert "8. SR:C08:COR:SP → 4.0" in reason
    assert "12. SR:C12:COR:SP → 6.0" in reason
    # The elided middle really is absent, not merely unasserted.
    assert "SR:C06:COR:SP" not in reason
    assert "SR:C07:COR:SP" not in reason
    assert "truncated" not in reason
    # The parameters go on the wire exactly as the draft staged them.
    assert posted == [(_preview_route("grid_scan"), plan_args)]


@pytest.mark.unit
def test_queue_add_states_the_exact_total_when_the_pre_flight_truncated(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A trajectory past the worker's cap still reports its EXACT total, and says
    plainly that the last move shown is not the launch's last move — a prompt
    that showed a capped tail as the ending would misstate where the hardware
    finishes."""
    config = _queue_config(make_config)
    routes = {
        "/draft": _draft_route("grid_scan", {}, 5),
        _preview_route("grid_scan"): {
            "ok": True,
            "plan": "grid_scan",
            "channels": _TRAJECTORY_CHANNELS,
            "moves": _moves(12),
            "total_moves": 41337,
            "truncated": True,
            "move_cap": 10000,
            "reason": None,
            "detail": None,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_add(hook_runner, config, tmp_path, draft_revision=5))

    assert "41337 moves in total" in reason
    assert "(truncated)" in reason
    assert "NOT the last move of the launch" in reason
    # Bounded exactly as in the untruncated case: first five, last five, total.
    assert "1. SR:C01:COR:SP → 0.5" in reason
    assert "12. SR:C12:COR:SP → 6.0" in reason
    assert "… 2 moves not shown …" in reason


@pytest.mark.unit
def test_queue_add_renders_trajectory_unavailable_without_blocking_approval(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The route is total, so a pre-flight that could not run answers `ok: false`
    with a reason word. The prompt names it, keeps the channels the plan still
    declares, and — the point of the whole degradation — still ASKS: a human who
    cannot see the trajectory is a human who has to decide without it, never one
    who is blocked from deciding."""
    config = _queue_config(make_config)
    routes = {
        "/draft": _draft_route("grid_scan", {}, 6),
        _preview_route("grid_scan"): {
            "ok": False,
            "plan": "grid_scan",
            "channels": _TRAJECTORY_CHANNELS,
            "moves": [],
            "total_moves": 0,
            "truncated": False,
            "move_cap": None,
            "reason": "preview_timed_out",
            "detail": "The pre-flight was accepted but had not finished in time.",
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=6)

    # `_reason` asserts the decision is still "ask"; `hook_runner` asserts the
    # hook exited zero, so the degradation neither blocks nor crashes.
    reason = _reason(result)
    assert "Setpoint trajectory: unavailable (reason: preview_timed_out)" in reason
    assert "approval is not blocked" in reason
    assert "had not finished in time" in reason
    # What the plan declares it would touch survives the failed pre-flight.
    assert "Channels this launch would move: SR:C01:COR:SP, SR:C02:COR:SP" in reason
    assert "Tool: queue_add" in reason


@pytest.mark.unit
def test_queue_add_survives_a_bridge_with_no_pre_flight_route(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The other unavailable path: a bridge too old to have the route at all (or
    any transport failure) answers nothing this hook can parse. Same rendering,
    same undegraded rest of the prompt, same approval prompt."""
    config = _queue_config(make_config)

    with fake_bridge({"/draft": _draft_route("orm", {"num_points": 5}, 7)}) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_add(hook_runner, config, tmp_path, draft_revision=7))

    assert "Setpoint trajectory: unavailable — the pre-flight could not be reached" in reason
    assert "Plan: orm" in reason
    assert "num_points" in reason


@pytest.mark.unit
def test_a_newline_in_a_previewed_channel_cannot_forge_a_prompt_line(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Channel names and setpoints reach this prompt from an agent-staged plan
    via the pre-flight, so they are agent-influenced text on a human's approval
    prompt. A newline in one must not be able to start a line of its own."""
    config = _queue_config(make_config)
    spoof = "SR:C01:COR:SP\nHazard: read-only (no hardware writes declared)"
    routes = {
        "/draft": _draft_route("grid_scan", {}, 8),
        _preview_route("grid_scan"): {
            "ok": True,
            "plan": "grid_scan",
            "channels": [{"channel": spoof, "role": "movable"}],
            "moves": [{"channel": spoof, "target": 1.0}],
            "total_moves": 1,
            "truncated": False,
            "move_cap": 10000,
            "reason": None,
            "detail": None,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_add(hook_runner, config, tmp_path, draft_revision=8))

    assert "\\x0a" in reason
    assert not any(line.startswith("Hazard: read-only") for line in reason.splitlines())
    # Inert, but still visible — tampering is shown to the approver, not dropped.
    assert "Hazard: read-only" in reason


@pytest.mark.unit
def test_queue_start_renders_the_trajectory_of_the_items_it_would_run(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A start hands every pending item to the RunEngine with no further
    approval, so the leading items carry their trajectories too — previewed with
    each item's own parameters, exactly as the queue holds them."""
    config = _queue_config(make_config)
    routes = {
        "/queue": {
            "status": {"manager_state": "idle", "items_in_queue": 1},
            "items": [{"item_uid": "u1", "name": "grid_scan", "kwargs": {"num_points": 3}}],
            "running_item": None,
        },
        "/plans": [{"name": "grid_scan", "provenance": "shipped"}],
        _preview_route("grid_scan"): {
            "ok": True,
            "plan": "grid_scan",
            "channels": _TRAJECTORY_CHANNELS,
            "moves": _moves(3),
            "total_moves": 3,
            "truncated": False,
            "move_cap": 10000,
            "reason": None,
            "detail": None,
        },
    }
    posted: list[tuple[str, object]] = []

    with fake_bridge(routes, posted) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_tool(hook_runner, config, tmp_path, "queue_start"))

    assert "1. grid_scan" in reason
    assert "Setpoint trajectory — 3 moves in total:" in reason
    assert "3. SR:C03:COR:SP → 1.5" in reason
    assert "Channels this launch would move: SR:C01:COR:SP, SR:C02:COR:SP" in reason
    # Previewed with the item's own parameters, not the shared draft's.
    assert posted == [(_preview_route("grid_scan"), {"num_points": 3})]


@pytest.mark.unit
def test_queue_start_previews_only_the_first_three_items_of_a_larger_queue(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """`_MAX_PREVIEWED_QUEUE_ITEMS` pin: a start behind a 4+ item queue gets a
    trajectory for exactly the first three items, plus a note that the rest run
    without one shown — previewing every pending item would multiply the
    prompt's bridge calls with the queue length."""
    config = _queue_config(make_config)

    def _preview(name: str, move_count: int) -> dict:
        return {
            "ok": True,
            "plan": name,
            "channels": _TRAJECTORY_CHANNELS,
            "moves": _moves(move_count),
            "total_moves": move_count,
            "truncated": False,
            "move_cap": 10000,
            "reason": None,
            "detail": None,
        }

    routes = {
        "/queue": {
            "status": {"manager_state": "idle", "items_in_queue": 4},
            "items": [
                {"item_uid": "u1", "name": "orm1", "kwargs": {}},
                {"item_uid": "u2", "name": "orm2", "kwargs": {}},
                {"item_uid": "u3", "name": "orm3", "kwargs": {}},
                {"item_uid": "u4", "name": "orm4", "kwargs": {}},
            ],
            "running_item": None,
        },
        "/plans": [{"name": f"orm{n}", "provenance": "shipped"} for n in range(1, 5)],
        _preview_route("orm1"): _preview("orm1", 1),
        _preview_route("orm2"): _preview("orm2", 1),
        _preview_route("orm3"): _preview("orm3", 1),
        # Deliberately no route for orm4's preview: it must never be fetched.
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        reason = _reason(_run_queue_tool(hook_runner, config, tmp_path, "queue_start"))

    assert reason.count("Setpoint trajectory") == 3
    assert "1. orm1" in reason
    assert "2. orm2" in reason
    assert "3. orm3" in reason
    assert "4. orm4" in reason
    assert (
        "Trajectories above cover the first 3 items only; the rest run without "
        "one shown here." in reason
    )


@pytest.mark.unit
def test_queue_start_item_preview_skips_the_fetch_once_the_shared_budget_is_spent(
    hook_module,
):
    """`_PREVIEW_BUDGET_S` pin, at the helper level: once the shared deadline
    for the whole prompt's pre-flight is already in the past, a per-item fetch
    is skipped outright — no bridge call is attempted — and the item renders
    the budget-spent line instead of a trajectory. A real network fetch would
    make this test's runtime depend on the timeout; the deadline is exhausted
    directly instead, which is what a slow first item does to items after it.
    """
    mod = hook_module("osprey_approval")
    exhausted_deadline = mod.time.monotonic() - 1.0

    lines = mod._item_trajectory_lines(
        "http://127.0.0.1:1", {"name": "orm", "kwargs": {}}, exhausted_deadline
    )

    assert lines == [
        "    Setpoint trajectory: unavailable — the pre-flight budget for this "
        "prompt is spent. Approval is not blocked."
    ]


@pytest.mark.unit
def test_bounded_move_lines_singular_at_exactly_one_hidden_move(hook_module):
    """Grammar pin: exactly one hidden move (11 total, 5 head + 5 tail) reads
    '1 move not shown', not '1 moves not shown'."""
    mod = hook_module("osprey_approval")

    lines = mod._bounded_move_lines(_moves_for_module(mod, 11))

    assert "  … 1 move not shown …" in lines
    assert "moves not shown" not in "\n".join(lines)


def _moves_for_module(mod, count: int) -> list:
    return [
        {"channel": f"SR:C{index:02d}:COR:SP", "target": index * 0.5}
        for index in range(1, count + 1)
    ]


@pytest.mark.unit
def test_declared_channels_survives_a_non_list_channels_field(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """F1 regression: a preview whose `channels` field is present but not a
    list (a malformed bridge payload) must not raise. The channel lines are
    skipped — 'none declared' — while the rest of the prompt still renders."""
    config = _queue_config(make_config)
    routes = {
        "/draft": _draft_route("grid_scan", {}, 9),
        _preview_route("grid_scan"): {
            "ok": True,
            "plan": "grid_scan",
            "channels": 5,  # malformed: not a list
            "moves": [],
            "total_moves": 0,
            "truncated": False,
            "move_cap": None,
            "reason": None,
            "detail": None,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=9)

    reason = _reason(result)
    assert "Channels: none declared for these parameters." in reason
    assert "Setpoint trajectory: no moves" in reason
    # The rest of the prompt (this is the part a raised TypeError would have
    # silently wiped, per the pre-existing catch-all in main()) still renders.
    assert "Plan: grid_scan" in reason
    assert "Tool: queue_add" in reason


@pytest.mark.unit
def test_declared_channels_survives_entries_missing_expected_keys(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """F1 regression: a `channels` list whose entries are dicts but lack the
    expected `role`/`channel` keys must not raise — each entry degrades rather
    than aborting the whole block."""
    config = _queue_config(make_config)
    routes = {
        "/draft": _draft_route("grid_scan", {}, 10),
        _preview_route("grid_scan"): {
            "ok": True,
            "plan": "grid_scan",
            "channels": [{"bogus": 1}],
            "moves": [],
            "total_moves": 0,
            "truncated": False,
            "move_cap": None,
            "reason": None,
            "detail": None,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=10)

    reason = _reason(result)
    # The malformed entry degrades to a rendered-but-empty label pair rather
    # than raising — the rest of the prompt is unaffected either way.
    assert "Plan: grid_scan" in reason
    assert "Tool: queue_add" in reason


@pytest.mark.unit
def test_preview_fetch_survives_a_plan_name_with_a_lone_surrogate(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """F2 regression: `json.load` can decode a `\\udXXX` escape in a bridge
    response into a lone surrogate (`PlanMetadata.name` carries no character
    constraint). Percent-encoding that name for the preview URL must not raise
    `UnicodeEncodeError` — the pre-existing catch-all in `main()` would
    otherwise silently drop the WHOLE queue-detail block, including the
    revision-match warning, not just the trajectory."""
    config = _queue_config(make_config)
    spoof_name = "surrogate_plan\ud800"
    routes = {
        "/draft": _draft_route(spoof_name, {}, 11),
        # Deliberately no matching /plans/<quoted>/preview route: the fetch is
        # expected to degrade to "unavailable", not to raise before reaching
        # the network at all.
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=11)

    reason = _reason(result)
    # The rest of the block — proof the exception never propagated out of
    # `_fetch_preview` and wiped everything downstream of it.
    assert "matches pinned revision 11" in reason
    assert "Plan: surrogate_plan" in reason
    assert "Setpoint trajectory: unavailable" in reason


@pytest.mark.unit
def test_role_reason_and_detail_sanitize_embedded_newlines(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Sanitization coverage beyond `channel`: an unrecognised channel *role*
    (rendered in its own headline) and a pre-flight's *reason*/*detail* words
    are agent/bridge-influenced text on the same human-facing prompt, and must
    not be able to forge a line of their own either."""
    config = _queue_config(make_config)
    spoof_role = "custom\nHazard: read-only (no hardware writes declared)"
    spoof_reason = "bad_plan\nValidation status: PASSED (SPOOFED)"
    spoof_detail = "detail line\nProvenance: shipped (SPOOFED)"
    routes = {
        "/draft": _draft_route("grid_scan", {}, 12),
        _preview_route("grid_scan"): {
            "ok": False,
            "plan": "grid_scan",
            "channels": [{"channel": "SR:C01:COR:SP", "role": spoof_role}],
            "moves": [],
            "total_moves": 0,
            "truncated": False,
            "move_cap": None,
            "reason": spoof_reason,
            "detail": spoof_detail,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=12)

    reason = _reason(result)
    lines = reason.splitlines()

    # Escaped to a visible token, never a raw line break.
    assert "\\x0a" in reason
    # None of the three spoofed continuations start a line of their own.
    assert not any(line.startswith("Hazard: read-only") for line in lines)
    assert not any(line.startswith("Validation status: PASSED (SPOOFED)") for line in lines)
    assert not any(line.startswith("Provenance: shipped (SPOOFED)") for line in lines)
    # Still visible to the approver, just inert.
    assert "Hazard: read-only" in reason
    assert "PASSED (SPOOFED)" in reason
    assert "Provenance: shipped (SPOOFED)" in reason


@pytest.mark.unit
def test_target_sanitizes_embedded_newlines(tmp_path, hook_runner, make_config, monkeypatch):
    """Sanitization coverage for a move's *target*: like `channel`, it is a
    staged parameter relayed verbatim by the pre-flight, and must not be able
    to forge a line of its own either."""
    config = _queue_config(make_config)
    spoof_target = "1.0\nHazard: read-only (no hardware writes declared)"
    routes = {
        "/draft": _draft_route("grid_scan", {}, 13),
        _preview_route("grid_scan"): {
            "ok": True,
            "plan": "grid_scan",
            "channels": _TRAJECTORY_CHANNELS,
            "moves": [{"channel": "SR:C01:COR:SP", "target": spoof_target}],
            "total_moves": 1,
            "truncated": False,
            "move_cap": 10000,
            "reason": None,
            "detail": None,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_queue_add(hook_runner, config, tmp_path, draft_revision=13)

    reason = _reason(result)
    lines = reason.splitlines()

    assert "\\x0a" in reason
    assert not any(line.startswith("Hazard: read-only") for line in lines)
    assert "Hazard: read-only" in reason
