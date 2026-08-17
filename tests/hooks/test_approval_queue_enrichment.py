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

Two things the rendered prompt must always carry:

* a revision-match line — a plain "matches" when `GET /draft`'s revision equals
  the pinned `draft_revision`, a LOUD "DRAFT CHANGED" warning (showing both
  revisions) when it does not: the human backstop against queuing a draft the
  agent pinned but that has since moved on; and
* for a session/unreviewed plan, the plan validator's documented obfuscation
  residual made legible — a `getattr`/string-concat body that passes the
  sandbox's AST scan (see `plan_validation.py`) is surfaced verbatim and
  labelled unmistakably agent-authored/unreviewed, so an approver who can SEE
  the source can refuse it where the automated stages could not.

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
    """Serves canned bodies for whatever paths `routes` maps.

    A route value that is `bytes` is written raw (used to serve a malformed,
    non-JSON body); anything else is JSON-encoded. An unmapped path is a 404.
    """

    routes: dict[str, object] = {}

    def do_GET(self):  # noqa: N802 (stdlib method name)
        body = self.routes.get(self.path, _MISSING)
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

    def log_message(self, format, *args):  # noqa: A002 (stdlib signature)
        pass  # keep test output quiet


@contextmanager
def fake_bridge(routes: dict[str, object]):
    """Runs a real threaded HTTP server for the duration of the `with` block."""
    handler_cls = type("_Handler", (_FakeBridgeHandler,), {"routes": routes})
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
    assert "left wherever the scan moved it" in reason


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
